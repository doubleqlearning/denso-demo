"""Retrieval evaluation on BEIR-format datasets: Recall@k, MRR@k, nDCG@k.

A BEIR dataset directory holds corpus.jsonl, queries.jsonl, and
qrels/<split>.tsv (https://github.com/beir-cellar/beir). Metrics are
document-level: chunk hits are reduced to each document's best chunk first.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytrec_eval
from beir.datasets.data_loader import GenericDataLoader

from backend.src.agents.rag.chunking import Document
from backend.src.agents.rag.store import ChunkHit, DocumentStore, SearchMode


K_VALUES = (1, 3, 5, 10, 100)
METRICS = ("Recall", "MRR", "nDCG")
# Retrieved documents kept per query in a saved run.
TOP_DOCS_SAVED = 10

Qrels = dict[str, dict[str, int]]
# {query_id: [doc_id, ...]} best first.
Rankings = dict[str, list[str]]


def load_corpus(dataset_dir: Path) -> list[Document]:
    corpus = GenericDataLoader(data_folder=str(dataset_dir)).load_corpus()
    return [
        Document(id=doc_id, text=doc.get("text") or "", title=doc.get("title") or "")
        for doc_id, doc in corpus.items()
    ]


def load_queries(dataset_dir: Path, split: str) -> tuple[dict[str, str], Qrels]:
    """Queries that have relevance judgements in the split, and those judgements."""

    _, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)
    return queries, qrels


def rankings_to_run(rankings: Rankings) -> dict[str, dict[str, float]]:
    """Rank-preserving scores for trec_eval, which orders by score, not list position."""

    return {
        qid: {doc_id: float(len(doc_ids) - rank) for rank, doc_id in enumerate(doc_ids)}
        for qid, doc_ids in rankings.items()
    }


def reciprocal_rank(relevant: Iterable[str], doc_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant)
    for rank, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def per_query_metrics(
    qrels: Qrels, rankings: Rankings, k_values: Sequence[int] = K_VALUES
) -> dict[str, dict[str, float]]:
    """Recall@k, MRR@k, nDCG@k per judged query; a query with no results scores 0."""

    k_values = sorted(set(k_values))
    ks = ",".join(str(k) for k in k_values)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"recall.{ks}", f"ndcg_cut.{ks}"})
    # trec_eval skips queries without results; they still count as misses here.
    scored = evaluator.evaluate(rankings_to_run({q: d for q, d in rankings.items() if d}))
    metrics = {}
    for qid, judgements in qrels.items():
        relevant = [doc_id for doc_id, grade in judgements.items() if grade > 0]
        trec = scored.get(qid, {})
        doc_ids = rankings.get(qid, [])
        row = {}
        for k in k_values:
            row[f"Recall@{k}"] = trec.get(f"recall_{k}", 0.0)
            row[f"MRR@{k}"] = reciprocal_rank(relevant, doc_ids, k)
            row[f"nDCG@{k}"] = trec.get(f"ndcg_cut_{k}", 0.0)
        metrics[qid] = row
    return metrics


def mean_metrics(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    return {
        name: sum(row[name] for row in per_query.values()) / len(per_query)
        for name in names
    }


@dataclass(frozen=True)
class ThresholdCalibration:
    """Result of calibrating a similarity threshold on labelled hits."""

    # None when no threshold reaches the target precision with enough support.
    threshold: float | None
    # Precision and recall of hits at or above the threshold.
    precision: float
    recall: float
    confident_hits: int
    labelled_hits: int
    relevant_hits: int


def calibrate_similarity_threshold(
    hits: Iterable[tuple[float, bool]],
    *,
    target_precision: float = 0.9,
    min_support: int = 10,
) -> ThresholdCalibration:
    """Lowest similarity threshold whose hits reach `target_precision`.

    `hits` are (vector_similarity, relevant) pairs from labelled searches, e.g.
    the top-k knowledge hits of benchmark questions judged against their gold
    sources. A hit at or above the threshold counts as confident, as in
    KnowledgeStore (min_vector_similarity), so the lowest qualifying threshold
    keeps the most relevant hits. At least `min_support` hits must clear it, so
    a few lucky top hits cannot justify a threshold. Only the vector side of
    `confident` is calibrated; lexical confidence uses term overlap.
    """

    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    labelled = sorted(
        ((float(score), bool(relevant)) for score, relevant in hits if not math.isnan(score)),
        key=lambda hit: hit[0],
        reverse=True,
    )
    total_relevant = sum(relevant for _, relevant in labelled)
    best: ThresholdCalibration | None = None
    seen = relevant_seen = 0
    for index, (score, relevant) in enumerate(labelled):
        seen += 1
        relevant_seen += relevant
        # Ties share one threshold: evaluate only after the last hit of a score.
        if index + 1 < len(labelled) and labelled[index + 1][0] == score:
            continue
        precision = relevant_seen / seen
        if seen >= min_support and precision >= target_precision:
            best = ThresholdCalibration(
                threshold=score,
                precision=precision,
                recall=relevant_seen / total_relevant if total_relevant else 0.0,
                confident_hits=seen,
                labelled_hits=len(labelled),
                relevant_hits=total_relevant,
            )
    return best or ThresholdCalibration(
        threshold=None, precision=0.0, recall=0.0, confident_hits=0,
        labelled_hits=len(labelled), relevant_hits=total_relevant,
    )


async def retrieve(
    store: DocumentStore,
    queries: Mapping[str, str],
    modes: Sequence[SearchMode],
    *,
    k: int,
    concurrency: int = 8,
) -> dict[SearchMode, dict[str, list[ChunkHit]]]:
    """Top-k documents per query for each mode; each query is embedded once."""

    semaphore = asyncio.Semaphore(concurrency)
    needs_embedding = any(mode is not SearchMode.KEYWORD for mode in modes)

    async def run(qid: str, query: str) -> tuple[str, dict[SearchMode, list[ChunkHit]]]:
        async with semaphore:
            embedding = await store.embed_query(query) if needs_embedding else None
            hits = {
                mode: await store.search_documents(
                    query, mode=mode, k=k, query_embedding=embedding
                )
                for mode in modes
            }
        return qid, hits

    results = await asyncio.gather(*(run(qid, query) for qid, query in queries.items()))
    return {mode: {qid: hits[mode] for qid, hits in results} for mode in modes}


def to_rankings(hits: Mapping[str, Sequence[ChunkHit]]) -> Rankings:
    return {qid: [hit.doc_id for hit in query_hits] for qid, query_hits in hits.items()}


def score_modes(
    qrels: Qrels,
    hits: Mapping[SearchMode, Mapping[str, Sequence[ChunkHit]]],
    k_values: Sequence[int] = K_VALUES,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Mean metrics per mode, and the per-query metrics they average."""

    summary, per_query = {}, {}
    for mode, mode_hits in hits.items():
        per_query[mode.value] = per_query_metrics(qrels, to_rankings(mode_hits), k_values)
        summary[mode.value] = mean_metrics(per_query[mode.value])
    return summary, per_query


def format_table(
    title: str, summary: Mapping[str, Mapping[str, float]], k_values: Sequence[int]
) -> str:
    lines = [title, f"  {'mode':<9} {'metric':<7}" + "".join(f"{'@' + str(k):>8}" for k in k_values)]
    for mode, metrics in summary.items():
        for index, metric in enumerate(METRICS):
            label = mode if index == 0 else ""
            values = "".join(f"{metrics[f'{metric}@{k}']:>8.4f}" for k in k_values)
            lines.append(f"  {label:<9} {metric:<7}{values}")
    return "\n".join(lines)


def save_run(
    path: Path,
    *,
    queries: Mapping[str, str],
    qrels: Qrels,
    hits: Mapping[SearchMode, Mapping[str, Sequence[ChunkHit]]],
    summary: Mapping[str, Mapping[str, float]],
    per_query: Mapping[str, Mapping[str, Mapping[str, float]]],
    **info: Any,
) -> None:
    """Write a JSON run: `info`, mean metrics, and each query's top documents and metrics."""

    run = {
        **info,
        "queries": len(queries),
        "metrics": summary,
        "per_query": {
            mode.value: {
                qid: {
                    "query": queries[qid],
                    "relevant": sorted(
                        doc_id for doc_id, grade in qrels[qid].items() if grade > 0
                    ),
                    "retrieved": [hit.doc_id for hit in mode_hits[qid][:TOP_DOCS_SAVED]],
                    **per_query[mode.value][qid],
                }
                for qid in queries
            }
            for mode, mode_hits in hits.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
