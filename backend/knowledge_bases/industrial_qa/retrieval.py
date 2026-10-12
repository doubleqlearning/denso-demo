"""Industrial QA H1 retrieval over stores; no benchmark files or run directories."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any

from pydantic import BaseModel

from backend.knowledge_bases.industrial_qa import CATALOG
from backend.knowledge_bases.industrial_qa.catalog import _spelling_re
from backend.src.agents.rag.store import ChunkHit
from backend.src.services import llm
from backend.src.storage import KnowledgeRecord
from backend.src.storage.knowledge_store import build_knowledge_filter


DOCUMENT_QUERY_INSTRUCTION = (
    "Given a question from a factory maintenance engineer, retrieve the equipment manual passage that answers it"
)
UNSCOPED_CONTEXT = frozenset({"case_card", "steps", "source_turns", "benchmark_run_id"})


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 8
    per_conversation: int = 3
    context_characters: int = 24000
    knowledge_candidates: int = 20
    manual_candidates: int = 48
    manual_rerank_candidates: int = 16

    def __post_init__(self):
        if min(self.k, self.context_characters, self.knowledge_candidates,
               self.manual_candidates, self.manual_rerank_candidates) < 1 or self.per_conversation < 0:
            raise ValueError("Invalid industrial QA retrieval limits")


def detected_scope(question: str):
    assets = CATALOG.asset_ids(question)
    families = {CATALOG.family_of(asset) for asset in assets}
    for family in CATALOG.families.values():
        if any(_spelling_re(value).search(question) for value in (family.name, *family.aliases)):
            families.add(family.name)
    component = CATALOG.component(question)
    if component:
        families.add(CATALOG.components[component].family)
    if variant := CATALOG.machine_variant(question):
        families.add(variant[0])
    families.discard(None)
    return assets, next(iter(families)) if len(families) == 1 else None


def manual_query(question, family):
    text = CATALOG.without_asset_ids(question)
    for pattern, _, _ in CATALOG._site_spellings:
        text = pattern.sub(" ", text)
    text = re.sub(r"\b(?:line|station)\s*\d+\b|\bcell\s+[A-Z]\b", " ", text, flags=re.I)
    return f"{family or ''} {re.sub(r'\s+', ' ', text).strip()}".strip()


def render_record(label, record):
    scope = {key: value for key, value in record.context.items()
             if key not in UNSCOPED_CONTEXT and value not in (None, "", [])}
    head = f"[{label}] {record.type.value}: {record.subject}"
    if scope:
        head += " " + json.dumps(scope, ensure_ascii=False)
    return f"{head}\n{record.claim}"


def reference_id(passage):
    return (passage.metadata.get("reference_doc_id") or passage.metadata.get("doc_id")
            or re.sub(r"_p\d+_c\d+$", "", passage.doc_id))


def page_start(passage):
    if passage.metadata.get("page_start") is not None:
        return int(passage.metadata["page_start"])
    match = re.search(r"_p(\d+)_c\d+$", passage.doc_id)
    return int(match[1]) if match else None


def render_passage(label, passage):
    meta = passage.metadata
    page = page_start(passage)
    pages = f", p. {page}" if page is not None else ""
    if page is not None and meta.get("page_end") not in (None, page):
        pages += f"-{meta['page_end']}"
    return f"[{label}] manual {reference_id(passage)}{pages}, {passage.title}\n{passage.text}"


@dataclass(frozen=True)
class Source:
    label: str
    record: KnowledgeRecord | None = None
    passage: ChunkHit | None = None
    candidate_label: str = ""
    confident: bool | None = None

    def __post_init__(self):
        if (self.record is None) == (self.passage is None):
            raise ValueError("A source must have exactly one record or passage")

    @property
    def text(self):
        return render_record(self.label, self.record) if self.record else render_passage(self.label, self.passage)

    @property
    def kind(self):
        return "knowledge" if self.record else "document"


@dataclass(frozen=True)
class RetrievalResult:
    sources: tuple[Source, ...]
    assets: tuple[str, ...] = ()
    family: str | None = None
    candidates: int = 0
    usage: dict[str, Any] | None = None


class Ranking(BaseModel):
    labels: list[str]


class IndustrialRetriever:
    def __init__(self, ranker, *, config=RetrievalConfig(), prompt: str):
        self.ranker, self.config, self.prompt = ranker, config, prompt

    async def retrieve(self, question, knowledge, documents, *, scope=None, context_characters=None, config=None):
        cfg = config if config is not None else self.config
        assets, family = detected_scope(question)
        local = CATALOG.lookup_scope(assets, family)
        filters = {"$and": [scope, local]} if scope and local else scope or local
        hits, passages = await asyncio.gather(
            knowledge.search_knowledge(question, k=cfg.knowledge_candidates, filters=filters),
            documents.search(manual_query(question, family), k=cfg.manual_candidates),
        )
        if not assets and cfg.per_conversation:
            counts = defaultdict(int)
            kept = []
            for hit in hits:
                key = str(hit.record.evidence_id)
                if counts[key] < cfg.per_conversation:
                    kept.append(hit)
                    counts[key] += 1
            hits = kept
        elif assets and not hits:
            general = {"context.machine_id": {"$exists": False}}
            compatible = build_knowledge_filter(context={"machine_model": family})
            parts = [part for part in (scope, general, compatible) if part]
            hits = await knowledge.search_knowledge(question, k=cfg.knowledge_candidates, filters={"$and": parts})
        expected_manual = CATALOG.families[family].manual if family else None
        allowed = [p for p in passages if not expected_manual or reference_id(p) == expected_manual]
        if not allowed:
            allowed = passages
        selected_passages = list({p.chunk_id: p for p in allowed}.values())[:cfg.manual_rerank_candidates]
        seeds = [p for p in selected_passages[:3] if re.search(r"procedure|steps?|connect|configur", p.text, re.I)]
        if seeds:
            neighbors = await documents.adjacent_chunks([p.chunk_id for p in seeds])
            for neighbor in neighbors:
                if any(reference_id(neighbor) == reference_id(seed)
                       and neighbor.title.split(" > ")[0] == seed.title.split(" > ")[0]
                       and (page_start(seed) is None or page_start(neighbor) is None
                            or abs(page_start(seed) - page_start(neighbor)) <= 2) for seed in seeds):
                    selected_passages.append(neighbor)
        selected_passages = list({p.chunk_id: p for p in selected_passages}.values())
        candidates = {f"K{i}": Source(f"K{i}", record=hit.record, confident=hit.confident) for i, hit in enumerate(hits, 1)}
        candidates.update({f"D{i}": Source(f"D{i}", passage=p) for i, p in enumerate(selected_passages, 1)})
        if not candidates:
            return RetrievalResult((), tuple(assets), family)
        values = {"QUESTION": question, "SOURCES": "\n\n".join(s.text for s in candidates.values())}
        response = await llm.ainvoke(self.ranker,
            re.sub(r"\{(QUESTION|SOURCES)\}", lambda m: values[m[1]], self.prompt))
        ranking = llm.validated(Ranking, llm.response_json(response, task="source ranking", env_prefix="ANSWER"), task="source ranking")
        if set(ranking.labels) - candidates.keys():
            raise llm.LLMResponseError("Reranker returned unknown source labels")
        counts, used, sources = {"K": 0, "D": 0}, 0, []
        budget = cfg.context_characters if context_characters is None else context_characters
        for label in dict.fromkeys(ranking.labels):
            item = candidates[label]
            kind, size = label[0], len(item.text)
            if counts[kind] >= cfg.k or used + size > budget:
                continue
            counts[kind] += 1
            sources.append(Source(f"{kind}{counts[kind]}", item.record, item.passage, label, item.confident))
            used += size
        if not sources:
            raise llm.LLMResponseError("Reranker selected no evidence within context budget")
        return RetrievalResult(tuple(sources), tuple(assets), family, len(candidates), getattr(response, "usage_metadata", None))
