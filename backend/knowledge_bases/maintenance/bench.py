"""Run the document baseline and isolated, traceable Case Card extraction.

    python -m backend.knowledge_bases.maintenance.bench ingest --run-dir data/maintenance_runs/v2
    python -m backend.knowledge_bases.maintenance.bench evaluate --run-dir data/maintenance_runs/v2

Uses the actual KnowledgeIngestionService, configured LLM/embedding, PostgreSQL,
and native search_knowledge. Gold annotations are loaded only for assessment.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

from backend.knowledge_bases import get_kb
from backend.knowledge_bases.base import PROJECT_ROOT
from backend.knowledge_bases.maintenance.prepare import prepare, read_source, transcript, VERSION
from backend.scripts.kb import build_parser as kb_parser, evaluate, index_kb_documents, load_retrieval_questions
from backend.src.agents.rag.chunking import Document
from backend.src.agents.rag.evaluation import mean_metrics, per_query_metrics, format_table
from backend.src.services.knowledge_extraction import KnowledgeExtractionService, KnowledgeExtractionError
from backend.src.services.knowledge_ingestion import IngestionInput, IngestionResult, KnowledgeIngestionService
from backend.src.storage import KnowledgeStore
from backend.src.storage.models import KnowledgeRecord


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A checkpoint survives interruption without leaving half a JSON document.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def validate_run_source(run: dict, source: dict) -> None:
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if run["source_sha256"] != fingerprint:
        raise ValueError("Source changed since extraction; ingest a new run before assessment")


async def snapshot(store: KnowledgeStore, path: Path) -> None:
    """Preserve the pre-run index, evidence and knowledge, including vectors."""
    payload = {}
    async with store.engine.connect() as conn:
        for table in ("raw_conversations", "knowledge_items", "chunks_full", "rag_indexes", "terms_full"):
            if await conn.scalar(text("SELECT to_regclass(:table)"), {"table": table}):
                payload[table] = list((await conn.scalars(
                    text(f'SELECT row_to_json(t) FROM "{table}" t')
                )).all())
    encoded = json.dumps(payload, default=str).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(encoded))
    if json.loads(gzip.decompress(path.read_bytes())) != json.loads(encoded):
        raise RuntimeError("Snapshot verification failed; index not changed")
    print("Pre-run snapshot:", {k: len(v) for k, v in payload.items()}, flush=True)


async def ingest_run(args: argparse.Namespace) -> int:
    manifest = prepare()
    source = read_source()
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    spec = get_kb("maintenance")
    extractor = KnowledgeExtractionService.from_env(
        spec.extract_prompt, max_tokens=8192, json_schema=True, output_model=spec.extraction_output_model
    )
    checkpoint = args.run_dir / "extraction.json"
    if checkpoint.exists():
        run = json.loads(checkpoint.read_text())
        if run["source_sha256"] != source_hash or run["extraction"] != extractor.provenance:
            raise ValueError("Source or extractor changed; choose a new --run-dir")
    else:
        run = {"run_id": f'maintenance-v2-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}',
               "started_at": datetime.now(timezone.utc).isoformat(),
               "version": VERSION, "source_sha256": source_hash, "dataset": manifest,
               "extraction_version": spec.extraction_version,
               "extraction": extractor.provenance, "cases": {}}
        save(checkpoint, run)
    store = await KnowledgeStore.for_kb(spec)
    try:
        if not run.get("documents_indexed"):
            await snapshot(store, args.run_dir / "before_ingest.json.gz")
            await index_kb_documents(spec, [Document(id=c["case_id"], title=c["title"], text=transcript(c))
                                            for c in source["cases"]], reindex=True)
            run["documents_indexed"] = True
            save(checkpoint, run)
        class RetryingExtractor:
            # Retry generation, never silently repair or fill extracted facts.
            async def aextract(self, raw_conversation):
                errors = []
                key = hashlib.sha256(raw_conversation.encode()).hexdigest()
                for attempt in range(3):
                    try:
                        result = await extractor.aextract(raw_conversation)
                        run.setdefault("generation_attempts", {})[key] = {"attempts": attempt+1, "errors": errors}
                        return result
                    except KnowledgeExtractionError as exc:
                        errors.append(str(exc))
                run.setdefault("generation_attempts", {})[key] = {"attempts": 3, "errors": errors}
                raise KnowledgeExtractionError(errors[-1])

        service = KnowledgeIngestionService(RetryingExtractor(), store, extraction_metadata={
            "version": spec.extraction_version, **extractor.provenance,
        })
        remaining = [c for c in source["cases"]
                     if run["cases"].get(c["case_id"], {}).get("status") != "processed"]
        started = time.perf_counter()
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start+args.batch_size]
            outcomes = await service.ingest_batch([
                IngestionInput(transcript(c), metadata={
                    "kb": "maintenance", "case_id": c["case_id"], "title": c["title"],
                    "benchmark_version": VERSION, "benchmark_run_id": run["run_id"],
                    "source_file": spec.dataset.path.name,
                }, context={"benchmark_run_id": run["run_id"]}) for c in batch
            ], concurrency=args.concurrency)
            for case, result in zip(batch, outcomes, strict=True):
                cid = case["case_id"]
                if isinstance(result, IngestionResult):
                    run["cases"][cid] = {"status": "processed", "evidence_id": str(result.evidence_id),
                                         "records": [r.model_dump(mode="json") for r in result.records]}
                    print(f'Case {cid}: {len(result.records)} items, evidence {result.evidence_id}', flush=True)
                else:
                    run["cases"][cid] = {"status": "failed", "error": str(result)}
                    print(f'Case {cid}: FAILED {result}', flush=True)
            save(checkpoint, run)
        run["last_ingest_seconds"] = time.perf_counter()-started
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(checkpoint, run)
    finally:
        await store.aclose()
    succeeded = sum(c["status"] == "processed" for c in run["cases"].values())
    count = sum(len(c.get("records", [])) for c in run["cases"].values())
    print(f'Ingested {succeeded}/40 cases; {count} records; {run["last_ingest_seconds"]:.1f}s', flush=True)
    return 0 if succeeded == 40 else 1


def deduplicate_case_hits(hits: list[dict], k: int) -> list[str]:
    """Map native record ranks to unique source cases, preserving first hit."""
    return list(dict.fromkeys(hit["case_id"] for hit in hits))[:k]


async def evaluate_run(args: argparse.Namespace) -> int:
    prepare()
    spec = get_kb("maintenance")
    run = json.loads((args.run_dir / "extraction.json").read_text())
    validate_run_source(run, read_source())
    if len(run["cases"]) != 40 or any(c["status"] != "processed" for c in run["cases"].values()):
        raise ValueError("Complete all 40 source extractions before evaluation")
    save(args.run_dir / "evaluation_inputs.json", {
        "run_id": run["run_id"], "source_sha256": run["source_sha256"],
        "questions_sha256": hashlib.sha256(spec.retrieval_questions.path.read_bytes()).hexdigest(),
        "extraction_prompt_sha256": run["extraction"]["prompt_sha256"],
        "embedding_model": os.getenv("EMBEDDING_MODEL"), "embedding_dim": os.getenv("EMBEDDING_DIM"),
    })
    doc_args = kb_parser().parse_args([
        "evaluate", "maintenance", "--output", str(args.run_dir / "documents_retrieval.json")
    ])
    await evaluate(spec, doc_args)
    queries, qrels = load_retrieval_questions(spec.retrieval_questions)
    by_evidence = {c["evidence_id"]: cid for cid, c in run["cases"].items()}
    # Strict run scope prevents old public knowledge and earlier runs from
    # improving or polluting this score. Use native top-20 record ranking.
    store = await KnowledgeStore.for_kb(spec)
    # Refresh this run only if the semantic indexing contract changed. Evidence
    # excerpts must not substitute for knowledge content in retrieval metrics.
    if run.get("knowledge_index_contract") != "case-card-content-without-quotes-v1":
        records = [KnowledgeRecord.model_validate(r) for c in run["cases"].values() for r in c["records"]]
        await store.embed_knowledge(records)
        async with store.engine.begin() as conn:
            result = await conn.execute(text(
                "UPDATE knowledge_items SET embedding=CAST(:vector AS vector), search_text=:search_text "
                "WHERE id=:id AND context->>'benchmark_run_id'=:run_id"
            ), [{"vector": json.dumps(r.embedding), "search_text": r.lexical_text(spec.embedding_context_keys),
                 "id": r.id, "run_id": run["run_id"]} for r in records])
            if result.rowcount != len(records):
                raise RuntimeError("Run index refresh did not update all records")
        run["knowledge_index_contract"] = "case-card-content-without-quotes-v1"
        save(args.run_dir / "extraction.json", run)
        print(f'Refreshed {len(records)} run records without evidence quotes in search text', flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    async def search(qid: str, question: str):
        async with semaphore:
            hits = await store.search_knowledge(question, k=20, filters={
                "context.benchmark_run_id": {"$eq": run["run_id"]},
            })
        records = [{"knowledge_id": str(h.record.id),
                    "case_id": by_evidence[str(h.record.evidence_id)],
                    "claim": h.record.claim, "subject": h.record.subject,
                    "vector_rank": h.vector_rank, "lexical_rank": h.lexical_rank,
                    "confident": h.confident} for h in hits]
        return qid, records
    try:
        results = dict(await asyncio.gather(*(search(qid, q) for qid, q in queries.items())))
    finally:
        await store.aclose()
    ks = [1, 3, 5, 10]
    # Source-level k: highest native hit for each case, in a fixed top-20
    # knowledge-item candidate pool. Not the same as k items supplied to an LLM.
    rankings = {qid: deduplicate_case_hits(hits, 10) for qid, hits in results.items()}
    metrics = per_query_metrics(qrels, rankings, ks)
    summary = mean_metrics(metrics)
    groups = {}
    from backend.knowledge_bases.maintenance.prepare import questions
    for row in questions(read_source()):
        if row["question_id"] in metrics:
            groups.setdefault(row["retrieval_type"], {})[row["question_id"]] = metrics[row["question_id"]]
    save(args.run_dir / "knowledge_retrieval.json", {
        "run_id": run["run_id"], "mode": "native_search_knowledge_hybrid",
        "ranking_unit": "unique_source_case", "record_candidate_pool": 20,
        "query_instruction": spec.embedding_query_instruction,
        "queries": len(queries), "metrics": summary,
        "groups": {name: mean_metrics(group) for name, group in groups.items()},
        "per_query": {qid: {"query": queries[qid], "relevant": list(qrels[qid]),
                            "retrieved": rankings[qid], "knowledge_hits": results[qid],
                            **metrics[qid]} for qid in queries},
    })
    print(format_table("Native knowledge retrieval: unique cases in top-20 item pool", {"hybrid": summary}, ks))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("ingest", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.concurrency < 1:
        parser.error("batch size and concurrency must be positive")
    load_dotenv(PROJECT_ROOT / ".env")
    return asyncio.run(ingest_run(args) if args.command == "ingest" else evaluate_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
