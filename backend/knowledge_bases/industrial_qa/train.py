"""Ingest all train conversations and evaluate H1+H3 on every train question.

All runs write to kb_industrial_qa, replacing unrelated older runs.
Reuse baseline records and already indexed manuals; only missing conversations
are extracted. New records use a separate run ID. Retrieval reads their union
with baseline, while consolidation may only change the new run's records.

python -m backend.knowledge_bases.industrial_qa.train all --run-dir PATH --from-run-dir SOURCE
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import asyncio
import json
import os
from pathlib import Path
import shutil
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import text

from backend.knowledge_bases.industrial_qa import SPEC, pilot
from backend.knowledge_bases.industrial_qa.runs import overwrite_run
from backend.knowledge_bases.industrial_qa.benchmark_adapter import CompleteAnswer, ScopedRetrieval, benchmark_scope, sha


def make_run(questions, baseline, directory):
    question_ids = sorted(q["question_id"] for q in questions)
    conversation_ids = sorted({q["conversation_id"] for q in questions})
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Train question IDs are duplicated")
    if baseline["split"] != "train" or set(baseline["conversation_ids"]) - set(conversation_ids):
        raise ValueError("Reuse checkpoint contains conversations outside train")
    reused = {cid: value for cid, value in baseline["conversations"].items()
              if value.get("status") == "processed" and cid in conversation_ids}
    # A JSON copy preserves the historical checkpoint unchanged.
    return {
        "run_id": f"industrial-qa-train-full-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha(str(directory))[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(), "split": "train", "seed": 42,
        "source": None, "question_ids": question_ids, "conversation_ids": conversation_ids,
        "distractor_conversation_ids": [], "consolidate": True,
        "reused_run_ids": list(dict.fromkeys([baseline["run_id"], *baseline.get("reused_run_ids", [])])), "reused_conversation_ids": sorted(reused),
        "new_conversation_ids": sorted(set(conversation_ids) - reused.keys()),
        "conversations": json.loads(json.dumps(reused)), "extraction": baseline["extraction"],
    }


def search_scope(run):
    return benchmark_scope(run)


def source_hashes(directory, manifest):
    """Repairs may inherit the unchanged extraction contract from a parent run."""
    result, visited = {}, set()
    while True:
        directory = directory.resolve()
        if directory in visited:
            raise ValueError("Source manifest ancestry contains a cycle")
        visited.add(directory)
        for path, digest in manifest.get("source_sha256", {}).items():
            result.setdefault(path, digest)  # the newest declared version wins
        parent = manifest.get("from_run_dir")
        if not parent:
            return result
        parent_directory = Path(parent)
        if not parent_directory.is_absolute():
            parent_directory = directory / parent_directory
        parent_file = parent_directory / "manifest.json"
        if not parent_file.is_file():
            return result
        contents = parent_file.read_text()
        if expected := manifest.get("source_manifest_sha256"):
            if sha(contents) != expected:
                raise ValueError("Source parent manifest changed since this snapshot")
        directory, manifest = parent_directory, json.loads(contents)


def prepare(args):
    baseline = json.loads((args.from_run_dir / "run.json").read_text())
    previous = json.loads((args.from_run_dir / "manifest.json").read_text())
    questions = pilot.gold_questions("train")
    public = pilot.public_questions("train")
    if {q["question_id"] for q in questions} != set(public):
        raise ValueError("Public train questions do not match the train split")
    test_convs = {q["conversation_id"] for q in pilot.gold_questions("test")}
    if {q["conversation_id"] for q in questions} & test_convs:
        raise ValueError("Train and test conversations overlap")
    for path, expected in previous["input_sha256"].items():
        if path.startswith("benchmark/public/") and sha((pilot.PROJECT_ROOT / path).read_text()) != expected:
            raise ValueError(f"Reuse inputs changed: {path}")
    hashes = source_hashes(args.from_run_dir, previous)
    for filename in ("schema.py", "extract.md", "catalog_types.yaml", "catalog_assets.yaml", "ontology.yaml"):
        path = Path(__file__).with_name(filename)
        expected = hashes.get(str(path.relative_to(pilot.PROJECT_ROOT)))
        if expected is None:
            raise ValueError(f"Source snapshot has no extraction contract hash: {filename}")
        if sha(path.read_text()) != expected:
            raise ValueError(f"Reuse extraction contract changed: {filename}")
    # Artifacts describe the source dataset, never a target database.
    database = SPEC.database
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "run.json"
    run = json.loads(path.read_text()) if path.exists() else make_run(questions, baseline, args.run_dir)
    if (run["question_ids"] != sorted(public)
            or run["conversation_ids"] != sorted({q["conversation_id"] for q in questions})
            or run["reused_run_ids"] != list(dict.fromkeys([baseline["run_id"], *baseline.get("reused_run_ids", [])]))):
        raise ValueError("Checkpoint does not cover the complete train split")
    pilot.save(path, run)
    args.question_ids = args.run_dir / "question_ids.txt"
    args.question_ids.write_text("\n".join(run["question_ids"]) + "\n")
    sources = [Path(__file__), Path(pilot.__file__), Path(__file__).with_name("runs.py"),
               *(Path(__file__).with_name(name) for name in
                 ("schema.py", "extract.md", "catalog_types.yaml", "catalog_assets.yaml", "ontology.yaml")),
               *(Path(__file__).with_name(n) for n in ("benchmark_adapter.py", "retrieval.py", "answering.py", "pipeline.py", "rerank.md", "answer.md"))]
    manifest = {
        "policy": "H1+H3", "database": database, "source_database": previous.get("database"),
        "corpus_policy": "replace previous runs in the official database", "run_id": run["run_id"], "split": "train",
        "question_ids": run["question_ids"], "conversation_ids": run["conversation_ids"],
        "reused_run_ids": run["reused_run_ids"], "from_run_dir": str(args.from_run_dir.resolve()),
        "reused_conversations": len(run["reused_conversation_ids"]),
        "new_conversations": len(run["new_conversation_ids"]), "manual_index_rebuilt": False,
        "retrieval_scope": search_scope(run), "consolidation_scope": pilot.run_filter(run),
        "k": args.k, "per_conversation": args.per_conversation,
        "context_default_characters": 24000, "concurrency": args.concurrency, "batch_size": args.batch_size,
        "verify": args.verify,
        "models": {key: os.getenv(key) for key in
                   ("OPENROUTER_MODEL", "EXTRACTION_MODEL", "ANSWER_MODEL", "EMBEDDING_MODEL",
                    "EMBEDDING_DIM", "ANSWER_THINKING", "EXTRACTION_THINKING")},
        "input_sha256": {str(path.relative_to(pilot.PROJECT_ROOT)): sha(path.read_text())
                         for path in [pilot.PUBLIC_DIR / "conversations.jsonl", pilot.PUBLIC_DIR / "corpus_chunks.jsonl",
                                      pilot.PUBLIC_DIR / "questions_train.jsonl", pilot.GOLD_DIR / "train.jsonl"]},
        "source_sha256": {str(path.relative_to(pilot.PROJECT_ROOT)): sha(path.read_text()) for path in sources},
    }
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Full-train configuration changed; preserve this run and use a new run directory")
    snapshot = args.run_dir / "snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copyfile(source, snapshot / ("experiment_" + source.name if source.parent.name == "experiments" else source.name))
    pilot.save(manifest_path, manifest)
    print(f"Full train: {len(run['question_ids'])} questions, {len(run['conversation_ids'])} conversations; "
          f"reuse {len(run['reused_conversation_ids'])}, new {len(run['new_conversation_ids'])}")
    return run


async def inventory(store, baseline_id):
    result = {}
    async with store.engine.connect() as connection:
        row = (await connection.execute(text(
            "SELECT count(*) AS rows, md5(string_agg(md5(row_to_json(t)::text), '' ORDER BY id)) "
            "AS fingerprint FROM chunks_full t"
        ))).mappings().one()
        result["manual_chunks"] = dict(row)
        row = (await connection.execute(text(
            "SELECT count(*) AS rows, md5(string_agg(md5(row_to_json(t)::text), '' ORDER BY id)) "
            "AS fingerprint FROM knowledge_items t WHERE context->>'benchmark_run_id' = :run"
        ), {"run": baseline_id})).mappings().one()
        result["baseline_records"] = dict(row)
    return result


async def ingest_missing(args, run):
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        async with overwrite_run(store, run) as report:
            pilot.save(args.run_dir / "corpus_replacement.json", report)
            return await _ingest_missing(args, run)
    finally:
        await store.aclose()


async def _ingest_missing(args, run):
    remaining = [cid for cid in run["conversation_ids"]
                 if run["conversations"].get(cid, {}).get("status") != "processed"]
    if not remaining:
        print("All train conversations already processed; no ingestion repeated")
        return 0
    conversations = pilot.public_conversations()
    extractor = pilot.KnowledgeExtractionService.from_env(
        SPEC.extract_prompt, max_tokens=pilot.EXTRACTION_MAX_TOKENS, json_schema=True,
        output_model=SPEC.extraction_output_model, ontology=SPEC.ontology,
    )
    run["extraction"] = extractor.provenance
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        # Only the new namespace is writable by consolidation. Reused records
        # are searched later via union, but cannot be changed by ingestion.
        service = pilot.KnowledgeIngestionService(
            extractor, store, extraction_metadata={"version": SPEC.extraction_version, **extractor.provenance},
            ontology=SPEC.ontology, normalizer=SPEC.record_normalizer,
            verifier=pilot.KnowledgeVerifier.from_env(args.verify),
            consolidator=pilot.KnowledgeConsolidator.from_env(store, scope=pilot.run_filter(run)),
        )
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start + args.batch_size]
            outcomes = await service.ingest_batch([
                pilot.IngestionInput(pilot.render_transcript(conversations[cid]),
                    metadata={"kb": SPEC.name, "conversation_id": cid, "benchmark_run_id": run["run_id"]},
                    context={"benchmark_run_id": run["run_id"]}) for cid in batch
            ], concurrency=args.concurrency)
            for cid, result in zip(batch, outcomes, strict=True):
                if isinstance(result, pilot.IngestionResult):
                    run["conversations"][cid] = {
                        "status": "processed", "evidence_id": str(result.evidence_id),
                        "records": [r.model_dump(mode="json") for r in result.records],
                        "merged": [r.model_dump(mode="json") for r in result.merged],
                        "consolidation": result.consolidation,
                    }
                    print(f"{cid}: {len(result.records)} records, {result.consolidation}")
                    for note in pilot.review_notes((*result.records, *result.merged)):
                        print(f"  {note}")
                else:
                    run["conversations"][cid] = {"status": "failed", "error": str(result)}
                    print(f"{cid}: FAILED {type(result).__name__}: {result}")
            pilot.save(args.run_dir / "run.json", run)
            print(f"Checkpoint: {sum(c['status'] == 'processed' for c in run['conversations'].values())}"
                  f"/{len(run['conversation_ids'])} train conversations processed")
    finally:
        await store.aclose()
    return int(any(run["conversations"].get(cid, {}).get("status") != "processed" for cid in run["conversation_ids"]))


async def execute(args):
    run = prepare(args)
    if args.command == "select":
        return 0
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        before = await inventory(store, run["reused_run_ids"][0])
        if before["manual_chunks"]["rows"] != len(pilot.corpus_chunks()):
            raise ValueError("The existing manual index is incomplete; refusing to rebuild implicitly")
        path = args.run_dir / "inventory_before.json"
        if path.exists() and json.loads(path.read_text()) != before:
            raise ValueError("Manuals or baseline records changed")
        pilot.save(path, before)
        expected = {row["id"]: row for cid in run["reused_conversation_ids"]
                    for row in run["conversations"][cid]["records"]}
        rows = await store.get_knowledge([UUID(key) for key in expected])
        if {str(row.id) for row in rows} != set(expected):
            raise ValueError("Reused baseline records are missing in the database")
        for row in rows:
            actual = row.model_dump(mode="json")
            if any(actual[key] != expected[str(row.id)][key]
                   for key in ("subject", "claim", "context", "type", "evidence_id")):
                raise ValueError("Reused baseline content changed")
    finally:
        await store.aclose()
    if args.command in ("ingest", "all"):
        if code := await ingest_missing(args, run):
            return code
    if args.command in ("answer", "evaluate", "all") and any(
        run["conversations"].get(cid, {}).get("status") != "processed" for cid in run["conversation_ids"]
    ):
        raise ValueError("All train conversations must be ingested before evaluation")
    if args.command in ("answer", "all"):
        old = json.loads((args.from_run_dir / "answers.json").read_text())
        budgets = {answer["question"]: answer["context_characters"] for answer in old}
        if code := await pilot.answer(args, run,
            retrieve=ScopedRetrieval(budgets, scope=search_scope(run)), generate=CompleteAnswer(),
            init_schema=False, checkpoint_every=4,
        ):
            return code
    if args.command in ("evaluate", "all"):
        if code := pilot.evaluate(args, run):
            return code
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        after = await inventory(store, run["reused_run_ids"][0])
        if after != before:
            raise ValueError("Manual chunks or baseline records changed during full-train run")
        pilot.save(args.run_dir / "inventory_after.json", {"inventory": after, "unchanged": True})
    finally:
        await store.aclose()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "ingest", "answer", "evaluate", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--from-run-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--per-conversation", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verify", choices=("quotes", "llm"), default="quotes")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    if min(args.k, args.concurrency, args.batch_size) < 1 or args.per_conversation < 0:
        parser.error("Invalid retrieval or batching options")
    load_dotenv(pilot.PROJECT_ROOT / ".env")
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
