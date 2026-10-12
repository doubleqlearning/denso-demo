"""Ingest all test conversations and evaluate H1+H3 on every test question.

The held-out check of a frozen solution. Test conversations are extracted
under a new run ID next to the train corpus of --reuse-run-dir, and retrieval
searches their union with the shared manuals, as the runtime would. Train
records and manuals must stay unchanged; consolidation may only change the
test run's records. Each step resumes from run.json.

python -m backend.knowledge_bases.industrial_qa.holdout all --run-dir PATH --reuse-run-dir TRAIN
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import text

from backend.knowledge_bases.industrial_qa import SPEC, pilot
from backend.knowledge_bases.industrial_qa.benchmark_adapter import CompleteAnswer, ScopedRetrieval, benchmark_scope, sha
from backend.knowledge_bases.industrial_qa.train import ingest_missing


SPLIT = "test"
SOURCES = ("holdout.py", "pilot.py", "runs.py", "train.py", "schema.py", "extract.md", "catalog_types.yaml",
           "catalog_assets.yaml", "ontology.yaml", "benchmark_adapter.py", "retrieval.py", "answering.py",
           "pipeline.py", "rerank.md", "answer.md")


def make_run(questions, reused):
    return {
        "run_id": f"industrial-qa-test-full-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "created_at": datetime.now(timezone.utc).isoformat(), "split": SPLIT, "seed": None, "source": None,
        "question_ids": sorted(q["question_id"] for q in questions),
        "conversation_ids": sorted({q["conversation_id"] for q in questions}),
        "distractor_conversation_ids": [], "consolidate": True, "reused_run_ids": reused, "conversations": {},
    }


def prepare(args):
    questions = pilot.gold_questions(SPLIT)
    public = pilot.public_questions(SPLIT)
    if {q["question_id"] for q in questions} != set(public) or len(public) != len(questions):
        raise ValueError("Public test questions do not match the test split")
    if {q["conversation_id"] for q in questions} & {q["conversation_id"] for q in pilot.gold_questions("train")}:
        raise ValueError("Train and test conversations overlap")
    train = json.loads((args.reuse_run_dir / "run.json").read_text())
    if train["split"] != "train":
        raise ValueError("--reuse-run-dir must be a train run")
    reused = list(dict.fromkeys([train["run_id"], *train.get("reused_run_ids", [])]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "run.json"
    run = json.loads(path.read_text()) if path.exists() else make_run(questions, reused)
    if run["split"] != SPLIT or run["question_ids"] != sorted(public) or run["reused_run_ids"] != reused:
        raise ValueError("Checkpoint does not cover the complete test split")
    pilot.save(path, run)
    args.question_ids = args.run_dir / "question_ids.txt"
    args.question_ids.write_text("\n".join(run["question_ids"]) + "\n")
    here = Path(__file__).parent
    sources = [here / name for name in SOURCES]
    manifest = {
        "policy": "H1+H3", "database": SPEC.database, "corpus_policy": "test run next to the reused train corpus; other runs are replaced",
        "run_id": run["run_id"], "split": SPLIT, "question_ids": run["question_ids"],
        "conversation_ids": run["conversation_ids"], "reused_run_ids": reused,
        "reuse_run_dir": str(args.reuse_run_dir.resolve()), "manual_index_rebuilt": False,
        "retrieval_scope": benchmark_scope(run), "consolidation_scope": pilot.run_filter(run),
        "k": args.k, "per_conversation": args.per_conversation, "context_default_characters": 24000,
        "concurrency": args.concurrency, "batch_size": args.batch_size, "verify": args.verify,
        "models": {key: os.getenv(key) for key in
                   ("OPENROUTER_MODEL", "EXTRACTION_MODEL", "ANSWER_MODEL", "EMBEDDING_MODEL",
                    "EMBEDDING_DIM", "ANSWER_THINKING", "EXTRACTION_THINKING", "MODELARK_MODELNAME")},
        "input_sha256": {str(path.relative_to(pilot.PROJECT_ROOT)): sha(path.read_text())
                         for path in [pilot.PUBLIC_DIR / "conversations.jsonl", pilot.PUBLIC_DIR / "corpus_chunks.jsonl",
                                      pilot.PUBLIC_DIR / f"questions_{SPLIT}.jsonl", pilot.GOLD_DIR / f"{SPLIT}.jsonl"]},
        "source_sha256": {str(path.relative_to(pilot.PROJECT_ROOT)): sha(path.read_text()) for path in sources},
    }
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Test configuration changed; preserve this run and use a new run directory")
    snapshot = args.run_dir / "snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    pilot.save(manifest_path, manifest)
    done = sum(c.get("status") == "processed" for c in run["conversations"].values())
    print(f"Test: {len(run['question_ids'])} questions, {len(run['conversation_ids'])} conversations; {done} extracted")
    return run, train


async def inventory(store, run_ids):
    """Fingerprints of the shared manuals and the reused train records."""
    result = {}
    async with store.engine.connect() as connection:
        for name, query, params in (
            ("manual_chunks", "SELECT count(*) AS rows, md5(string_agg(md5(row_to_json(t)::text), '' ORDER BY id)) "
             "AS fingerprint FROM chunks_full t", {}),
            ("reused_records", "SELECT count(*) AS rows, md5(string_agg(md5(row_to_json(t)::text), '' ORDER BY id)) "
             "AS fingerprint FROM knowledge_items t WHERE context->>'benchmark_run_id' = ANY(:runs)", {"runs": run_ids}),
        ):
            result[name] = dict((await connection.execute(text(query), params)).mappings().one())
    return result


async def check_reused(store, train):
    expected = {row["id"]: row for conversation in train["conversations"].values()
                if conversation.get("status") == "processed" for row in conversation["records"]}
    rows = await store.get_knowledge([UUID(key) for key in expected])
    if {str(row.id) for row in rows} != set(expected):
        raise ValueError("Reused train records are missing; restore the train snapshot first")
    for row in rows:
        actual = row.model_dump(mode="json")
        if any(actual[key] != expected[str(row.id)][key] for key in ("subject", "claim", "context", "type", "evidence_id")):
            raise ValueError("Reused train content changed")


async def execute(args):
    run, train = prepare(args)
    if args.command == "select":
        return 0
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        await check_reused(store, train)
        before = await inventory(store, run["reused_run_ids"])
    finally:
        await store.aclose()
    if before["manual_chunks"]["rows"] != len(pilot.corpus_chunks()):
        raise ValueError("The existing manual index is incomplete; refusing to rebuild implicitly")
    path = args.run_dir / "inventory_before.json"
    if path.exists() and json.loads(path.read_text()) != before:
        raise ValueError("Manuals or reused train records changed")
    pilot.save(path, before)
    if args.command in ("ingest", "all"):
        if code := await ingest_missing(args, run):
            return code
    if args.command in ("answer", "evaluate", "all") and any(
        run["conversations"].get(cid, {}).get("status") != "processed" for cid in run["conversation_ids"]
    ):
        raise ValueError("All test conversations must be ingested before evaluation")
    if args.command in ("answer", "all"):
        if code := await pilot.answer(args, run, retrieve=ScopedRetrieval(scope=benchmark_scope(run)),
                                      generate=CompleteAnswer(), init_schema=False, checkpoint_every=4):
            return code
    if args.command in ("evaluate", "all"):
        if code := pilot.evaluate(args, run):
            return code
    store = await pilot.KnowledgeStore.for_kb(SPEC, init_schema=False)
    try:
        after = await inventory(store, run["reused_run_ids"])
    finally:
        await store.aclose()
    if after != before:
        raise ValueError("Manual chunks or reused train records changed during the test run")
    pilot.save(args.run_dir / "inventory_after.json", {"inventory": after, "unchanged": True})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "ingest", "answer", "evaluate", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-run-dir", type=Path, required=True, help="train run whose corpus stays searchable")
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
