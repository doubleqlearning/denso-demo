"""Publish and restore portable industrial_qa knowledge and benchmark snapshots.

The Hugging Face dataset contains extracted ``knowledge_items`` and, when
requested, selected question-answering artifacts/reports. Raw conversations and
manual chunks are rebuilt from ``benchmark/public`` during restore; embeddings
are recomputed with the pulling machine's EMBEDDING_* configuration.

Examples:
    uv run python -m backend.knowledge_bases.industrial_qa.snapshot export \
        --output data/industrial_qa_snapshot
    uv run python -m backend.knowledge_bases.industrial_qa.snapshot push \
        --repo-id ORG/industrial-qa-knowledge --include-latest-test
    uv run python -m backend.knowledge_bases.industrial_qa.snapshot pull \
        --repo-id ORG/industrial-qa-knowledge
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.knowledge_bases.base import PROJECT_ROOT
from backend.knowledge_bases.industrial_qa import SPEC
from backend.src.services.knowledge_ingestion import transcript_sha256
from backend.src.storage import ConversationStatus, KnowledgeRecord, KnowledgeStore, StorageError
from backend.src.storage.databases import database_url, server_url_from_env


FORMAT_NAME = "industrial_qa_knowledge_snapshot"
FORMAT_VERSION = 2
KNOWLEDGE_FILE = "knowledge_items.jsonl"
MANIFEST_FILE = "manifest.json"
CARD_FILE = "README.md"
ARTIFACTS_DIR = "benchmark_artifacts"
ARTIFACT_MANIFEST_FILE = "artifact_manifest.json"
ARTIFACT_REPORT_FILE = "REPORT.md"
PUBLIC_DIR = PROJECT_ROOT / "benchmark" / "public"
BENCHMARK_INPUTS = (
    PUBLIC_DIR / "conversations.jsonl",
    PUBLIC_DIR / "corpus_chunks.jsonl",
)
EXCLUDED_TABLES = (
    "raw_conversations",
    SPEC.document_table,
    f"{SPEC.document_table}_terms",
    "rag_indexes",
)

# Files that preserve the benchmark outputs and enough provenance to audit or
# resume the answer/evaluation stages.  The public source data and hidden gold
# labels remain in the repository and are deliberately not duplicated here.
ARTIFACT_FILES = {
    "manifest.json": "run configuration, model settings, and input hashes",
    "question_ids.txt": "exact evaluated question subset",
    "run.json": "extraction and consolidation checkpoint",
    "answers.json": "answer, retrieval, citation, and generation traces",
    "predictions.jsonl": "cited/reranked evaluator input",
    "predictions_uncited.jsonl": "pre-citation-rerank evaluator input",
    "results.json": "full benchmark metrics and judge details",
    "results_uncited.json": "retrieval and MC metrics before citation reranking",
    "inventory_before.json": "database inventory before the test run",
    "inventory_after.json": "database inventory after the test run",
    "corpus_replacement.json": "retained/removed benchmark corpus provenance",
}
REQUIRED_ARTIFACT_FILES = {
    "manifest.json",
    "question_ids.txt",
    "run.json",
    "answers.json",
    "predictions.jsonl",
    "results.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _benchmark_hashes() -> dict[str, str]:
    missing = [path for path in BENCHMARK_INPUTS if not path.is_file()]
    if missing:
        raise ValueError(
            "Missing benchmark public input(s): " + ", ".join(str(path) for path in missing)
        )
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in BENCHMARK_INPUTS
    }


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe {label} path: {value!r}")
    return path


def _safe_run_id(value: Any) -> str:
    run_id = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(f"Invalid benchmark run_id: {run_id!r}")
    return run_id


def find_latest_complete_test_run() -> Path:
    """Return the newest completed test run using its recorded creation time."""

    root = PROJECT_ROOT / "data" / "industrial_qa_runs"
    candidates: list[tuple[str, Path]] = []
    for run_file in root.rglob("run.json"):
        try:
            run = json.loads(run_file.read_text(encoding="utf-8"))
            results = json.loads((run_file.parent / "results.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("split") != "test":
            continue
        questions = run.get("question_ids") or []
        if not questions or results.get("n_answered") != len(questions):
            continue
        if any(not (run_file.parent / name).is_file() for name in REQUIRED_ARTIFACT_FILES):
            continue
        if not (run_file.parent / "snapshot").is_dir():
            continue
        candidates.append((str(run.get("created_at") or ""), run_file.parent))
    if not candidates:
        raise ValueError(f"No completed test benchmark run found under {root}")
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def _metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _artifact_report(run: dict[str, Any], results: dict[str, Any]) -> str:
    overall = results.get("overall") or {}
    metric_order = (
        "recall@1", "recall@3", "recall@5", "recall@10",
        "context_precision@1", "context_precision@3", "context_precision@5",
        "context_precision@10", "mc_accuracy", "answer_score", "answer_correct",
    )
    rows = "\n".join(
        f"| `{name}` | {_metric(overall[name])} |"
        for name in metric_order if name in overall
    )
    fact_capture = results.get("fact_capture") or {}
    fact_line = ""
    if fact_capture:
        fact_line = (
            f"\n- Fact capture: precision {_metric(fact_capture.get('precision'))}, "
            f"recall {_metric(fact_capture.get('recall'))} across "
            f"{fact_capture.get('n_conversations', 0)} conversations.\n"
        )
    return f"""# Benchmark report: {run['run_id']}

- Split: `{run.get('split')}`
- Created: `{run.get('created_at', 'unknown')}`
- Questions: {results.get('n_answered', 0)}/{results.get('n_questions', 0)} answered
{fact_line}
| Metric | Score |
| --- | ---: |
{rows}

`results.json` is the authoritative full report. `answers.json` contains the
retrieval and generation traces, while both `predictions*.jsonl` files can be
fed back to `benchmark/evaluate.py`. `run_checkpoint.json` is the original
historical extraction checkpoint; a pulled `--run-dir` also receives a portable
`run.json` whose evidence UUIDs match the restored database.
"""


def _copy_benchmark_artifacts(output: Path, run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Copy selected QA artifacts and record per-file integrity metadata."""

    destination_root = output / ARTIFACTS_DIR
    if destination_root.exists():
        shutil.rmtree(destination_root)
    result: dict[str, Any] = {}
    for source in run_dirs:
        source = source.resolve()
        run_file = source / "run.json"
        if not run_file.is_file():
            raise ValueError(f"Artifact directory has no run.json: {source}")
        run = json.loads(run_file.read_text(encoding="utf-8"))
        run_id = _safe_run_id(run.get("run_id"))
        if run_id in result:
            raise ValueError(f"Duplicate benchmark artifact run: {run_id}")
        missing = sorted(name for name in REQUIRED_ARTIFACT_FILES if not (source / name).is_file())
        if missing:
            raise ValueError(
                f"Benchmark artifact run {run_id} is incomplete; missing: " + ", ".join(missing)
            )
        if not (source / "snapshot").is_dir():
            raise ValueError(f"Benchmark artifact run {run_id} has no source snapshot directory")

        relative_root = Path(ARTIFACTS_DIR) / run_id
        destination = output / relative_root
        destination.mkdir(parents=True, exist_ok=True)
        roles: dict[str, str] = {}
        for name, role in ARTIFACT_FILES.items():
            path = source / name
            if not path.is_file():
                continue
            destination_name = "run_checkpoint.json" if name == "run.json" else name
            shutil.copy2(path, destination / destination_name)
            roles[destination_name] = role
        shutil.copytree(source / "snapshot", destination / "snapshot")
        for path in sorted((destination / "snapshot").rglob("*")):
            if path.is_file():
                roles[str(path.relative_to(destination))] = "frozen source/prompt/config provenance"

        results = json.loads((source / "results.json").read_text(encoding="utf-8"))
        (destination / ARTIFACT_REPORT_FILE).write_text(
            _artifact_report(run, results), encoding="utf-8"
        )
        roles[ARTIFACT_REPORT_FILE] = "human-readable benchmark summary"
        files = {
            relative: {
                "sha256": _sha256(destination / relative),
                "size_bytes": (destination / relative).stat().st_size,
                "role": role,
            }
            for relative, role in sorted(roles.items())
        }
        artifact_manifest = {
            "format": "industrial_qa_benchmark_artifacts",
            "format_version": 1,
            "run_id": run_id,
            "split": run.get("split"),
            "created_at": run.get("created_at"),
            "question_count": len(run.get("question_ids") or []),
            "files": files,
        }
        _write_json(destination / ARTIFACT_MANIFEST_FILE, artifact_manifest)
        result[run_id] = {
            "path": str(relative_root),
            "split": run.get("split"),
            "question_count": len(run.get("question_ids") or []),
            "artifact_manifest": ARTIFACT_MANIFEST_FILE,
        }
    return result


def _validate_artifacts(directory: Path, manifest: dict[str, Any]) -> None:
    for run_id, info in (manifest.get("artifact_runs") or {}).items():
        _safe_run_id(run_id)
        relative_root = _safe_relative_path(str(info.get("path") or ""), label="artifact")
        root = directory / relative_root
        artifact_manifest_name = str(
            info.get("artifact_manifest") or ARTIFACT_MANIFEST_FILE
        )
        artifact_manifest_path = root / _safe_relative_path(
            artifact_manifest_name, label="artifact manifest"
        )
        if not artifact_manifest_path.is_file():
            raise ValueError(f"Snapshot has no artifact manifest for {run_id}")
        artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        if (
            artifact_manifest.get("format") != "industrial_qa_benchmark_artifacts"
            or artifact_manifest.get("format_version") != 1
            or artifact_manifest.get("run_id") != run_id
        ):
            raise ValueError(f"Unsupported benchmark artifact manifest for {run_id}")
        for relative, file_info in (artifact_manifest.get("files") or {}).items():
            path = root / _safe_relative_path(relative, label="artifact file")
            if not path.is_file():
                raise ValueError(f"Snapshot artifact is missing: {relative_root / relative}")
            if path.stat().st_size != file_info.get("size_bytes") or _sha256(path) != file_info.get("sha256"):
                raise ValueError(f"Checksum mismatch for artifact {relative_root / relative}")


def _sanitized_runs(paths: Sequence[Path], run_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Keep one or more benchmark selections, never their duplicated records."""

    keep = (
        "run_id", "created_at", "split", "seed", "source", "question_ids",
        "conversation_ids", "distractor_conversation_ids", "consolidate",
        "reused_run_ids", "reused_conversation_ids", "new_conversation_ids",
        "extraction",
    )
    result: dict[str, dict[str, Any]] = {}
    declared: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Benchmark run file does not exist: {path}")
        run = json.loads(path.read_text(encoding="utf-8"))
        run_id = run.get("run_id")
        if not run_id:
            raise ValueError(f"Benchmark run file has no run_id: {path}")
        value = {key: run[key] for key in keep if key in run}
        if run_id in result and result[run_id] != value:
            raise ValueError(f"Conflicting benchmark metadata for run {run_id}")
        result[run_id] = value
        declared.update({run_id, *(run.get("reused_run_ids") or [])})
    if paths and run_ids - declared:
        raise ValueError(
            "The supplied --run-file values do not describe snapshot run(s): "
            + ", ".join(sorted(run_ids - declared))
        )
    return result


def dataset_card(manifest: dict[str, Any]) -> str:
    artifact_runs = manifest.get("artifact_runs") or {}
    artifact_section = ""
    if artifact_runs:
        listed = "\n".join(
            f"- `{run_id}` ({info.get('split')}, {info.get('question_count')} questions)"
            for run_id, info in sorted(artifact_runs.items())
        )
        artifact_section = f"""
Included benchmark artifact bundles:

{listed}

Each bundle under `{ARTIFACTS_DIR}/<run_id>/` includes checksummed answer and
retrieval traces, evaluator inputs, full JSON reports, database inventories, the
historical run checkpoint, and the frozen source/prompt/config snapshot. A short
`{ARTIFACT_REPORT_FILE}` makes the headline metrics visible on the Hub.
"""
    return f"""---
pretty_name: Industrial QA extracted knowledge snapshot
tags:
- text
- knowledge-base
- industrial-maintenance
- postgresql
---

# Industrial QA extracted knowledge snapshot

This dataset is a portable snapshot of **{manifest['record_count']} extracted
knowledge records** from the `industrial_qa` knowledge base.

It intentionally excludes raw conversations, manual chunks, PostgreSQL search
indexes, and embeddings. Those inputs already live in the repository under
`benchmark/public/`; the restore command validates their SHA-256 checksums,
recreates the evidence rows, indexes the manuals, remaps `evidence_id`, and
recomputes embeddings locally.

Files:

- `{KNOWLEDGE_FILE}`: extracted records plus `source_conversation_id`.
- `{MANIFEST_FILE}`: format version, counts, checksums, and optional benchmark
  run selection metadata.
{artifact_section}

Restore from the source repository:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot pull \\
  --repo-id <owner/dataset>
```

The destination `knowledge_items` table must be empty, unless it already
contains exactly this snapshot (a resumed pull). Embeddings are not part of
the snapshot and retrieval quality therefore depends on the local
`EMBEDDING_*` configuration.
"""


async def read_database_records() -> list[dict[str, Any]]:
    """Read knowledge plus a stable raw-data reference, never raw text or vectors."""

    engine = create_async_engine(database_url(server_url_from_env(), SPEC.database))
    statement = text(
        """
        SELECT k.id, k.subject, k.claim, k.context, k.type, k.stated_by,
               k.created_at, k.valid_from, k.valid_to, k.conflict_with,
               k.status, k.supported_by, k.contradicted_by, k.superseded_by,
               k.revision, r.metadata ->> 'conversation_id' AS source_conversation_id
        FROM knowledge_items AS k
        LEFT JOIN raw_conversations AS r ON r.id = k.evidence_id
        ORDER BY k.id
        """
    )
    try:
        async with engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
    finally:
        await engine.dispose()
    if not rows:
        raise ValueError(f"{SPEC.database}.knowledge_items is empty")

    exported = []
    for raw in rows:
        row = dict(raw)
        conversation_id = row.pop("source_conversation_id", None)
        if not conversation_id:
            raise ValueError(
                f"Knowledge record {row['id']} has no benchmark conversation_id evidence"
            )
        record = KnowledgeRecord.model_validate({**row, "evidence_id": None})
        payload = record.model_dump(mode="json")
        payload.pop("evidence_id", None)
        payload["source_conversation_id"] = conversation_id
        exported.append(payload)
    return exported


async def export_snapshot(
    output: Path,
    *,
    run_files: Sequence[Path] = (),
    artifact_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Export a self-describing, deterministic snapshot directory."""

    rows = await read_database_records()
    output.mkdir(parents=True, exist_ok=True)
    knowledge_path = output / KNOWLEDGE_FILE
    count = _write_jsonl(knowledge_path, rows)
    run_ids = {
        str(run_id)
        for row in rows
        if (run_id := (row.get("context") or {}).get("benchmark_run_id"))
    }
    artifact_runs = _copy_benchmark_artifacts(output, artifact_dirs)
    artifact_run_files = [directory / "run.json" for directory in artifact_dirs]
    all_run_files = list(dict.fromkeys([*run_files, *artifact_run_files]))
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "knowledge_base": SPEC.name,
        "source_database": SPEC.database,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": count,
        "conversation_count": len({row["source_conversation_id"] for row in rows}),
        "run_ids": sorted(run_ids),
        "knowledge_file": KNOWLEDGE_FILE,
        "knowledge_sha256": _sha256(knowledge_path),
        "benchmark_files": _benchmark_hashes(),
        "excluded_tables": list(EXCLUDED_TABLES),
        "embeddings": {
            "included": False,
            "source_model": os.getenv("EMBEDDING_MODEL"),
            "source_dimension": os.getenv("EMBEDDING_DIM"),
            "restore": "recompute",
        },
        "benchmark_runs": _sanitized_runs(all_run_files, run_ids),
        "artifact_runs": artifact_runs,
    }
    _write_json(output / MANIFEST_FILE, manifest)
    (output / CARD_FILE).write_text(dataset_card(manifest), encoding="utf-8")
    return manifest


def load_snapshot(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate format, checksum, record schema, links, and local raw inputs."""

    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"Snapshot has no {MANIFEST_FILE}: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported snapshot format/version: {manifest.get('format')!r} "
            f"v{manifest.get('format_version')!r}"
        )
    knowledge_path = directory / str(manifest.get("knowledge_file") or KNOWLEDGE_FILE)
    if not knowledge_path.is_file():
        raise ValueError(f"Snapshot has no knowledge file: {knowledge_path}")
    if _sha256(knowledge_path) != manifest.get("knowledge_sha256"):
        raise ValueError(f"Checksum mismatch for {knowledge_path.name}")
    rows = _read_jsonl(knowledge_path)
    if len(rows) != manifest.get("record_count"):
        raise ValueError("Snapshot record_count does not match knowledge_items.jsonl")

    ids: set[UUID] = set()
    records: list[KnowledgeRecord] = []
    for index, row in enumerate(rows, start=1):
        if "embedding" in row or "evidence_id" in row:
            raise ValueError(f"Snapshot row {index} contains a non-portable storage field")
        if not row.get("source_conversation_id"):
            raise ValueError(f"Snapshot row {index} has no source_conversation_id")
        record = KnowledgeRecord.model_validate({**row, "evidence_id": None})
        if record.id in ids:
            raise ValueError(f"Snapshot repeats knowledge id {record.id}")
        ids.add(record.id)
        records.append(record)
    for record in records:
        links = [*record.conflict_with, *record.supported_by, *record.contradicted_by]
        if record.superseded_by:
            links.append(record.superseded_by)
        if unknown := set(links) - ids:
            raise ValueError(
                f"Knowledge record {record.id} links to ids outside the snapshot: "
                + ", ".join(map(str, sorted(unknown)))
            )

    expected_hashes = manifest.get("benchmark_files") or {}
    actual_hashes = _benchmark_hashes()
    for relative, expected in expected_hashes.items():
        if actual_hashes.get(relative) != expected:
            raise ValueError(
                f"Local {relative} does not match the benchmark version used by this snapshot"
            )
    _validate_artifacts(directory, manifest)
    return manifest, rows


def _public_conversations() -> dict[str, dict[str, Any]]:
    return {
        row["conversation_id"]: row
        for row in _read_jsonl(PUBLIC_DIR / "conversations.jsonl")
    }


async def _existing_knowledge_ids(store: KnowledgeStore) -> set[UUID]:
    async with store.engine.connect() as connection:
        return set((await connection.execute(text("SELECT id FROM knowledge_items"))).scalars())


async def _restore_conversations(
    store: KnowledgeStore,
    rows: Sequence[dict[str, Any]],
    *,
    source: str,
) -> tuple[dict[str, UUID], list[UUID]]:
    """Recreate only evidence rows referenced by the extracted knowledge."""

    from backend.knowledge_bases.industrial_qa.pilot import render_transcript

    public = _public_conversations()
    conversation_ids = list(dict.fromkeys(row["source_conversation_id"] for row in rows))
    if missing := set(conversation_ids) - public.keys():
        raise ValueError(
            "Snapshot references conversations absent from benchmark/public: "
            + ", ".join(sorted(missing))
        )

    mapping: dict[str, UUID] = {}
    created: list[UUID] = []
    async with store.engine.connect() as connection:
        existing = (
            await connection.execute(
                text(
                    "SELECT id, metadata FROM raw_conversations "
                    "WHERE metadata ->> 'conversation_id' = ANY(:conversation_ids) "
                    "ORDER BY created_at, id"
                ),
                {"conversation_ids": conversation_ids},
            )
        ).mappings().all()
    for row in existing:
        conversation_id = (row["metadata"] or {}).get("conversation_id")
        mapping.setdefault(conversation_id, row["id"])

    for conversation_id in conversation_ids:
        if conversation_id in mapping:
            continue
        transcript = render_transcript(public[conversation_id])
        conversation = await store.add_conversation(
            transcript,
            metadata={
                "kb": SPEC.name,
                "conversation_id": conversation_id,
                "transcript_sha256": transcript_sha256(transcript),
                "restored_from_snapshot": source,
            },
        )
        mapping[conversation_id] = conversation.id
        created.append(conversation.id)
    return mapping, created


def _records_for_restore(
    rows: Sequence[dict[str, Any]], evidence: dict[str, UUID]
) -> list[KnowledgeRecord]:
    return [
        KnowledgeRecord.model_validate(
            {
                **{key: value for key, value in row.items() if key != "source_conversation_id"},
                "evidence_id": evidence[row["source_conversation_id"]],
            }
        )
        for row in rows
    ]


async def _insert_records(store: KnowledgeStore, records: Sequence[KnowledgeRecord]) -> None:
    """Insert FK targets before records that supersede them."""

    pending = {record.id: record for record in records}
    inserted: set[UUID] = set()
    while pending:
        batch = [
            record
            for record in pending.values()
            if record.superseded_by is None or record.superseded_by in inserted
        ]
        if not batch:
            raise ValueError("Snapshot contains a cycle in superseded_by links")
        await store.add_knowledge(batch)
        inserted.update(record.id for record in batch)
        for record in batch:
            pending.pop(record.id)


async def _index_manuals() -> tuple[int, int]:
    from backend.knowledge_bases.industrial_qa.pilot import (
        DOCUMENT_QUERY_INSTRUCTION,
        chunk_document,
        corpus_chunks,
    )
    from backend.knowledge_bases.stores import open_document_store
    from backend.src.agents.rag.indexing import index_documents

    chunks = corpus_chunks()
    store = await open_document_store(
        SPEC, query_instruction=DOCUMENT_QUERY_INSTRUCTION
    )
    try:
        added = await index_documents(
            store, [chunk_document(chunk) for chunk in chunks.values()], SPEC.chunking
        )
        total = await store.count_chunks()
    finally:
        await store.aclose()
    return added, total


def _selected_run_id(manifest: dict[str, Any], run_id: str | None) -> str:
    choices = manifest.get("benchmark_runs") or {}
    if not choices:
        raise ValueError(
            "Snapshot has no benchmark run metadata; push it with --run-file first"
        )
    if run_id is None:
        if len(choices) != 1:
            raise ValueError(
                "Snapshot has multiple benchmark runs; pass --run-id with one of: "
                + ", ".join(sorted(choices))
            )
        run_id = next(iter(choices))
    if run_id not in choices:
        raise ValueError(
            f"Snapshot has no benchmark metadata for {run_id!r}; available: "
            + ", ".join(sorted(choices))
        )
    return run_id


def _restore_artifact_bundle(
    snapshot_directory: Path,
    destination: Path,
    manifest: dict[str, Any],
    run_id: str,
) -> bool:
    info = (manifest.get("artifact_runs") or {}).get(run_id)
    if not info:
        return False
    source = snapshot_directory / _safe_relative_path(str(info["path"]), label="artifact")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return True


def _write_run_directory(
    directory: Path,
    manifest: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    records: Sequence[KnowledgeRecord],
    *,
    run_id: str | None = None,
) -> None:
    run_id = _selected_run_id(manifest, run_id)
    choices = manifest["benchmark_runs"]
    metadata = choices[run_id]
    by_conversation: dict[str, list[KnowledgeRecord]] = defaultdict(list)
    for row, record in zip(rows, records, strict=True):
        by_conversation[row["source_conversation_id"]].append(record)
    run = dict(metadata)
    selected = set(run.get("conversation_ids") or ())
    run["conversations"] = {
        conversation_id: {
            "status": "processed",
            "evidence_id": str(group[0].evidence_id),
            "records": [record.model_dump(mode="json") for record in group],
            "merged": [],
            "consolidation": {},
        }
        for conversation_id, group in by_conversation.items()
        if conversation_id in selected
    }
    missing = set(run.get("conversation_ids") or ()) - run["conversations"].keys()
    if missing:
        raise ValueError(
            "Benchmark run includes conversations with no stored records: "
            + ", ".join(sorted(missing))
        )
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "run.json", run)
    (directory / "question_ids.txt").write_text(
        "".join(f"{question_id}\n" for question_id in run.get("question_ids") or ()),
        encoding="utf-8",
    )


async def restore_snapshot(
    directory: Path,
    *,
    source: str,
    index_manuals: bool = True,
    run_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, int]:
    """Restore evidence + knowledge and optionally the manual index/run state."""

    manifest, rows = load_snapshot(directory)
    snapshot_ids = {UUID(row["id"]) for row in rows}
    store = await KnowledgeStore.for_kb(SPEC)
    created_conversations: list[UUID] = []
    records: list[KnowledgeRecord] = []
    imported = 0
    try:
        existing = await _existing_knowledge_ids(store)
        if existing and existing != snapshot_ids:
            raise ValueError(
                f"{SPEC.database}.knowledge_items already contains {len(existing)} record(s) "
                "that are not exactly this snapshot; use a fresh database"
            )
        evidence, created_conversations = await _restore_conversations(
            store, rows, source=source
        )
        records = _records_for_restore(rows, evidence)
        if not existing:
            await _insert_records(store, records)
            imported = len(records)
        for conversation_id in created_conversations:
            await store.mark_conversation(
                conversation_id, ConversationStatus.PROCESSED
            )
    finally:
        await store.aclose()

    manual_added = manual_total = 0
    if index_manuals:
        manual_added, manual_total = await _index_manuals()
    artifacts_restored = False
    if run_dir is not None:
        selected_run_id = _selected_run_id(manifest, run_id)
        artifacts_restored = _restore_artifact_bundle(
            directory, run_dir, manifest, selected_run_id
        )
        _write_run_directory(run_dir, manifest, rows, records, run_id=run_id)
    return {
        "knowledge_imported": imported,
        "knowledge_total": len(rows),
        "conversations_created": len(created_conversations),
        "manual_chunks_added": manual_added,
        "manual_chunks_total": manual_total,
        "benchmark_artifact_bundles_restored": int(artifacts_restored),
    }


def _repo_id(value: str | None) -> str:
    repo_id = value or os.getenv("HF_DATASET_REPO_ID")
    if not repo_id:
        raise ValueError("Pass --repo-id or set HF_DATASET_REPO_ID")
    return repo_id


async def push_snapshot(
    repo_id: str,
    *,
    private: bool,
    run_files: Sequence[Path],
    artifact_dirs: Sequence[Path] = (),
) -> str:
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="industrial-qa-snapshot-") as temporary:
        directory = Path(temporary)
        manifest = await export_snapshot(
            directory, run_files=run_files, artifact_dirs=artifact_dirs
        )
        api = HfApi(token=os.getenv("HF_TOKEN") or None)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=directory,
            commit_message=(
                f"Update industrial_qa knowledge snapshot ({manifest['record_count']} records)"
            ),
        )
    return str(commit.commit_url)


async def pull_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    index_manuals: bool,
    run_dir: Path | None,
    run_id: str | None,
) -> dict[str, int]:
    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=[KNOWLEDGE_FILE, MANIFEST_FILE, CARD_FILE, f"{ARTIFACTS_DIR}/**"],
        )
    )
    source = f"hf://datasets/{repo_id}@{revision or 'main'}"
    return await restore_snapshot(
        downloaded,
        source=source,
        index_manuals=index_manuals,
        run_dir=run_dir,
        run_id=run_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Export the current DB snapshot locally")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--run-file", type=Path, action="append", default=[],
        help="Benchmark run.json to preserve; repeat for multiple runs",
    )
    export.add_argument(
        "--artifacts-dir", type=Path, action="append", default=[],
        help="Completed benchmark run directory to bundle; repeat for multiple runs",
    )
    export.add_argument(
        "--include-latest-test", action="store_true",
        help="Bundle the newest completed test run under data/industrial_qa_runs",
    )

    push = commands.add_parser("push", help="Export and upload a Hugging Face dataset")
    push.add_argument("--repo-id", help="owner/dataset (or HF_DATASET_REPO_ID)")
    push.add_argument("--private", action="store_true", help="Create a private dataset repo")
    push.add_argument(
        "--run-file", type=Path, action="append", default=[],
        help="Benchmark run.json to preserve; repeat for multiple runs",
    )
    push.add_argument(
        "--artifacts-dir", type=Path, action="append", default=[],
        help="Completed benchmark run directory to bundle; repeat for multiple runs",
    )
    push.add_argument(
        "--include-latest-test", action="store_true",
        help="Bundle the newest completed test run under data/industrial_qa_runs",
    )

    import_ = commands.add_parser("import", help="Restore a local snapshot directory")
    import_.add_argument("--input", type=Path, required=True)
    import_.add_argument("--skip-manuals", action="store_true")
    import_.add_argument("--run-dir", type=Path)
    import_.add_argument("--run-id", help="Benchmark run to reconstruct in --run-dir")

    pull = commands.add_parser("pull", help="Download a dataset and restore the DB")
    pull.add_argument("--repo-id", help="owner/dataset (or HF_DATASET_REPO_ID)")
    pull.add_argument("--revision", help="Hub branch, tag, or commit (default: main)")
    pull.add_argument("--skip-manuals", action="store_true")
    pull.add_argument("--run-dir", type=Path)
    pull.add_argument("--run-id", help="Benchmark run to reconstruct in --run-dir")
    return parser


def _requested_artifact_dirs(args: argparse.Namespace) -> list[Path]:
    directories = list(args.artifacts_dir)
    if args.include_latest_test:
        directories.append(find_latest_complete_test_run())
    return list(dict.fromkeys(directory.resolve() for directory in directories))


async def run(args: argparse.Namespace) -> int:
    if args.command == "export":
        artifact_dirs = _requested_artifact_dirs(args)
        manifest = await export_snapshot(
            args.output, run_files=args.run_file, artifact_dirs=artifact_dirs
        )
        print(
            f"Exported {manifest['record_count']} knowledge records from "
            f"{manifest['conversation_count']} conversations and "
            f"{len(manifest['artifact_runs'])} benchmark artifact bundle(s) to {args.output}"
        )
        return 0
    if args.command == "push":
        repo_id = _repo_id(args.repo_id)
        artifact_dirs = _requested_artifact_dirs(args)
        url = await push_snapshot(
            repo_id,
            private=args.private,
            run_files=args.run_file,
            artifact_dirs=artifact_dirs,
        )
        print(f"Pushed industrial_qa knowledge snapshot: {url}")
        return 0
    if args.command == "import":
        result = await restore_snapshot(
            args.input,
            source=str(args.input.resolve()),
            index_manuals=not args.skip_manuals,
            run_dir=args.run_dir,
            run_id=args.run_id,
        )
    else:
        repo_id = _repo_id(args.repo_id)
        result = await pull_snapshot(
            repo_id,
            revision=args.revision,
            index_manuals=not args.skip_manuals,
            run_dir=args.run_dir,
            run_id=args.run_id,
        )
    print("Restored " + ", ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except (OSError, ValueError, StorageError) as exc:
        print(f"Error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
