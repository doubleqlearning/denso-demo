from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from backend.knowledge_bases.industrial_qa import snapshot
from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import KnowledgeRecord


def snapshot_row(**overrides):
    values = {
        "id": str(uuid4()),
        "subject": "MR-JE connector",
        "claim": "Re-seat CN1 before replacement.",
        "context": {"benchmark_run_id": "run-1", "machine_model": "MR-JE"},
        "type": "procedure",
        "stated_by": None,
        "created_at": "2026-10-10T00:00:00Z",
        "valid_from": None,
        "valid_to": None,
        "conflict_with": [],
        "status": "draft",
        "supported_by": [],
        "contradicted_by": [],
        "superseded_by": None,
        "revision": {"verify": {"passed": True}},
        "source_conversation_id": "conv_001_1",
    }
    values.update(overrides)
    return values


def write_snapshot(directory: Path, rows: list[dict], *, run=None):
    knowledge = directory / snapshot.KNOWLEDGE_FILE
    snapshot._write_jsonl(knowledge, rows)
    manifest = {
        "format": snapshot.FORMAT_NAME,
        "format_version": snapshot.FORMAT_VERSION,
        "record_count": len(rows),
        "knowledge_file": snapshot.KNOWLEDGE_FILE,
        "knowledge_sha256": snapshot._sha256(knowledge),
        "benchmark_files": snapshot._benchmark_hashes(),
        "benchmark_runs": {run["run_id"]: run} if run else {},
    }
    snapshot._write_json(directory / snapshot.MANIFEST_FILE, manifest)
    return manifest


def write_artifact_run(directory: Path, *, run_id="run-1") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    run = {
        "run_id": run_id,
        "created_at": "2026-10-10T00:00:00+00:00",
        "split": "test",
        "question_ids": ["q1"],
        "conversation_ids": ["conv_001_1"],
    }
    payloads = {
        "manifest.json": {"run_id": run_id, "split": "test"},
        "run.json": run,
        "answers.json": [{"question_id": "q1", "answer": "answer"}],
        "results.json": {
            "split": "test", "n_questions": 1, "n_answered": 1,
            "overall": {"recall@1": 1.0, "mc_accuracy": 1.0},
        },
    }
    for name, value in payloads.items():
        snapshot._write_json(directory / name, value)
    (directory / "question_ids.txt").write_text("q1\n", encoding="utf-8")
    (directory / "predictions.jsonl").write_text(
        '{"type":"answer","question_id":"q1"}\n', encoding="utf-8"
    )
    source = directory / "snapshot"
    source.mkdir()
    (source / "pipeline.py").write_text("# frozen\n", encoding="utf-8")
    return run


class SnapshotFormatTests(unittest.TestCase):
    def test_round_trip_format_has_no_raw_text_embedding_or_evidence_uuid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = snapshot_row()
            write_snapshot(directory, [row])

            manifest, loaded = snapshot.load_snapshot(directory)

        self.assertEqual(manifest["record_count"], 1)
        self.assertEqual(loaded, [row])
        self.assertNotIn("embedding", loaded[0])
        self.assertNotIn("evidence_id", loaded[0])
        self.assertEqual(loaded[0]["source_conversation_id"], "conv_001_1")

    def test_checksum_detects_a_modified_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_snapshot(directory, [snapshot_row()])
            with (directory / snapshot.KNOWLEDGE_FILE).open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                snapshot.load_snapshot(directory)

    def test_links_must_stay_inside_the_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = snapshot_row(supported_by=[str(uuid4())])
            write_snapshot(directory, [row])
            with self.assertRaisesRegex(ValueError, "outside the snapshot"):
                snapshot.load_snapshot(directory)

    def test_run_file_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            path.write_text(json.dumps({
                "run_id": "run-1",
                "split": "train",
                "question_ids": ["q1"],
                "conversation_ids": ["conv_001_1"],
                "conversations": {"conv_001_1": {"records": ["large duplicate"]}},
            }))
            runs = snapshot._sanitized_runs([path], {"run-1"})
        self.assertEqual(runs["run-1"]["question_ids"], ["q1"])
        self.assertNotIn("conversations", runs["run-1"])

    def test_parser_supports_hub_and_local_flows(self):
        parser = snapshot.build_parser()
        push = parser.parse_args([
            "push", "--repo-id", "org/data", "--private",
            "--run-file", "train.json", "--run-file", "test.json",
            "--include-latest-test", "--artifacts-dir", "completed-test",
        ])
        pull = parser.parse_args(["pull", "--repo-id", "org/data", "--revision", "v1"])
        local = parser.parse_args(["import", "--input", "snapshot", "--skip-manuals"])
        self.assertTrue(push.private)
        self.assertEqual(push.run_file, [Path("train.json"), Path("test.json")])
        self.assertTrue(push.include_latest_test)
        self.assertEqual(push.artifacts_dir, [Path("completed-test")])
        self.assertEqual(pull.revision, "v1")
        self.assertTrue(local.skip_manuals)

    def test_artifact_bundle_is_checksummed_and_has_human_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_artifact_run(source)
            output = root / "snapshot"
            output.mkdir()
            rows = [snapshot_row()]
            manifest = write_snapshot(output, rows)
            manifest["artifact_runs"] = snapshot._copy_benchmark_artifacts(
                output, [source]
            )
            snapshot._write_json(output / snapshot.MANIFEST_FILE, manifest)

            loaded, _ = snapshot.load_snapshot(output)
            bundle = output / snapshot.ARTIFACTS_DIR / "run-1"
            report = (bundle / snapshot.ARTIFACT_REPORT_FILE).read_text(encoding="utf-8")

        self.assertIn("run-1", loaded["artifact_runs"])
        self.assertIn("Benchmark report: run-1", report)

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_artifact_run(source)
            output = root / "snapshot"
            output.mkdir()
            manifest = write_snapshot(output, [snapshot_row()])
            manifest["artifact_runs"] = snapshot._copy_benchmark_artifacts(
                output, [source]
            )
            snapshot._write_json(output / snapshot.MANIFEST_FILE, manifest)
            bundle = output / snapshot.ARTIFACTS_DIR / "run-1"
            (bundle / "answers.json").write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Checksum mismatch for artifact"):
                snapshot.load_snapshot(output)


class SnapshotRestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_replaces_portable_source_with_local_evidence_uuid(self):
        evidence_id = uuid4()
        row = snapshot_row()
        (record,) = snapshot._records_for_restore(
            [row], {"conv_001_1": evidence_id}
        )
        self.assertEqual(record.evidence_id, evidence_id)
        self.assertEqual(str(record.id), row["id"])

    async def test_superseded_target_is_inserted_first(self):
        target = KnowledgeRecord(
            subject="old", claim="old claim", type=KnowledgeType.HEURISTIC
        )
        replacement = KnowledgeRecord(
            subject="new",
            claim="new claim",
            type=KnowledgeType.HEURISTIC,
            superseded_by=target.id,
        )
        store = SimpleNamespace(add_knowledge=AsyncMock())

        await snapshot._insert_records(store, [replacement, target])

        first = store.add_knowledge.await_args_list[0].args[0]
        second = store.add_knowledge.await_args_list[1].args[0]
        self.assertEqual([record.id for record in first], [target.id])
        self.assertEqual([record.id for record in second], [replacement.id])

    async def test_run_directory_uses_remapped_evidence_ids(self):
        evidence_id = uuid4()
        row = snapshot_row()
        record = KnowledgeRecord.model_validate({
            **{key: value for key, value in row.items() if key != "source_conversation_id"},
            "evidence_id": evidence_id,
        })
        manifest = {"benchmark_runs": {"run-1": {
            "run_id": "run-1",
            "split": "train",
            "question_ids": ["q1"],
            "conversation_ids": ["conv_001_1"],
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            snapshot._write_run_directory(directory, manifest, [row], [record])
            run = json.loads((directory / "run.json").read_text())
            questions = (directory / "question_ids.txt").read_text()
        self.assertEqual(run["conversations"]["conv_001_1"]["evidence_id"], str(evidence_id))
        self.assertEqual(questions, "q1\n")

    async def test_artifact_restore_preserves_checkpoint_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_artifact_run(source)
            snapshot_dir = root / "snapshot"
            snapshot_dir.mkdir()
            artifact_runs = snapshot._copy_benchmark_artifacts(snapshot_dir, [source])
            destination = root / "restored"

            copied = snapshot._restore_artifact_bundle(
                snapshot_dir,
                destination,
                {"artifact_runs": artifact_runs},
                "run-1",
            )

            self.assertTrue(copied)
            self.assertTrue((destination / "run_checkpoint.json").is_file())
            self.assertTrue((destination / "answers.json").is_file())
            self.assertTrue((destination / "snapshot" / "pipeline.py").is_file())


if __name__ == "__main__":
    unittest.main()
