"""Offline adapter tests: source fidelity, benchmark isolation and shared ingestion."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from backend.knowledge_bases import get_kb
from backend.knowledge_bases.assetopsbench.prepare import (
    CORPUS_COLUMNS, GITHUB_BASE, HF_BASE, QUESTION_FILES, REFERENCE_FILES,
    REFERENCE_ROOT, prepare, question_rows, reference_rows, sha256,
)
from backend.scripts.kb import build_parser, ingest, read_transcripts
from backend.src.agents.agent.tools import search_input_model
from backend.src.services.knowledge_extraction import KnowledgeExtractionService
from backend.tests.test_knowledge_extraction_service import FakeLLM
from backend.tests.test_knowledge_ingestion_service import FakeStore


def fixture_sources() -> dict[str, bytes]:
    payloads = {}
    for source in REFERENCE_FILES:
        record = {source.identity_key: "entry-1", "description": "Sample reference"}
        if source.source_type == "asset_profile":
            record = {
                "assetnum": "Chiller 6", "siteid": "MAIN", "assettype": "CHILLER",
                "sensors": ["Chiller 6 Oil Pressure", "Chiller 6 Supply Temperature"],
                "manufacturer": None,
            }
        if source.path.endswith(".json"):
            raw = json.dumps([record]).encode()
        else:
            handle = io.StringIO(newline="")
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
            raw = handle.getvalue().encode()
        payloads[f"{GITHUB_BASE}/{REFERENCE_ROOT}/{source.path}"] = raw
    for config, path in QUESTION_FILES.items():
        record = {
            "id": 1, "text": f"QUESTION_ONLY_{config}",
            "characteristic_form": "GOLD_ONLY_SENTINEL", "hint": "HINT_ONLY_SENTINEL",
            "failure_event": "LABEL_ONLY_SENTINEL", "deterministic": False,
        }
        payloads[f"{HF_BASE}/{path}"] = json.dumps(record).encode() + b"\n"
    return payloads


def fixture_client(payloads: dict[str, bytes]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=payloads[str(request.url)])
    ))


class PrepareTest(unittest.TestCase):
    def test_preserves_records_and_separates_all_evaluation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory, fixture_client(fixture_sources()) as client:
            output = Path(directory)
            manifest = prepare(output, client=client)
            spec = get_kb("assetopsbench")
            items = list(read_transcripts(output / "corpus.csv", transcript_column=spec.dataset.transcript_column))
            questions = [json.loads(line) for line in (output / "questions.jsonl").read_text().splitlines()]
            with (output / "corpus.csv").open(newline="") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames), CORPUS_COLUMNS)
            self.assertEqual(len(items), len(REFERENCE_FILES))
            self.assertEqual(len({item.metadata["doc_id"] for item in items}), len(items))
            self.assertEqual(len(questions), 6)
            self.assertEqual(len({q["question_id"] for q in questions}), 6)
            self.assertIs(questions[0]["source_record"]["deterministic"], False)
            self.assertEqual(questions[0]["source_record"]["characteristic_form"], "GOLD_ONLY_SENTINEL")
            for token in ("QUESTION_ONLY", "GOLD_ONLY", "HINT_ONLY", "LABEL_ONLY"):
                self.assertNotIn(token, (output / "corpus.csv").read_text())
            self.assertIn('"manufacturer": null', items[0].transcript)
            self.assertIn('"Chiller 6 Oil Pressure"', items[0].transcript)
            self.assertEqual(items[0].metadata["asset_id"], "Chiller 6")
            self.assertEqual(items[0].metadata["site_id"], "MAIN")
            self.assertNotIn("asset_id", items[1].metadata)
            self.assertFalse(manifest["questions_in_corpus"])
            for name, details in manifest["outputs"].items():
                self.assertEqual(details["sha256"], sha256((output / name).read_bytes()))
            # No timestamps or ordering noise; cached rebuild is byte-for-byte reproducible.
            before = {name: (output / name).read_bytes() for name in ("corpus.csv", "questions.jsonl", "manifest.json")}
            with httpx.Client(transport=httpx.MockTransport(
                lambda request: self.fail("Offline rebuild attempted a network call")
            )) as offline_client:
                prepare(output, client=offline_client, offline=True)
            self.assertEqual(before, {name: (output / name).read_bytes() for name in before})

    def test_stable_document_ids_and_explicit_non_exhaustive_flag(self) -> None:
        source = next(s for s in REFERENCE_FILES if s.source_type == "failure_modes")
        records = [{"asset_class": "pump", "failure_modes": ["seal leakage"], "exhaustive": False},
                   {"asset_class": "motor", "failure_modes": ["bearing damage"]}]
        rows = reference_rows(source, json.dumps(records).encode())
        reversed_rows = reference_rows(source, json.dumps(records[::-1]).encode())
        self.assertEqual([r["doc_id"] for r in rows], [r["doc_id"] for r in reversed_rows[::-1]])
        self.assertIn('"exhaustive": false', rows[0]["document"])
        self.assertEqual(rows[0]["asset_type"], "pump")
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            reference_rows(source, json.dumps([records[0], records[0]]).encode())

    def test_invalid_sources_fail_without_replacing_previous_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, fixture_client(fixture_sources()) as client:
            output = Path(directory)
            prepare(output, client=client)
            before = (output / "corpus.csv").read_bytes()
            broken = fixture_sources()
            broken[f"{HF_BASE}/{QUESTION_FILES['scenarios']}"] = b'{"id": 1}\n'
            with fixture_client(broken) as broken_client:
                with self.assertRaisesRegex(ValueError, "nonempty text"):
                    prepare(output, client=broken_client, force=True)
            self.assertEqual((output / "corpus.csv").read_bytes(), before)

    def test_rejects_missing_cache_malformed_records_and_duplicate_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Offline source missing"):
                prepare(Path(directory), offline=True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            prepare(offline=True, force=True)
        for raw in (b'{}', b'[]', b'[{}]', b'[{"assetnum": ""}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reference_rows(REFERENCE_FILES[0], raw)
        with self.assertRaisesRegex(ValueError, "malformed CSV"):
            reference_rows(REFERENCE_FILES[1], b"asset,description\npump\n")
        raw = b'{"id": 1, "text": "q"}\n'
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            question_rows("scenarios", raw + raw)

    def test_registered_for_ingestion_and_dynamic_retrieval(self) -> None:
        spec = get_kb("assetopsbench")
        self.assertEqual(spec.database, "kb_assetopsbench")
        self.assertEqual(spec.dataset.id_column, "doc_id")
        self.assertEqual(spec.dataset.path.name, "corpus.csv")
        self.assertEqual(spec.extract_prompt.count("{RAW_CONVERSATION}"), 1)
        model = search_input_model(spec.context_filters)
        parsed = model(query=" Which sensors are installed? ", asset_id=" Chiller 6 ", site_id=" MAIN ")
        self.assertEqual((parsed.asset_id, parsed.site_id), ("Chiller 6", "MAIN"))
        self.assertIn("source_type", model.model_json_schema()["properties"])
        self.assertTrue(spec.embedding_query_instruction)


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_csv_uses_shared_extraction_ingestion_and_resume(self) -> None:
        spec = get_kb("assetopsbench")
        llm = FakeLLM(json.dumps({"knowledge_items": [{
            "subject": "Chiller 6 installed sensors", "claim": "Chiller 6 lists an Oil Pressure sensor.",
            "type": "fact", "context": {"site_id": "WRONG"},
        }]}))
        extractor = KnowledgeExtractionService(llm, spec.extract_prompt)
        store = FakeStore()
        store.processed_metadata_values = mock.AsyncMock(return_value=set())
        store.aclose = mock.AsyncMock()
        with tempfile.TemporaryDirectory() as directory, fixture_client(fixture_sources()) as client:
            output = Path(directory)
            prepare(output, client=client)
            args = build_parser().parse_args([
                "ingest", "assetopsbench", "--file", str(output / "corpus.csv"), "--limit", "1",
            ])
            with (
                mock.patch("backend.scripts.kb.KnowledgeExtractionService.from_env", return_value=extractor),
                mock.patch("backend.scripts.kb.KnowledgeStore.for_kb", new_callable=mock.AsyncMock, return_value=store),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(await ingest(spec, args), 0)
                record, = store.knowledge
                self.assertEqual(record.context["asset_id"], "Chiller 6")
                self.assertEqual(record.context["site_id"], "MAIN")
                self.assertEqual(record.context["source_type"], "asset_profile")
                evidence = store.conversations[record.evidence_id]
                self.assertEqual(evidence.metadata["extraction"]["version"], spec.extraction_version)
                self.assertIn("raw.githubusercontent.com/IBM/AssetOpsBench/", evidence.metadata["source_url"])
                self.assertIn("Chiller 6 Oil Pressure", evidence.transcript)
                # All IDs processed -> no further evidence or model invocation.
                store.processed_metadata_values.return_value = {
                    item.metadata["doc_id"] for item in read_transcripts(
                        output / "corpus.csv", transcript_column="document",
                    )
                }
                self.assertEqual(await ingest(spec, args), 0)
                self.assertEqual(len(store.conversations), 1)
                store.processed_metadata_values.assert_awaited_with("doc_id")


if __name__ == "__main__":
    unittest.main()
