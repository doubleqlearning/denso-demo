"""Unit tests for knowledge storage models and retrieval filters (no database)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from backend.src.services.knowledge_extraction import KnowledgeItem, KnowledgeType
from backend.src.storage import (
    KnowledgeRecord,
    KnowledgeSearchHit,
    KnowledgeStatus,
    build_knowledge_filter,
)
from backend.src.storage.knowledge_store import (
    DEFAULT_MIN_VECTOR_SIMILARITY,
    MAX_LEXICAL_TERMS,
    _compile_knowledge_filter,
    lexical_or_query,
    min_vector_similarity_from_env,
)


def _record(**overrides) -> KnowledgeRecord:
    values = {
        "subject": "rear photoelectric sensor",
        "claim": "Oil mist on the lens causes an unstable signal; clean it before replacing.",
        "context": {"machine_id": "M01", "condition": {"environment": "oil mist"}},
        "type": KnowledgeType.HEURISTIC,
        "evidence_id": uuid4(),
        "stated_by": "Maintenance Expert",
        "created_at": datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc),
        "valid_from": date(2026, 1, 1),
        "valid_to": None,
    }
    values.update(overrides)
    return KnowledgeRecord(**values)


class BuildKnowledgeFilterTest(unittest.TestCase):
    def test_no_constraints_returns_none(self) -> None:
        self.assertIsNone(build_knowledge_filter())
        self.assertIsNone(build_knowledge_filter(subject="  ", context={"machine_id": "", "line": None}))

    def test_single_constraint_is_not_wrapped(self) -> None:
        self.assertEqual(
            build_knowledge_filter(knowledge_type="rule"),
            {"type": {"$eq": "rule"}},
        )

    def test_subject_uses_escaped_case_insensitive_substring(self) -> None:
        self.assertEqual(
            build_knowledge_filter(subject=" 50%_sensor "),
            {"subject": {"$ilike": "%50\\%\\_sensor%"}},
        )

    def test_context_filters_keep_unscoped_knowledge(self) -> None:
        self.assertEqual(
            build_knowledge_filter(context={"machine_id": " M01 ", "machine_model": "X200"}),
            {
                "$and": [
                    {
                        "$or": [
                            {"context.machine_id": {"$eq": "M01"}},
                            {"context.machine_id": {"$exists": False}},
                        ]
                    },
                    {
                        "$or": [
                            {"context.machine_model": {"$eq": "X200"}},
                            {"context.machine_model": {"$exists": False}},
                        ]
                    },
                ]
            },
        )

    def test_exact_context_filter_requires_the_key(self) -> None:
        self.assertEqual(
            build_knowledge_filter(
                context={"source_type": "slack"}, exact_context_keys={"source_type"}
            ),
            {"context.source_type": {"$eq": "slack"}},
        )

    def test_valid_on_accepts_open_ended_ranges(self) -> None:
        day = date(2026, 9, 25)
        self.assertEqual(
            build_knowledge_filter(valid_on=day),
            {
                "$and": [
                    {"$or": [{"valid_from": {"$exists": False}}, {"valid_from": {"$lte": day}}]},
                    {"$or": [{"valid_to": {"$exists": False}}, {"valid_to": {"$gte": day}}]},
                ]
            },
        )

    def test_invalid_knowledge_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_knowledge_filter(knowledge_type="opinion")

    def test_compiles_filter_values_as_parameters(self) -> None:
        filters = build_knowledge_filter(
            subject="sensor%' OR TRUE --",
            context={"machine_id": "M01"},
            valid_on=date(2026, 9, 25),
        )
        sql, parameters = _compile_knowledge_filter(filters)

        self.assertNotIn("OR TRUE", sql)
        self.assertIn("k.subject ILIKE :filter_0", sql)
        self.assertIn("k.context ->> 'machine_id'", sql)
        self.assertIn("k.valid_from <=", sql)
        self.assertEqual(parameters["filter_0"], "%sensor\\%' OR TRUE --%")

    def test_filter_compiler_rejects_unknown_fields_and_operators(self) -> None:
        with self.assertRaisesRegex(ValueError, "field"):
            _compile_knowledge_filter({"embedding": {"$eq": "secret"}})
        with self.assertRaisesRegex(ValueError, "operator"):
            _compile_knowledge_filter({"subject": {"$regex": ".*"}})
        with self.assertRaisesRegex(ValueError, "context key"):
            _compile_knowledge_filter({"context.x') OR TRUE --": {"$eq": "1"}})


class KnowledgeRecordTest(unittest.TestCase):
    def test_from_extracted_attaches_ingestion_metadata(self) -> None:
        evidence_id = uuid4()
        item = KnowledgeItem(
            subject="spindle bearing",
            claim="Replace when vibration exceeds 7 mm/s.",
            context={"machine_model": "X200"},
            type=KnowledgeType.RULE,
            valid_from=date(2026, 3, 1),
        )
        record = KnowledgeRecord.from_extracted(
            item, evidence_id=evidence_id, stated_by="Expert A"
        )
        self.assertIsInstance(record.id, UUID)
        self.assertEqual(record.evidence_id, evidence_id)
        self.assertEqual(record.stated_by, "Expert A")
        self.assertEqual(record.type, KnowledgeType.RULE)
        self.assertEqual(record.valid_from, date(2026, 3, 1))
        self.assertEqual(record.conflict_with, [])
        self.assertEqual(record.status, KnowledgeStatus.APPROVED)
        self.assertIsNotNone(record.created_at.tzinfo)
        # The record must not share the extracted item's context dict.
        record.context["machine_id"] = "M02"
        self.assertNotIn("machine_id", item.context)

    def test_embedding_text_leads_with_key_context(self) -> None:
        record = _record(
            context={"shift": "night", "condition": {"temperature": "80C"}, "machine_id": "M01"}
        )
        self.assertEqual(
            record.embedding_text(["machine_id", "machine_model", "condition"]),
            "rear photoelectric sensor: Oil mist on the lens causes an unstable signal; "
            "clean it before replacing.\n"
            'Context: machine_id=M01; condition={"temperature": "80C"}; shift=night',
        )

    def test_embedding_text_sorts_context_without_leading_keys(self) -> None:
        record = _record(context={"shift": "night", "machine_id": "M01"})
        self.assertTrue(
            record.embedding_text().endswith("Context: machine_id=M01; shift=night")
        )

    def test_embedding_text_without_context(self) -> None:
        record = _record(context={})
        self.assertNotIn("Context:", record.embedding_text())

    def test_rendered_structure_and_run_provenance_are_not_indexed(self) -> None:
        # The claim already renders the card and steps; indexing them again would
        # double their weight, and evidence quotes must not mask extraction gaps.
        record = _record(context={"machine_id": "M01", "benchmark_run_id": "private-run",
                                  "steps": ["Isolate power."],
                                  "case_card": {"action": "Clean the approved optical path.",
                                                "evidence": [{"quote": "UNEXTRACTED FACT"}]}})
        for text in ("Clean the approved optical path", "UNEXTRACTED FACT",
                     "Isolate power", "private-run"):
            self.assertNotIn(text, record.lexical_text())
        self.assertIn("machine_id=M01", record.embedding_text())
        self.assertEqual(record.context["case_card"]["evidence"][0]["quote"], "UNEXTRACTED FACT")

    def test_lexical_text_contains_subject_claim_and_stable_context(self) -> None:
        record = _record(
            subject="valve AX-220",
            claim="Keep 80 mm of service clearance.",
            context={"machine_id": "M01", "component_model": "S14"},
        )
        self.assertEqual(record.lexical_text(), record.embedding_text())
        self.assertIn("valve AX-220", record.lexical_text())
        self.assertIn("Keep 80 mm", record.lexical_text())
        self.assertIn("machine_id=M01", record.lexical_text())
        self.assertIn("component_model=S14", record.lexical_text())
        self.assertEqual(record.to_metadata()["search_text"], record.lexical_text())

    def test_embedding_is_excluded_from_dumps_and_repr(self) -> None:
        record = _record(embedding=[0.1, 0.2, 0.3])
        self.assertNotIn("embedding", record.model_dump())
        self.assertNotIn("embedding", record.model_dump_json())
        self.assertNotIn("0.1", repr(record))
        self.assertEqual(record.embedding, [0.1, 0.2, 0.3])

    def test_metadata_round_trips_through_document(self) -> None:
        conflict = uuid4()
        record = _record(conflict_with=[conflict])
        metadata = record.to_metadata()
        self.assertEqual(metadata["type"], "heuristic")
        self.assertEqual(metadata["evidence_id"], str(record.evidence_id))
        self.assertEqual(metadata["conflict_with"], "{" + str(conflict) + "}")

        restored = KnowledgeRecord.from_document(
            Document(id=str(record.id), page_content=record.claim, metadata=metadata)
        )
        self.assertEqual(restored, record)

    def test_empty_conflicts_become_empty_array_literal(self) -> None:
        record = _record()
        self.assertEqual(record.to_metadata()["conflict_with"], "{}")
        restored = KnowledgeRecord.from_document(
            Document(id=str(record.id), page_content=record.claim, metadata=record.to_metadata())
        )
        self.assertEqual(restored.conflict_with, [])

    def test_lifecycle_fields_round_trip_through_document(self) -> None:
        case, newer = uuid4(), uuid4()
        record = _record(
            status=KnowledgeStatus.DRAFT, supported_by=[case], contradicted_by=[],
            superseded_by=newer, revision={"action": "merge", "from": [str(case)]},
        )
        metadata = record.to_metadata()
        self.assertEqual(metadata["status"], "draft")
        self.assertEqual(metadata["supported_by"], "{" + str(case) + "}")
        self.assertEqual(metadata["contradicted_by"], "{}")
        self.assertEqual(metadata["superseded_by"], str(newer))
        restored = KnowledgeRecord.from_document(Document(
            id=str(record.id), page_content=record.claim,
            metadata={**metadata, "revision": '{"action": "merge", "from": ["%s"]}' % case},
        ))
        self.assertEqual(restored, record)

    def test_rows_without_lifecycle_columns_are_approved_and_unlinked(self) -> None:
        record = _record()
        metadata = record.to_metadata()
        for key in ("status", "supported_by", "contradicted_by", "superseded_by", "revision"):
            metadata.pop(key)
        restored = KnowledgeRecord.from_document(
            Document(id=str(record.id), page_content=record.claim, metadata=metadata)
        )
        self.assertEqual(restored.status, KnowledgeStatus.APPROVED)
        self.assertEqual((restored.supported_by, restored.superseded_by, restored.revision),
                         ([], None, {}))

    def test_from_document_accepts_native_database_types(self) -> None:
        record = _record(conflict_with=[uuid4()])
        metadata = {
            **record.to_metadata(),
            "evidence_id": record.evidence_id,
            "conflict_with": list(record.conflict_with),
        }
        restored = KnowledgeRecord.from_document(
            Document(id=str(record.id), page_content=record.claim, metadata=metadata)
        )
        self.assertEqual(restored, record)

    def test_from_document_parses_json_context(self) -> None:
        record = _record()
        metadata = {**record.to_metadata(), "context": '{"machine_id": "M01"}'}
        restored = KnowledgeRecord.from_document(
            Document(id=str(record.id), page_content=record.claim, metadata=metadata)
        )
        self.assertEqual(restored.context, {"machine_id": "M01"})


class KnowledgeSearchHitTest(unittest.TestCase):
    def test_structured_hit_allows_one_retriever_to_be_absent(self) -> None:
        hit = KnowledgeSearchHit(
            record=_record(),
            hybrid_score=1 / 61,
            vector_rank=1,
            vector_distance=0.12,
        )
        self.assertEqual(hit.vector_rank, 1)
        self.assertIsNone(hit.lexical_rank)
        self.assertIsNone(hit.lexical_score)
        self.assertAlmostEqual(hit.vector_similarity, 0.88)
        with self.assertRaises(FrozenInstanceError):
            hit.hybrid_score = 0.5

    def test_lexical_only_hit_has_no_vector_similarity(self) -> None:
        hit = KnowledgeSearchHit(
            record=_record(),
            hybrid_score=1 / 61,
            lexical_rank=1,
            lexical_score=0.8,
        )
        self.assertIsNone(hit.vector_similarity)


class LexicalOrQueryTest(unittest.TestCase):
    def test_joins_terms_with_or_and_keeps_identifiers(self) -> None:
        self.assertEqual(
            lexical_or_query("Jig alignment on AX-220, station 12 (S14.b)?"),
            "Jig or alignment or on or AX-220 or station or 12 or S14.b",
        )

    def test_drops_operators_duplicates_and_negation(self) -> None:
        self.assertEqual(
            lexical_or_query('"sensor" OR -sensor or Sensor lens'),
            "sensor or lens",
        )

    def test_caps_terms_and_handles_empty_input(self) -> None:
        query = " ".join(f"t{i}" for i in range(MAX_LEXICAL_TERMS + 5))
        self.assertEqual(len(lexical_or_query(query).split(" or ")), MAX_LEXICAL_TERMS)
        self.assertEqual(lexical_or_query("?!"), "")


class MinVectorSimilarityConfigTest(unittest.TestCase):
    def test_uses_default_when_environment_is_missing_or_blank(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                min_vector_similarity_from_env(), DEFAULT_MIN_VECTOR_SIMILARITY
            )
        with patch.dict(
            "os.environ", {"KNOWLEDGE_MIN_VECTOR_SIMILARITY": "  "}, clear=True
        ):
            self.assertEqual(
                min_vector_similarity_from_env(), DEFAULT_MIN_VECTOR_SIMILARITY
            )

    def test_reads_threshold_from_environment(self) -> None:
        with patch.dict(
            "os.environ", {"KNOWLEDGE_MIN_VECTOR_SIMILARITY": "0.72"}, clear=True
        ):
            self.assertEqual(min_vector_similarity_from_env(), 0.72)

    def test_rejects_invalid_threshold(self) -> None:
        for value in ("not-a-number", "-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"KNOWLEDGE_MIN_VECTOR_SIMILARITY": value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    min_vector_similarity_from_env()


if __name__ == "__main__":
    unittest.main()
