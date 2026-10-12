"""Unit tests for the transcript → knowledge ingestion pipeline."""

from __future__ import annotations

import asyncio
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.services.knowledge_extraction import (
    KnowledgeExtractionError,
    KnowledgeExtractionResult,
    KnowledgeItem,
    KnowledgeType,
)
from backend.src.services.knowledge_ingestion import (
    IngestionInput,
    IngestionResult,
    KnowledgeIngestionService,
)
from backend.src.storage import ConversationStatus, KnowledgeStatus, RawConversation


class FakeExtractor:
    """Returns `result`, or per-transcript results from a dict keyed by transcript."""

    def __init__(self, result, *, delay: float = 0.0):
        self.result = result
        self.delay = delay
        self.transcripts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def aextract(self, raw_conversation: str) -> KnowledgeExtractionResult:
        self.transcripts.append(raw_conversation)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        result = self.result
        if isinstance(result, dict):
            result = result[raw_conversation]
        if isinstance(result, Exception):
            raise result
        return result


class FakeStore:
    def __init__(
        self,
        *,
        fail_on_write: bool = False,
        fail_on_embed: bool = False,
        fail_write_for: str | None = None,
    ):
        self.fail_on_write = fail_on_write
        self.fail_on_embed = fail_on_embed
        self.fail_write_for = fail_write_for
        self.conversations: dict = {}
        self.knowledge: list = []
        self.embed_calls: list[int] = []

    async def add_conversation(self, transcript, *, stated_by=None, metadata=None):
        conversation = RawConversation(
            id=uuid4(),
            transcript=transcript,
            stated_by=stated_by,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        self.conversations[conversation.id] = conversation
        return conversation

    async def mark_conversation(self, conversation_id, status, *, error=None):
        conversation = self.conversations[conversation_id]
        self.conversations[conversation_id] = conversation.model_copy(
            update={"status": status, "error": error}
        )

    async def embed_knowledge(self, records):
        self.embed_calls.append(len(records))
        if self.fail_on_embed:
            raise RuntimeError("embedding server down")
        for record in records:
            record.embedding = [0.0]

    async def add_knowledge(self, records, *, updates=()):
        if self.fail_on_write:
            raise RuntimeError("database down")
        if self.fail_write_for and any(
            self.conversations[r.evidence_id].transcript == self.fail_write_for
            for r in records
        ):
            raise RuntimeError("row rejected")
        self.knowledge.extend(records)
        return [record.id for record in records]


EXTRACTED = KnowledgeExtractionResult(
    knowledge_items=[
        KnowledgeItem(
            subject="rear photoelectric sensor",
            claim="Clean the lens before replacing the sensor.",
            context={"machine_id": "M01"},
            type=KnowledgeType.PROCEDURE,
        ),
        KnowledgeItem(
            subject="rear photoelectric sensor",
            claim="Oil mist builds up on the lens after a few weeks.",
            context={"machine_id": "M01"},
            type=KnowledgeType.FACT,
        ),
    ]
)


class KnowledgeIngestionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_complete_unit_and_records_extraction_provenance(self) -> None:
        claim = "Only after approval:\n1. Configure the destination.\n2. Enable export.\nVerify delivery; disable if 403 persists."
        extracted = KnowledgeExtractionResult(knowledge_items=[KnowledgeItem(
            subject="Audit export enablement", claim=claim, type=KnowledgeType.PROCEDURE,
        )])
        store = FakeStore()
        provenance = {"version": "complete-units-v2", "prompt_sha256": "abc", "model": "test-model"}
        metadata = {"doc_id": "doc-1", "extraction": {"version": "untrusted"}}
        result = await KnowledgeIngestionService(
            FakeExtractor(extracted), store, extraction_metadata=provenance,
        ).ingest("source", metadata=metadata)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].claim, claim)
        self.assertEqual(store.conversations[result.evidence_id].metadata["extraction"], provenance)
        self.assertEqual(store.conversations[result.evidence_id].metadata["doc_id"], "doc-1")
        self.assertEqual(metadata["extraction"], {"version": "untrusted"})

    async def test_failed_extraction_also_keeps_provenance(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(
            FakeExtractor(KnowledgeExtractionError("bad JSON")), store,
            extraction_metadata={"version": "v2"},
        )
        with self.assertRaises(KnowledgeExtractionError):
            await service.ingest("source")
        conversation, = store.conversations.values()
        self.assertEqual(conversation.metadata["extraction"], {"version": "v2"})
        self.assertEqual(conversation.status, ConversationStatus.FAILED)

    async def test_dataset_context_overrides_extracted_context(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)

        result = await service.ingest(
            "Expert: clean the lens first.",
            context={"machine_id": "M09", "source_type": "slack"},
        )

        for record in result.records:
            self.assertEqual(record.context, {"machine_id": "M09", "source_type": "slack"})
        # The extracted items themselves are left untouched.
        self.assertEqual(EXTRACTED.knowledge_items[0].context, {"machine_id": "M01"})

    async def test_normalizer_runs_per_transcript_before_the_ontology(self) -> None:
        from backend.src.services.ontology import Ontology

        seen = []

        def normalizer(transcript, records):
            seen.append((transcript, [dict(record.context) for record in records]))
            for record in records:
                record.context["machine_id"] = "machine 1"

        ontology = Ontology.from_dict(
            {"keys": {"machine_id": {"values": {"M01": {"aliases": ["machine 1"]}}}}}
        )
        service = KnowledgeIngestionService(
            FakeExtractor(EXTRACTED), FakeStore(), ontology=ontology, normalizer=normalizer
        )
        result = await service.ingest("Expert: clean the lens first.", context={"line": "L1"})

        transcript, contexts = seen[0]
        self.assertEqual(transcript, "Expert: clean the lens first.")
        # The normalizer sees the input context and the whole group.
        self.assertEqual(contexts, [{"machine_id": "M01", "line": "L1"}] * 2)
        self.assertEqual([r.context["machine_id"] for r in result.records], ["M01", "M01"])
        self.assertIn("normalized", result.records[0].revision["ontology"])

    async def test_ingest_links_records_to_evidence(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)

        result = await service.ingest(
            "  Expert: clean the lens first.  ",
            stated_by="Maintenance Expert",
            metadata={"case_id": "1"},
        )

        conversation = store.conversations[result.evidence_id]
        self.assertEqual(conversation.status, ConversationStatus.PROCESSED)
        self.assertEqual(conversation.transcript, "Expert: clean the lens first.")
        self.assertEqual(conversation.metadata, {"case_id": "1"})
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.knowledge_ids, tuple(r.id for r in store.knowledge))
        for record in store.knowledge:
            self.assertEqual(record.evidence_id, result.evidence_id)
            self.assertEqual(record.stated_by, "Maintenance Expert")

    async def test_records_default_to_approved_without_links(self) -> None:
        store = FakeStore()
        await KnowledgeIngestionService(FakeExtractor(EXTRACTED), store).ingest("Expert: x")
        self.assertEqual({r.status for r in store.knowledge}, {KnowledgeStatus.APPROVED})
        self.assertEqual([r.supported_by for r in store.knowledge], [[], []])

    async def test_support_links_resolve_within_each_conversation(self) -> None:
        def extracted(name: str) -> KnowledgeExtractionResult:
            return KnowledgeExtractionResult(knowledge_items=[
                KnowledgeItem(subject=name, claim="Case.", type=KnowledgeType.CASE),
                KnowledgeItem(subject=name, claim="Case.", type=KnowledgeType.CASE),
                KnowledgeItem(subject=name, claim="Lesson.", type=KnowledgeType.HEURISTIC,
                              supported_by=[1, 0]),
            ])

        store = FakeStore()
        service = KnowledgeIngestionService(
            FakeExtractor({"Expert: a": extracted("a"), "Expert: b": extracted("b")}),
            store, knowledge_status=KnowledgeStatus.DRAFT,
        )
        results = await service.ingest_batch(
            [IngestionInput("Expert: a"), IngestionInput("Expert: b")]
        )
        for result in results:
            first, second, lesson = result.records
            self.assertEqual(lesson.supported_by, [second.id, first.id])
            self.assertEqual({r.status for r in result.records}, {KnowledgeStatus.DRAFT})

    async def test_conversation_without_knowledge_is_processed(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(
            FakeExtractor(KnowledgeExtractionResult()), store
        )
        result = await service.ingest("Hello, how are you?")
        self.assertEqual(result.records, ())
        self.assertEqual(
            store.conversations[result.evidence_id].status, ConversationStatus.PROCESSED
        )

    async def test_extraction_failure_marks_conversation_failed(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(
            FakeExtractor(KnowledgeExtractionError("bad json")), store
        )
        with self.assertRaises(KnowledgeExtractionError):
            await service.ingest("Expert: something")

        (conversation,) = store.conversations.values()
        self.assertEqual(conversation.status, ConversationStatus.FAILED)
        self.assertEqual(conversation.error, "KnowledgeExtractionError: bad json")
        self.assertEqual(store.knowledge, [])

    async def test_storage_failure_marks_conversation_failed(self) -> None:
        store = FakeStore(fail_on_write=True)
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)
        with self.assertRaises(RuntimeError):
            await service.ingest("Expert: something")
        (conversation,) = store.conversations.values()
        self.assertEqual(conversation.status, ConversationStatus.FAILED)

    async def test_empty_transcript_is_rejected_before_storage(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)
        with self.assertRaises(ValueError):
            await service.ingest("   ")
        self.assertEqual(store.conversations, {})



class KnowledgeIngestionBatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_results_align_with_inputs_and_failures_are_isolated(self) -> None:
        store = FakeStore()
        extractor = FakeExtractor(
            {
                "a": EXTRACTED,
                "b": KnowledgeExtractionError("bad json"),
                "c": KnowledgeExtractionResult(),
            }
        )
        service = KnowledgeIngestionService(extractor, store)

        results = await service.ingest_batch(
            [
                IngestionInput("a", stated_by="Expert", metadata={"row": "1"}),
                IngestionInput("b"),
                IngestionInput("   "),
                IngestionInput("c"),
            ],
            concurrency=4,
        )

        first, second, third, fourth = results
        self.assertIsInstance(first, IngestionResult)
        self.assertEqual(len(first.records), 2)
        self.assertIsInstance(second, KnowledgeExtractionError)
        self.assertIsInstance(third, ValueError)
        self.assertEqual(fourth.records, ())
        # The empty transcript is never stored.
        self.assertEqual(len(store.conversations), 3)
        statuses = {c.transcript: c.status for c in store.conversations.values()}
        self.assertEqual(
            statuses,
            {
                "a": ConversationStatus.PROCESSED,
                "b": ConversationStatus.FAILED,
                "c": ConversationStatus.PROCESSED,
            },
        )
        self.assertEqual(store.conversations[first.evidence_id].metadata, {"row": "1"})
        self.assertTrue(all(r.stated_by == "Expert" for r in first.records))

    async def test_embeds_the_whole_batch_in_one_call(self) -> None:
        store = FakeStore()
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)
        await service.ingest_batch([IngestionInput(t) for t in "abc"], concurrency=3)
        self.assertEqual(store.embed_calls, [6])
        self.assertEqual(len(store.knowledge), 6)

    async def test_batch_embedding_failure_falls_back_to_per_conversation(self) -> None:
        store = FakeStore(fail_on_embed=True)
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)
        with self.assertLogs("backend.src.services.knowledge_ingestion", "WARNING"):
            results = await service.ingest_batch([IngestionInput(t) for t in "ab"])
        # add_knowledge (here: the fake) is left to embed each group itself.
        self.assertTrue(all(isinstance(r, IngestionResult) for r in results))

    async def test_write_failure_only_fails_that_conversation(self) -> None:
        store = FakeStore(fail_write_for="b")
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), store)
        first, second = await service.ingest_batch([IngestionInput("a"), IngestionInput("b")])
        self.assertIsInstance(first, IngestionResult)
        self.assertIsInstance(second, RuntimeError)
        failed = [c for c in store.conversations.values() if c.transcript == "b"][0]
        self.assertEqual(failed.status, ConversationStatus.FAILED)
        self.assertEqual(failed.error, "RuntimeError: row rejected")

    async def test_limits_concurrent_extractions(self) -> None:
        extractor = FakeExtractor(KnowledgeExtractionResult(), delay=0.01)
        service = KnowledgeIngestionService(extractor, FakeStore())
        await service.ingest_batch([IngestionInput(str(i)) for i in range(10)], concurrency=3)
        self.assertEqual(extractor.max_in_flight, 3)

    async def test_rejects_non_positive_concurrency(self) -> None:
        service = KnowledgeIngestionService(FakeExtractor(EXTRACTED), FakeStore())
        with self.assertRaises(ValueError):
            await service.ingest_batch([IngestionInput("a")], concurrency=0)


if __name__ == "__main__":
    unittest.main()
