"""Integration tests against a real PostgreSQL + pgvector database.

Run with the compose database up:
    docker compose -f deployments/docker-compose.yml up -d
    TEST_DATABASE_URL=postgresql+psycopg://voice_agent:voice_agent@localhost:5434/voice_agent \
        uv run python -m unittest backend.tests.test_knowledge_store_integration -v

Tables are created in a throwaway schema, so existing data is untouched.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import (
    ConversationStatus,
    KnowledgeRecord,
    KnowledgeStore,
    StorageError,
    build_knowledge_filter,
)


DATABASE_URL = os.getenv("TEST_DATABASE_URL")
DIM = 16


@unittest.skipUnless(DATABASE_URL, "TEST_DATABASE_URL is not set")
class KnowledgeStoreIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.schema = f"test_{uuid4().hex[:12]}"
        self.store = KnowledgeStore.from_url(
            DATABASE_URL,
            DeterministicFakeEmbedding(size=DIM),
            DIM,
            schema=self.schema,
        )
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA {self.schema}"))

    async def asyncTearDown(self) -> None:
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {self.schema} CASCADE"))
        await self.store.aclose()

    async def _seed(self) -> tuple[UUID, dict[str, KnowledgeRecord]]:
        await self.store.init_schema()
        conversation = await self.store.add_conversation(
            "Expert: on M01 clean the rear sensor lens first.",
            stated_by="Maintenance Expert",
            metadata={"case_id": "1"},
        )
        records = {
            "m01": KnowledgeRecord(
                subject="rear photoelectric sensor",
                claim="Clean the lens before replacing the sensor.",
                context={"machine_id": "M01", "condition": {"environment": "oil mist"}},
                type=KnowledgeType.PROCEDURE,
                evidence_id=conversation.id,
                stated_by="Maintenance Expert",
            ),
            "m02": KnowledgeRecord(
                subject="rear photoelectric sensor",
                claim="Check the connector before anything else.",
                context={"machine_id": "M02"},
                type=KnowledgeType.HEURISTIC,
                evidence_id=conversation.id,
            ),
            "general": KnowledgeRecord(
                subject="photoelectric sensor",
                claim="Operating temperature must stay below 60C.",
                type=KnowledgeType.FACT,
                evidence_id=conversation.id,
            ),
            "expired": KnowledgeRecord(
                subject="photoelectric sensor",
                claim="Use lens cleaner type A.",
                type=KnowledgeType.RULE,
                evidence_id=conversation.id,
                valid_from=date(2024, 1, 1),
                valid_to=date(2025, 1, 1),
            ),
        }
        records["general"].conflict_with = [records["expired"].id]
        await self.store.add_knowledge(list(records.values()))
        return conversation.id, records

    async def test_init_schema_is_idempotent(self) -> None:
        await self.store.init_schema()
        await self.store.init_schema()

    async def test_init_schema_backfills_legacy_rows_and_creates_gin_index(self) -> None:
        await self.store.init_schema()
        table = f'"{self.schema}"."knowledge_items"'
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN search_vector"))
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN search_text"))
            record_id = uuid4()
            await conn.execute(
                text(
                    f"INSERT INTO {table} "
                    "(id, claim, embedding, subject, type, context, conflict_with) "
                    "VALUES (:id, :claim, CAST(:embedding AS vector), :subject, "
                    ":type, CAST(:context AS jsonb), CAST(:conflicts AS uuid[]))"
                ),
                {
                    "id": record_id,
                    "claim": "Clean connector S14 before replacement.",
                    "embedding": "[" + ",".join("0" for _ in range(DIM)) + "]",
                    "subject": "connector",
                    "type": "procedure",
                    "context": '{"machine_id": "M01"}',
                    "conflicts": "{}",
                },
            )

        await self.store.init_schema()

        async with self.store.engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        f"SELECT search_text, search_vector::text AS search_vector "
                        f"FROM {table} WHERE id = :id"
                    ),
                    {"id": record_id},
                )
            ).mappings().one()
            index_exists = await conn.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_indexes "
                    "WHERE schemaname = :schema AND indexname = :index_name)"
                ),
                {
                    "schema": self.schema,
                    "index_name": "knowledge_items_search_vector_gin_idx",
                },
            )
        self.assertIn("M01", row["search_text"])
        self.assertIn("'s14'", row["search_vector"])
        self.assertTrue(index_exists)

    async def test_dimension_mismatch_is_detected(self) -> None:
        await self.store.init_schema()
        other = KnowledgeStore(
            self.store.engine, DeterministicFakeEmbedding(size=8), 8, schema=self.schema
        )
        with self.assertRaises(StorageError):
            await other.init_schema()

    async def test_conversation_round_trip(self) -> None:
        await self.store.init_schema()
        conversation = await self.store.add_conversation(
            "hello", stated_by="Expert", metadata={"case_id": "7"}
        )
        self.assertEqual(conversation.status, ConversationStatus.PENDING)
        await self.store.mark_conversation(
            conversation.id, ConversationStatus.FAILED, error="boom"
        )
        stored = await self.store.get_conversation(conversation.id)
        self.assertEqual(stored.metadata, {"case_id": "7"})
        self.assertEqual(stored.status, ConversationStatus.FAILED)
        self.assertEqual(stored.error, "boom")
        self.assertIsNone(await self.store.get_conversation(uuid4()))

    async def test_records_round_trip_with_native_types(self) -> None:
        evidence_id, records = await self._seed()
        wanted = [records["general"].id, uuid4(), records["m01"].id]
        general, m01 = await self.store.get_knowledge(wanted)

        self.assertEqual(general.id, records["general"].id)
        self.assertEqual(general.conflict_with, [records["expired"].id])
        self.assertEqual(m01.context, records["m01"].context)
        self.assertEqual(m01.evidence_id, evidence_id)
        self.assertEqual(m01.type, KnowledgeType.PROCEDURE)
        self.assertEqual(m01.conflict_with, [])
        self.assertIsNone(m01.embedding)
        (expired,) = await self.store.get_knowledge([records["expired"].id])
        self.assertEqual((expired.valid_from, expired.valid_to), (date(2024, 1, 1), date(2025, 1, 1)))

    async def test_search_filters(self) -> None:
        _, records = await self._seed()

        async def ids(**filters) -> set[UUID]:
            hits = await self.store.search_knowledge(
                "sensor", filters=build_knowledge_filter(**filters), k=10
            )
            for hit in hits:
                self.assertIsInstance(hit.hybrid_score, float)
                self.assertTrue(
                    hit.vector_rank is not None or hit.lexical_rank is not None
                )
                if hit.vector_rank is not None:
                    self.assertIsInstance(hit.vector_distance, float)
                if hit.lexical_rank is not None:
                    self.assertIsInstance(hit.lexical_score, float)
                    self.assertGreaterEqual(hit.lexical_matched_terms, 1)
                self.assertIsInstance(hit.confident, bool)
            return {hit.record.id for hit in hits}

        everything = {record.id for record in records.values()}
        self.assertEqual(await ids(), everything)
        self.assertEqual(await ids(context={"machine_id": "M01"}), everything - {records["m02"].id})
        self.assertEqual(
            await ids(valid_on=date(2026, 9, 25)), everything - {records["expired"].id}
        )
        self.assertEqual(await ids(knowledge_type="heuristic"), {records["m02"].id})
        self.assertEqual(
            await ids(subject="REAR photo"), {records["m01"].id, records["m02"].id}
        )
        self.assertEqual(
            await ids(context={"machine_id": "M01"}, subject="rear", valid_on=date(2026, 9, 25)),
            {records["m01"].id},
        )

    async def test_indexed_context_keys_exact_filter_and_resume_ids(self) -> None:
        store = KnowledgeStore.from_url(
            DATABASE_URL,
            DeterministicFakeEmbedding(size=DIM),
            DIM,
            schema=self.schema,
            embedding_context_keys=["title"],
            indexed_context_keys=["source_type"],
        )
        try:
            await store.init_schema()
            done = await store.add_conversation("Slack thread", metadata={"doc_id": "d1"})
            await store.add_conversation("Pending email", metadata={"doc_id": "d2"})
            await store.mark_conversation(done.id, ConversationStatus.PROCESSED)
            self.assertEqual(await store.processed_metadata_values("doc_id"), {"d1"})

            slack, unscoped = (
                KnowledgeRecord(
                    subject="eng-security",
                    claim="Quarantine partner logs before deleting them.",
                    context=context,
                    type=KnowledgeType.RULE,
                    evidence_id=done.id,
                )
                for context in ({"source_type": "slack", "title": "eng-security"}, {})
            )
            await store.add_knowledge([slack, unscoped])
            self.assertEqual(
                slack.to_metadata(["title"])["search_text"].splitlines()[-1],
                "Context: title=eng-security; source_type=slack",
            )

            hits = await store.search_knowledge(
                "quarantine partner logs",
                filters=build_knowledge_filter(
                    context={"source_type": "slack"}, exact_context_keys={"source_type"}
                ),
            )
            self.assertEqual([hit.record.id for hit in hits], [slack.id])

            async with store.engine.connect() as conn:
                indexes = set(
                    (
                        await conn.execute(
                            text("SELECT indexname FROM pg_indexes WHERE schemaname = :s"),
                            {"s": self.schema},
                        )
                    ).scalars()
                )
            self.assertIn("knowledge_items_source_type_idx", indexes)
            self.assertNotIn("knowledge_items_machine_id_idx", indexes)
        finally:
            await store.aclose()

    async def test_hybrid_search_promotes_exact_identifiers_and_is_stable(self) -> None:
        evidence_id, records = await self._seed()
        exact_records = {
            "AX-220": KnowledgeRecord(
                subject="pneumatic valve",
                claim="Leave 80 millimeters of service clearance.",
                context={"machine_model": "AX-220"},
                type=KnowledgeType.RULE,
                evidence_id=evidence_id,
            ),
            "S14": KnowledgeRecord(
                subject="connector S14",
                claim="Keep the connector facing downward during washdown.",
                context={"component_model": "S14"},
                type=KnowledgeType.PROCEDURE,
                evidence_id=evidence_id,
            ),
        }
        await self.store.add_knowledge(list(exact_records.values()))

        expected = {
            "M01": records["m01"].id,
            "AX-220": exact_records["AX-220"].id,
            "S14": exact_records["S14"].id,
        }
        for query, expected_id in expected.items():
            with self.subTest(query=query):
                first = await self.store.search_knowledge(query, k=3)
                second = await self.store.search_knowledge(query, k=3)
                self.assertEqual(first[0].record.id, expected_id)
                self.assertIsNotNone(first[0].lexical_rank)
                self.assertGreater(first[0].hybrid_score, 0)
                self.assertTrue(first[0].confident)
                self.assertEqual(
                    [hit.record.id for hit in first],
                    [hit.record.id for hit in second],
                )
                self.assertEqual(
                    [hit.hybrid_score for hit in first],
                    [hit.hybrid_score for hit in second],
                )

    async def test_sentence_query_matches_any_stemmed_term(self) -> None:
        _, records = await self._seed()
        hits = await self.store.search_knowledge(
            "How should I be cleaning the lens of the rear sensors?", k=4
        )
        by_id = {hit.record.id: hit for hit in hits}

        # Stemmed query terms: clean, len, rear, sensor.
        m01 = by_id[records["m01"].id]
        self.assertEqual(m01.lexical_rank, 1)
        self.assertEqual(m01.lexical_matched_terms, 4)
        self.assertTrue(m01.confident)
        # One shared term out of four is not enough on the lexical side alone.
        general = by_id[records["general"].id]
        self.assertEqual(general.lexical_matched_terms, 1)
        self.assertEqual(
            [hit.confident for hit in hits],
            sorted((hit.confident for hit in hits), reverse=True),
        )

    async def test_init_schema_migrates_simple_search_vector_to_english(self) -> None:
        _, records = await self._seed()
        table = f'"{self.schema}"."knowledge_items"'
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN search_vector"))
            await conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN search_vector TSVECTOR "
                    "GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, search_text)) STORED"
                )
            )

        await self.store.init_schema()
        await self.store.init_schema()

        async with self.store.engine.connect() as conn:
            expression = await conn.scalar(
                text(
                    "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef AS d "
                    "JOIN pg_attribute AS a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
                    "WHERE d.adrelid = CAST(:table AS regclass) AND a.attname = 'search_vector'"
                ),
                {"table": table},
            )
            index_exists = await conn.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_indexes "
                    "WHERE schemaname = :schema AND indexname = :index_name)"
                ),
                {
                    "schema": self.schema,
                    "index_name": "knowledge_items_search_vector_gin_idx",
                },
            )
        self.assertIn("'english'::regconfig", expression)
        self.assertTrue(index_exists)
        hits = await self.store.search_knowledge("cleaning lenses", k=1)
        self.assertEqual(hits[0].record.id, records["m01"].id)

    async def test_weak_vector_only_results_are_kept_but_not_confident(self) -> None:
        await self._seed()
        strict_store = KnowledgeStore(
            self.store.engine,
            DeterministicFakeEmbedding(size=DIM),
            DIM,
            schema=self.schema,
            min_vector_similarity=1.0,
        )
        hits = await strict_store.search_knowledge("zzzznosuchtoken", k=3)

        self.assertEqual(len(hits), 3)
        self.assertTrue(all(hit.lexical_rank is None for hit in hits))
        self.assertFalse(any(hit.confident for hit in hits))

    async def test_lexical_matches_are_confident_below_similarity_threshold(self) -> None:
        _, records = await self._seed()
        strict_store = KnowledgeStore(
            self.store.engine,
            DeterministicFakeEmbedding(size=DIM),
            DIM,
            schema=self.schema,
            min_vector_similarity=1.0,
        )
        hits = await strict_store.search_knowledge("M01", k=3)

        self.assertEqual(hits[0].record.id, records["m01"].id)
        self.assertTrue(hits[0].confident)
        self.assertEqual(hits[0].lexical_matched_terms, 1)
        self.assertFalse(any(hit.confident for hit in hits[1:]))

    async def test_reembed_resizes_column_and_keeps_records(self) -> None:
        _, records = await self._seed()
        resized = KnowledgeStore(
            self.store.engine, DeterministicFakeEmbedding(size=8), 8, schema=self.schema
        )

        self.assertEqual(await resized.reembed_knowledge(), len(records))
        await resized.init_schema()

        table = f'"{self.schema}"."knowledge_items"'
        async with self.store.engine.connect() as conn:
            dims = set(
                (
                    await conn.execute(
                        text(f"SELECT vector_dims(embedding) FROM {table}")
                    )
                ).scalars()
            )
            hnsw_exists = await conn.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_indexes "
                    "WHERE schemaname = :schema AND indexname = :index_name)"
                ),
                {"schema": self.schema, "index_name": "knowledge_items_embedding_hnsw_idx"},
            )
        self.assertEqual(dims, {8})
        self.assertTrue(hnsw_exists)
        (m01,) = await resized.get_knowledge([records["m01"].id])
        self.assertEqual(m01.claim, records["m01"].claim)
        hits = await resized.search_knowledge("rear sensor lens", k=4)
        self.assertIn(records["m01"].id, {hit.record.id for hit in hits})
        # New records use the new dimension too.
        await resized.add_knowledge(
            [
                KnowledgeRecord(
                    subject="s",
                    claim="c",
                    type=KnowledgeType.FACT,
                    evidence_id=records["m01"].evidence_id,
                )
            ]
        )

    async def test_reembed_failure_keeps_old_embeddings(self) -> None:
        await self._seed()
        wrong_size = KnowledgeStore(
            self.store.engine,
            DeterministicFakeEmbedding(size=4),
            8,
            schema=self.schema,
        )
        with self.assertRaises(StorageError):
            await wrong_size.reembed_knowledge()
        await self.store.init_schema()  # still vector(16)

    async def test_unknown_evidence_is_rejected(self) -> None:
        await self.store.init_schema()
        record = KnowledgeRecord(
            subject="s", claim="c", type=KnowledgeType.FACT, evidence_id=uuid4()
        )
        with self.assertRaises(StorageError):
            await self.store.add_knowledge([record])


if __name__ == "__main__":
    unittest.main()
