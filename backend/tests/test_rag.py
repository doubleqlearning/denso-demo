"""Tests for document chunking, rank fusion, retrieval metrics, and the chunk store.

The store tests need a PostgreSQL + pgvector database:
    docker compose -f deployments/docker-compose.yml up -d
    TEST_DATABASE_URL=postgresql+psycopg://voice_agent:voice_agent@localhost:5434/voice_agent \
        uv run python -m unittest backend.tests.test_rag -v

Store tables are created in a throwaway schema, so existing data is untouched.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from backend.src.agents.rag import (
    ChunkHit,
    ChunkingConfig,
    Document,
    DocumentStore,
    SearchMode,
    chunk_document,
    chunk_documents,
)
from backend.src.agents.rag.evaluation import (
    calibrate_similarity_threshold,
    mean_metrics,
    per_query_metrics,
    rankings_to_run,
    reciprocal_rank,
)
from backend.src.agents.rag.indexing import index_documents
from backend.src.agents.rag.store import best_chunk_per_document, rrf_fuse
from backend.src.storage import StorageError


DATABASE_URL = os.getenv("TEST_DATABASE_URL")
DIM = 16


def _quiet(message: str) -> None:
    pass


def _hit(chunk_id: str, doc_id: str | None = None) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id.split("#")[0],
        chunk_index=0,
        title="",
        text="",
        score=0.0,
    )


class ChunkingTest(unittest.TestCase):
    def test_short_document_stays_whole(self) -> None:
        document = Document("d1", "One sentence. Two sentences.", title="Title")
        chunks = chunk_document(document, ChunkingConfig(max_words=50))
        self.assertEqual([chunk.id for chunk in chunks], ["d1#0"])
        self.assertEqual(chunks[0].search_text, "Title\nOne sentence. Two sentences.")

    def test_no_limit_keeps_long_document_whole(self) -> None:
        document = Document("d1", " ".join(["word"] * 500) + ".")
        self.assertEqual(len(chunk_document(document, ChunkingConfig())), 1)

    def test_packs_sentences_with_overlap(self) -> None:
        text = "A a a. B b b. C c c. D d d."
        chunks = chunk_document(
            Document("d1", text), ChunkingConfig(max_words=6, overlap_sentences=1)
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["A a a. B b b.", "B b b. C c c.", "C c c. D d d."],
        )
        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])

    def test_without_overlap_chunks_are_disjoint(self) -> None:
        chunks = chunk_document(
            Document("d1", "A a a. B b b. C c c."),
            ChunkingConfig(max_words=6, overlap_sentences=0),
        )
        self.assertEqual([chunk.text for chunk in chunks], ["A a a. B b b.", "C c c."])

    def test_long_sentence_is_split_into_word_windows(self) -> None:
        chunks = chunk_document(
            Document("d1", "one two three four five six seven"),
            ChunkingConfig(max_words=3, overlap_sentences=1),
        )
        self.assertTrue(all(len(chunk.text.split()) <= 3 for chunk in chunks))
        self.assertEqual(
            " ".join(chunk.text for chunk in chunks), "one two three four five six seven"
        )

    def test_abbreviations_and_decimals_do_not_end_sentences(self) -> None:
        chunks = chunk_document(
            Document("d1", "Dose was 3.5 mg e.g. daily. Next one here."),
            ChunkingConfig(max_words=6, overlap_sentences=0),
        )
        self.assertEqual(chunks[0].text, "Dose was 3.5 mg e.g. daily.")

    def test_empty_document_has_no_chunks(self) -> None:
        self.assertEqual(chunk_documents([Document("d1", "  ")], ChunkingConfig()), [])

    def test_label_names_the_setup(self) -> None:
        self.assertEqual(ChunkingConfig().label, "full")
        self.assertEqual(ChunkingConfig(max_words=128, overlap_sentences=1).label, "w128_o1")
        with self.assertRaises(ValueError):
            ChunkingConfig(max_words=0)


class FusionTest(unittest.TestCase):
    def test_rrf_rewards_agreement(self) -> None:
        fused = rrf_fuse(
            [_hit("a#0"), _hit("b#0"), _hit("c#0")],
            [_hit("c#0"), _hit("d#0")],
        )
        self.assertEqual(fused[0].chunk_id, "c#0")
        self.assertEqual((fused[0].vector_rank, fused[0].lexical_rank), (3, 1))
        # a (vector 1) and d (lexical 2) are single-list hits ranked by RRF score.
        self.assertEqual([hit.chunk_id for hit in fused[1:]], ["a#0", "b#0", "d#0"])

    def test_ties_go_to_the_better_single_rank_then_id(self) -> None:
        fused = rrf_fuse([_hit("b#0")], [_hit("a#0")])
        self.assertEqual([hit.chunk_id for hit in fused], ["a#0", "b#0"])

    def test_best_chunk_per_document_keeps_order(self) -> None:
        hits = [_hit("a#1"), _hit("b#0"), _hit("a#0"), _hit("c#2")]
        self.assertEqual(
            [hit.chunk_id for hit in best_chunk_per_document(hits)], ["a#1", "b#0", "c#2"]
        )


class MetricsTest(unittest.TestCase):
    def test_run_scores_follow_list_order(self) -> None:
        run = rankings_to_run({"q": ["z", "a"]})
        self.assertGreater(run["q"]["z"], run["q"]["a"])

    def test_reciprocal_rank_respects_cutoff(self) -> None:
        self.assertEqual(reciprocal_rank({"b"}, ["a", "b"], 2), 0.5)
        self.assertEqual(reciprocal_rank({"b"}, ["a", "b"], 1), 0.0)

    def test_metrics_per_query(self) -> None:
        qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
        rankings = {"q1": ["x", "a", "b"], "q2": ["c"]}
        metrics = per_query_metrics(qrels, rankings, (1, 3))
        self.assertEqual(metrics["q1"]["Recall@1"], 0.0)
        self.assertEqual(metrics["q1"]["Recall@3"], 1.0)
        self.assertEqual(metrics["q1"]["MRR@3"], 0.5)
        self.assertEqual(metrics["q2"]["nDCG@1"], 1.0)
        self.assertEqual(metrics["q2"]["MRR@1"], 1.0)

    def test_queries_without_results_count_as_misses(self) -> None:
        qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
        metrics = per_query_metrics(qrels, {"q1": ["a"], "q2": []}, (1,))
        self.assertEqual(mean_metrics(metrics)["Recall@1"], 0.5)
        self.assertEqual(mean_metrics(metrics)["nDCG@1"], 0.5)

    def test_non_relevant_judgements_are_ignored(self) -> None:
        metrics = per_query_metrics({"q": {"a": 0, "b": 1}}, {"q": ["a", "b"]}, (1, 2))
        self.assertEqual(metrics["q"]["MRR@2"], 0.5)
        self.assertEqual(metrics["q"]["Recall@1"], 0.0)



class ThresholdCalibrationTest(unittest.TestCase):
    def test_picks_lowest_threshold_meeting_target_precision(self) -> None:
        hits = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.5, False), (0.4, False)]
        calibration = calibrate_similarity_threshold(hits, target_precision=0.75, min_support=1)
        # 0.6 keeps 3 of 4 hits relevant; 0.5 drops precision to 0.6.
        self.assertEqual(calibration.threshold, 0.6)
        self.assertEqual(calibration.precision, 0.75)
        self.assertEqual(calibration.recall, 1.0)
        self.assertEqual(calibration.confident_hits, 4)
        self.assertEqual((calibration.labelled_hits, calibration.relevant_hits), (6, 3))

    def test_tied_scores_share_one_threshold(self) -> None:
        hits = [(0.8, True), (0.5, True), (0.5, False)]
        calibration = calibrate_similarity_threshold(hits, target_precision=0.9, min_support=1)
        self.assertEqual(calibration.threshold, 0.8)

    def test_min_support_rejects_thresholds_backed_by_few_hits(self) -> None:
        hits = [(0.9, True), (0.8, False), (0.7, False)]
        calibration = calibrate_similarity_threshold(hits, target_precision=0.9, min_support=2)
        self.assertIsNone(calibration.threshold)
        self.assertEqual(calibration.labelled_hits, 3)

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_similarity_threshold([], target_precision=0)
        with self.assertRaises(ValueError):
            calibrate_similarity_threshold([], min_support=0)


@unittest.skipUnless(DATABASE_URL, "TEST_DATABASE_URL is not set")
class DocumentStoreIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.schema = f"test_{uuid4().hex[:12]}"
        self.store = DocumentStore.from_url(
            DATABASE_URL, DeterministicFakeEmbedding(size=DIM), DIM, schema=self.schema
        )
        async with self.store._engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA {self.schema}"))
        await self.store.init_schema()
        documents = [
            Document("d1", "Aspirin reduces the risk of heart attack.", title="Aspirin"),
            Document("d2", "Vitamin D deficiency is linked to bone loss.", title="Vitamin D"),
            Document(
                "d3",
                "Statins lower cholesterol. Statins also reduce inflammation markers.",
                title="Statins",
            ),
        ]
        await self.store.add_chunks(
            chunk_documents(documents, ChunkingConfig(max_words=4, overlap_sentences=0))
        )

    async def asyncTearDown(self) -> None:
        async with self.store._engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {self.schema} CASCADE"))
        await self.store.aclose()

    async def test_adjacent_passages_and_search_preserve_manual_metadata(self) -> None:
        documents = [
            Document("manual_p0021_c01", "Connector connection procedure.", title="Connectors",
                     metadata={"reference_doc_id": "manual", "page_start": 21, "page_end": 22}),
            Document("manual_p0022_c01", "Inspect the connector latch.", title="Connectors",
                     metadata={"reference_doc_id": "manual", "page_start": 22, "page_end": 23}),
            # Legacy indexed passages have only their stable document IDs.
            Document("manual_p0023_c01", "Replace the connector only if damaged.", title="Connectors"),
            Document("other_p0022_c01", "Another manual.", title="Other"),
        ]
        await self.store.add_chunks(chunk_documents(documents, ChunkingConfig()))
        neighbors = await self.store.adjacent_chunks(["manual_p0022_c01#0"])
        self.assertEqual({hit.doc_id for hit in neighbors}, {"manual_p0021_c01", "manual_p0023_c01"})
        self.assertEqual(next(hit for hit in neighbors if hit.doc_id == "manual_p0021_c01").metadata["page_end"], 22)
        self.assertEqual(await self.store.adjacent_chunks([]), [])
        read = await self.store.get_document("manual_p0022_c01")
        self.assertEqual(read[0].metadata["page_end"], 23)
        searched = await self.store.search("connector", k=10)
        self.assertEqual(next(hit for hit in searched if hit.doc_id == "manual_p0021_c01").metadata["reference_doc_id"], "manual")

    async def test_keyword_search_matches_stemmed_terms(self) -> None:
        hits = await self.store.keyword_search("statin inflammation", k=5)
        self.assertEqual(hits[0].chunk_id, "d3#1")
        self.assertEqual(hits[0].lexical_rank, 1)
        self.assertEqual(await self.store.keyword_search("zebra", k=5), [])

    async def test_keyword_search_weights_rare_terms_higher(self) -> None:
        # "statin" is in every d3 chunk, "aspirin" only in d1's: BM25's IDF
        # puts the aspirin chunk first although both match one term twice.
        hits = await self.store.keyword_search("statins aspirin", k=10)
        self.assertEqual(hits[0].doc_id, "d1")
        self.assertEqual(
            [hit.score for hit in hits], sorted((hit.score for hit in hits), reverse=True)
        )
        self.assertTrue(all(hit.score > 0 for hit in hits))

    async def test_tables_indexed_before_bm25_are_backfilled(self) -> None:
        expected = await self.store.keyword_search("statins aspirin", k=10)
        table = f'"{self.schema}"."{self.store.table_name}"'
        async with self.store._engine.begin() as conn:
            await conn.execute(text(f'DROP TABLE "{self.schema}".chunks_terms'))
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN term_count"))
        await self.store.init_schema()
        self.assertEqual(await self.store.keyword_search("statins aspirin", k=10), expected)

    async def test_semantic_search_finds_an_exact_match_first(self) -> None:
        # The fake embedding is a hash of the text: identical text, identical vector.
        hits = await self.store.search(
            "Vitamin D\nVitamin D deficiency is", mode=SearchMode.SEMANTIC, k=3
        )
        self.assertEqual(hits[0].chunk_id, "d2#0")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        self.assertEqual([hit.vector_rank for hit in hits], [1, 2, 3])

    async def test_hybrid_and_document_search(self) -> None:
        hits = await self.store.search("statins cholesterol", mode=SearchMode.HYBRID, k=10)
        self.assertEqual(len({hit.chunk_id for hit in hits}), len(hits))
        self.assertEqual(hits[0].doc_id, "d3")
        documents = await self.store.search_documents("statins", mode=SearchMode.HYBRID, k=10)
        self.assertEqual(sorted(hit.doc_id for hit in documents), ["d1", "d2", "d3"])

    async def test_get_document_returns_chunks_in_order(self) -> None:
        chunks = await self.store.get_document("d3")
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(
            " ".join(chunk.text for chunk in chunks),
            "Statins lower cholesterol. Statins also reduce inflammation markers.",
        )
        self.assertEqual(await self.store.get_document("missing"), [])

    async def test_index_documents_resumes_and_reindexes(self) -> None:
        chunking = ChunkingConfig(max_words=4, overlap_sentences=0)
        await self.store.drop()
        await self.store.init_schema()
        documents = [Document("a", "One two."), Document("b", "Three four.")]
        with mock.patch.dict(os.environ, {"EMBEDDING_DIM": str(DIM)}):
            added = await index_documents(self.store, documents[:1], chunking, progress=_quiet)
            self.assertEqual(added, 1)
            self.assertEqual(await index_documents(self.store, documents, chunking, progress=_quiet), 1)
            self.assertEqual(
                await index_documents(
                    self.store, documents, chunking, reindex=True, progress=_quiet
                ),
                2,
            )
            with self.assertRaises(StorageError):
                await index_documents(self.store, documents, ChunkingConfig(), progress=_quiet)

    async def test_resume_and_config_guard(self) -> None:
        self.assertEqual(await self.store.indexed_doc_ids(), {"d1", "d2", "d3"})
        chunks = await self.store.count_chunks()
        await self.store.add_chunks(
            chunk_documents([Document("d1", "Aspirin reduces the risk of heart attack.")],
                            ChunkingConfig(max_words=4, overlap_sentences=0))
        )
        self.assertEqual(await self.store.count_chunks(), chunks)
        await self.store.ensure_config({"chunk_words": 4})
        await self.store.ensure_config({"chunk_words": 4})
        with self.assertRaises(StorageError):
            await self.store.ensure_config({"chunk_words": 8})
        await self.store.drop()
        self.assertIsNone(await self.store.config())


if __name__ == "__main__":
    unittest.main()
