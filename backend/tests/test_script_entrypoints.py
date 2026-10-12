"""Unit tests for the standalone ingestion and retrieval CLI interfaces."""

from __future__ import annotations

import tempfile
import unittest
import io
from pathlib import Path
from unittest import mock

from backend.knowledge_bases import RetrievalQuestions, get_kb
from backend.scripts.kb import (
    TranscriptInput,
    build_parser,
    evaluate,
    ingest,
    load_retrieval_questions,
    read_transcripts,
    reembed,
    reset,
    to_documents,
)
from backend.src.services.knowledge_extraction import KnowledgeExtractionService
from backend.tests.test_knowledge_extraction_service import FakeLLM
from backend.tests.test_knowledge_ingestion_service import FakeStore


class IngestEntrypointTest(unittest.TestCase):
    def test_reads_one_text_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.txt"
            path.write_text("  Expert: Clean the sensor.  ", encoding="utf-8")

            (item,) = read_transcripts(path, stated_by="Expert A")

        self.assertEqual(item.transcript, "Expert: Clean the sensor.")
        self.assertEqual(item.stated_by, "Expert A")
        self.assertEqual(item.metadata, {"source_file": "conversation.txt"})

    def test_reads_csv_rows_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.csv"
            path.write_text(
                "case_id,stated_by,raw_conversation\n"
                '1,Expert A,"Expert: Clean the lens."\n',
                encoding="utf-8",
            )

            (item,) = read_transcripts(path)

        self.assertEqual(item.stated_by, "Expert A")
        self.assertEqual(item.metadata["case_id"], "1")
        self.assertEqual(item.metadata["source_row"], "2")

    def test_rejects_csv_without_transcript_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.csv"
            path.write_text("text\nhello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transcript column"):
                list(read_transcripts(path))

    def test_rejects_empty_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.txt"
            path.write_text("   ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                list(read_transcripts(path))

    def test_ingest_parser_requires_only_a_knowledge_base(self) -> None:
        args = build_parser().parse_args(["ingest", "maintenance"])
        self.assertEqual(args.kb, "maintenance")
        self.assertIsNone(args.file)
        self.assertEqual(args.batch_size, 1)
        self.assertIsNone(args.concurrency)

    def test_ingest_parser_accepts_batch_options(self) -> None:
        args = build_parser().parse_args(
            ["ingest", "maintenance", "--batch-size", "8", "--concurrency", "4"]
        )
        self.assertEqual((args.batch_size, args.concurrency), (8, 4))
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["ingest", "maintenance", "--batch-size", "0"])

    def test_ingest_parser_accepts_a_file_override(self) -> None:
        args = build_parser().parse_args(
            ["ingest", "maintenance", "--file", "conversation.txt", "--limit", "3"]
        )
        self.assertEqual(args.file, Path("conversation.txt"))
        self.assertEqual(args.limit, 3)

    def test_ingest_parser_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["ingest", "maintenance", "--limit", "0"])


class DocumentRetrievalEntrypointTest(unittest.TestCase):
    def test_loads_questions_with_several_gold_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.csv"
            path.write_text(
                "question_id,question,gold_case_ids\n"
                "1,Why is the sensor unstable?,1\n"
                '2,Which cases mention oil?," 6;12 ;19 "\n',
                encoding="utf-8",
            )
            queries, qrels = load_retrieval_questions(
                RetrievalQuestions(path, gold_column="gold_case_ids")
            )
            self.assertEqual(queries["2"], "Which cases mention oil?")
            self.assertEqual(qrels, {"1": {"1": 1}, "2": {"6": 1, "12": 1, "19": 1}})
            with self.assertRaisesRegex(ValueError, "gold_doc_ids"):
                load_retrieval_questions(RetrievalQuestions(path))

    def test_maintenance_questions_reference_indexed_cases(self) -> None:
        spec = get_kb("maintenance")
        queries, qrels = load_retrieval_questions(spec.retrieval_questions)
        case_ids = {item.metadata["case_id"] for item in read_transcripts(spec.dataset.path)}
        self.assertEqual(len(queries), 156)
        self.assertLessEqual({doc for gold in qrels.values() for doc in gold}, case_ids)

    def test_documents_are_keyed_by_the_id_column(self) -> None:
        items = [
            TranscriptInput("Expert: a.", None, {"case_id": "7", "topic": "Lens"}, "f.csv:2"),
            TranscriptInput("Expert: b.", None, {}, "f.csv:3"),
        ]
        documents = to_documents(items, id_column="case_id", title_column="topic")
        self.assertEqual([(d.id, d.title) for d in documents], [("7", "Lens"), ("f.csv:3", "")])
        with self.assertRaisesRegex(ValueError, "more than once"):
            to_documents([items[0], items[0]], id_column="case_id", title_column=None)

    def test_evaluate_parser_defaults(self) -> None:
        args = build_parser().parse_args(["evaluate", "maintenance"])
        self.assertEqual(args.modes, ["semantic", "keyword", "hybrid"])
        self.assertEqual(args.k_values, [1, 3, 5, 10])
        self.assertIsNone(args.query_instruction)


class QueryEntrypointTest(unittest.TestCase):
    def test_query_parser_takes_knowledge_base_and_question(self) -> None:
        args = build_parser().parse_args(["query", "maintenance", "How do I clean M01?"])
        self.assertEqual(args.kb, "maintenance")
        self.assertEqual(args.question, "How do I clean M01?")

    def test_reset_requires_explicit_confirmation_flag(self) -> None:
        args = build_parser().parse_args(["reset", "maintenance"])
        self.assertFalse(args.yes)


class IngestionCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_command_saves_active_extraction_version_and_prompt(self) -> None:
        spec = get_kb("enterprise_rag")
        extractor = KnowledgeExtractionService(FakeLLM('{"knowledge_items":[]}'), spec.extract_prompt)
        store = FakeStore()
        store.processed_metadata_values = mock.AsyncMock(return_value=set())
        store.find_processed_conversation = mock.AsyncMock(return_value=None)
        store.aclose = mock.AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("A document to extract.")
            args = build_parser().parse_args(["ingest", "enterprise_rag", "--file", str(path)])
            with (
                mock.patch("backend.scripts.kb.KnowledgeExtractionService.from_env", return_value=extractor),
                mock.patch("backend.scripts.kb.KnowledgeStore.for_kb", new_callable=mock.AsyncMock, return_value=store),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(await ingest(spec, args), 0)
        conversation, = store.conversations.values()
        self.assertEqual(conversation.metadata["extraction"], {
            "version": "complete-units-v2", **extractor.provenance,
        })
        store.aclose.assert_awaited_once()

    async def test_documents_ingest_indexes_without_extraction(self) -> None:
        args = build_parser().parse_args(["ingest", "maintenance", "--limit", "2", "--reindex"])
        with (
            mock.patch("backend.scripts.kb.KnowledgeExtractionService.from_env") as extractor,
            mock.patch(
                "backend.scripts.kb.index_kb_documents", new_callable=mock.AsyncMock, return_value=0
            ) as index,
        ):
            self.assertEqual(await ingest(get_kb("maintenance"), args), 0)
        extractor.assert_not_called()
        spec, documents = index.await_args.args
        self.assertEqual([document.id for document in documents], ["1", "2"])
        self.assertTrue(documents[0].text.strip())
        self.assertEqual(index.await_args.kwargs, {"reindex": True})

    async def test_evaluate_and_reembed_follow_the_retrieval_mode(self) -> None:
        args = build_parser().parse_args(["evaluate", "enterprise_rag"])
        with self.assertRaisesRegex(ValueError, "needs documents"):
            await evaluate(get_kb("enterprise_rag"), args)
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(await reembed(get_kb("maintenance")), 2)
        self.assertIn("--reindex", stderr.getvalue())

    async def test_reset_requires_yes_and_targets_only_enterprise_database(self) -> None:
        with (
            mock.patch("backend.scripts.kb.drop_database", new_callable=mock.AsyncMock, return_value=True) as drop,
            mock.patch("backend.scripts.kb.server_url_from_env", return_value="server"),
            mock.patch("sys.stdout", new_callable=io.StringIO),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            self.assertEqual(await reset(get_kb("enterprise_rag"), confirmed=False), 2)
            drop.assert_not_awaited()
            self.assertEqual(await reset(get_kb("enterprise_rag"), confirmed=True), 0)
            drop.assert_awaited_once_with("server", "kb_enterprise_rag")


if __name__ == "__main__":
    unittest.main()
