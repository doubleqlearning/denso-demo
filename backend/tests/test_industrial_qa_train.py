from pathlib import Path
import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from uuid import uuid4

from backend.knowledge_bases.industrial_qa import train
from backend.knowledge_bases.industrial_qa import benchmark_adapter
from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import KnowledgeRecord


class FullTrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_checkpoint_survives_interruption(self):
        questions = {qid: {"question": qid, "options": {"A": "ok"}} for qid in ("q1", "q2")}
        run = {"run_id": "run", "split": "train", "question_ids": ["q1", "q2"],
               "conversation_ids": [], "conversations": {}}
        store = SimpleNamespace(aclose=AsyncMock())
        documents = SimpleNamespace(aclose=AsyncMock())
        waiting = asyncio.Event()

        async def retrieve(*_args):
            return {}, []

        async def generate(question, *_args):
            if question["question"] == "q2":
                await waiting.wait()
            return train.pilot.AnswerOutput(answer="ok", choice="A", sources=[]), {}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train.pilot, "public_questions", return_value=questions), \
                patch.object(train.pilot, "public_conversations", return_value={}), \
                patch.object(train.pilot, "corpus_chunks", return_value={}), \
                patch.object(train.pilot.llm, "chat_client_from_env", return_value=object()), \
                patch.object(train.pilot.KnowledgeStore, "for_kb", AsyncMock(return_value=store)), \
                patch.object(train.pilot, "open_document_store", AsyncMock(return_value=documents)):
            args = SimpleNamespace(run_dir=Path(directory), concurrency=2)
            task = asyncio.create_task(train.pilot.answer(args, run, retrieve=retrieve, generate=generate,
                                                        init_schema=False, checkpoint_every=1))
            path = args.run_dir / "answers.json"
            for _ in range(100):
                if path.exists():
                    break
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual([a["question_id"] for a in json.loads(path.read_text())], ["q1"])
            store.aclose.assert_awaited_once()
            documents.aclose.assert_awaited_once()

    def test_multiple_questions_ingest_each_conversation_once_and_preserve_baseline(self):
        questions = [{"question_id": "q2", "conversation_id": "c1"},
                     {"question_id": "q1", "conversation_id": "c1"},
                     {"question_id": "q3", "conversation_id": "c2"}]
        baseline = {"split": "train", "run_id": "base", "conversation_ids": ["c1"],
                    "conversations": {"c1": {"status": "processed", "records": []}}, "extraction": {}}
        run = train.make_run(questions, baseline, Path("full-train"))
        self.assertEqual(run["question_ids"], ["q1", "q2", "q3"])
        self.assertEqual(run["conversation_ids"], ["c1", "c2"])
        self.assertEqual(run["new_conversation_ids"], ["c2"])
        run["conversations"]["c1"]["records"].append("new")
        self.assertEqual(baseline["conversations"]["c1"]["records"], [])

    def test_reuse_outside_train_is_rejected(self):
        baseline = {"split": "train", "conversation_ids": ["test-only"]}
        with self.assertRaisesRegex(ValueError, "outside train"):
            train.make_run([{"question_id": "q1", "conversation_id": "c1"}], baseline, Path("run"))

    async def test_retrieval_union_does_not_widen_consolidation_scope(self):
        run = {"run_id": "new", "reused_run_ids": ["base"]}
        union = train.search_scope(run)
        self.assertEqual(union, {"$or": [
            {"context.benchmark_run_id": {"$eq": "new"}},
            {"context.benchmark_run_id": {"$eq": "base"}},
        ]})
        self.assertEqual(train.pilot.run_filter(run), {"context.benchmark_run_id": {"$eq": "new"}})
        rec = KnowledgeRecord(subject="PLC", claim="Check cable.", type=KnowledgeType.RULE,
                              evidence_id=uuid4(), context={"machine_id": "FX5-PLC963"})
        store = SimpleNamespace(search_knowledge=AsyncMock(return_value=[SimpleNamespace(record=rec, confident=True)]))
        documents = SimpleNamespace(search=AsyncMock(return_value=[]))
        with patch.object(benchmark_adapter.llm, "chat_client_from_env", return_value=object()), \
                patch.object(benchmark_adapter.llm, "ainvoke", AsyncMock(return_value='{"labels":["K1"]}')):
            retrieval = benchmark_adapter.ScopedRetrieval({}, scope=union)
            await retrieval("FX5-PLC963", store, documents, run,
                            SimpleNamespace(k=8, per_conversation=3), {}, {}, {})
        self.assertEqual(store.search_knowledge.call_args.kwargs["filters"]["$and"][0], union)


if __name__ == "__main__":
    unittest.main()
