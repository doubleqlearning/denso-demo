"""Unit tests for the knowledge retrieval agent (fake model, fake store)."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr, ValidationError

from backend.knowledge_bases import Retrieval, maintenance
from backend.src.agents.agent.service import (
    DocumentCitation,
    InvalidKnowledgeCitation,
    KnowledgeAgentService,
    KnowledgeAgentUnavailable,
    citations_from_evidence,
    public_tool_input,
    public_tool_output,
)
from backend.src.agents.agent.state import merge_evidence
from backend.src.agents.agent.tools import EvidenceIdInput
from backend.src.agents.rag.store import ChunkHit, SearchMode
from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import (
    KnowledgeRecord,
    KnowledgeSearchHit,
    KnowledgeStatus,
    RawConversation,
    StorageError,
)


EVIDENCE_ID = uuid4()
RECORD = KnowledgeRecord(
    subject="rear photoelectric sensor",
    claim="Clean the lens before replacing the sensor.",
    context={"machine_id": "M01"},
    type=KnowledgeType.PROCEDURE,
    evidence_id=EVIDENCE_ID,
    stated_by="Maintenance Expert",
    valid_from=date(2026, 1, 1),
)


class ScriptedChatModel(BaseChatModel):
    """Returns queued AI messages in order and records what it was shown."""

    responses: list[AIMessage]
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._calls.append(list(messages))
        message = self.responses[len(self._calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=message)])


def _hit(record: KnowledgeRecord = RECORD, *, confident: bool = True) -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        record=record,
        hybrid_score=0.0325,
        vector_rank=1,
        lexical_rank=2,
        vector_distance=0.12,
        lexical_score=0.4,
        lexical_matched_terms=2,
        confident=confident,
    )


class FakeStore:
    def __init__(
        self, *, fail: bool = False, hits: list[KnowledgeSearchHit] | None = None,
        transcript: str = "Expert: clean the lens before replacing the sensor.",
    ):
        self.fail = fail
        self.transcript = transcript
        self.hits = [_hit()] if hits is None else hits
        self.searches: list[tuple[str, Any, int]] = []

    async def search_knowledge(self, query, *, filters=None, k=5):
        if self.fail:
            raise StorageError("connection refused")
        self.searches.append((query, filters, k))
        return list(self.hits)

    async def get_knowledge(self, ids):
        return [RECORD] if RECORD.id in ids else []

    async def get_conversation(self, conversation_id):
        if conversation_id != EVIDENCE_ID:
            return None
        return RawConversation(
            id=EVIDENCE_ID,
            transcript=self.transcript,
            stated_by="Maintenance Expert",
            created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )


def _search_call(call_id: str = "call_1", **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge", "args": args, "id": call_id}],
    )


def _tool_payload(message: BaseMessage) -> dict[str, Any]:
    assert isinstance(message, ToolMessage)
    return json.loads(message.content)


def _service(store: FakeStore, *responses: AIMessage) -> tuple[KnowledgeAgentService, ScriptedChatModel]:
    model = ScriptedChatModel(responses=list(responses))
    service = KnowledgeAgentService(
        store=store,
        model=model,
        system_prompt="Answer from the knowledge base.",
        timeout_seconds=10,
        context_filters=maintenance.SPEC.context_filters,
        trace_context_keys=maintenance.SPEC.trace_context_keys,
    )
    return service, model


class CitationTest(unittest.TestCase):
    def evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": "knowledge",
                "knowledge_id": str(RECORD.id),
                "subject": RECORD.subject,
                "claim": RECORD.claim,
                "type": "procedure",
                "evidence_id": str(EVIDENCE_ID),
                "stated_by": "Maintenance Expert",
            }
        ]

    def test_resolves_citations_once_in_order(self) -> None:
        answer = f"Clean the lens [K:{RECORD.id}]. Really [K:{str(RECORD.id).upper()}]."
        (citation,) = citations_from_evidence(answer, self.evidence())
        self.assertEqual(citation.knowledge_id, str(RECORD.id))
        self.assertEqual(citation.evidence_id, str(EVIDENCE_ID))
        self.assertEqual(citation.type, "procedure")

    def test_unknown_citation_is_rejected(self) -> None:
        with self.assertRaises(InvalidKnowledgeCitation):
            citations_from_evidence(f"Nope [K:{uuid4()}]", self.evidence())

    def test_answer_without_markers_has_no_citations(self) -> None:
        self.assertEqual(citations_from_evidence("No evidence found.", self.evidence()), ())


class MergeEvidenceTest(unittest.TestCase):
    def test_deduplicates_and_enriches(self) -> None:
        merged = merge_evidence(
            [{"kind": "knowledge", "knowledge_id": "a", "claim": "x", "stated_by": None}],
            [
                {"kind": "knowledge", "knowledge_id": "a", "claim": "", "stated_by": "E"},
                {"kind": "conversation", "evidence_id": "a"},
            ],
        )
        self.assertEqual(
            merged,
            [
                {"kind": "knowledge", "knowledge_id": "a", "claim": "x", "stated_by": "E"},
                {"kind": "conversation", "evidence_id": "a"},
            ],
        )


class PublicTraceTest(unittest.TestCase):
    def test_evidence_paging_is_validated_and_allowlisted(self) -> None:
        for invalid in ({"offset": -1}, {"limit": 0}, {"limit": 6001}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                EvidenceIdInput(evidence_id=EVIDENCE_ID, **invalid)
        self.assertEqual(
            public_tool_input("get_evidence", {
                "evidence_id": str(EVIDENCE_ID), "offset": 6000, "limit": 2000,
                "private": "hidden",
            }),
            {"evidence_id": str(EVIDENCE_ID), "offset": 6000, "limit": 2000},
        )

    def test_tool_input_is_allowlisted(self) -> None:
        self.assertEqual(
            public_tool_input(
                "search_knowledge",
                {
                    "query": "sensor",
                    "machine_id": "M01",
                    "api_key": "must-not-leak",
                },
                context_fields={"machine_id"},
            ),
            {"query": "sensor", "machine_id": "M01"},
        )
        # Context arguments are only public for the KB that declares them.
        self.assertEqual(
            public_tool_input("search_knowledge", {"query": "q", "machine_id": "M01"}),
            {"query": "q"},
        )

    def test_knowledge_summary_shows_only_trace_context_keys(self) -> None:
        output = public_tool_output(
            "search_knowledge",
            {
                "ok": True,
                "match": "confident",
                "knowledge": [
                    {"id": "k1", "context": {"machine_id": "M01", "note": "private"}}
                ],
            },
            context_fields={"machine_id"},
        )
        self.assertEqual(output["items"][0]["context"], {"machine_id": "M01"})

    def test_evidence_trace_omits_transcript_and_metadata(self) -> None:
        output = public_tool_output(
            "get_evidence",
            {
                "ok": True,
                "evidence": {
                    "id": str(EVIDENCE_ID),
                    "stated_by": "Maintenance Expert",
                    "metadata": {"secret": "hidden"},
                    "transcript": "private transcript",
                    "truncated": False,
                },
            },
        )
        self.assertNotIn("private transcript", str(output))
        self.assertNotIn("metadata", str(output))
        self.assertEqual(output["evidence"]["returned_transcript_chars"], 18)


class KnowledgeAgentServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_reads_long_evidence_in_pages_without_exposing_it_in_trace(self) -> None:
        source = "Đ" * 6000 + "β" * 1500 + "Final rollback condition."
        calls = [
            AIMessage(content="", tool_calls=[{
                "name": "get_evidence", "id": f"page_{i}",
                "args": {"evidence_id": str(EVIDENCE_ID), **args},
            }])
            for i, args in enumerate(({}, {"offset": 6000, "limit": 1500}, {"offset": 7500}))
        ]
        service, model = _service(FakeStore(transcript=source), *calls, AIMessage(content="Read."))
        events = [e async for e in service.start_stream("Read the source", thread_id="pages").events]
        pages = [_tool_payload(model.calls[i][-1])["evidence"] for i in (1, 2, 3)]
        self.assertEqual("".join(p["transcript"] for p in pages), source)
        self.assertEqual([p["next_offset"] for p in pages], [6000, 7500, None])
        self.assertTrue(all(p["total_chars"] == len(source) for p in pages))
        traces = [e.data["output"]["evidence"] for e in events if e.event == "tool.completed"]
        self.assertEqual([p["offset"] for p in traces], [0, 6000, 7500])
        self.assertNotIn("Final rollback condition", str(traces))
        self.assertNotIn("transcript", {k for p in traces for k in p})

    async def test_evidence_past_end_terminates_with_empty_page(self) -> None:
        service, model = _service(
            FakeStore(transcript="short"),
            AIMessage(content="", tool_calls=[{
                "name": "get_evidence", "id": "past_end",
                "args": {"evidence_id": str(EVIDENCE_ID), "offset": 100},
            }]),
            AIMessage(content="Done."),
        )
        await service.ask("Read")
        page = _tool_payload(model.calls[1][-1])["evidence"]
        self.assertEqual(page["transcript"], "")
        self.assertIsNone(page["next_offset"])

    async def test_answers_with_validated_citation(self) -> None:
        store = FakeStore()
        service, model = _service(
            store,
            _search_call(query="unstable sensor signal", machine_id="M01"),
            AIMessage(content=f"Clean the lens before replacing it [K:{RECORD.id}]."),
        )

        with self.assertLogs("backend.src.agents.agent.tools", level="INFO") as logs:
            answer = await service.ask(
                "The rear sensor on M01 is unstable, what should I do?"
            )

        self.assertEqual(answer.answer, f"Clean the lens before replacing it [K:{RECORD.id}].")
        (citation,) = answer.citations
        self.assertEqual(UUID(citation.knowledge_id), RECORD.id)
        self.assertEqual(citation.stated_by, "Maintenance Expert")

        query, filters, k = store.searches[0]
        self.assertEqual(query, "unstable sensor signal")
        self.assertEqual(filters["$or"][0], {"context.machine_id": {"$eq": "M01"}})
        self.assertEqual(k, 5)

        tool_message = model.calls[1][-1]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertIn('"validity": "current"', tool_message.content)
        self.assertIn('"hybrid_score": 0.0325', tool_message.content)
        self.assertIn('"vector_rank": 1', tool_message.content)
        self.assertIn('"lexical_rank": 2', tool_message.content)
        self.assertIn('"vector_similarity": 0.88', tool_message.content)
        payload = _tool_payload(tool_message)
        self.assertEqual(payload["match"], "confident")
        self.assertEqual(payload["searches_remaining"], 2)
        self.assertNotIn("note", payload)
        self.assertNotIn("embedding", tool_message.content)
        self.assertIn("[Turn context: current UTC time is", model.calls[0][-1].content)
        self.assertTrue(any("hybrid_score" in line for line in logs.output))

    async def test_stream_emits_normalized_trace_and_citations(self) -> None:
        service, _ = _service(
            FakeStore(),
            _search_call(query="sensor"),
            AIMessage(content=f"Clean it [K:{RECORD.id}]."),
        )
        session = service.start_stream("sensor?", thread_id="t1")
        events = [event async for event in session.events]
        names = [event.event for event in events]
        self.assertEqual(names[0], "run.started")
        self.assertIn("tool.started", names)
        self.assertIn("tool.completed", names)
        self.assertEqual(names[-2:], ["answer.citations", "run.completed"])
        self.assertEqual(
            [event.sequence for event in events], list(range(1, len(events) + 1))
        )
        self.assertTrue(all(event.request_id == session.request_id for event in events))
        started = next(event for event in events if event.event == "tool.started")
        self.assertEqual(started.data["tool"], "search_knowledge")
        self.assertEqual(started.data["input"], {"query": "sensor"})
        completed = next(event for event in events if event.event == "tool.completed")
        self.assertEqual(completed.data["output"]["count"], 1)
        self.assertEqual(completed.data["output"]["match"], "confident")
        self.assertIs(
            completed.data["output"]["items"][0]["retrieval"]["confident"], True
        )
        self.assertNotIn("claim", str(completed.data))
        text = "".join(
            event.data["text"] for event in events if event.event == "answer.delta"
        )
        self.assertEqual(text, f"Clean it [K:{RECORD.id}].")

    async def test_only_confident_hits_are_returned(self) -> None:
        weak = KnowledgeRecord(
            subject="conveyor belt",
            claim="Check belt tension weekly.",
            type=KnowledgeType.RULE,
            evidence_id=EVIDENCE_ID,
        )
        store = FakeStore(hits=[_hit(), _hit(weak, confident=False)])
        service, model = _service(
            store,
            _search_call(query="unstable rear sensor"),
            AIMessage(content="Nothing."),
        )
        await service.ask("sensor?")
        payload = _tool_payload(model.calls[1][-1])
        self.assertEqual(payload["match"], "confident")
        self.assertEqual([item["id"] for item in payload["knowledge"]], [str(RECORD.id)])

    async def test_payloads_omit_unpopulated_conflicts(self) -> None:
        service, model = _service(
            FakeStore(),
            _search_call(query="unstable rear sensor"),
            AIMessage(content="", tool_calls=[{
                "name": "get_knowledge", "id": "read", "args": {"knowledge_id": str(RECORD.id)},
            }]),
            AIMessage(content="Done."),
        )
        await service.ask("sensor?")
        searched, read = (_tool_payload(model.calls[i][-1]) for i in (1, 2))
        self.assertEqual(read["knowledge"]["id"], str(RECORD.id))
        self.assertNotIn("conflicting_knowledge", read)
        for record in (searched["knowledge"][0], read["knowledge"]):
            self.assertNotIn("conflict_with", record)

    async def test_search_is_compact_and_get_knowledge_returns_the_card(self) -> None:
        case = KnowledgeRecord(
            subject="WG-21", claim="Cold start.\nObservation: Misfire.", type=KnowledgeType.CASE,
            context={"machine_id": "M03", "case_card": {"observation": "Misfire."}},
            evidence_id=EVIDENCE_ID, status=KnowledgeStatus.DRAFT,
        )
        lesson = KnowledgeRecord(
            subject="weld gun", claim="Check the connector first.\n1. Isolate power.",
            type=KnowledgeType.PROCEDURE, context={"steps": ["Isolate power."]},
            evidence_id=EVIDENCE_ID, supported_by=[case.id], status=KnowledgeStatus.APPROVED,
        )

        class CaseStore(FakeStore):
            async def get_knowledge(self, ids):
                return [case] if case.id in ids else []

        service, model = _service(
            CaseStore(hits=[_hit(case), _hit(lesson)]),
            _search_call(query="weld gun misfire"),
            AIMessage(content="", tool_calls=[{
                "name": "get_knowledge", "id": "read", "args": {"knowledge_id": str(case.id)},
            }]),
            AIMessage(content="Done."),
        )
        await service.ask("misfire?")
        searched, read = (_tool_payload(model.calls[i][-1]) for i in (1, 2))
        found_case, found_lesson = searched["knowledge"]
        self.assertEqual(found_case["context"], {"machine_id": "M03"})
        self.assertEqual(found_case["status"], "draft")
        self.assertNotIn("supported_by", found_case)
        self.assertEqual(found_lesson["context"], {})
        self.assertEqual(found_lesson["supported_by"], [str(case.id)])
        self.assertEqual(found_lesson["status"], "approved")
        self.assertEqual(read["knowledge"]["context"]["case_card"], {"observation": "Misfire."})
        for record in (found_case, found_lesson, read["knowledge"]):
            self.assertNotIn("revision", record)
            self.assertNotIn("superseded_by", record)

    async def test_weak_results_are_returned_as_low_confidence(self) -> None:
        records = [
            KnowledgeRecord(
                subject=f"component {i}",
                claim=f"Claim {i}.",
                type=KnowledgeType.FACT,
                evidence_id=EVIDENCE_ID,
            )
            for i in range(5)
        ]
        store = FakeStore(hits=[_hit(record, confident=False) for record in records])
        service, model = _service(
            store,
            _search_call(query="unstable rear sensor"),
            AIMessage(content=f"Partial match: claim 0 [K:{records[0].id}]."),
        )
        answer = await service.ask("sensor?")
        payload = _tool_payload(model.calls[1][-1])
        self.assertEqual(payload["match"], "low_confidence")
        self.assertIn("nearest records", payload["note"])
        self.assertEqual(
            [item["id"] for item in payload["knowledge"]],
            [str(record.id) for record in records[:3]],
        )
        self.assertTrue(
            all(item["retrieval"]["confident"] is False for item in payload["knowledge"])
        )
        # Low-confidence records are still citable when the model uses them.
        self.assertEqual(answer.citations[0].knowledge_id, str(records[0].id))

    async def test_empty_result_is_labelled_none(self) -> None:
        service, model = _service(
            FakeStore(hits=[]),
            _search_call(query="unstable rear sensor"),
            AIMessage(content="The knowledge base has no answer."),
        )
        await service.ask("sensor?")
        payload = _tool_payload(model.calls[1][-1])
        self.assertEqual(payload["match"], "none")
        self.assertEqual(payload["knowledge"], [])

    async def test_search_budget_is_enforced_per_question(self) -> None:
        store = FakeStore(hits=[])
        service, model = _service(
            store,
            *(_search_call(f"call_{i}", query=f"attempt {i}") for i in range(4)),
            AIMessage(content="The knowledge base has no answer."),
            # Next question gets a fresh budget.
            _search_call("call_next", query="new question"),
            AIMessage(content="Still nothing."),
        )
        [e async for e in service.start_stream("sensor?", thread_id="budget").events]
        self.assertEqual(
            [query for query, _, _ in store.searches],
            [f"attempt {i}" for i in range(3)],
        )
        self.assertEqual(_tool_payload(model.calls[3][-1])["searches_remaining"], 0)
        refused = _tool_payload(model.calls[4][-1])
        self.assertEqual(refused["error"], "search_budget_exhausted")

        [e async for e in service.start_stream("another?", thread_id="budget").events]
        self.assertEqual(store.searches[-1][0], "new question")

    async def test_invented_citation_is_rejected(self) -> None:
        service, _ = _service(
            FakeStore(),
            _search_call(query="sensor"),
            AIMessage(content=f"Replace it [K:{uuid4()}]."),
        )
        with self.assertRaises(InvalidKnowledgeCitation):
            await service.ask("sensor?")

    async def test_storage_failure_is_reported_as_unavailable(self) -> None:
        service, _ = _service(
            FakeStore(fail=True),
            _search_call(query="sensor"),
            AIMessage(content="unused"),
        )
        with self.assertRaises(KnowledgeAgentUnavailable) as caught:
            await service.ask("sensor?")
        self.assertEqual(caught.exception.error_type, "StorageError")

    async def test_evidence_resets_between_turns(self) -> None:
        service, _ = _service(
            FakeStore(),
            _search_call(query="sensor"),
            AIMessage(content=f"Clean it [K:{RECORD.id}]."),
            # Second turn: no retrieval, so the earlier citation is no longer valid.
            AIMessage(content=f"As I said [K:{RECORD.id}]."),
        )
        first = [e async for e in service.start_stream("sensor?", thread_id="t").events]
        self.assertEqual(first[-1].event, "run.completed")
        with self.assertRaises(InvalidKnowledgeCitation):
            [e async for e in service.start_stream("again?", thread_id="t").events]


class FakeDocumentStore:
    """Two passages of conversation 12 and one of conversation 3."""

    def __init__(self) -> None:
        self.chunks = {
            "12": [
                ChunkHit("12#0", "12", 0, "", "Expert: clean the lens first.", 0.0),
                ChunkHit("12#1", "12", 1, "", "Expert: then recalibrate.", 0.0),
            ],
            "3": [ChunkHit("3#0", "3", 0, "", "Expert: check the cable.", 0.0)],
        }
        self.searches: list[tuple[str, SearchMode, int]] = []

    async def search_documents(self, query, *, mode, k):
        self.searches.append((query, mode, k))
        return [
            ChunkHit("12#0", "12", 0, "", "Expert: clean the lens first.", 0.032, 1, 2),
            ChunkHit("3#0", "3", 0, "", "Expert: check the cable.", 0.016, 2, None),
        ][:k]

    async def get_document(self, doc_id):
        return list(self.chunks.get(doc_id, []))


def _call(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _document_service(
    store: FakeDocumentStore, *responses: AIMessage
) -> tuple[KnowledgeAgentService, ScriptedChatModel]:
    model = ScriptedChatModel(responses=list(responses))
    service = KnowledgeAgentService(
        store=store,
        model=model,
        system_prompt="Answer from the documents.",
        timeout_seconds=10,
        retrieval=Retrieval.DOCUMENTS,
    )
    return service, model


class DocumentCitationTest(unittest.TestCase):
    def test_resolves_document_markers_against_returned_documents(self) -> None:
        evidence = [{"kind": "document", "doc_id": "12", "title": None}]
        self.assertEqual(
            citations_from_evidence("Clean it [D:12]. Again [D:12].", evidence),
            (DocumentCitation(doc_id="12", title=None),),
        )
        with self.assertRaises(InvalidKnowledgeCitation):
            citations_from_evidence("Replace it [D:7].", evidence)

    def test_document_trace_omits_passage_text(self) -> None:
        self.assertEqual(
            public_tool_input("search_documents", {"query": "lens", "k": 3, "x": "hidden"}),
            {"query": "lens", "k": 3},
        )
        output = public_tool_output(
            "search_documents",
            {
                "ok": True,
                "documents": [
                    {
                        "doc_id": "12",
                        "chunk_id": "12#0",
                        "title": "",
                        "text": "private passage",
                        "retrieval": {"hybrid_score": 0.03, "vector_rank": 1, "lexical_rank": None},
                    }
                ],
            },
        )
        self.assertNotIn("private passage", str(output))
        self.assertEqual(output["documents"][0]["retrieval"], {"hybrid_score": 0.03, "vector_rank": 1})
        output = public_tool_output(
            "get_document",
            {"ok": True, "document": {"doc_id": "12", "passages": ["private passage"], "start": 0}},
        )
        self.assertNotIn("private passage", str(output))
        self.assertEqual(output["document"]["returned_chars"], 15)


class DocumentAgentServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_answers_with_document_citations(self) -> None:
        store = FakeDocumentStore()
        service, model = _document_service(
            store,
            _call("search_documents", query="Why is the rear sensor unstable?"),
            _call("get_document", "call_2", doc_id="12"),
            AIMessage(content="Clean the lens, then recalibrate [D:12]."),
        )
        answer = await service.ask("Rear sensor unstable?")

        self.assertEqual(answer.citations, (DocumentCitation(doc_id="12", title=None),))
        self.assertEqual(store.searches, [("Why is the rear sensor unstable?", SearchMode.HYBRID, 5)])
        found = _tool_payload(model.calls[1][-1])
        self.assertEqual([item["doc_id"] for item in found["documents"]], ["12", "3"])
        self.assertEqual(found["documents"][0]["retrieval"]["lexical_rank"], 2)
        self.assertEqual(found["searches_remaining"], 2)
        document = _tool_payload(model.calls[2][-1])["document"]
        self.assertEqual(
            document["passages"], ["Expert: clean the lens first.", "Expert: then recalibrate."]
        )
        self.assertIsNone(document["next_start"])

    async def test_stream_trace_and_logs_omit_document_text(self) -> None:
        service, _ = _document_service(
            FakeDocumentStore(),
            _call("search_documents", query="lens"),
            AIMessage(content="Clean the lens [D:12]."),
        )
        with self.assertLogs("backend.src.agents.agent.service", level="INFO") as logs:
            events = [
                event.to_dict()
                async for event in service.start_stream("lens?", thread_id="t").events
            ]
        completed = next(event for event in events if event["type"] == "tool.completed")
        self.assertEqual(completed["data"]["output"]["count"], 2)
        self.assertNotIn("clean the lens first", json.dumps(events))
        self.assertNotIn("clean the lens first", "\n".join(logs.output))
        citations = next(event for event in events if event["type"] == "answer.citations")
        self.assertEqual(citations["data"]["citations"], [{"doc_id": "12", "title": None}])

    async def test_missing_document_and_budget(self) -> None:
        store = FakeDocumentStore()
        service, model = _document_service(
            store,
            _call("get_document", doc_id="99"),
            *(_call("search_documents", f"call_{i}", query=f"try {i}") for i in range(4)),
            AIMessage(content="The knowledge base has no answer."),
        )
        await service.ask("Unknown?")
        self.assertEqual(_tool_payload(model.calls[1][-1])["error"], "document_not_found")
        self.assertEqual(len(store.searches), 3)
        self.assertEqual(_tool_payload(model.calls[5][-1])["error"], "search_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
