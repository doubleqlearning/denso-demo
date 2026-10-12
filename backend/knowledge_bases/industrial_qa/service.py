"""Industrial QA application service using the shared H1/H3 pipeline."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import re
from time import monotonic
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError

from backend.knowledge_bases.stores import open_document_store
from backend.knowledge_bases.industrial_qa.pipeline import IndustrialQaPipeline
from backend.knowledge_bases.industrial_qa.retrieval import DOCUMENT_QUERY_INSTRUCTION, detected_scope
from backend.src.agents.agent.config import RUN_TIMEOUT_SECONDS
from backend.src.agents.agent.service import (
    CITATION_RE, DOCUMENT_CITATION_RE,
    AgentStreamSession, InvalidKnowledgeCitation, KnowledgeAgentError,
    KnowledgeAgentService, KnowledgeAgentUnavailable, KnowledgeAgentUpstreamError,
    _StreamEventFactory, citations_from_evidence,
)
from backend.src.storage import StorageError


@dataclass
class _Thread:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    history: list[tuple[str, str]] = field(default_factory=list)


def answer_with_citations(written, retrieval):
    """Resolve ephemeral labels to actual records/chunks before exposing text."""
    sources = {source.label: source for source in retrieval.sources}
    evidence, markers = [], []
    for label in dict.fromkeys(written.sources):
        if label not in sources:
            raise InvalidKnowledgeCitation("Answer cites an unknown source label")
        source = sources[label]
        if source.record is not None:
            record = source.record
            evidence.append({"kind": "knowledge", "knowledge_id": str(record.id),
                "subject": record.subject, "claim": record.claim, "type": record.type.value,
                "evidence_id": str(record.evidence_id), "stated_by": record.stated_by})
            markers.append(f"[K:{record.id}]")
        else:
            passage = source.passage
            evidence.append({"kind": "document", "doc_id": passage.doc_id, "title": passage.title})
            markers.append(f"[D:{passage.doc_id}]")
    # Writer returns prose and structured labels; validate any full IDs it
    # happens to put in prose as well, then normalize short label markers.
    for marker in re.findall(r"\[(?:K|D):[^\]]*\]", written.answer):
        if not (CITATION_RE.fullmatch(marker) or DOCUMENT_CITATION_RE.fullmatch(marker)):
            raise InvalidKnowledgeCitation("Answer contains a malformed citation")
    citations_from_evidence(written.answer, evidence)
    def replace_label(match):
        label = match[1]
        if label not in written.sources:
            raise InvalidKnowledgeCitation("Answer cites a label missing from its sources")
        source = sources[label]
        return f"[K:{source.record.id}]" if source.record else f"[D:{source.passage.doc_id}]"
    answer = re.sub(r"\[([KD]\d+)\]", replace_label, written.answer).strip()
    if markers:
        answer += "\n\n" + " ".join(marker for marker in dict.fromkeys(markers) if marker not in answer)
    return answer.strip(), citations_from_evidence(answer, evidence)


class IndustrialQaService(KnowledgeAgentService):
    """Same HTTP/CLI interface, with KB-specific retrieval and answer policy."""
    def __init__(self, spec, store, *, documents=None, pipeline=None,
                 timeout_seconds=RUN_TIMEOUT_SECONDS):
        self._spec, self._store = spec, store
        self._documents, self._owns_documents = documents, documents is None
        self._document_lock = asyncio.Lock()
        self.pipeline = pipeline if pipeline is not None else IndustrialQaPipeline()
        self._timeout_seconds = timeout_seconds
        self._threads: dict[str, _Thread] = {}

    async def _document_store(self):
        async with self._document_lock:
            if self._documents is None:
                self._documents = await open_document_store(self._spec,
                    query_instruction=DOCUMENT_QUERY_INSTRUCTION, init_schema=False)
        return self._documents

    def start_stream(self, query: str, *, thread_id: str):
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        request_id = uuid4().hex
        return AgentStreamSession(request_id, thread_id,
            self._stream(query.strip(), thread_id=thread_id, request_id=request_id))

    async def _stream(self, query, *, thread_id, request_id):
        events = _StreamEventFactory(request_id=request_id, thread_id=thread_id)
        started = monotonic()
        yield events.create("run.started", {"query_chars": len(query)})
        thread = self._threads.setdefault(thread_id, _Thread())
        try:
            async with asyncio.timeout(self._timeout_seconds), thread.lock:
                documents = await self._document_store()
                retrieval_query = query
                assets, family = detected_scope(query)
                if not assets and not family:
                    for prior_query, _ in reversed(thread.history):
                        if any(detected_scope(prior_query)):
                            retrieval_query = prior_query + "\nFollow-up question: " + query
                            break
                call_id = f"{request_id}:retrieval"
                yield events.create("tool.started", {"tool_call_id": call_id,
                    "tool": "industrial_qa_retrieval", "phase": "retrieving",
                    "input": {"query": retrieval_query[:2000]}})
                retrieved = await self.pipeline.retrieve(retrieval_query, self._store, documents)
                yield events.create("tool.completed", {"tool_call_id": call_id,
                    "tool": "industrial_qa_retrieval", "status": "success",
                    "duration_ms": int((monotonic() - started) * 1000),
                    "output": {"count": len(retrieved.sources), "candidate_count": retrieved.candidates,
                        "assets": list(retrieved.assets), "family": retrieved.family,
                        "knowledge_ids": [str(s.record.id) for s in retrieved.sources if s.record],
                        "document_ids": [s.passage.doc_id for s in retrieved.sources if s.passage]}})
                history = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in thread.history)[-6000:]
                written, _ = await self.pipeline.write(query, retrieved, history=history)
                answer, citations = answer_with_citations(written, retrieved)
                thread.history = [*thread.history, (query, written.answer)][-5:]
                yield events.create("answer.delta", {"text": answer})
                yield events.create("answer.citations", {"citations": [asdict(c) for c in citations]})
                yield events.create("run.completed", {"duration_ms": int((monotonic()-started)*1000),
                    "citation_count": len(citations)})
        except KnowledgeAgentError as exc:
            exc.request_id = request_id
            raise
        except (StorageError, SQLAlchemyError) as exc:
            raise KnowledgeAgentUnavailable("Industrial QA stores are unavailable",
                request_id=request_id, error_type=type(exc).__name__, reason=str(exc)) from exc
        except Exception as exc:
            raise KnowledgeAgentUpstreamError("Industrial QA pipeline failed",
                request_id=request_id, error_type=type(exc).__name__, reason=str(exc)) from exc

    async def delete_thread(self, thread_id):
        thread = self._threads.get(thread_id)
        if thread:
            async with thread.lock:
                thread.history.clear()
                self._threads.pop(thread_id, None)

    async def aclose(self):
        if self._owns_documents and self._documents is not None:
            await self._documents.aclose()
            self._documents = None
        self._threads.clear()
