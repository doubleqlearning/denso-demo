"""Application service that runs the knowledge agent and validates citations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Collection, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any
from uuid import uuid4

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite
from openai import APIError
from sqlalchemy.exc import SQLAlchemyError

from backend.knowledge_bases.base import AnswerPipeline, ContextFilter, KnowledgeBaseSpec, Retrieval
from backend.src.agents.agent.config import (
    RECURSION_LIMIT,
    RUN_TIMEOUT_SECONDS,
    build_chat_model_from_env,
)
from backend.src.agents.agent.graph import build_graph
from backend.src.agents.agent.state import Evidence, KnowledgeAgentContext
from backend.src.agents.rag.store import DocumentStore
from backend.src.storage import KnowledgeStore, StorageError


logger = logging.getLogger(__name__)
CITATION_RE = re.compile(
    r"\[K:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
)
DOCUMENT_CITATION_RE = re.compile(r"\[D:([^\]\s]{1,200})\]")
LOG_CONTENT_LIMIT = 2_000
STREAM_EVENT_VERSION = 1
PUBLIC_TRACE_STRING_LIMIT = 2_000
# Context keys are knowledge-base specific: callers pass the ones a trace may
# show (KnowledgeBaseSpec.trace_context_keys) as context_fields.
PUBLIC_TOOL_INPUT_FIELDS = {
    "search_knowledge": {
        "query",
        "subject",
        "knowledge_type",
        "valid_on",
        "k",
    },
    "get_knowledge": {"knowledge_id"},
    "get_evidence": {"evidence_id", "offset", "limit"},
    "search_documents": {"query", "k"},
    "get_document": {"doc_id", "start"},
}
# Source text in tool payloads: kept out of logs and traces.
SOURCE_TEXT_KEYS = frozenset({"transcript", "text", "passages"})
SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;]+)"
)
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
DATABASE_URL_RE = re.compile(
    r"(?i)(postgres(?:ql)?(?:\+\w+)?://)([^\s/@:]+):([^\s/@]+)@"
)


def _safe_text(value: object, *, limit: int = LOG_CONTENT_LIMIT) -> str:
    text = SECRET_VALUE_RE.sub(r"\1\2<redacted>", str(value))
    text = OPENAI_KEY_RE.sub("<redacted>", text)
    text = DATABASE_URL_RE.sub(r"\1<redacted>:<redacted>@", text)
    if len(text) > limit:
        return f"{text[:limit]}... <truncated {len(text) - limit} chars>"
    return text


class KnowledgeAgentError(RuntimeError):
    """Base error carrying a safe diagnostic for API responses."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        error_type: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.error_type = error_type or type(self).__name__
        self.reason = _safe_text(reason or message, limit=1_000)

    def public_detail(self) -> dict[str, str]:
        detail = {
            "message": str(self),
            "error_type": self.error_type,
            "reason": self.reason,
        }
        if self.request_id:
            detail["request_id"] = self.request_id
        return detail


class KnowledgeAgentUnavailable(KnowledgeAgentError):
    """The knowledge base or embedding service cannot currently serve requests."""


class KnowledgeAgentUpstreamError(KnowledgeAgentError):
    """The model invocation failed, timed out, or produced no answer."""


class InvalidKnowledgeCitation(KnowledgeAgentError):
    """The model cited a knowledge or document id that no tool returned."""


@dataclass(frozen=True)
class KnowledgeCitation:
    knowledge_id: str
    subject: str | None
    claim: str | None
    type: str | None
    evidence_id: str | None
    stated_by: str | None


@dataclass(frozen=True)
class DocumentCitation:
    doc_id: str
    title: str | None


Citation = KnowledgeCitation | DocumentCitation


def citation_from_dict(item: dict[str, Any]) -> Citation:
    return DocumentCitation(**item) if "doc_id" in item else KnowledgeCitation(**item)


@dataclass(frozen=True)
class KnowledgeAnswer:
    answer: str
    citations: tuple[Citation, ...]
    request_id: str = ""


@dataclass(frozen=True)
class AgentStreamEvent:
    version: int
    sequence: int
    event: str
    request_id: str
    thread_id: str
    timestamp: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned public event envelope."""

        return {
            "version": self.version,
            "sequence": self.sequence,
            "type": self.event,
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass(frozen=True)
class AgentStreamSession:
    request_id: str
    thread_id: str
    events: AsyncIterator[AgentStreamEvent]


class _StreamEventFactory:
    """Create monotonically sequenced trace events for one agent run."""

    def __init__(self, *, request_id: str, thread_id: str) -> None:
        self.request_id = request_id
        self.thread_id = thread_id
        self.sequence = 0

    def create(self, event: str, data: dict[str, Any] | None = None) -> AgentStreamEvent:
        self.sequence += 1
        return AgentStreamEvent(
            version=STREAM_EVENT_VERSION,
            sequence=self.sequence,
            event=event,
            request_id=self.request_id,
            thread_id=self.thread_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            data=data or {},
        )


def _content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and isinstance(block.get("text"), str))
        )
    return str(content)


def _public_value(value: Any) -> Any:
    """Convert trace values to bounded JSON-safe values."""

    if isinstance(value, str):
        return value[:PUBLIC_TRACE_STRING_LIMIT]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    return _public_value(enum_value if enum_value is not None else str(value))


def public_tool_input(
    tool_name: str, arguments: Any, *, context_fields: Collection[str] = ()
) -> dict[str, Any]:
    """Allowlist the tool arguments that may be exposed to a trace consumer."""

    if not isinstance(arguments, dict):
        return {}
    allowed = PUBLIC_TOOL_INPUT_FIELDS.get(tool_name)
    if allowed is None:
        return {}
    if tool_name == "search_knowledge":
        allowed = allowed | set(context_fields)
    return {
        key: _public_value(value)
        for key, value in arguments.items()
        if key in allowed and value is not None
    }


def _public_knowledge_summary(
    item: Any, *, context_fields: Collection[str] = ()
) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    context = item.get("context")
    public_context = (
        {
            key: _public_value(value)
            for key, value in context.items()
            if key in context_fields
        }
        if isinstance(context, dict)
        else {}
    )
    summary = {
        "knowledge_id": item.get("id") or item.get("knowledge_id"),
        "subject": item.get("subject"),
        "type": item.get("type"),
        "status": item.get("status"),
        "validity": item.get("validity"),
    }
    if public_context:
        summary["context"] = public_context
    retrieval = item.get("retrieval")
    if isinstance(retrieval, dict):
        summary["retrieval"] = {
            key: _public_value(retrieval.get(key))
            for key in (
                "hybrid_score",
                "vector_rank",
                "lexical_rank",
                "vector_similarity",
                "lexical_score",
                "lexical_matched_terms",
                "confident",
            )
            if retrieval.get(key) is not None
        }
    return {
        key: _public_value(value)
        for key, value in summary.items()
        if value is not None
    }


def _public_document_summary(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    summary = {key: item.get(key) for key in ("doc_id", "chunk_id", "title")}
    retrieval = item.get("retrieval")
    if isinstance(retrieval, dict):
        summary["retrieval"] = {
            key: retrieval.get(key)
            for key in ("hybrid_score", "vector_rank", "lexical_rank")
            if retrieval.get(key) is not None
        }
    return {key: _public_value(value) for key, value in summary.items() if value is not None}


def public_tool_output(
    tool_name: str, payload: Any, *, context_fields: Collection[str] = ()
) -> dict[str, Any]:
    """Return a useful trace summary without exposing claims or raw transcripts."""

    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid_tool_output"}
    ok = payload.get("ok") is True
    result: dict[str, Any] = {"ok": ok}
    if not ok:
        if payload.get("error"):
            result["error"] = _public_value(payload["error"])
        return result

    if tool_name == "search_knowledge":
        knowledge = payload.get("knowledge")
        items = knowledge if isinstance(knowledge, list) else []
        result.update(
            {
                "match": _public_value(payload.get("match")),
                "count": len(items),
                "items": [
                    _public_knowledge_summary(item, context_fields=context_fields)
                    for item in items
                ],
            }
        )
    elif tool_name == "get_knowledge":
        result["knowledge"] = _public_knowledge_summary(
            payload.get("knowledge"), context_fields=context_fields
        )
    elif tool_name == "get_evidence":
        evidence = payload.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        transcript = evidence.get("transcript")
        result["evidence"] = {
            key: _public_value(value)
            for key, value in {
                "evidence_id": evidence.get("id"),
                "stated_by": evidence.get("stated_by"),
                "created_at": evidence.get("created_at"),
                "returned_transcript_chars": (
                    len(transcript) if isinstance(transcript, str) else 0
                ),
                "truncated": evidence.get("truncated"),
                "offset": evidence.get("offset"),
                "total_chars": evidence.get("total_chars"),
                "next_offset": evidence.get("next_offset"),
            }.items()
            if value is not None
        }
    elif tool_name == "search_documents":
        documents = payload.get("documents")
        items = documents if isinstance(documents, list) else []
        result.update(
            {"count": len(items), "documents": [_public_document_summary(item) for item in items]}
        )
    elif tool_name == "get_document":
        document = payload.get("document")
        document = document if isinstance(document, dict) else {}
        passages = document.get("passages")
        passages = passages if isinstance(passages, list) else []
        result["document"] = {
            key: _public_value(value)
            for key, value in {
                "doc_id": document.get("doc_id"),
                "title": document.get("title"),
                "returned_passages": len(passages),
                "returned_chars": sum(len(item) for item in passages if isinstance(item, str)),
                "start": document.get("start"),
                "total_passages": document.get("total_passages"),
                "truncated": document.get("truncated"),
                "next_start": document.get("next_start"),
            }.items()
            if value is not None
        }
    return result


def _tool_message_payload(message: ToolMessage) -> dict[str, Any]:
    try:
        payload = json.loads(_content_text(message))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_tool_output"}
    return payload if isinstance(payload, dict) else {"ok": False, "error": "invalid_tool_output"}


def _root_exception(exc: BaseException) -> BaseException:
    current = exc
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "exceptions", None)
        if nested:
            current = nested[0]
            continue
        cause = current.__cause__ or current.__context__
        if cause is None:
            break
        current = cause
    return current


def _contains_exception(
    exc: BaseException, classes: tuple[type[BaseException], ...]
) -> bool:
    if isinstance(exc, classes):
        return True
    nested = getattr(exc, "exceptions", None) or ()
    if any(_contains_exception(item, classes) for item in nested):
        return True
    cause = exc.__cause__ or exc.__context__
    return cause is not None and _contains_exception(cause, classes)


def _diagnostic(exc: BaseException) -> tuple[str, str]:
    root = _root_exception(exc)
    reason = _safe_text(root, limit=1_000) or "No error detail was supplied by the dependency"
    return type(root).__name__, reason


def _sanitize_tool_payload(value: Any, *, key: str | None = None) -> Any:
    if key in SOURCE_TEXT_KEYS and isinstance(value, str):
        return f"<omitted {len(value)} chars>"
    if key in SOURCE_TEXT_KEYS and isinstance(value, list):
        return f"<omitted {len(value)} items>"
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_tool_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_tool_payload(item) for item in value]
    return value


def _logged_message_content(message: Any) -> str:
    text = _content_text(message).strip()
    if isinstance(message, ToolMessage):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return _safe_text(text)
        text = json.dumps(_sanitize_tool_payload(payload), ensure_ascii=False, default=str)
    return _safe_text(text)


def _log_new_messages(request_id: str, messages: list[Any], already_logged: int) -> int:
    for index, message in enumerate(messages[already_logged:], start=already_logged):
        tool_calls = getattr(message, "tool_calls", None) or []
        logger.info(
            "Knowledge agent message request_id=%s index=%d role=%s tool_call_id=%s "
            "tool_calls=%s content=%s",
            request_id,
            index,
            getattr(message, "type", type(message).__name__),
            getattr(message, "tool_call_id", None),
            _safe_text(json.dumps(tool_calls, ensure_ascii=False, default=str)),
            _logged_message_content(message),
        )
    return len(messages)


def _tool_status(tool_name: str) -> str:
    return "searching" if tool_name.startswith("search_") else "reading"


def citations_from_evidence(answer: str, evidence: list[Evidence]) -> tuple[Citation, ...]:
    """Resolve [K:<id>] and [D:<id>] markers against what the tools actually returned."""
    return (
        _knowledge_citations(answer, evidence)
        + _document_citations(answer, evidence)
    )


def _document_citations(answer: str, evidence: list[Evidence]) -> tuple[DocumentCitation, ...]:
    documents = {
        str(item["doc_id"]): item
        for item in evidence
        if item.get("kind") == "document" and item.get("doc_id") is not None
    }
    ordered_ids: list[str] = []
    for doc_id in DOCUMENT_CITATION_RE.findall(answer):
        if doc_id not in documents:
            raise InvalidKnowledgeCitation(
                f"Citation D:{doc_id} is not present in retrieved evidence"
            )
        if doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
    return tuple(
        DocumentCitation(doc_id=doc_id, title=documents[doc_id].get("title"))
        for doc_id in ordered_ids
    )


def _knowledge_citations(answer: str, evidence: list[Evidence]) -> tuple[KnowledgeCitation, ...]:
    records = {
        str(item["knowledge_id"]).lower(): item
        for item in evidence
        if item.get("kind") == "knowledge" and item.get("knowledge_id")
    }
    ordered_ids: list[str] = []
    for raw_id in CITATION_RE.findall(answer):
        knowledge_id = raw_id.lower()
        if knowledge_id not in records:
            raise InvalidKnowledgeCitation(
                f"Citation K:{knowledge_id} is not present in retrieved evidence"
            )
        if knowledge_id not in ordered_ids:
            ordered_ids.append(knowledge_id)
    return tuple(
        KnowledgeCitation(
            knowledge_id=knowledge_id,
            subject=records[knowledge_id].get("subject"),
            claim=records[knowledge_id].get("claim"),
            type=records[knowledge_id].get("type"),
            evidence_id=records[knowledge_id].get("evidence_id"),
            stated_by=records[knowledge_id].get("stated_by"),
        )
        for knowledge_id in ordered_ids
    )


class KnowledgeAgentService:
    """Runs the retrieval agent for one question or a multi-turn thread."""

    def __init__(
        self,
        *,
        store: KnowledgeStore | DocumentStore,
        model: Any,
        system_prompt: str,
        timeout_seconds: float = RUN_TIMEOUT_SECONDS,
        checkpointer: InMemorySaver | None = None,
        context_filters: Sequence[ContextFilter] = (),
        trace_context_keys: Collection[str] = (),
        retrieval: Retrieval = Retrieval.KNOWLEDGE,
        knowledge_filter_builder=None,
    ) -> None:
        self._store = store
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._checkpointer = checkpointer or InMemorySaver()
        # Filter arguments are always safe to show; add them to the trace keys.
        self._trace_context_keys = frozenset(
            [*trace_context_keys, *(item.key for item in context_filters)]
        )
        self._graph = build_graph(
            model,
            system_prompt=system_prompt,
            context_filters=context_filters,
            retrieval=retrieval,
            checkpointer=self._checkpointer,
            **({"filter_builder": knowledge_filter_builder} if knowledge_filter_builder else {}),
        )

    @classmethod
    def from_kb(
        cls, spec: KnowledgeBaseSpec, store: KnowledgeStore | DocumentStore
    ) -> "KnowledgeAgentService":
        """Agent with the knowledge base's prompt over its (already open) store."""

        if spec.answer_pipeline is AnswerPipeline.INDUSTRIAL_QA:
            from backend.knowledge_bases.industrial_qa.service import IndustrialQaService
            return IndustrialQaService(spec, store)
        knowledge = spec.retrieval is Retrieval.KNOWLEDGE
        return cls(
            store=store,
            model=build_chat_model_from_env(),
            system_prompt=spec.agent_prompt,
            context_filters=spec.context_filters if knowledge else (),
            trace_context_keys=spec.trace_context_keys if knowledge else (),
            retrieval=spec.retrieval,
            **({"knowledge_filter_builder": spec.knowledge_filter_builder}
               if knowledge and spec.knowledge_filter_builder else {}),
        )

    def start_stream(self, query: str, *, thread_id: str) -> AgentStreamSession:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        request_id = uuid4().hex
        return AgentStreamSession(
            request_id=request_id,
            thread_id=thread_id,
            events=self._stream_graph(
                query=query.strip(), thread_id=thread_id, request_id=request_id
            ),
        )

    async def _stream_graph(
        self,
        *,
        query: str,
        thread_id: str,
        request_id: str,
    ) -> AsyncIterator[AgentStreamEvent]:
        graph = self._graph
        started = monotonic()
        now = datetime.now(timezone.utc)
        context = KnowledgeAgentContext(store=self._store, now=now, request_id=request_id)
        config = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        }
        snapshot = await graph.aget_state(config)
        baseline_messages = len((snapshot.values or {}).get("messages") or [])
        logged_messages = baseline_messages
        observed_messages = baseline_messages
        emitted_tool_calls: set[str] = set()
        emitted_tool_results: set[str] = set()
        tool_names: dict[str, str] = {}
        tool_started_at: dict[str, float] = {}
        current_generation: str | None = None
        current_generation_uses_tools = False
        visible_answer = ""
        result: dict[str, Any] | None = None
        events = _StreamEventFactory(request_id=request_id, thread_id=thread_id)

        logger.info(
            "Knowledge agent started request_id=%s thread_id=%s query_chars=%d "
            "history_messages=%d",
            request_id,
            thread_id,
            len(query),
            baseline_messages,
        )
        yield events.create("run.started", {"query_chars": len(query)})

        turn_message = f"[Turn context: current UTC time is {now.isoformat()}]\n\n{query}"
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for part in graph.astream(
                    {
                        "messages": [HumanMessage(content=turn_message)],
                        "evidence": Overwrite([]),
                    },
                    config=config,
                    context=context,
                    stream_mode=["messages", "values"],
                    version="v2",
                ):
                    part_type = part.get("type")
                    if part_type == "messages":
                        message, metadata = part["data"]
                        if metadata.get("langgraph_node") != "model":
                            continue
                        generation = str(getattr(message, "id", None) or "model")
                        if generation != current_generation:
                            current_generation = generation
                            current_generation_uses_tools = False
                        calls = [
                            *(getattr(message, "tool_calls", None) or []),
                            *(getattr(message, "tool_call_chunks", None) or []),
                        ]
                        if calls:
                            # Text streamed before a tool call was not the answer.
                            if not current_generation_uses_tools and visible_answer:
                                visible_answer = ""
                                yield events.create("answer.reset")
                            current_generation_uses_tools = True
                            continue
                        if current_generation_uses_tools:
                            continue
                        delta = _content_text(message)
                        if delta:
                            visible_answer += delta
                            yield events.create("answer.delta", {"text": delta})
                        continue

                    if part_type != "values":
                        continue
                    result = part["data"]
                    messages = list(result.get("messages") or [])
                    logged_messages = _log_new_messages(request_id, messages, logged_messages)
                    for observed in messages[observed_messages:]:
                        if isinstance(observed, AIMessage):
                            for index, call in enumerate(observed.tool_calls or []):
                                tool_name = str(call.get("name") or "")
                                call_id = str(
                                    call.get("id")
                                    or f"{getattr(observed, 'id', 'model')}:{index}:{tool_name}"
                                )
                                if not tool_name or call_id in emitted_tool_calls:
                                    continue
                                emitted_tool_calls.add(call_id)
                                tool_names[call_id] = tool_name
                                tool_started_at[call_id] = monotonic()
                                yield events.create(
                                    "tool.started",
                                    {
                                        "tool_call_id": call_id,
                                        "tool": tool_name,
                                        "phase": _tool_status(tool_name),
                                        "input": public_tool_input(
                                            tool_name,
                                            call.get("args") or {},
                                            context_fields=self._trace_context_keys,
                                        ),
                                    },
                                )
                        elif isinstance(observed, ToolMessage):
                            call_id = str(observed.tool_call_id or "")
                            if not call_id or call_id in emitted_tool_results:
                                continue
                            emitted_tool_results.add(call_id)
                            tool_name = str(
                                getattr(observed, "name", None)
                                or tool_names.get(call_id)
                                or "unknown_tool"
                            )
                            payload = _tool_message_payload(observed)
                            started_at = tool_started_at.get(call_id)
                            duration_ms = (
                                int((monotonic() - started_at) * 1000)
                                if started_at is not None
                                else None
                            )
                            trace_data: dict[str, Any] = {
                                "tool_call_id": call_id,
                                "tool": tool_name,
                                "status": (
                                    "success" if payload.get("ok") is True else "error"
                                ),
                                "output": public_tool_output(
                                    tool_name,
                                    payload,
                                    context_fields=self._trace_context_keys,
                                ),
                            }
                            if duration_ms is not None:
                                trace_data["duration_ms"] = duration_ms
                            yield events.create("tool.completed", trace_data)
                    observed_messages = len(messages)
        except TimeoutError as exc:
            reason = f"Agent exceeded the {self._timeout_seconds:g}s run timeout"
            logger.warning(
                "Knowledge agent timed out request_id=%s thread_id=%s", request_id, thread_id
            )
            raise KnowledgeAgentUpstreamError(
                "Knowledge agent request timed out",
                request_id=request_id,
                error_type="TimeoutError",
                reason=reason,
            ) from exc
        except asyncio.CancelledError:
            logger.info(
                "Knowledge agent cancelled request_id=%s thread_id=%s", request_id, thread_id
            )
            raise
        except Exception as exc:
            error_type, reason = _diagnostic(exc)
            retrieval_failed = _contains_exception(exc, (StorageError, SQLAlchemyError))
            logger.exception(
                "Knowledge agent failed request_id=%s thread_id=%s phase=%s "
                "error_type=%s reason=%s",
                request_id,
                thread_id,
                "retrieval" if retrieval_failed else "graph",
                error_type,
                reason,
            )
            if retrieval_failed:
                raise KnowledgeAgentUnavailable(
                    "Knowledge base is unavailable",
                    request_id=request_id,
                    error_type=error_type,
                    reason=reason,
                ) from exc
            message = (
                "Model or embedding request failed"
                if _contains_exception(exc, (APIError,))
                else "Knowledge agent run failed"
            )
            raise KnowledgeAgentUpstreamError(
                message, request_id=request_id, error_type=error_type, reason=reason
            ) from exc

        if result is None:
            raise KnowledgeAgentUpstreamError(
                "Knowledge agent returned no result",
                request_id=request_id,
                error_type="EmptyGraphResult",
            )

        messages = list(result.get("messages") or [])
        final = next(
            (
                message
                for message in reversed(messages[baseline_messages:])
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )
        answer = _content_text(final).strip() if final is not None else ""
        if not answer:
            raise KnowledgeAgentUpstreamError(
                "Knowledge agent returned no final answer",
                request_id=request_id,
                error_type="MissingFinalAnswer",
            )

        evidence = list(result.get("evidence") or [])
        try:
            citations = citations_from_evidence(answer, evidence)
        except InvalidKnowledgeCitation as exc:
            logger.warning(
                "Knowledge agent invalid citation request_id=%s thread_id=%s reason=%s",
                request_id,
                thread_id,
                exc,
            )
            raise InvalidKnowledgeCitation(
                "Knowledge agent returned an invalid citation",
                request_id=request_id,
                reason=str(exc),
            ) from exc

        if visible_answer != answer:
            if visible_answer:
                yield events.create("answer.reset")
            yield events.create("answer.delta", {"text": answer})
        yield events.create(
            "answer.citations",
            {"citations": [asdict(citation) for citation in citations]},
        )
        yield events.create(
            "run.completed",
            {
                "duration_ms": int((monotonic() - started) * 1000),
                "citation_count": len(citations),
            },
        )

        current_messages = messages[baseline_messages:]
        logger.info(
            "Knowledge agent completed request_id=%s thread_id=%s duration_ms=%d "
            "model_calls=%d tool_calls=%d evidence=%d citations=%d",
            request_id,
            thread_id,
            int((monotonic() - started) * 1000),
            sum(isinstance(message, AIMessage) for message in current_messages),
            sum(
                len(message.tool_calls)
                for message in current_messages
                if isinstance(message, AIMessage)
            ),
            len(evidence),
            len(citations),
        )

    async def ask(self, query: str) -> KnowledgeAnswer:
        """Answer one standalone question; the thread is discarded afterwards."""
        thread_id = uuid4().hex
        session = self.start_stream(query, thread_id=thread_id)
        answer = ""
        citations: tuple[Citation, ...] = ()
        completed = False
        try:
            async for event in session.events:
                if event.event == "answer.delta":
                    answer += str(event.data.get("text") or "")
                elif event.event == "answer.reset":
                    answer = ""
                elif event.event == "answer.citations":
                    citations = tuple(
                        citation_from_dict(item)
                        for item in event.data.get("citations") or ()
                    )
                elif event.event == "run.completed":
                    completed = True
        finally:
            await self.delete_thread(thread_id)
        if not completed:
            raise KnowledgeAgentUpstreamError(
                "Knowledge agent stream ended before completion",
                request_id=session.request_id,
                error_type="IncompleteStream",
            )
        return KnowledgeAnswer(
            answer=answer, citations=citations, request_id=session.request_id
        )

    async def delete_thread(self, thread_id: str) -> None:
        await self._checkpointer.adelete_thread(thread_id)

    async def aclose(self) -> None:
        """Release service-owned resources; the caller owns its primary store."""
