"""HTTP streaming API for the knowledge agent's public trace."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.knowledge_bases import KnowledgeBaseSpec
from backend.knowledge_bases.stores import open_store
from backend.src.agents.agent.service import (
    STREAM_EVENT_VERSION,
    AgentStreamEvent,
    KnowledgeAgentError,
    KnowledgeAgentService,
)
from backend.src.agents.rag.store import DocumentStore
from backend.src.storage import KnowledgeStore, StorageError
from backend.src.storage.databases import DatabaseAdminError


router = APIRouter(prefix="/v1/knowledge-agent", tags=["knowledge-agent"])


class KnowledgeAgentPool:
    """One agent per knowledge base, opened on its first request."""

    def __init__(
        self,
        specs: Mapping[str, KnowledgeBaseSpec],
        *,
        agents: Mapping[str, Any] | None = None,
    ) -> None:
        self._specs = dict(specs)
        # Pre-built agents, e.g. fakes in HTTP tests.
        self._agents: dict[str, Any] = dict(agents or {})
        self._stores: list[KnowledgeStore | DocumentStore] = []
        self._lock = asyncio.Lock()

    def names(self) -> list[str]:
        return sorted({*self._specs, *self._agents})

    def description(self, name: str) -> str:
        spec = self._specs.get(name)
        return spec.description.strip() if spec else ""

    async def get(self, name: str) -> Any:
        if name in self._agents:
            return self._agents[name]
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(name)
        async with self._lock:
            if name not in self._agents:
                store = await open_store(spec)
                try:
                    agent = KnowledgeAgentService.from_kb(spec, store)
                except Exception:
                    await store.aclose()
                    raise
                self._stores.append(store)
                self._agents[name] = agent
        return self._agents[name]

    async def aclose(self) -> None:
        for agent in self._agents.values():
            if close := getattr(agent, "aclose", None):
                await close()
        self._agents.clear()
        for store in self._stores:
            await store.aclose()
        self._stores.clear()


class AgentQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=200)

    @field_validator("query", "thread_id", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class KnowledgeAgentStreamRequest(AgentQueryRequest):
    kb: str = Field(min_length=1, max_length=64)

    @field_validator("kb", mode="before")
    @classmethod
    def strip_kb(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ScopedAgentQueryRequest(AgentQueryRequest):
    # A KB is selected by the URL; reject a conflicting KB in the body.
    model_config = {"extra": "forbid"}


def encode_sse(event: AgentStreamEvent) -> str:
    """Encode one normalized event using the Server-Sent Events wire format."""

    event_id = f"{event.request_id}:{event.sequence}"
    payload = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event.event}\ndata: {payload}\n\n"


def _failure_event(
    *,
    request_id: str,
    thread_id: str,
    sequence: int,
    error: KnowledgeAgentError | Exception,
) -> AgentStreamEvent:
    if isinstance(error, KnowledgeAgentError):
        detail = error.public_detail()
    else:
        detail = {
            "message": "Knowledge agent request failed",
            "error_type": type(error).__name__,
        }
    return AgentStreamEvent(
        version=STREAM_EVENT_VERSION,
        sequence=sequence,
        event="run.failed",
        request_id=request_id,
        thread_id=thread_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        data=detail,
    )


def _pool(request: Request) -> KnowledgeAgentPool:
    pool = getattr(request.app.state, "knowledge_agents", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Knowledge agent is not initialized")
    return pool


async def _service(request: Request, kb: str) -> KnowledgeAgentService:
    try:
        return await _pool(request).get(kb)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Unknown knowledge base: {kb}"
        ) from None
    except (StorageError, DatabaseAdminError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base {kb} is unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/knowledge-bases")
async def list_knowledge_bases(request: Request) -> dict[str, list[dict[str, str]]]:
    pool = _pool(request)
    return {
        "knowledge_bases": [
            {"name": name, "description": pool.description(name)}
            for name in pool.names()
        ]
    }


@router.post("/stream", response_class=StreamingResponse)
async def stream_knowledge_agent(
    request: Request,
    payload: Annotated[KnowledgeAgentStreamRequest, Body(...)],
) -> StreamingResponse:
    """Compatibility endpoint: select a knowledge base in the JSON body."""
    service = await _service(request, payload.kb)
    return _stream_response(service, payload)


@router.post("/{kb}/stream", response_class=StreamingResponse)
async def stream_scoped_knowledge_agent(
    kb: str,
    request: Request,
    payload: Annotated[ScopedAgentQueryRequest, Body(...)],
) -> StreamingResponse:
    """Stream a KB's agent, including /enterprise_rag/stream, with its own history."""
    service = await _service(request, kb)
    return _stream_response(service, payload)


def _stream_response(
    service: KnowledgeAgentService, payload: AgentQueryRequest
) -> StreamingResponse:
    session = service.start_stream(payload.query, thread_id=payload.thread_id)

    async def event_source():
        last_sequence = 0
        try:
            async for event in session.events:
                last_sequence = event.sequence
                yield encode_sse(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield encode_sse(
                _failure_event(
                    request_id=session.request_id,
                    thread_id=session.thread_id,
                    sequence=last_sequence + 1,
                    error=exc,
                )
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
