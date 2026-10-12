"""Typed runtime context and ephemeral evidence state for one agent run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

from langchain.agents import AgentState

from backend.src.agents.rag.store import DocumentStore
from backend.src.storage import KnowledgeStore


Evidence = dict[str, Any]


_IDENTITY_FIELDS = {"knowledge": "knowledge_id", "document": "doc_id"}


def _evidence_key(item: Evidence) -> tuple[str, str]:
    kind = str(item.get("kind", "unknown"))
    identity = item.get(_IDENTITY_FIELDS.get(kind, "evidence_id"))
    return kind, str(identity or "")


def merge_evidence(
    current: list[Evidence] | None, additions: list[Evidence] | None
) -> list[Evidence]:
    """Deduplicate evidence and let later, deeper reads enrich search hits."""
    merged: dict[tuple[str, str], Evidence] = {}
    order: list[tuple[str, str]] = []
    for item in [*(current or []), *(additions or [])]:
        key = _evidence_key(item)
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        enriched = dict(merged[key])
        for field, value in item.items():
            if value is not None and value != "" and value != []:
                enriched[field] = value
        merged[key] = enriched
    return [merged[key] for key in order]


class KnowledgeAgentState(AgentState):
    evidence: Annotated[list[Evidence], merge_evidence]


@dataclass(frozen=True)
class KnowledgeAgentContext:
    # KnowledgeStore for knowledge retrieval, DocumentStore for document retrieval.
    store: KnowledgeStore | DocumentStore
    now: datetime
    request_id: str = ""
