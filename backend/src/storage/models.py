"""Persistence models for the knowledge base and its source evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from backend.src.services.knowledge_extraction import KnowledgeItem, KnowledgeType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationStatus(str, Enum):
    """Processing state of a raw conversation in the ingestion pipeline."""

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class KnowledgeStatus(str, Enum):
    """Review state of a knowledge record."""

    # Extracted but not yet reviewed by an expert; still searchable.
    DRAFT = "draft"
    APPROVED = "approved"
    # Retired or replaced; kept for citations and history, never searched.
    DEPRECATED = "deprecated"


class RawConversation(BaseModel):
    """A stored transcript that knowledge records cite as evidence."""

    id: UUID
    transcript: str
    stated_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.PENDING
    error: str | None = None
    created_at: datetime


class KnowledgeRecord(BaseModel):
    """One canonical knowledge row, as stored in the `knowledge_items` table."""

    id: UUID = Field(default_factory=uuid4)
    subject: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    type: KnowledgeType
    evidence_id: UUID | None = None
    stated_by: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    valid_from: date | None = None
    valid_to: date | None = None
    conflict_with: list[UUID] = Field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    # Cases a generalized record was drawn from, and cases that contradict it.
    supported_by: list[UUID] = Field(default_factory=list)
    contradicted_by: list[UUID] = Field(default_factory=list)
    # The record that replaced this one; replaced records are not searched.
    superseded_by: UUID | None = None
    # Why the record last changed: action, reason, actor, run, previous ids.
    revision: dict[str, Any] = Field(default_factory=dict)
    # Vectors are storage detail: keep them out of dumps, tool payloads, and logs.
    embedding: list[float] | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_extracted(
        cls,
        item: KnowledgeItem,
        *,
        evidence_id: UUID,
        stated_by: str | None = None,
        created_at: datetime | None = None,
        status: KnowledgeStatus = KnowledgeStatus.DRAFT,
    ) -> "KnowledgeRecord":
        """Attach ingestion metadata to one LLM-extracted knowledge item.

        item.supported_by holds positions within the extraction result; the
        caller resolves them to record ids once every record exists.
        """

        return cls(
            subject=item.subject,
            claim=item.claim,
            context=dict(item.context),
            type=item.type,
            evidence_id=evidence_id,
            stated_by=stated_by,
            created_at=created_at or _utcnow(),
            valid_from=item.valid_from,
            valid_to=item.valid_to,
            status=status,
        )

    def embedding_text(self, leading_keys: Sequence[str] = ()) -> str:
        """Text whose vector represents this record: subject, claim, and scope.

        Context keys in `leading_keys` (the knowledge base's
        embedding_context_keys) come first; the rest follow alphabetically.
        """

        parts = [f"{self.subject}: {self.claim}"]
        # Benchmark run IDs are provenance, not semantic scope. Structured
        # content (a Case Card, procedure steps) is already rendered into the
        # claim, so it is not indexed twice; Case Card evidence quotes are never
        # indexed, as they could hide omissions from the extracted knowledge.
        scope_context = {
            key: value for key, value in self.context.items() if key not in _UNINDEXED_CONTEXT
        }
        ordered = [key for key in leading_keys if key in scope_context]
        ordered += sorted(key for key in scope_context if key not in ordered)
        scope = "; ".join(
            f"{key}={_compact(scope_context[key])}" for key in ordered
        )
        if scope:
            parts.append(f"Context: {scope}")
        return "\n".join(parts)

    def lexical_text(self, leading_keys: Sequence[str] = ()) -> str:
        """Stable text indexed by PostgreSQL full-text search."""

        return self.embedding_text(leading_keys)

    def to_metadata(self, leading_keys: Sequence[str] = ()) -> dict[str, Any]:
        """Values for the vector store's metadata columns."""

        return {
            "subject": self.subject,
            "type": self.type.value,
            "context": self.context,
            "search_text": self.lexical_text(leading_keys),
            "evidence_id": str(self.evidence_id) if self.evidence_id else None,
            "stated_by": self.stated_by,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "conflict_with": _uuid_array(self.conflict_with),
            "status": self.status.value,
            "supported_by": _uuid_array(self.supported_by),
            "contradicted_by": _uuid_array(self.contradicted_by),
            "superseded_by": str(self.superseded_by) if self.superseded_by else None,
            "revision": self.revision,
        }

    @classmethod
    def from_document(cls, document: Document) -> "KnowledgeRecord":
        """Rebuild a record from a row returned by the vector store."""

        metadata = document.metadata
        context = metadata.get("context") or {}
        if isinstance(context, str):
            context = json.loads(context)
        revision = metadata.get("revision") or {}
        if isinstance(revision, str):
            revision = json.loads(revision)
        return cls(
            id=document.id,
            subject=metadata["subject"],
            claim=document.page_content,
            context=context,
            type=metadata["type"],
            evidence_id=metadata.get("evidence_id"),
            stated_by=metadata.get("stated_by"),
            created_at=metadata["created_at"],
            valid_from=metadata.get("valid_from"),
            valid_to=metadata.get("valid_to"),
            conflict_with=_parse_uuid_array(metadata.get("conflict_with")),
            status=metadata.get("status") or KnowledgeStatus.DRAFT,
            supported_by=_parse_uuid_array(metadata.get("supported_by")),
            contradicted_by=_parse_uuid_array(metadata.get("contradicted_by")),
            superseded_by=metadata.get("superseded_by"),
            revision=revision,
        )


@dataclass(frozen=True)
class KnowledgeLinkUpdate:
    """Links added to a stored record when new knowledge relates to it.

    Content is never edited in place: a changed idea is a new record that
    supersedes the old one, so citations of the old one stay valid.
    """

    record_id: UUID
    supported_by: tuple[UUID, ...] = ()
    contradicted_by: tuple[UUID, ...] = ()
    conflict_with: tuple[UUID, ...] = ()
    # Retires the record (status deprecated) in favour of this one.
    superseded_by: UUID | None = None
    # Appended to revision["history"], to explain the change.
    event: dict[str, Any] | None = None


@dataclass(frozen=True)
class KnowledgeSearchHit:
    """One hybrid-search result with per-retriever diagnostics."""

    record: KnowledgeRecord
    hybrid_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    vector_distance: float | None = None
    lexical_score: float | None = None
    lexical_matched_terms: int | None = None
    # True when the hit clears the similarity threshold or shares enough terms.
    confident: bool = False

    @property
    def vector_similarity(self) -> float | None:
        """Cosine similarity corresponding to pgvector's cosine distance."""

        if self.vector_distance is None:
            return None
        return 1.0 - self.vector_distance


# Context keys left out of the embedded and full-text indexed text.
_UNINDEXED_CONTEXT = frozenset({"benchmark_run_id", "case_card", "source_turns", "steps"})


def _uuid_array(ids: Sequence[UUID]) -> str:
    # A Postgres array literal lets the server coerce it to UUID[], including
    # the empty case that a bare Python list cannot type.
    return "{" + ",".join(str(i) for i in ids) + "}"


def _parse_uuid_array(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, str):
        return [i for i in value.strip("{}").split(",") if i]
    return list(value)


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)
