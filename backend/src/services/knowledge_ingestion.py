"""Turn raw conversations into stored, embedded knowledge records.

Optional steps, in order: normalize record scope with the knowledge base's
record normalizer (e.g. an equipment catalog) and its ontology, verify records
against their transcript, and consolidate them
with the stored knowledge. Each step only annotates records (revision) or
links them; every record is stored as a draft for expert review.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from backend.src.services.knowledge_consolidation import ConsolidationPlan, KnowledgeConsolidator
from backend.src.services.knowledge_extraction import KnowledgeExtractionResult
from backend.src.services.knowledge_verification import KnowledgeVerifier
from backend.src.services.ontology import Ontology
from backend.src.storage.models import (
    ConversationStatus,
    KnowledgeLinkUpdate,
    KnowledgeRecord,
    RawConversation,
)


logger = logging.getLogger(__name__)


# Rewrites the scope of the records extracted from one transcript in place.
RecordNormalizer = Callable[[str, list[KnowledgeRecord]], None]


class SupportsExtract(Protocol):
    async def aextract(self, raw_conversation: str) -> KnowledgeExtractionResult: ...


class SupportsKnowledgeWrite(Protocol):
    async def add_conversation(
        self,
        transcript: str,
        *,
        stated_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RawConversation: ...

    async def mark_conversation(
        self,
        conversation_id: UUID,
        status: ConversationStatus,
        *,
        error: str | None = None,
    ) -> None: ...

    async def embed_knowledge(self, records: Sequence[KnowledgeRecord]) -> None: ...

    async def add_knowledge(
        self,
        records: Sequence[KnowledgeRecord],
        *,
        updates: Sequence[KnowledgeLinkUpdate] = (),
    ) -> list[UUID]: ...

    # Only called when a consolidator is configured.
    async def find_processed_conversation(
        self, transcript_sha256: str, *, without_key: str | None = None
    ) -> UUID | None: ...


@dataclass(frozen=True)
class IngestionInput:
    transcript: str
    stated_by: str | None = None
    metadata: dict[str, Any] | None = field(default=None)
    # Merged into the context of every record extracted from this transcript,
    # overriding what the LLM wrote for the same keys (e.g. the source system).
    context: dict[str, Any] | None = field(default=None)


@dataclass(frozen=True)
class IngestionResult:
    evidence_id: UUID
    # Records stored; lessons merged into stored ones are not among them.
    records: tuple[KnowledgeRecord, ...]
    # Lessons merged into stored ones (revision["consolidation"]["target"]).
    merged: tuple[KnowledgeRecord, ...] = ()
    # Consolidation actions taken, e.g. {"add": 2, "merge": 1, "supports": 3}.
    consolidation: dict[str, int] = field(default_factory=dict)
    # Set, with no records, when the same transcript was already ingested.
    duplicate_of: UUID | None = None

    @property
    def knowledge_ids(self) -> tuple[UUID, ...]:
        return tuple(record.id for record in self.records)


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def transcript_sha256(transcript: str) -> str:
    return hashlib.sha256(transcript.strip().encode("utf-8")).hexdigest()


class KnowledgeIngestionService:
    """Store transcripts as evidence, extract knowledge, then embed and write it."""

    def __init__(
        self, extractor: SupportsExtract, store: SupportsKnowledgeWrite,
        *,
        extraction_metadata: dict[str, Any] | None = None,
        ontology: Ontology | None = None,
        normalizer: RecordNormalizer | None = None,
        verifier: KnowledgeVerifier | None = None,
        consolidator: KnowledgeConsolidator | None = None,
    ):
        self._extractor = extractor
        self._store = store
        self._extraction_metadata = dict(extraction_metadata or {})
        self._ontology = ontology
        self._normalizer = normalizer
        self._verifier = verifier
        self._consolidator = consolidator

    async def ingest(
        self,
        transcript: str,
        *,
        stated_by: str | None = None,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> IngestionResult:
        (result,) = await self.ingest_batch(
            [IngestionInput(transcript, stated_by, metadata, context)], concurrency=1
        )
        if isinstance(result, BaseException):
            raise result
        return result

    async def ingest_batch(
        self,
        inputs: Sequence[IngestionInput],
        *,
        concurrency: int = 8,
    ) -> list[IngestionResult | Exception]:
        """Ingest many transcripts with up to `concurrency` LLM calls in flight.

        Returns one entry per input, in order: its result, or the exception
        that failed it. One failed transcript does not affect the others.
        """

        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        results: list[IngestionResult | Exception | None] = [None] * len(inputs)

        # Evidence is persisted first so a failed extraction stays traceable.
        conversations: dict[int, RawConversation] = {}
        seen: dict[str, UUID] = {}
        for index, item in enumerate(inputs):
            if not item.transcript or not item.transcript.strip():
                results[index] = ValueError("transcript must not be empty")
                continue
            digest = transcript_sha256(item.transcript)
            if self._consolidator is not None:
                # Ingesting a transcript twice would count its cases twice as
                # support for the lessons it is merged into.
                duplicate_of = seen.get(digest) or await self._store.find_processed_conversation(
                    digest, without_key=self._consolidator.isolation_key
                )
                if duplicate_of is not None:
                    results[index] = IngestionResult(duplicate_of, (), duplicate_of=duplicate_of)
                    continue
            metadata = dict(item.metadata or {})
            metadata["transcript_sha256"] = digest
            if self._extraction_metadata:
                # Dataset metadata cannot spoof the active extraction settings.
                metadata["extraction"] = dict(self._extraction_metadata)
            conversations[index] = await self._store.add_conversation(
                item.transcript.strip(), stated_by=item.stated_by, metadata=metadata
            )
            seen[digest] = conversations[index].id

        semaphore = asyncio.Semaphore(concurrency)

        async def extract(index: int) -> KnowledgeExtractionResult:
            async with semaphore:
                return await self._extractor.aextract(conversations[index].transcript)

        indexes = list(conversations)
        extracted = await asyncio.gather(
            *(extract(index) for index in indexes), return_exceptions=True
        )

        records: dict[int, list[KnowledgeRecord]] = {}
        for index, outcome in zip(indexes, extracted, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):  # e.g. CancelledError
                    raise outcome
                await self._fail(conversations[index], outcome)
                results[index] = outcome
                continue
            group = [
                KnowledgeRecord.from_extracted(
                    item,
                    evidence_id=conversations[index].id,
                    stated_by=inputs[index].stated_by,
                )
                for item in outcome.knowledge_items
            ]
            # Support links point to positions in the result; store record ids.
            for record, item in zip(group, outcome.knowledge_items, strict=True):
                record.supported_by = [group[position].id for position in item.supported_by]
            records[index] = group
            if inputs[index].context:
                for record in group:
                    record.context = {**record.context, **inputs[index].context}
            if self._normalizer is not None:
                self._normalizer(conversations[index].transcript, group)
            for record in group:
                if self._ontology is not None:
                    record.context, report = self._ontology.normalize(record.context)
                    if ontology_report := report.to_dict():
                        record.revision = {**record.revision, "ontology": ontology_report}

        if self._verifier is not None:
            await asyncio.gather(*(
                self._verify(conversations[index].transcript, group, semaphore)
                for index, group in records.items()
            ))

        # One embedding call for the whole batch; if it fails, add_knowledge
        # embeds each conversation's records on its own, isolating the failure.
        batch_records = [record for group in records.values() for record in group]
        if batch_records:
            try:
                await self._store.embed_knowledge(batch_records)
            except Exception:
                logger.warning(
                    "Batch embedding failed; embedding per conversation", exc_info=True
                )

        # Conversations are consolidated one at a time, so each one sees the
        # lessons the previous ones stored.
        for index, group in records.items():
            conversation = conversations[index]
            try:
                plan = await self._plan(group)
                await self._store.add_knowledge(plan.records, updates=plan.updates)
            except Exception as exc:
                await self._fail(conversation, exc)
                results[index] = exc
                continue
            await self._store.mark_conversation(conversation.id, ConversationStatus.PROCESSED)
            stored = {record.id for record in plan.records}
            results[index] = IngestionResult(
                evidence_id=conversation.id,
                records=tuple(plan.records),
                merged=tuple(record for record in group if record.id not in stored),
                consolidation=plan.summary,
            )

        return results  # type: ignore[return-value]  # every slot is filled

    async def _verify(
        self, transcript: str, records: list[KnowledgeRecord], semaphore: asyncio.Semaphore
    ) -> None:
        assert self._verifier is not None
        async with semaphore:
            try:
                outcomes = await self._verifier.averify(transcript, records)
            except Exception as exc:
                # Verification is advisory; it never blocks storing a record.
                logger.warning("Verification failed", exc_info=True)
                outcomes = [{**self._verifier.provenance, "error": _error_text(exc)}] * len(records)
        for record, outcome in zip(records, outcomes, strict=True):
            record.revision = {**record.revision, "verify": outcome}

    async def _plan(self, records: list[KnowledgeRecord]) -> ConsolidationPlan:
        if self._consolidator is None or not records:
            return ConsolidationPlan(records=records)
        try:
            return await self._consolidator.aplan(records)
        except Exception as exc:
            # Unrelated records are better than none: store all as new.
            logger.warning("Consolidation failed; adding records as new", exc_info=True)
            for record in records:
                record.revision = {
                    **record.revision,
                    "consolidation": {"action": "add", "error": _error_text(exc)[:500]},
                }
            return ConsolidationPlan(records=records)

    async def _fail(self, conversation: RawConversation, exc: Exception) -> None:
        await self._store.mark_conversation(
            conversation.id, ConversationStatus.FAILED, error=_error_text(exc)
        )
