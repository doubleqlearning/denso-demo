"""Typed Case Card and lesson extraction schema for the industrial QA benchmark."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.src.services.knowledge_extraction import (
    KnowledgeExtractionResult,
    KnowledgeItem,
    KnowledgeType,
    StructuredExtraction,
    scope_context,
)


_TURN_ID_RE = re.compile(r"\[?T?0*(\d+)\]?")
_TURN_MARKER_RE = re.compile(r"\s*\[(T\d+)\]\s*")


def canonical_turn_id(value: Any) -> Any:
    """Canonical T05-style turn ID of "T5", "[T05]" or 5; other values unchanged."""

    match = _TURN_ID_RE.fullmatch(str(value).strip()) if value is not None else None
    return f"T{int(match.group(1)):02d}" if match else value


class CaseEvidence(BaseModel):
    """A verbatim transcript excerpt and the turn it is quoted from."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(pattern=r"^T\d{2,}$")
    quote: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _split_turn_marker(cls, data: Any) -> Any:
        # Models copy the transcript's "[T05]" line marker into the quote
        # instead of turn_id; the quote is then not verbatim.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        quote = str(data.get("quote") or "")
        if markers := _TURN_MARKER_RE.findall(quote):
            data["quote"] = _TURN_MARKER_RE.sub(" ", quote).strip()
            data["turn_id"] = data.get("turn_id") or markers[0]
        data["turn_id"] = canonical_turn_id(data.get("turn_id"))
        return data


class CaseCard(BaseModel):
    """Context, observation, reasoning, action and outcome of one incident."""

    model_config = ConfigDict(extra="forbid")

    context: str
    observation: str
    reasoning: str
    action: str
    outcome: str | None = None
    evidence: list[CaseEvidence] = Field(min_length=1)


class IndustrialScope(BaseModel):
    """Where a case or lesson applies; every key is decided, null ones are not stored."""

    model_config = ConfigDict(extra="forbid")

    machine_id: str | None
    machine_model: str | None = Field(description="Canonical equipment family: MR-JE, FX5 or i4L.")
    machine_model_specific: str | None = Field(
        default=None, description="Specific manufacturer model or order code, e.g. MR-JE-70A; not a family or asset ID."
    )
    component_id: str | None
    component_model: str | None
    line_id: str | None
    station: str | None
    product_model: str | None
    configuration: str | None
    serial_range: str | None
    condition: str | None
    failure_mode: str | None


class IndustrialCase(BaseModel):
    """One incident whose retrieval claim is rendered from a Case Card."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    scope: IndustrialScope
    case_card: CaseCard
    valid_from: date | None = None
    valid_to: date | None = None

    def to_item(self) -> KnowledgeItem:
        return KnowledgeItem(
            subject=self.subject,
            claim=render_case_claim(self.case_card),
            context={**scope_context(self.scope), "case_card": self.case_card.model_dump()},
            type=KnowledgeType.CASE,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
        )


class IndustrialLesson(BaseModel):
    """One reusable requirement, practice, heuristic, or ordered procedure."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    scope: IndustrialScope
    claim: str = Field(min_length=1)
    steps: list[str] | None = None
    mandatory: bool = False
    # Expert turns stating or endorsing the lesson.
    source_turns: list[str] = Field(min_length=1)
    case_refs: list[int] = Field(default_factory=list)
    valid_from: date | None = None
    valid_to: date | None = None

    @field_validator("source_turns", mode="before")
    @classmethod
    def _canonical_turns(cls, value: Any) -> Any:
        return [canonical_turn_id(item) for item in value] if isinstance(value, list) else value

    @property
    def knowledge_type(self) -> KnowledgeType:
        if self.mandatory:
            return KnowledgeType.RULE
        if self.steps:
            return KnowledgeType.PROCEDURE
        return KnowledgeType.HEURISTIC

    def to_item(self, case_positions: list[int]) -> KnowledgeItem:
        context = {**scope_context(self.scope), "source_turns": list(self.source_turns)}
        claim = self.claim.strip()
        if self.steps:
            context["steps"] = list(self.steps)
            claim += "\n" + "\n".join(
                f"{number}. {step}" for number, step in enumerate(self.steps, 1)
            )
        return KnowledgeItem(
            subject=self.subject,
            claim=claim,
            context=context,
            type=self.knowledge_type,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            supported_by=case_positions,
        )


class IndustrialQaExtraction(StructuredExtraction):
    """Everything extracted from one benchmark conversation.

    Lessons come first: the model writes the reusable ideas, then cases that
    keep only what is specific to the incident.
    """

    lessons: list[IndustrialLesson] = Field(default_factory=list)
    cases: list[IndustrialCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_case_refs(self) -> "IndustrialQaExtraction":
        for index, lesson in enumerate(self.lessons):
            if any(not 0 <= ref < len(self.cases) for ref in lesson.case_refs):
                raise ValueError(f"lessons.{index}.case_refs points to no case")
        return self

    def to_result(self) -> KnowledgeExtractionResult:
        return KnowledgeExtractionResult(knowledge_items=[
            *(case.to_item() for case in self.cases),
            *(lesson.to_item(sorted(set(lesson.case_refs))) for lesson in self.lessons),
        ])


def render_case_claim(card: CaseCard) -> str:
    """Build self-contained retrieval text from the structured card."""

    facets = [
        ("", card.context),
        ("Observation: ", card.observation),
        ("Reasoning: ", card.reasoning),
        ("Action: ", card.action),
        ("Outcome: ", card.outcome or "not stated"),
    ]
    return "\n".join(label + value.strip() for label, value in facets if value.strip())
