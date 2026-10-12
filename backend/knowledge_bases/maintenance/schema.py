"""Typed extraction schema for maintenance knowledge: cases and lessons.

Sent to the LLM as the response JSON schema and used to validate its output.
A case is one incident kept whole as a Case Card; a lesson is one reusable,
generalized idea drawn from cases. Both are stored as ordinary knowledge rows.
Text that only restates other fields (a case's claim, a lesson's numbered
steps) is rendered here, never written by the LLM, so it cannot drift.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.src.services.knowledge_extraction import (
    KnowledgeExtractionResult,
    KnowledgeItem,
    KnowledgeType,
    StructuredExtraction,
    scope_context,
)


class CaseEvidence(BaseModel):
    """A verbatim transcript excerpt supporting the case."""

    model_config = ConfigDict(extra="forbid")

    # Only when the transcript provides [T01]-style turn IDs; never invented.
    turn_id: str | None = None
    quote: str = Field(min_length=1)


class CaseCard(BaseModel):
    """Context, observation, reasoning, action and outcome of one incident."""

    model_config = ConfigDict(extra="forbid")

    context: str
    observation: str
    reasoning: str
    action: str
    # Null when the transcript states no outcome or pending status.
    outcome: str | None = None
    evidence: list[CaseEvidence] = Field(min_length=1)


class MaintenanceScope(BaseModel):
    """Where knowledge applies. Closed, so filters and matching stay reliable.

    Leave a key null when the source does not establish it; null keys are not
    stored, and a record without a key applies to every value of it.
    """

    model_config = ConfigDict(extra="forbid")

    machine_id: str | None = None
    machine_model: str | None = None
    component_id: str | None = None
    component_model: str | None = None
    line_id: str | None = None
    station: str | None = None
    product_model: str | None = None
    configuration: str | None = None
    serial_range: str | None = None
    # A short tag such as "cold start" or "ambient above 35 C", not prose.
    condition: str | None = None


class MaintenanceCase(BaseModel):
    """One incident; its claim is rendered from the Case Card."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    valid_from: date | None = None
    valid_to: date | None = None
    context: MaintenanceScope = Field(default_factory=MaintenanceScope)
    case_card: CaseCard

    def to_item(self) -> KnowledgeItem:
        return KnowledgeItem(
            subject=self.subject,
            claim=render_case_claim(self.case_card),
            context={**scope_context(self.context), "case_card": self.case_card.model_dump()},
            type=KnowledgeType.CASE,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
        )


class MaintenanceLesson(BaseModel):
    """One generalized idea: a requirement, a practice, or an ordered plan."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    # Ordered steps when the lesson is a reusable plan.
    steps: list[str] | None = None
    # True only when the source states it as a requirement or prohibition.
    mandatory: bool = False
    valid_from: date | None = None
    valid_to: date | None = None
    context: MaintenanceScope = Field(default_factory=MaintenanceScope)
    # Positions in `cases` of the incidents this lesson was drawn from.
    case_refs: list[int] = Field(default_factory=list)

    @property
    def knowledge_type(self) -> KnowledgeType:
        # A requirement matters more than its shape: the steps stay in the claim.
        if self.mandatory:
            return KnowledgeType.RULE
        if self.steps:
            return KnowledgeType.PROCEDURE
        return KnowledgeType.HEURISTIC

    def to_item(self, case_positions: list[int]) -> KnowledgeItem:
        context = scope_context(self.context)
        claim = self.claim.strip()
        if self.steps:
            context["steps"] = list(self.steps)
            claim += "\n" + "\n".join(f"{n}. {step}" for n, step in enumerate(self.steps, 1))
        return KnowledgeItem(
            subject=self.subject,
            claim=claim,
            context=context,
            type=self.knowledge_type,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            supported_by=case_positions,
        )


class MaintenanceExtraction(StructuredExtraction):
    """Everything extracted from one maintenance conversation."""

    cases: list[MaintenanceCase] = Field(default_factory=list)
    lessons: list[MaintenanceLesson] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_case_refs(self) -> "MaintenanceExtraction":
        for index, lesson in enumerate(self.lessons):
            if any(not 0 <= ref < len(self.cases) for ref in lesson.case_refs):
                raise ValueError(f"lessons.{index}.case_refs points to no case")
        return self

    def to_result(self) -> KnowledgeExtractionResult:
        # Cases come first, so a case's position in `cases` is its item position.
        return KnowledgeExtractionResult(knowledge_items=[
            *(case.to_item() for case in self.cases),
            *(lesson.to_item(sorted(set(lesson.case_refs))) for lesson in self.lessons),
        ])


def render_case_claim(card: CaseCard) -> str:
    """Self-contained retrieval text of a case, built from its card."""

    facets = [
        ("", card.context),
        ("Observation: ", card.observation),
        ("Reasoning: ", card.reasoning),
        ("Action: ", card.action),
        ("Outcome: ", card.outcome or "not stated"),
    ]
    return "\n".join(label + value.strip() for label, value in facets if value.strip())
