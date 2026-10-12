"""Extract structured, reusable knowledge from raw conversations."""

from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.src.services import llm
from backend.src.services.llm import LLMResponseError
from backend.src.services.ontology import Ontology


RAW_CONVERSATION_PLACEHOLDER = "{RAW_CONVERSATION}"


class KnowledgeType(str, Enum):
    """Knowledge categories allowed by the extraction prompt."""

    FACT = "fact"
    RULE = "rule"
    CASE = "case"
    PROCEDURE = "procedure"
    HEURISTIC = "heuristic"


class KnowledgeItem(BaseModel):
    """One reusable knowledge statement extracted from a conversation."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    type: KnowledgeType
    valid_from: date | None = None
    valid_to: date | None = None
    # Positions of other items of the same result that this item generalizes,
    # e.g. the cases a lesson was drawn from; ingestion stores their ids.
    supported_by: list[int] = Field(default_factory=list)


class KnowledgeExtractionResult(BaseModel):
    """Validated structured output returned by the service."""

    model_config = ConfigDict(extra="forbid")

    knowledge_items: list[KnowledgeItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_support_links(self) -> "KnowledgeExtractionResult":
        count = len(self.knowledge_items)
        for index, item in enumerate(self.knowledge_items):
            if any(ref == index or not 0 <= ref < count for ref in item.supported_by):
                raise ValueError(f"knowledge_items.{index}.supported_by points to no other item")
        return self


class StructuredExtraction(BaseModel):
    """Base of a knowledge base's typed extraction output.

    A subclass describes exactly what the LLM must return, so the LLM gets a
    precise JSON schema and its output is validated against it. to_result()
    turns it into plain knowledge items, so storage is the same for every
    knowledge base. Field order is the order the JSON schema lists keys in.
    """

    model_config = ConfigDict(extra="forbid")

    def to_result(self) -> KnowledgeExtractionResult:
        raise NotImplementedError


def scope_context(scope: BaseModel | None) -> dict[str, Any]:
    """Context of a typed scope model without its null keys.

    An absent scope key must stay absent: context filters treat a missing key
    as "applies to every machine", unlike an explicit null.
    """

    if scope is None:
        return {}
    return {key: value for key, value in scope.model_dump().items() if value is not None}


class SupportsInvoke(Protocol):
    """Small client interface used to keep the service easy to unit test."""

    def invoke(self, input: str) -> Any: ...


class KnowledgeExtractionError(LLMResponseError):
    """Raised when the LLM response cannot be parsed or validated."""


class KnowledgeExtractionService:
    """LLM-backed service for turning a transcript into knowledge items.

    `prompt_template` is the knowledge base's extraction prompt; it must contain
    {RAW_CONVERSATION}, which is replaced by the transcript. `output_model` is
    an optional typed output schema (see StructuredExtraction); `ontology`
    constrains its closed context keys and is recorded in the provenance.
    """

    def __init__(
        self,
        client: SupportsInvoke,
        prompt_template: str,
        *,
        output_model: type[StructuredExtraction] | None = None,
        ontology: Ontology | None = None,
    ):
        if RAW_CONVERSATION_PLACEHOLDER not in prompt_template:
            raise ValueError(
                f"Extraction prompt is missing its {RAW_CONVERSATION_PLACEHOLDER} placeholder"
            )
        self._client = client
        self._prompt_template = prompt_template
        self._output_model = output_model
        self._ontology = ontology

    @staticmethod
    def output_schema(
        output_model: type[StructuredExtraction] | None = None,
        ontology: Ontology | None = None,
    ) -> dict[str, Any]:
        """JSON schema of the extraction output, typed when output_model is set."""

        schema = (output_model or KnowledgeExtractionResult).model_json_schema()
        return ontology.constrain_schema(schema) if ontology is not None else schema

    @property
    def provenance(self) -> dict[str, Any]:
        """Non-secret settings identifying the extraction that produced a row."""
        details: dict[str, Any] = {
            "prompt_sha256": hashlib.sha256(self._prompt_template.encode("utf-8")).hexdigest(),
            **llm.client_provenance(self._client),
        }
        if self._output_model is not None:
            details["output_model"] = self._output_model.__name__
        if self._ontology is not None:
            details["ontology_sha256"] = self._ontology.sha256
        return details

    @classmethod
    def from_env(
        cls,
        prompt_template: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        json_schema: bool = False,
        output_model: type[StructuredExtraction] | None = None,
        ontology: Ontology | None = None,
    ) -> "KnowledgeExtractionService":
        """Build the OpenAI-compatible client configured by environment variables.

        EXTRACTION_* settings override the shared OPENROUTER_* ones, so bulk
        extraction can use another endpoint than the agent, e.g. a ModelArk
        batch inference endpoint or EXTRACTION_SERVICE_TIER=flex.

        Output is always validated against `output_model` when it is set;
        json_schema additionally sends that schema as the response format.
        """

        response_format = None
        if json_schema:
            response_format = llm.json_schema_format(
                "knowledge_extraction", cls.output_schema(output_model, ontology)
            )
        elif json_mode:
            response_format = {"type": "json_object"}
        client = llm.chat_client_from_env(
            "EXTRACTION",
            max_tokens=max_tokens,
            response_format=response_format,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        return cls(client, prompt_template, output_model=output_model, ontology=ontology)

    def extract(self, raw_conversation: str) -> KnowledgeExtractionResult:
        """Extract and validate knowledge items from one raw conversation."""

        prompt = self._checked_prompt(raw_conversation)
        return self._parse(llm.invoke(self._client, prompt, error=KnowledgeExtractionError))

    async def aextract(self, raw_conversation: str) -> KnowledgeExtractionResult:
        """Async extract, so many conversations can be in flight at once.

        Clients without `ainvoke` run their blocking `invoke` in a thread.
        """

        prompt = self._checked_prompt(raw_conversation)
        return self._parse(
            await llm.ainvoke(self._client, prompt, error=KnowledgeExtractionError)
        )

    def _checked_prompt(self, raw_conversation: str) -> str:
        if not raw_conversation or not raw_conversation.strip():
            raise ValueError("raw_conversation must not be empty")
        return self.build_prompt(raw_conversation)

    def _parse(self, response: Any) -> KnowledgeExtractionResult:
        payload = llm.response_json(
            response, task="extraction", env_prefix="EXTRACTION", error=KnowledgeExtractionError
        )
        try:
            if self._output_model is None:
                return KnowledgeExtractionResult.model_validate(payload)
            return self._output_model.model_validate(payload).to_result()
        except ValidationError as exc:
            # Keep diagnostics useful without logging extracted document contents.
            raise KnowledgeExtractionError(
                "The LLM extraction JSON failed schema validation: "
                + llm.validation_summary(exc)
            ) from exc

    def build_prompt(self, raw_conversation: str) -> str:
        """Insert a transcript into the knowledge base's extraction prompt."""

        return self._prompt_template.replace(
            RAW_CONVERSATION_PLACEHOLDER, raw_conversation.strip(), 1
        )
