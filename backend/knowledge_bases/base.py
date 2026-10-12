"""Definition of one knowledge base: its dataset, prompts, and database."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.src.agents.rag.chunking import ChunkingConfig
from backend.src.services.knowledge_extraction import (
    RAW_CONVERSATION_PLACEHOLDER,
    StructuredExtraction,
)
from backend.src.services.ontology import Ontology

if TYPE_CHECKING:
    from backend.src.storage import KnowledgeRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_PROMPT_FILE = "extract.md"
AGENT_PROMPT_FILE = "agent.md"
DOCUMENTS_AGENT_PROMPT_FILE = "agent_documents.md"
EMBEDDING_QUERY_PROMPT_FILE = "embedding_query.md"
ONTOLOGY_FILE = "ontology.yaml"
DATABASE_PREFIX = "kb_"
# The database name kb_<name> must stay a plain PostgreSQL identifier.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,59}$")
# Context keys become tool arguments and SQL expressions: keep them plain.
_CONTEXT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _check_context_key(key: str) -> str:
    if not _CONTEXT_KEY_RE.match(key):
        raise ValueError(
            f"Invalid context key {key!r}: use lowercase letters, digits, and "
            "underscores, starting with a letter"
        )
    return key


class Retrieval(str, Enum):
    """What the agent searches."""

    # Knowledge records an LLM extracted from each source, linked to it as evidence.
    KNOWLEDGE = "knowledge"
    # Chunks of the sources themselves (traditional RAG); no extraction step.
    DOCUMENTS = "documents"


class AnswerPipeline(str, Enum):
    AGENT = "agent"
    INDUSTRIAL_QA = "industrial_qa"


@dataclass(frozen=True)
class DatasetSpec:
    """Default ingestion input: a .csv (one transcript per row), .txt, or .md file."""

    path: Path
    transcript_column: str = "raw_conversation"
    stated_by_column: str = "stated_by"
    # Column identifying a row; ingest skips rows already processed in the KB.
    # Document retrieval uses it as the document id (cited as [D:<id>]).
    id_column: str | None = None
    # Document retrieval: column prepended to the text as the document title.
    title_column: str | None = None
    # Columns copied into the context of every record extracted from the row,
    # so search can filter on them (e.g. the source system of a document).
    context_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for column in self.context_columns:
            _check_context_key(column)


@dataclass(frozen=True)
class RetrievalQuestions:
    """Questions labelled with the dataset ids of the sources that answer them."""

    path: Path
    id_column: str = "question_id"
    question_column: str = "question"
    gold_column: str = "gold_doc_ids"
    # Separates several gold ids in one cell, e.g. "1;2;13".
    gold_separator: str = ";"


@dataclass(frozen=True)
class ContextFilter:
    """A search_knowledge argument that filters records on one context key."""

    key: str
    # Shown to the agent as the argument's description.
    description: str
    # Also return records whose context lacks the key, e.g. knowledge that
    # names no machine applies to every machine. False requires a match.
    include_unscoped: bool = True

    def __post_init__(self) -> None:
        _check_context_key(self.key)
        if not self.description.strip():
            raise ValueError(f"Context filter {self.key!r} needs a description")


@dataclass(frozen=True)
class KnowledgeBaseSpec:
    """Everything that differs between knowledge bases.

    Infrastructure (database server, LLM, embedding model) stays in the
    environment; each knowledge base gets its own database on that server.
    """

    name: str
    extract_prompt: str
    agent_prompt: str
    description: str = ""
    dataset: DatasetSpec | None = None
    # PostgreSQL full-text search configuration for lexical retrieval.
    text_search_config: str = "english"
    # None falls back to KNOWLEDGE_MIN_VECTOR_SIMILARITY.
    min_vector_similarity: float | None = None
    # What the agent searches; DOCUMENTS needs no extraction prompt.
    retrieval: Retrieval = Retrieval.KNOWLEDGE
    answer_pipeline: AnswerPipeline = AnswerPipeline.AGENT
    # Document retrieval: how sources are split before embedding.
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # Gold-labelled questions for `kb evaluate` (document retrieval).
    retrieval_questions: RetrievalQuestions | None = None
    # Extra search_knowledge arguments; each filters on one context key.
    # Knowledge retrieval only: document chunks carry no extracted context.
    context_filters: tuple[ContextFilter, ...] = ()
    # Optional KB policy for combining filters (e.g. exact asset OR generic family).
    knowledge_filter_builder: Callable[..., dict[str, Any] | None] | None = None
    # Context keys that lead the embedded text of a record (the rest follow
    # alphabetically) because they most change what a claim means.
    embedding_context_keys: tuple[str, ...] = ()
    # Query-side task instruction for instruction-tuned embedding models.
    # Empty means queries are embedded as-is; documents never get this prefix.
    embedding_query_instruction: str = ""
    # Human-readable extraction contract version; prompt hash tracks exact edits.
    extraction_version: str = "1"
    # Typed schema the extraction output must match; None accepts plain
    # knowledge items with any context dictionary.
    extraction_output_model: type[StructuredExtraction] | None = None
    # Legacy compatibility for knowledge bases that still declare review policy.
    # Ingestion now stores extracted records as drafts regardless of this flag.
    review_required: bool = False
    # Known values of context keys; extracted context is normalized to it.
    ontology: Ontology | None = None
    # Rewrites the scope of the records of one transcript before the ontology
    # (see KnowledgeIngestionService), e.g. with an equipment catalog.
    record_normalizer: Callable[[str, list[KnowledgeRecord]], None] | None = None

    def __post_init__(self) -> None:
        keys = [item.key for item in self.context_filters]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate context filters in {self.name!r}")
        for key in self.embedding_context_keys:
            _check_context_key(key)
        if self.ontology is not None:
            for key in self.ontology.keys:
                _check_context_key(key)
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid knowledge base name {self.name!r}: use lowercase "
                "letters, digits, and underscores, starting with a letter"
            )
        object.__setattr__(self, "retrieval", Retrieval(self.retrieval))
        object.__setattr__(self, "answer_pipeline", AnswerPipeline(self.answer_pipeline))
        if self.answer_pipeline is AnswerPipeline.INDUSTRIAL_QA and self.retrieval is not Retrieval.KNOWLEDGE:
            raise ValueError("Industrial QA pipeline requires a knowledge store")
        if (
            self.retrieval is Retrieval.KNOWLEDGE
            and RAW_CONVERSATION_PLACEHOLDER not in self.extract_prompt
        ):
            raise ValueError(
                f"Extraction prompt of {self.name!r} is missing "
                f"{RAW_CONVERSATION_PLACEHOLDER}"
            )
        if not self.agent_prompt.strip():
            raise ValueError(f"Agent prompt of {self.name!r} is empty")

    @property
    def database(self) -> str:
        return f"{DATABASE_PREFIX}{self.name}"

    @property
    def document_table(self) -> str:
        """Chunk table of document retrieval, one per chunking setup."""

        return f"chunks_{self.chunking.label}"

    @property
    def trace_context_keys(self) -> tuple[str, ...]:
        """Context keys whose values may appear in public agent traces."""

        keys = [item.key for item in self.context_filters]
        return tuple(dict.fromkeys([*keys, *self.embedding_context_keys]))

    @classmethod
    def from_directory(cls, directory: Path, **fields) -> "KnowledgeBaseSpec":
        """Load prompts and the optional ontology.yaml and embedding_query.md.

        The agent prompt is agent.md for knowledge retrieval and
        agent_documents.md for document retrieval; extract.md is optional for
        document retrieval. An explicit embedding_query_instruction overrides
        the file, including an empty string to disable the query prefix for
        this knowledge base.
        """

        if "embedding_query_instruction" not in fields:
            query_prompt = directory / EMBEDDING_QUERY_PROMPT_FILE
            fields["embedding_query_instruction"] = (
                query_prompt.read_text(encoding="utf-8").strip()
                if query_prompt.is_file()
                else ""
            )

        if "ontology" not in fields and (directory / ONTOLOGY_FILE).is_file():
            fields["ontology"] = Ontology.from_file(directory / ONTOLOGY_FILE)

        documents = Retrieval(fields.get("retrieval", Retrieval.KNOWLEDGE)) is Retrieval.DOCUMENTS
        extract_prompt = directory / EXTRACT_PROMPT_FILE
        agent_prompt = directory / (DOCUMENTS_AGENT_PROMPT_FILE if documents else AGENT_PROMPT_FILE)
        return cls(
            name=fields.pop("name", directory.name),
            extract_prompt=(
                extract_prompt.read_text(encoding="utf-8").strip()
                if extract_prompt.is_file() or not documents
                else ""
            ),
            agent_prompt=agent_prompt.read_text(encoding="utf-8").strip(),
            **fields,
        )
