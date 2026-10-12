"""PostgreSQL + pgvector knowledge base backed by langchain-postgres."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage.models import (
    ConversationStatus,
    KnowledgeLinkUpdate,
    KnowledgeRecord,
    KnowledgeSearchHit,
    KnowledgeStatus,
    RawConversation,
)

if TYPE_CHECKING:
    from backend.knowledge_bases.base import KnowledgeBaseSpec


KNOWLEDGE_TABLE = "knowledge_items"
RAW_CONVERSATION_TABLE = "raw_conversations"
ID_COLUMN = "id"
CONTENT_COLUMN = "claim"
EMBEDDING_COLUMN = "embedding"
METADATA_COLUMNS = [
    "subject",
    "type",
    "context",
    "search_text",
    "evidence_id",
    "stated_by",
    "created_at",
    "valid_from",
    "valid_to",
    "conflict_with",
    "status",
    "supported_by",
    "contradicted_by",
    "superseded_by",
    "revision",
]
# Columns of a stored record, as read back by search.
RECORD_COLUMNS = (
    "id", "subject", "claim", "context", "type", "evidence_id", "stated_by", "created_at",
    "valid_from", "valid_to", "conflict_with", "status", "supported_by", "contradicted_by",
    "superseded_by", "revision",
)
# pgvector cannot build an HNSW index over wider `vector` columns.
HNSW_MAX_DIM = 2000
DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 20
MIN_SEARCH_CANDIDATES = 20
MAX_SEARCH_CANDIDATES = 100
SEARCH_CANDIDATE_MULTIPLIER = 4
RRF_RANK_CONSTANT = 60
# Cosine similarity at or above which a vector hit counts as confident. The
# threshold only labels hits; it never removes the nearest candidates.
DEFAULT_MIN_VECTOR_SIMILARITY = 0.3
# A lexical hit is confident when it shares at least
# max(min(LEXICAL_CONFIDENT_MIN_TERMS, n), ceil(LEXICAL_CONFIDENT_TERM_FRACTION * n))
# of the n distinct query lexemes.
LEXICAL_CONFIDENT_MIN_TERMS = 2
LEXICAL_CONFIDENT_TERM_FRACTION = 0.3
MAX_LEXICAL_TERMS = 32
# Stemming and stop words for full-text search; changing it rebuilds search_vector.
DEFAULT_TEXT_SEARCH_CONFIG = "english"
MIN_VECTOR_SIMILARITY_ENV = "KNOWLEDGE_MIN_VECTOR_SIMILARITY"
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Keeps identifiers such as AX-220, S14.b, or v1.2 in one token.
_LEXICAL_TOKEN_RE = re.compile(r"\w+(?:[-./]\w+)*")

_KNOWLEDGE_TYPES_SQL = ", ".join(f"'{kind.value}'" for kind in KnowledgeType)
_STATUSES_SQL = ", ".join(f"'{status.value}'" for status in ConversationStatus)
_KNOWLEDGE_STATUSES_SQL = ", ".join(f"'{status.value}'" for status in KnowledgeStatus)
_RECORD_SELECT_SQL = ", ".join(f"k.{column}" for column in RECORD_COLUMNS)
# Deprecated and replaced records stay readable by id but are never searched.
_SEARCHABLE_SQL = (
    f"k.status <> '{KnowledgeStatus.DEPRECATED.value}' AND k.superseded_by IS NULL"
)


class StorageError(RuntimeError):
    """Raised when the knowledge base cannot complete a database operation."""


def _validate_min_vector_similarity(value: float) -> float:
    """Validate a cosine-similarity threshold used by vector retrieval."""

    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_vector_similarity must be between 0 and 1")
    return threshold


def min_vector_similarity_from_env() -> float:
    """Read the retrieval threshold, falling back to the production default."""

    raw = os.getenv(MIN_VECTOR_SIMILARITY_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MIN_VECTOR_SIMILARITY
    try:
        return _validate_min_vector_similarity(float(raw))
    except ValueError as exc:
        raise ValueError(
            f"{MIN_VECTOR_SIMILARITY_ENV} must be a number between 0 and 1"
        ) from exc


def lexical_or_query(query: str) -> str:
    """Rewrite free text as a websearch OR query so any shared term can match."""

    seen: set[str] = set()
    tokens: list[str] = []
    for token in _LEXICAL_TOKEN_RE.findall(query):
        folded = token.casefold()
        # A bare "or" would be read as an operator; it is a stop word anyway.
        if folded == "or" or folded in seen:
            continue
        seen.add(folded)
        tokens.append(token)
        if len(tokens) == MAX_LEXICAL_TERMS:
            break
    return " or ".join(tokens)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_knowledge_filter(
    *,
    subject: str | None = None,
    knowledge_type: KnowledgeType | str | None = None,
    context: Mapping[str, str | None] | None = None,
    exact_context_keys: Collection[str] = (),
    valid_on: date | None = None,
) -> dict[str, Any] | None:
    """Translate retrieval constraints into a PGVectorStore metadata filter.

    Each `context` value must equal the record's context value for that key.
    Records whose context lacks the key still match (knowledge that names no
    machine applies to every machine) unless the key is in exact_context_keys.
    """

    conditions: list[dict[str, Any]] = []
    if subject and subject.strip():
        conditions.append({"subject": {"$ilike": f"%{_escape_like(subject.strip())}%"}})
    if knowledge_type:
        conditions.append({"type": {"$eq": KnowledgeType(knowledge_type).value}})
    for key, value in (context or {}).items():
        if value is None or not value.strip():
            continue
        field = f"context.{key}"
        condition = {field: {"$eq": value.strip()}}
        if key not in exact_context_keys:
            condition = {"$or": [condition, {field: {"$exists": False}}]}
        conditions.append(condition)
    if valid_on is not None:
        conditions.append(
            {"$or": [{"valid_from": {"$exists": False}}, {"valid_from": {"$lte": valid_on}}]}
        )
        conditions.append(
            {"$or": [{"valid_to": {"$exists": False}}, {"valid_to": {"$gte": valid_on}}]}
        )
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


_FILTER_FIELDS = {
    "subject": "k.subject",
    "type": "k.type",
    "valid_from": "k.valid_from",
    "valid_to": "k.valid_to",
}
_FILTER_OPERATORS = {"$eq", "$ilike", "$exists", "$lte", "$gte"}


def _context_key(key: str) -> str:
    if not _IDENTIFIER_RE.match(key):
        raise ValueError(f"Invalid context key: {key!r}")
    return key


def _filter_column(field: str) -> str | None:
    if field in _FILTER_FIELDS:
        return _FILTER_FIELDS[field]
    if field.startswith("context."):
        # Validated as a plain identifier, so it is safe to inline.
        return f"k.context ->> '{_context_key(field.removeprefix('context.'))}'"
    return None


def _compile_knowledge_filter(
    filters: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Compile the filter grammar emitted by build_knowledge_filter to safe SQL."""

    if filters is None:
        return "TRUE", {}

    parameters: dict[str, Any] = {}

    def bind(value: Any) -> str:
        name = f"filter_{len(parameters)}"
        parameters[name] = value
        return f":{name}"

    def compile_node(node: Any) -> str:
        if not isinstance(node, dict) or len(node) != 1:
            raise ValueError("Knowledge filter nodes must contain exactly one key")
        key, value = next(iter(node.items()))
        if key in {"$and", "$or"}:
            if not isinstance(value, list) or not value:
                raise ValueError(f"{key} requires a non-empty list")
            joiner = " AND " if key == "$and" else " OR "
            return "(" + joiner.join(compile_node(item) for item in value) + ")"

        column = _filter_column(key)
        if column is None:
            raise ValueError(f"Unsupported knowledge filter field: {key}")
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError(f"Filter for {key} must contain exactly one operator")
        operator, operand = next(iter(value.items()))
        if operator not in _FILTER_OPERATORS:
            raise ValueError(f"Unsupported knowledge filter operator: {operator}")
        if operator == "$exists":
            if not isinstance(operand, bool):
                raise ValueError("$exists requires a boolean")
            return f"{column} IS {'NOT ' if operand else ''}NULL"

        placeholder = bind(operand)
        if operator == "$eq":
            return f"{column} = {placeholder}"
        if operator == "$ilike":
            return f"{column} ILIKE {placeholder} ESCAPE '\\'"
        if operator == "$lte":
            return f"{column} <= {placeholder}"
        return f"{column} >= {placeholder}"

    return compile_node(filters), parameters


class KnowledgeStore:
    """Knowledge records (with embeddings) and the raw conversations they cite."""

    def __init__(
        self,
        engine: AsyncEngine,
        embeddings: Embeddings,
        vector_size: int,
        *,
        schema: str = "public",
        min_vector_similarity: float = DEFAULT_MIN_VECTOR_SIMILARITY,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
        embedding_context_keys: Sequence[str] = (),
        indexed_context_keys: Sequence[str] = (),
    ) -> None:
        if not _IDENTIFIER_RE.match(schema):
            raise ValueError(f"Invalid schema name: {schema!r}")
        if not _IDENTIFIER_RE.match(text_search_config):
            raise ValueError(f"Invalid text search config: {text_search_config!r}")
        self._engine = engine
        self._schema = schema
        self._text_search_config = text_search_config
        self._embedding_context_keys = tuple(embedding_context_keys)
        # Context keys that search filters on get an expression index.
        self._indexed_context_keys = tuple(
            _context_key(key) for key in indexed_context_keys
        )
        self._pg_engine = PGEngine.from_engine(engine)
        self._embeddings = embeddings
        self._vector_size = vector_size
        self._min_vector_similarity = _validate_min_vector_similarity(
            min_vector_similarity
        )
        self._vector_store: PGVectorStore | None = None
        self._vector_store_lock = asyncio.Lock()

    @classmethod
    def from_url(
        cls,
        database_url: str,
        embeddings: Embeddings,
        vector_size: int,
        *,
        schema: str = "public",
        min_vector_similarity: float = DEFAULT_MIN_VECTOR_SIMILARITY,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
        embedding_context_keys: Sequence[str] = (),
        indexed_context_keys: Sequence[str] = (),
        **engine_kwargs: Any,
    ) -> "KnowledgeStore":
        """Create a store over a `postgresql+psycopg://` connection URL."""

        engine_kwargs.setdefault("pool_pre_ping", True)
        return cls(
            create_async_engine(database_url, **engine_kwargs),
            embeddings,
            vector_size,
            schema=schema,
            min_vector_similarity=min_vector_similarity,
            text_search_config=text_search_config,
            embedding_context_keys=embedding_context_keys,
            indexed_context_keys=indexed_context_keys,
        )

    @classmethod
    async def for_kb(
        cls,
        spec: KnowledgeBaseSpec,
        *,
        embeddings: Embeddings | None = None,
        init_schema: bool = True,
    ) -> "KnowledgeStore":
        """Open a knowledge base's own database, creating it and its schema if missing.

        The server comes from DATABASE_URL, the embedding model from EMBEDDING_*.
        Query instructions come from the KB spec. Explicitly supplied embedding
        clients are used as-is, including their own query formatting.
        Pass init_schema=False to re-embed after EMBEDDING_DIM changed, which
        init_schema would otherwise reject.
        """

        from backend.src.storage.databases import ensure_database, server_url_from_env
        from backend.src.storage.embeddings import (
            build_embeddings_from_env,
            embedding_dim_from_env,
        )

        store = cls.from_url(
            await ensure_database(server_url_from_env(), spec.database),
            embeddings if embeddings is not None else build_embeddings_from_env(
                query_instruction=spec.embedding_query_instruction
            ),
            embedding_dim_from_env(),
            min_vector_similarity=(
                min_vector_similarity_from_env()
                if spec.min_vector_similarity is None
                else spec.min_vector_similarity
            ),
            text_search_config=spec.text_search_config,
            embedding_context_keys=spec.embedding_context_keys,
            indexed_context_keys=[item.key for item in spec.context_filters],
        )
        try:
            if init_schema:
                await store.init_schema()
        except BaseException:
            await store.aclose()
            raise
        return store

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def min_vector_similarity(self) -> float:
        return self._min_vector_similarity

    def _table(self, name: str) -> str:
        return f'"{self._schema}"."{name}"'

    async def aclose(self) -> None:
        await self._engine.dispose()

    @asynccontextmanager
    async def _errors(self, action: str) -> AsyncIterator[None]:
        try:
            yield
        except SQLAlchemyError as exc:
            raise StorageError(f"Knowledge store failed to {action}: {exc}") from exc

    # ------------------------------------------------------------------ schema

    async def init_schema(self) -> None:
        """Create the extension, tables, constraints, and indexes if missing."""

        dim = self._vector_size
        raw_table = self._table(RAW_CONVERSATION_TABLE)
        knowledge_table = self._table(KNOWLEDGE_TABLE)
        text_search_config = self._text_search_config
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS {raw_table} (
                id UUID PRIMARY KEY,
                transcript TEXT NOT NULL,
                stated_by TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ({_STATUSES_SQL})),
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            # Column names must match the PGVectorStore mapping in _vector().
            f"""
            CREATE TABLE IF NOT EXISTS {knowledge_table} (
                {ID_COLUMN} UUID PRIMARY KEY,
                {CONTENT_COLUMN} TEXT NOT NULL,
                {EMBEDDING_COLUMN} vector({dim}) NOT NULL,
                subject TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ({_KNOWLEDGE_TYPES_SQL})),
                context JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                search_text TEXT NOT NULL,
                search_vector TSVECTOR GENERATED ALWAYS AS
                    (to_tsvector('{text_search_config}'::regconfig, search_text)) STORED,
                evidence_id UUID REFERENCES {raw_table}(id),
                stated_by TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                valid_from DATE,
                valid_to DATE,
                conflict_with UUID[] NOT NULL DEFAULT '{{}}',
                CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to)
            )
            """,
            # Review and consolidation columns, added in place to older tables.
            # Nothing is approved without an expert, existing rows included.
            f"""
            ALTER TABLE {knowledge_table}
                ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ({_KNOWLEDGE_STATUSES_SQL})),
                ADD COLUMN IF NOT EXISTS supported_by UUID[] NOT NULL DEFAULT '{{}}',
                ADD COLUMN IF NOT EXISTS contradicted_by UUID[] NOT NULL DEFAULT '{{}}',
                ADD COLUMN IF NOT EXISTS superseded_by UUID REFERENCES {knowledge_table}(id),
                ADD COLUMN IF NOT EXISTS revision JSONB NOT NULL DEFAULT '{{}}'::jsonb
            """,
            f"ALTER TABLE {knowledge_table} ALTER COLUMN status SET DEFAULT 'draft'",
            f"ALTER TABLE {knowledge_table} ADD COLUMN IF NOT EXISTS search_text TEXT",
            f"""
            UPDATE {knowledge_table}
            SET search_text = subject || ': ' || claim ||
                CASE WHEN context = '{{}}'::jsonb THEN ''
                     ELSE E'\nContext: ' || context::text END
            WHERE search_text IS NULL
            """,
            f"ALTER TABLE {knowledge_table} ALTER COLUMN search_text SET NOT NULL",
            # A search_vector built with another text search config is dropped
            # (with its GIN index) so it is regenerated below.
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_attribute AS a
                    JOIN pg_attrdef AS d
                      ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                    WHERE a.attrelid = '{knowledge_table}'::regclass
                      AND a.attname = 'search_vector'
                      AND NOT a.attisdropped
                      AND pg_get_expr(d.adbin, d.adrelid)
                          NOT LIKE '%''{text_search_config}''::regconfig%'
                ) THEN
                    ALTER TABLE {knowledge_table} DROP COLUMN search_vector;
                END IF;
            END
            $$
            """,
            f"""
            ALTER TABLE {knowledge_table} ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
            GENERATED ALWAYS AS
                (to_tsvector('{text_search_config}'::regconfig, search_text)) STORED
            """,
            f"CREATE INDEX IF NOT EXISTS {KNOWLEDGE_TABLE}_evidence_id_idx "
            f"ON {knowledge_table} (evidence_id)",
            f"CREATE INDEX IF NOT EXISTS {KNOWLEDGE_TABLE}_search_vector_gin_idx "
            f"ON {knowledge_table} USING gin (search_vector)",
            *(
                f"CREATE INDEX IF NOT EXISTS {KNOWLEDGE_TABLE}_{key}_idx "
                f"ON {knowledge_table} ((context->>'{key}'))"
                for key in self._indexed_context_keys
            ),
        ]
        if dim <= HNSW_MAX_DIM:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {KNOWLEDGE_TABLE}_embedding_hnsw_idx "
                f"ON {knowledge_table} USING hnsw ({EMBEDDING_COLUMN} vector_cosine_ops)"
            )

        async with self._errors("initialise the schema"):
            async with self._engine.begin() as conn:
                for statement in statements:
                    await conn.execute(text(statement))
                stored_dim = await conn.scalar(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = CAST(:table AS regclass) AND attname = :column"
                    ),
                    {"table": knowledge_table, "column": EMBEDDING_COLUMN},
                )
        if stored_dim != dim:
            raise StorageError(
                f"{KNOWLEDGE_TABLE}.{EMBEDDING_COLUMN} is vector({stored_dim}) but "
                f"EMBEDDING_DIM is {dim}; re-embed into a new table or fix the setting"
            )

    async def _vector(self) -> PGVectorStore:
        if self._vector_store is not None:
            return self._vector_store
        async with self._vector_store_lock:
            if self._vector_store is None:
                async with self._errors("open the vector store"):
                    self._vector_store = await PGVectorStore.create(
                        self._pg_engine,
                        embedding_service=self._embeddings,
                        table_name=KNOWLEDGE_TABLE,
                        schema_name=self._schema,
                        id_column=ID_COLUMN,
                        content_column=CONTENT_COLUMN,
                        embedding_column=EMBEDDING_COLUMN,
                        metadata_columns=METADATA_COLUMNS,
                        metadata_json_column=None,
                    )
        return self._vector_store

    # ------------------------------------------------------- raw conversations

    async def add_conversation(
        self,
        transcript: str,
        *,
        stated_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RawConversation:
        """Store one transcript as pending evidence."""

        async with self._errors("store a conversation"):
            async with self._engine.begin() as conn:
                row = (
                    await conn.execute(
                        text(
                            f"INSERT INTO {self._table(RAW_CONVERSATION_TABLE)} "
                            "(id, transcript, stated_by, metadata) "
                            "VALUES (:id, :transcript, :stated_by, CAST(:metadata AS jsonb)) "
                            "RETURNING *"
                        ),
                        {
                            "id": uuid4(),
                            "transcript": transcript,
                            "stated_by": stated_by,
                            "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                        },
                    )
                ).mappings().one()
        return RawConversation.model_validate(dict(row))

    async def mark_conversation(
        self,
        conversation_id: UUID,
        status: ConversationStatus,
        *,
        error: str | None = None,
    ) -> None:
        async with self._errors("update a conversation"):
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"UPDATE {self._table(RAW_CONVERSATION_TABLE)} "
                        "SET status = :status, error = :error WHERE id = :id"
                    ),
                    {"id": conversation_id, "status": status.value, "error": error},
                )

    async def processed_metadata_values(
        self, key: str, *, without_key: str | None = None
    ) -> set[str]:
        """Values of metadata[key] over conversations that were fully ingested.

        Conversations whose metadata has `without_key` (e.g. benchmark runs)
        are left out.
        """

        async with self._errors("read ingested conversations"):
            async with self._engine.connect() as conn:
                values = (
                    await conn.execute(
                        text(
                            f"SELECT DISTINCT metadata ->> :key "
                            f"FROM {self._table(RAW_CONVERSATION_TABLE)} "
                            "WHERE status = :status AND metadata ? :key "
                            "AND NOT (CAST(:without_key AS text) IS NOT NULL "
                            "AND metadata ? CAST(:without_key AS text))"
                        ),
                        {
                            "key": key,
                            "status": ConversationStatus.PROCESSED.value,
                            "without_key": without_key,
                        },
                    )
                ).scalars()
                return {value for value in values if value is not None}

    async def find_processed_conversation(
        self, transcript_sha256: str, *, without_key: str | None = None
    ) -> UUID | None:
        """Earliest fully ingested conversation with this transcript hash."""

        async with self._errors("read ingested conversations"):
            async with self._engine.connect() as conn:
                return await conn.scalar(
                    text(
                        f"SELECT id FROM {self._table(RAW_CONVERSATION_TABLE)} "
                        "WHERE status = :status "
                        "AND metadata ->> 'transcript_sha256' = :sha "
                        "AND NOT (CAST(:without_key AS text) IS NOT NULL "
                        "AND metadata ? CAST(:without_key AS text)) "
                        "ORDER BY created_at, id LIMIT 1"
                    ),
                    {
                        "sha": transcript_sha256,
                        "status": ConversationStatus.PROCESSED.value,
                        "without_key": without_key,
                    },
                )

    async def get_conversation(self, conversation_id: UUID) -> RawConversation | None:
        async with self._errors("read a conversation"):
            async with self._engine.connect() as conn:
                row = (
                    await conn.execute(
                        text(f"SELECT * FROM {self._table(RAW_CONVERSATION_TABLE)} WHERE id = :id"),
                        {"id": conversation_id},
                    )
                ).mappings().one_or_none()
        return RawConversation.model_validate(dict(row)) if row else None

    # ------------------------------------------------------------- knowledge

    async def embed_knowledge(self, records: Sequence[KnowledgeRecord]) -> None:
        """Fill in missing embeddings with one (client-chunked) embedding call."""

        missing = [record for record in records if record.embedding is None]
        if not missing:
            return
        vectors = await self._embeddings.aembed_documents(
            [record.embedding_text(self._embedding_context_keys) for record in missing]
        )
        for record, vector in zip(missing, vectors, strict=True):
            record.embedding = list(vector)

    async def add_knowledge(
        self,
        records: Sequence[KnowledgeRecord],
        *,
        updates: Sequence[KnowledgeLinkUpdate] = (),
    ) -> list[UUID]:
        """Embed (if not done yet) and store records; returns their ids.

        `updates` link stored records to the new ones (or to each other) in
        the same transaction, so a failure leaves neither half behind.
        """

        await self.embed_knowledge(records)
        for record in records:
            if len(record.embedding or []) != self._vector_size:
                raise StorageError(
                    f"Embedding for knowledge {record.id} has "
                    f"{len(record.embedding or [])} dimensions, expected {self._vector_size}"
                )
        if not records and not updates:
            return []

        columns = [ID_COLUMN, CONTENT_COLUMN, EMBEDDING_COLUMN, *METADATA_COLUMNS]
        insert = text(
            f"INSERT INTO {self._table(KNOWLEDGE_TABLE)} "
            f"({', '.join(columns)}) VALUES ({', '.join(':' + c for c in columns)})"
        )
        rows = [
            {
                ID_COLUMN: record.id,
                CONTENT_COLUMN: record.claim,
                EMBEDDING_COLUMN: json.dumps(record.embedding, separators=(",", ":")),
                **{
                    column: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, dict) else value
                    for column, value in record.to_metadata(self._embedding_context_keys).items()
                },
            }
            for record in records
        ]
        async with self._errors("store knowledge"):
            async with self._engine.begin() as conn:
                if rows:
                    await conn.execute(insert, rows)
                if updates:
                    await self._apply_link_updates(conn, updates)
        return [record.id for record in records]

    async def _apply_link_updates(
        self, conn: Any, updates: Sequence[KnowledgeLinkUpdate]
    ) -> None:
        knowledge_table = self._table(KNOWLEDGE_TABLE)
        ids = list(dict.fromkeys(update.record_id for update in updates))
        current = {
            row["id"]: dict(row)
            for row in (
                await conn.execute(
                    text(
                        "SELECT id, supported_by, contradicted_by, conflict_with, "
                        f"superseded_by, status, revision FROM {knowledge_table} "
                        "WHERE id = ANY(:ids) ORDER BY id FOR UPDATE"
                    ),
                    {"ids": ids},
                )
            ).mappings()
        }
        if missing := [str(i) for i in ids if i not in current]:
            raise StorageError(f"Cannot link unknown knowledge: {', '.join(missing)}")
        for update in updates:
            row = current[update.record_id]
            for column in ("supported_by", "contradicted_by", "conflict_with"):
                row[column] = list(
                    dict.fromkeys([*(row[column] or []), *getattr(update, column)])
                )
            # The first replacement wins; a later one is only recorded.
            if update.superseded_by is not None and row["superseded_by"] is None:
                row["superseded_by"] = update.superseded_by
                row["status"] = KnowledgeStatus.DEPRECATED.value
            if update.event is not None:
                revision = dict(row["revision"] or {})
                revision["history"] = [*revision.get("history", []), update.event]
                row["revision"] = revision
        await conn.execute(
            text(
                f"UPDATE {knowledge_table} SET "
                "supported_by = CAST(:supported_by AS uuid[]), "
                "contradicted_by = CAST(:contradicted_by AS uuid[]), "
                "conflict_with = CAST(:conflict_with AS uuid[]), "
                "superseded_by = :superseded_by, status = :status, "
                "revision = CAST(:revision AS jsonb) WHERE id = :id"
            ),
            [
                {
                    "id": row["id"],
                    "supported_by": row["supported_by"],
                    "contradicted_by": row["contradicted_by"],
                    "conflict_with": row["conflict_with"],
                    "superseded_by": row["superseded_by"],
                    "status": row["status"],
                    "revision": json.dumps(row["revision"], ensure_ascii=False, default=str),
                }
                for row in current.values()
            ],
        )

    async def similar_knowledge(
        self,
        record: KnowledgeRecord,
        *,
        k: int = DEFAULT_SEARCH_K,
        types: Collection[KnowledgeType | str] = (),
        filters: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[KnowledgeRecord, float]]:
        """Searchable records nearest to an embedded record, with cosine similarity.

        Used to find what new knowledge duplicates, refines, or contradicts.
        """

        if record.embedding is None:
            await self.embed_knowledge([record])
        where_sql, parameters = _compile_knowledge_filter(filters)
        type_sql = "TRUE"
        if types:
            type_sql = "k.type = ANY(:types)"
            parameters["types"] = [KnowledgeType(kind).value for kind in types]
        statement = text(
            f"""
            SELECT {_RECORD_SELECT_SQL},
                   1 - (k.{EMBEDDING_COLUMN} <=> CAST(:embedding AS vector)) AS similarity
            FROM {self._table(KNOWLEDGE_TABLE)} AS k
            WHERE {where_sql} AND {type_sql} AND {_SEARCHABLE_SQL} AND k.id <> :id
              AND 1 - (k.{EMBEDDING_COLUMN} <=> CAST(:embedding AS vector)) >= :min_similarity
            ORDER BY k.{EMBEDDING_COLUMN} <=> CAST(:embedding AS vector), k.id
            LIMIT :k
            """
        )
        parameters.update({
            "embedding": json.dumps(record.embedding, separators=(",", ":")),
            "id": record.id,
            "k": max(1, min(k, MAX_SEARCH_K)),
            "min_similarity": min_similarity,
        })
        async with self._errors("find similar knowledge"):
            async with self._engine.connect() as conn:
                rows = (await conn.execute(statement, parameters)).mappings().all()
        return [
            (
                KnowledgeRecord.model_validate({key: row[key] for key in RECORD_COLUMNS}),
                float(row["similarity"]),
            )
            for row in rows
        ]

    async def reembed_knowledge(self) -> int:
        """Recompute every stored embedding with the current model, in place.

        Resizes the embedding column when the model's dimension differs. All
        vectors are computed before the table is touched, and the swap runs in
        one transaction, so a failure leaves the old embeddings intact.
        """

        knowledge_table = self._table(KNOWLEDGE_TABLE)
        async with self._errors("read knowledge for re-embedding"):
            async with self._engine.connect() as conn:
                rows = (
                    await conn.execute(
                        text(
                            f"SELECT id, subject, claim, context, type "
                            f"FROM {knowledge_table} ORDER BY id"
                        )
                    )
                ).mappings().all()
        records = [KnowledgeRecord.model_validate(dict(row)) for row in rows]
        vectors = await self._embeddings.aembed_documents(
            [record.embedding_text(self._embedding_context_keys) for record in records]
        )
        for record, vector in zip(records, vectors, strict=True):
            if len(vector) != self._vector_size:
                raise StorageError(
                    f"Embedding for knowledge {record.id} has {len(vector)} "
                    f"dimensions, expected {self._vector_size}"
                )

        dim = self._vector_size
        async with self._errors("re-embed knowledge"):
            async with self._engine.begin() as conn:
                # Block writers so no row is added between the read and the swap.
                await conn.execute(
                    text(f"LOCK TABLE {knowledge_table} IN SHARE ROW EXCLUSIVE MODE")
                )
                current_ids = set(
                    (
                        await conn.execute(text(f"SELECT id FROM {knowledge_table}"))
                    ).scalars()
                )
                if current_ids != {record.id for record in records}:
                    raise StorageError(
                        "Knowledge changed while re-embedding; run it again"
                    )
                stored_dim = await conn.scalar(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = CAST(:table AS regclass) AND attname = :column"
                    ),
                    {"table": knowledge_table, "column": EMBEDDING_COLUMN},
                )
                if stored_dim != dim:
                    # Old vectors cannot be cast to the new size; drop the ANN
                    # index (init_schema recreates it) and refill the column.
                    await conn.execute(
                        text(
                            f'DROP INDEX IF EXISTS "{self._schema}".'
                            f"{KNOWLEDGE_TABLE}_embedding_hnsw_idx"
                        )
                    )
                    await conn.execute(
                        text(
                            f"ALTER TABLE {knowledge_table} "
                            f"ALTER COLUMN {EMBEDDING_COLUMN} DROP NOT NULL, "
                            f"ALTER COLUMN {EMBEDDING_COLUMN} TYPE vector({dim}) USING NULL"
                        )
                    )
                if records:
                    await conn.execute(
                        text(
                            f"UPDATE {knowledge_table} "
                            f"SET {EMBEDDING_COLUMN} = CAST(:embedding AS vector) "
                            "WHERE id = :id"
                        ),
                        [
                            {
                                "id": record.id,
                                "embedding": json.dumps(
                                    list(vector), separators=(",", ":")
                                ),
                            }
                            for record, vector in zip(records, vectors, strict=True)
                        ],
                    )
                await conn.execute(
                    text(
                        f"ALTER TABLE {knowledge_table} "
                        f"ALTER COLUMN {EMBEDDING_COLUMN} SET NOT NULL"
                    )
                )
        # PGVectorStore caches column metadata; reopen it on next use.
        self._vector_store = None
        return len(records)

    async def update_knowledge_scope(
        self, records: Sequence[KnowledgeRecord], *, revision_keys: Sequence[str] = ()
    ) -> int:
        """Store the new context of records in place, without re-extraction.

        Re-embeds the records and refreshes their search text, since both
        include the context; of the revision only `revision_keys` are written,
        so links and history recorded since they were read are kept. Returns
        the number of rows updated.
        """

        if not records:
            return 0
        for record in records:
            record.embedding = None
        await self.embed_knowledge(records)
        for record in records:
            if len(record.embedding or []) != self._vector_size:
                raise StorageError(
                    f"Embedding for knowledge {record.id} has "
                    f"{len(record.embedding or [])} dimensions, expected {self._vector_size}"
                )
        statement = text(
            f"UPDATE {self._table(KNOWLEDGE_TABLE)} SET "
            "context = CAST(:context AS jsonb), search_text = :search_text, "
            f"{EMBEDDING_COLUMN} = CAST(:embedding AS vector), "
            "revision = COALESCE(revision, '{}'::jsonb) || CAST(:revision AS jsonb) "
            "WHERE id = :id RETURNING id"
        )
        updated = 0
        async with self._errors("update knowledge scope"):
            async with self._engine.begin() as conn:
                for record in records:
                    result = await conn.execute(statement, {
                        "id": record.id,
                        "context": json.dumps(record.context, ensure_ascii=False, default=str),
                        "search_text": record.lexical_text(self._embedding_context_keys),
                        "embedding": json.dumps(record.embedding, separators=(",", ":")),
                        "revision": json.dumps(
                            {key: record.revision[key] for key in revision_keys
                             if key in record.revision},
                            ensure_ascii=False, default=str,
                        ),
                    })
                    updated += len(result.scalars().all())
        return updated

    async def search_knowledge(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = DEFAULT_SEARCH_K,
    ) -> list[KnowledgeSearchHit]:
        """Fuse vector and PostgreSQL full-text ranks with RRF.

        The nearest vector candidates are always kept. Each hit is labelled
        ``confident`` when its cosine similarity reaches the configured
        threshold or it shares enough query terms; confident hits come first.
        """

        result_k = max(1, min(k, MAX_SEARCH_K))
        candidate_k = min(
            MAX_SEARCH_CANDIDATES,
            max(MIN_SEARCH_CANDIDATES, result_k * SEARCH_CANDIDATE_MULTIPLIER),
        )
        where_sql, filter_parameters = _compile_knowledge_filter(filters)
        query_embedding = list(await self._embeddings.aembed_query(query))
        if len(query_embedding) != self._vector_size:
            raise StorageError(
                f"Query embedding has {len(query_embedding)} dimensions, "
                f"expected {self._vector_size}"
            )

        knowledge_table = self._table(KNOWLEDGE_TABLE)
        text_search_config = self._text_search_config
        statement = text(
            f"""
            WITH query_values AS (
                SELECT CAST(:query_embedding AS vector) AS embedding,
                       websearch_to_tsquery(
                           '{text_search_config}'::regconfig, :lexical_query
                       ) AS lexical_query,
                       tsvector_to_array(
                           to_tsvector('{text_search_config}'::regconfig, :query)
                       ) AS query_terms
            ),
            vector_scored AS (
                SELECT k.id,
                       k.{EMBEDDING_COLUMN} <=> q.embedding AS vector_distance
                FROM {knowledge_table} AS k
                CROSS JOIN query_values AS q
                WHERE {where_sql} AND {_SEARCHABLE_SQL}
            ),
            vector_candidates AS (
                SELECT id, vector_distance
                FROM vector_scored
                ORDER BY vector_distance ASC, id ASC
                LIMIT :candidate_k
            ),
            vector_ranked AS (
                SELECT id, vector_distance,
                       row_number() OVER (
                           ORDER BY vector_distance ASC, id ASC
                       )::integer AS vector_rank
                FROM vector_candidates
            ),
            lexical_candidates AS (
                SELECT k.id,
                       ts_rank_cd(k.search_vector, q.lexical_query) AS lexical_score,
                       cardinality(ARRAY(
                           SELECT unnest(tsvector_to_array(k.search_vector))
                           INTERSECT
                           SELECT unnest(q.query_terms)
                       ))::integer AS lexical_matched_terms,
                       cardinality(q.query_terms) AS query_term_count
                FROM {knowledge_table} AS k
                CROSS JOIN query_values AS q
                WHERE {where_sql} AND {_SEARCHABLE_SQL}
                  AND k.search_vector @@ q.lexical_query
                ORDER BY lexical_score DESC, k.id ASC
                LIMIT :candidate_k
            ),
            lexical_ranked AS (
                SELECT id, lexical_score, lexical_matched_terms, query_term_count,
                       row_number() OVER (
                           ORDER BY lexical_score DESC, id ASC
                       )::integer AS lexical_rank
                FROM lexical_candidates
            ),
            combined AS (
                SELECT COALESCE(v.id, l.id) AS id,
                       v.vector_rank,
                       v.vector_distance,
                       l.lexical_rank,
                       l.lexical_score,
                       l.lexical_matched_terms,
                       l.query_term_count
                FROM vector_ranked AS v
                FULL OUTER JOIN lexical_ranked AS l USING (id)
            ),
            scored AS (
                SELECT combined.*,
                       COALESCE(1.0 / (:rrf_k + vector_rank), 0.0) +
                       COALESCE(1.0 / (:rrf_k + lexical_rank), 0.0) AS hybrid_score,
                       COALESCE(vector_distance <= :max_vector_distance, FALSE)
                       OR COALESCE(
                           lexical_matched_terms >= GREATEST(
                               LEAST(:lexical_min_terms, query_term_count),
                               ceil(:lexical_term_fraction * query_term_count)
                           ),
                           FALSE
                       ) AS confident
                FROM combined
            )
            SELECT {_RECORD_SELECT_SQL},
                   scored.hybrid_score, scored.vector_rank,
                   scored.lexical_rank, scored.vector_distance, scored.lexical_score,
                   scored.lexical_matched_terms, scored.confident
            FROM scored
            JOIN {knowledge_table} AS k ON k.id = scored.id
            ORDER BY scored.confident DESC,
                     scored.hybrid_score DESC,
                     ((scored.vector_rank IS NOT NULL)::integer +
                      (scored.lexical_rank IS NOT NULL)::integer) DESC,
                     LEAST(COALESCE(scored.vector_rank, 2147483647),
                           COALESCE(scored.lexical_rank, 2147483647)) ASC,
                     scored.lexical_rank ASC NULLS LAST,
                     scored.vector_distance ASC NULLS LAST,
                     k.id ASC
            LIMIT :result_k
            """
        )
        parameters = {
            **filter_parameters,
            "query": query,
            "lexical_query": lexical_or_query(query),
            "query_embedding": json.dumps(query_embedding, separators=(",", ":")),
            "candidate_k": candidate_k,
            "result_k": result_k,
            "rrf_k": RRF_RANK_CONSTANT,
            "max_vector_distance": 1.0 - self._min_vector_similarity,
            "lexical_min_terms": LEXICAL_CONFIDENT_MIN_TERMS,
            "lexical_term_fraction": LEXICAL_CONFIDENT_TERM_FRACTION,
        }
        async with self._errors("search knowledge"):
            async with self._engine.connect() as conn:
                rows = (await conn.execute(statement, parameters)).mappings().all()

        return [
            KnowledgeSearchHit(
                record=KnowledgeRecord.model_validate(
                    {key: row[key] for key in RECORD_COLUMNS}
                ),
                hybrid_score=float(row["hybrid_score"]),
                vector_rank=row["vector_rank"],
                lexical_rank=row["lexical_rank"],
                vector_distance=(
                    float(row["vector_distance"])
                    if row["vector_distance"] is not None
                    else None
                ),
                lexical_score=(
                    float(row["lexical_score"])
                    if row["lexical_score"] is not None
                    else None
                ),
                lexical_matched_terms=row["lexical_matched_terms"],
                confident=bool(row["confident"]),
            )
            for row in rows
        ]

    async def get_knowledge(self, ids: Sequence[UUID]) -> list[KnowledgeRecord]:
        """Fetch records by id, in the requested order, skipping unknown ids."""

        if not ids:
            return []
        store = await self._vector()
        async with self._errors("read knowledge"):
            documents = await store.aget_by_ids([str(i) for i in ids])
        by_id = {UUID(doc.id): KnowledgeRecord.from_document(doc) for doc in documents}
        return [by_id[i] for i in ids if i in by_id]
