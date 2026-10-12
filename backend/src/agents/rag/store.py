"""Chunk index over PostgreSQL + pgvector with semantic, keyword, and hybrid search.

Uses the same retrieval engine as the knowledge store: the shared embedding
client, pgvector cosine distance, PostgreSQL full-text parsing over an OR query
(lexical_or_query), and reciprocal rank fusion. Keyword search ranks by Okapi
BM25 rather than ts_rank_cd, which has no IDF or length normalisation. BM25
runs in plain SQL over a postings table ({table}_terms: lexeme, chunk, term
frequency) derived from each chunk's tsvector, so stemming and stop words match
the full-text index and no extension is needed. Unlike search_knowledge it
returns plain rankings without confidence labels, so each mode can be
evaluated on its own.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from backend.src.agents.rag.chunking import Chunk
from backend.src.storage.knowledge_store import (
    DEFAULT_TEXT_SEARCH_CONFIG,
    HNSW_MAX_DIM,
    MIN_SEARCH_CANDIDATES,
    RRF_RANK_CONSTANT,
    StorageError,
    lexical_or_query,
)


DATABASE_PREFIX = "rag_"
INDEX_TABLE = "rag_indexes"
# Chunks fetched per requested document, so documents split into several
# chunks still fill k distinct documents.
DOCUMENT_OVERFETCH = 3
# pgvector returns at most hnsw.ef_search rows from an HNSW scan (default 40).
MAX_EF_SEARCH = 1000
# BM25 term-frequency saturation and length normalisation (Pyserini/BEIR defaults).
BM25_K1 = 0.9
BM25_B = 0.4
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    text: str
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


def rrf_fuse(
    vector_hits: Sequence[ChunkHit],
    lexical_hits: Sequence[ChunkHit],
    *,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[ChunkHit]:
    """Fuse two rankings by reciprocal rank; ties go to the better single rank."""

    vector_ranks = {hit.chunk_id: rank for rank, hit in enumerate(vector_hits, start=1)}
    lexical_ranks = {hit.chunk_id: rank for rank, hit in enumerate(lexical_hits, start=1)}
    hits = {hit.chunk_id: hit for hit in (*lexical_hits, *vector_hits)}
    fused = []
    for chunk_id, hit in hits.items():
        vector_rank = vector_ranks.get(chunk_id)
        lexical_rank = lexical_ranks.get(chunk_id)
        score = sum(
            1.0 / (rank_constant + rank)
            for rank in (vector_rank, lexical_rank)
            if rank is not None
        )
        fused.append(
            ChunkHit(
                chunk_id=chunk_id,
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
                title=hit.title,
                text=hit.text,
                score=score,
                vector_rank=vector_rank,
                lexical_rank=lexical_rank,
                metadata=hit.metadata,
            )
        )
    fused.sort(
        key=lambda hit: (
            -hit.score,
            min(hit.vector_rank or 2**31, hit.lexical_rank or 2**31),
            hit.chunk_id,
        )
    )
    return fused


def best_chunk_per_document(hits: Sequence[ChunkHit]) -> list[ChunkHit]:
    """Keep each document's highest-ranked chunk, in ranking order."""

    seen: set[str] = set()
    documents = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            documents.append(hit)
    return documents


class DocumentStore:
    """Chunks of one dataset in table `table`, with their embeddings."""

    def __init__(
        self,
        engine: AsyncEngine,
        embeddings: Embeddings,
        vector_size: int,
        *,
        table: str = "chunks",
        schema: str = "public",
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
    ) -> None:
        if bm25_k1 < 0 or not 0 <= bm25_b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self._engine = engine
        self._embeddings = embeddings
        self._vector_size = vector_size
        self._table_name = _identifier(table)
        self._terms_name = _identifier(f"{table}_terms")
        self._schema = _identifier(schema)
        self._text_search_config = _identifier(text_search_config)
        self._bm25_k1 = float(bm25_k1)
        self._bm25_b = float(bm25_b)

    @classmethod
    def from_url(
        cls, database_url: str, embeddings: Embeddings, vector_size: int, **options: Any
    ) -> "DocumentStore":
        return cls(
            create_async_engine(database_url, pool_pre_ping=True),
            embeddings,
            vector_size,
            **options,
        )

    @classmethod
    async def for_dataset(cls, dataset: str, **options: Any) -> "DocumentStore":
        """Open database rag_<dataset> on the DATABASE_URL server, creating it if missing."""

        return await cls.for_database(DATABASE_PREFIX + dataset, **options)

    @classmethod
    async def for_database(
        cls,
        database: str,
        *,
        query_instruction: str = "",
        embeddings: Embeddings | None = None,
        init_schema: bool = True,
        **options: Any,
    ) -> "DocumentStore":
        """Open `database` on the DATABASE_URL server, creating it if missing.

        The embedding model comes from EMBEDDING_*; query_instruction prefixes
        queries only. An explicitly supplied embedding client is used as-is.
        """

        from backend.src.storage.databases import ensure_database, server_url_from_env
        from backend.src.storage.embeddings import (
            build_embeddings_from_env,
            embedding_dim_from_env,
        )

        store = cls.from_url(
            await ensure_database(server_url_from_env(), database),
            embeddings
            if embeddings is not None
            else build_embeddings_from_env(query_instruction=query_instruction),
            embedding_dim_from_env(),
            **options,
        )
        try:
            if init_schema:
                await store.init_schema()
        except BaseException:
            await store.aclose()
            raise
        return store

    @property
    def table_name(self) -> str:
        return self._table_name

    def _table(self, name: str) -> str:
        return f'"{self._schema}"."{name}"'

    async def aclose(self) -> None:
        await self._engine.dispose()

    @asynccontextmanager
    async def _errors(self, action: str) -> AsyncIterator[None]:
        try:
            yield
        except SQLAlchemyError as exc:
            raise StorageError(f"Document store failed to {action}: {exc}") from exc

    # ------------------------------------------------------------------ schema

    async def init_schema(self) -> None:
        table = self._table(self._table_name)
        name = self._table_name
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS {self._table(INDEX_TABLE)} (
                table_name TEXT PRIMARY KEY,
                config JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                search_text TEXT NOT NULL,
                search_vector TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('{self._text_search_config}'::regconfig, search_text)
                ) STORED,
                embedding vector({self._vector_size}) NOT NULL,
                term_count INTEGER
            )
            """,
            # Tables indexed before BM25 get the column here and are backfilled below.
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS term_count INTEGER",
            f"CREATE INDEX IF NOT EXISTS {name}_doc_id_idx ON {table} (doc_id)",
            f"CREATE INDEX IF NOT EXISTS {name}_search_vector_idx "
            f"ON {table} USING gin (search_vector)",
            f"CREATE INDEX IF NOT EXISTS {name}_unindexed_terms_idx "
            f"ON {table} (id) WHERE term_count IS NULL",
            f"""
            CREATE TABLE IF NOT EXISTS {self._table(self._terms_name)} (
                lexeme TEXT NOT NULL,
                chunk_id TEXT NOT NULL REFERENCES {table} (id) ON DELETE CASCADE,
                tf INTEGER NOT NULL,
                PRIMARY KEY (lexeme, chunk_id)
            )
            """,
        ]
        if self._vector_size <= HNSW_MAX_DIM:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {name}_embedding_idx "
                f"ON {table} USING hnsw (embedding vector_cosine_ops)"
            )
        async with self._errors("initialise the schema"):
            async with self._engine.begin() as conn:
                for statement in statements:
                    await conn.execute(text(statement))
                stored_dim = await conn.scalar(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = CAST(:table AS regclass) AND attname = 'embedding'"
                    ),
                    {"table": table},
                )
                await self._index_terms(conn)
        if stored_dim != self._vector_size:
            raise StorageError(
                f"{self._table_name}.embedding is vector({stored_dim}) but the "
                f"embedding size is {self._vector_size}; rebuild the index"
            )

    async def ensure_config(self, config: Mapping[str, Any]) -> None:
        """Record how the index is built; refuse to mix chunks from another setup."""

        async with self._errors("read the index configuration"):
            async with self._engine.begin() as conn:
                stored = await conn.scalar(
                    text(
                        f"SELECT config FROM {self._table(INDEX_TABLE)} "
                        "WHERE table_name = :table"
                    ),
                    {"table": self._table_name},
                )
                if stored is None:
                    await conn.execute(
                        text(
                            f"INSERT INTO {self._table(INDEX_TABLE)} (table_name, config) "
                            "VALUES (:table, CAST(:config AS jsonb))"
                        ),
                        {"table": self._table_name, "config": json.dumps(dict(config))},
                    )
                    return
        if stored != dict(config):
            raise StorageError(
                f"Index {self._table_name} was built with {stored}, not {dict(config)}; "
                "rebuild it"
            )

    async def config(self) -> dict[str, Any] | None:
        async with self._errors("read the index configuration"):
            async with self._engine.connect() as conn:
                return await conn.scalar(
                    text(
                        f"SELECT config FROM {self._table(INDEX_TABLE)} "
                        "WHERE table_name = :table"
                    ),
                    {"table": self._table_name},
                )

    async def drop(self) -> None:
        """Delete the chunk table and its recorded configuration."""

        async with self._errors("drop the index"):
            async with self._engine.begin() as conn:
                await conn.execute(text(f"DROP TABLE IF EXISTS {self._table(self._terms_name)}"))
                await conn.execute(text(f"DROP TABLE IF EXISTS {self._table(self._table_name)}"))
                await conn.execute(
                    text(f"DELETE FROM {self._table(INDEX_TABLE)} WHERE table_name = :table"),
                    {"table": self._table_name},
                )

    # ----------------------------------------------------------------- indexing

    async def indexed_doc_ids(self) -> set[str]:
        async with self._errors("read indexed documents"):
            async with self._engine.connect() as conn:
                rows = await conn.execute(
                    text(f"SELECT DISTINCT doc_id FROM {self._table(self._table_name)}")
                )
                return set(rows.scalars())

    async def count_chunks(self) -> int:
        async with self._errors("count chunks"):
            async with self._engine.connect() as conn:
                return int(
                    await conn.scalar(
                        text(f"SELECT count(*) FROM {self._table(self._table_name)}")
                    )
                )

    async def _index_terms(self, conn: Any) -> None:
        """Add BM25 postings and lengths for chunks that have none yet."""

        table = self._table(self._table_name)
        await conn.execute(
            text(
                f"""
                INSERT INTO {self._table(self._terms_name)} (lexeme, chunk_id, tf)
                SELECT t.lexeme, c.id, cardinality(t.positions)
                FROM {table} AS c,
                     unnest(c.search_vector) AS t(lexeme, positions, weights)
                WHERE c.term_count IS NULL
                ON CONFLICT DO NOTHING
                """
            )
        )
        # Document length counts every token kept by the parser, as in BM25.
        await conn.execute(
            text(
                f"""
                UPDATE {table} AS c
                SET term_count = coalesce(
                    (SELECT sum(cardinality(t.positions))
                     FROM unnest(c.search_vector) AS t(lexeme, positions, weights)),
                    0
                )
                WHERE c.term_count IS NULL
                """
            )
        )

    async def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks in one transaction; existing chunk ids are kept."""

        if not chunks:
            return 0
        vectors = await self._embeddings.aembed_documents(
            [chunk.search_text for chunk in chunks]
        )
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self._vector_size:
                raise StorageError(
                    f"Embedding for chunk {chunk.id} has {len(vector)} dimensions, "
                    f"expected {self._vector_size}"
                )
            rows.append(
                {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.index,
                    "title": chunk.title,
                    "content": chunk.text,
                    "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                    "search_text": chunk.search_text,
                    "embedding": json.dumps(list(vector), separators=(",", ":")),
                }
            )
        async with self._errors("store chunks"):
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table(self._table_name)}
                            (id, doc_id, chunk_index, title, content, metadata,
                             search_text, embedding)
                        VALUES (:id, :doc_id, :chunk_index, :title, :content,
                                CAST(:metadata AS jsonb), :search_text,
                                CAST(:embedding AS vector))
                        ON CONFLICT (id) DO NOTHING
                        """
                    ),
                    rows,
                )
                await self._index_terms(conn)
        return len(rows)

    async def get_document(self, doc_id: str) -> list[ChunkHit]:
        """A document's chunks in order; consecutive chunks may share sentences."""

        statement = text(
            f"""
            SELECT id, doc_id, chunk_index, title, content, metadata
            FROM {self._table(self._table_name)}
            WHERE doc_id = :doc_id
            ORDER BY chunk_index
            """
        )
        async with self._errors("read a document"):
            async with self._engine.connect() as conn:
                rows = (await conn.execute(statement, {"doc_id": doc_id})).mappings().all()
        return [
            ChunkHit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                chunk_index=row["chunk_index"],
                title=row["title"],
                text=row["content"],
                score=0.0,
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]

    async def adjacent_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkHit]:
        """Immediate indexed neighbors; supports legacy page-chunk document IDs."""
        if not chunk_ids:
            return []
        statement = text(f"""
            WITH ordered AS (
                SELECT id,
                    lag(id) OVER w AS previous_id, lead(id) OVER w AS next_id
                FROM {self._table(self._table_name)}
                WINDOW w AS (
                    PARTITION BY COALESCE(metadata->>'reference_doc_id', metadata->>'doc_id',
                        regexp_replace(doc_id, '_p[0-9]+_c[0-9]+$', ''))
                    ORDER BY CASE WHEN metadata->>'page_start' ~ '^[0-9]+$'
                        THEN (metadata->>'page_start')::integer
                        ELSE COALESCE(substring(doc_id from '_p([0-9]+)')::integer, 0) END,
                        doc_id, chunk_index
                )
            ), neighbors AS (
                SELECT previous_id AS id FROM ordered WHERE id = ANY(:ids)
                UNION SELECT next_id FROM ordered WHERE id = ANY(:ids)
            )
            SELECT c.id, c.doc_id, c.chunk_index, c.title, c.content, c.metadata
            FROM {self._table(self._table_name)} c JOIN neighbors n ON n.id = c.id
            ORDER BY c.doc_id, c.chunk_index
        """)
        async with self._errors("read adjacent passages"):
            async with self._engine.connect() as conn:
                rows = (await conn.execute(statement, {"ids": list(chunk_ids)})).mappings().all()
        return [ChunkHit(chunk_id=row["id"], doc_id=row["doc_id"], chunk_index=row["chunk_index"],
                         title=row["title"], text=row["content"], score=0.0,
                         metadata=row.get("metadata") or {}) for row in rows]

    # ------------------------------------------------------------------ search

    async def embed_query(self, query: str) -> list[float]:
        vector = list(await self._embeddings.aembed_query(query))
        if len(vector) != self._vector_size:
            raise StorageError(
                f"Query embedding has {len(vector)} dimensions, expected {self._vector_size}"
            )
        return vector

    async def semantic_search(
        self, query_embedding: Sequence[float], *, k: int
    ) -> list[ChunkHit]:
        """Nearest chunks by cosine distance; score is cosine similarity."""

        statement = text(
            f"""
            SELECT id, doc_id, chunk_index, title, content, metadata,
                   embedding <=> CAST(:embedding AS vector) AS distance
            FROM {self._table(self._table_name)}
            ORDER BY distance
            LIMIT :k
            """
        )
        ef_search = min(MAX_EF_SEARCH, max(40, k))
        async with self._errors("run semantic search"):
            async with self._engine.begin() as conn:
                await conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
                rows = (
                    await conn.execute(
                        statement,
                        {
                            "embedding": json.dumps(
                                list(query_embedding), separators=(",", ":")
                            ),
                            "k": k,
                        },
                    )
                ).mappings().all()
        return [
            ChunkHit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                chunk_index=row["chunk_index"],
                title=row["title"],
                text=row["content"],
                score=1.0 - float(row["distance"]),
                vector_rank=rank,
                metadata=row.get("metadata") or {},
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def keyword_search(self, query: str, *, k: int) -> list[ChunkHit]:
        """Chunks sharing any query term, ranked by BM25.

        idf = ln(1 + (N - df + 0.5) / (df + 0.5)) as in Lucene; N, df, and the
        average chunk length are read from the index at query time.
        """

        lexical_query = lexical_or_query(query)
        if not lexical_query:
            return []
        chunks = self._table(self._table_name)
        terms = self._table(self._terms_name)
        statement = text(
            f"""
            WITH query_terms AS (
                SELECT DISTINCT t.lexeme
                FROM unnest(
                    to_tsvector('{self._text_search_config}'::regconfig, :lexical_query)
                ) AS t(lexeme, positions, weights)
            ),
            corpus AS (
                SELECT count(*)::float8 AS n,
                       greatest(avg(term_count)::float8, 1.0) AS avgdl
                FROM {chunks}
            ),
            idf AS (
                SELECT p.lexeme,
                       ln(1.0 + (corpus.n - count(*) + 0.5) / (count(*) + 0.5)) AS idf
                FROM {terms} AS p
                JOIN query_terms USING (lexeme)
                CROSS JOIN corpus
                GROUP BY p.lexeme, corpus.n
            ),
            scores AS (
                SELECT p.chunk_id,
                       sum(
                           idf.idf * p.tf * (:k1 + 1.0)
                           / (p.tf + :k1 * (1.0 - :b + :b * c.term_count / corpus.avgdl))
                       ) AS score
                FROM {terms} AS p
                JOIN idf USING (lexeme)
                JOIN {chunks} AS c ON c.id = p.chunk_id
                CROSS JOIN corpus
                GROUP BY p.chunk_id
                ORDER BY score DESC, p.chunk_id
                LIMIT :k
            )
            SELECT c.id, c.doc_id, c.chunk_index, c.title, c.content, c.metadata, s.score
            FROM scores AS s
            JOIN {chunks} AS c ON c.id = s.chunk_id
            ORDER BY s.score DESC, c.id
            """
        )
        params = {
            "lexical_query": lexical_query,
            "k": k,
            "k1": self._bm25_k1,
            "b": self._bm25_b,
        }
        async with self._errors("run keyword search"):
            async with self._engine.connect() as conn:
                rows = (await conn.execute(statement, params)).mappings().all()
        return [
            ChunkHit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                chunk_index=row["chunk_index"],
                title=row["title"],
                text=row["content"],
                score=float(row["score"]),
                lexical_rank=rank,
                metadata=row.get("metadata") or {},
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        k: int = 10,
        query_embedding: Sequence[float] | None = None,
    ) -> list[ChunkHit]:
        """Top-k chunks. Hybrid fuses top max(k, 20) semantic and keyword chunks."""

        if k <= 0:
            raise ValueError("k must be positive")
        mode = SearchMode(mode)
        if mode is SearchMode.KEYWORD:
            return await self.keyword_search(query, k=k)
        if query_embedding is None:
            query_embedding = await self.embed_query(query)
        if mode is SearchMode.SEMANTIC:
            return await self.semantic_search(query_embedding, k=k)
        candidates = max(k, MIN_SEARCH_CANDIDATES)
        vector_hits = await self.semantic_search(query_embedding, k=candidates)
        lexical_hits = await self.keyword_search(query, k=candidates)
        return rrf_fuse(vector_hits, lexical_hits)[:k]

    async def search_documents(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        k: int = 10,
        query_embedding: Sequence[float] | None = None,
    ) -> list[ChunkHit]:
        """Top-k documents, each represented by its best chunk (MaxP)."""

        hits = await self.search(
            query, mode=mode, k=k * DOCUMENT_OVERFETCH, query_embedding=query_embedding
        )
        return best_chunk_per_document(hits)[:k]
