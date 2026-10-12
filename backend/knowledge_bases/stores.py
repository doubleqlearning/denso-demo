"""Open the store a knowledge base's agent searches."""

from __future__ import annotations

from backend.knowledge_bases.base import KnowledgeBaseSpec, Retrieval
from backend.src.agents.rag.store import DocumentStore
from backend.src.storage import KnowledgeStore


async def open_document_store(
    spec: KnowledgeBaseSpec, *, query_instruction: str | None = None, init_schema: bool = True
) -> DocumentStore:
    """Chunk table spec.document_table in the knowledge base's database."""

    return await DocumentStore.for_database(
        spec.database,
        query_instruction=(
            spec.embedding_query_instruction if query_instruction is None else query_instruction
        ),
        init_schema=init_schema,
        table=spec.document_table,
        text_search_config=spec.text_search_config,
    )


async def open_store(spec: KnowledgeBaseSpec) -> KnowledgeStore | DocumentStore:
    """KnowledgeStore for knowledge retrieval, DocumentStore for document retrieval."""

    if spec.retrieval is Retrieval.DOCUMENTS:
        return await open_document_store(spec)
    return await KnowledgeStore.for_kb(spec)
