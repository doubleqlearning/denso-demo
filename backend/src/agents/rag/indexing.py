"""Chunk, embed, and store documents in batches, resuming an interrupted run."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from backend.src.agents.rag.chunking import ChunkingConfig, Document, chunk_documents
from backend.src.agents.rag.store import DocumentStore
from backend.src.storage.embeddings import embedding_dim_from_env


# Documents per embedding call and insert transaction; a document's chunks
# always land in the same batch, so resume never sees half a document.
INDEX_BATCH_DOCS = 128


def index_config(chunking: ChunkingConfig) -> dict[str, Any]:
    """What an index was built with; chunks from another setup must not mix in."""

    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_dim": embedding_dim_from_env(),
        "chunk_words": chunking.max_words,
        "overlap_sentences": chunking.overlap_sentences if chunking.max_words else None,
    }


async def index_documents(
    store: DocumentStore,
    documents: Sequence[Document],
    chunking: ChunkingConfig,
    *,
    reindex: bool = False,
    batch_docs: int = INDEX_BATCH_DOCS,
    progress: Callable[[str], None] = print,
) -> int:
    """Index the documents not yet in the store; return the chunks added."""

    if reindex:
        await store.drop()
        await store.init_schema()
    await store.ensure_config(index_config(chunking))
    done = await store.indexed_doc_ids()
    remaining = [document for document in documents if document.id not in done]
    progress(
        f"[{store.table_name}] {len(documents)} documents, {len(documents) - len(remaining)} "
        f"already indexed, indexing {len(remaining)}"
    )
    added = 0
    for start in range(0, len(remaining), batch_docs):
        batch = remaining[start : start + batch_docs]
        added += await store.add_chunks(chunk_documents(batch, chunking))
        progress(f"  {start + len(batch)}/{len(remaining)} documents, {added} chunks")
    return added
