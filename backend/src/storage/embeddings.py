"""Embedding model construction for the knowledge base."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


# Local OpenAI-compatible servers (e.g. text-embeddings-inference) ignore the key,
# but the OpenAI client refuses to start without one.
LOCAL_API_KEY_PLACEHOLDER = "not-needed"


class QueryInstructionEmbeddings(Embeddings):
    """Prefix queries with a task instruction; documents are embedded as-is.

    Instruction-tuned models such as harrier-oss-v1, Qwen3-Embedding, or
    e5-instruct expect ``Instruct: <task>\nQuery: <query>`` on the query side only.
    """

    def __init__(self, embeddings: Embeddings, instruction: str) -> None:
        self.embeddings = embeddings
        self.instruction = instruction

    def _query(self, text: str) -> str:
        return f"Instruct: {self.instruction}\nQuery: {text}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(self._query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(self._query(text))


def embedding_dim_from_env() -> int:
    """Vector size of the `knowledge_items.embedding` column."""

    raw = os.getenv("EMBEDDING_DIM")
    if not raw:
        raise ValueError("Missing required embedding configuration: EMBEDDING_DIM")
    try:
        dim = int(raw)
    except ValueError as exc:
        raise ValueError("EMBEDDING_DIM must be a positive integer") from exc
    if dim <= 0:
        raise ValueError("EMBEDDING_DIM must be a positive integer")
    return dim


def _positive_int_from_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_embeddings_from_env(*, query_instruction: str = "") -> Embeddings:
    """Build a client from shared infrastructure settings and a KB instruction.

    The instruction is supplied by the knowledge base, never by a global
    environment variable. Empty instructions leave queries unchanged.
    """

    model = os.getenv("EMBEDDING_MODEL")
    base_url = os.getenv("EMBEDDING_BASE_URL") or None
    api_key = os.getenv("EMBEDDING_API_KEY") or (
        LOCAL_API_KEY_PLACEHOLDER if base_url else None
    )
    missing = [
        name
        for name, value in (("EMBEDDING_API_KEY", api_key), ("EMBEDDING_MODEL", model))
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required embedding configuration: " + ", ".join(missing)
        )

    # Only send `dimensions` when asked: many OpenAI-compatible providers
    # reject the parameter, while OpenAI's text-embedding-3 models honour it.
    dimensions = (
        embedding_dim_from_env()
        if os.getenv("EMBEDDING_SEND_DIMENSIONS", "").lower() in {"1", "true", "yes"}
        else None
    )
    options = {}
    # Servers such as text-embeddings-inference cap inputs per request.
    batch_size = _positive_int_from_env("EMBEDDING_BATCH_SIZE")
    if batch_size is not None:
        options["chunk_size"] = batch_size
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        dimensions=dimensions,
        # Tiktoken pre-chunking only matches OpenAI's own tokenizer.
        check_embedding_ctx_length=False,
        **options,
    )
    instruction = query_instruction.strip()
    if instruction:
        return QueryInstructionEmbeddings(embeddings, instruction)
    return embeddings
