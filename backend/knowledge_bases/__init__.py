"""Registry of knowledge bases.

To add one, copy the `maintenance` package (extract.md, agent.md, SPEC) under a
new name and register its SPEC below.
"""

from __future__ import annotations

from backend.knowledge_bases import industrial_qa, maintenance
from backend.knowledge_bases.base import (
    ContextFilter,
    AnswerPipeline,
    DatasetSpec,
    KnowledgeBaseSpec,
    Retrieval,
    RetrievalQuestions,
)


KNOWLEDGE_BASES: dict[str, KnowledgeBaseSpec] = {
    spec.name: spec for spec in (maintenance.SPEC, industrial_qa.SPEC)
}


def get_kb(name: str) -> KnowledgeBaseSpec:
    try:
        return KNOWLEDGE_BASES[name]
    except KeyError:
        available = ", ".join(sorted(KNOWLEDGE_BASES)) or "none"
        raise ValueError(
            f"Unknown knowledge base {name!r}; available: {available}"
        ) from None


__all__ = [
    "KNOWLEDGE_BASES",
    "AnswerPipeline",
    "ContextFilter",
    "DatasetSpec",
    "KnowledgeBaseSpec",
    "Retrieval",
    "RetrievalQuestions",
    "get_kb",
]
