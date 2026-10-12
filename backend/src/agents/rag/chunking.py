"""Split documents into sentence-aligned chunks for indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# A sentence ends at . ! or ? followed by whitespace and an uppercase letter,
# digit, or opening bracket, so "e.g. the" and "3.5 mm" stay in one sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"

    @property
    def search_text(self) -> str:
        """Embedded and keyword-indexed text; the title keeps every chunk on topic."""

        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class ChunkingConfig:
    """max_words=None keeps each document whole (one chunk per document)."""

    max_words: int | None = None
    overlap_sentences: int = 1

    def __post_init__(self) -> None:
        if self.max_words is not None and self.max_words <= 0:
            raise ValueError("max_words must be positive")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must not be negative")

    @property
    def label(self) -> str:
        """Short identifier used in index table names."""

        if self.max_words is None:
            return "full"
        return f"w{self.max_words}_o{self.overlap_sentences}"


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_BOUNDARY_RE.split(text.strip()) if part]


def _word_count(units: list[str]) -> int:
    return sum(len(unit.split()) for unit in units)


def _units(text: str, max_words: int) -> list[str]:
    """Sentences, with any sentence longer than max_words split into word windows."""

    units: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        units.extend(
            " ".join(words[start : start + max_words])
            for start in range(0, len(words), max_words)
        )
    return units


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Pack whole sentences into chunks of at most max_words words (title excluded).

    Consecutive chunks share the last overlap_sentences sentences when they fit.
    Documents within the limit stay a single chunk.
    """

    text = document.text.strip()
    if not text and not document.title:
        return []
    if config.max_words is None or len(text.split()) <= config.max_words:
        groups = [[text]]
    else:
        groups = []
        current: list[str] = []
        for unit in _units(text, config.max_words):
            if current and _word_count(current) + len(unit.split()) > config.max_words:
                groups.append(current)
                current = current[-config.overlap_sentences :] if config.overlap_sentences else []
                while current and _word_count(current) + len(unit.split()) > config.max_words:
                    current = current[1:]
            current.append(unit)
        groups.append(current)
    return [
        Chunk(
            doc_id=document.id,
            index=index,
            text=" ".join(group),
            title=document.title,
            metadata=document.metadata,
        )
        for index, group in enumerate(groups)
    ]


def chunk_documents(documents: list[Document], config: ChunkingConfig) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, config)]
