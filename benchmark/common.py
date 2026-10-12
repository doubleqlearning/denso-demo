"""Shared paths, JSONL helpers and light text utilities for the benchmark pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
PROCESSED = ROOT / "processed"
GENERATION = ROOT / "generation"
GOLD = ROOT / "benchmark"
PUBLIC = ROOT / "public"
PROMPTS = ROOT / "prompts"

DOCUMENTS_PATH = PROCESSED / "documents.jsonl"
CHUNKS_PATH = PROCESSED / "chunks.jsonl"
SCENARIOS_PATH = GENERATION / "scenarios.jsonl"
SCENARIO_REJECTS_PATH = GENERATION / "scenario_rejects.jsonl"
CONVERSATIONS_PATH = GENERATION / "conversations.jsonl"
VALIDATION_PATH = GENERATION / "validation.jsonl"
QA_DRAFTS_PATH = GENERATION / "qa_drafts.jsonl"
QA_PATH = GOLD / "qa.jsonl"
TRAIN_PATH = GOLD / "train.jsonl"
TEST_PATH = GOLD / "test.jsonl"
MANIFEST_PATH = GOLD / "manifest.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def render(template: str, **values: str) -> str:
    """Fill {{name}} placeholders; JSON braces in the template stay untouched."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def prompt_hash(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()[:12]


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- text matching

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into onto over under about as is are was were be
been being do does did done have has had having it its this that these those there here which who whom whose what
when where why how not no nor so than too very can could should would will shall may might must just also only
any all each every some such both either neither other same own more most less few many much one two i you he she
we they me him her us them my your his our their yes ok okay let lets s t ll ve re d m get got make made
""".split())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _stem(word: str) -> str:
    for suffix in ("ing", "edly", "ed", "ies", "es", "s", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.\-/]*[a-z0-9]|[a-z0-9]", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS}


def overlap(needle: str, haystack: str) -> float:
    """Share of the needle's content words that occur in the haystack."""
    words = content_words(needle)
    if not words:
        return 0.0
    return len(words & content_words(haystack)) / len(words)


def quote_in(quote: str, text: str) -> bool:
    """Verbatim check tolerant to whitespace, case and PDF ligature/quote differences."""
    def canon(s: str) -> str:
        s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("“", '"').replace("”", '"')
        s = s.replace("‑", "-").replace("–", "-").replace("—", "-")
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    q = canon(quote)
    return bool(q) and q in canon(text)

