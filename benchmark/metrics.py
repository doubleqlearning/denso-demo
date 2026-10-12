"""Pure metric functions (no IO, no LLM). Used by evaluate.py."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable


def parse_item(item: dict | str) -> dict:
    """Accept {"type": ..., ...} dicts, a chunk_id string, or "conv_id#turn_id"."""
    if isinstance(item, dict):
        return item
    if "#" in item:
        cid, turn = item.rsplit("#", 1)
        return {"type": "conversation", "id": cid, "turn_id": int(turn)}
    return {"type": "document", "id": item}


def is_hit(item: dict, gold: dict) -> bool:
    """A retrieved item matches a gold evidence unit.

    Documents match by exact chunk_id, or by doc_id + a page inside the gold chunk's page range (for
    systems that chunk the manuals themselves). Conversation evidence matches by conversation + turn.
    """
    if gold["type"] == "document":
        if item.get("type", "document") != "document":
            return False
        if item.get("id") == gold["id"]:
            return True
        page = item.get("page")
        return (item.get("doc_id") == gold["doc_id"] and page is not None
                and gold["pages"][0] <= int(page) <= gold["pages"][1])
    return (item.get("type") == "conversation" and item.get("id") == gold["id"]
            and item.get("turn_id") == gold["turn_id"])


def recall_at_k(retrieved: list[dict], gold: list[dict], k: int) -> float:
    if not gold:
        return 0.0
    top = retrieved[:k]
    return sum(any(is_hit(r, g) for r in top) for g in gold) / len(gold)


def context_precision_at_k(retrieved: list[dict], gold: list[dict], k: int) -> float:
    """RAGAS-style: mean of precision@i over the ranks i ≤ k that hold a relevant item."""
    hits, total = 0, 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if any(is_hit(r, g) for g in gold):
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def answer_score(judgement: dict, n_points: int) -> tuple[float, bool]:
    """Key-point coverage in [0, 1] (partial = 0.5) and the binary 'correct' verdict."""
    weights = {"covered": 1.0, "partial": 0.5}
    statuses = [p.get("status", "missing") for p in judgement.get("key_points", [])][:n_points]
    score = sum(weights.get(s, 0.0) for s in statuses) / max(n_points, 1)
    contradiction = bool(judgement.get("contradiction")) or "contradicted" in statuses
    return score, score >= 0.8 and not contradiction


def fact_precision_recall(judgement: dict, n_extracted: int, gold_expert_ids: list[str],
                          gold_ids: list[str]) -> dict:
    """Precision: extracted facts that match any gold fact. Recall: expert facts covered by extraction."""
    matched_extracted = {e["index"] for e in judgement.get("extracted", [])
                         if e.get("matches") and set(e["matches"]) & set(gold_ids)}
    covered = {g for e in judgement.get("extracted", []) for g in (e.get("matches") or [])}
    covered_expert = [g for g in gold_expert_ids if g in covered]
    return {
        "precision": len(matched_extracted) / n_extracted if n_extracted else 0.0,
        "recall": len(covered_expert) / len(gold_expert_ids) if gold_expert_ids else 0.0,
        "covered_expert_ids": covered_expert,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def grouped_means(rows: list[dict], key: Callable[[dict], str], fields: list[str]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {g: {"n": len(rs), **{f: mean(r[f] for r in rs if f in r) for f in fields}}
            for g, rs in sorted(groups.items())}
