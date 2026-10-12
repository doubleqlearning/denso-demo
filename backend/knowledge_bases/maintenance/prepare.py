"""Materialize the versioned synthetic automotive benchmark without using an LLM.

    python -m backend.knowledge_bases.maintenance.prepare

The tracked benchmark/cases.json is the source of truth. Gold cards and answers
are never included in the ingestion CSV. Original v1 sources remain archived.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from backend.knowledge_bases.base import DATA_DIR


SOURCE = Path(__file__).parent / "benchmark"
OUTPUT = DATA_DIR / "maintainance"  # Preserve the existing on-disk spelling.
VERSION = "automotive-maintenance-v2"
CONVERSATIONS = OUTPUT / "expert_knowledge_extraction_benchmark_40_conversations_en.csv"
QUESTIONS = OUTPUT / "retrieval_evaluation_questions_answers_v2.csv"
ANSWER_QUESTIONS = OUTPUT / "answer_accuracy_questions_v2.csv"
CARD_FIELDS = ("context", "observation", "reasoning", "action", "outcome", "evidence")


def read_source() -> dict:
    return json.loads((SOURCE / "cases.json").read_text(encoding="utf-8"))


def transcript(case: dict) -> str:
    return "\n".join(
        f'[{turn["turn_id"]}] {turn["speaker"]}: {turn["text"]}'
        for turn in case["turns"]
    )


def validate_source(source: dict) -> None:
    cases = source["cases"]
    ids = [case["case_id"] for case in cases]
    if len(ids) != 40 or set(ids) != {str(i) for i in range(1, 41)}:
        raise ValueError("Source must contain exactly cases 1 through 40 without duplicates")
    for case in cases:
        turns = case["turns"]
        lookup = {t["turn_id"]: t["text"] for t in turns}
        if len(lookup) != len(turns) or len(turns) < 8:
            raise ValueError(f'{case["case_id"]}: missing or duplicate turns')
        card = case["gold_card"]
        if set(card) != set(CARD_FIELDS) or any(not card[f] for f in CARD_FIELDS):
            raise ValueError(f'{case["case_id"]}: incomplete gold card')
        for evidence in card["evidence"]:
            if evidence["quote"] not in lookup.get(evidence["turn_id"], ""):
                raise ValueError(f'{case["case_id"]}: invalid gold evidence')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def questions(source: dict) -> list[dict]:
    with (SOURCE / "legacy_questions.csv").open(encoding="utf-8-sig") as handle:
        legacy = list(csv.DictReader(handle))
    # Broad v1 questions need new corpus-wide judgements; do not leave new
    # relevant cases falsely labelled as distractors.
    updates = {
        "42": "1;2;13;14",
        "43": "1;2;4;5;6;8;9;10;12;13;16;17;18;19;21;22;23;24;26;27;29;30;31;32;33;34;35;36;37;38;39;40",
        "44": "8;9;10;22;23;32",
    }
    cases = {c["case_id"]: c for c in source["cases"]}
    rows = []
    for row in legacy:
        row["gold_case_ids"] = updates.get(row["question_id"], row["gold_case_ids"])
        if row["question_id"] in updates:
            field = "action" if row["question_id"] == "44" else "reasoning"
            row["sample_answer"] = " ".join(
                f'Case {cid}: {cases[cid]["gold_card"][field]}'
                for cid in row["gold_case_ids"].split(";")
            )
        row["answerable"] = "true"
        row["gold_evidence"] = ";".join(
            f'{cid}:{ev["turn_id"]}' for cid in row["gold_case_ids"].split(";")
            for ev in cases[cid]["gold_card"]["evidence"]
        )
        rows.append(row)
    rows.extend(source["case_questions"])
    rows.extend(source["challenge_questions"])
    seen = set()
    for row in rows:
        if row["question_id"] in seen:
            raise ValueError("Duplicate question id")
        seen.add(row["question_id"])
        gold = row["gold_case_ids"].split(";") if row["gold_case_ids"] else []
        if any(cid not in cases for cid in gold):
            raise ValueError("Question refers to a nonexistent case")
        if (row["answerable"] == "true") != bool(gold):
            raise ValueError("Answerability and qrels disagree")
        for ref in row["gold_evidence"].split(";") if row["gold_evidence"] else []:
            cid, tid = ref.split(":")
            if cid not in gold or tid not in {t["turn_id"] for t in cases[cid]["turns"]}:
                raise ValueError("Invalid question evidence turn")
        row["required_answer_facts"] = json.dumps(
            [s.strip() for s in row["sample_answer"].split(". ") if s.strip()], ensure_ascii=False,
        )
    return rows


def prepare() -> dict:
    source = read_source()
    validate_source(source)
    rows = [{
        "case_id": case["case_id"], "title": case["title"],
        "complexity": case["complexity"], "domain": case["domain"],
        "benchmark_version": VERSION, "synthetic": "true",
        "raw_conversation": transcript(case),
    } for case in source["cases"]]
    all_questions = questions(source)
    positive = [r for r in all_questions if r["answerable"] == "true"]
    write_csv(CONVERSATIONS, rows)
    write_csv(QUESTIONS, positive)
    write_csv(ANSWER_QUESTIONS, all_questions)
    counts = [sum(len(t["text"].split()) for t in case["turns"]) for case in source["cases"]]
    manifest = {
        "version": VERSION, "synthetic": True, "language": "en",
        "cases": len(rows), "retrieval_questions": len(positive),
        "answer_questions": len(all_questions), "unanswerable_questions": len(all_questions)-len(positive),
        "content_words": {"min": min(counts), "max": max(counts), "mean": sum(counts)/len(counts)},
        "turns": {"min": min(len(c["turns"]) for c in source["cases"]),
                  "max": max(len(c["turns"]) for c in source["cases"])},
    }
    (OUTPUT / "benchmark_v2_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
