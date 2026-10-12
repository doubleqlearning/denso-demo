"""Score a system's predictions against the hidden gold labels.

    uv run python benchmark/evaluate.py --predictions run.jsonl --split test --out results.json
    uv run python benchmark/evaluate.py --predictions run.jsonl --skip-llm     # retrieval + MC only
    uv run python benchmark/evaluate.py --predictions run.jsonl --facts        # + fact precision/recall
    uv run python benchmark/evaluate.py --predictions run.jsonl --question-ids ids.txt  # a sample only

Predictions JSONL (one object per line):
    {"type": "answer", "question_id": "q_0001", "answer": "...", "choice": "B",
     "retrieved": [{"type": "document", "id": "<chunk_id>"},
                   {"type": "document", "doc_id": "omron_i4l", "page": 141},
                   {"type": "conversation", "id": "conv_001_1", "turn_id": 4}]}
    {"type": "extraction", "conversation_id": "conv_001_1", "facts": ["...", "..."]}
Missing predictions score 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from common import CONVERSATIONS_PATH, QA_PATH, SCENARIOS_PATH, TEST_PATH, TRAIN_PATH, load_prompt, read_jsonl, render
from llm_client import LLMClient, run_all
from metrics import (answer_score, context_precision_at_k, fact_precision_recall, grouped_means, mean, parse_item,
                     recall_at_k)

SPLITS = {"test": TEST_PATH, "train": TRAIN_PATH, "all": QA_PATH}


async def judge_answers(llm: LLMClient, questions: list[dict], answers: dict[str, str]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    async def judge(q: dict) -> tuple[str, dict]:
        prompt = render(
            load_prompt("judge_answer.md"),
            question=q["question"],
            reference_answer=q["reference_answer"],
            key_points="\n".join(f"{i}. {p}" for i, p in enumerate(q["answer_key_points"], start=1)),
            answer=answers[q["question_id"]],
        )
        return q["question_id"], await llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)

    jobs = [lambda q=q: judge(q) for q in questions if answers.get(q["question_id"], "").strip()]
    await run_all(jobs, lambda r: results.__setitem__(r[0], r[1]), label="judge-answer")
    return results


async def judge_facts(llm: LLMClient, conversation_ids: list[str], extracted: dict[str, list[str]],
                      scenarios: dict[str, dict], conv_scenario: dict[str, str]) -> list[dict]:
    rows: list[dict] = []

    async def judge(cid: str) -> dict:
        s = scenarios[conv_scenario[cid]]
        gold = ([(f["fact_id"], f["fact"]) for f in s["expert_facts"]]
                + [(f"{s['scenario_id']}_obs{i}", o) for i, o in enumerate(s["observations"], start=1)]
                + [(f"{s['scenario_id']}_doc{i}", e["fact"]) for i, e in enumerate(s["source_evidence"], start=1)])
        facts = extracted.get(cid, [])
        expert_ids = [f["fact_id"] for f in s["expert_facts"]]
        if not facts:
            return {"conversation_id": cid, "precision": 0.0, "recall": 0.0, "n_extracted": 0,
                    "covered_expert_ids": [], "expert_types": {f["fact_id"]: f["type"] for f in s["expert_facts"]}}
        prompt = render(load_prompt("judge_facts.md"),
                        gold_facts="\n".join(f"[{gid}] {text}" for gid, text in gold),
                        extracted_facts="\n".join(f"[{i}] {f}" for i, f in enumerate(facts)))
        judgement = await llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
        scores = fact_precision_recall(judgement, len(facts), expert_ids, [g for g, _ in gold])
        return {"conversation_id": cid, "n_extracted": len(facts), **scores,
                "expert_types": {f["fact_id"]: f["type"] for f in s["expert_facts"]}}

    await run_all([lambda c=c: judge(c) for c in conversation_ids], rows.append, label="judge-facts")
    return rows


def markdown_table(title: str, table: dict[str, dict], fields: list[str]) -> str:
    lines = [f"\n### {title}", "| group | n | " + " | ".join(fields) + " |",
             "|---|---|" + "---|" * len(fields)]
    for group, row in table.items():
        lines.append(f"| {group} | {row['n']} | " + " | ".join(f"{row.get(f, 0):.3f}" for f in fields) + " |")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    questions = read_jsonl(SPLITS[args.split])
    if not questions:
        raise SystemExit("No gold questions; run generate_qa.py first")
    if args.question_ids:
        wanted = set(Path(args.question_ids).read_text(encoding="utf-8").split())
        questions = [q for q in questions if q["question_id"] in wanted]
        if missing := wanted - {q["question_id"] for q in questions}:
            raise SystemExit(f"Not in split {args.split}: {', '.join(sorted(missing))}")
    preds = read_jsonl(Path(args.predictions))
    answers = {p["question_id"]: p for p in preds if p.get("type", "answer") == "answer" and "question_id" in p}
    extracted = {p["conversation_id"]: p.get("facts", []) for p in preds if p.get("type") == "extraction"}
    ks = [int(k) for k in args.k.split(",")]

    rows = []
    for q in questions:
        pred = answers.get(q["question_id"], {})
        retrieved = [parse_item(r) for r in pred.get("retrieved", [])]
        row = {"question_id": q["question_id"], "knowledge_source": q["knowledge_source"],
               "difficulty": q["difficulty"]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(retrieved, q["required_evidence"], k)
            row[f"context_precision@{k}"] = context_precision_at_k(retrieved, q["required_evidence"], k)
        row["mc_accuracy"] = float(str(pred.get("choice", "")).strip().upper()[:1] == q["correct_option"])
        rows.append(row)

    llm = None
    if not args.skip_llm and any(a.get("answer") for a in answers.values()):
        llm = LLMClient()
        judgements = await judge_answers(llm, questions, {k: v.get("answer", "") for k, v in answers.items()})
        for q, row in zip(questions, rows):
            j = judgements.get(q["question_id"])
            score, correct = answer_score(j, len(q["answer_key_points"])) if j else (0.0, False)
            row["answer_score"], row["answer_correct"] = score, float(correct)
            row["judge"] = j

    fact_rows = []
    if args.facts:
        conversations = read_jsonl(CONVERSATIONS_PATH)
        scenarios = {s["scenario_id"]: s for s in read_jsonl(SCENARIOS_PATH)}
        conv_scenario = {c["conversation_id"]: c["scenario_id"] for c in conversations}
        subset = args.split != "all" or args.question_ids
        split_convs = sorted({q["conversation_id"] for q in questions}) if subset else sorted(conv_scenario)
        llm = llm or LLMClient()
        fact_rows = await judge_facts(llm, split_convs, extracted, scenarios, conv_scenario)

    fields = [f"recall@{k}" for k in ks] + [f"context_precision@{k}" for k in ks] + ["mc_accuracy"]
    if rows and "answer_score" in rows[0]:
        fields += ["answer_score", "answer_correct"]
    results = {
        "split": args.split,
        "n_questions": len(questions),
        "n_answered": sum(q["question_id"] in answers for q in questions),
        "overall": {f: mean(r[f] for r in rows if f in r) for f in fields},
        "by_knowledge_source": grouped_means(rows, lambda r: r["knowledge_source"], fields),
        "by_difficulty": grouped_means(rows, lambda r: r["difficulty"], fields),
        "by_cell": grouped_means(rows, lambda r: f"{r['knowledge_source']}/{r['difficulty']}", fields),
    }
    if fact_rows:
        by_type: dict[str, list[float]] = {}
        for r in fact_rows:
            for fid, ftype in r["expert_types"].items():
                by_type.setdefault(ftype, []).append(float(fid in r["covered_expert_ids"]))
        results["fact_capture"] = {
            "n_conversations": len(fact_rows),
            "precision": mean(r["precision"] for r in fact_rows),
            "recall": mean(r["recall"] for r in fact_rows),
            "recall_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        }
    results["per_question"] = rows
    if fact_rows:
        results["per_conversation_facts"] = fact_rows

    print(f"## Results ({args.split}: {results['n_answered']}/{len(questions)} answered)")
    print(markdown_table("Overall", {"all": {"n": len(rows), **results["overall"]}}, fields))
    print(markdown_table("By knowledge source", results["by_knowledge_source"], fields))
    print(markdown_table("By difficulty", results["by_difficulty"], fields))
    if fact_rows:
        fc = results["fact_capture"]
        print(f"\n### Fact capture\nprecision {fc['precision']:.3f} | recall {fc['recall']:.3f} | "
              f"recall by type {fc['recall_by_type']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=list(SPLITS), default="test")
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--question-ids", help="file of question ids (whitespace-separated): score only these")
    parser.add_argument("--skip-llm", action="store_true", help="no answer judge (retrieval + MC only)")
    parser.add_argument("--facts", action="store_true", help="also judge fact precision/recall (LLM)")
    parser.add_argument("--out")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
