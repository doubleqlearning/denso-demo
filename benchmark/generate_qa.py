"""Steps 5–6 — generate Q&A from selected gold evidence, then build the benchmark splits.

    uv run python benchmark/generate_qa.py            # plan slots, generate, validate, build
    uv run python benchmark/generate_qa.py --build    # only rebuild qa/train/test/public from drafts

Evidence is selected first (doc chunk and/or expert turns), then the LLM writes a question, a
reference answer and a multiple-choice version from that evidence only. Slots are planned to hit the
knowledge_source × difficulty targets exactly (10% over-generation absorbs rejects). Resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from common import (CHUNKS_PATH, CONVERSATIONS_PATH, DOCUMENTS_PATH, GENERATION, MANIFEST_PATH, PUBLIC, QA_DRAFTS_PATH,
                    QA_PATH, SCENARIOS_PATH, TEST_PATH, TRAIN_PATH, append_jsonl, content_words, load_prompt,
                    normalize, overlap, prompt_hash, read_jsonl, render, write_jsonl)
from llm_client import LLMClient, run_all

QA_SLOTS_PATH = GENERATION / "qa_slots.jsonl"

TARGET = {
    ("expert_only", "easy"): 100, ("expert_only", "medium"): 70, ("expert_only", "hard"): 30,
    ("docs_only", "easy"): 80, ("docs_only", "medium"): 50, ("docs_only", "hard"): 20,
    ("expert_and_docs", "easy"): 20, ("expert_and_docs", "medium"): 80, ("expert_and_docs", "hard"): 50,
}
OVERGENERATE = 1.10
# (doc units, expert facts) per cell; hard cells fall back to the second pattern when needed.
PATTERNS = {
    ("expert_only", "easy"): [(0, 1)], ("expert_only", "medium"): [(0, 1)], ("expert_only", "hard"): [(0, 2)],
    ("docs_only", "easy"): [(1, 0)], ("docs_only", "medium"): [(1, 0)], ("docs_only", "hard"): [(2, 0), (1, 0)],
    ("expert_and_docs", "easy"): [(1, 1)], ("expert_and_docs", "medium"): [(1, 1)],
    ("expert_and_docs", "hard"): [(1, 2), (2, 1)],
}
TEST_RATIO = 0.8
QUESTION_TYPES = {"factual", "procedure", "diagnosis", "causal", "comparison"}
LETTERS = "ABCD"

SOURCE_INSTRUCTIONS = {
    "expert_only": "ask about the site-specific know-how of the expert for this asset. Mention the asset id "
                   "({asset_id}). The answer must come from the expert's statements, not general manual knowledge.",
    "docs_only": "ask a general question about this equipment model that the manual excerpt answers. Do NOT "
                 "mention any site, line or asset id — the question is about the product in general.",
    "expert_and_docs": "the question is about this asset's situation and needs BOTH the manual's official "
                       "information and the expert's site know-how to be fully answered. Mention the asset id "
                       "({asset_id}). The reference answer must combine both.",
}
DIFFICULTY_INSTRUCTIONS = {
    "easy": "ask directly about one stated fact; the answer is stated almost explicitly in the evidence.",
    "medium": "describe the situation by its symptoms or goal in different words than the evidence; the answer "
              "requires linking the symptom to the right action, cause or value.",
    "hard": "the answer requires combining ALL evidence units (multi-hop), or handling a condition, an exception, "
            "the order of steps, or the corrected advice rather than an initial wrong idea. Someone who sees only "
            "one evidence unit must not be able to answer fully.",
}
BANNED_PHRASES = re.compile(r"\b(conversation|excerpt|evidence|the text|transcript|the expert said|"
                            r"according to the (manual|expert|document))\b", re.I)


# ---------------------------------------------------------------- slot planning


class Planner:
    """Assigns evidence to slots: balanced over conversations, no evidence set used twice."""

    def __init__(self, conversations: list[dict], scenarios: dict[str, dict], existing: list[dict], seed: int):
        self.rng = random.Random(seed + len(existing))
        self.convs = conversations
        self.scenarios = scenarios
        self.n_slots: Counter[str] = Counter()
        self.fact_use: dict[str, Counter[str]] = defaultdict(Counter)
        self.doc_use: Counter[str] = Counter()  # by doc unit key, all sources
        self.docs_only_units: set[str] = set()
        self.docs_only_chunks: Counter[str] = Counter()
        self.signatures: set[tuple] = set()
        for slot in existing:
            self._register(slot)

    @staticmethod
    def unit_key(unit: dict) -> str:
        return unit["chunk_id"] + "|" + normalize(unit["quote"])[:80]

    def _register(self, slot: dict) -> None:
        self.n_slots[slot["conversation_id"]] += 1
        for f in slot["fact_ids"]:
            self.fact_use[f][slot["knowledge_source"]] += 1
        for u in slot["doc_units"]:
            self.doc_use[self.unit_key(u)] += 1
            if slot["knowledge_source"] == "docs_only":
                self.docs_only_units.add(self.unit_key(u))
                self.docs_only_chunks[u["chunk_id"]] += 1
        self.signatures.add(self._signature(slot["knowledge_source"], slot["doc_units"], slot["fact_ids"]))

    def _signature(self, source: str, units: list[dict], facts: list[str]) -> tuple:
        return (source, tuple(sorted(self.unit_key(u) for u in units)), tuple(sorted(facts)))

    def _facts(self, conv: dict, k: int, source: str) -> list[str] | None:
        ids = list(conv["fact_turns"])
        free = [f for f in ids if self.fact_use[f][source] == 0 and sum(self.fact_use[f].values()) < 2]
        if len(free) < k:
            return None
        self.rng.shuffle(free)
        free.sort(key=lambda f: sum(self.fact_use[f].values()))
        return free[:k]

    def _docs(self, scenario: dict, k: int, source: str) -> list[dict] | None:
        units = [{"chunk_id": e["chunk_id"], "quote": e["quote"], "fact": e["fact"]}
                 for e in scenario["source_evidence"]]
        if source == "docs_only":
            units = [u for u in units if self.unit_key(u) not in self.docs_only_units
                     and self.docs_only_chunks[u["chunk_id"]] < 2]
            self.rng.shuffle(units)
            units.sort(key=lambda u: self.docs_only_chunks[u["chunk_id"]])
        else:
            self.rng.shuffle(units)
            units.sort(key=lambda u: self.doc_use[self.unit_key(u)])
        if k == 2:
            # Two units must come from different chunks to be a real two-source question.
            for i, a in enumerate(units):
                for b in units[i + 1:]:
                    if a["chunk_id"] != b["chunk_id"]:
                        return [a, b]
            return None
        return units[:k] if len(units) >= k else None

    def plan(self, needs: dict[tuple[str, str], int], next_index: int) -> list[dict]:
        slots = []
        max_per_conv = math.ceil((sum(self.n_slots.values()) + sum(needs.values())) / max(len(self.convs), 1)) + 1
        order = sorted(needs, key=lambda c: ["hard", "medium", "easy"].index(c[1]))
        for cell in order:
            source, difficulty = cell
            for _ in range(needs[cell]):
                convs = self.convs[:]
                self.rng.shuffle(convs)
                convs.sort(key=lambda c: self.n_slots[c["conversation_id"]])
                slot = None
                for conv in convs:
                    if self.n_slots[conv["conversation_id"]] >= max_per_conv:
                        continue
                    scenario = self.scenarios[conv["scenario_id"]]
                    for n_docs, n_facts in PATTERNS[cell]:
                        facts = self._facts(conv, n_facts, source) if n_facts else []
                        units = self._docs(scenario, n_docs, source) if n_docs else []
                        if facts is None or units is None:
                            continue
                        if self._signature(source, units, facts) in self.signatures:
                            continue
                        slot = {
                            "slot_id": f"slot_{next_index:04d}",
                            "knowledge_source": source,
                            "difficulty": difficulty,
                            "conversation_id": conv["conversation_id"],
                            "scenario_id": conv["scenario_id"],
                            "group_id": scenario["group_id"],
                            "fact_ids": facts,
                            "doc_units": units,
                            "correct_letter": LETTERS[next_index % 4],
                        }
                        break
                    if slot:
                        break
                if slot is None:
                    print(f"[plan] no evidence left for {cell}; generate more scenarios/conversations")
                    break
                self._register(slot)
                slots.append(slot)
                next_index += 1
        return slots


def cell_of(row: dict) -> tuple[str, str]:
    return row["knowledge_source"], row["difficulty"]


def update_plan(conversations: list[dict], scenarios: dict[str, dict], scale: float, seed: int) -> list[dict]:
    slots = read_jsonl(QA_SLOTS_PATH)
    drafts = {d["slot_id"]: d for d in read_jsonl(QA_DRAFTS_PATH)}
    accepted, pending = Counter(), Counter()
    for s in slots:
        draft = drafts.get(s["slot_id"])
        if draft is None:
            pending[cell_of(s)] += 1
        elif draft["status"] == "accepted":
            accepted[cell_of(s)] += 1
    needs = {}
    for cell, target in TARGET.items():
        want = math.ceil(round(target * scale) * OVERGENERATE)
        if accepted[cell] < round(target * scale):
            needs[cell] = max(0, want - accepted[cell] - pending[cell])
    needs = {c: n for c, n in needs.items() if n}
    if needs:
        planner = Planner(conversations, scenarios, slots, seed)
        new = planner.plan(needs, next_index=len(slots) + 1)
        for s in new:
            append_jsonl(QA_SLOTS_PATH, s)
        slots += new
        print(f"[plan] added {len(new)} slots: {dict(Counter(f'{a}/{b}' for a, b in map(cell_of, new)))}")
    return slots


# ---------------------------------------------------------------- generation


def evidence_units(slot: dict, conv: dict, scenario: dict, chunks: dict[str, dict]) -> list[dict]:
    """Gold evidence units with the text shown to the generator."""
    units = []
    for u in slot["doc_units"]:
        c = chunks[u["chunk_id"]]
        units.append({
            "type": "document", "id": c["chunk_id"], "doc_id": c["doc_id"],
            "pages": [c["page_start"], c["page_end"]],
            "prompt": (f'[document {c["chunk_id"]}] section: {" > ".join(c["section_path"])} '
                       f'(pages {c["page_start"]}-{c["page_end"]})\nKey statement: "{u["quote"]}"\n'
                       f'Full excerpt:\n{c["text"]}'),
            "text": u["quote"] + " " + u["fact"],
        })
    turns = {t["turn_id"]: t for t in conv["turns"]}
    roles = {s["id"]: s["role"].replace("_", " ") for s in conv["speakers"]}
    facts = {f["fact_id"]: f for f in scenario["expert_facts"]}
    for fid in slot["fact_ids"]:
        mapped = conv["fact_turns"][fid]
        shown = sorted(set(mapped) | {m - 1 for m in mapped if m - 1 in turns})
        lines = [f'{"" if t in mapped else "(context) "}turn {t} — {roles[turns[t]["speaker"]]}: {turns[t]["text"]}'
                 for t in shown]
        units.append({
            "type": "conversation", "id": conv["conversation_id"], "turn_ids": mapped, "fact_id": fid,
            "prompt": f'[conversation {conv["conversation_id"]}]\n' + "\n".join(lines)
                      + f'\nKnow-how stated: "{facts[fid]["fact"]}"',
            "text": " ".join(turns[t]["text"] for t in mapped) + " " + facts[fid]["fact"],
        })
    return units


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def validate(qa: dict, slot: dict, units: list[dict], anchor: dict) -> list[str]:
    errors = []
    for k in ("question", "question_type", "reference_answer", "answer_key_points", "correct_option", "distractors"):
        if not qa.get(k):
            errors.append(f"missing {k}")
    if errors:
        return errors
    q, answer = qa["question"], qa["reference_answer"]
    if qa["question_type"] not in QUESTION_TYPES:
        errors.append(f"bad question_type {qa['question_type']}")
    if not 6 <= len(q.split()) <= 50:
        errors.append("question must be 10-40 words")
    if BANNED_PHRASES.search(q):
        errors.append("question refers to the evidence itself")
    evidence_text = " ".join(u["text"] for u in units)
    full_text = evidence_text + " " + " ".join(u["prompt"] for u in units)
    if ngrams(q, 10) & ngrams(full_text, 10):
        errors.append("question copies a long phrase from the evidence")
    asset = anchor["asset_id"].lower()
    if slot["knowledge_source"] == "docs_only":
        if asset in q.lower() or anchor["line"].lower() in q.lower():
            errors.append("docs_only question must not mention the site or asset")
    elif asset not in q.lower():
        errors.append(f"question must mention the asset id {anchor['asset_id']}")
    if overlap(answer, full_text) < 0.4:
        errors.append("reference answer is not grounded in the evidence")
    for u in units:  # every evidence unit must contribute to the answer
        if len((content_words(answer) | content_words(" ".join(qa["answer_key_points"])))
               & content_words(u["text"])) < 3:
            errors.append(f"answer does not use evidence {u['id']}")
    if not 1 <= len(qa["answer_key_points"]) <= 4:
        errors.append("answer_key_points must have 1-4 items")
    options = [qa["correct_option"]] + list(qa["distractors"])
    if len(qa["distractors"]) != 3 or len({normalize(o) for o in options}) != 4 or not all(o.strip() for o in options):
        errors.append("need 3 distinct non-empty distractors")
    else:
        if any(overlap(d, qa["correct_option"]) > 0.9 and overlap(qa["correct_option"], d) > 0.9
               for d in qa["distractors"]):
            errors.append("a distractor is a paraphrase of the correct option")
        longest = max(len(d.split()) for d in qa["distractors"])
        if len(qa["correct_option"].split()) > longest + 1:
            errors.append("the correct option is the longest option; make at least one distractor as long and "
                          "as specific as the correct option")
    return errors


async def generate_slot(llm: LLMClient, slot: dict, conv: dict, scenario: dict, chunks: dict[str, dict],
                        attempts: int) -> dict:
    units = evidence_units(slot, conv, scenario, chunks)
    a = scenario["anchor"]
    source = slot["knowledge_source"]
    prompt = render(
        load_prompt("generate_qa.md"),
        machine=scenario["machine"],
        anchor=(f"{a['asset_id']} at {a['site']}, {a['line']}, {a['station']}" if source != "docs_only"
                else "(general product question)"),
        evidence="\n\n".join(u["prompt"] for u in units),
        knowledge_source=source,
        source_instructions=SOURCE_INSTRUCTIONS[source].format(asset_id=a["asset_id"]),
        difficulty=slot["difficulty"],
        difficulty_instructions=DIFFICULTY_INSTRUCTIONS[slot["difficulty"]],
        misconceptions="; ".join(scenario["misconceptions"]) if source != "docs_only" else "(none given)",
    )
    errors: list[str] = []
    qa: dict = {}
    for attempt in range(1, attempts + 1):
        messages = [{"role": "user", "content": prompt}]
        if errors:
            messages.append({"role": "user", "content": "Your previous answer was rejected:\n- " + "\n- ".join(errors)
                             + "\nReturn a corrected JSON object."})
        qa = await llm.chat_json(messages, temperature=0.7)
        qa = qa if isinstance(qa, dict) else {}
        errors = validate(qa, slot, units, a)
        if not errors:
            return {"slot_id": slot["slot_id"], "status": "accepted", "attempts": attempt, "qa": qa,
                    "evidence": [{k: v for k, v in u.items() if k not in ("prompt", "text")} for u in units]}
    return {"slot_id": slot["slot_id"], "status": "rejected", "attempts": attempts, "errors": errors, "qa": qa}


# ---------------------------------------------------------------- build


def required_evidence(evidence: list[dict]) -> list[dict]:
    out, seen = [], set()
    for e in evidence:
        if e["type"] == "document":
            key = ("document", e["id"])
            if key not in seen:
                seen.add(key)
                out.append({"type": "document", "id": e["id"], "doc_id": e["doc_id"], "pages": e["pages"]})
        else:
            for t in e["turn_ids"]:
                key = ("conversation", e["id"], t)
                if key not in seen:
                    seen.add(key)
                    out.append({"type": "conversation", "id": e["id"], "turn_id": t})
    return out


def split_groups(questions: list[dict], seed: int) -> set[str]:
    """Group ids for train: whole groups (shared chunks + their conversations), ~20% of every cell."""
    rng = random.Random(seed)
    per_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    totals = Counter()
    for q in questions:
        per_group[q["group_id"]].append(cell_of(q))
        totals[cell_of(q)] += 1
    caps = {c: round(n * (1 - TEST_RATIO)) for c, n in totals.items()}
    counts: Counter = Counter()
    train: set[str] = set()
    groups = sorted(per_group)
    rng.shuffle(groups)
    for g in groups:
        need = Counter(per_group[g])
        if sum(counts.values()) + len(per_group[g]) <= sum(caps.values()) and \
                all(counts[c] + n <= caps[c] + 1 for c, n in need.items()):
            train.add(g)
            counts.update(need)
    return train


def build(seed: int, scale: float) -> None:
    slots = {s["slot_id"]: s for s in read_jsonl(QA_SLOTS_PATH)}
    drafts = [d for d in read_jsonl(QA_DRAFTS_PATH) if d["status"] == "accepted" and d["slot_id"] in slots]
    scenarios = {s["scenario_id"]: s for s in read_jsonl(SCENARIOS_PATH)}
    conversations = read_jsonl(CONVERSATIONS_PATH)
    chunks = read_jsonl(CHUNKS_PATH)

    # Drop near-duplicate questions, then keep exactly the target per cell (planned order).
    kept, seen_q = [], []
    for d in sorted(drafts, key=lambda d: d["slot_id"]):
        q = d["qa"]["question"]
        if any(overlap(q, o) > 0.85 and overlap(o, q) > 0.85 for o in seen_q):
            continue
        seen_q.append(q)
        kept.append(d)
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for d in kept:
        by_cell[cell_of(slots[d["slot_id"]])].append(d)
    selected = []
    for cell, target in TARGET.items():
        want = round(target * scale)
        have = by_cell.get(cell, [])
        if len(have) < want:
            print(f"[build] WARNING {cell[0]}/{cell[1]}: {len(have)}/{want} questions")
        selected += have[:want]

    questions = []
    for d in sorted(selected, key=lambda d: (slots[d["slot_id"]]["conversation_id"], d["slot_id"])):
        slot, qa = slots[d["slot_id"]], d["qa"]
        rng = random.Random(f"{seed}-{slot['slot_id']}")
        distractors = list(qa["distractors"])
        rng.shuffle(distractors)
        letter = slot["correct_letter"]
        options, it = {}, iter(distractors)
        for L in LETTERS:
            options[L] = qa["correct_option"] if L == letter else next(it)
        questions.append({
            "question_id": f"q_{len(questions) + 1:04d}",
            "scenario_id": slot["scenario_id"],
            "conversation_id": slot["conversation_id"],
            "group_id": slot["group_id"],
            "question": qa["question"],
            "question_type": qa["question_type"],
            "difficulty": slot["difficulty"],
            "knowledge_source": slot["knowledge_source"],
            "required_evidence": required_evidence(d["evidence"]),
            "gold_fact_ids": slot["fact_ids"],
            "reference_answer": qa["reference_answer"],
            "answer_key_points": qa["answer_key_points"],
            "options": options,
            "correct_option": letter,
        })

    train_groups = split_groups(questions, seed)
    for q in questions:
        q["split"] = "train" if q["group_id"] in train_groups else "test"
    train = [q for q in questions if q["split"] == "train"]
    test = [q for q in questions if q["split"] == "test"]
    write_jsonl(QA_PATH, questions)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(TEST_PATH, test)

    # Public inputs for the system under test: no scenario, facts, fact_turns or answers.
    write_jsonl(PUBLIC / "corpus_chunks.jsonl", [
        {k: c[k] for k in ("chunk_id", "doc_id", "section_path", "page_start", "page_end", "text")} for c in chunks])
    write_jsonl(PUBLIC / "conversations.jsonl", [
        {"conversation_id": c["conversation_id"],
         "speakers": [{"id": s["id"], "role": s["role"]} for s in c["speakers"]],
         "turns": c["turns"]} for c in conversations])
    for name, rows in (("questions_train.jsonl", train), ("questions_test.jsonl", test)):
        write_jsonl(PUBLIC / name, [{"question_id": q["question_id"], "question": q["question"],
                                     "options": q["options"]} for q in rows])

    convs_used = {q["conversation_id"] for q in questions}
    train_convs = {q["conversation_id"] for q in train}
    test_convs = {q["conversation_id"] for q in test}
    assert not train_convs & test_convs, "conversation leaked across splits"
    n_turns = [c["n_turns"] for c in conversations]
    n_words = [c["n_words"] for c in conversations]

    def table(rows: list[dict]) -> dict:
        return {f"{a}/{b}": n for (a, b), n in sorted(Counter(map(cell_of, rows)).items())}

    manifest = {
        "name": "factory-expert-knowhow-benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": "en",
        "seed": seed,
        "scale": scale,
        "documents": [{k: d[k] for k in ("doc_id", "file", "title", "n_pages", "n_chunks")}
                      for d in read_jsonl(DOCUMENTS_PATH)],
        "n_chunks": len(chunks),
        "n_scenarios": len(scenarios),
        "n_conversations": len(conversations),
        "n_conversations_with_questions": len(convs_used),
        "conversation_turns": {"min": min(n_turns), "mean": round(sum(n_turns) / len(n_turns), 1),
                               "max": max(n_turns)} if n_turns else {},
        "conversation_words": {"min": min(n_words), "mean": round(sum(n_words) / len(n_words), 1),
                               "max": max(n_words)} if n_words else {},
        "conversation_controls": {k: dict(Counter(str(c["controls"][k]) for c in conversations))
                                  for k in ("length", "difficulty", "num_speakers", "irrelevant_ratio")},
        "n_questions": len(questions),
        "n_train": len(train),
        "n_test": len(test),
        "knowledge_source": dict(Counter(q["knowledge_source"] for q in questions)),
        "difficulty": dict(Counter(q["difficulty"] for q in questions)),
        "cells": {"all": table(questions), "train": table(train), "test": table(test)},
        "question_type": dict(Counter(q["question_type"] for q in questions)),
        "correct_option": dict(sorted(Counter(q["correct_option"] for q in questions).items())),
        # Share of questions whose correct option has strictly more words than every distractor (chance ≈ 0.25).
        "correct_option_strictly_longest": round(sum(
            all(len(q["options"][q["correct_option"]].split()) > len(o.split())
                for letter, o in q["options"].items() if letter != q["correct_option"])
            for q in questions) / max(len(questions), 1), 3),
        "questions_per_doc": dict(Counter(scenarios[q["scenario_id"]]["doc_id"] for q in questions)),
        "prompts": {p: prompt_hash(p) for p in ("generate_scenario.md", "generate_conversation.md",
                                                "generate_qa.md")},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[build] {len(questions)} questions ({len(train)} train / {len(test)} test) from "
          f"{len(convs_used)} conversations")
    print(json.dumps(manifest["cells"]["all"], indent=1))


async def main_async(args: argparse.Namespace) -> None:
    if not args.build:
        scenarios = {s["scenario_id"]: s for s in read_jsonl(SCENARIOS_PATH)}
        conversations = read_jsonl(CONVERSATIONS_PATH)
        chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS_PATH)}
        if not conversations:
            raise SystemExit("Run generate_conversations.py first")
        slots = update_plan(conversations, scenarios, args.target_scale, args.seed)
        done = {d["slot_id"] for d in read_jsonl(QA_DRAFTS_PATH)}
        convs = {c["conversation_id"]: c for c in conversations}
        todo = [s for s in slots if s["slot_id"] not in done]
        print(f"{len(slots)} slots, {len(todo)} to generate")
        if todo:
            llm = LLMClient()
            jobs = [lambda s=s: generate_slot(llm, s, convs[s["conversation_id"]], scenarios[s["scenario_id"]],
                                              chunks, args.attempts) for s in todo]
            finished = await run_all(jobs, lambda r: append_jsonl(QA_DRAFTS_PATH, r), label="qa")
            drafts = read_jsonl(QA_DRAFTS_PATH)
            print(f"qa drafts: {sum(d['status'] == 'accepted' for d in drafts)} accepted, "
                  f"{sum(d['status'] == 'rejected' for d in drafts)} rejected; LLM calls {llm.calls}, "
                  f"tokens {llm.tokens}")
            if not finished:
                return
            # Rejected slots leave gaps: plan replacements once and generate them.
            slots = update_plan(conversations, scenarios, args.target_scale, args.seed)
            done = {d["slot_id"] for d in read_jsonl(QA_DRAFTS_PATH)}
            todo = [s for s in slots if s["slot_id"] not in done]
            if todo:
                jobs = [lambda s=s: generate_slot(llm, s, convs[s["conversation_id"]],
                                                  scenarios[s["scenario_id"]], chunks, args.attempts)
                        for s in todo]
                if not await run_all(jobs, lambda r: append_jsonl(QA_DRAFTS_PATH, r), label="qa-refill"):
                    return
    build(args.seed, args.target_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-scale", type=float, default=1.0, help="scale the 500-question target (pilot)")
    parser.add_argument("--attempts", type=int, default=2, help="generation + regenerations per slot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build", action="store_true", help="only rebuild outputs from existing drafts")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
