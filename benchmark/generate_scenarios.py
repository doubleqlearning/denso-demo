"""Step 2 — generate scenario fact sheets from groups of linked manual chunks.

    uv run python benchmark/generate_scenarios.py --n-scenarios 200

Each scenario separates official evidence (verbatim quotes from chunks) from synthetic expert facts.
Rule validation rejects a scenario whose quotes are not found in the cited chunks. Resumable: groups
that already produced scenarios are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re

from common import (CHUNKS_PATH, DOCUMENTS_PATH, SCENARIO_REJECTS_PATH, SCENARIOS_PATH, append_jsonl, dumps,
                    load_prompt, overlap, quote_in, read_jsonl, render)
from llm_client import LLMClient, run_all

MACHINES = {
    "omron_i4l": ("OMRON i4L SCARA robot", "i4L-R"),
    "melservo_je": ("Mitsubishi MELSERVO-JE servo amplifier (MR-JE) with HG servo motor", "MR-JE-AX"),
    "fx5_comm": ("Mitsubishi MELSEC iQ-F FX5 PLC communication network", "FX5-PLC"),
}
DOC_SHARE = {"omron_i4l": 0.30, "melservo_je": 0.35, "fx5_comm": 0.35}
PRIORITY = re.compile(r"troubleshoot|error|alarm|maintenance|warning|fault|inspection|precaution", re.I)

FACT_TYPES = {"tacit_heuristic", "historical_experience", "workaround", "root_cause_hint", "safety_practice",
              "local_parameter"}
TOPICS = {"alarm", "error", "troubleshooting", "maintenance", "warning", "motor", "encoder", "motion", "safety",
          "communication"}

# Conversation controls, assigned here so the mix is balanced across the whole dataset.
LENGTHS = [("short", 30), ("medium", 45), ("long", 25)]
DIFFICULTIES = [("direct", 30), ("implicit", 25), ("correction_reference", 25), ("noisy", 20)]
SITES = ["Plant Hanoi", "Plant Bac Ninh", "Plant Hai Phong", "Plant Da Nang", "Plant Binh Duong", "Plant Dong Nai"]
AREAS = ["Assembly", "Packaging", "Machining", "Pick-and-place", "Inspection", "Palletizing", "Dispensing"]


def balanced(options: list[tuple[str, int]], n: int, rng: random.Random) -> list[str]:
    """Exactly n values in the given percentage mix, shuffled."""
    values = [v for v, w in options for _ in range(n * w // 100)]
    values += [options[i % len(options)][0] for i in range(n - len(values))]
    rng.shuffle(values)
    return values


def build_groups(chunks: list[dict]) -> dict[str, list[list[dict]]]:
    """Candidate groups per document: single chunks and adjacent pairs of the same section."""
    groups: dict[str, list[list[dict]]] = {}
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for doc_id, doc_chunks in by_doc.items():
        out = []
        for i, c in enumerate(doc_chunks):
            if not c["scenario_candidate"]:
                continue
            out.append([c])
            nxt = doc_chunks[i + 1] if i + 1 < len(doc_chunks) else None
            if (nxt and nxt["scenario_candidate"] and nxt["section_path"][:2] == c["section_path"][:2]
                    and c["n_tokens"] + nxt["n_tokens"] <= 1300):
                out.append([c, nxt])
        groups[doc_id] = out
    return groups


def plan_groups(chunks: list[dict], n_scenarios: int, seed: int) -> list[dict]:
    """Deterministic plan: which chunk groups to use, how many scenarios each, plus controls and anchors."""
    rng = random.Random(seed)
    groups = build_groups(chunks)
    plan = []
    for doc_id, share in DOC_SHARE.items():
        quota = round(n_scenarios * share)
        candidates = groups.get(doc_id, [])
        # Troubleshooting/maintenance sections first, then the rest; pairs and singles interleaved.
        rng.shuffle(candidates)
        candidates.sort(key=lambda g: 0 if PRIORITY.search(" ".join(g[0]["section_path"])) else 1)
        n_chunks_single = len({c["chunk_id"] for g in candidates for c in g})
        per_group = 2 if n_chunks_single < quota * 1.2 else 1
        used: set[str] = set()
        produced = 0
        for g in candidates:
            if produced >= quota:
                break
            ids = [c["chunk_id"] for c in g]
            if any(i in used for i in ids):
                continue  # each chunk seeds at most one group
            used.update(ids)
            k = min(per_group, quota - produced)
            plan.append({"doc_id": doc_id, "chunk_ids": ids, "n_scenarios": k})
            produced += k
    rng.shuffle(plan)

    n_total = sum(p["n_scenarios"] for p in plan)
    lengths = balanced(LENGTHS, n_total, rng)
    difficulties = balanced(DIFFICULTIES, n_total, rng)
    asset_numbers = rng.sample(range(100, 1000), n_total)
    k = 0
    for g_idx, p in enumerate(plan, start=1):
        p["group_id"] = f"g{g_idx:03d}"
        p["slots"] = []
        for j in range(p["n_scenarios"]):
            difficulty = difficulties[k]
            irrelevant = {"noisy": rng.choice([0.2, 0.4]), "direct": 0.0}.get(difficulty, rng.choice([0.0, 0.2]))
            machine, prefix = MACHINES[p["doc_id"]]
            line = rng.randint(1, 12)
            p["slots"].append({
                "scenario_id": f"scn_{g_idx:03d}_{j + 1}",
                "machine": machine,
                "anchor": {
                    "site": rng.choice(SITES),
                    "line": f"Line {line}",
                    "station": f"{rng.choice(AREAS)} cell {chr(65 + rng.randrange(6))}",
                    "asset_id": f"{prefix}{asset_numbers[k]}",
                },
                "conversation_controls": {
                    "length": lengths[k],
                    "difficulty": difficulty,
                    "num_speakers": 3 if rng.random() < 0.3 else 2,
                    "irrelevant_ratio": irrelevant,
                },
            })
            k += 1
    return plan


def chunk_block(c: dict) -> str:
    return (f'<chunk id="{c["chunk_id"]}" section="{" > ".join(c["section_path"])}" '
            f'pages="{c["page_start"]}-{c["page_end"]}">\n{c["text"]}\n</chunk>')


def validate(raw: dict, group: dict, chunks: dict[str, dict]) -> list[str]:
    errors = []
    for k in ("title", "topic", "source_evidence", "observations", "expert_facts", "misconceptions"):
        if not raw.get(k):
            errors.append(f"missing {k}")
    if errors:
        return errors
    if raw["topic"] not in TOPICS:
        errors.append(f"bad topic {raw['topic']}")
    if not 1 <= len(raw["source_evidence"]) <= 3:
        errors.append("source_evidence must have 1-3 items")
    for ev in raw["source_evidence"]:
        cid = ev.get("chunk_id")
        if cid not in group["chunk_ids"] or cid not in chunks:
            errors.append(f"unknown chunk_id {cid}")
        elif not quote_in(ev.get("quote", ""), chunks[cid]["text"]):
            errors.append(f"quote not found verbatim in {cid}: {ev.get('quote', '')[:80]!r}")
        elif len(ev.get("quote", "").split()) < 5 or not ev.get("fact"):
            errors.append("quote too short or fact missing")
    if not 2 <= len(raw["observations"]) <= 4:
        errors.append("observations must have 2-4 items")
    if not 2 <= len(raw["expert_facts"]) <= 4:
        errors.append("expert_facts must have 2-4 items")
    source = " ".join(chunks[c]["text"] for c in group["chunk_ids"] if c in chunks)
    for f in raw["expert_facts"]:
        if f.get("type") not in FACT_TYPES:
            errors.append(f"bad expert fact type {f.get('type')}")
        if not f.get("fact") or len(f["fact"].split()) > 45:
            errors.append("expert fact empty or too long")
        elif overlap(f["fact"], source) > 0.85:
            errors.append(f"expert fact restates the manual: {f['fact'][:80]!r}")
    return errors


def to_scenario(raw: dict, slot: dict, group: dict) -> dict:
    sid = slot["scenario_id"]
    return {
        "scenario_id": sid,
        "group_id": group["group_id"],
        "doc_id": group["doc_id"],
        "title": raw["title"],
        "topic": raw["topic"],
        "machine": slot["machine"],
        "anchor": slot["anchor"],
        "source_chunk_ids": group["chunk_ids"],
        "source_evidence": [{"chunk_id": e["chunk_id"], "quote": e["quote"], "fact": e["fact"]}
                            for e in raw["source_evidence"]],
        "observations": raw["observations"],
        "expert_facts": [{"fact_id": f"{sid}_ef{i}", "fact": f["fact"], "type": f["type"]}
                         for i, f in enumerate(raw["expert_facts"], start=1)],
        "misconceptions": raw["misconceptions"],
        "conversation_controls": slot["conversation_controls"],
    }


async def generate_group(llm: LLMClient, group: dict, chunks: dict[str, dict], titles: dict[str, str],
                         attempts: int) -> dict:
    slots = group["slots"]
    n = len(slots)
    anchor = "; ".join(
        f"scenario {i + 1}: {s['anchor']['asset_id']} at {s['anchor']['site']}, {s['anchor']['line']}, "
        f"{s['anchor']['station']}" for i, s in enumerate(slots))
    prompt = render(
        load_prompt("generate_scenario.md"),
        machine=slots[0]["machine"],
        manual_title=titles[group["doc_id"]],
        chunks="\n\n".join(chunk_block(chunks[c]) for c in group["chunk_ids"]),
        n_scenarios=str(n),
        diversity_note=("The two scenarios must be clearly different incidents (different symptom and different "
                        "expert know-how)." if n > 1 else ""),
        anchor=anchor,
    )
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        messages = [{"role": "user", "content": prompt}]
        if errors:
            messages.append({"role": "user", "content": "Your previous answer was rejected:\n- "
                             + "\n- ".join(errors[:8]) + "\nReturn a corrected JSON object."})
        data = await llm.chat_json(messages, temperature=0.8)
        raws = data.get("scenarios", []) if isinstance(data, dict) else []
        errors = [] if len(raws) == n else [f"expected {n} scenarios, got {len(raws)}"]
        for i, raw in enumerate(raws[:n]):
            errors += [f"scenario {i + 1}: {e}" for e in validate(raw, group, chunks)]
        if not errors:
            return {"group": group, "scenarios": [to_scenario(r, s, group) for r, s in zip(raws, slots)],
                    "attempts": attempt}
    return {"group": group, "scenarios": [], "errors": errors, "attempts": attempts}


async def main_async(args: argparse.Namespace) -> None:
    chunk_rows = read_jsonl(CHUNKS_PATH)
    if not chunk_rows:
        raise SystemExit("Run parse_docs.py first")
    chunks = {c["chunk_id"]: c for c in chunk_rows}
    titles = {d["doc_id"]: d["title"] for d in read_jsonl(DOCUMENTS_PATH)}
    plan = plan_groups(chunk_rows, args.n_scenarios, args.seed)
    if args.limit_groups:
        plan = plan[: args.limit_groups]
    done = {s["group_id"] for s in read_jsonl(SCENARIOS_PATH)}
    done |= {r["group_id"] for r in read_jsonl(SCENARIO_REJECTS_PATH)}
    todo = [g for g in plan if g["group_id"] not in done]
    print(f"plan: {len(plan)} groups / {sum(g['n_scenarios'] for g in plan)} scenarios; {len(todo)} groups to do")
    if not todo:
        return
    llm = LLMClient()

    def on_result(result: dict) -> None:
        for s in result["scenarios"]:
            append_jsonl(SCENARIOS_PATH, s)
        if not result["scenarios"]:
            append_jsonl(SCENARIO_REJECTS_PATH, {"group_id": result["group"]["group_id"],
                                                 "chunk_ids": result["group"]["chunk_ids"],
                                                 "errors": result["errors"]})

    jobs = [lambda g=g: generate_group(llm, g, chunks, titles, args.attempts) for g in todo]
    await run_all(jobs, on_result, label="scenarios")
    total = read_jsonl(SCENARIOS_PATH)
    print(f"scenarios.jsonl: {len(total)} scenarios; LLM calls {llm.calls}, tokens {llm.tokens}")
    if args.show:
        print(dumps(total[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--limit-groups", type=int, default=0, help="only the first N planned groups (pilot)")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="print the last scenario")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
