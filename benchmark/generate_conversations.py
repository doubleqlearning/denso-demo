"""Steps 3–4 — generate a canonical expert conversation per scenario and validate it with rules.

    uv run python benchmark/generate_conversations.py

A conversation is accepted only if every rule check passes; otherwise it is regenerated (up to
--attempts) and finally dropped. All attempts are logged to generation/validation.jsonl. Resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib

from common import (CHUNKS_PATH, CONVERSATIONS_PATH, SCENARIOS_PATH, VALIDATION_PATH, append_jsonl, load_prompt,
                    overlap, quote_in, read_jsonl, render)
from llm_client import LLMClient, run_all

LENGTH_BANDS = {"short": (6, 8), "medium": (10, 14), "long": (18, 26)}
LENGTH_TOLERANCE = 1
GROUNDING_MIN_OVERLAP = 0.5

DIFFICULTY_INSTRUCTIONS = {
    "direct": "The expert states each piece of know-how plainly, answering the junior's questions in a "
              "logical order.",
    "implicit": "The know-how comes out indirectly: the expert asks questions back, tells short anecdotes "
                "(\"last time this happened...\") and spreads the advice over several turns. Still keep each "
                "fact's key terms.",
    "correction_reference": "Early on, a non-expert proposes one of the wrong ideas listed above and the expert "
                            "explicitly corrects it (\"No, don't ... —\"). Later the expert refers back to an "
                            "earlier turn (\"like I said about ...\").",
    "noisy": "The conversation is realistic and messy: interruptions, small talk and other machines are "
             "mentioned between the useful turns.",
}

EXPERT_ROLES = ["maintenance_engineer", "senior_technician", "controls_engineer"]
JUNIOR_ROLES = ["operator", "junior_technician", "maintenance_apprentice"]


def pick(options: list[str], key: str) -> str:
    return options[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(options)]


def speakers_for(scenario: dict) -> list[dict]:
    sid = scenario["scenario_id"]
    speakers = [
        {"id": "junior_01", "role": pick(JUNIOR_ROLES, sid + "j"), "is_expert": False},
        {"id": "expert_01", "role": pick(EXPERT_ROLES, sid + "e"), "is_expert": True},
    ]
    if scenario["conversation_controls"]["num_speakers"] == 3:
        speakers.append({"id": "supervisor_01", "role": "shift_supervisor", "is_expert": False})
    return speakers


def build_prompt(scenario: dict, speakers: list[dict]) -> str:
    controls = scenario["conversation_controls"]
    lo, hi = LENGTH_BANDS[controls["length"]]
    ratio = controls["irrelevant_ratio"]
    if ratio:
        n_off = max(1, round(ratio * (lo + hi) / 2))
        irrelevant = (f"About {n_off} turns ({int(ratio * 100)}%) are off-topic (shift schedule, lunch, other "
                      "machines, spare-parts delivery) and contain no know-how.")
    else:
        irrelevant = "None: every turn is about the incident."
    a = scenario["anchor"]
    return render(
        load_prompt("generate_conversation.md"),
        title=scenario["title"],
        machine=scenario["machine"],
        anchor=f"{a['asset_id']} — {a['site']}, {a['line']}, {a['station']}",
        asset_id=a["asset_id"],
        doc_facts="\n".join(f"- {e['fact']}" for e in scenario["source_evidence"]),
        observations="\n".join(f"- {o}" for o in scenario["observations"]),
        expert_facts="\n".join(f"- [{f['fact_id']}] {f['fact']}" for f in scenario["expert_facts"]),
        misconceptions="\n".join(f"- {m}" for m in scenario["misconceptions"]),
        speakers="\n".join(f"- {s['id']}: {s['role'].replace('_', ' ')}"
                           + (" (the senior expert)" if s["is_expert"] else "") for s in speakers),
        n_turns_min=str(lo),
        n_turns_max=str(hi),
        difficulty_instructions=DIFFICULTY_INSTRUCTIONS[controls["difficulty"]],
        irrelevant_instructions=irrelevant,
    )


def validate(scenario: dict, speakers: list[dict], data: dict, chunks: dict[str, dict]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    turns = data.get("turns") or []
    fact_turns = data.get("fact_turns") or {}
    speaker_ids = {s["id"] for s in speakers}
    experts = {s["id"] for s in speakers if s["is_expert"]}
    by_id = {t.get("turn_id"): t for t in turns}

    # ids_valid: contiguous turn ids, declared speakers, length band.
    ids_errors = []
    if [t.get("turn_id") for t in turns] != list(range(1, len(turns) + 1)):
        ids_errors.append("turn_id must be 1..N in order")
    for t in turns:
        if t.get("speaker") not in speaker_ids:
            ids_errors.append(f"turn {t.get('turn_id')}: unknown speaker {t.get('speaker')}")
        if not str(t.get("text", "")).strip():
            ids_errors.append(f"turn {t.get('turn_id')}: empty text")
    lo, hi = LENGTH_BANDS[scenario["conversation_controls"]["length"]]
    if not lo - LENGTH_TOLERANCE <= len(turns) <= hi + LENGTH_TOLERANCE:
        ids_errors.append(f"{len(turns)} turns, expected {lo}-{hi}")
    errors += ids_errors

    # document_support_valid: official evidence still resolves to the cited chunks.
    doc_errors = [f"quote not in {e['chunk_id']}" for e in scenario["source_evidence"]
                  if e["chunk_id"] not in chunks or not quote_in(e["quote"], chunks[e["chunk_id"]]["text"])]
    errors += doc_errors

    # conversation_support_valid: every expert fact maps to existing turns spoken by an expert.
    conv_errors, ground_errors = [], []
    for fact in scenario["expert_facts"]:
        mapped = fact_turns.get(fact["fact_id"]) or []
        if not mapped:
            conv_errors.append(f"{fact['fact_id']}: no mapped turn")
            continue
        bad = [m for m in mapped if m not in by_id or by_id[m].get("speaker") not in experts]
        if bad:
            conv_errors.append(f"{fact['fact_id']}: mapped turns {bad} missing or not spoken by the expert")
            continue
        # all_gold_facts_grounded: the fact's content words appear in its mapped turns.
        text = " ".join(by_id[m]["text"] for m in mapped)
        score = overlap(fact["fact"], text)
        if score < GROUNDING_MIN_OVERLAP:
            ground_errors.append(f"{fact['fact_id']}: only {score:.0%} of the fact's key terms appear in turns "
                                 f"{mapped}; state it with its specifics")
    unknown = set(fact_turns) - {f["fact_id"] for f in scenario["expert_facts"]}
    if unknown:
        conv_errors.append(f"unknown fact ids in fact_turns: {sorted(unknown)}")
    if scenario["anchor"]["asset_id"].lower() not in " ".join(t.get("text", "") for t in turns).lower():
        ground_errors.append(f"asset id {scenario['anchor']['asset_id']} never mentioned")
    errors += conv_errors + ground_errors

    checks = {
        "ids_valid": not ids_errors,
        "document_support_valid": not doc_errors,
        "conversation_support_valid": not conv_errors,
        "all_gold_facts_grounded": not ground_errors,
    }
    return checks, errors


async def generate_one(llm: LLMClient, scenario: dict, chunks: dict[str, dict], attempts: int) -> dict:
    sid = scenario["scenario_id"]
    cid = "conv_" + sid.removeprefix("scn_")
    speakers = speakers_for(scenario)
    prompt = build_prompt(scenario, speakers)
    log: list[dict] = []
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        messages = [{"role": "user", "content": prompt}]
        if errors:
            messages.append({"role": "user", "content": "A previous attempt was rejected:\n- " + "\n- ".join(errors)
                             + "\nWrite a new conversation that fixes these problems."})
        data = await llm.chat_json(messages, temperature=0.9)
        checks, errors = validate(scenario, speakers, data if isinstance(data, dict) else {}, chunks)
        accept = all(checks.values())
        log.append({"scenario_id": sid, "conversation_id": cid, "attempt": attempt, "checks": checks,
                    "errors": errors, "accept": accept, "final": accept or attempt == attempts})
        if accept:
            turns = [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"].strip()}
                     for t in data["turns"]]
            conversation = {
                "conversation_id": cid,
                "scenario_id": sid,
                "controls": scenario["conversation_controls"],
                "speakers": speakers,
                "turns": turns,
                "fact_turns": {f["fact_id"]: data["fact_turns"][f["fact_id"]] for f in scenario["expert_facts"]},
                "n_turns": len(turns),
                "n_words": sum(len(t["text"].split()) for t in turns),
            }
            return {"conversation": conversation, "log": log}
    return {"conversation": None, "log": log}


async def main_async(args: argparse.Namespace) -> None:
    scenarios = read_jsonl(SCENARIOS_PATH)
    chunks = {c["chunk_id"]: c for c in read_jsonl(CHUNKS_PATH)}
    done = {c["scenario_id"] for c in read_jsonl(CONVERSATIONS_PATH)}
    done |= {v["scenario_id"] for v in read_jsonl(VALIDATION_PATH) if v.get("final")}
    todo = [s for s in scenarios if s["scenario_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(scenarios)} scenarios, {len(todo)} to generate")
    if not todo:
        return
    llm = LLMClient()

    def on_result(result: dict) -> None:
        for row in result["log"]:
            append_jsonl(VALIDATION_PATH, row)
        if result["conversation"]:
            append_jsonl(CONVERSATIONS_PATH, result["conversation"])

    jobs = [lambda s=s: generate_one(llm, s, chunks, args.attempts) for s in todo]
    await run_all(jobs, on_result, label="conversations")
    log = read_jsonl(VALIDATION_PATH)
    finals = [v for v in log if v.get("final")]
    print(f"conversations.jsonl: {len(read_jsonl(CONVERSATIONS_PATH))} accepted; "
          f"{sum(not v['accept'] for v in finals)} dropped; first-attempt pass rate "
          f"{sum(v['accept'] for v in log if v['attempt'] == 1)}/{sum(v['attempt'] == 1 for v in log)}; "
          f"LLM calls {llm.calls}, tokens {llm.tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="only N scenarios (pilot)")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
