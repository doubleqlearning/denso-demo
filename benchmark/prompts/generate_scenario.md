You are building a benchmark that tests whether an AI system can capture, store and reuse the
know-how of senior maintenance experts in a factory. You write realistic shop-floor
**scenario fact sheets** grounded in excerpts from an official equipment manual.

## Equipment
Machine: {{machine}}
Manual: {{manual_title}}

## Manual excerpts (official evidence)
{{chunks}}

## Task
Write exactly {{n_scenarios}} scenario fact sheet(s). Each scenario is one concrete incident on the
shop floor (a fault, alarm, abnormal behaviour, maintenance decision or communication problem) that a
junior colleague brings to a senior expert. {{diversity_note}}

Each scenario has four kinds of content. Keep them strictly separate:

1. `source_evidence` (1–3 items) — facts taken from the manual excerpts above.
   - `chunk_id`: the id of the excerpt the fact comes from (only ids listed above).
   - `quote`: a VERBATIM span copied character-for-character from that excerpt (one or two sentences,
     or one table row; 8–60 words). Do not fix typos, do not shorten with "...".
   - `fact`: a one-sentence paraphrase of what the quote says. No information beyond the quote.
2. `observations` (2–4) — symptoms or context the junior reports (what happened, when, how often,
   recent changes). Concrete: numbers, times, readings, alarm codes consistent with the manual.
3. `expert_facts` (2–4) — SYNTHETIC tacit know-how of the senior expert that is NOT written in the
   manual: heuristics, site history, workarounds, root-cause hints, safety habits, local parameters.
   - Each is ONE atomic, self-contained, checkable claim in one sentence (≤ 25 words), with concrete
     specifics (a number, a component, an order of checks, a past event). Avoid vague advice.
     Do not chain two claims with "because", "and then", "so" — split them or keep the important one.
   - Must not contradict the manual and must not simply restate it.
   - `type` is one of: tacit_heuristic, historical_experience, workaround, root_cause_hint,
     safety_practice, local_parameter. Use at least two different types.
4. `misconceptions` (1–2) — plausible but WRONG actions or beliefs a less experienced person might
   suggest for this incident (e.g. replace a part too early, wrong check order). The expert facts or
   the manual must make clear why each is wrong.

Refer to the equipment by its site anchor so the incident is unique: {{anchor}}.
Write everything in English.

## Output
Return ONE JSON object and nothing else:

{"scenarios": [
  {
    "title": "short incident title",
    "topic": "one of: alarm, error, troubleshooting, maintenance, warning, motor, encoder, motion, safety, communication",
    "source_evidence": [{"chunk_id": "...", "quote": "...", "fact": "..."}],
    "observations": ["..."],
    "expert_facts": [{"fact": "...", "type": "tacit_heuristic"}],
    "misconceptions": ["..."]
  }
]}
