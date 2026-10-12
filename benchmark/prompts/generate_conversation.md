You write realistic, natural shop-floor conversations for a benchmark that tests whether an AI system
can capture the tacit know-how of senior maintenance experts from what they say.

## Scenario
Incident: {{title}}
Machine: {{machine}}
Location: {{anchor}}

What the manual says (official facts — speakers may mention them in their own words):
{{doc_facts}}

What the junior colleague observes:
{{observations}}

Expert know-how (GOLD facts — every one must be said by the expert):
{{expert_facts}}

Plausible wrong ideas someone may raise:
{{misconceptions}}

## Speakers
{{speakers}}

## Conversation controls
- Length: {{n_turns_min}}–{{n_turns_max}} turns (one turn = one speaker's uninterrupted utterance).
- Style: {{difficulty_instructions}}
- Off-topic turns: {{irrelevant_instructions}}

## Rules
- The conversation starts with the junior describing the problem. The equipment's asset id
  ({{asset_id}}) must be mentioned at least once.
- Every expert fact must be stated by an expert speaker, clearly and with the same specifics
  (numbers, component names, parameter names, dates, order of checks). You may rephrase naturally,
  but keep its key terms so the fact is recognisable. One fact may be split across at most two turns.
- The expert must not state know-how that contradicts the gold facts or the manual.
- Spoken, informal English as used on a factory floor; short turns (1–4 sentences). No narration,
  no stage directions, no bullet lists, no markdown.
- Do not mention "the scenario", "gold facts" or "the benchmark".

## Output
Return ONE JSON object and nothing else:

{"turns": [{"turn_id": 1, "speaker": "<speaker id>", "text": "..."}],
 "fact_turns": {"<fact_id>": [<turn_id>, ...]}}

`fact_turns` maps EVERY expert fact id to the turn id(s) where the expert states it (final, correct
version — not the turn where a wrong idea is raised).
