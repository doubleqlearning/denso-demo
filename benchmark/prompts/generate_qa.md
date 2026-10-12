You write one benchmark question for a factory knowledge assistant. The question must be answerable
ONLY from the evidence below. You never see anything else, so do not add outside knowledge.

## Equipment
{{machine}} — {{anchor}}

## Evidence (all of it is required to answer)
{{evidence}}

## Question requirements
Knowledge source: **{{knowledge_source}}** — {{source_instructions}}
Difficulty: **{{difficulty}}** — {{difficulty_instructions}}

- Write the question the way a technician on the shop floor would ask a knowledge assistant:
  standalone, specific, one question (no "and also..."), 10–40 words.
- Never refer to the evidence itself ("in the conversation", "the excerpt", "the expert said",
  "the text", "according to the manual") and never copy a long phrase from the evidence.
- `question_type`: factual | procedure | diagnosis | causal | comparison.
- `reference_answer`: 1–3 sentences, complete and correct, using only the evidence.
- `answer_key_points`: 1–4 short atomic statements that a correct answer must contain.

## Multiple-choice version
- `correct_option`: the correct answer, concise (≤ 25 words). Keep only what makes it correct.
- Length balance (important): all four options have about the same length and level of detail
  (within a few words), so the correct option never stands out as the longest or most specific.
- `distractors`: exactly 3 wrong options. Each is plausible to a non-expert, matches the correct
  option in style and length, and is clearly wrong given the evidence (e.g. the generic textbook
  fix, the wrong order of checks, a wrong number/parameter, or one of these common wrong ideas:
  {{misconceptions}}). No "all/none of the above", no options that are partly correct.

## Output
Return ONE JSON object and nothing else:

{"question": "...", "question_type": "...", "reference_answer": "...",
 "answer_key_points": ["..."], "correct_option": "...", "distractors": ["...", "...", "..."]}
