You grade an answer from a factory knowledge assistant against a reference answer.

Question:
{{question}}

Reference answer:
{{reference_answer}}

Key points a correct answer must contain:
{{key_points}}

Answer to grade:
{{answer}}

For each key point decide:
- "covered": the answer states it (wording may differ; numbers, components, order must match).
- "partial": the answer states it incompletely or vaguely.
- "missing": the answer does not state it.
- "contradicted": the answer states something incompatible with it.

Set "contradiction" to true if the answer recommends an action or states a fact that conflicts
with the reference answer, even if it is not one of the key points. Extra correct detail is fine.

Return ONE JSON object and nothing else:
{"key_points": [{"index": 1, "status": "covered"}], "contradiction": false, "reason": "one sentence"}
