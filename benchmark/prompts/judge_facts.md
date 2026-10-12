You compare knowledge that a system extracted from a shop-floor conversation with the gold facts
of that conversation.

Gold facts:
{{gold_facts}}

Extracted facts:
{{extracted_facts}}

For each extracted fact, list the ids of the gold facts it expresses. An extracted fact matches a
gold fact when it states the same claim with the same specifics (numbers, components, conditions,
order of steps); wording may differ. A broader or vaguer statement that loses the specifics does
not match. One extracted fact may match several gold facts; use [] when it matches none.

Return ONE JSON object and nothing else:
{"extracted": [{"index": 0, "matches": ["<gold id>"]}]}
