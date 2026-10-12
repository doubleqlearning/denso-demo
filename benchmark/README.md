# Factory Expert Know-how Benchmark

A benchmark for systems that **capture, store and reuse the know-how of senior experts** on the
factory floor. The system under test reads official equipment manuals and expert conversations,
extracts knowledge into a knowledge base, retrieves it and answers technicians' questions:

```
reference manuals + expert conversations
        → knowledge extraction → knowledge base → retrieval → Q&A
```

The benchmark is generated from **scenario fact sheets** that carry gold facts, gold evidence and
reference answers. These gold labels are **hidden from the system**: it only receives the manual
chunks, the raw conversations and the questions (see [Public inputs](#public-inputs)).

It measures three things:

| Stage | Metric | Gold label used |
|---|---|---|
| Capture (optional) | Fact Precision / Recall (LLM judge) | scenario expert facts, observations, doc facts |
| Retrieval | Recall@K, Context Precision@K | `required_evidence` (doc chunks, conversation turns) |
| Answer | Answer Correctness (LLM judge), MC Accuracy (exact match) | `reference_answer`, `answer_key_points`, `correct_option` |

Everything (sources, conversations, questions, prompts) is in English.

## Dataset at a glance

- **3 manuals** (1,224 pages) → 1,067 section-aware chunks of ~300–700 tokens.
- **195 scenarios** → 195 validated conversations (7–26 turns, mean 13.9; 204–649 words, mean 371;
  2–3 speakers), 193 of which are referenced by questions.
- **500 questions** (100 train / 400 test), each with a free-text reference answer **and** a
  4-option multiple-choice version. Correct letters are balanced (A 124 / B 130 / C 125 / D 121
  in the current build) and the correct option is strictly the longest in only 23.6% of
  questions (chance ≈ 25%), so option length is not a usable shortcut.

| knowledge_source | easy | medium | hard | total |
|---|---|---|---|---|
| expert_only | 100 | 70 | 30 | 200 |
| docs_only | 80 | 50 | 20 | 150 |
| expert_and_docs | 20 | 80 | 50 | 150 |
| **total** | 200 | 200 | 100 | 500 |

Exact counts per cell and split, conversation length statistics, option-letter balance,
`correct_option_strictly_longest` and prompt hashes are recorded in `benchmark/manifest.json` after each build.

### Sources

| doc_id | File | Pages | Content |
|---|---|---|---|
| `omron_i4l` | `i658_i4l_series_users_manual_en.pdf` | 176 | OMRON i4L SCARA robot user's manual (installation, operation, troubleshooting, maintenance) |
| `melservo_je` | `sh030166engf.pdf` | 92 | Mitsubishi MELSERVO-JE servo amplifier troubleshooting (alarm/warning cause–check–action tables) |
| `fx5_comm` | `sh082625engh.pdf` | 956 | Mitsubishi MELSEC iQ-F FX5 communication manual (Ethernet, serial, Modbus, SLMP, troubleshooting) |

Scenarios are spread ≈30% / 35% / 35% over the three manuals so the 956-page FX5 manual does not
dominate. The PDFs are git-ignored; place them in `sources/` before parsing.

## Pipeline

```
sources/*.pdf
  │ 1. parse_docs.py            text layer check → text + tables + layout → section path from
  │                             bookmarks → header/footer cleanup → chunks with provenance
  ▼
processed/chunks.jsonl  (official evidence pool; scenario_candidate=true for alarm/error/
  │                      troubleshooting/maintenance/warning/motor/encoder/motion/safety/communication)
  │ 2. generate_scenarios.py    1–2 chunks of one section → LLM → scenario fact sheet(s)
  ▼                             (rule check: quotes are verbatim in the cited chunk)
generation/scenarios.jsonl
  │ 3. generate_conversations.py  fact sheet + controls → canonical conversation with turn_ids
  │ 4.                            rule validation → accept, or regenerate (≤3 attempts) / drop
  ▼
generation/conversations.jsonl, generation/validation.jsonl
  │ 5. generate_qa.py           plan slots (cell targets) → select gold evidence FIRST →
  │ 6.                          LLM writes question + answer + MC options ONLY from that evidence
  │                             → rule validation → build, split, public export
  ▼
benchmark/{qa,train,test}.jsonl + manifest.json        (gold, hidden)
public/{corpus_chunks,conversations,questions_*}.jsonl   (what the system sees)
```

### 1. Parse manuals — `parse_docs.py`

PyMuPDF only (no OCR, no heavy layout models):

1. **Text layer detection** — pages with < 20 characters of text and images are listed in
   `documents.jsonl` as `needs_ocr_pages` and skipped (none in the current sources).
2. **Extraction** — lines with font size and bounding boxes; rotated side tabs are dropped.
3. **Headings / tables** — the section path comes from the PDF bookmarks, located on the page by
   matching the heading line. Tables come from `page.find_tables()` and are rendered as Markdown
   rows; text inside tables is not duplicated.
4. **Header/footer cleanup** — lines in the top 8% / bottom 8% of the page that repeat on ≥ 3
   pages (digits normalised) or look like page numbers are removed. Lines broken into separate
   blocks are re-joined into sentences; hyphenation is repaired.
5. **Chunking** — paragraphs and table rows are packed per section into ~300–700 token chunks
   (token ≈ 4 characters); chunks never cross a top-level section; long tables repeat their
   header row; small trailing pieces are merged.
6. **Provenance** — every chunk keeps `doc_id`, `section_path`, `page_start`, `page_end`.

### 2. Scenario fact sheets — `generate_scenarios.py`

A deterministic plan picks groups of 1–2 adjacent candidate chunks of the same section
(troubleshooting/maintenance sections first) and pre-assigns each scenario a unique site anchor
and its conversation controls, so the mix is balanced over the whole dataset. Each group yields
1 scenario (2 for the shorter OMRON manual).

The schema strictly separates document facts from synthetic expert knowledge:

```json
{
  "scenario_id": "scn_001_1",
  "group_id": "g001",
  "doc_id": "melservo_je",
  "title": "Intermittent AL. 17.4 board error on MR-JE-AX205 after adjacent VFD installation",
  "topic": "alarm",
  "machine": "Mitsubishi MELSERVO-JE servo amplifier (MR-JE) with HG servo motor",
  "anchor": {"site": "Plant Bac Ninh", "line": "Line 5", "station": "Machining cell A", "asset_id": "MR-JE-AX205"},
  "source_chunk_ids": ["melservo_je_p0022_c01"],
  "source_evidence": [
    {"chunk_id": "melservo_je_p0022_c01",
     "quote": "Disconnect the cables except for the power supply, and then check the repeatability.",
     "fact": "The manual instructs to disconnect all cables except power and check whether the alarm repeats."}
  ],
  "observations": ["MR-JE-AX205 tripped AL. 17.4 three times this week, always around 14:00 ..."],
  "expert_facts": [
    {"fact_id": "scn_001_1_ef2",
     "fact": "Before replacing any MR-JE, we reroute the SW1 cable at least 15 cm away from the VFD output cables.",
     "type": "workaround"}
  ],
  "misconceptions": ["Replacing the servo amplifier immediately because the manual lists board error ..."],
  "conversation_controls": {"length": "medium", "difficulty": "implicit", "num_speakers": 3, "irrelevant_ratio": 0.0}
}
```

- `source_evidence` **must be supported by the manual**: `quote` is a verbatim span of the cited
  chunk (checked by rule), `fact` paraphrases only that quote.
- `expert_facts` are **synthetic** tacit know-how that is *not* in the manual (an expert fact that
  restates the chunk is rejected): `tacit_heuristic`, `historical_experience`, `workaround`,
  `root_cause_hint`, `safety_practice`, `local_parameter`. Each is one atomic, checkable claim.
- `misconceptions` are plausible wrong ideas used for correction-style conversations and for
  multiple-choice distractors.
- The `anchor.asset_id` is unique per scenario, so expert-specific questions are unambiguous
  across ~200 similar conversations.

### 3. Canonical conversations — `generate_conversations.py`

Input: the fact sheet (doc facts, observations, expert facts, misconceptions) and its controls.

| Control | Values |
|---|---|
| `length` | `short` 6–8 turns (30%), `medium` 10–14 (45%), `long` 18–26 (25%) |
| `difficulty` | `direct` (facts stated plainly, 30%), `implicit` (facts spread over turns as anecdotes and questions, 25%), `correction_reference` (a non-expert proposes a wrong idea, the expert corrects it and later refers back, 25%), `noisy` (interruptions and small talk, 20%) |
| `num_speakers` | 2 (junior + expert) or 3 (+ shift supervisor) |
| `irrelevant_ratio` | share of off-topic turns: 0.0, 0.2 or 0.4 (`noisy` ≥ 0.2, `direct` = 0) |

```json
{
  "conversation_id": "conv_001_1",
  "scenario_id": "scn_001_1",
  "controls": {"length": "medium", "difficulty": "implicit", "num_speakers": 3, "irrelevant_ratio": 0.0},
  "speakers": [{"id": "junior_01", "role": "operator", "is_expert": false},
               {"id": "expert_01", "role": "maintenance_engineer", "is_expert": true}],
  "turns": [{"turn_id": 1, "speaker": "junior_01", "text": "MR-JE-AX205 on Line 5 tripped AL. 17.4 again ..."}],
  "fact_turns": {"scn_001_1_ef1": [4], "scn_001_1_ef2": [6, 8]},
  "n_turns": 12,
  "n_words": 410
}
```

`turn_id` is mandatory and `fact_turns` (gold) maps every expert fact to the expert turn(s) that
state its final, corrected version. Once accepted, the conversation is part of the golden dataset.

### 4. Conversation validation (rule-based, no LLM)

Every attempt is logged to `generation/validation.jsonl`:

```json
{"scenario_id": "scn_001_1", "conversation_id": "conv_001_1", "attempt": 1,
 "checks": {"ids_valid": true, "document_support_valid": true,
            "conversation_support_valid": true, "all_gold_facts_grounded": true},
 "errors": [], "accept": true, "final": true}
```

| Check | Rule |
|---|---|
| `ids_valid` | turn ids are 1..N in order, every speaker is declared, turn count within the length band (±1) |
| `document_support_valid` | every cited `chunk_id` exists and every `quote` is found verbatim (whitespace/case-insensitive) in it |
| `conversation_support_valid` | every expert fact maps to ≥ 1 existing turn, spoken by the expert (source_type=expert ⇒ evidence from the conversation) |
| `all_gold_facts_grounded` | ≥ 50% of each expert fact's content words appear in its mapped turns; the asset id is mentioned |

`accept=false` → the conversation is regenerated with the errors as feedback (up to 3 attempts),
then the scenario is dropped.

### 5–6. Questions from evidence — `generate_qa.py`

Questions are generated **from selected evidence, not from the conversation directly**:

```
plan slot (knowledge_source × difficulty)
   → select gold evidence (doc chunk quote and/or expert fact turns)
   → LLM writes question + reference answer + MC options ONLY from that evidence
   → rule validation (regenerate once, else drop the slot)
```

Evidence per cell:

| knowledge_source | easy / medium | hard |
|---|---|---|
| `expert_only` | 1 expert fact (its turns) | 2 expert facts |
| `docs_only` | 1 manual chunk | 2 chunks of the same section (or 1 chunk with a condition/order) |
| `expert_and_docs` | 1 chunk + 1 expert fact | 1 chunk + 2 expert facts (or 2 chunks + 1 fact) |

Difficulty:

- **easy** — one stated fact; the answer is almost explicit in the evidence.
- **medium** — the question describes symptoms or goals in other words; the answer links symptom → action, cause or value.
- **hard** — needs all evidence units (multi-hop), a condition/exception/order of steps, or the corrected advice instead of an initial wrong idea.

`expert_only` and `expert_and_docs` questions name the asset id (site-specific know-how);
`docs_only` questions are general product questions and never name a site or asset. Each evidence
set is used at most once and each expert fact at most twice; the planner spreads ~3 questions per
conversation and over-generates 10% so every cell is filled exactly after rejects.

Rule validation of a question: required fields; 1–4 key points; the question does not mention
"conversation/excerpt/evidence" and does not copy a 10-word phrase; asset id present (expert
questions) or absent (docs_only); the reference answer is grounded in the evidence (≥ 40% content
word overlap) and uses **every** evidence unit; 3 distinct distractors, none a paraphrase of the
correct option, and the correct option is not conspicuously longer than the distractors.
Near-duplicate questions are removed at build time.

Gold record (`benchmark/qa.jsonl`, `train.jsonl`, `test.jsonl`; abridged example):

```json
{
  "question_id": "q_0003",
  "scenario_id": "scn_001_1",
  "conversation_id": "conv_001_1",
  "group_id": "g001",
  "question": "MR-JE-AX205 on Line 5 keeps showing AL. 17.4 intermittently, and a new welder sits 5 m away. How should we troubleshoot before swapping the amplifier?",
  "question_type": "diagnosis",
  "difficulty": "medium",
  "knowledge_source": "expert_and_docs",
  "required_evidence": [
    {"type": "document", "id": "melservo_je_p0022_c01", "doc_id": "melservo_je", "pages": [22, 23]},
    {"type": "conversation", "id": "conv_001_1", "turn_id": 6}
  ],
  "gold_fact_ids": ["scn_001_1_ef3"],
  "reference_answer": "Disconnect all cables except the power supply and check repeatability; if it does not repeat, ...",
  "answer_key_points": ["Disconnect cables except the power supply and check repeatability.", "..."],
  "options": {"A": "...", "B": "...", "C": "...", "D": "Disconnect all cables except power and check repeatability; ..."},
  "correct_option": "D",
  "split": "test"
}
```

`question_type` ∈ `factual`, `procedure`, `diagnosis`, `causal`, `comparison`. Correct letters
are assigned round-robin, so A–D are balanced.

**Split**: whole scenario groups (scenarios sharing source chunks, and their conversations) go to
either train or test, ~20% / 80% per knowledge_source × difficulty cell. No conversation or
source chunk appears in both splits. Use train for prompt tuning / few-shot, report on test.

### Public inputs

The system under test may only read `public/`:

| File | Content |
|---|---|
| `corpus_chunks.jsonl` | all 1,067 manual chunks: `chunk_id, doc_id, section_path, page_start, page_end, text` |
| `conversations.jsonl` | `conversation_id`, `speakers` (id, role), `turns` — no scenario, facts or `fact_turns` |
| `questions_train.jsonl`, `questions_test.jsonl` | `question_id`, `question`, `options` |

A system may also ingest the PDFs in `sources/` with its own chunking; retrieval is then matched
by document page (below).

## Evaluation — `evaluate.py`

The system writes one predictions JSONL:

```json
{"type": "answer", "question_id": "q_0001", "answer": "free-text answer", "choice": "B",
 "retrieved": [{"type": "document", "id": "melservo_je_p0022_c01"},
               {"type": "document", "doc_id": "omron_i4l", "page": 141},
               {"type": "conversation", "id": "conv_001_1", "turn_id": 4}]}
{"type": "extraction", "conversation_id": "conv_001_1", "facts": ["extracted knowledge item", "..."]}
```

`retrieved` is ranked. A document item matches a gold chunk by exact `chunk_id`, or by `doc_id` +
a `page` inside the chunk's page range. A conversation item matches by conversation id + `turn_id`.
Strings are accepted as shorthand: `"melservo_je_p0022_c01"` or `"conv_001_1#4"`.

```bash
uv run python benchmark/evaluate.py --predictions run.jsonl --split test --out results.json
uv run python benchmark/evaluate.py --predictions run.jsonl --skip-llm   # retrieval + MC only, no LLM
uv run python benchmark/evaluate.py --predictions run.jsonl --facts      # also fact precision/recall
```

| Metric | Definition |
|---|---|
| **Recall@K** | share of the question's gold evidence units found in the top K retrieved items (K = 1, 3, 5, 10) |
| **Context Precision@K** | RAGAS-style: mean of precision@i over the ranks i ≤ K that hold a gold unit (rewards ranking gold evidence first) |
| **MC Accuracy** | `choice` equals `correct_option` (exact match, no LLM) |
| **Answer Score** | LLM judge marks each key point covered (1) / partial (0.5) / missing / contradicted; score = mean |
| **Answer Correctness** | answer score ≥ 0.8 and no contradiction with the reference answer |
| **Fact Recall** *(optional)* | share of gold expert facts expressed by at least one extracted fact (per conversation, then averaged; also per expert-fact type) |
| **Fact Precision** *(optional)* | share of extracted facts that match a gold fact (expert facts, observations or document facts of that scenario) |

Results are reported overall, by `knowledge_source`, by `difficulty` and per cell; `--out` also
keeps per-question rows and judge outputs. Missing predictions score 0.

Sanity checks: on the full test split an oracle retriever (gold evidence) scores Recall@5 = 1.0
and Context Precision = 1.0 (Recall@1 = 0.78 because multi-evidence questions cannot be covered by
one item). On the pilot set an oracle (gold evidence, reference answers, correct letters,
gold facts + one junk fact) scores 1.0 on recall, MC, answer correctness and fact recall, with
fact precision equal to the share of real facts. Answering with a distractor scores ~0.1 answer
score and 0% correctness. Empty predictions score 0.

## End-to-end example

A real test-split conversation (`conv_057_1`) and its three questions, one per knowledge source,
scored for a deliberately imperfect system. Files: [`examples/predictions_example.jsonl`](examples/predictions_example.jsonl)
and [`examples/results_example.json`](examples/results_example.json).

### 1. Input — what the system reads (`public/` only)

Manual chunk (`public/corpus_chunks.jsonl`, abridged):

```json
{"chunk_id": "melservo_je_p0069_c01", "doc_id": "melservo_je",
 "section_path": ["1. SERVO AMPLIFIER TROUBLESHOOTING", "1.6 Trouble which does not trigger an alarm/warning"],
 "page_start": 69, "page_end": 69,
 "text": "| Description | Cause | Checkpoint | Action | Target |\n| The servo motor does not operate. | The setting of the torque limit is incorrect. | Check if the torque limit value is \"0\". [A], [C]: [Pr. PA11] and [Pr. PA12] ... | Set it correctly. |\n| A machine is interfering with the servo motor. | Check if a machine is interfering. | Remove the interference. | ..."}
```

Conversation (`public/conversations.jsonl`, 8 turns, abridged; no facts or `fact_turns`):

```json
{"conversation_id": "conv_057_1",
 "speakers": [{"id": "junior_01", "role": "junior_technician"}, {"id": "expert_01", "role": "senior_technician"}],
 "turns": [
  {"turn_id": 1, "speaker": "junior_01", "text": "Hey, I'm stuck on MR-JE-AX932 on Line 8. The servo motor won't rotate when I give the start command, but there's no alarm or warning ... should I swap it?"},
  {"turn_id": 2, "speaker": "expert_01", "text": "Hold on, don't swap anything yet. On Line 8, first check for mechanical binding by manually turning the ball screw. If it moves freely, then we go check parameters."},
  {"turn_id": 4, "speaker": "expert_01", "text": "We had a similar incident in 2022 on Line 8, and it was a piece of swarf jamming the coupling, not electrical settings. ... make sure the safety gate is closed, because the brake release signal on this machine is interlocked with the gate."},
  {"turn_id": 6, "speaker": "expert_01", "text": "Right, when PA11 is zero, temporarily set it to 100% to test motor movement. But revert it immediately after diagnosis, don't leave it like that."}
 ]}
```

Questions (`public/questions_test.jsonl`):

```json
{"question_id": "q_0158", "question": "On MR-JE-AX932, the servo motor won't move and PA11 currently reads zero — what temporary setting should I try to test movement, and what must I do afterward?",
 "options": {"A": "Temporarily set PA11 to 0% to test movement, then leave it at that value.",
             "B": "Permanently set PA11 to 100% to test movement, then check for alarms afterward.",
             "C": "Temporarily set PA11 to 100% to test movement, then revert it right after diagnosis.",
             "D": "Temporarily set PA11 to 50% to test movement, then revert it after a restart."}}
{"question_id": "q_0159", "question": "MR-JE-AX932 on Line 8 won't rotate after a tool change with no alarm shown — what should I check first and what parameter could be causing this?",
 "options": {"A": "Check the electromagnetic brake power supply first, then inspect the command pulse wiring.",
             "B": "Check ball screw binding manually, then verify torque limit parameter is not set to 0.",
             "C": "Swap the servo amplifier immediately, then verify the control mode parameter is correct.",
             "D": "Turn on SON and LSP/LSN signals first, then check the torque limit parameter value."}}
{"question_id": "q_0160", "question": "On an MR-JE servo amplifier, the servo motor won't run and no alarm or warning shows. What should I check if the machine itself is interfering with the motor?",
 "options": {"A": "Check whether a machine is interfering and remove the interference.",
             "B": "Turn on the electromagnetic brake power supply.",
             "C": "Check the torque limit value and set it correctly.",
             "D": "Check the SON state and turn SON on."}}
```

### 2. Output — what the system writes (`predictions.jsonl`)

```json
{"type": "extraction", "conversation_id": "conv_057_1", "facts": [
  "On Line 8, check for mechanical binding by turning the ball screw by hand before checking parameters.",
  "When PA11 (torque limit) is 0, temporarily set it to 100% to test motor movement and revert it right after diagnosis.",
  "The brake release on MR-JE-AX932's machine is interlocked with the safety gate, so close the gate before troubleshooting.",
  "Servo amplifiers sometimes need replacement."]}
{"type": "answer", "question_id": "q_0158", "choice": "C",
 "answer": "Set PA11 to 100% temporarily to check whether the motor moves, then put PA11 back right after the diagnosis.",
 "retrieved": [{"type": "conversation", "id": "conv_057_1", "turn_id": 6}, {"type": "conversation", "id": "conv_057_1", "turn_id": 8},
               {"type": "document", "id": "melservo_je_p0069_c01"}]}
{"type": "answer", "question_id": "q_0159", "choice": "B",
 "answer": "First turn the ball screw by hand to rule out mechanical binding; on Line 8 that is always the first check.",
 "retrieved": [{"type": "conversation", "id": "conv_057_1", "turn_id": 2}, {"type": "conversation", "id": "conv_057_1", "turn_id": 4},
               {"type": "conversation", "id": "conv_057_1", "turn_id": 8}]}
{"type": "answer", "question_id": "q_0160", "choice": "A",
 "answer": "Check whether a machine is interfering with the servo motor and remove the interference.",
 "retrieved": [{"type": "document", "id": "melservo_je_p0070_c01"}, {"type": "document", "doc_id": "melservo_je", "page": 69}]}
```

The second item of q_0160 matches by `doc_id` + `page`, as a system with its own chunking would report it.

### 3. Gold (hidden) and scores

Gold for these questions (from `benchmark/test.jsonl`, abridged):

```json
{"question_id": "q_0158", "knowledge_source": "expert_only", "difficulty": "medium",
 "required_evidence": [{"type": "conversation", "id": "conv_057_1", "turn_id": 6}],
 "reference_answer": "Temporarily set PA11 to 100% to test motor movement, then revert it immediately after diagnosis so it isn't left in that state.",
 "answer_key_points": ["Set PA11 temporarily to 100%", "This is done to test motor movement", "Revert PA11 immediately after diagnosis"],
 "correct_option": "C"}
{"question_id": "q_0159", "knowledge_source": "expert_and_docs", "difficulty": "medium",
 "required_evidence": [{"type": "document", "id": "melservo_je_p0069_c01", "doc_id": "melservo_je", "pages": [69, 69]},
                       {"type": "conversation", "id": "conv_057_1", "turn_id": 2}],
 "reference_answer": "First check for mechanical binding by manually turning the ball screw on Line 8; if it moves freely, check whether the torque limit parameter is set to 0, as an incorrect torque limit setting prevents motor operation without triggering an alarm.",
 "answer_key_points": ["Check for mechanical binding first by manually turning the ball screw on Line 8",
                       "If the ball screw moves freely, check parameters",
                       "An incorrect torque limit setting (value of 0) causes the motor not to operate without an alarm"],
 "correct_option": "B"}
{"question_id": "q_0160", "knowledge_source": "docs_only", "difficulty": "easy",
 "required_evidence": [{"type": "document", "id": "melservo_je_p0069_c01", "doc_id": "melservo_je", "pages": [69, 69]}],
 "reference_answer": "If a machine is interfering with the servo motor, check whether a machine is interfering and remove the interference.",
 "correct_option": "A"}
```

`uv run python benchmark/evaluate.py --predictions benchmark/examples/predictions_example.jsonl --split test --facts --k 1,3,5`
gives these per-question rows:

| Question | Cell | Recall@1 | Recall@3 | Context Precision@3 | MC | Answer score | Correct |
|---|---|---|---|---|---|---|---|
| q_0158 | expert_only / medium | 1.0 | 1.0 | 1.0 | ✔ C | 1.00 | ✔ |
| q_0159 | expert_and_docs / medium | 0.5 | 0.5 | 1.0 | ✔ B | 0.33 | ✘ |
| q_0160 | docs_only / easy | 0.0 | 1.0 | 0.5 | ✔ A | 1.00 | ✔ |

- **q_0159** — the system used only the expert's know-how and ignored the manual: the chunk was
  never retrieved (Recall 0.5) and the judge marked 2 of 3 key points missing (check parameters
  when the screw turns freely; torque limit = 0). It still picked the right option, which is why
  free-text Answer Correctness is the stricter measure.
- **q_0160** — the right chunk was found but ranked second: Recall@1 = 0, Context Precision@3 = 0.5.
- **Fact capture for `conv_057_1`** — precision 0.75, recall 0.75. The extraction covers ef1, ef3 and
  ef4 but misses **ef2** (the 2022 swarf jam, `historical_experience`), and "Servo amplifiers
  sometimes need replacement." matches no gold fact.

The "Overall" table printed for this file is close to 0 because the other 397 test questions are
unanswered and score 0; read the per-question rows (`per_question` in the results JSON).

## Reproduce

Environment (`.env` at the repository root, OpenAI-compatible endpoint):

```
MODELARK_BASEURL=...
MODELARK_APIKEY=...
MODELARK_MODELNAME=...
```

`llm_client.py` is an async client with a 50 requests/minute sliding-window limiter, 20 concurrent
calls, retries with backoff and thinking disabled (DeepSeek on ModelArk). When the provider's usage
quota is exhausted, the scripts stop cleanly and keep all finished results; re-running the same
command resumes.

```bash
uv sync
uv run python benchmark/parse_docs.py                         # ~1.5 min, no LLM
uv run python benchmark/generate_scenarios.py --n-scenarios 200
uv run python benchmark/generate_conversations.py
uv run python benchmark/generate_qa.py                        # plan → generate → validate → build
uv run python benchmark/generate_qa.py --build                # rebuild outputs only
```

Pilot runs: `generate_scenarios.py --limit-groups 5`, `generate_conversations.py --limit 5`,
`generate_qa.py --target-scale 0.05`. All steps are seeded (`--seed 42`) and resumable; delete the
corresponding `generation/*.jsonl` file to regenerate a step.

| File | Written by | Content |
|---|---|---|
| `processed/documents.jsonl` | parse_docs | per manual: title, sha256, pages, OCR pages, bookmarks |
| `processed/chunks.jsonl` | parse_docs | all chunks with provenance, `topics`, `scenario_candidate` |
| `generation/scenarios.jsonl` | generate_scenarios | accepted scenario fact sheets |
| `generation/scenario_rejects.jsonl` | generate_scenarios | groups that failed validation after all attempts |
| `generation/conversations.jsonl` | generate_conversations | accepted conversations with gold `fact_turns` |
| `generation/validation.jsonl` | generate_conversations | every validation attempt |
| `generation/qa_slots.jsonl`, `qa_drafts.jsonl` | generate_qa | planned evidence slots and generated drafts |
| `benchmark/qa.jsonl`, `train.jsonl`, `test.jsonl`, `manifest.json` | generate_qa | gold benchmark |
| `public/*.jsonl` | generate_qa | system inputs |
| `examples/predictions_example.jsonl`, `results_example.json` | hand-written / evaluate | the [end-to-end example](#end-to-end-example) |

Prompts live in `prompts/` (`generate_scenario.md`, `generate_conversation.md`, `generate_qa.md`,
`judge_answer.md`, `judge_facts.md`).

## Limitations

- Expert facts, observations and conversations are **synthetic** (LLM-generated). They are
  realistic and checked by rules, but not reviewed by real experts.
- Validation is rule-based (verbatim quotes, turn mapping, lexical grounding). It catches missing
  or misattributed facts but not every subtle contradiction; the LLM judges add variance.
- Generic manual questions (`docs_only`) may also be answerable from other, near-identical
  passages of the same manual; such hits count as misses for Recall@K.
- Page-based matching is coarser than chunk ids when chunk page ranges overlap.
- No OCR: scanned pages would be skipped (none in the current sources).
