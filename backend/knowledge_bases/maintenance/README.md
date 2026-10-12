# Automotive maintenance benchmark v2

40 **synthetic** English conversations for automotive factory operations and
maintenance. Cases 1–20 preserve the original turns verbatim and add shift,
measurement, diagnostic, repair and acceptance context. Cases 21–40 cover
body-in-white welding, tightening, pressing, adhesive/sealant, coating, CNC,
utilities, leak testing, AGV and handover. Machine settings are scoped fictional
case specifications, not department-wide operating instructions.

The dialogues contain 292–434 content words (mean 336.9), 8–12 turns and benchmark
difficulties: normal readings that hide faults, failed swaps, product/configuration
exceptions, warm-state recurrence, containment, proposed versus completed actions,
partial recovery and pending release. The v1 originals contain 44–172 words
including speaker labels.

## Sources and ground truth

- `benchmark/cases.json`: tracked canonical conversations, five semantic gold
  facets plus source evidence, 80 case questions and 40 challenge questions.
- `benchmark/legacy_conversations.csv`, `legacy_questions.csv`: archived v1 inputs.
- `prepare.py`: deterministic CSV materialization. Gold cards and answers never
  enter extraction or document embeddings; the ingest CSV contains source text
  and neutral metadata only.

```sh
uv run python -m backend.knowledge_bases.maintenance.prepare
```

Generated under `data/maintainance/` (existing directory spelling retained):

| File | Content |
|---|---|
| `expert_knowledge_extraction_benchmark_40_conversations_en.csv` | 40 source dialogues with stable case and turn IDs |
| `retrieval_evaluation_questions_answers_v2.csv` | 156 positive retrieval questions, `sample_answer`, `required_answer_facts`, `gold_case_ids`, `gold_evidence` |
| `answer_accuracy_questions_v2.csv` | Same 156 plus 8 missing-specification/pending-outcome questions with expected abstention |
| `benchmark_v2_manifest.json` | Version, language, counts and length statistics |

The 156 positives comprise 44 legacy questions, 80 case questions, 20 paraphrases
without asset IDs and 12 explicit comparisons requiring two sources. Broad legacy
queries 42–44 have corpus-wide v2 judgements; new relevant cases must not be scored
as false positives. `gold_evidence` lists expert turns as a support pool, not minimal
answer spans. `required_answer_facts` splits the authored reference answer into
sentences, not an independently reviewed atomic-fact rubric. Abstention rows have
empty positive qrels and are excluded from Recall/MRR/nDCG for later answer tests.

## Ingest, retrieval and extraction inspection

Maintenance's default agent still retrieves **documents**. Standard ingest only
indexes source conversations and does not run extraction:

```sh
uv run python -m backend.scripts.kb ingest maintenance --reindex
uv run python -m backend.scripts.kb evaluate maintenance
```

For both the source baseline and fresh extraction on all cases:

```sh
uv run python -m backend.knowledge_bases.maintenance.bench ingest \
  --run-dir data/maintenance_runs/my-v2-run --batch-size 4 --concurrency 4
uv run python -m backend.knowledge_bases.maintenance.bench evaluate \
  --run-dir data/maintenance_runs/my-v2-run
uv run python -m backend.knowledge_bases.maintenance.audit \
  --run-dir data/maintenance_runs/my-v2-run --judge --concurrency 4
uv run python -m backend.knowledge_bases.maintenance.report \
  --run-dir data/maintenance_runs/my-v2-run --output docs/maintenance-extraction-review.html
```

The runner uses the real `KnowledgeIngestionService`, configured LLM/embedding and
`kb_maintenance` PostgreSQL. It saves a verified gzip JSON snapshot of existing
rows/vectors before rebuilding `chunks_full`. Existing knowledge/evidence are
retained. New records carry `context.benchmark_run_id`; evaluation filters strictly
to that run. The same folder resumes its checkpoint; a new folder extracts all 40
from scratch. Changed source or extractor rejects incompatible resume.

Prompt version: `maintenance-case-card-v2`. The six Case Card fields are stored in
`context.case_card`, preserving the existing schema. `claim` is a self-contained
retrieval summary. The runner explicitly enables JSON Schema output; the provider
must support this mode. Up to three generation attempts are logged; schema failures
are never silently repaired. Prompt hash, model, token limit, thinking, response
schema, evidence IDs and record IDs are saved. Evaluation saves question-file and
embedding provenance.

Evidence quotes and run IDs are excluded from embedding/lexical text so long source
excerpts cannot hide missing extraction. Metadata retains quotes for inspection.
Only this run's knowledge vectors and lexical text are refreshed for the contract.

Document metrics use each source's best chunk. Native knowledge metrics map the
production hybrid ranking's top **20 items** to unique source cases in first-hit
order, then score Recall/MRR/nDCG@1/3/5/10. These measure source retrieval, not the
proportion of required items in an LLM context or answer accuracy. Candidate budgets
differ; this is not an equal-budget controlled comparison. A query with 32 relevant
cases cannot achieve Recall@5 above 5/32, so inspect broad-query groups separately.

## Audit limits

Deterministic checks verify card shape and exact evidence substrings in the stated
source turn. They do not certify all statements. Optional LLM review assesses five
gold facets against the union of extracted content, excluding evidence quotes.
`AUDIT_MODEL` may select another reviewer; the default matches extraction and can
miss errors or report false omissions.

The self-contained reviewer shows original turns, actual cards, gold facets,
questions/answers, quote checks and separate spot-review notes when
`manual_review.json` exists. Cited-turn counts are not semantic coverage. See the
checked run's [assessment](../../../docs/maintenance-rag-extraction-audit.md).

40 synthetic sources form a compact diagnostic benchmark with limited distractors.
Strong scores do not establish quality on noisy production transcripts. Answer
accuracy and prompt A/B experiments have not been run.
