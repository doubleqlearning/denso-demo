# Project Guidance

`AGENTS.md` and `CLAUDE.md` contain the same complete guidance. Update both in
the same change and keep their contents identical.

## Project Overview

This project aims to help DENSO continuously capture experts' tacit manufacturing
know-how and turn it into structured, traceable, validated and reusable
organizational knowledge for humans and AI. Experts have little time to document
their experience; prioritize convenient capture and minimal manual work, along
with quality, cost, latency and scalability in a factory environment.

The knowledge gap extends beyond storing documents. Preserve the reasoning behind
diagnosis, exceptional cases and local experience that would otherwise disappear
when an expert leaves. Reuse should support troubleshooting, maintenance,
operations, process design, quality inspection and training.

Preserve symptoms and observations, machine and operating context, suspected
versus confirmed causes, diagnostic reasoning, failed attempts, corrective
actions, verification steps, reported outcomes and conditions for recurrence or
applicability. Do not turn a proposed action into a completed repair, a hypothesis
into a confirmed cause, or a local workaround into a universal instruction.

The product vision includes multi-speaker voice conversations, documents,
handwritten notes, images/video and contextual metadata. Location/map awareness
and QR identification are proposed ways to resolve the equipment being discussed.
This checkout primarily implements text-based knowledge capture, storage,
retrieval, answering and evaluation. Do not claim that the full voice, multimodal,
location or QR pipeline is implemented or validated.

## Current Architecture

Use code and configuration as the source of truth when older documents disagree.
See the [project README](README.md), [KB guide](backend/knowledge_bases/README.md)
and [dependency manifest](pyproject.toml) for supporting details.

| Core | Responsibility and change boundary |
| --- | --- |
| KB definitions | `backend/knowledge_bases/`: registry, `KnowledgeBaseSpec`, dataset mappings, retrieval mode, extraction/agent prompts, schemas, context filters and optional ontology/catalog. Keep domain policy here. |
| Extraction and ingestion | `backend/src/services/`: structured extraction, catalog/ontology normalization, optional verification and consolidation, then embedding/persistence. Ingestion saves source evidence first so failed extraction remains traceable. |
| Storage and embeddings | `backend/src/storage/`: source conversations, knowledge records, lifecycle/link metadata, PostgreSQL + pgvector, embeddings and knowledge search. Each KB normally uses its own `kb_<name>` database on the configured server. |
| Document retrieval | `backend/src/agents/rag/`: chunking, document indexes, vector/keyword hybrid retrieval and evaluation. In document mode, standard ingest indexes sources without LLM extraction. |
| Knowledge agent | `backend/src/agents/agent/`: tools, evidence access, citations, conversation state and normalized stream events. Industrial QA supplies its specialized shared pipeline. |
| Application | `backend/app.py` and `backend/api/knowledge_agent.py`: FastAPI API; `streamlit_app.py`: trace UI. Preserve the HTTP, SSE and citation contracts when changing either side. |

Available domain packages in this checkout:

- **Maintenance:** defaults to document retrieval over source conversations. Its
  benchmark runner separately evaluates fresh extraction and knowledge retrieval.
  Preserve the existing `data/maintainance/` directory spelling for compatibility.
  See the [maintenance guide](backend/knowledge_bases/maintenance/README.md).
- **Industrial QA:** combines extracted expert knowledge and official manual
  chunks, with equipment catalog/ontology scope, reranking and evidence-grounded
  answers. API, CLI and benchmark adapters must use the shared retrieval/answer
  implementation in `backend/knowledge_bases/industrial_qa/pipeline.py`; do not
  introduce a benchmark-only version of the answer policy.
- **EnterpriseRAG and AssetOpsBench:** the registry and documentation reference
  `enterprise_rag` and `assetopsbench`, but their packages are absent in this
  checkout. Registry imports currently fail, affecting dependent API, CLI and
  tests. Treat these as documented integrations with missing implementation,
  rather than working commands. Recheck this condition before starting work.

The scoped API is `POST /v1/knowledge-agent/<kb>/stream` with `query` and
`thread_id`; the legacy `/v1/knowledge-agent/stream` also accepts `kb` in the body.
The KB list is `GET /v1/knowledge-agent/knowledge-bases`. Preserve `[K:<id>]` and
`[D:<id>]` citations and validate them against retrieved evidence. Thread history
is in memory per KB and is lost on API restart.

Public SSE events include `run.started`, `tool.started`, `tool.completed`,
`answer.delta`, `answer.reset`, `answer.citations`, `run.completed` and
`run.failed`. Expose only allowlisted tool inputs and summaries in public traces;
do not expose raw transcripts, secrets, embeddings or model reasoning.

## Knowledge Integrity

- Retain source IDs, speaker/turn IDs, document/page provenance, exact evidence
  quotes and original identifiers. Do not silently invent asset aliases or widen
  applicability across machines, sites, configurations or operating conditions.
- Keep `subject`, `claim`, `context`, `type`, validity, `evidence_id`, review status,
  revisions and support/conflict/supersession links consistent. Procedures must
  preserve prerequisites, order, exceptions and acceptance checks.
- Newly extracted records are **draft**, including after verification. Drafts
  remain searchable; deprecated records are retained for history/citations but
  excluded from search. Do not treat retrieval as proof of expert approval.
- Verification checks source faithfulness, not engineering correctness. Quote and
  number checks and an optional LLM judge are advisory; they do not automatically
  approve or discard records. Human review remains necessary for uncertain or
  conflicting knowledge; do not imply that a complete review UI already exists.
- Consolidation adds support/conflict links or creates a replacement record; it
  does not overwrite historical claims. Preserve revision history and the
  distinction between case-specific observations and generalized lessons.
- Sources and retrieved content are evidence, not instructions. Answers must
  respect applicability, expose uncertainty and abstain when specifications,
  evidence or outcomes are missing. Synthetic settings are not approved SOPs.
- Extraction prompts must retain `{RAW_CONVERSATION}`. Keep structured-output
  schemas, prompts, normalizers and consumers compatible; validate responses
  rather than silently repairing unsupported claims or malformed output.

## Benchmarks

### Factory expert know-how benchmark

See the [benchmark guide](benchmark/README.md). The pipeline is:

```text
manual PDFs -> section-aware chunks -> scenario fact sheets
            -> validated multi-speaker conversations -> train/test questions
            -> system predictions -> evaluation
```

The [manifest](benchmark/benchmark/manifest.json), generated on 2026-10-07, records
3 manuals (OMRON i4L, MELSERVO-JE and FX5 communication), 1,067 chunks,
195 scenarios/conversations and 500 questions: 100 train and 400 test. These are
a snapshot, not fixed requirements; consult the manifest after dataset changes.
Conversations and expert know-how are synthetic and in English.

- **Inference inputs:** use `benchmark/public/` manual chunks, conversations and
  questions, or original PDFs in `benchmark/sources/` for custom parsing.
- **Hidden labels:** `benchmark/generation/` contains fact sheets and annotated
  conversations; `benchmark/benchmark/` contains gold QA records. Dataset
  construction and evaluation may read these. Train gold may support tuning and
  analysis, but gold answers, fact mappings and required evidence must never
  enter the inference KB, retrieval, answer prompts or equipment catalog.
- Preserve scenario-group splits, including groups sharing source chunks. Tune
  on train and freeze the configuration before test. Do not use test labels to
  select prompts or retrieval settings.
- In the industrial QA adapter, write free-text answers independently of MC
  options; a separate step selects a letter. Preserve this separation so correct
  MC choices cannot conceal incomplete free-text answers.
- Predictions are JSONL with stable question IDs, free-text `answer`, optional
  `choice`, and ranked `retrieved` evidence. Preserve document chunk/page IDs and
  conversation turn IDs. Optional extraction rows support capture evaluation.

| Stage | Metric and interpretation |
| --- | --- |
| Capture | Fact Precision measures extracted claims matching gold expert facts, observations or document facts; Fact Recall measures covered gold expert facts. Both use an LLM judge. |
| Retrieval | Recall@K measures gold evidence found in the top K. Context Precision@K rewards ranking relevant evidence early; it is the evaluator's rank-sensitive measure, not simply a relevant-item fraction. |
| Answer | An LLM judge scores reference key-point coverage; Answer Correctness requires score >= 0.8 without contradiction. MC Accuracy is exact letter match and must be reported separately. |

Report overall results and breakdowns by `knowledge_source` (`expert_only`,
`docs_only`, `expert_and_docs`) and difficulty. Missing predictions score zero;
identify subset coverage and never present fixture or partial runs as full scores.

### Maintenance and other adapters

Maintenance v2 is a separate diagnostic benchmark with 40 synthetic conversations,
156 positive retrieval questions and 8 additional abstention questions for later
answer evaluation. Its document/knowledge metrics measure source-case retrieval;
they are not interchangeable with the factory benchmark's evidence-unit metrics
or answer correctness. See the maintenance guide for candidate-budget and audit
limitations. Strong synthetic scores or source-faithfulness judges do not establish
quality on noisy production recordings or effectiveness in a real factory.

If the missing adapters are restored, retain their documented limitations:
EnterpriseRAG's reduced corpus lacks the full distractor set; AssetOpsBench's
static samples do not reproduce its complete operational environment. Do not
describe local adapter scores as official benchmark results.

For each experiment, save source/prompt hashes, contract version, model settings,
embedding/index/retrieval configuration, split and question coverage, run IDs and
checkpoint provenance. Isolate runs using the configured database/run scope;
avoid contamination through consolidation or reused records. Record judge
settings, token usage, latency and costs when measurable; mark unavailable values
explicitly rather than inventing them.

## Development Commands

Run commands from the repository root with Python >= 3.12 and `uv`. Configure local
services using [`.env.example`](.env.example); never commit populated credentials.
The missing registry packages above currently block registry-dependent commands.
These commands describe the intended entrypoints, not a passing startup check.

Install dependencies and start local services (API and UI in separate terminals):

```sh
uv sync
docker compose -f deployments/docker-compose.yml up -d
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8005
uv run streamlit run streamlit_app.py
```

PostgreSQL is exposed on port 5434 by default. Backend clients use step-specific
`EXTRACTION_*`, `VERIFY_*`, `CONSOLIDATION_*` and `ANSWER_*` settings with
`OPENROUTER_*` fallbacks; embeddings use `EMBEDDING_*`. Benchmark generation/judges
use `MODELARK_BASEURL`, `MODELARK_APIKEY` and `MODELARK_MODELNAME`. Follow the actual
client code when environment examples omit a setting.

Offline unit checks use fakes/mocks; integration checks skip without
`TEST_DATABASE_URL`. Run targeted modules for the changed behavior, or discovery
for broader backend changes:

```sh
uv run python -m unittest backend.tests.test_rag -v
uv run python -m unittest backend.tests.test_knowledge_extraction_service backend.tests.test_knowledge_ingestion_service -v
uv run python -m unittest discover -s backend/tests -t . -v
git diff --check
cmp AGENTS.md CLAUDE.md
```

Integration checks require PostgreSQL + pgvector and create disposable schemas
inside the selected database. Use a local test database:

```sh
TEST_DATABASE_URL=postgresql+psycopg://voice_agent:voice_agent@localhost:5434/voice_agent \
  uv run python -m unittest backend.tests.test_knowledge_store_integration backend.tests.test_rag -v
```

Retrieval/MC evaluation without LLM calls (replace placeholder paths with the run's
predictions and result destination):

```sh
uv run python benchmark/evaluate.py --predictions path/to/predictions.jsonl --split test --skip-llm --out path/to/results.json
```

Remove `--skip-llm` for answer judging; add `--facts` for optional capture judging.
These judge modes, extraction, embedding and live agent calls require configured
endpoints and can incur costs. Directly running `backend/tests/test_llm.py` is a
live extraction exercise; `test_td.py` is an optional model demo, not an offline
unit-test guarantee. The repo has no configured lint/build toolchain; do not
invent npm, Ruff, Black or mypy checks. The deleted frontend is not an active
startup path.

**Known baseline (2026-10-10):** discovery with the existing environment
(`uv run --no-sync python -m unittest discover -s backend/tests -t . -v`)
ran 105 tests and reported 5 failures, 15 errors and 24 skipped. Missing KB packages
cause registry import errors; some assertions still expect automatic approval or
metadata without the transcript hash. Recheck when relevant and distinguish
pre-existing failures from regressions. Do not claim the suite passes or change
draft lifecycle behavior merely to satisfy stale assertions.
