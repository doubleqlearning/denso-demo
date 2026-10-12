# FACTORY MEMO

**A solution for capturing expert manufacturing know-how and turning it into traceable, reusable knowledge for engineers and AI.**

This repository implements text-based knowledge extraction, storage, retrieval
and evidence-grounded answers, with a synthetic benchmark and a Streamlit UI.
Voice, multimodal capture, location awareness and QR identification are planned
capabilities.

## 1. Environment setup

Requirements: **Python 3.12+**, **uv**, and **Docker with Docker Compose**.
Run all commands from the repository root.

```bash
uv python install 3.12
uv venv --python 3.12
uv sync
cp .env.example .env
```

`uv run` automatically uses `.venv`. To activate it manually on macOS/Linux:
`source .venv/bin/activate`.

Edit `.env` with your provider's settings. Add the shared `OPENROUTER_*` values
below; they are supported by the code but absent from `.env.example`.
Replace every placeholder.

```dotenv
DATABASE_URL=postgresql+psycopg://voice_agent:voice_agent@localhost:5434/voice_agent

# Shared OpenAI-compatible chat endpoint
OPENROUTER_API_KEY=<your-api-key>
OPENROUTER_MODEL=<your-chat-model>
OPENROUTER_BASE_URL=<your-chat-api-base-url>

# OpenAI-compatible embedding endpoint
EMBEDDING_API_KEY=<your-api-key>
EMBEDDING_MODEL=<your-embedding-model>
EMBEDDING_BASE_URL=<your-embedding-api-base-url>
EMBEDDING_DIM=<model-output-size>
```

`EMBEDDING_DIM` must match the model's vector size. A local embedding server can
omit its API key when `EMBEDDING_BASE_URL` is set. Keep populated `.env` files out
of Git.

| Optional settings | Purpose |
| --- | --- |
| `EXTRACTION_*`, `VERIFY_*`, `CONSOLIDATION_*`, `ANSWER_*` | Override shared chat settings for each step. |
| `MODELARK_BASEURL`, `MODELARK_APIKEY`, `MODELARK_MODELNAME` | Benchmark generation and LLM judging. Runners fall back to extraction/shared chat settings for judging. |
| `HF_TOKEN` | Access private snapshots; public pulls need no token. |

## 2. Database initialization

Start PostgreSQL + pgvector and check readiness:

```bash
docker compose -f deployments/docker-compose.yml up -d
docker compose -f deployments/docker-compose.yml exec postgres pg_isready -U voice_agent -d voice_agent
```

PostgreSQL is exposed on **localhost:5434**. The application creates each
`kb_<name>` database, tables and vector extension on first use. The current
registry contains `industrial_qa` and `maintenance`.

For the Industrial QA demo, restore the published knowledge snapshot:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot pull \
  --repo-id duonganh2973/industrial_qa \
  --run-dir data/industrial_qa_runs/restored-test \
  --run-id industrial-qa-test-full-20261010T164959Z
```

This populates `kb_industrial_qa`, rebuilds knowledge/manual embeddings and
restores the selected benchmark artifacts when bundled. It requires a compatible
snapshot and matching local `benchmark/public/` files. Use a fresh database or
one containing the same snapshot; unrelated records cause the import to stop.

To build knowledge from scratch instead, use the fresh benchmark workflow below.
See the [snapshot guide](backend/knowledge_bases/industrial_qa/SNAPSHOT.md) for
local imports and other restore options.

## 3. API and Streamlit UI

**Terminal 1 — API:**

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8005
```

**Terminal 2 — UI:**

```bash
uv run streamlit run streamlit_app.py
```

Open **http://localhost:8501**, select **industrial_qa**, and ask a question.
The UI connects to the API on port **8005** and displays answers, citations and
tool traces. Check API health at **http://localhost:8005/healthz**.
Conversation history is cleared when the API restarts.

## 4. Factory benchmark

The [dataset manifest](benchmark/benchmark/manifest.json) records **3 manuals,
1,067 chunks, 195 synthetic conversations and 500 questions**: **100 train /
400 test**. All benchmark content is in English.

### Score restored predictions

After restoring the snapshot, recompute retrieval and multiple-choice metrics
without LLM calls:

```bash
uv run python benchmark/evaluate.py \
  --predictions data/industrial_qa_runs/restored-test/predictions.jsonl \
  --split test \
  --question-ids data/industrial_qa_runs/restored-test/question_ids.txt \
  --skip-llm \
  --out data/industrial_qa_runs/restored-test/results_recomputed.json
```

This scores saved predictions; it does not generate new answers. Remove
`--skip-llm` to judge answers with the configured `MODELARK_*` endpoint.

### Run fresh extraction, answers and evaluation

These runs call LLM/embedding endpoints and incur usage costs. **Ingestion
replaces unrelated records in `kb_industrial_qa`**, retaining the current and
explicitly reused runs. Stop API/UI use of this KB while rebuilding it.

Run a **10-question train pilot**, then expand to **all 100 train questions**:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.pilot all \
  --run-dir data/industrial_qa_runs/pilot-local \
  --split train --sample 10

uv run python -m backend.knowledge_bases.industrial_qa.train all \
  --run-dir data/industrial_qa_runs/train-local \
  --from-run-dir data/industrial_qa_runs/pilot-local
```

Freeze model, prompt and retrieval settings before running **all 400 test
questions**:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.holdout all \
  --run-dir data/industrial_qa_runs/test-local \
  --reuse-run-dir data/industrial_qa_runs/train-local
```

Each run saves checkpoints/configuration, `answers.json`, `predictions.jsonl`
and `results.json`. Repeat the same command to resume; use new run directories
for new experiments. Adding `--skip-llm` skips judges only: extraction and answer
generation still call models.

Use only `benchmark/public/` as inference data. Hidden labels in
`benchmark/generation/` and `benchmark/benchmark/` are for construction and
evaluation. Tune on train and report pilot coverage separately from full scores.
