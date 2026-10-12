# Industrial QA knowledge snapshots

This workflow publishes the useful, expensive-to-reproduce parts of
`kb_industrial_qa`: the extracted rows in `knowledge_items` and optionally a
checksummed question-answering artifact bundle from completed benchmark runs.

It does **not** publish:

- `raw_conversations` (rebuilt from `benchmark/public/conversations.jsonl`)
- `chunks_full`, `chunks_full_terms`, or `rag_indexes` (rebuilt from
  `benchmark/public/corpus_chunks.jsonl`)
- vector embeddings (recomputed with the puller's embedding model)

The dataset contains `knowledge_items.jsonl`, `manifest.json`, a Hugging Face
dataset card, and optional `benchmark_artifacts/<run-id>/` directories. Each
knowledge row replaces its database-specific `evidence_id` with
`source_conversation_id`. Pulling recreates the evidence row from the
repository's public benchmark data, maps the new UUID back into the record, and
preserves knowledge UUIDs and knowledge-to-knowledge links.

For a bundled benchmark run, the workflow keeps:

- `answers.json` for answer, retrieval, citation, token-usage, and generation traces
- `predictions.jsonl` and `predictions_uncited.jsonl` as evaluator inputs
- `results.json` and `results_uncited.json` as full metric reports
- `manifest.json`, `question_ids.txt`, inventories, and corpus replacement metadata
- `run_checkpoint.json` as the historical extraction/consolidation checkpoint
- `snapshot/` as frozen source, prompt, catalog, and ontology provenance
- generated `REPORT.md` and `artifact_manifest.json` for a readable summary and
  per-file SHA-256 verification

The bundle does not copy public conversations, manual chunks, hidden gold
labels, or source PDFs. Those continue to come from the matching source checkout.

## Environment

Install the locked dependencies and start PostgreSQL + pgvector:

```bash
uv sync
docker compose -f deployments/docker-compose.yml up -d
```

Set these values in `.env`:

```dotenv
DATABASE_URL=postgresql+psycopg://voice_agent:voice_agent@localhost:5434/voice_agent

# Needed on pull because both knowledge and manual embeddings are rebuilt.
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=...
EMBEDDING_BASE_URL=...
EMBEDDING_DIM=...

# owner/dataset; can instead be passed with --repo-id.
HF_DATASET_REPO_ID=your-org/industrial-qa-knowledge
# Write token for push or read token for a private dataset. A public pull needs no token.
HF_TOKEN=hf_...
```

The embedding model and dimension must be mutually compatible. They need not
match the publisher's model because neither knowledge nor manual vectors are
uploaded; both are rebuilt together on pull.

## Inspect a local export

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot export \
  --output data/industrial_qa_snapshot \
  --include-latest-test
```

`--run-file` is optional and repeatable. It stores only benchmark
selection/configuration, not the duplicated `conversations` payload. Include
each run that another user should be able to reconstruct. The supplied files,
together with their `reused_run_ids`, must cover all run namespaces currently
present in `knowledge_items`.

`--include-latest-test` locates the newest completed `test` run by the
`created_at` value in `run.json`, requires complete answers/results and a frozen
source snapshot, and bundles it. Its `run.json` automatically supplies the run
metadata, so a separate `--run-file` is usually unnecessary. Use repeatable
`--artifacts-dir path/to/run` when you want explicit runs instead.

## Push

Create a write token in Hugging Face, set `HF_TOKEN`, then run:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot push \
  --repo-id your-org/industrial-qa-knowledge \
  --include-latest-test
```

For the current checkout, that resolves to
`data/industrial_qa_runs/test-v1-20261010` and run
`industrial-qa-test-full-20261010T164959Z`. That run declares the reused train
and baseline run IDs, so it covers the run namespaces currently in the database.
If the database changes later and the command reports uncovered run IDs, add
their metadata with repeatable `--run-file` options.

Add `--private` when the dataset must be private. The command creates the
dataset repository if needed and uploads one new Hub commit. It never uploads
`.env`, database credentials, raw transcripts, manual chunks, or embeddings.

## Pull and rebuild the database

Use a fresh `kb_industrial_qa` database (or one already containing exactly the
same snapshot):

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot pull \
  --repo-id your-org/industrial-qa-knowledge \
  --run-dir data/industrial_qa_runs/restored-test \
  --run-id industrial-qa-test-full-20261010T164959Z
```

Pull performs these steps:

1. Download the snapshot at `--revision` (default: the repository's default
   branch).
2. Verify the snapshot checksum and the SHA-256 hashes of the local public
   conversations/manual chunks.
3. Recreate only the raw conversation rows referenced by the knowledge
   snapshot and remap their new evidence UUIDs.
4. Validate and insert all extracted records while preserving their IDs and
   links, then recompute knowledge embeddings.
5. Ingest all public manual chunks through the normal code path, rebuilding
   vector/full-text/BM25 indexes.
6. If `--run-dir` is present, restore that run's QA artifacts and reports when
   available, then reconstruct a portable `run.json` whose evidence UUIDs match
   the new database. The exact historical checkpoint remains available as
   `run_checkpoint.json`. If several runs were published, use `--run-id` to
   choose one.

To verify the published retrieval and MC scores without model calls:

```bash
uv run python benchmark/evaluate.py \
  --predictions data/industrial_qa_runs/restored-test/predictions.jsonl \
  --split test \
  --question-ids data/industrial_qa_runs/restored-test/question_ids.txt \
  --k 1,3,5,10 \
  --skip-llm \
  --out data/industrial_qa_runs/restored-test/results_recomputed.json
```

This deterministically recomputes retrieval and multiple-choice metrics from
the saved predictions. Re-running answer generation or LLM judging requires the
recorded model endpoints and can differ due to provider/model nondeterminism;
the saved `answers.json` and full `results.json` are the audit record for the
original run.

The command refuses to merge the snapshot into unrelated knowledge records.
This is intentional: an implicit merge could silently corrupt consolidation
links or mix benchmark runs. A repeated pull of the exact same snapshot is
safe and resumes manual indexing if it was interrupted. `--skip-manuals` is
available for a knowledge-only restore, but that database is not complete for
the combined H1/H3 benchmark.

For an offline/local transfer, use the same format without Hugging Face:

```bash
uv run python -m backend.knowledge_bases.industrial_qa.snapshot import \
  --input data/industrial_qa_snapshot \
  --run-dir data/industrial_qa_runs/restored
```

Hugging Face references: [upload files](https://huggingface.co/docs/huggingface_hub/en/guides/upload),
[download snapshots](https://huggingface.co/docs/huggingface_hub/main/guides/download), and
[dataset cards](https://huggingface.co/docs/hub/main/datasets-cards).
