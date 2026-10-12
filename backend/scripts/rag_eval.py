"""Index a BEIR dataset and evaluate semantic, keyword, and hybrid retrieval.

    # Chunk, embed, and index the corpus (resumes where it stopped)
    uv run python -m backend.scripts.rag_eval index data/datasets/scifact

    # Recall@k, MRR@k, nDCG@k on the test split for every search mode
    uv run python -m backend.scripts.rag_eval evaluate data/datasets/scifact

    # Inspect one query
    uv run python -m backend.scripts.rag_eval query data/datasets/scifact "..." --mode hybrid

Each dataset gets database rag_<directory name> on the DATABASE_URL server and
one table per chunking setup (chunks_full, chunks_w128_o1, ...), so setups can
be compared side by side. Pass the same --chunk-words to index and evaluate.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.knowledge_bases.base import DATA_DIR, PROJECT_ROOT
from backend.src.agents.rag.chunking import ChunkingConfig
from backend.src.agents.rag.evaluation import (
    K_VALUES,
    format_table,
    load_corpus,
    load_queries,
    retrieve,
    save_run,
    score_modes,
)
from backend.src.agents.rag.indexing import index_documents
from backend.src.agents.rag.store import DocumentStore, SearchMode
from backend.src.storage import StorageError
from backend.src.storage.databases import DatabaseAdminError


RUNS_DIR = DATA_DIR / "rag_eval"
# Query-side instructions for instruction-tuned embedding models (MTEB prompts).
QUERY_INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
}


def _chunking(args: argparse.Namespace) -> ChunkingConfig:
    return ChunkingConfig(max_words=args.chunk_words, overlap_sentences=args.overlap)


def _query_instruction(args: argparse.Namespace) -> str:
    if args.query_instruction is not None:
        return args.query_instruction
    return QUERY_INSTRUCTIONS.get(args.dataset.name, "")


async def _open_store(args: argparse.Namespace) -> DocumentStore:
    return await DocumentStore.for_dataset(
        args.dataset.name,
        query_instruction=_query_instruction(args),
        table=f"chunks_{_chunking(args).label}",
    )


async def run_index(args: argparse.Namespace) -> int:
    documents = load_corpus(args.dataset)[: args.limit]
    store = await _open_store(args)
    started = time.perf_counter()
    try:
        await index_documents(store, documents, _chunking(args), reindex=args.reindex)
        total = await store.count_chunks()
    finally:
        await store.aclose()
    print(f"[{store.table_name}] {total} chunks in {time.perf_counter() - started:.1f}s")
    return 0


async def run_evaluate(args: argparse.Namespace) -> int:
    queries, qrels = load_queries(args.dataset, args.split)
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
        qrels = {qid: qrels[qid] for qid in queries}
    modes = [SearchMode(mode) for mode in args.modes]
    k_values = sorted(set(args.k_values))
    store = await _open_store(args)
    try:
        index_config = await store.config()
        chunks = await store.count_chunks()
        if not chunks:
            print(
                f"Error: {store.table_name} is empty; run `index` with the same "
                "--chunk-words first",
                file=sys.stderr,
            )
            return 2
        started = time.perf_counter()
        hits = await retrieve(
            store, queries, modes, k=max(k_values), concurrency=args.concurrency
        )
        elapsed = time.perf_counter() - started
    finally:
        await store.aclose()

    summary, per_query = score_modes(qrels, hits, k_values)
    print()
    print(
        format_table(
            f"{args.dataset.name} {args.split}: {len(queries)} queries, "
            f"{store.table_name} ({chunks} chunks), {elapsed:.1f}s",
            summary,
            k_values,
        )
    )
    output = args.output or RUNS_DIR / args.dataset.name / (
        f"{store.table_name}-{args.split}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    save_run(
        output,
        queries=queries,
        qrels=qrels,
        hits=hits,
        summary=summary,
        per_query=per_query,
        dataset=args.dataset.name,
        split=args.split,
        table=store.table_name,
        chunks=chunks,
        index_config=index_config,
        query_instruction=_query_instruction(args),
        k_values=k_values,
    )
    print(f"\nRun saved to {output}")
    return 0


async def run_query(args: argparse.Namespace) -> int:
    store = await _open_store(args)
    try:
        hits = await store.search(args.query, mode=SearchMode(args.mode), k=args.k)
    finally:
        await store.aclose()
    for rank, hit in enumerate(hits, start=1):
        ranks = f"vector={hit.vector_rank} lexical={hit.lexical_rank}"
        print(f"{rank:>2}. {hit.chunk_id}  score={hit.score:.4f}  {ranks}")
        print(f"    {hit.title}")
        print(f"    {hit.text[:200]}{'...' if len(hit.text) > 200 else ''}")
    return 0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _int_list(raw: str) -> list[int]:
    return [_positive_int(part) for part in raw.split(",") if part.strip()]


def _mode_list(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    for mode in modes:
        SearchMode(mode)
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("dataset", type=Path, help="BEIR dataset directory")
    common.add_argument(
        "--chunk-words", type=_positive_int,
        help="Split documents longer than this many words (default: one chunk per document)",
    )
    common.add_argument("--overlap", type=int, default=1, help="Sentences shared by consecutive chunks")
    common.add_argument(
        "--query-instruction",
        help="Query-side embedding instruction (default: per dataset, empty if unknown)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", parents=[common], help="Chunk, embed, and index the corpus")
    index.add_argument("--limit", type=_positive_int, help="Index only the first N documents")
    index.add_argument("--reindex", action="store_true", help="Drop this chunking setup's table first")

    evaluate = commands.add_parser("evaluate", parents=[common], help="Recall, MRR, nDCG per mode")
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument(
        "--modes", type=_mode_list, default=[mode.value for mode in SearchMode],
        help="Comma-separated: semantic,keyword,hybrid",
    )
    evaluate.add_argument("--k-values", type=_int_list, default=list(K_VALUES))
    evaluate.add_argument("--limit", type=_positive_int, help="Evaluate only the first N queries")
    evaluate.add_argument("--concurrency", type=_positive_int, default=8)
    evaluate.add_argument("--output", type=Path, help=f"Run file (default: under {RUNS_DIR})")

    query = commands.add_parser("query", parents=[common], help="Show the chunks one query retrieves")
    query.add_argument("query")
    query.add_argument("--mode", choices=[mode.value for mode in SearchMode], default="hybrid")
    query.add_argument("-k", type=_positive_int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(level=logging.WARNING)
    if args.overlap < 0:
        print("Error: --overlap must not be negative", file=sys.stderr)
        return 2
    if not (args.dataset / "corpus.jsonl").is_file():
        print(f"Error: {args.dataset}/corpus.jsonl not found", file=sys.stderr)
        return 2
    commands = {"index": run_index, "evaluate": run_evaluate, "query": run_query}
    try:
        return asyncio.run(commands[args.command](args))
    except (OSError, ValueError, StorageError, DatabaseAdminError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
