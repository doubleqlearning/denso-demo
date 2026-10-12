"""Manage and query knowledge bases, each stored in its own database.

Examples:
    uv run python -m backend.scripts.kb list
    uv run python -m backend.scripts.kb ingest maintenance --limit 5
    uv run python -m backend.scripts.kb ingest maintenance --batch-size 8
    uv run python -m backend.scripts.kb ingest enterprise_rag --limit 20 --batch-size 8
    uv run python -m backend.scripts.kb ingest maintenance --file transcript.txt
    uv run python -m backend.scripts.kb ingest maintenance --extract --verify llm --limit 5
    uv run python -m backend.scripts.kb query maintenance "How do I fix the unstable sensor on M01?"
    uv run python -m backend.scripts.kb evaluate maintenance
    uv run python -m backend.scripts.kb reembed maintenance
    uv run python -m backend.scripts.kb reset maintenance --yes

Knowledge bases with retrieval="knowledge" extract knowledge records with an
LLM at ingest; retrieval="documents" chunks and embeds the sources as they are
(no LLM), and `evaluate` scores that retrieval against labelled questions;
`ingest --extract` runs knowledge extraction on such a knowledge base too.
Extracted records are normalized to the knowledge base's ontology, verified
against their transcript, and consolidated with stored knowledge; all are
stored as drafts for expert review.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from backend.knowledge_bases import KNOWLEDGE_BASES, KnowledgeBaseSpec, get_kb
from backend.knowledge_bases.base import DATA_DIR, PROJECT_ROOT, Retrieval, RetrievalQuestions
from backend.knowledge_bases.stores import open_document_store, open_store
from backend.src.agents.agent import KnowledgeAgentError, KnowledgeAgentService
from backend.src.agents.agent.service import DocumentCitation
from backend.src.agents.rag.chunking import Document
from backend.src.agents.rag.evaluation import (
    Qrels,
    format_table,
    retrieve,
    save_run,
    score_modes,
)
from backend.src.agents.rag.indexing import index_documents
from backend.src.agents.rag.store import SearchMode
from backend.src.services.knowledge_consolidation import DEFAULT_ISOLATION_KEY, KnowledgeConsolidator
from backend.src.services.knowledge_extraction import KnowledgeExtractionService
from backend.src.services.knowledge_ingestion import (
    IngestionInput,
    IngestionResult,
    KnowledgeIngestionService,
)
from backend.src.services.knowledge_verification import KnowledgeVerifier, VerifyMode
from backend.src.storage import KnowledgeRecord, KnowledgeStore, StorageError
from backend.src.storage.databases import (
    DatabaseAdminError,
    database_exists,
    drop_database,
    server_url_from_env,
)


DEFAULT_TRANSCRIPT_COLUMN = "raw_conversation"
DEFAULT_STATED_BY_COLUMN = "stated_by"
# Headroom for reasoning models, whose thinking tokens count against max_tokens.
EXTRACTION_MAX_TOKENS = 8192
TEXT_SUFFIXES = {".txt", ".md"}
EVAL_RUNS_DIR = DATA_DIR / "kb_eval"
EVAL_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class TranscriptInput:
    transcript: str
    stated_by: str | None
    metadata: dict[str, str]
    label: str


def read_transcripts(
    path: Path,
    *,
    transcript_column: str = DEFAULT_TRANSCRIPT_COLUMN,
    stated_by_column: str = DEFAULT_STATED_BY_COLUMN,
    stated_by: str | None = None,
) -> Iterator[TranscriptInput]:
    """Yield normalized ingestion inputs without touching external services."""

    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        transcript = path.read_text(encoding="utf-8-sig").strip()
        if not transcript:
            raise ValueError(f"Input file is empty: {path}")
        yield TranscriptInput(
            transcript=transcript,
            stated_by=stated_by,
            metadata={"source_file": path.name},
            label=path.name,
        )
        return

    if suffix != ".csv":
        raise ValueError("Unsupported input format; expected .txt, .md, or .csv")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if transcript_column not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain transcript column {transcript_column!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            transcript = (row.get(transcript_column) or "").strip()
            if not transcript:
                raise ValueError(f"CSV row {row_number} has an empty transcript")
            row_stated_by = stated_by
            if row_stated_by is None:
                row_stated_by = (row.get(stated_by_column) or "").strip() or None
            metadata = {
                key: value
                for key, raw_value in row.items()
                if key not in {transcript_column, stated_by_column}
                and raw_value is not None
                and (value := raw_value.strip())
            }
            metadata.update({"source_file": path.name, "source_row": str(row_number)})
            yield TranscriptInput(
                transcript=transcript,
                stated_by=row_stated_by,
                metadata=metadata,
                label=f"{path.name}:{row_number}",
            )


def load_retrieval_questions(questions: RetrievalQuestions) -> tuple[dict[str, str], Qrels]:
    """Questions by id and their gold document ids (qrels), from a labelled CSV."""

    queries: dict[str, str] = {}
    qrels: Qrels = {}
    with questions.path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = (questions.id_column, questions.question_column, questions.gold_column)
        if missing := [column for column in columns if column not in (reader.fieldnames or [])]:
            raise ValueError(f"{questions.path.name} has no column(s) {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            qid = (row[questions.id_column] or "").strip()
            question = (row[questions.question_column] or "").strip()
            gold = [
                doc_id.strip()
                for doc_id in (row[questions.gold_column] or "").split(questions.gold_separator)
                if doc_id.strip()
            ]
            if not qid or not question or not gold:
                raise ValueError(
                    f"{questions.path.name} row {row_number} needs an id, a question, "
                    "and gold ids"
                )
            if qid in queries:
                raise ValueError(f"{questions.path.name} repeats question id {qid!r}")
            queries[qid] = question
            qrels[qid] = {doc_id: 1 for doc_id in gold}
    return queries, qrels


def to_documents(
    items: list[TranscriptInput], *, id_column: str | None, title_column: str | None
) -> list[Document]:
    """Documents keyed by the dataset id column (else the input label)."""

    documents = [
        Document(
            id=item.metadata.get(id_column, item.label) if id_column else item.label,
            text=item.transcript,
            title=item.metadata.get(title_column, "") if title_column else "",
        )
        for item in items
    ]
    ids = [document.id for document in documents]
    if duplicate := next((doc_id for doc_id in ids if ids.count(doc_id) > 1), None):
        raise ValueError(f"Document id {duplicate!r} appears more than once")
    return documents


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List registered knowledge bases")

    ingest = commands.add_parser(
        "ingest", help="Extract knowledge from transcripts, or index them for document retrieval"
    )
    ingest.add_argument("kb")
    ingest.add_argument(
        "--file",
        type=Path,
        help=".txt, .md, or .csv input (default: the knowledge base's dataset)",
    )
    ingest.add_argument(
        "--transcript-column",
        help="CSV transcript column (default: from the knowledge base's dataset)",
    )
    ingest.add_argument(
        "--stated-by-column",
        help="CSV speaker column when present (default: from the knowledge base's dataset)",
    )
    ingest.add_argument(
        "--stated-by",
        help="Speaker applied to every input; overrides the CSV speaker column",
    )
    ingest.add_argument("--limit", type=_positive_int, help="Process at most this many transcripts")
    ingest.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Transcripts per batch: extracted concurrently, embedded in one call "
        "(default: 1, one at a time)",
    )
    ingest.add_argument(
        "--concurrency",
        type=_positive_int,
        help="Maximum concurrent LLM extraction calls (default: --batch-size)",
    )
    ingest.add_argument(
        "--no-resume",
        action="store_true",
        help="Ingest rows whose dataset id is already processed again "
        "(by default they are skipped, so an interrupted run can continue)",
    )
    ingest.add_argument(
        "--reindex",
        action="store_true",
        help="Document retrieval: drop the chunk table and index everything again",
    )
    ingest.add_argument(
        "--extract",
        action="store_true",
        help="Document retrieval: extract knowledge records instead of indexing chunks",
    )
    ingest.add_argument(
        "--verify",
        choices=[mode.value for mode in VerifyMode],
        default=VerifyMode.QUOTES.value,
        help="Check records against their transcript: quotes and numbers, plus an LLM "
        "judge for llm (VERIFY_* settings); results are stored, records stay drafts "
        "(default: quotes)",
    )
    ingest.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Store every record as new instead of merging or linking it with stored "
        "lessons (CONSOLIDATION_* settings)",
    )
    ingest.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the batch with the first failed transcript instead of continuing",
    )

    query = commands.add_parser("query", help="Ask the knowledge agent one question")
    query.add_argument("kb")
    query.add_argument("question")

    evaluate = commands.add_parser(
        "evaluate",
        help="Recall/MRR/nDCG of document retrieval on the knowledge base's labelled questions",
    )
    evaluate.add_argument("kb")
    evaluate.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in SearchMode],
        default=[mode.value for mode in SearchMode],
    )
    evaluate.add_argument("--k-values", type=_positive_int, nargs="+", default=list(EVAL_K_VALUES))
    evaluate.add_argument(
        "--query-instruction",
        help="Override the embedding query instruction (\"\" disables it), to compare prompts",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        help=f"Run JSON (default: {EVAL_RUNS_DIR.relative_to(PROJECT_ROOT)}/<kb>/)",
    )

    reembed = commands.add_parser(
        "reembed", help="Recompute embeddings after changing EMBEDDING_MODEL / EMBEDDING_DIM"
    )
    reembed.add_argument("kb")

    reset = commands.add_parser("reset", help="Drop the knowledge base's database")
    reset.add_argument("kb")
    reset.add_argument("--yes", action="store_true", help="Confirm deleting all its data")
    return parser


async def list_kbs() -> int:
    try:
        server_url = server_url_from_env()
    except ValueError:
        server_url = None
    for spec in KNOWLEDGE_BASES.values():
        state = "unknown"
        if server_url:
            try:
                state = "exists" if await database_exists(server_url, spec.database) else "not created"
            except DatabaseAdminError:
                state = "server unavailable"
        print(f"{spec.name}\tdatabase={spec.database} ({state})\t{spec.description.strip()}")
    return 0


async def ingest(spec: KnowledgeBaseSpec, args: argparse.Namespace) -> int:
    dataset = spec.dataset
    path = args.file or (dataset.path if dataset else None)
    if path is None:
        raise ValueError(f"Knowledge base {spec.name!r} has no default dataset; pass --file")
    inputs = read_transcripts(
        path,
        transcript_column=args.transcript_column
        or (dataset.transcript_column if dataset else DEFAULT_TRANSCRIPT_COLUMN),
        stated_by_column=args.stated_by_column
        or (dataset.stated_by_column if dataset else DEFAULT_STATED_BY_COLUMN),
        stated_by=args.stated_by,
    )
    # Read (and validate) the input before creating the database.
    items = list(inputs)
    if not items:
        raise ValueError(f"Input file contains no transcripts: {path}")
    id_column = dataset.id_column if dataset else None
    if spec.retrieval is Retrieval.DOCUMENTS and not args.extract:
        documents = to_documents(
            items, id_column=id_column, title_column=dataset.title_column if dataset else None
        )
        return await index_kb_documents(spec, documents[: args.limit], reindex=args.reindex)
    context_columns = dataset.context_columns if dataset else ()

    extractor = KnowledgeExtractionService.from_env(
        spec.extract_prompt,
        max_tokens=EXTRACTION_MAX_TOKENS,
        # A typed output schema is also sent as the response format.
        json_schema=spec.extraction_output_model is not None,
        output_model=spec.extraction_output_model,
        ontology=spec.ontology,
    )
    verifier = KnowledgeVerifier.from_env(args.verify)
    store = await KnowledgeStore.for_kb(spec)
    concurrency = args.concurrency or args.batch_size
    succeeded = failed = knowledge_items = 0
    started = time.perf_counter()
    try:
        if id_column and not args.no_resume:
            # Benchmark runs ingest the same rows into the same tables.
            done = await store.processed_metadata_values(
                id_column, without_key=DEFAULT_ISOLATION_KEY
            )
            remaining = [item for item in items if item.metadata.get(id_column) not in done]
            if skipped := len(items) - len(remaining):
                print(f"[{spec.name}] Skipping {skipped} already processed rows ({id_column})")
            items = remaining
        if args.limit:
            items = items[: args.limit]
        ingestion = KnowledgeIngestionService(
            extractor, store,
            extraction_metadata={"version": spec.extraction_version, **extractor.provenance},
            ontology=spec.ontology,
            normalizer=spec.record_normalizer,
            verifier=verifier,
            consolidator=None if args.no_consolidate else KnowledgeConsolidator.from_env(store),
        )
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            results = await ingestion.ingest_batch(
                [
                    IngestionInput(
                        item.transcript,
                        stated_by=item.stated_by,
                        metadata={**item.metadata, "kb": spec.name},
                        context={
                            column: item.metadata[column]
                            for column in context_columns
                            if column in item.metadata
                        },
                    )
                    for item in batch
                ],
                concurrency=concurrency,
            )
            # Failed transcripts are marked failed in the DB; the rest are kept.
            for item, result in zip(batch, results, strict=True):
                if isinstance(result, IngestionResult) and result.duplicate_of:
                    succeeded += 1
                    print(f"{item.label}: already ingested (evidence {result.duplicate_of})")
                elif isinstance(result, IngestionResult):
                    succeeded += 1
                    knowledge_items += len(result.records)
                    actions = ", ".join(f"{n} {name}" for name, n in result.consolidation.items())
                    print(
                        f"{item.label}: {len(result.records)} knowledge items "
                        f"(evidence {result.evidence_id})" + (f"; {actions}" if actions else "")
                    )
                    for note in review_notes(result.records):
                        print(f"  {note}")
                else:
                    failed += 1
                    print(
                        f"{item.label}: FAILED {type(result).__name__}: {result}",
                        file=sys.stderr,
                    )
            if failed and args.fail_fast:
                break
    finally:
        await store.aclose()

    elapsed = time.perf_counter() - started
    print(
        f"[{spec.name}] Ingestion complete: {succeeded} succeeded, {failed} failed, "
        f"{knowledge_items} knowledge items in {elapsed:.1f}s "
        f"(batch size {args.batch_size}, concurrency {concurrency})"
    )
    return 1 if failed else 0


def review_notes(records: tuple[KnowledgeRecord, ...]) -> list[str]:
    """One line per record whose ontology or verification needs an expert."""

    notes = []
    for record in records:
        problems = []
        ontology = record.revision.get("ontology") or {}
        for section in ("unknown", "misplaced", "conflicts"):
            if ontology.get(section):
                problems.append(f"ontology {section}: {ontology[section]}")
        verify = record.revision.get("verify") or {}
        if verify and not verify.get("passed", True):
            details = verify.get("issues") or [verify.get("error") or verify.get("verdict")]
            problems.append(
                "verify" + (f" {verify['verdict']}" if verify.get("verdict") else "")
                + ": " + "; ".join(map(str, details))
            )
        if problems:
            notes.append(f"{record.type.value} {record.subject!r}: " + " | ".join(problems))
    return notes


async def index_kb_documents(
    spec: KnowledgeBaseSpec, documents: list[Document], *, reindex: bool = False
) -> int:
    """Chunk and embed the documents into spec.document_table; already indexed ids are skipped."""

    store = await open_document_store(spec)
    started = time.perf_counter()
    try:
        added = await index_documents(store, documents, spec.chunking, reindex=reindex)
        total = await store.count_chunks()
    finally:
        await store.aclose()
    print(
        f"[{spec.name}] Indexing complete: {added} chunks added, {total} in "
        f"{spec.database}.{spec.document_table} ({time.perf_counter() - started:.1f}s)"
    )
    return 0


async def evaluate(spec: KnowledgeBaseSpec, args: argparse.Namespace) -> int:
    if spec.retrieval is not Retrieval.DOCUMENTS:
        raise ValueError(
            f"{spec.name!r} uses {spec.retrieval.value} retrieval; evaluate needs documents"
        )
    if spec.retrieval_questions is None:
        raise ValueError(f"{spec.name!r} has no retrieval_questions")
    queries, qrels = load_retrieval_questions(spec.retrieval_questions)
    modes = [SearchMode(mode) for mode in args.modes]
    k_values = sorted({1, *args.k_values})
    store = await open_document_store(spec, query_instruction=args.query_instruction)
    try:
        chunks = await store.count_chunks()
        if not chunks:
            raise ValueError(f"{spec.document_table} is empty; run `kb ingest {spec.name}` first")
        indexed = await store.indexed_doc_ids()
        index_config = await store.config()
        hits = await retrieve(store, queries, modes, k=max(k_values))
    finally:
        await store.aclose()

    if unknown := sorted({doc for gold in qrels.values() for doc in gold} - indexed):
        print(f"Warning: gold ids not in the index: {', '.join(unknown)}", file=sys.stderr)
    summary, per_query = score_modes(qrels, hits, k_values)
    print(
        format_table(
            f"[{spec.name}] {len(queries)} questions, {len(indexed)} documents, "
            f"{chunks} chunks ({spec.document_table})",
            summary,
            k_values,
        )
    )
    # Questions whose top result is not a gold document, per mode.
    for mode in modes:
        misses = [qid for qid, metrics in per_query[mode.value].items() if metrics["MRR@1"] < 1]
        listed = f": {', '.join(misses)}" if misses else ""
        print(f"  {mode.value}: top result not gold for {len(misses)} questions{listed}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or EVAL_RUNS_DIR / spec.name / f"{spec.document_table}_{timestamp}.json"
    save_run(
        output,
        queries=queries,
        qrels=qrels,
        hits=hits,
        summary=summary,
        per_query=per_query,
        kb=spec.name,
        questions=str(spec.retrieval_questions.path.relative_to(PROJECT_ROOT))
        if spec.retrieval_questions.path.is_relative_to(PROJECT_ROOT)
        else str(spec.retrieval_questions.path),
        table=spec.document_table,
        chunks=chunks,
        index_config=index_config,
        query_instruction=(
            spec.embedding_query_instruction
            if args.query_instruction is None
            else args.query_instruction
        ),
        k_values=k_values,
    )
    print(f"Saved {output}")
    return 0


async def query(spec: KnowledgeBaseSpec, question: str) -> int:
    store = await open_store(spec)
    agent = None
    try:
        agent = KnowledgeAgentService.from_kb(spec, store)
        answer = await agent.ask(question)
    finally:
        if agent is not None:
            await agent.aclose()
        await store.aclose()

    print(answer.answer)
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            if isinstance(citation, DocumentCitation):
                title = f" {citation.title}" if citation.title else ""
                print(f"- [D:{citation.doc_id}]{title}")
                continue
            print(
                f"- [K:{citation.knowledge_id}] ({citation.type}) "
                f"{citation.subject}: {citation.claim} "
                f"— evidence {citation.evidence_id}"
            )
    return 0


async def reembed(spec: KnowledgeBaseSpec) -> int:
    if spec.retrieval is Retrieval.DOCUMENTS:
        # Chunks keep no copy outside the index: rebuild it from the dataset.
        print(
            f"[{spec.name}] Document retrieval: run `kb ingest {spec.name} --reindex`",
            file=sys.stderr,
        )
        return 2
    # init_schema would reject a column whose size no longer matches EMBEDDING_DIM.
    store = await KnowledgeStore.for_kb(spec, init_schema=False)
    try:
        count = await store.reembed_knowledge()
        # Recreates indexes (HNSW when the new dimension allows it) and
        # verifies the column now matches EMBEDDING_DIM.
        await store.init_schema()
    finally:
        await store.aclose()
    print(f"[{spec.name}] Re-embedded {count} knowledge items")
    return 0


async def reset(spec: KnowledgeBaseSpec, *, confirmed: bool) -> int:
    if not confirmed:
        print(
            f"This drops database {spec.database} and all of its knowledge; "
            "re-run with --yes to confirm.",
            file=sys.stderr,
        )
        return 2
    dropped = await drop_database(server_url_from_env(), spec.database)
    print(f"[{spec.name}] {'Dropped' if dropped else 'No'} database {spec.database}")
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.command == "list":
        return await list_kbs()
    spec = get_kb(args.kb)
    if args.command == "ingest":
        return await ingest(spec, args)
    if args.command == "query":
        return await query(spec, args.question)
    if args.command == "evaluate":
        return await evaluate(spec, args)
    if args.command == "reembed":
        return await reembed(spec)
    return await reset(spec, confirmed=args.yes)


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(level=logging.WARNING)
    try:
        return asyncio.run(run(args))
    except (
        OSError,
        ValueError,
        StorageError,
        DatabaseAdminError,
        KnowledgeAgentError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
