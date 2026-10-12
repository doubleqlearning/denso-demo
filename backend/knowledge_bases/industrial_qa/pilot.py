"""Run a small sample of the benchmark/ training questions end to end.

    uv run python -m backend.knowledge_bases.industrial_qa.pilot all --run-dir data/industrial_qa_runs/train-pilot10

Steps (each resumes from <run-dir>/run.json; `all` runs them in order):

- select: pick --sample questions of the train split (of one --source only, if
  given), one per conversation, spread over the knowledge_source x difficulty
  cells; or the questions listed in --question-ids, to rerun an earlier sample.
- ingest: index every manual chunk as a document (embeddings only), and
  extract knowledge from the sampled questions' conversations (plus
  --distractors other conversations of the split): Case Cards and lessons,
  normalized to the equipment catalog (catalog.py) and the failure-mode
  ontology, verified against the transcript and
  consolidated with the records of the same run only (--no-consolidate skips
  it).
- renormalize: rescope the stored records of --from-run-dir with the current
  equipment catalog and ontology, without extracting again, and copy its
  sample into a new --run-dir; then run answer and evaluate there to measure
  the catalog alone. The records are updated in place (they belong to the run
  id), so the old run keeps its result files but not its database state.
- answer: search the run's knowledge records (at most --per-conversation of one
  conversation) and the manual chunks for each question (its knowledge_source
  is not public), answer it with one LLM call, and write predictions.jsonl for
  benchmark/evaluate.py. The evidence ranking puts the sources the answer cites
  first, in citation order, then the others; predictions_uncited.jsonl keeps
  the ranking before that. Questions answered by an earlier attempt are not
  asked again.
- evaluate: score the sample with benchmark/evaluate.py (retrieval and MC;
  answers and extracted facts too when a judge is configured: MODELARK_*,
  else the EXTRACTION_* / OPENROUTER_* model) into results.json, plus the gold
  evidence among the turns of the top k records, the gold expert facts stated
  at a turn some extracted record points to, and ingestion quality
  (verification, scope, catalog changes, case/lesson overlap, consolidation
  errors).

The system under test reads only benchmark/public/; gold files are used to
choose the sample and to score it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.knowledge_bases.base import PROJECT_ROOT
from backend.knowledge_bases.industrial_qa import CATALOG, SPEC
from backend.knowledge_bases.industrial_qa.runs import overwrite_run, require_official_database
from backend.knowledge_bases.stores import open_document_store
from backend.scripts.kb import EXTRACTION_MAX_TOKENS, review_notes
from backend.src.agents.rag.chunking import Document
from backend.src.agents.rag.indexing import index_documents
from backend.src.services import llm
from backend.src.services.knowledge_consolidation import KnowledgeConsolidator
from backend.src.services.knowledge_extraction import KnowledgeExtractionService, KnowledgeType
from backend.src.services.knowledge_ingestion import (
    IngestionInput,
    IngestionResult,
    KnowledgeIngestionService,
)
from backend.src.services.knowledge_verification import KnowledgeVerifier, VerifyMode
from backend.src.storage import KnowledgeRecord, KnowledgeStore


BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
PUBLIC_DIR = BENCHMARK_DIR / "public"
GOLD_DIR = BENCHMARK_DIR / "benchmark"
GENERATION_DIR = BENCHMARK_DIR / "generation"
SPLITS = ("train",)
# The KB's query instruction asks for knowledge records; chunks are manual passages.
DOCUMENT_QUERY_INSTRUCTION = (
    "Given a question from a factory maintenance engineer, retrieve the "
    "equipment manual passage that answers it"
)
# Reasoning models spend part of this budget thinking before the JSON.
ANSWER_MAX_TOKENS = 8192
# Turns a lesson maps to when it quotes none (see record_turns).
LESSON_TURNS = 2
RETRIEVED_LIMIT = 20
# Knowledge hits fetched per requested record, so the per-conversation cap
# still leaves k records.
OVERFETCH = 2
# Context keys that are provenance or rendered elsewhere, not a record's scope.
UNSCOPED_CONTEXT = frozenset({"case_card", "steps", "source_turns", "benchmark_run_id"})
# Speaker roles whose turns carry expert knowledge; records point to these only
# (the others report symptoms or ask).
EXPERT_ROLES = frozenset({"maintenance_engineer", "controls_engineer", "senior_technician"})
_WORD_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_STOP_WORDS = frozenset(
    "a an and are as at be before but by can do for from has have if in is it "
    "its not of on or so that the then this to was we when with you your".split()
)


# ----------------------------------------------------------------- data files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A checkpoint survives interruption without leaving half a JSON document.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gold_questions(split: str) -> list[dict[str, Any]]:
    return read_jsonl(GOLD_DIR / f"{split}.jsonl")


def gold_conversations() -> dict[str, dict[str, Any]]:
    """Conversations with fact_turns: the turns stating each expert fact."""

    return {c["conversation_id"]: c for c in read_jsonl(GENERATION_DIR / "conversations.jsonl")}


def public_questions(split: str) -> dict[str, dict[str, Any]]:
    return {q["question_id"]: q for q in read_jsonl(PUBLIC_DIR / f"questions_{split}.jsonl")}


def public_conversations() -> dict[str, dict[str, Any]]:
    return {c["conversation_id"]: c for c in read_jsonl(PUBLIC_DIR / "conversations.jsonl")}


def corpus_chunks() -> dict[str, dict[str, Any]]:
    return {c["chunk_id"]: c for c in read_jsonl(PUBLIC_DIR / "corpus_chunks.jsonl")}


# --------------------------------------------------------------------- select


def select_questions(questions: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n questions from distinct conversations, round-robin over the cells.

    Cells are knowledge_source x difficulty; a seeded shuffle within each cell
    makes the sample reproducible.
    """

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: q["question_id"]):
        cells[(question["knowledge_source"], question["difficulty"])].append(question)
    rng = random.Random(seed)
    for members in cells.values():
        rng.shuffle(members)
    picked: list[dict[str, Any]] = []
    conversations: set[str] = set()
    while len(picked) < n:
        added = False
        for cell in sorted(cells):
            while cells[cell] and cells[cell][0]["conversation_id"] in conversations:
                cells[cell].pop(0)
            if cells[cell] and len(picked) < n:
                question = cells[cell].pop(0)
                picked.append(question)
                conversations.add(question["conversation_id"])
                added = True
        if not added:
            break
    return picked


def select(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_dir / "run.json"
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("split") != "train":
            raise SystemExit("Pilot runs require the train split; use a new --run-dir for this pilot")
        train = gold_questions("train")
        train_ids = {q["question_id"] for q in train}
        train_convs = {q["conversation_id"] for q in train}
        if (set(run["question_ids"]) - train_ids
                or set(run["conversation_ids"]) - train_convs):
            raise SystemExit("Pilot checkpoint contains inputs outside the train split; use a new --run-dir")
        print(f"Sample of {run['run_id']}: {len(run['question_ids'])} questions, "
              f"{len(run['conversation_ids'])} conversations (from {path})")
        return run
    if args.split != "train":
        raise SystemExit("Pilot runs require the train split")
    questions = gold_questions("train")
    if args.question_ids:
        by_id = {q["question_id"]: q for q in questions}
        wanted = args.question_ids.read_text(encoding="utf-8").split()
        if unknown := [qid for qid in wanted if qid not in by_id]:
            raise SystemExit(f"Not train questions: {', '.join(unknown)}")
        picked = [by_id[qid] for qid in wanted]
    else:
        picked = select_questions(
            [q for q in questions if args.source in (None, q["knowledge_source"])],
            args.sample, args.seed,
        )
    conversation_ids = [q["conversation_id"] for q in picked]
    others = sorted({q["conversation_id"] for q in questions} - set(conversation_ids))
    distractors = random.Random(args.seed).sample(others, min(args.distractors, len(others)))
    run = {
        "run_id": f"industrial-qa-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "seed": args.seed,
        "source": args.source,
        "question_ids": [q["question_id"] for q in picked],
        "conversation_ids": conversation_ids + distractors,
        "distractor_conversation_ids": distractors,
        "consolidate": not args.no_consolidate,
        "conversations": {},
    }
    save(path, run)
    (args.run_dir / "question_ids.txt").write_text(
        "".join(f"{qid}\n" for qid in run["question_ids"]), encoding="utf-8"
    )
    for question in picked:
        print(f"{question['question_id']} {question['knowledge_source']}/{question['difficulty']} "
              f"{question['conversation_id']}: {question['question']}")
    print(f"Selected {len(picked)} questions, {len(run['conversation_ids'])} conversations "
          f"({len(distractors)} distractors) into {path}")
    return run


# --------------------------------------------------------------------- ingest


def run_filter(run: dict[str, Any]) -> dict[str, Any]:
    return {"context.benchmark_run_id": {"$eq": run["run_id"]}}


def extracted(conversation: dict[str, Any]) -> list[KnowledgeRecord]:
    """Every record extracted from a processed conversation, merged ones included."""

    return [
        KnowledgeRecord.model_validate(record)
        for record in [*conversation["records"], *conversation.get("merged", [])]
    ]


def turn_label(turn_id: int) -> str:
    return f"T{turn_id:02d}"


def render_transcript(conversation: dict[str, Any]) -> str:
    """[T01] speaker (role): text lines, so Case Card quotes carry turn ids."""

    roles = {speaker["id"]: speaker["role"] for speaker in conversation["speakers"]}
    return "\n".join(
        f"[{turn_label(turn['turn_id'])}] {turn['speaker']} ({roles.get(turn['speaker'], 'unknown')}): "
        f"{turn['text']}"
        for turn in conversation["turns"]
    )


def expert_turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    roles = {speaker["id"]: speaker["role"] for speaker in conversation["speakers"]}
    return [turn for turn in conversation["turns"] if roles.get(turn["speaker"]) in EXPERT_ROLES]


def chunk_document(chunk: dict[str, Any]) -> Document:
    return Document(id=chunk["chunk_id"], title=" > ".join(chunk["section_path"]), text=chunk["text"],
                    metadata={"reference_doc_id": chunk["doc_id"], "page_start": chunk["page_start"],
                              "page_end": chunk["page_end"], "section_path": chunk["section_path"]})


async def ingest(args: argparse.Namespace, run: dict[str, Any], *, extractor=None) -> int:
    store = await KnowledgeStore.for_kb(require_official_database(SPEC))
    try:
        async with overwrite_run(store, run) as report:
            save(args.run_dir / "corpus_replacement.json", report)
            return await _ingest(args, run, extractor=extractor)
    finally:
        await store.aclose()


async def _ingest(args: argparse.Namespace, run: dict[str, Any], *, extractor=None) -> int:
    chunks = corpus_chunks()
    documents = await open_document_store(SPEC, query_instruction=DOCUMENT_QUERY_INSTRUCTION)
    try:
        await index_documents(documents, [chunk_document(c) for c in chunks.values()], SPEC.chunking)
    finally:
        await documents.aclose()

    conversations = public_conversations()
    remaining = [
        cid for cid in run["conversation_ids"]
        if run["conversations"].get(cid, {}).get("status") != "processed"
    ]
    if not remaining:
        print("All sampled conversations already extracted")
        return 0
    extractor = extractor or KnowledgeExtractionService.from_env(
        SPEC.extract_prompt,
        max_tokens=EXTRACTION_MAX_TOKENS,
        json_schema=True,
        output_model=SPEC.extraction_output_model,
        ontology=SPEC.ontology,
    )
    run["extraction"] = extractor.provenance
    store = await KnowledgeStore.for_kb(require_official_database(SPEC))
    try:
        # Scoped: a run must not merge into, or link, other runs' records.
        consolidator = (
            KnowledgeConsolidator.from_env(store, scope=run_filter(run))
            if run.get("consolidate") else None
        )
        service = KnowledgeIngestionService(
            extractor, store,
            extraction_metadata={"version": SPEC.extraction_version, **extractor.provenance},
            ontology=SPEC.ontology,
            normalizer=SPEC.record_normalizer,
            verifier=KnowledgeVerifier.from_env(args.verify),
            consolidator=consolidator,
        )
        results = await service.ingest_batch(
            [
                IngestionInput(
                    render_transcript(conversations[cid]),
                    metadata={"kb": SPEC.name, "conversation_id": cid,
                              "benchmark_run_id": run["run_id"]},
                    context={"benchmark_run_id": run["run_id"]},
                )
                for cid in remaining
            ],
            concurrency=args.concurrency,
        )
    finally:
        await store.aclose()
    for cid, result in zip(remaining, results, strict=True):
        if isinstance(result, IngestionResult):
            run["conversations"][cid] = {
                "status": "processed",
                "evidence_id": str(result.evidence_id),
                "records": [record.model_dump(mode="json") for record in result.records],
                "merged": [record.model_dump(mode="json") for record in result.merged],
                "consolidation": result.consolidation,
            }
            actions = ", ".join(f"{n} {name}" for name, n in result.consolidation.items())
            if errors := sum("error" in r.revision.get("consolidation", {}) for r in result.records):
                actions += f"; consolidation failed for {errors}"
            print(f"{cid}: {len(result.records)} records" + (f" ({actions})" if actions else ""))
            for note in review_notes((*result.records, *result.merged)):
                print(f"  {note}")
        else:
            run["conversations"][cid] = {"status": "failed", "error": str(result)}
            print(f"{cid}: FAILED {type(result).__name__}: {result}", file=sys.stderr)
    save(args.run_dir / "run.json", run)
    failed = sum(c["status"] != "processed" for c in run["conversations"].values())
    return 1 if failed else 0


# ---------------------------------------------------------------- renormalize

# Revision keys a re-normalization rewrites; links and history are kept.
SCOPE_REVISION_KEYS = ("catalog", "ontology")


def normalize_scope(transcript: str, records: list[KnowledgeRecord]) -> None:
    """The scope steps of ingestion: the record normalizer, then the ontology.

    The new ontology report keeps the spellings normalized before and replaces
    what still needs review.
    """

    if SPEC.record_normalizer is not None:
        SPEC.record_normalizer(transcript, records)
    if SPEC.ontology is None:
        return
    for record in records:
        record.context, report = SPEC.ontology.normalize(record.context)
        old = record.revision.get("ontology") or {}
        merged = {
            section: {**old.get(section, {}), **new}
            for section, new in report.to_dict().items()
        }
        for section in ("normalized", "implied"):
            if old.get(section) and section not in merged:
                merged[section] = old[section]
        if merged != old:
            record.revision = {**record.revision, "ontology": merged}


def start_renormalized_run(args: argparse.Namespace) -> None:
    """Copy the sample of --from-run-dir into a new --run-dir."""

    if args.from_run_dir is None:
        raise SystemExit("renormalize needs --from-run-dir, the run whose records to rescope")
    if (args.run_dir / "run.json").exists():
        return  # resumed
    source = args.from_run_dir / "run.json"
    if not source.is_file():
        raise SystemExit(f"No run.json in {args.from_run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads(source.read_text(encoding="utf-8"))
    run["renormalized_from"] = str(args.from_run_dir)
    save(args.run_dir / "run.json", run)
    if (args.from_run_dir / "question_ids.txt").is_file():
        shutil.copy(args.from_run_dir / "question_ids.txt", args.run_dir / "question_ids.txt")


async def renormalize(args: argparse.Namespace, run: dict[str, Any]) -> int:
    """Rescope the run's stored records with the current catalog and ontology.

    Updates the records in place (context, search text, embedding and the
    catalog/ontology revision) and mirrors them in run.json; extraction,
    verification and consolidation are not repeated. The records belong to
    the run id, so --from-run-dir keeps its result files but its database
    records change too.
    """

    conversations = public_conversations()
    keys = SPEC.embedding_context_keys
    totals: dict[str, int] = defaultdict(int)
    store = await KnowledgeStore.for_kb(require_official_database(SPEC))
    try:
        for cid, conversation in run["conversations"].items():
            if conversation.get("status") != "processed":
                continue
            rows = [*conversation["records"], *conversation.get("merged", [])]
            ids = [UUID(row["id"]) for row in conversation["records"]]
            stored = await store.get_knowledge(ids)
            if missing := len(ids) - len(stored):
                print(f"{cid}: {missing} records not in the database", file=sys.stderr)
            stored_ids = {record.id for record in stored}
            # Merged records are not stored but count in ingestion quality.
            group = [*stored, *(
                KnowledgeRecord.model_validate(row) for row in conversation.get("merged", [])
                if UUID(row["id"]) not in stored_ids
            )]
            before = {
                record.id: (dict(record.context), record.embedding_text(keys))
                for record in group
            }
            normalize_scope(render_transcript(conversations[cid]), group)
            changed = [
                record for record in group
                if (record.context, record.embedding_text(keys)) != before[record.id]
            ]
            updated = await store.update_knowledge_scope(
                [record for record in changed if record.id in stored_ids],
                revision_keys=SCOPE_REVISION_KEYS,
            )
            by_id = {str(record.id): record for record in group}
            for row in rows:
                if (record := by_id.get(row["id"])) is not None:
                    row["context"] = record.context
                    row["revision"] = {**row.get("revision", {}), **{
                        key: record.revision[key]
                        for key in SCOPE_REVISION_KEYS if key in record.revision
                    }}
            totals["records"] += len(group)
            totals["changed"] += len(changed)
            totals["updated"] += updated
            totals["with_machine_id"] += sum(bool(r.context.get("machine_id")) for r in group)
            print(f"{cid}: {len(changed)}/{len(group)} records rescoped, {updated} updated")
    finally:
        await store.aclose()
    run["renormalized"] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": CATALOG.sha256,
        "ontology_sha256": SPEC.ontology.sha256 if SPEC.ontology else None,
        **totals,
    }
    save(args.run_dir / "run.json", run)
    print("Renormalized: " + json.dumps(dict(totals)))
    return 0


# --------------------------------------------------------------------- answer


def tokens(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOP_WORDS}


def _by_overlap(text: str, turns: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(shared words, turn id) of every turn, most shared first, then by turn."""

    words = tokens(text)
    return sorted(
        ((len(words & tokens(turn["text"])), turn["turn_id"]) for turn in turns),
        key=lambda item: (-item[0], item[1]),
    )


def record_turns(
    record: KnowledgeRecord,
    turns: list[dict[str, Any]],
    question: str = "",
    *,
    limit: int = LESSON_TURNS,
) -> list[int]:
    """Conversation turns a record points to, for turn-level retrieval scoring.

    `turns` are the candidates (the expert turns, see expert_turns). A Case
    Card names the turns of its evidence quotes, a lesson its source_turns;
    they are ranked by the words they share with the question (a card quotes
    the symptoms before the expert's answer). A record naming no turn (one of
    an older extraction) maps to the `limit` turns sharing the most words with
    its claim.
    """

    evidence = (record.context.get("case_card") or {}).get("evidence") or []
    named = [item.get("turn_id") for item in evidence] + list(record.context.get("source_turns") or [])
    cited = {
        int(match.group(1))
        for label in named if (match := re.fullmatch(r"T0*(\d+)", str(label or "")))
    }
    cited_turns = [turn for turn in turns if turn["turn_id"] in cited]
    if cited_turns:
        return [turn_id for _, turn_id in _by_overlap(question, cited_turns)]
    ranked = _by_overlap(f"{record.subject} {record.claim}", turns)
    return [turn_id for overlap, turn_id in ranked[:limit] if overlap]


def covered_facts(
    records: list[KnowledgeRecord], turns: list[dict[str, Any]], fact_turns: dict[str, list[int]]
) -> list[str]:
    """Expert facts stated in a turn that some record points to (see record_turns)."""

    pointed = {turn for record in records for turn in record_turns(record, turns)}
    return [fact_id for fact_id, fact in fact_turns.items() if pointed & set(fact)]


def concatenate(
    rankings: list[list[dict[str, Any]]], limit: int | None = RETRIEVED_LIMIT
) -> list[dict[str, Any]]:
    """Items of each ranking in turn, dropping repeats."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in (item for ranking in rankings for item in ranking):
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged[:limit]


def interleave(*rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Alternate items of several rankings, dropping repeats."""

    width = max(map(len, rankings), default=0)
    return concatenate(
        [[ranking[i] for ranking in rankings if i < len(ranking)] for i in range(width)]
    )


def retrieved_ranking(
    record_items: list[list[dict[str, Any]]], chunk_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Record turns (in record order) and manual chunks alternately."""

    return interleave(concatenate(record_items, limit=None), chunk_items)


def cited_ranking(
    sources: dict[str, dict[str, Any]], cited: list[str], uncited: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Items of the cited sources in citation order, then the uncited ranking.

    The answer model has read every source, so its citations rerank them.
    """

    return concatenate([*(sources[label]["items"] for label in cited), uncited])


def cap_per_conversation(hits: list[Any], cap: int, k: int) -> list[Any]:
    """The first k hits, with at most `cap` of one conversation (0: no cap)."""

    kept: list[Any] = []
    per_conversation: dict[str, int] = defaultdict(int)
    for hit in hits:
        evidence = str(hit.record.evidence_id)
        if cap and per_conversation[evidence] >= cap:
            continue
        per_conversation[evidence] += 1
        kept.append(hit)
    return kept[:k]


class AnswerOutput(BaseModel):
    answer: str = Field(min_length=1)
    choice: Literal["A", "B", "C", "D"]
    sources: list[str] = Field(default_factory=list)


def render_record(label: str, record: KnowledgeRecord) -> str:
    scope = {
        key: value for key, value in record.context.items()
        if key not in UNSCOPED_CONTEXT and value not in (None, "", [])
    }
    head = f"[{label}] {record.type.value}: {record.subject}"
    if scope:
        head += " " + json.dumps(scope, ensure_ascii=False)
    return f"{head}\n{record.claim}"


def render_chunk(label: str, chunk: dict[str, Any]) -> str:
    pages = f"p. {chunk['page_start']}" + (
        f"-{chunk['page_end']}" if chunk["page_end"] != chunk["page_start"] else ""
    )
    section = " > ".join(chunk["section_path"])
    return f"[{label}] manual {chunk['doc_id']}, {pages}, {section}\n{chunk['text']}"


def answer_prompt(question: dict[str, Any], sources: list[str]) -> str:
    # Historical ablations load their own prompt only when explicitly used.
    # Importing the official benchmark adapters needs no experiment files.
    baseline = Path(__file__).with_name("baseline_answer.md").read_text(encoding="utf-8").strip()
    options = "\n".join(f"{letter}. {text}" for letter, text in sorted(question["options"].items()))
    return (
        baseline.replace("{QUESTION}", question["question"])
        .replace("{OPTIONS}", options)
        .replace("{SOURCES}", "\n\n".join(sources) or "(no sources found)")
    )


async def answer(
    args: argparse.Namespace, run: dict[str, Any], *, retrieve=None, generate=None, init_schema=True,
    checkpoint_every=0, official=True,
) -> int:
    if official:
        from backend.knowledge_bases.industrial_qa.benchmark_adapter import CompleteAnswer, ScopedRetrieval
        retrieve = retrieve if retrieve is not None else ScopedRetrieval()
        generate = generate if generate is not None else CompleteAnswer()
    processed = {cid: c for cid, c in run["conversations"].items() if c["status"] == "processed"}
    if missing := [cid for cid in run["conversation_ids"] if cid not in processed]:
        print(f"Warning: {len(missing)} conversations not extracted: {', '.join(missing)}",
              file=sys.stderr)
    by_evidence = {c["evidence_id"]: cid for cid, c in processed.items()}
    conversations = public_conversations()
    chunks = corpus_chunks()
    questions = public_questions(run["split"])
    client = llm.chat_client_from_env(
        "ANSWER",
        max_tokens=ANSWER_MAX_TOKENS,
        response_format=llm.json_schema_format("benchmark_answer", AnswerOutput),
    )
    store = await KnowledgeStore.for_kb(require_official_database(SPEC), init_schema=init_schema)
    documents = await open_document_store(
        SPEC, query_instruction=DOCUMENT_QUERY_INSTRUCTION, init_schema=init_schema
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def search(question: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        if retrieve is not None:
            return await retrieve(
                question, store, documents, run, args, by_evidence, conversations, chunks
            )
        knowledge_hits, chunk_hits = await asyncio.gather(
            store.search_knowledge(
                question, k=args.k * (OVERFETCH if args.per_conversation else 1),
                filters=run_filter(run),
            ),
            documents.search(question, k=args.k),
        )
        knowledge_hits = cap_per_conversation(knowledge_hits, args.per_conversation, args.k)
        sources: dict[str, dict[str, Any]] = {}
        record_items: list[list[dict[str, Any]]] = []
        for rank, hit in enumerate(knowledge_hits, start=1):
            record = hit.record
            cid = by_evidence.get(str(record.evidence_id))
            turn_ids = record_turns(record, expert_turns(conversations[cid]), question) if cid else []
            items = [{"type": "conversation", "id": cid, "turn_id": t} for t in turn_ids]
            record_items.append(items)
            sources[f"K{rank}"] = {
                "text": render_record(f"K{rank}", record), "items": items,
                "knowledge_id": str(record.id), "type": record.type.value,
                "subject": record.subject, "conversation_id": cid,
                "confident": hit.confident,
            }
        for rank, hit in enumerate(chunk_hits, start=1):
            sources[f"D{rank}"] = {
                "text": render_chunk(f"D{rank}", chunks[hit.doc_id]),
                "items": [{"type": "document", "id": hit.doc_id}], "chunk_id": hit.doc_id,
            }
        chunk_items = [{"type": "document", "id": hit.doc_id} for hit in chunk_hits]
        return sources, retrieved_ranking(record_items, chunk_items)

    async def solve(qid: str) -> dict[str, Any]:
        question = questions[qid]
        async with semaphore:
            sources, retrieved = await search(question["question"])
            if generate is None:
                prompt = answer_prompt(question, [s["text"] for s in sources.values()])
                response = await llm.ainvoke(client, prompt)
                output = llm.validated(
                    AnswerOutput,
                    llm.response_json(response, task="answer", env_prefix="ANSWER"),
                    task="answer",
                )
                trace = {"usage": getattr(response, "usage_metadata", None)}
            else:
                output, trace = await generate(question, sources, client)
        print(f"{qid}: {output.choice} ({len(sources)} sources)")
        cited = list(dict.fromkeys(label for label in output.sources if label in sources))
        outcome = {
            "question_id": qid,
            "question": question["question"],
            "answer": output.answer,
            "choice": output.choice,
            "cited": cited,
            "retrieved": cited_ranking(sources, cited, retrieved),
            "retrieved_uncited": retrieved,
            "sources": {label: {k: v for k, v in s.items() if k != "text"}
                        for label, s in sources.items()},
            "context_characters": sum(len(s["text"]) for s in sources.values()),
            "answer_trace": trace,
        }
        if checkpoint_every:
            done[qid] = outcome
            if len(done) % checkpoint_every == 0:
                save(path, [done[q] for q in run["question_ids"] if q in done])
        return outcome

    # Answers of an earlier attempt are kept; only missing questions are asked.
    path = args.run_dir / "answers.json"
    done = {a["question_id"]: a for a in json.loads(path.read_text())} if path.exists() else {}
    remaining = [qid for qid in run["question_ids"] if qid not in done]
    try:
        outcomes = await asyncio.gather(*(solve(qid) for qid in remaining), return_exceptions=True)
    finally:
        await store.aclose()
        await documents.aclose()
    for qid, outcome in zip(remaining, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            print(f"{qid}: FAILED {type(outcome).__name__}: {outcome}", file=sys.stderr)
        else:
            done[qid] = outcome
    answers = [done[qid] for qid in run["question_ids"] if qid in done]
    save(path, answers)
    extractions = [
        # Every extracted record is one fact; case claims render the whole card.
        {"type": "extraction", "conversation_id": cid,
         "facts": [record.claim for record in extracted(processed[cid])]}
        for cid in run["conversation_ids"] if cid in processed
    ]
    for name, ranking in (("predictions", "retrieved"), ("predictions_uncited", "retrieved_uncited")):
        write_jsonl(args.run_dir / f"{name}.jsonl", [
            {"type": "answer", "question_id": a["question_id"], "answer": a["answer"],
             "choice": a["choice"], "retrieved": a.get(ranking, a["retrieved"])}
            for a in answers
        ] + extractions)
    print(f"Answered {len(answers)}/{len(run['question_ids'])}; wrote "
          f"{args.run_dir / 'predictions.jsonl'}")
    return 0 if len(answers) == len(run["question_ids"]) else 1


# ------------------------------------------------------------------- evaluate


def judge_env() -> dict[str, str] | None:
    """Environment for benchmark/evaluate.py's judge, None when no model is set.

    MODELARK_* wins; otherwise the judge is the EXTRACTION_* (else
    OPENROUTER_*) model, which may also be the model under test.
    """

    def setting(name: str) -> str | None:
        return os.getenv(f"EXTRACTION_{name}") or os.getenv(f"OPENROUTER_{name}")

    judge = {
        "MODELARK_BASEURL": os.getenv("MODELARK_BASEURL")
        or setting("BASE_URL") or llm.DEFAULT_LLM_BASE_URL,
        "MODELARK_APIKEY": os.getenv("MODELARK_APIKEY") or setting("API_KEY"),
        "MODELARK_MODELNAME": os.getenv("MODELARK_MODELNAME") or setting("MODEL"),
    }
    if not all(judge.values()):
        return None
    return {**os.environ, **judge}  # type: ignore[dict-item]


def run_evaluator(
    args: argparse.Namespace, run: dict[str, Any], predictions: str, out: Path,
    env: dict[str, str] | None, *, facts: bool = True,
) -> int:
    command = [
        sys.executable, str(BENCHMARK_DIR / "evaluate.py"),
        "--predictions", str(args.run_dir / predictions),
        "--split", run["split"],
        "--question-ids", str(args.run_dir / "question_ids.txt"),
        "--k", "1,3,5,10",
        "--out", str(out),
    ]
    if env is None:
        command.append("--skip-llm")
    elif facts:
        command.append("--facts")
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False, env=env).returncode


def evaluate(args: argparse.Namespace, run: dict[str, Any], *, facts: bool = True) -> int:
    """facts=False judges answers only, for runs whose extraction did not change."""
    results = args.run_dir / "results.json"
    env = None if args.skip_llm else judge_env()
    if env is None and not args.skip_llm:
        print("No judge model is configured: scoring retrieval and MC only")
    elif env is not None:
        print(f"Judge: {env['MODELARK_MODELNAME']}")
    if code := run_evaluator(args, run, "predictions.jsonl", results, env, facts=facts):
        return code
    scores = json.loads(results.read_text(encoding="utf-8"))
    uncited = args.run_dir / "results_uncited.json"
    if (args.run_dir / "predictions_uncited.jsonl").exists():
        # Retrieval before the citation rerank; the answers are the same.
        if code := run_evaluator(args, run, "predictions_uncited.jsonl", uncited, None):
            return code
        scores["uncited_retrieval"] = json.loads(uncited.read_text(encoding="utf-8"))["overall"]
    answers = json.loads((args.run_dir / "answers.json").read_text(encoding="utf-8"))
    if answers:
        scores["record_recall"] = record_recall(answers, run["split"])
        print("Record recall: " + " | ".join(
            f"{name} {value:.3f}" for name, value in scores["record_recall"].get("overall", {}).items()
        ))
    scores["extraction_turn_coverage"] = extraction_coverage(run)
    scores["ingestion_quality"] = ingestion_quality(run)
    if answers:
        scores["retrieval_diversity"] = retrieval_diversity(answers)
    save(results, scores)
    coverage = scores["extraction_turn_coverage"]
    print(f"Extraction: {coverage['records']} records; expert facts stated at a turn a record "
          f"points to: {coverage['covered']}/{coverage['expert_facts']}")
    print("Ingestion: " + json.dumps(scores["ingestion_quality"]["summary"]))
    return 0


def record_recall(
    answers: list[dict[str, Any]], split: str, ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, Any]:
    """Share of a question's gold evidence among the turns of its top k records.

    Unlike the turn ranking, a record counts as one result whatever the number
    of turns it points to.
    """

    gold = {q["question_id"]: q for q in gold_questions(split)}
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for answer_ in answers:
        question = gold[answer_["question_id"]]
        evidence = question["required_evidence"]
        records = [s for label, s in answer_["sources"].items() if label.startswith("K")]
        row = {}
        for k in ks:
            pointed = {(item["id"], item.get("turn_id")) for s in records[:k] for item in s["items"]}
            hits = sum((item["id"], item.get("turn_id")) in pointed for item in evidence)
            row[f"recall@{k}"] = hits / len(evidence) if evidence else 0.0
        rows["overall"].append(row)
        rows[question["knowledge_source"]].append(row)
    return {
        group: {name: sum(r[name] for r in group_rows) / len(group_rows) for name in group_rows[0]}
        for group, group_rows in rows.items()
    }


def extraction_coverage(run: dict[str, Any]) -> dict[str, Any]:
    """Gold expert facts stated at a turn some extracted record points to."""

    turns = {cid: expert_turns(c) for cid, c in public_conversations().items()}
    fact_turns = {cid: c["fact_turns"] for cid, c in gold_conversations().items()}
    per_conversation = {}
    for cid, conversation in run["conversations"].items():
        if conversation.get("status") != "processed":
            continue
        records = extracted(conversation)
        per_conversation[cid] = {
            "records": len(records),
            "covered": covered_facts(records, turns[cid], fact_turns[cid]),
            "expert_facts": len(fact_turns[cid]),
        }
    return {
        "records": sum(c["records"] for c in per_conversation.values()),
        "covered": sum(len(c["covered"]) for c in per_conversation.values()),
        "expert_facts": sum(c["expert_facts"] for c in per_conversation.values()),
        "per_conversation": per_conversation,
    }


def ingestion_quality(run: dict[str, Any]) -> dict[str, Any]:
    """Verification, scope, case/lesson overlap and consolidation of the records."""

    records = [
        record for conversation in run["conversations"].values()
        if conversation.get("status") == "processed" for record in extracted(conversation)
    ]
    by_conversation: dict[str, list[KnowledgeRecord]] = defaultdict(list)
    for record in records:
        by_conversation[str(record.evidence_id)].append(record)
    overlaps = []
    for group in by_conversation.values():
        case_words = set().union(*(
            tokens(record.claim) for record in group if record.type is KnowledgeType.CASE
        ))
        for record in group:
            if record.type is not KnowledgeType.CASE and (words := tokens(record.claim)):
                overlaps.append(len(words & case_words) / len(words))
    lessons = [record for record in records if record.type is not KnowledgeType.CASE]
    consolidation = [record.revision.get("consolidation") or {} for record in lessons]
    types: dict[str, int] = defaultdict(int)
    for record in records:
        types[record.type.value] += 1
    summary = {
        "records": len(records),
        "verify_passed": sum(bool((r.revision.get("verify") or {}).get("passed")) for r in records),
        "scoped": sum(any(k not in UNSCOPED_CONTEXT for k in r.context) for r in records),
        "lessons_with_source_turns": sum(bool(r.context.get("source_turns")) for r in lessons),
        "lesson_case_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0,
        "lessons_mostly_in_case": sum(share >= 0.8 for share in overlaps),
        "consolidation_errors": sum("error" in note for note in consolidation),
        "with_machine_id": sum(bool(r.context.get("machine_id")) for r in records),
        "catalog_changed": sum(bool(r.revision.get("catalog")) for r in records),
        "catalog_inherited": sum(bool((r.revision.get("catalog") or {}).get("inherited")) for r in records),
        "ontology_unknown": sum(bool((r.revision.get("ontology") or {}).get("unknown")) for r in records),
    }
    actions: dict[str, int] = defaultdict(int)
    for note in consolidation:
        actions[note.get("action", "none")] += 1
    return {"summary": summary, "types": dict(types), "consolidation_actions": dict(actions)}


def retrieval_diversity(answers: list[dict[str, Any]], k: int = 10) -> dict[str, float]:
    """Mean distinct conversations among a question's top k records."""

    distinct = [
        len({s["conversation_id"] for label, s in a["sources"].items() if label.startswith("K")
             and int(label[1:]) <= k})
        for a in answers
    ]
    return {f"conversations@{k}": round(sum(distinct) / len(distinct), 3) if distinct else 0.0}


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=("select", "ingest", "renormalize", "answer", "evaluate", "all")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--from-run-dir", type=Path,
        help="renormalize: the run whose stored records to rescope; its sample is "
             "copied into a new --run-dir",
    )
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--sample", type=int, default=10, help="questions to run (default 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--question-ids", type=Path,
        help="file of train question ids to run instead of a sampled selection",
    )
    parser.add_argument(
        "--source", choices=("expert_only", "expert_and_docs", "docs_only"),
        help="only questions of this knowledge_source",
    )
    parser.add_argument(
        "--distractors", type=int, default=0,
        help="other conversations of the split to extract, so the sample's "
             "conversations are not the only knowledge searched",
    )
    parser.add_argument("--verify", choices=[m.value for m in VerifyMode], default="quotes")
    parser.add_argument("--k", type=int, default=8, help="records and chunks searched per question")
    parser.add_argument(
        "--per-conversation", type=int, default=3,
        help="most records of one conversation among the k (0: no cap)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--skip-llm", action="store_true", help="evaluate without the answer and fact judge"
    )
    parser.add_argument(
        "--no-consolidate", action="store_true",
        help="store each conversation's records as new (new runs only)",
    )
    return parser


async def run_async(args: argparse.Namespace) -> int:
    if args.command == "renormalize":
        start_renormalized_run(args)
    run = select(args)
    if args.command == "select":
        return 0
    if args.command == "renormalize":
        return await renormalize(args, run)
    if args.command in ("ingest", "all"):
        if code := await ingest(args, run):
            return code
    if args.command in ("answer", "all"):
        if code := await answer(args, run):
            return code
    if args.command in ("evaluate", "all"):
        return evaluate(args, run)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if (args.sample < 1 or args.k < 1 or args.concurrency < 1 or args.distractors < 0
            or args.per_conversation < 0):
        raise SystemExit("--sample, --k and --concurrency must be positive")
    load_dotenv(PROJECT_ROOT / ".env")
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
