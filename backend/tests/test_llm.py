#!/usr/bin/env python3
"""Run the knowledge extraction service against a benchmark conversation.

Examples:
    uv run --env-file .env python backend/tests/test_llm.py
    uv run --env-file .env python backend/tests/test_llm.py --case-id 7
    uv run --env-file .env python backend/tests/test_llm.py --conversation "Expert: ..."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.knowledge_bases import get_kb  # noqa: E402
from backend.src.services.knowledge_extraction import (  # noqa: E402
    KnowledgeExtractionError,
    KnowledgeExtractionService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured knowledge from a raw conversation."
    )
    parser.add_argument(
        "--kb",
        default="maintenance",
        help="Knowledge base whose extraction prompt and dataset to use.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--conversation",
        help="Raw conversation text. If omitted, a row is loaded from the CSV.",
    )
    source.add_argument(
        "--case-id",
        default="1",
        help="Benchmark case_id to load (default: 1).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Benchmark CSV path (default: the knowledge base's dataset).",
    )
    return parser


def load_benchmark_conversation(csv_path: Path, case_id: str) -> str:
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"case_id", "raw_conversation"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns: {', '.join(sorted(required_columns))}"
            )
        for row in reader:
            if row["case_id"] == case_id:
                return row["raw_conversation"]
    raise ValueError(f"case_id {case_id!r} was not found in {csv_path}")


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        spec = get_kb(args.kb)
        csv_path = args.csv or (spec.dataset.path if spec.dataset else None)
        if not args.conversation and csv_path is None:
            raise ValueError(f"Knowledge base {spec.name!r} has no dataset; pass --csv")
        conversation = args.conversation or load_benchmark_conversation(
            csv_path, args.case_id
        )
        service = KnowledgeExtractionService.from_env(spec.extract_prompt)
        result = service.extract(conversation)
    except (OSError, ValueError, KnowledgeExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
