"""Build a self-contained visual source/gold/extraction reviewer from saved runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.knowledge_bases.maintenance.prepare import SOURCE, read_source, questions
from backend.knowledge_bases.maintenance.bench import validate_run_source


def build_report(run_dir: Path, output: Path) -> None:
    run = json.loads((run_dir / "extraction.json").read_text())
    source = read_source()
    validate_run_source(run, source)
    def optional(name):
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else None
    payload = {"source": source, "questions": questions(source), "run": {"cases": run["cases"]},
               "audit": optional("audit.json"), "documents": optional("documents_retrieval.json"),
               "knowledge": optional("knowledge_retrieval.json"),
               "manual": optional("manual_review.json"),
               "provenance": {k:v for k,v in run.items() if k not in ("cases", "generation_attempts")}}
    if payload["audit"] is None or "summary" not in payload["audit"]:
        raise ValueError("Finish an audit before generating the reviewer")
    payload["source"] = {"version": source["version"], "cases": source["cases"]}
    if payload["knowledge"]:
        payload["knowledge"]["per_query"] = {
            qid:{k:v for k,v in row.items() if k != "knowledge_hits"}
            for qid,row in payload["knowledge"]["per_query"].items()
        }
    template = (SOURCE / "review_template.html").read_text()
    # Source strings are rendered via textContent. Escape '<' so no source text
    # can terminate the application/json script element.
    content = template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run_dir, args.output)
