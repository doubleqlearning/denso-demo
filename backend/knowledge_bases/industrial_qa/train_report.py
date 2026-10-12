"""Report a completed full-train run from artifacts without external calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from backend.knowledge_bases.industrial_qa import pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path,
                        default=pilot.PROJECT_ROOT / "data/industrial_qa_runs/ablations-20261010-v1/paired-combined")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    run = json.loads((root / "run.json").read_text())
    results = json.loads((root / "results.json").read_text())
    answers = json.loads((root / "answers.json").read_text())
    train = pilot.gold_questions("train")
    if set(run["question_ids"]) != {q["question_id"] for q in train}:
        raise ValueError("Run does not cover every train question")
    if set(run["conversation_ids"]) != {q["conversation_id"] for q in train}:
        raise ValueError("Run does not cover every train conversation")
    if any(run["conversations"].get(cid, {}).get("status") != "processed" for cid in run["conversation_ids"]):
        raise ValueError("Ingestion is incomplete")
    if {a["question_id"] for a in answers} != set(run["question_ids"]) or results["n_answered"] != len(train):
        raise ValueError("Some train questions have no answer")
    reuse = json.loads((root / "inventory_after.json").read_text())
    if not reuse["unchanged"]:
        raise ValueError("Manual chunks or baseline records were modified")
    pct = lambda value: f"{value * 100:.2f}%"
    overall, uncited = results["overall"], results["uncited_retrieval"]
    rows = results["per_question"]
    incorrect = [r for r in rows if not r["answer_correct"]]
    contradictions = [r["question_id"] for r in rows if (r.get("judge") or {}).get("contradiction")]
    quality = results["ingestion_quality"]["summary"]
    fact = results["fact_capture"]
    lines = ["# Industrial QA — H1+H3 trên toàn bộ train split", "",
        f"Đã ingest đủ {len(run['conversation_ids'])} conversations train: tái sử dụng "
        f"{len(run['reused_conversation_ids'])} baseline conversations, extract/ingest mới "
        f"{len(run['new_conversation_ids'])} conversations. Đã trả lời và chấm "
        f"{results['n_answered']}/{len(train)} câu train. Không ingest conversation thuộc test.", "",
        f"Tái sử dụng {reuse['inventory']['manual_chunks']['rows']} manual chunks. Fingerprint của "
        "manual index và baseline records giống hệt trước/sau. Records mới nằm trong namespace "
        "riêng; retrieval đọc union baseline + full-train, consolidation chỉ được sửa namespace mới. "
        "Toàn bộ câu hỏi được retrieve/rerank và trả lời lại vì corpus đã tăng từ 20 lên 38 conversations.", "",
        "| Metric | Trước answer/citation | Sau citation |", "|---|---:|---:|"]
    for metric in ("recall@1", "recall@3", "recall@5", "recall@10", "context_precision@10"):
        lines.append(f"| {metric} | {pct(uncited[metric])} | {pct(overall[metric])} |")
    lines += ["", f"MC accuracy: **{pct(overall['mc_accuracy'])}**. Free-text correctness: "
        f"**{pct(overall['answer_correct'])}** ({len(train) - len(incorrect)}/{len(train)}). "
        f"Answer score: **{pct(overall['answer_score'])}**. Contradiction flags: {len(contradictions)}.", "",
        "| Knowledge source | Câu | Cited R@5 | Cited R@10 | MC accuracy | Free-text correctness | Answer score |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for source, metrics in results["by_knowledge_source"].items():
        values = [metrics[field] for field in ("recall@5", "recall@10", "mc_accuracy", "answer_correct", "answer_score")]
        lines.append(f"| {source} | {metrics['n']} | " + " | ".join(map(pct, values)) + " |")
    previous = json.loads((args.pilot_dir / "results.json").read_text())
    prior_questions = {r["question_id"] for r in previous["per_question"]}
    subset = [r for r in rows if r["question_id"] in prior_questions]
    summary = {field: statistics.mean(r[field] for r in subset)
               for field in ("recall@5", "recall@10", "answer_correct", "answer_score", "mc_accuracy")}
    changes = [{"question_id": row["question_id"],
                "before": next(r["answer_correct"] for r in previous["per_question"] if r["question_id"] == row["question_id"]),
                "after": row["answer_correct"]} for row in subset]
    pilot.save(root / "pilot_subset_comparison.json", {
        "pilot_path": str(args.pilot_dir.resolve()), "n": len(subset), "full_train_on_pilot_questions": summary,
        "original_pilot": previous["overall"], "correctness_changes": changes,
    })
    lines += ["", "**Đối chiếu trên đúng các câu pilot cũ**", "",
        f"Trên {len(subset)} câu pilot cũ khi retrieval từ đủ 38 conversations, free-text correctness "
        f"là {pct(summary['answer_correct'])}, so với {pct(previous['overall']['answer_correct'])} trước đó. "
        "Đây là một lượt chạy mới; corpus mở rộng và variance của LLM/reranker/judge đều có thể ảnh hưởng. "
        "Không so sánh trực tiếp overall của 100 câu với 20 câu để kết luận riêng tác động của corpus.", "",
        "**Ingestion và fact capture**", "",
        f"{quality['records']} extracted records (bao gồm merged nếu có); "
        f"{quality['verify_passed']} verify passed; consolidation errors {quality['consolidation_errors']}; "
        f"ontology unknown {quality['ontology_unknown']}. LLM fact precision {pct(fact['precision'])}, "
        f"recall {pct(fact['recall'])}. Fact metrics lần này được chấm trên toàn bộ train conversations, "
        "không lấy điểm từ pilot 20 conversations. Turn coverage và quote verification không bảo đảm "
        "mọi chi tiết trong claim đã được giữ đầy đủ.", "",
        "**Các câu chưa đạt free-text correctness**", ""]
    for row in incorrect:
        lines.append(f"- {row['question_id']} ({row['knowledge_source']}/{row['difficulty']}): "
                     f"score {row['answer_score']:.3f}; {(row.get('judge') or {}).get('reason', '')}")
    lines += ["", "Đây là điểm trên train split, chưa phải held-out test. Judge dùng cùng model family "
        "với extraction/answer; mỗi câu trong full train tương đương một điểm phần trăm accuracy.", "",
        "**Artifacts**", ""]
    for name in ("results.json", "results_uncited.json", "answers.json", "predictions.jsonl", "run.json",
                 "manifest.json", "inventory_before.json", "inventory_after.json", "pilot_subset_comparison.json"):
        lines.append(f"- [{name}]({root / name})")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    main()
