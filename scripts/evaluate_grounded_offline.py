from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "no information",
    "not enough information",
    "cannot determine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded-answer quality offline using the grounded eval set and prediction file."
    )
    parser.add_argument("--eval-set", default="evaluation/hpr_grounded_eval_offline.json")
    parser.add_argument("--predictions", required=True, help="JSON file containing model answers and citations.")
    parser.add_argument("--output-dir", default="evaluation/out/grounded_offline")
    parser.add_argument("--weights", default=None, help="Optional JSON object or file for metric weights.")
    return parser.parse_args()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _contains_abstain(answer: str) -> bool:
    low = _normalize_text(answer)
    return any(pattern in low for pattern in ABSTAIN_PATTERNS)


def _keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    low = _normalize_text(answer)
    found = sum(1 for kw in keywords if _normalize_text(kw) and _normalize_text(kw) in low)
    return found / len(keywords)


def _citation_presence(answer: str, citations: list[dict]) -> float:
    if citations:
        return 1.0
    return 1.0 if re.search(r"\[\d+\]", str(answer or "")) else 0.0


def _citation_source_recall(pred_citations: list[dict], expected_evidence: list[dict]) -> float | None:
    expected_sources = {str(item.get("source", "")).strip() for item in expected_evidence if item.get("source")}
    if not expected_sources:
        return None
    pred_sources = {str(item.get("source", "")).strip() for item in pred_citations if item.get("source")}
    if not pred_sources:
        return 0.0
    found = sum(1 for src in expected_sources if src in pred_sources)
    return found / len(expected_sources)


def _score_total(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    total = 0.0
    total_weight = 0.0
    for metric, value in scores.items():
        if value is None:
            continue
        weight = float(weights.get(metric, 1.0))
        total += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total / total_weight


def _load_weights(raw_weights: str | None) -> dict[str, float]:
    base = {
        "abstain_accuracy": 0.35,
        "citation_presence": 0.2,
        "citation_source_recall": 0.25,
        "keyword_coverage": 0.2,
    }
    if not raw_weights:
        return base
    raw = raw_weights.strip()
    if raw.startswith("{"):
        base.update(json.loads(raw))
        return base
    path = Path(raw)
    base.update(json.loads(path.read_text(encoding="utf-8")))
    return base


def evaluate(eval_set: dict, predictions: list[dict], weights: dict[str, float]) -> tuple[list[dict], dict]:
    pred_by_question = {str(item.get("question", "")).strip(): item for item in predictions}
    per_question: list[dict] = []
    grouped_scores: dict[str, list[dict[str, float | None]]] = defaultdict(list)

    for item in eval_set.get("records", []):
        question = str(item.get("question", "")).strip()
        expected_behavior = str(item.get("expected_behavior", "answer"))
        required_keywords = [str(x) for x in item.get("required_keywords", [])]
        expected_evidence = list(item.get("evidence", []))
        qtype = str(item.get("type", "unknown"))

        pred = pred_by_question.get(question, {})
        pred_answer = str(pred.get("answer", ""))
        pred_citations = list(pred.get("citations", []))

        abstain_pred = _contains_abstain(pred_answer)
        if expected_behavior == "abstain":
            abstain_accuracy = 1.0 if abstain_pred else 0.0
            keyword_cov = None
            source_recall = None
        else:
            abstain_accuracy = 0.0 if abstain_pred else 1.0
            keyword_cov = _keyword_coverage(pred_answer, required_keywords)
            source_recall = _citation_source_recall(pred_citations, expected_evidence)

        scores = {
            "abstain_accuracy": abstain_accuracy,
            "citation_presence": _citation_presence(pred_answer, pred_citations),
            "citation_source_recall": source_recall,
            "keyword_coverage": keyword_cov,
        }
        total = _score_total(scores, weights)
        scores["total_score"] = total

        row = {
            "question": question,
            "type": qtype,
            "expected_behavior": expected_behavior,
            "prediction_found": bool(pred),
            "answer": pred_answer,
            "citations": pred_citations,
            "scores": scores,
        }
        per_question.append(row)
        grouped_scores[qtype].append(scores)

    summary: dict[str, dict[str, float | None]] = {}
    for qtype, score_rows in grouped_scores.items():
        bucket: dict[str, float | None] = {}
        metric_names = ("abstain_accuracy", "citation_presence", "citation_source_recall", "keyword_coverage", "total_score")
        for metric in metric_names:
            vals = [row.get(metric) for row in score_rows if row.get(metric) is not None]
            bucket[metric] = (sum(vals) / len(vals)) if vals else None
        summary[qtype] = bucket

    overall_metric_names = ("abstain_accuracy", "citation_presence", "citation_source_recall", "keyword_coverage", "total_score")
    overall: dict[str, float | None] = {}
    for metric in overall_metric_names:
        vals = [row["scores"].get(metric) for row in per_question if row["scores"].get(metric) is not None]
        overall[metric] = (sum(vals) / len(vals)) if vals else None
    summary["overall"] = overall

    return per_question, summary


def main() -> int:
    args = parse_args()
    eval_set = _load_json(Path(args.eval_set))
    predictions = _load_json(Path(args.predictions))
    if not isinstance(predictions, list):
        raise ValueError("--predictions must point to a JSON array of result objects.")

    weights = _load_weights(args.weights)
    results, summary = evaluate(eval_set=eval_set, predictions=predictions, weights=weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        for qtype, metrics in summary.items():
            handle.write(f"{qtype}:\n")
            for metric, value in metrics.items():
                if value is None:
                    handle.write(f"  {metric}: n/a\n")
                else:
                    handle.write(f"  {metric}: {value:.3f}\n")
            handle.write("\n")

    print(f"Wrote grounded offline evaluation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
