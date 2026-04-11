from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_mvp.bm25 import BM25Index, ChunkRecord, SearchHit, tokenize


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"https?://\S+")
DATE_TIME_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}(?:,\s*\d{1,2}:\d{2})?\b")
PAGE_COUNTER_RE = re.compile(r"\b\d+/\d+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline grounded-answer evaluation set from an existing "
            "question file and local BM25 index."
        )
    )
    parser.add_argument("--seed-file", default="evaluation/hpr_eval.json")
    parser.add_argument("--index-path", default=".rag/index.json")
    parser.add_argument("--output-file", default="evaluation/hpr_grounded_eval_offline.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--evidence-per-question", type=int, default=2)
    parser.add_argument("--max-quote-chars", type=int, default=220)
    return parser.parse_args()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_pdf_artifacts(text: str) -> str:
    cleaned = str(text or "")
    cleaned = URL_RE.sub("", cleaned)
    cleaned = DATE_TIME_RE.sub("", cleaned)
    cleaned = PAGE_COUNTER_RE.sub("", cleaned)
    cleaned = re.sub(r"\bHPR volume \d+ issue \d+[^.]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,-;")
    return cleaned.strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _strip_pdf_artifacts(_clean_text(text))
    if not cleaned:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    parts = [p for p in parts if len(p) >= 20]
    return parts if parts else [cleaned]


def _overlap_score(question_tokens: set[str], sentence: str) -> tuple[int, float]:
    sent_tokens = set(tokenize(sentence))
    overlap = len(question_tokens & sent_tokens)
    density = overlap / max(1, len(sent_tokens))
    return overlap, density


def _sentence_quality_penalty(sentence: str) -> float:
    lower = sentence.lower()
    penalty = 0.0
    if "gov.uk" in lower or "http" in lower or "www." in lower:
        penalty += 0.8
    if re.search(r"\b\d{2}/\d{2}/\d{4}\b", sentence):
        penalty += 0.8
    letters = sum(ch.isalpha() for ch in sentence)
    if letters < 20:
        penalty += 0.5
    if len(sentence) > 260:
        penalty += 0.3
    return penalty


def _pick_quote(question: str, chunk: ChunkRecord, max_chars: int) -> str:
    question_tokens = set(tokenize(question))
    sentences = _split_sentences(chunk.text)
    if not sentences:
        return ""

    ranked = sorted(sentences, key=lambda s: (_overlap_score(question_tokens, s), -_sentence_quality_penalty(s)), reverse=True)
    best = ranked[0]
    best = _strip_pdf_artifacts(best)
    if len(best) <= max_chars:
        return best
    return best[: max_chars - 3].rstrip() + "..."


def _unique_hits_by_source(hits: list[SearchHit], n: int) -> list[SearchHit]:
    out: list[SearchHit] = []
    seen_sources: set[str] = set()
    for hit in hits:
        source = hit.chunk.source
        if source in seen_sources:
            continue
        out.append(hit)
        seen_sources.add(source)
        if len(out) >= n:
            break
    return out


def _build_reference_answer(evidence: list[dict], expected_keywords: list[str]) -> str:
    if not evidence:
        return "I don't know."

    parts: list[str] = []
    for i, item in enumerate(evidence, start=1):
        quote = _strip_pdf_artifacts(_clean_text(str(item.get("quote", ""))))
        if quote:
            parts.append(f"{quote} [{i}]")
        if len(parts) >= 2:
            break

    answer = " ".join(parts[:2])
    if not answer:
        return "I don't know."

    if expected_keywords:
        lower_answer = answer.lower()
        missing = [kw for kw in expected_keywords if kw.lower() not in lower_answer]
        if len(missing) == len(expected_keywords):
            top_kw = expected_keywords[0]
            answer = f"{answer} This indicates {top_kw}."

    return answer


def build_grounded_set(
    *,
    seed_file: Path,
    index_path: Path,
    output_file: Path,
    top_k: int,
    evidence_per_question: int,
    max_quote_chars: int,
) -> dict:
    seed_data = json.loads(seed_file.read_text(encoding="utf-8"))
    index = BM25Index.load(index_path)

    records: list[dict] = []

    for i, item in enumerate(seed_data, start=1):
        question = str(item.get("question", "")).strip()
        qtype = str(item.get("type", "unknown"))
        expected_keywords = [str(x) for x in item.get("expected_keywords", [])]
        expected_behavior = "abstain" if bool(item.get("edge_case", False)) else "answer"

        hits = index.search(question, top_k=max(1, int(top_k)))
        selected_hits = _unique_hits_by_source(hits, max(1, int(evidence_per_question)))

        evidence = []
        for hit in selected_hits:
            evidence.append(
                {
                    "source": hit.chunk.source,
                    "page": hit.chunk.page,
                    "chunk_id": hit.chunk.id,
                    "retrieval_score": round(float(hit.score), 6),
                    "quote": _pick_quote(question, hit.chunk, max_chars=max_quote_chars),
                }
            )

        reference_answer = (
            "I don't know."
            if expected_behavior == "abstain"
            else _build_reference_answer(evidence, expected_keywords)
        )

        record = {
            "id": f"hpr-grounded-{i:03d}",
            "question": question,
            "type": qtype,
            "expected_behavior": expected_behavior,
            "reference_answer": reference_answer,
            "required_keywords": expected_keywords,
            "evidence": evidence,
            "rubric": {
                "must_be_grounded_in_evidence": True,
                "must_include_citation_markers": expected_behavior == "answer",
                "allow_abstention": expected_behavior == "abstain",
                "min_supported_claims": 0 if expected_behavior == "abstain" else 1,
            },
            "notes": (
                "Edge-case abstention test."
                if expected_behavior == "abstain"
                else "Answer should stay within cited evidence and avoid unsupported claims."
            ),
        }
        records.append(record)

    dataset = {
        "dataset_name": "hpr_grounded_eval_offline",
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_seed_file": str(seed_file.as_posix()),
        "index_snapshot": {
            "path": str(index_path.as_posix()),
            "created_at": index.created_at,
            "num_docs": index.num_docs,
        },
        "scoring_focus": [
            "groundedness",
            "citation correctness",
            "safe abstention",
        ],
        "records": records,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return dataset


def main() -> int:
    args = parse_args()
    dataset = build_grounded_set(
        seed_file=Path(args.seed_file),
        index_path=Path(args.index_path),
        output_file=Path(args.output_file),
        top_k=args.top_k,
        evidence_per_question=args.evidence_per_question,
        max_quote_chars=args.max_quote_chars,
    )
    print(
        "Built grounded eval set:",
        f"{args.output_file}",
        f"(records={len(dataset['records'])})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
