from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .bm25 import SearchHit, tokenize

_INSUFFICIENT_EVIDENCE_MARKERS = (
    "i don't know",
    "i do not know",
    "could not find relevant context",
    "relevance score is too low",
    "insufficient evidence",
    "not enough evidence",
)

_CITATION_PREVIEW_MAX_LEN = 140
_PROPER_NOUN_CASE_MAP = {
    "uk": "UK",
    "england": "England",
    "nhs": "NHS",
    "ukhsa": "UKHSA",
}
_MID_SENTENCE_CONNECTORS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "via",
    "was",
    "were",
    "which",
    "with",
}


@dataclass
class Citation:
    label: str
    source: str
    chunk_id: str
    page: int | None
    score: float
    preview: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]


def generate_answer_llm(
    question: str,
    hits: list[SearchHit],
    model: str,
    max_context_chars: int,
    temperature: float,
) -> AnswerResult:
    if not hits:
        return AnswerResult(answer="I don't know.", citations=[])
    top_hits = hits[:5]
    if not _has_minimum_question_support(question, top_hits):
        return AnswerResult(answer="I don't know.", citations=_build_citations(top_hits))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from exc

    sources_block = _build_sources_block(top_hits, max_context_chars=max_context_chars)
    if not sources_block:
        return AnswerResult(answer="I don't know.", citations=_build_citations(top_hits))

    client = OpenAI(api_key=api_key)
    system_instruction = (
        "Answer ONLY from the provided sources. Do not infer beyond explicit evidence. "
        "If evidence is missing, indirect, or ambiguous, reply exactly: I don't know. "
        "Cite factual claims with references like [1], [2]."
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Sources:\n"
        f"{sources_block}\n\n"
        "Rules:\n"
        "- Use only the provided sources.\n"
        "- If the answer is not supported by the sources, reply exactly: I don't know.\n"
        "- If the question asks for a specific year/policy/treatment and that exact detail is absent, reply exactly: I don't know.\n"
        "- Cite factual claims with source references like [1] or [1][2].\n"
    )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        instructions=system_instruction,
        input=user_prompt,
    )
    answer_text = _extract_response_text(response).strip() or "I don't know."
    if not _passes_answer_guardrails(answer_text):
        answer_text = "I don't know."

    return AnswerResult(
        answer=answer_text,
        citations=_build_citations(top_hits),
    )


def generate_answer(question: str, hits: list[SearchHit], max_sentences: int = 4) -> AnswerResult:
    if not hits:
        return AnswerResult(
            answer="I could not find relevant context in the index.",
            citations=[],
        )

    top_score = hits[0].score
    if top_score <= 0.0:
        return AnswerResult(
            answer="I found context, but the relevance score is too low to answer confidently.",
            citations=_build_citations(hits[:3]),
        )

    query_terms = set(tokenize(question))
    candidates: list[tuple[float, str, int]] = []
    supplemental: list[tuple[float, str, int]] = []

    for i, hit in enumerate(hits[:5], start=1):
        for sentence in _split_sentences(hit.chunk.text):
            cleaned_sentence = _clean_sentence(sentence)
            if not cleaned_sentence:
                continue
            sentence_terms = set(tokenize(cleaned_sentence))
            overlap = len(query_terms & sentence_terms) if query_terms else 0
            is_title_like = (
                not re.search(r"[.!?]$", cleaned_sentence)
                and len(cleaned_sentence.split()) <= 12
            )
            if is_title_like:
                if overlap < 2:
                    continue
            if overlap == 0:
                if i <= 2 and not is_title_like:
                    supplemental.append((hit.score * 0.1, cleaned_sentence, i))
                continue
            density = overlap / max(1, len(sentence_terms))
            length_penalty = 0.2 if len(cleaned_sentence) > 280 else 0.0
            score = (overlap * 2.0) + (density * 2.0) + (hit.score * 0.25) - length_penalty
            candidates.append((score, cleaned_sentence, i))

    if not candidates:
        fallback = hits[0].chunk.text.strip()
        snippet = fallback[:400].rstrip() + ("..." if len(fallback) > 400 else "")
        return AnswerResult(
            answer=f"{snippet} [1]",
            citations=_build_citations(hits[:3]),
        )

    candidates.sort(key=lambda row: row[0], reverse=True)
    selected: list[tuple[str, int]] = []
    seen_norm: set[str] = set()

    current_chars = 0
    for _, sentence, citation_idx in candidates:
        norm = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not norm or norm in seen_norm:
            continue
        projected = current_chars + len(sentence) + 6
        if selected and projected > 650:
            continue
        selected.append((sentence, citation_idx))
        seen_norm.add(norm)
        current_chars = projected
        if len(selected) >= max_sentences:
            break

    if not selected:
        selected = [(hits[0].chunk.text[:240].strip(), 1)]
    elif len(selected) < 2 and supplemental:
        for _, sentence, citation_idx in sorted(supplemental, key=lambda row: row[0], reverse=True):
            norm = re.sub(r"\W+", " ", sentence.lower()).strip()
            if not norm or norm in seen_norm:
                continue
            selected.append((sentence, citation_idx))
            seen_norm.add(norm)
            break

    answer_parts = [f"{sentence} [{citation_idx}]" for sentence, citation_idx in selected]
    answer_text = " ".join(answer_parts)

    return AnswerResult(
        answer=answer_text,
        citations=_build_citations(hits[:5]),
    )


def _build_citations(hits: list[SearchHit]) -> list[Citation]:
    citations: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        preview = _build_citation_preview(hit.chunk.text)
        if len(preview) > _CITATION_PREVIEW_MAX_LEN:
            preview = preview[: _CITATION_PREVIEW_MAX_LEN - 3].rstrip() + "..."
        citations.append(
            Citation(
                label=f"[{i}]",
                source=hit.chunk.source,
                chunk_id=hit.chunk.id,
                page=hit.chunk.page,
                score=hit.score,
                preview=preview,
            )
        )
    return citations


def _build_citation_preview(raw_text: str) -> str:
    text = _normalize_preview_whitespace(raw_text)
    text = _clean_sentence(text)
    text = _shift_to_next_sentence_boundary(text)
    text = _fix_common_proper_nouns(text)
    text = _capitalize_sentence_starts(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_preview_whitespace(text: str) -> str:
    # Preserve content while smoothing common PDF extraction artifacts.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[\n\t\f\v]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_mid_sentence_start(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0] in {",", ";", ":", ")", "]", "}"}:
        return True
    match = re.match(r"[A-Za-z]+", stripped)
    if not match:
        return False
    first_word = match.group(0)
    if len(first_word) == 1 and first_word.islower():
        return True
    return first_word.islower() and first_word in _MID_SENTENCE_CONNECTORS


def _shift_to_next_sentence_boundary(text: str) -> str:
    if not _looks_like_mid_sentence_start(text):
        return text

    match = re.search(r'[.!?]\s+([\"\'(\[]?[A-Z])', text)
    if not match:
        return text

    candidate = text[match.start(1):].lstrip()
    if len(candidate) < 24:
        return text
    return candidate


def _fix_common_proper_nouns(text: str) -> str:
    for lowered, canonical in _PROPER_NOUN_CASE_MAP.items():
        text = re.sub(rf"\b{lowered}\b", canonical, text, flags=re.IGNORECASE)
    return text


def _capitalize_sentence_starts(text: str) -> str:
    text = re.sub(
        r'^([\"\'(\[]*)([a-z])',
        lambda match: f"{match.group(1)}{match.group(2).upper()}",
        text,
    )
    return re.sub(
        r'([.!?]\s+)([\"\'(\[]*)([a-z])',
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3).upper()}",
        text,
    )


def _build_sources_block(hits: list[SearchHit], max_context_chars: int) -> str:
    max_context_chars = max(1, int(max_context_chars))

    blocks: list[str] = []
    total_len = 0

    for i, hit in enumerate(hits, start=1):
        header = f"[{i}] source={hit.chunk.source} chunk={hit.chunk.id} score={hit.score:.3f}"
        text = hit.chunk.text.strip()
        block = f"{header}\n{text}" if text else header
        separator_len = 2 if blocks else 0

        if total_len + separator_len + len(block) <= max_context_chars:
            if blocks:
                total_len += 2
            blocks.append(block)
            total_len += len(block)
            continue

        if blocks:
            break

        # First source is too large: keep the full header, then truncate only the body.
        if not text:
            blocks.append(header)
            break

        min_needed = len(header) + 1
        if max_context_chars <= min_needed:
            break

        room_for_text = max_context_chars - min_needed
        truncated_text = text[:room_for_text].rstrip()
        if len(truncated_text) < len(text):
            truncated_text = truncated_text.rstrip() + "..."
        blocks.append(f"{header}\n{truncated_text}")
        break

    return "\n\n".join(blocks)


def _extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)

    return ""


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    sentences: list[str] = []
    for line in [ln.strip() for ln in text.split("\n") if ln.strip()]:
        if line.startswith("#"):
            sentences.append(line)
            continue
        normalized = re.sub(r"\s+", " ", line)
        sentences.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()])
    return sentences


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = text.replace("`", "")
    return text.strip()


def _has_minimum_question_support(question: str, hits: list[SearchHit]) -> bool:
    if not hits:
        return False

    question_text = str(question).lower()
    query_terms = set(tokenize(question_text))
    if not query_terms:
        return True

    year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", question_text)
    if year_tokens:
        evidence_text = " ".join(hit.chunk.text.lower() for hit in hits)
        if not any(year in evidence_text for year in year_tokens):
            return False

    max_overlap = 0
    for hit in hits:
        overlap = len(query_terms & set(tokenize(hit.chunk.text)))
        if overlap > max_overlap:
            max_overlap = overlap

    return max_overlap >= 2


def _passes_answer_guardrails(answer_text: str) -> bool:
    text = (answer_text or "").strip()
    if not text:
        return False
    if is_insufficient_evidence_answer(text) or "no information" in text.lower():
        return True
    return bool(re.search(r"\[\d+\]", text))


def is_insufficient_evidence_answer(answer_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer_text or "").strip().lower())
    if not normalized:
        return True
    return any(marker in normalized for marker in _INSUFFICIENT_EVIDENCE_MARKERS)
