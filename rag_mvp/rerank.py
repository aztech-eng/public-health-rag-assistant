from __future__ import annotations

import math
from typing import Dict, List

from .bm25 import SearchHit, tokenize
from .embeddings import embed_texts


_DISABLED_RERANK_MODES = {"none", "off", "disabled", "false", "0"}


def rerank_hits(
    *,
    question: str,
    hits: List[SearchHit],
    top_k: int,
    rerank_mode: str = "lexical",
    embed_model: str | None = None,
) -> List[SearchHit]:
    """
    Re-rank retrieved candidates for stronger context relevance.

    Modes:
    - none: keep original ranking
    - lexical: token/bigram/phrase relevance reranking
    - semantic: embedding cosine similarity reranking (fallback to lexical on error)
    - auto: semantic when available, else lexical
    """
    deduped_hits = _dedupe_hits_by_chunk_id(hits)
    if not deduped_hits:
        return []

    mode = _normalize_mode(rerank_mode)
    limit = max(1, int(top_k))
    if mode == "none":
        return _reindex_hits(deduped_hits[:limit])

    base_scores = _normalize_scores_by_max(deduped_hits)
    lexical_scores = _lexical_scores(question=question, hits=deduped_hits)

    semantic_scores: Dict[str, float] = {}
    if mode in {"semantic", "auto"} and embed_model:
        try:
            semantic_scores = _semantic_scores(question=question, hits=deduped_hits, embed_model=embed_model)
        except Exception:
            semantic_scores = {}

    if mode == "semantic" and semantic_scores:
        combined = _combine_semantic(base_scores, lexical_scores, semantic_scores)
    elif mode == "auto" and semantic_scores:
        combined = _combine_semantic(base_scores, lexical_scores, semantic_scores)
    else:
        combined = _combine_lexical(base_scores, lexical_scores)

    ranked = sorted(
        deduped_hits,
        key=lambda hit: (
            combined.get(hit.chunk.id, 0.0),
            base_scores.get(hit.chunk.id, 0.0),
        ),
        reverse=True,
    )

    out: list[SearchHit] = []
    for rank, hit in enumerate(ranked[:limit], start=1):
        out.append(
            SearchHit(
                rank=rank,
                score=float(combined.get(hit.chunk.id, 0.0)),
                chunk=hit.chunk,
                retriever=hit.retriever,
            )
        )
    return out


def _normalize_mode(rerank_mode: str) -> str:
    mode = str(rerank_mode).strip().lower()
    if mode in _DISABLED_RERANK_MODES:
        return "none"
    if mode in {"semantic", "auto", "lexical"}:
        return mode
    return "lexical"


def _dedupe_hits_by_chunk_id(hits: List[SearchHit]) -> List[SearchHit]:
    best_by_id: dict[str, SearchHit] = {}
    order: list[str] = []

    for hit in hits:
        chunk_id = hit.chunk.id
        if chunk_id not in best_by_id:
            best_by_id[chunk_id] = hit
            order.append(chunk_id)
            continue
        if float(hit.score) > float(best_by_id[chunk_id].score):
            best_by_id[chunk_id] = hit

    return [best_by_id[cid] for cid in order]


def _normalize_scores_by_max(hits: List[SearchHit]) -> Dict[str, float]:
    if not hits:
        return {}
    max_score = max(float(hit.score) for hit in hits)
    if max_score <= 0.0:
        return {hit.chunk.id: 0.0 for hit in hits}
    return {hit.chunk.id: float(hit.score) / max_score for hit in hits}


def _lexical_scores(*, question: str, hits: List[SearchHit]) -> Dict[str, float]:
    query_tokens = tokenize(question)
    query_terms = set(query_tokens)
    query_bigrams = _bigrams(query_tokens)
    normalized_question = " ".join(question.lower().split())

    if not query_terms:
        return {hit.chunk.id: 0.0 for hit in hits}

    scores: dict[str, float] = {}
    for hit in hits:
        chunk_tokens = tokenize(hit.chunk.text)
        chunk_terms = set(chunk_tokens)
        overlap = len(query_terms & chunk_terms) / len(query_terms)

        chunk_bigrams = _bigrams(chunk_tokens)
        bigram_overlap = 0.0
        if query_bigrams:
            bigram_overlap = len(query_bigrams & chunk_bigrams) / len(query_bigrams)

        phrase_boost = 0.0
        if normalized_question and len(normalized_question) >= 8:
            normalized_chunk = " ".join(hit.chunk.text.lower().split())
            if normalized_question in normalized_chunk:
                phrase_boost = 1.0

        lexical_score = (0.65 * overlap) + (0.25 * bigram_overlap) + (0.10 * phrase_boost)
        scores[hit.chunk.id] = float(lexical_score)

    return scores


def _semantic_scores(*, question: str, hits: List[SearchHit], embed_model: str) -> Dict[str, float]:
    texts = [question, *[hit.chunk.text for hit in hits]]
    vectors = embed_texts(texts, model=embed_model)
    if len(vectors) != len(texts):
        return {}

    query_vec = vectors[0]
    query_norm = _l2_norm(query_vec)
    if query_norm <= 0.0:
        return {}

    scores: dict[str, float] = {}
    for hit, vector in zip(hits, vectors[1:]):
        similarity = _cosine_similarity(query_vec, vector, query_norm=query_norm)
        # Map cosine [-1, 1] to [0, 1] for stable blending with lexical/base signals.
        scores[hit.chunk.id] = max(0.0, min(1.0, 0.5 * (similarity + 1.0)))
    return scores


def _combine_lexical(
    base_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
) -> Dict[str, float]:
    ids = set(base_scores) | set(lexical_scores)
    combined: dict[str, float] = {}
    for chunk_id in ids:
        combined[chunk_id] = (0.30 * base_scores.get(chunk_id, 0.0)) + (0.70 * lexical_scores.get(chunk_id, 0.0))
    return combined


def _combine_semantic(
    base_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
    semantic_scores: Dict[str, float],
) -> Dict[str, float]:
    ids = set(base_scores) | set(lexical_scores) | set(semantic_scores)
    combined: dict[str, float] = {}
    for chunk_id in ids:
        combined[chunk_id] = (
            (0.10 * base_scores.get(chunk_id, 0.0))
            + (0.20 * lexical_scores.get(chunk_id, 0.0))
            + (0.70 * semantic_scores.get(chunk_id, 0.0))
        )
    return combined


def _bigrams(tokens: List[str]) -> set[str]:
    if len(tokens) < 2:
        return set()
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _l2_norm(vector: List[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def _cosine_similarity(query_vec: List[float], other_vec: List[float], *, query_norm: float) -> float:
    other_norm = _l2_norm(other_vec)
    if query_norm <= 0.0 or other_norm <= 0.0:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(query_vec, other_vec))
    return dot / (query_norm * other_norm)


def _reindex_hits(hits: List[SearchHit]) -> List[SearchHit]:
    out: list[SearchHit] = []
    for rank, hit in enumerate(hits, start=1):
        out.append(SearchHit(rank=rank, score=float(hit.score), chunk=hit.chunk, retriever=hit.retriever))
    return out
