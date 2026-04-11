from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rag_mvp.bm25 import BM25Index, SearchHit
from rag_mvp.rerank import rerank_hits
from rag_mvp.vector_store import query as vector_query


def retrieve_hits(
    question: str,
    retrieval_mode: str,
    top_k: int = 5,
    bm25_index_path: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    embed_model: Optional[str] = None,
    hybrid_fusion: str = "rrf",
    hybrid_bm25_weight: float = 0.5,
    hybrid_vector_weight: float = 0.5,
    hybrid_rrf_k: int = 60,
    hybrid_candidate_k: int = 20,
    rerank_mode: str = "lexical",
    rerank_candidate_k: int = 20,
) -> List[SearchHit]:
    """
    Unified retrieval function supporting bm25, vector, and hybrid modes.
    Returns a list of SearchHit objects.
    """
    retrieval_mode = retrieval_mode.lower()
    if retrieval_mode == "bm25":
        if bm25_index_path is None:
            raise ValueError("bm25_index_path must be provided for bm25 retrieval")
        index = BM25Index.load(bm25_index_path)
        candidate_k = _candidate_pool_size(
            top_k=top_k,
            rerank_mode=rerank_mode,
            rerank_candidate_k=rerank_candidate_k,
        )
        hits = index.search(question, top_k=candidate_k)
        return rerank_hits(
            question=question,
            hits=hits,
            top_k=top_k,
            rerank_mode=rerank_mode,
            embed_model=embed_model,
        )
    elif retrieval_mode == "vector":
        if chroma_dir is None or embed_model is None:
            raise ValueError("chroma_dir and embed_model must be provided for vector retrieval")
        candidate_k = _candidate_pool_size(
            top_k=top_k,
            rerank_mode=rerank_mode,
            rerank_candidate_k=rerank_candidate_k,
        )
        hits = vector_query(
            question=question,
            top_k=candidate_k,
            persist_dir=chroma_dir,
            embed_model=embed_model,
        )
        return rerank_hits(
            question=question,
            hits=hits,
            top_k=top_k,
            rerank_mode=rerank_mode,
            embed_model=embed_model,
        )
    elif retrieval_mode == "hybrid":
        if bm25_index_path is None:
            raise ValueError("bm25_index_path must be provided for hybrid retrieval")
        index = BM25Index.load(bm25_index_path)
        candidate_k = max(int(top_k), int(hybrid_candidate_k))
        if _is_rerank_enabled(rerank_mode):
            candidate_k = max(candidate_k, max(1, int(rerank_candidate_k)))
        bm25_hits = index.search(question, top_k=candidate_k)
        vector_hits = []
        if chroma_dir is not None and embed_model is not None:
            try:
                vector_hits = vector_query(
                    question=question,
                    top_k=candidate_k,
                    persist_dir=chroma_dir,
                    embed_model=embed_model,
                )
            except Exception:
                pass
        merged_hits = _merge_hybrid_hits(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            top_k=candidate_k,
            fusion=hybrid_fusion,
            bm25_weight=hybrid_bm25_weight,
            vector_weight=hybrid_vector_weight,
            rrf_k=hybrid_rrf_k,
        )
        return rerank_hits(
            question=question,
            hits=merged_hits,
            top_k=top_k,
            rerank_mode=rerank_mode,
            embed_model=embed_model,
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")


def _merge_hybrid_hits(
    bm25_hits: List[SearchHit],
    vector_hits: List[SearchHit],
    top_k: int,
    fusion: str,
    bm25_weight: float,
    vector_weight: float,
    rrf_k: int,
) -> List[SearchHit]:
    fusion_mode = str(fusion).lower()
    if fusion_mode == "legacy":
        return _merge_legacy_hits(bm25_hits=bm25_hits, vector_hits=vector_hits, top_k=top_k)
    if fusion_mode == "weighted":
        return _merge_weighted_hits(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
    if fusion_mode == "rrf":
        return _merge_rrf_hits(
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=rrf_k,
        )
    raise ValueError(f"Unknown hybrid fusion mode: {fusion}")


def _merge_legacy_hits(bm25_hits: List[SearchHit], vector_hits: List[SearchHit], top_k: int) -> List[SearchHit]:
    # Legacy behavior: prioritize BM25, then fill remaining slots with vector-only hits.
    seen = set(h.chunk.id for h in bm25_hits)
    merged = bm25_hits[:]
    for hit in vector_hits:
        if hit.chunk.id not in seen:
            merged.append(hit)
            seen.add(hit.chunk.id)
        if len(merged) >= top_k:
            break
    return merged[:top_k]


def _merge_weighted_hits(
    bm25_hits: List[SearchHit],
    vector_hits: List[SearchHit],
    top_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> List[SearchHit]:
    weight_bm25, weight_vector = _normalized_weights(bm25_weight, vector_weight)
    bm25_norm = _normalize_scores_by_max(bm25_hits)
    vector_norm = _normalize_scores_by_max(vector_hits)
    merged = _collect_chunks(bm25_hits, vector_hits)

    ranked: list[Tuple[float, str]] = []
    for chunk_id in merged:
        score = (weight_bm25 * bm25_norm.get(chunk_id, 0.0)) + (weight_vector * vector_norm.get(chunk_id, 0.0))
        if score > 0.0:
            ranked.append((score, chunk_id))
    ranked.sort(key=lambda item: item[0], reverse=True)

    return _build_ranked_hits(ranked=ranked, merged_chunks=merged, top_k=top_k)


def _merge_rrf_hits(
    bm25_hits: List[SearchHit],
    vector_hits: List[SearchHit],
    top_k: int,
    bm25_weight: float,
    vector_weight: float,
    rrf_k: int,
) -> List[SearchHit]:
    weight_bm25, weight_vector = _normalized_weights(bm25_weight, vector_weight)
    k = max(1, int(rrf_k))
    merged = _collect_chunks(bm25_hits, vector_hits)
    scores: dict[str, float] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + (weight_bm25 / (k + rank))
    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + (weight_vector / (k + rank))

    ranked = sorted(((score, chunk_id) for chunk_id, score in scores.items() if score > 0.0), reverse=True)
    return _build_ranked_hits(ranked=ranked, merged_chunks=merged, top_k=top_k)


def _build_ranked_hits(
    ranked: List[Tuple[float, str]],
    merged_chunks: Dict[str, object],
    top_k: int,
) -> List[SearchHit]:
    out: list[SearchHit] = []
    for rank, (score, chunk_id) in enumerate(ranked[: max(1, int(top_k))], start=1):
        chunk = merged_chunks.get(chunk_id)
        if chunk is None:
            continue
        out.append(SearchHit(rank=rank, score=float(score), chunk=chunk, retriever="hybrid"))
    return out


def _collect_chunks(bm25_hits: List[SearchHit], vector_hits: List[SearchHit]) -> Dict[str, object]:
    merged: dict[str, object] = {}
    for hit in bm25_hits:
        merged[hit.chunk.id] = hit.chunk
    for hit in vector_hits:
        merged[hit.chunk.id] = hit.chunk
    return merged


def _normalize_scores_by_max(hits: List[SearchHit]) -> Dict[str, float]:
    if not hits:
        return {}
    max_score = max(float(hit.score) for hit in hits)
    if max_score <= 0:
        return {hit.chunk.id: 0.0 for hit in hits}
    return {hit.chunk.id: float(hit.score) / max_score for hit in hits}


def _normalized_weights(bm25_weight: float, vector_weight: float) -> Tuple[float, float]:
    bw = max(0.0, float(bm25_weight))
    vw = max(0.0, float(vector_weight))
    total = bw + vw
    if total <= 0:
        return 0.5, 0.5
    return bw / total, vw / total


def _candidate_pool_size(*, top_k: int, rerank_mode: str, rerank_candidate_k: int) -> int:
    pool = max(1, int(top_k))
    if not _is_rerank_enabled(rerank_mode):
        return pool
    return max(pool, max(1, int(rerank_candidate_k)))


def _is_rerank_enabled(rerank_mode: str) -> bool:
    disabled = {"none", "off", "disabled", "false", "0"}
    return str(rerank_mode).strip().lower() not in disabled
