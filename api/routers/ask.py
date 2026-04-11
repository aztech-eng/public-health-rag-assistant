from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.models.requests import AskRequest
from api.models.responses import AskResponse, CitationResponse, TimingsMs
from api.settings import DEFAULT_INSUFFICIENT_REASON, get_settings
from rag_mvp.answering import Citation, is_insufficient_evidence_answer
from rag_mvp.bm25 import SearchHit
from rag_mvp.pipeline import (
    DefaultGenerator,
    GenerationOptions,
    RAGPipeline,
    RetrievalOptions,
    infer_retriever_used,
)


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_DIR = ".rag/chroma"

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ask"])
_DISABLED_RERANK_MODES = {"none", "off", "disabled", "false", "0"}

ASK_REQUEST_EXAMPLES = {
    "supported": {
        "summary": "Supported Question",
        "description": "Typical request with standard free-text question input.",
        "value": {
            "question": "What disease notifications are highlighted?",
            "index_path": ".rag/index.json",
            "retriever": "hybrid",
            "top_k": 5,
            "rerank_mode": "lexical",
            "rerank_candidate_k": 20,
            "use_llm": False,
        },
    },
    "multiline": {
        "summary": "Multiline Question",
        "description": "Valid JSON-serialized multiline input. Newlines/tabs are accepted.",
        "value": {
            "question": "Please summarize:\n- major outbreaks\n- vaccination updates\tif present",
            "index_path": ".rag/index.json",
            "retriever": "hybrid",
            "top_k": 5,
            "rerank_mode": "lexical",
            "rerank_candidate_k": 20,
            "use_llm": False,
        },
    },
}


def _log_llm_fallback(exc: Exception) -> None:
    logger.warning("LLM generation failed; falling back to extractive mode: %s", exc)


pipeline = RAGPipeline(
    generator=DefaultGenerator(
        fallback_to_extractive=True,
        on_llm_error=_log_llm_fallback,
    )
)


@router.post("/ask", response_model=AskResponse)
def ask(
    request: Annotated[
        AskRequest,
        Body(openapi_examples=ASK_REQUEST_EXAMPLES),
    ]
) -> AskResponse:
    settings = get_settings()
    index_path = Path(request.index_path)
    chroma_dir = Path(DEFAULT_CHROMA_DIR)
    retrieval_options = RetrievalOptions(
        retrieval_mode=request.retriever,
        top_k=request.top_k,
        bm25_index_path=index_path,
        chroma_dir=chroma_dir,
        embed_model=DEFAULT_EMBED_MODEL,
        rerank_mode=request.rerank_mode,
        rerank_candidate_k=request.rerank_candidate_k,
    )
    generation_options = GenerationOptions(
        use_llm=request.use_llm,
        model=request.model,
        max_context_chars=request.max_context_chars,
        temperature=float(request.temperature),
    )

    retrieve_started = time.perf_counter()
    try:
        hits = pipeline.retrieve(
            question=request.question,
            options=retrieval_options,
        )
    except Exception as exc:
        retrieve_ms = _elapsed_ms(retrieve_started)
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Retrieval failed: {exc}",
                "timings_ms": {
                    "retrieve": retrieve_ms,
                    "generate": 0,
                    "total": retrieve_ms,
                },
            },
        ) from exc
    retrieve_ms = _elapsed_ms(retrieve_started)
    top_score = _top_score(hits)

    if not hits:
        return _insufficient_evidence_response(
            request=request,
            hits=hits,
            retrieve_ms=retrieve_ms,
            generate_ms=0,
            reason_hint="no_retrieval_hits",
            top_score=top_score,
        )
    if top_score <= 0.0:
        return _insufficient_evidence_response(
            request=request,
            hits=hits,
            retrieve_ms=retrieve_ms,
            generate_ms=0,
            reason_hint="non_positive_relevance_score",
            top_score=top_score,
        )
    if _is_rerank_enabled(request.rerank_mode) and top_score < settings.ask_min_evidence_score:
        return _insufficient_evidence_response(
            request=request,
            hits=hits,
            retrieve_ms=retrieve_ms,
            generate_ms=0,
            reason_hint="below_rerank_threshold",
            top_score=top_score,
        )

    generate_started = time.perf_counter()
    result = pipeline.generate(
        question=request.question,
        hits=hits,
        options=generation_options,
    )
    generate_ms = _elapsed_ms(generate_started)
    if is_insufficient_evidence_answer(result.answer) or not result.citations:
        return _insufficient_evidence_response(
            request=request,
            hits=hits,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            reason_hint="generation_insufficient_evidence",
            top_score=top_score,
        )

    total_ms = retrieve_ms + generate_ms
    return AskResponse(
        answer=result.answer,
        supported=True,
        citations=[_to_citation_response(c) for c in result.citations],
        reason=None,
        retriever_used=infer_retriever_used(request.retriever, hits),
        top_k=max(1, int(request.top_k)),
        timings_ms=TimingsMs(retrieve=retrieve_ms, generate=generate_ms, total=total_ms),
    )


def _to_citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        label=citation.label,
        source=citation.source,
        page=citation.page,
        chunk_id=citation.chunk_id,
        score=float(citation.score),
        preview=citation.preview,
    )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _top_score(hits: list[SearchHit]) -> float:
    if not hits:
        return 0.0
    return float(max(hit.score for hit in hits))


def _is_rerank_enabled(rerank_mode: str) -> bool:
    return str(rerank_mode).strip().lower() not in _DISABLED_RERANK_MODES


def _insufficient_evidence_response(
    *,
    request: AskRequest,
    hits: list[SearchHit],
    retrieve_ms: int,
    generate_ms: int,
    reason_hint: str,
    top_score: float,
) -> AskResponse:
    settings = get_settings()
    logger.info(
        "Returning insufficient-evidence response (reason=%s, hits=%d, top_score=%.3f, threshold=%.3f)",
        reason_hint,
        len(hits),
        float(top_score),
        settings.ask_min_evidence_score,
    )
    total_ms = retrieve_ms + generate_ms
    return AskResponse(
        answer=settings.ask_insufficient_answer,
        supported=False,
        citations=[],
        reason=DEFAULT_INSUFFICIENT_REASON,
        retriever_used=infer_retriever_used(request.retriever, hits),
        top_k=max(1, int(request.top_k)),
        timings_ms=TimingsMs(retrieve=retrieve_ms, generate=generate_ms, total=total_ms),
    )
