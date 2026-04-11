from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from .answering import AnswerResult, generate_answer, generate_answer_llm
from .bm25 import SearchHit
from .retrieval import retrieve_hits


RetrieverName = Literal["bm25", "vector", "hybrid"]
RerankMode = Literal["none", "lexical", "semantic", "auto"]


@dataclass(frozen=True)
class RetrievalOptions:
    retrieval_mode: RetrieverName = "hybrid"
    top_k: int = 5
    bm25_index_path: Path | None = None
    chroma_dir: Path | None = None
    embed_model: str | None = None
    hybrid_fusion: str = "rrf"
    hybrid_bm25_weight: float = 0.5
    hybrid_vector_weight: float = 0.5
    hybrid_rrf_k: int = 60
    hybrid_candidate_k: int = 20
    rerank_mode: RerankMode = "lexical"
    rerank_candidate_k: int = 20


@dataclass(frozen=True)
class GenerationOptions:
    use_llm: bool = False
    model: str = "gpt-4.1-mini"
    max_context_chars: int = 6000
    temperature: float = 0.2


class Retriever(Protocol):
    def retrieve(self, *, question: str, options: RetrievalOptions) -> list[SearchHit]:
        ...


class Generator(Protocol):
    def generate(self, *, question: str, hits: list[SearchHit], options: GenerationOptions) -> AnswerResult:
        ...


class DefaultRetriever:
    def retrieve(self, *, question: str, options: RetrievalOptions) -> list[SearchHit]:
        return retrieve_hits(
            question=question,
            retrieval_mode=options.retrieval_mode,
            top_k=max(1, int(options.top_k)),
            bm25_index_path=options.bm25_index_path,
            chroma_dir=options.chroma_dir,
            embed_model=options.embed_model,
            hybrid_fusion=options.hybrid_fusion,
            hybrid_bm25_weight=options.hybrid_bm25_weight,
            hybrid_vector_weight=options.hybrid_vector_weight,
            hybrid_rrf_k=options.hybrid_rrf_k,
            hybrid_candidate_k=options.hybrid_candidate_k,
            rerank_mode=options.rerank_mode,
            rerank_candidate_k=max(1, int(options.rerank_candidate_k)),
        )


class DefaultGenerator:
    def __init__(
        self,
        *,
        fallback_to_extractive: bool = True,
        on_llm_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._fallback_to_extractive = fallback_to_extractive
        self._on_llm_error = on_llm_error

    def generate(self, *, question: str, hits: list[SearchHit], options: GenerationOptions) -> AnswerResult:
        if not options.use_llm:
            return generate_answer(question, hits)

        try:
            return generate_answer_llm(
                question=question,
                hits=hits,
                model=options.model,
                max_context_chars=max(1, int(options.max_context_chars)),
                temperature=float(options.temperature),
            )
        except Exception as exc:
            if self._on_llm_error is not None:
                self._on_llm_error(exc)
            if not self._fallback_to_extractive:
                raise
            return generate_answer(question, hits)


@dataclass(frozen=True)
class RAGRunResult:
    hits: list[SearchHit]
    answer: AnswerResult
    retriever_used: RetrieverName


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self._retriever = retriever or DefaultRetriever()
        self._generator = generator or DefaultGenerator()

    def retrieve(self, *, question: str, options: RetrievalOptions) -> list[SearchHit]:
        return self._retriever.retrieve(question=question, options=options)

    def generate(self, *, question: str, hits: list[SearchHit], options: GenerationOptions) -> AnswerResult:
        return self._generator.generate(question=question, hits=hits, options=options)

    def run(
        self,
        *,
        question: str,
        retrieval: RetrievalOptions,
        generation: GenerationOptions,
    ) -> RAGRunResult:
        hits = self.retrieve(question=question, options=retrieval)
        answer = self.generate(question=question, hits=hits, options=generation)
        retriever_used = infer_retriever_used(retrieval.retrieval_mode, hits)
        return RAGRunResult(hits=hits, answer=answer, retriever_used=retriever_used)


def infer_retriever_used(requested: RetrieverName, hits: list[SearchHit]) -> RetrieverName:
    if not hits:
        return requested
    retrievers = {hit.retriever for hit in hits if hit.retriever in {"bm25", "vector", "hybrid"}}
    if len(retrievers) == 1:
        only = next(iter(retrievers))
        if only == "bm25":
            return "bm25"
        if only == "vector":
            return "vector"
        if only == "hybrid":
            return "hybrid"
    return requested
