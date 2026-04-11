from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for offline mode
    load_dotenv = None

from .bm25 import BM25Index, SearchHit
from .ingestion import ingest_source_dir
from .pipeline import DefaultGenerator, GenerationOptions, RAGPipeline, RetrievalOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RAG MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a folder of .md/.txt/.pdf files into a BM25 index")
    ingest.add_argument("source_dir", type=Path, help="Directory containing source documents")
    ingest.add_argument("--index-path", type=Path, default=Path(".rag/index.json"), help="Where to write the index")
    ingest.add_argument("--chunk-chars", type=int, default=900, help="Target chunk size in characters")
    ingest.add_argument("--overlap-chars", type=int, default=120, help="Chunk overlap in characters")
    ingest.add_argument(
        "--retriever",
        choices=("bm25", "vector", "hybrid"),
        default="hybrid",
        help="Retriever mode to prepare at ingest time (bm25 skips vector indexing)",
    )
    ingest.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embeddings model for vector indexing/query",
    )
    ingest.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path(".rag/chroma"),
        help="Persistent Chroma directory for vector retrieval",
    )
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question against an existing index")
    ask.add_argument("question", type=str, help="Question to ask")
    ask.add_argument("--index-path", type=Path, default=Path(".rag/index.json"), help="Path to index JSON")
    ask.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    ask.add_argument(
        "--retriever",
        choices=("bm25", "vector", "hybrid"),
        default="hybrid",
        help="Retriever backend to use",
    )
    ask.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embeddings model for vector/hybrid retrieval",
    )
    ask.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path(".rag/chroma"),
        help="Persistent Chroma directory for vector retrieval",
    )
    ask.add_argument(
        "--rerank-mode",
        type=str,
        choices=("none", "lexical", "semantic", "auto"),
        default="lexical",
        help="Optional reranking strategy applied to retrieval candidates",
    )
    ask.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=20,
        help="Candidate pool size before reranking (>= top-k)",
    )
    ask.add_argument("--llm", action="store_true", help="Use OpenAI LLM generation instead of extractive mode")
    ask.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model for --llm mode")
    ask.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Max characters of retrieved source context sent to the LLM",
    )
    ask.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for --llm mode")
    ask.add_argument("--show-context", action="store_true", help="Print retrieved context chunks")
    ask.set_defaults(func=cmd_ask)

    stats = subparsers.add_parser("stats", help="Show index stats")
    stats.add_argument("--index-path", type=Path, default=Path(".rag/index.json"), help="Path to index JSON")
    stats.set_defaults(func=cmd_stats)

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        result = ingest_source_dir(
            source_dir=args.source_dir,
            index_path=args.index_path,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            retriever=args.retriever,
            embed_model=args.embed_model,
            chroma_dir=args.chroma_dir,
            allow_vector_failure=True,
        )
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    if result.vector_error is not None:
        print(_format_vector_index_warning(result.vector_error), file=sys.stderr)

    print(f"Indexed {result.files_indexed} files into {result.chunks_created} chunks")
    print(f"Index path: {args.index_path}")
    print(f"Avg tokens per chunk: {result.avg_tokens_per_chunk:.1f}")
    print(f"BM25 terms: {result.bm25_terms}")
    if args.retriever == "bm25":
        print("Vector index: skipped (--retriever bm25)")
    elif result.vector_indexed:
        print(f"Vector index: upserted {result.chunks_created} chunks into {args.chroma_dir}")
    else:
        print("Vector index: unavailable (BM25 index was still created)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    retrieval_options = RetrievalOptions(
        retrieval_mode=str(args.retriever),
        top_k=max(1, int(args.top_k)),
        bm25_index_path=getattr(args, "index_path", None),
        chroma_dir=getattr(args, "chroma_dir", None),
        embed_model=getattr(args, "embed_model", None),
        rerank_mode=str(getattr(args, "rerank_mode", "lexical")),
        rerank_candidate_k=max(1, int(getattr(args, "rerank_candidate_k", 20))),
    )
    generation_options = GenerationOptions(
        use_llm=bool(args.llm),
        model=str(args.model),
        max_context_chars=max(1, int(args.max_context_chars)),
        temperature=float(args.temperature),
    )
    pipeline = _build_cli_pipeline()

    try:
        hits = _retrieve_with_hybrid_fallback(
            pipeline=pipeline,
            question=args.question,
            options=retrieval_options,
        )
    except Exception as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1

    result = pipeline.generate(
        question=args.question,
        hits=hits,
        options=generation_options,
    )

    print(f"Q: {args.question}")
    print()
    print("Answer:")
    print(result.answer)
    print()

    if result.citations:
        print("Citations:")
        for citation in result.citations:
            page_part = f"p{citation.page}, " if citation.page is not None else ""
            print(
                f"  {citation.label} {citation.source} "
                f"({page_part}chunk_id={citation.chunk_id}, score={citation.score:.3f})"
            )
            print(f"      {citation.preview}")
    else:
        print("Citations: none")

    if args.show_context and hits:
        print()
        print("Retrieved Context:")
        for i, hit in enumerate(hits, start=1):
            retriever_part = f" | retriever={hit.retriever}" if hit.retriever else ""
            page_part = f" | page=p{hit.chunk.page}" if hit.chunk.page is not None else ""
            print(
                f"--- [{i}] {hit.chunk.source} | score={hit.score:.3f}{retriever_part}"
                f"{page_part} | chunk={hit.chunk.id}"
            )
            print(hit.chunk.text)
            print()

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    if not args.index_path.exists():
        print(f"Index not found: {args.index_path}", file=sys.stderr)
        return 1

    index = BM25Index.load(args.index_path)
    avg_chars = (sum(c.char_len for c in index.chunks) / max(1, len(index.chunks))) if index.chunks else 0.0
    avg_tokens = (sum(c.token_count for c in index.chunks) / max(1, len(index.chunks))) if index.chunks else 0.0
    unique_sources = len({c.source for c in index.chunks})

    print(f"Index path: {args.index_path}")
    print(f"Created: {index.created_at}")
    print(f"Chunks: {len(index.chunks)}")
    print(f"Sources: {unique_sources}")
    print(f"Terms: {len(index.postings)}")
    print(f"Avg chunk chars: {avg_chars:.1f}")
    print(f"Avg chunk tokens: {avg_tokens:.1f}")
    return 0


def _format_llm_warning(exc: Exception) -> str:
    detail_str, message = _describe_exception(exc)
    return (
        f"Warning: LLM generation failed ({detail_str}): {message}. "
        "Falling back to extractive mode."
    )


def _format_vector_index_warning(exc: Exception) -> str:
    detail_str, message = _describe_exception(exc)
    return (
        f"Warning: vector indexing failed ({detail_str}): {message}. "
        "Continuing with BM25 index only."
    )


def _format_hybrid_vector_warning(exc: Exception) -> str:
    detail_str, message = _describe_exception(exc)
    return (
        f"Warning: hybrid vector retrieval failed ({detail_str}): {message}. "
        "Using BM25 results only."
    )


def _describe_exception(exc: Exception) -> tuple[str, str]:
    exc_type = exc.__class__.__name__
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc)
    err_type = getattr(exc, "type", None)

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_obj = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(error_obj, dict):
            code = code or error_obj.get("code")
            err_type = err_type or error_obj.get("type")
            message = error_obj.get("message") or message

    details: list[str] = [exc_type]
    if status_code is not None:
        details.append(f"status={status_code}")
    if code:
        details.append(f"code={code}")
    if err_type:
        details.append(f"type={err_type}")
    return ", ".join(details), message


def _print_llm_warning(exc: Exception) -> None:
    print(_format_llm_warning(exc), file=sys.stderr)


def _build_cli_pipeline() -> RAGPipeline:
    return RAGPipeline(
        generator=DefaultGenerator(
            fallback_to_extractive=True,
            on_llm_error=_print_llm_warning,
        )
    )


def _retrieve_with_hybrid_fallback(
    *,
    pipeline: RAGPipeline,
    question: str,
    options: RetrievalOptions,
) -> list[SearchHit]:
    try:
        return pipeline.retrieve(question=question, options=options)
    except Exception as exc:
        if options.retrieval_mode == "hybrid" and options.bm25_index_path is not None:
            print(_format_hybrid_vector_warning(exc), file=sys.stderr)
            index = _load_bm25_index(options.bm25_index_path)
            return index.search(question, top_k=max(1, int(options.top_k)))
        raise


def _load_bm25_index(index_path: Path) -> BM25Index:
    if not index_path.exists():
        raise RuntimeError(f"Index not found: {index_path}")
    return BM25Index.load(index_path)


def main(argv: list[str] | None = None) -> int:
    if load_dotenv is not None:
        load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
