from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Index, ChunkRecord, tokenize
from .chunking import chunk_text, discover_text_files, extract_pdf_pages, read_text_file


@dataclass
class ChunkBuildResult:
    files_indexed: int
    chunks: list[ChunkRecord]
    total_tokens: int


@dataclass
class IngestPathResult:
    files_indexed: int
    chunks_created: int
    index_path: Path
    bm25_terms: int
    avg_tokens_per_chunk: float
    vector_indexed: bool
    vector_error: Exception | None = None


@dataclass
class IngestTextResult:
    source: str
    chunks_created: int
    index_path: Path
    total_index_chunks: int


def ingest_source_dir(
    *,
    source_dir: Path,
    index_path: Path,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    retriever: str = "hybrid",
    embed_model: str = "text-embedding-3-small",
    chroma_dir: Path = Path(".rag/chroma"),
    allow_vector_failure: bool = True,
) -> IngestPathResult:
    build = build_chunks_from_dir(
        source_dir=source_dir,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )

    index = BM25Index.build(build.chunks)
    index.save(index_path)

    vector_indexed = False
    vector_error: Exception | None = None
    if str(retriever) != "bm25":
        try:
            upsert_vector_chunks(build.chunks, chroma_dir=chroma_dir, embed_model=embed_model)
            vector_indexed = True
        except Exception as exc:
            if not allow_vector_failure:
                raise
            vector_error = exc

    return IngestPathResult(
        files_indexed=build.files_indexed,
        chunks_created=len(build.chunks),
        index_path=index_path,
        bm25_terms=len(index.postings),
        avg_tokens_per_chunk=(build.total_tokens / max(1, len(build.chunks))),
        vector_indexed=vector_indexed,
        vector_error=vector_error,
    )


def ingest_text(
    *,
    text: str,
    source: str,
    index_path: Path = Path(".rag/index.json"),
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    embed_model: str = "text-embedding-3-small",
    chroma_dir: Path = Path(".rag/chroma"),
) -> IngestTextResult:
    chunks = build_chunks_from_text(
        text=text,
        source=source,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )

    existing_chunks: list[ChunkRecord] = []
    if index_path.exists():
        existing_chunks = BM25Index.load(index_path).chunks

    merged_chunks = _merge_chunks_by_id(existing_chunks, chunks)
    bm25_index = BM25Index.build(merged_chunks)
    bm25_index.save(index_path)

    upsert_vector_chunks(chunks, chroma_dir=chroma_dir, embed_model=embed_model)

    return IngestTextResult(
        source=source,
        chunks_created=len(chunks),
        index_path=index_path,
        total_index_chunks=len(merged_chunks),
    )


def build_chunks_from_dir(
    *,
    source_dir: Path,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> ChunkBuildResult:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    files = discover_text_files(source_dir)
    if not files:
        raise ValueError(f"No supported files found under {source_dir} (.md, .markdown, .txt, .pdf)")

    chunks: list[ChunkRecord] = []
    total_tokens = 0

    for file_path in files:
        rel_path = file_path.relative_to(source_dir).as_posix()
        try:
            file_chunks = _build_chunks_for_file(
                file_path=file_path,
                source_name=rel_path,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to ingest {rel_path}: {exc}") from exc
        for chunk in file_chunks:
            total_tokens += chunk.token_count
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(f"No non-empty chunks could be created from files under {source_dir}")

    return ChunkBuildResult(files_indexed=len(files), chunks=chunks, total_tokens=total_tokens)


def build_chunks_from_text(
    *,
    text: str,
    source: str,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> list[ChunkRecord]:
    source_name = str(source).strip().replace("\\", "/")
    if not source_name:
        raise ValueError("source must not be empty")

    normalized_text = str(text)
    if not normalized_text.strip():
        raise ValueError("text must not be empty")

    chunk_texts = chunk_text(
        normalized_text,
        chunk_chars=max(1, int(chunk_chars)),
        overlap_chars=max(0, int(overlap_chars)),
    )
    if not chunk_texts:
        raise ValueError("No chunks could be created from the provided text")

    chunks: list[ChunkRecord] = []
    for chunk_index, chunk_value in enumerate(chunk_texts):
        token_count = len(tokenize(chunk_value))
        chunks.append(
            ChunkRecord(
                id=f"{source_name}#c{chunk_index}",
                source=source_name,
                chunk_index=chunk_index,
                page=None,
                text=chunk_value,
                char_len=len(chunk_value),
                token_count=token_count,
            )
        )
    return chunks


def upsert_vector_chunks(chunks: list[ChunkRecord], *, chroma_dir: Path, embed_model: str) -> None:
    if not chunks:
        return
    from .vector_store import upsert_chunks

    upsert_chunks(chunks=chunks, persist_dir=chroma_dir, embed_model=embed_model)


def _build_chunks_for_file(
    *,
    file_path: Path,
    source_name: str,
    chunk_chars: int,
    overlap_chars: int,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    if file_path.suffix.lower() == ".pdf":
        page_records = extract_pdf_pages(file_path)
        for page_num, page_text in page_records:
            page_chunks = chunk_text(
                page_text,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
            for chunk_index, text in enumerate(page_chunks):
                token_count = len(tokenize(text))
                records.append(
                    ChunkRecord(
                        id=f"{source_name}#p{page_num}#c{chunk_index}",
                        source=source_name,
                        chunk_index=chunk_index,
                        page=page_num,
                        text=text,
                        char_len=len(text),
                        token_count=token_count,
                    )
                )
        return records

    raw_text = read_text_file(file_path)
    file_chunks = chunk_text(raw_text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    for chunk_index, text in enumerate(file_chunks):
        token_count = len(tokenize(text))
        records.append(
            ChunkRecord(
                id=f"{source_name}#c{chunk_index}",
                source=source_name,
                chunk_index=chunk_index,
                page=None,
                text=text,
                char_len=len(text),
                token_count=token_count,
            )
        )
    return records


def _merge_chunks_by_id(existing: list[ChunkRecord], new_chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    merged: dict[str, ChunkRecord] = {chunk.id: chunk for chunk in existing}
    for chunk in new_chunks:
        merged[chunk.id] = chunk
    return list(merged.values())
