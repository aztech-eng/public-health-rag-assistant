from __future__ import annotations

from pathlib import Path

from .bm25 import ChunkRecord, SearchHit, tokenize
from .embeddings import embed_texts


COLLECTION_NAME = "rag_mvp_chunks"


def upsert_chunks(chunks: list[ChunkRecord], persist_dir: Path, embed_model: str) -> None:
    if not chunks:
        return

    collection = _get_collection(persist_dir)
    batch_size = 128

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(texts, model=embed_model)
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=texts,
            metadatas=[_chunk_metadata(chunk) for chunk in batch],
            embeddings=embeddings,
        )


def query(question: str, top_k: int, persist_dir: Path, embed_model: str) -> list[SearchHit]:
    top_k = max(1, int(top_k))
    collection = _get_collection(persist_dir)

    query_embedding = embed_texts([question], model=embed_model)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids_rows = results.get("ids") or []
    if not ids_rows:
        return []

    docs_rows = results.get("documents") or []
    metas_rows = results.get("metadatas") or []
    dist_rows = results.get("distances") or []

    ids = ids_rows[0] if ids_rows else []
    docs = docs_rows[0] if docs_rows else []
    metas = metas_rows[0] if metas_rows else []
    dists = dist_rows[0] if dist_rows else []

    hits: list[SearchHit] = []
    for i, chunk_id in enumerate(ids, start=1):
        doc_text = ""
        if i - 1 < len(docs) and isinstance(docs[i - 1], str):
            doc_text = docs[i - 1]

        metadata = {}
        if i - 1 < len(metas) and isinstance(metas[i - 1], dict):
            metadata = metas[i - 1]

        distance = 0.0
        if i - 1 < len(dists):
            try:
                distance = float(dists[i - 1])
            except (TypeError, ValueError):
                distance = 0.0

        chunk = _chunk_from_result(chunk_id=str(chunk_id), text=doc_text, metadata=metadata)
        score = 1.0 / (1.0 + max(0.0, distance))
        hits.append(SearchHit(rank=i, score=score, chunk=chunk, retriever="vector"))

    return hits


def _get_collection(persist_dir: Path):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt") from exc

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_metadata(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "source": chunk.source,
        "page": chunk.page if chunk.page is not None else -1,
        "chunk_index": chunk.chunk_index,
    }


def _chunk_from_result(chunk_id: str, text: str, metadata: dict[str, object]) -> ChunkRecord:
    source_raw = metadata.get("source", "")
    chunk_index_raw = metadata.get("chunk_index", 0)
    page_raw = metadata.get("page", -1)

    source = str(source_raw) if source_raw is not None else ""

    try:
        chunk_index = int(chunk_index_raw)
    except (TypeError, ValueError):
        chunk_index = 0

    try:
        page_val = int(page_raw)
    except (TypeError, ValueError):
        page_val = -1
    page = page_val if page_val >= 0 else None

    token_count = len(tokenize(text))
    return ChunkRecord(
        id=chunk_id,
        source=source,
        chunk_index=chunk_index,
        page=page,
        text=text,
        char_len=len(text),
        token_count=token_count,
    )

