from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.models.requests import DEFAULT_INDEX_PATH, IngestPathRequest, IngestTextRequest
from api.models.responses import IngestPathResponse, IngestTextResponse
from rag_mvp.ingestion import ingest_source_dir, ingest_text


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_DIR = ".rag/chroma"


router = APIRouter(tags=["Ingestion"])


@router.post("/ingest_path", response_model=IngestPathResponse)
def ingest_path(request: IngestPathRequest) -> IngestPathResponse:
    source_dir = Path(request.source_dir)
    index_path = Path(request.index_path)

    try:
        result = ingest_source_dir(
            source_dir=source_dir,
            index_path=index_path,
            retriever="hybrid",
            embed_model=DEFAULT_EMBED_MODEL,
            chroma_dir=Path(DEFAULT_CHROMA_DIR),
            # Keep ingestion demo-safe: build BM25 index even if vector indexing is unavailable.
            allow_vector_failure=True,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestPathResponse(
        files_indexed=result.files_indexed,
        chunks_created=result.chunks_created,
        index_path=request.index_path,
        status="ok",
    )


@router.post("/ingestText", response_model=IngestTextResponse, include_in_schema=False)
@router.post("/ingest_text", response_model=IngestTextResponse, include_in_schema=False)
def ingest_text_endpoint(request: IngestTextRequest) -> IngestTextResponse:
    try:
        result = ingest_text(
            text=request.text,
            source=request.source,
            index_path=Path(DEFAULT_INDEX_PATH),
            embed_model=DEFAULT_EMBED_MODEL,
            chroma_dir=Path(DEFAULT_CHROMA_DIR),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestTextResponse(
        source=result.source,
        chunks_created=result.chunks_created,
        index_path=DEFAULT_INDEX_PATH,
        status="ok",
    )
