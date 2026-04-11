from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class IngestPathResponse(BaseModel):
    files_indexed: int
    chunks_created: int
    index_path: str
    status: Literal["ok"]


class IngestTextResponse(BaseModel):
    source: str
    chunks_created: int
    index_path: str
    status: Literal["ok"]


class CitationResponse(BaseModel):
    label: str
    source: str
    page: int | None
    chunk_id: str
    score: float
    preview: str


class TimingsMs(BaseModel):
    retrieve: int
    generate: int
    total: int


class AskResponse(BaseModel):
    answer: str
    supported: bool
    citations: list[CitationResponse]
    reason: Literal["insufficient_evidence"] | None = None
    retriever_used: Literal["bm25", "vector", "hybrid"]
    top_k: int
    timings_ms: TimingsMs
