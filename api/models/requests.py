from __future__ import annotations

import re
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


DEFAULT_INDEX_PATH = ".rag/index.json"
_QUESTION_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "question": "What disease notifications are highlighted in this report?",
                    "index_path": ".rag/index.json",
                    "retriever": "hybrid",
                    "top_k": 5,
                    "rerank_mode": "lexical",
                    "rerank_candidate_k": 20,
                    "use_llm": False,
                },
                {
                    "question": (
                        "Summarize key points from this pasted text:\n"
                        "- mention notable outbreaks\n"
                        "- mention vaccination updates\tif present"
                    ),
                    "index_path": ".rag/index.json",
                    "retriever": "hybrid",
                    "top_k": 5,
                    "rerank_mode": "lexical",
                    "rerank_candidate_k": 20,
                    "use_llm": False,
                },
            ]
        },
    )

    question: str = Field(
        min_length=1,
        strict=True,
        description=(
            "Free-text user question. Newlines and tabs are allowed. "
            "Use normal JSON serialization (for example `json=payload`) rather than manual string interpolation."
        ),
    )
    index_path: str = DEFAULT_INDEX_PATH
    retriever: Literal["bm25", "vector", "hybrid"] = Field(
        default="hybrid",
        validation_alias=AliasChoices("retriever", "retrieval_mode"),
    )
    top_k: int = Field(default=5, ge=1)
    rerank_mode: Literal["none", "lexical", "semantic", "auto"] = "lexical"
    rerank_candidate_k: int = Field(default=20, ge=1)
    use_llm: bool = False
    model: str = "gpt-4.1-mini"
    max_context_chars: int = Field(default=6000, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, value: str) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        sanitized = _QUESTION_CONTROL_CHARS.sub("", normalized).strip()
        if not sanitized:
            raise ValueError("question must contain visible text")
        return sanitized


class IngestPathRequest(BaseModel):
    source_dir: str = Field(min_length=1)
    index_path: str = DEFAULT_INDEX_PATH


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
