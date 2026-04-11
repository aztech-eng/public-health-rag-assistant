from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "if", "but", "not", "can", "could", "should",
    "would", "into", "about", "what", "which", "when", "where", "who", "why", "how",
    "you", "your", "we", "our", "they", "their", "i", "me", "my", "mine", "do", "does",
    "did", "done", "than", "then", "there", "here", "also", "such", "any", "all", "each",
    "per", "via", "over", "under", "up", "down", "out", "more", "most", "less", "least",
}


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in (m.group(0).lower() for m in TOKEN_RE.finditer(text))
        if len(tok) > 1 and tok not in STOPWORDS
    ]


@dataclass
class ChunkRecord:
    id: str
    source: str
    chunk_index: int
    page: int | None
    text: str
    char_len: int
    token_count: int

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkRecord":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            chunk_index=int(data["chunk_index"]),
            page=(int(data["page"]) if data.get("page") is not None else None),
            text=str(data["text"]),
            char_len=int(data["char_len"]),
            token_count=int(data["token_count"]),
        )


@dataclass
class SearchHit:
    rank: int
    score: float
    chunk: ChunkRecord
    retriever: str | None = None


@dataclass
class BM25Index:
    version: int
    created_at: str
    k1: float
    b: float
    avg_doc_len: float
    num_docs: int
    chunks: list[ChunkRecord]
    doc_lengths: list[int]
    postings: dict[str, list[list[int]]]

    @classmethod
    def build(cls, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        postings_map: dict[str, dict[int, int]] = defaultdict(dict)
        doc_lengths: list[int] = []

        for doc_idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            term_counts = Counter(tokens)
            doc_len = sum(term_counts.values())
            if doc_len <= 0:
                doc_len = 1
            doc_lengths.append(doc_len)

            for term, tf in term_counts.items():
                postings_map[term][doc_idx] = tf

        avg_doc_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 1.0
        postings = {
            term: [[doc_idx, tf] for doc_idx, tf in sorted(doc_map.items())]
            for term, doc_map in postings_map.items()
        }

        return cls(
            version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            k1=k1,
            b=b,
            avg_doc_len=avg_doc_len,
            num_docs=len(chunks),
            chunks=chunks,
            doc_lengths=doc_lengths,
            postings=postings,
        )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
        if self.num_docs == 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        query_counts = Counter(query_terms)

        for term, qtf in query_counts.items():
            postings = self.postings.get(term)
            if not postings:
                continue

            df = len(postings)
            # BM25 idf (positive, stable for small corpora)
            idf = math.log(1.0 + ((self.num_docs - df + 0.5) / (df + 0.5)))

            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-9, self.avg_doc_len)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(1e-9, denom))
                scores[doc_idx] += term_score * (1.0 + 0.15 * (qtf - 1))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits: list[SearchHit] = []
        for rank, (doc_idx, score) in enumerate(ranked[: max(1, top_k)], start=1):
            if score < min_score:
                continue
            hits.append(SearchHit(rank=rank, score=score, chunk=self.chunks[doc_idx], retriever="bm25"))
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "created_at": self.created_at,
            "k1": self.k1,
            "b": self.b,
            "avg_doc_len": self.avg_doc_len,
            "num_docs": self.num_docs,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "doc_lengths": self.doc_lengths,
            "postings": self.postings,
        }
        path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=int(data["version"]),
            created_at=str(data["created_at"]),
            k1=float(data["k1"]),
            b=float(data["b"]),
            avg_doc_len=float(data["avg_doc_len"]),
            num_docs=int(data["num_docs"]),
            chunks=[ChunkRecord.from_dict(c) for c in data["chunks"]],
            doc_lengths=[int(x) for x in data["doc_lengths"]],
            postings={str(k): [[int(a), int(b)] for a, b in v] for k, v in data["postings"].items()},
        )
