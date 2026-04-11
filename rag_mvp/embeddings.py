from __future__ import annotations

import os


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    if not texts:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Vector/hybrid retrieval requires OpenAI embeddings."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []
    batch_size = 128

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        rows = sorted(response.data, key=lambda item: item.index)
        embeddings.extend([list(row.embedding) for row in rows])

    return embeddings

