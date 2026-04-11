from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

import api.settings as api_settings
from api.main import app
from api.routers import ask as ask_router
from rag_mvp.answering import AnswerResult, Citation
from rag_mvp.bm25 import ChunkRecord, SearchHit


@dataclass
class FakePipeline:
    hits: list[SearchHit]
    answer_result: AnswerResult
    retrieve_question: str | None = None
    generate_question: str | None = None
    generate_calls: int = 0

    def retrieve(self, *, question: str, options: object) -> list[SearchHit]:
        self.retrieve_question = question
        return list(self.hits)

    def generate(self, *, question: str, hits: list[SearchHit], options: object) -> AnswerResult:
        self.generate_calls += 1
        self.generate_question = question
        return self.answer_result


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    api_settings.get_settings.cache_clear()
    yield
    api_settings.get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_valid_supported_question(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    hit = _make_hit(score=0.92)
    answer_result = AnswerResult(
        answer="Supported answer [1]",
        citations=[
            Citation(
                label="[1]",
                source=hit.chunk.source,
                chunk_id=hit.chunk.id,
                page=hit.chunk.page,
                score=hit.score,
                preview="Preview",
            )
        ],
    )
    fake = FakePipeline(hits=[hit], answer_result=answer_result)
    monkeypatch.setattr(ask_router, "pipeline", fake)

    response = client.post("/ask", json={"question": "What are the key highlights?"})
    body = response.json()

    assert response.status_code == 200
    assert body["supported"] is True
    assert body["reason"] is None
    assert body["citations"]
    assert body["answer"] == "Supported answer [1]"


def test_valid_unsupported_question_returns_structured_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASK_MIN_EVIDENCE_SCORE", "0.2")
    api_settings.get_settings.cache_clear()

    low_score_hit = _make_hit(score=0.05)
    fake = FakePipeline(hits=[low_score_hit], answer_result=AnswerResult(answer="Should not be used", citations=[]))
    monkeypatch.setattr(ask_router, "pipeline", fake)

    response = client.post("/ask", json={"question": "What policy changes are proposed for 2030?"})
    body = response.json()

    assert response.status_code == 200
    assert body["supported"] is False
    assert body["reason"] == "insufficient_evidence"
    assert body["citations"] == []
    assert body["answer"] == "I don't know based on the indexed public health evidence."
    assert fake.generate_calls == 0


def test_malformed_json_body_returns_validation_error(client: TestClient) -> None:
    # Literal newline inside a JSON string is invalid JSON and should fail transport validation.
    malformed_json = '{"question":"Line 1\nLine 2"}'
    response = client.post("/ask", content=malformed_json, headers={"Content-Type": "application/json"})
    body = response.json()

    assert response.status_code == 422
    assert body["message"].startswith("Malformed JSON request body")
    assert any(item.get("type") == "json_invalid" for item in body["detail"])


def test_multiline_question_input_is_preserved(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    question = "Line one\nLine two\tTabbed"
    hit = _make_hit(score=0.9)
    answer_result = AnswerResult(
        answer="Grounded answer [1]",
        citations=[
            Citation(
                label="[1]",
                source=hit.chunk.source,
                chunk_id=hit.chunk.id,
                page=hit.chunk.page,
                score=hit.score,
                preview="Preview",
            )
        ],
    )
    fake = FakePipeline(hits=[hit], answer_result=answer_result)
    monkeypatch.setattr(ask_router, "pipeline", fake)

    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert fake.retrieve_question == question


def test_hidden_control_chars_are_sanitized(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_question = "What\u0007 is\tin\nreport?\u0000"
    expected_sanitized = "What is\tin\nreport?"
    hit = _make_hit(score=0.88)
    answer_result = AnswerResult(
        answer="Grounded answer [1]",
        citations=[
            Citation(
                label="[1]",
                source=hit.chunk.source,
                chunk_id=hit.chunk.id,
                page=hit.chunk.page,
                score=hit.score,
                preview="Preview",
            )
        ],
    )
    fake = FakePipeline(hits=[hit], answer_result=answer_result)
    monkeypatch.setattr(ask_router, "pipeline", fake)

    response = client.post("/ask", json={"question": raw_question})

    assert response.status_code == 200
    assert fake.retrieve_question == expected_sanitized
    assert not any(
        (ord(ch) < 32 and ch not in {"\n", "\t"}) or ord(ch) == 127
        for ch in expected_sanitized
    )


def _make_hit(*, score: float) -> SearchHit:
    chunk = ChunkRecord(
        id="doc#p1#c0",
        source="sample.pdf",
        chunk_index=0,
        page=1,
        text="Sample evidence sentence for testing.",
        char_len=37,
        token_count=6,
    )
    return SearchHit(rank=1, score=score, chunk=chunk, retriever="bm25")
