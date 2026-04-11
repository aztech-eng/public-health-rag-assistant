from __future__ import annotations

from rag_mvp.answering import _build_citations
from rag_mvp.bm25 import ChunkRecord, SearchHit


def test_preview_fixes_lowercase_sentence_start_for_england() -> None:
    preview = _citation_preview("england) Evidence reviews support tighter controls.")
    assert preview == "England) Evidence reviews support tighter controls."


def test_preview_normalizes_broken_newlines_and_sentence_case() -> None:
    preview = _citation_preview("first line\nsecond line.\nthird line follows.")
    assert preview == "First line second line. Third line follows."


def test_preview_collapses_repeated_spaces() -> None:
    preview = _citation_preview("This    has   too      many spaces.")
    assert preview == "This has too many spaces."


def test_preview_fixes_common_proper_nouns() -> None:
    preview = _citation_preview("guidance in uk and nhs updates from ukhsa in england.")
    assert preview == "Guidance in UK and NHS updates from UKHSA in England."


def test_preview_shifts_when_starting_mid_sentence() -> None:
    text = "of B. stabilis on the patient's condition was difficult. Distribution of cases in england was limited."
    preview = _citation_preview(text)
    assert preview == "Distribution of cases in England was limited."


def test_preview_keeps_cleaned_substring_when_no_sentence_boundary_exists() -> None:
    preview = _citation_preview("n scabies diagnoses at SHSs from 2014 to 2024")
    assert preview == "N scabies diagnoses at SHSs from 2014 to 2024"


def test_preview_length_limit_matches_existing_behavior() -> None:
    long_text = "a" * 180
    preview = _citation_preview(long_text)
    assert len(preview) == 140
    assert preview.endswith("...")


def _citation_preview(text: str) -> str:
    hit = _make_hit(text=text)
    return _build_citations([hit])[0].preview


def _make_hit(*, text: str) -> SearchHit:
    chunk = ChunkRecord(
        id="doc#p1#c0",
        source="sample.pdf",
        chunk_index=0,
        page=1,
        text=text,
        char_len=len(text),
        token_count=max(1, len(text.split())),
    )
    return SearchHit(rank=1, score=0.9, chunk=chunk, retriever="bm25")
