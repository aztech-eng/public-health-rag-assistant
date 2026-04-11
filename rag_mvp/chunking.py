from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


def discover_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    import logging
    import concurrent.futures
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is not installed. Run: pip install -r requirements.txt") from exc

    def safe_extract_text(page, timeout=10):
        # Extract text with a timeout (seconds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: (page.extract_text() or "").replace("\r\n", "\n").strip())
            try:
                return future.result(timeout=timeout)
            except Exception as e:
                return e

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            result = safe_extract_text(page, timeout=10)
            if isinstance(result, Exception):
                logging.warning(f"Failed to extract text from {path} page {page_num}: {result}")
                continue
            text = result
        except Exception as e:
            logging.warning(f"Exception extracting text from {path} page {page_num}: {e}")
            continue
        if not text:
            continue
        pages.append((page_num, text))
    return pages


def chunk_text(text: str, chunk_chars: int = 900, overlap_chars: int = 120) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_large_block(para, chunk_chars, overlap_chars))
            continue

        if not current:
            current = para
            continue

        candidate = f"{current}\n\n{para}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue

        chunks.append(current.strip())
        carry = _tail_overlap(current, overlap_chars)
        current = f"{carry}\n\n{para}".strip() if carry else para

        if len(current) > chunk_chars:
            chunks.extend(_split_large_block(current, chunk_chars, overlap_chars))
            current = ""

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_large_block(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return _sliding_char_chunks(text, chunk_chars, overlap_chars)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        chunks.append(current.strip())
        carry = _tail_overlap(current, overlap_chars)
        current = f"{carry} {sentence}".strip() if carry else sentence
        if len(current) > chunk_chars:
            chunks.extend(_sliding_char_chunks(current, chunk_chars, overlap_chars))
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def _sliding_char_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - max(0, overlap_chars))
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    tail = text[-overlap_chars:].strip()
    return re.sub(r"\s+", " ", tail)


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
