"""Tests for ragcore.store.ingest — file reading, chunking, ingestion."""

from __future__ import annotations

import io
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from ragcore.store.ingest import chunk_text, extract_text


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_respects_size():
    """No chunk produced by chunk_text should exceed chunk_size characters."""
    long_text = " ".join(["word"] * 500)
    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 100, f"Chunk too long: {len(chunk)} chars"


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_short_text_single_chunk():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


def test_ingest_txt_file(ingestor, store):
    """Ingesting a TXT file must produce at least one chunk in the store."""
    content = b"This is a plain text document about machine learning, AI, and retrieval systems."
    count = ingestor.ingest("sample.txt", content)

    assert count >= 1
    assert store.count() >= 1


def test_ingest_unsupported_format():
    """extract_text must raise ValueError for unknown extensions."""
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("file.xyz", b"some data")


def test_extract_text_txt():
    pages = extract_text("hello.txt", b"Hello, world!")
    assert len(pages) == 1
    text, page = pages[0]
    assert "Hello" in text
    assert page == 0


def test_ingest_pdf_file(ingestor, store):
    """Ingesting a mocked PDF must produce chunks."""
    # Stub pypdf.PdfReader to return one page of text
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "PDF content about retrieval augmented generation."

    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = MagicMock(return_value=fake_reader)

    with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        # Force reimport of extract_text internals by calling _read_pdf directly
        from ragcore.store.ingest import _read_pdf

        pages = _read_pdf(b"fake-pdf-bytes")

    assert len(pages) == 1
    assert "retrieval" in pages[0][0]
    assert pages[0][1] == 1  # page number 1-indexed


def test_chunk_overlap_produces_multiple_chunks():
    """A text longer than chunk_size must produce more than one chunk."""
    text = "A" * 300
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
