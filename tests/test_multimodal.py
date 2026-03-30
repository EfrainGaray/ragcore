"""Tests for ragcore.multimodal — extract_images and multi-modal ingestion."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub pytesseract before any ragcore.multimodal import
# ---------------------------------------------------------------------------

if "pytesseract" not in sys.modules:
    _tess_mod = types.ModuleType("pytesseract")
    _tess_mod.image_to_string = MagicMock(return_value="extracted OCR text")
    _tess_mod.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
    sys.modules["pytesseract"] = _tess_mod

# Stub PIL / Pillow
if "PIL" not in sys.modules:
    _pil_mod = types.ModuleType("PIL")
    _pil_image_mod = types.ModuleType("PIL.Image")

    class _FakeImage:
        pass

    _pil_image_mod.Image = _FakeImage
    _pil_image_mod.open = MagicMock(return_value=_FakeImage())
    _pil_mod.Image = _pil_image_mod
    sys.modules["PIL"] = _pil_mod
    sys.modules["PIL.Image"] = _pil_image_mod

# Ensure fitz (PyMuPDF) is stubbed if not installed
if "fitz" not in sys.modules:
    _fitz_mod = types.ModuleType("fitz")

    class _FakeDoc:
        def __init__(self):
            self._pages = []

        def __iter__(self):
            return iter(self._pages)

        def __len__(self):
            return len(self._pages)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    _fitz_mod.open = MagicMock(return_value=_FakeDoc())
    sys.modules["fitz"] = _fitz_mod


# ---------------------------------------------------------------------------
# Minimal PNG and JPEG byte sequences (valid headers)
# ---------------------------------------------------------------------------

_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # fake minimal PNG bytes
_JPG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 50   # fake minimal JPEG bytes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_images_returns_tuples():
    """extract_images must return a list of (str, int) tuples."""
    with patch("pytesseract.image_to_string", return_value="some text"):
        from ragcore.multimodal import extract_images

        result = extract_images("photo.png", _PNG_HEADER)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        text, page = item
        assert isinstance(text, str)
        assert isinstance(page, int)


def test_extract_images_png_fallback():
    """When OCR raises (Tesseract not found), must return at least [('image', 0)]."""
    import pytesseract

    tess_err = sys.modules["pytesseract"].TesseractNotFoundError

    with patch("pytesseract.image_to_string", side_effect=tess_err("not found")):
        from ragcore.multimodal import extract_images

        result = extract_images("photo.png", _PNG_HEADER)

    assert len(result) >= 1
    assert result[0][1] == 0  # page number is 0
    assert "image" in result[0][0]  # text contains "image" (may include filename)


def test_extract_images_jpg_supported():
    """extract_images must accept .jpg extension without raising."""
    with patch("pytesseract.image_to_string", return_value="jpg text"):
        from ragcore.multimodal import extract_images

        result = extract_images("photo.jpg", _JPG_HEADER)

    assert isinstance(result, list)
    assert len(result) >= 1


def test_ingest_supports_png_extension(ingestor):
    """Ingestor.ingest('test.png', data) must not raise UnsupportedExtension."""
    with patch("pytesseract.image_to_string", return_value="image text content"):
        try:
            # The ingestor may delegate to extract_images internally
            ingestor.ingest("test.png", _PNG_HEADER)
        except Exception as exc:
            exc_type = type(exc).__name__
            # Only an UnsupportedExtension (or similar) is a failure here;
            # any other behaviour (including a no-op) is acceptable.
            assert "UnsupportedExtension" not in exc_type and "Unsupported" not in str(exc), (
                f"ingestor raised UnsupportedExtension for .png: {exc}"
            )


def test_multimodal_page_number_is_int():
    """Page numbers in extract_images output must be integers, not floats or strings."""
    with patch("pytesseract.image_to_string", return_value="page content"):
        from ragcore.multimodal import extract_images

        result = extract_images("scan.png", _PNG_HEADER)

    for _text, page in result:
        assert isinstance(page, int), f"Expected int page number, got {type(page)}: {page}"
