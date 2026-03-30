"""Multi-modal ingestion — extracts text from image files."""
from __future__ import annotations

from loguru import logger


def _ocr_image(data: bytes) -> str:
    """Run OCR on image bytes. Returns extracted text or empty string."""
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img).strip()
        return text
    except ImportError:
        logger.debug("pytesseract/Pillow not installed — OCR unavailable")
        return ""
    except Exception as exc:
        logger.warning("OCR failed: {}", exc)
        return ""


def extract_images(filename: str, data: bytes) -> list[tuple[str, int]]:
    """Extract text from image file (PNG, JPG, JPEG).

    Returns list of (text, page_number) tuples.
    Falls back to ("image", 0) if OCR is unavailable or fails.
    """
    text = _ocr_image(data)
    if not text:
        # Provide minimal placeholder so the file still gets indexed
        import os
        name = os.path.splitext(os.path.basename(filename))[0]
        text = f"image: {name}"
    return [(text, 0)]
