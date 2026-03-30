"""Document reading, chunking, embedding, and storing."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import BinaryIO

from loguru import logger

from ragcore.config import Settings
from ragcore.store.chroma import RagStore


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _read_txt(data: bytes) -> list[tuple[str, int]]:
    """Return list of (text, page_number).  TXT = single page."""
    return [(data.decode("utf-8", errors="replace"), 0)]


def _read_pdf(data: bytes) -> list[tuple[str, int]]:
    """Extract text page-by-page from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((text, i + 1))
    return pages


def _read_docx(data: bytes) -> list[tuple[str, int]]:
    """Extract paragraphs from a DOCX file (single logical page)."""
    from docx import Document

    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(text, 0)]


def _read_xlsx(data: bytes) -> list[tuple[str, int]]:
    """Convert each sheet to CSV-like text."""
    import pandas as pd

    pages = []
    xls = pd.ExcelFile(io.BytesIO(data))
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(xls, sheet_name=sheet)
        pages.append((df.to_csv(index=False), i))
    return pages


def _read_text(data: bytes) -> list[tuple[str, int]]:
    """Plain-text read for prose / markup / config files."""
    return [(data.decode("utf-8", errors="replace"), 0)]


def _read_code(data: bytes) -> list[tuple[str, int]]:
    """Plain-text read for source-code files."""
    return [(data.decode("utf-8", errors="replace"), 0)]


# Extensions treated as source code (smaller chunks, file_type="code")
CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".js", ".go", ".rs", ".java",
})

# Default chunk size for code files (preserves function/class boundaries better)
CODE_CHUNK_SIZE = 256

_READERS = {
    ".txt": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
    ".md": _read_text,
    ".json": _read_text,
    ".yaml": _read_text,
    ".yml": _read_text,
    ".toml": _read_text,
    ".csv": _read_text,
    # Source-code files
    ".py": _read_code,
    ".ts": _read_code,
    ".js": _read_code,
    ".go": _read_code,
    ".rs": _read_code,
    ".java": _read_code,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_READERS.keys())


def extract_text(filename: str, data: bytes) -> list[tuple[str, int]]:
    """Dispatch to the appropriate reader based on file extension.

    Returns a list of (text, page_number) tuples.
    Raises ValueError for unsupported extensions.
    """
    ext = Path(filename).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file format: {ext!r}")
    return reader(data)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[str]:
    """Simple character-level sliding-window chunker.

    Splits on word boundaries where possible to avoid cutting mid-word.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to break at a whitespace boundary, unless we're at end-of-text
        if end < length:
            # Search backwards for a whitespace character
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # If we consumed the whole text in one chunk, stop.
        if end >= length:
            break

        # Advance with overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1  # guarantee forward progress
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class Ingestor:
    """Coordinates reading → chunking → embedding → storing."""

    def __init__(
        self,
        store: RagStore,
        embedding_model,
        settings: Settings,
    ) -> None:
        self._store = store
        self._model = embedding_model
        self._settings = settings

    def ingest(self, filename: str, data: bytes) -> int:
        """Ingest a document.  Returns number of child chunks stored."""
        pages = extract_text(filename, data)
        logger.info("Ingesting {} — {} page(s)", filename, len(pages))

        ext = Path(filename).suffix.lower()
        is_code = ext in CODE_EXTENSIONS
        file_type = "code" if is_code else "document"
        effective_chunk_size = CODE_CHUNK_SIZE if is_code else self._settings.chunk_size

        use_parent_child = (
            getattr(self._settings, "parent_child_chunks", False) and not is_code
        )
        parent_chunk_size = getattr(self._settings, "parent_chunk_size", 1536)

        all_chunks: list[dict] = []
        parent_chunks: list[dict] = []
        chunk_index = 0

        for text, page in pages:
            if use_parent_child:
                # 1. Create parent chunks first
                p_chunks = chunk_text(
                    text,
                    chunk_size=parent_chunk_size,
                    chunk_overlap=self._settings.chunk_overlap,
                )
                for p_text in p_chunks:
                    parent_id = str(uuid.uuid4())
                    ns_parent_id = f"{self._store._namespace}:{parent_id}"
                    # Store parent chunk (not embedded, not searched — metadata flag)
                    parent_chunks.append(
                        {
                            "id": parent_id,
                            "content": p_text,
                            "filename": filename,
                            "page": page,
                            "chunk_index": chunk_index,
                            "file_type": file_type,
                            "chunk_type": "parent",
                            "is_parent": True,
                        }
                    )

                    # 2. Create child chunks within this parent
                    c_chunks = chunk_text(
                        p_text,
                        chunk_size=effective_chunk_size,
                        chunk_overlap=self._settings.chunk_overlap,
                    )
                    for c_text in c_chunks:
                        all_chunks.append(
                            {
                                "id": str(uuid.uuid4()),
                                "content": c_text,
                                "filename": filename,
                                "page": page,
                                "chunk_index": chunk_index,
                                "file_type": file_type,
                                "chunk_type": "child",
                                "parent_id": ns_parent_id,
                            }
                        )
                        chunk_index += 1
            else:
                text_chunks = chunk_text(
                    text,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=self._settings.chunk_overlap,
                )
                for chunk in text_chunks:
                    all_chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "content": chunk,
                            "filename": filename,
                            "page": page,
                            "chunk_index": chunk_index,
                            "file_type": file_type,
                        }
                    )
                    chunk_index += 1

        if not all_chunks:
            logger.warning("No chunks produced for {}", filename)
            return 0

        # Embed parent chunks so they can be retrieved by content.
        if parent_chunks:
            p_texts = [c["content"] for c in parent_chunks]
            p_raw = self._model.encode(p_texts, show_progress_bar=False)
            if hasattr(p_raw, "tolist"):
                p_embeddings = p_raw.tolist()
            else:
                p_embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in p_raw]
            for chunk, emb in zip(parent_chunks, p_embeddings):
                chunk["embedding"] = emb
            self._store.add_chunks(parent_chunks)
            logger.debug("Stored {} parent chunks for {}", len(parent_chunks), filename)

        # Embed all child chunks in one batch
        texts = [c["content"] for c in all_chunks]
        raw = self._model.encode(texts, show_progress_bar=False)
        # Support numpy arrays (LocalEmbedder) and plain lists (OpenAICompatibleEmbedder)
        if hasattr(raw, "tolist"):
            embeddings = raw.tolist()
        else:
            embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw]

        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        self._store.add_chunks(all_chunks)
        logger.info("Stored {} chunks for {}", len(all_chunks), filename)
        return len(all_chunks)
