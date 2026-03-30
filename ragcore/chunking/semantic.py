from __future__ import annotations
import re
import numpy as np


class SemanticChunker:
    """Splits text by sentence boundaries, merges until cosine similarity drops."""

    def __init__(self, embed_model, threshold: float = 0.5, max_chunk_size: int = 512) -> None:
        self._model = embed_model
        self._threshold = threshold
        self._max = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s]

        if not sentences:
            return []

        raw = self._model.encode(sentences, show_progress_bar=False)

        # Normalise each row to unit length for cosine similarity via dot product
        vecs: list[list[float]] = []
        for row in raw:
            arr = list(row.tolist() if hasattr(row, 'tolist') else row)
            norm = sum(x * x for x in arr) ** 0.5
            if norm > 0:
                arr = [x / norm for x in arr]
            vecs.append(arr)

        def _mean_vec(indices: list[int]) -> list[float]:
            n = len(indices)
            length = len(vecs[0])
            mv = [0.0] * length
            for i in indices:
                for j, val in enumerate(vecs[i]):
                    mv[j] += val
            return [x / n for x in mv]

        def _cosine(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        chunks: list[str] = []
        buffer_indices: list[int] = [0]
        buffer_text = sentences[0]

        for idx in range(1, len(sentences)):
            mean = _mean_vec(buffer_indices)
            sim = _cosine(mean, vecs[idx])
            candidate_text = buffer_text + " " + sentences[idx]

            if sim >= self._threshold and len(candidate_text) <= self._max:
                buffer_indices.append(idx)
                buffer_text = candidate_text
            else:
                flushed = buffer_text.strip()
                if flushed:
                    chunks.append(flushed)
                buffer_indices = [idx]
                buffer_text = sentences[idx]

        flushed = buffer_text.strip()
        if flushed:
            chunks.append(flushed)

        return chunks
