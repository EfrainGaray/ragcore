"""Multi-vector retrieval with MaxSim scoring (simplified late interaction)."""
from __future__ import annotations

import re
import numpy as np
from loguru import logger


class MultiVectorRetriever:
    """Index chunks as multiple sentence-level embeddings, search with MaxSim."""

    def __init__(self, embed_model, store=None) -> None:
        self._embed_model = embed_model
        self._store = store  # Optional RagStore reference (not used internally, for future integration)
        # In-memory index: doc_id -> list of sentence embeddings
        self._index: dict[str, list[np.ndarray]] = {}
        # doc_id -> original chunk dict
        self._chunks: dict[str, dict] = {}

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def index(self, chunks: list[dict]) -> None:
        """Index chunks by embedding each sentence.

        chunks: list of dicts with "id" and "content" keys
        """
        for chunk in chunks:
            doc_id = chunk["id"]
            text = chunk.get("content", "")
            sentences = self._split_sentences(text)

            if not sentences:
                # Use whole text as single "sentence"
                sentences = [text] if text.strip() else []

            if not sentences:
                continue

            raw = self._embed_model.encode(sentences, show_progress_bar=False)
            embeddings = []
            for row in raw:
                arr = np.array(row.tolist() if hasattr(row, "tolist") else list(row))
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                embeddings.append(arr)

            self._index[doc_id] = embeddings
            self._chunks[doc_id] = chunk

        logger.debug("MultiVectorRetriever: indexed {} chunks", len(self._index))

    def _maxsim(self, query_embeddings: list[np.ndarray], doc_id: str) -> float:
        """MaxSim: for each query vector, find max cosine sim in doc, average over queries."""
        doc_embeddings = self._index.get(doc_id, [])
        if not doc_embeddings or not query_embeddings:
            return 0.0

        doc_matrix = np.stack(doc_embeddings)   # shape: (n_doc_sents, dim)

        scores = []
        for q_vec in query_embeddings:
            # cosine sims: q_vec already normalized, doc_matrix rows normalized
            sims = doc_matrix @ q_vec  # shape: (n_doc_sents,)
            scores.append(float(np.max(sims)))

        return float(np.mean(scores))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using MaxSim scoring.

        Returns list of dicts with id, content, score, and all other chunk fields.
        """
        if not self._index:
            return []

        # Embed query sentences
        q_sentences = self._split_sentences(query)
        if not q_sentences:
            q_sentences = [query]

        raw = self._embed_model.encode(q_sentences, show_progress_bar=False)
        q_embeddings = []
        for row in raw:
            arr = np.array(row.tolist() if hasattr(row, "tolist") else list(row))
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            q_embeddings.append(arr)

        # Score all docs
        scored = []
        for doc_id in self._index:
            score = self._maxsim(q_embeddings, doc_id)
            chunk = dict(self._chunks[doc_id])
            chunk["score"] = score
            scored.append(chunk)

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
