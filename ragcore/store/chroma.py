"""ChromaDB wrapper for ragcore."""

from __future__ import annotations

from datetime import datetime, timezone
from collections import defaultdict

import chromadb
from loguru import logger

from ragcore.config import Settings
from ragcore.models import DocumentInfo


class RagStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "RagStore initialised — collection={} path={}",
            settings.chroma_collection,
            settings.chroma_path,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add pre-embedded chunks to the collection.

        Each chunk dict must contain:
            id, content, embedding, filename, page, chunk_index
        """
        if not chunks:
            return

        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [
            {
                "filename": c["filename"],
                "page": str(c["page"]),
                "chunk_index": int(c["chunk_index"]),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Added {} chunks to collection", len(chunks))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """Vector search.  Returns up to *top_k* results."""
        where = filters if filters else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]

        for doc_id, content, meta, distance in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            score = 1.0 - distance
            hits.append(
                {
                    "id": doc_id,
                    "content": content,
                    "score": score,
                    "filename": meta.get("filename", ""),
                    "page": meta.get("page", "0"),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "metadata": {
                        k: v
                        for k, v in meta.items()
                        if k not in {"filename", "page", "chunk_index"}
                    },
                }
            )

        return hits

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks belonging to *filename*.  Returns count deleted."""
        result = self._collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted {} chunks for filename={}", len(ids), filename)
        return len(ids)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def list_documents(self) -> list[DocumentInfo]:
        """Aggregate stored chunks by filename and return DocumentInfo list."""
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = defaultdict(int)
        added_ats: dict[str, str] = {}

        for meta in result["metadatas"]:
            fname = meta.get("filename", "unknown")
            counts[fname] += 1
            added_ats.setdefault(fname, meta.get("added_at", ""))

        return [
            DocumentInfo(filename=fname, chunks=counts[fname], added_at=added_ats[fname])
            for fname in sorted(counts)
        ]

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()
