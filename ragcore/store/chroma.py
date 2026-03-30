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

    def __init__(self, settings: Settings, namespace: str | None = None) -> None:
        self._settings = settings
        self._namespace = namespace if namespace is not None else settings.chroma_namespace
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "RagStore initialised — collection={} path={} namespace={}",
            settings.chroma_collection,
            settings.chroma_path,
            self._namespace,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _ns_id(self, chunk_id: str) -> str:
        """Prepend namespace to a chunk ID."""
        return f"{self._namespace}:{chunk_id}"

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add pre-embedded chunks to the collection.

        Each chunk dict must contain:
            id, content, embedding, filename, page, chunk_index
        """
        if not chunks:
            return

        ids = [self._ns_id(c["id"]) for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [
            {
                "filename": c["filename"],
                "page": str(c["page"]),
                "chunk_index": int(c["chunk_index"]),
                "added_at": datetime.now(timezone.utc).isoformat(),
                "namespace": self._namespace,
                "file_type": c.get("file_type", "document"),
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Added {} chunks to collection (namespace={})", len(chunks), self._namespace)

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
        # Merge caller filters with namespace filter
        ns_filter: dict = {"namespace": self._namespace}
        if filters:
            # Combine with $and when both are present
            where: dict | None = {"$and": [{k: v} for k, v in {**ns_filter, **filters}.items()]}
        else:
            where = ns_filter
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
                        if k not in {"filename", "page", "chunk_index", "namespace"}
                    },
                }
            )

        return hits

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks belonging to *filename* in the current namespace."""
        result = self._collection.get(
            where={"$and": [{"filename": filename}, {"namespace": self._namespace}]},
            include=["metadatas"],
        )
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info(
                "Deleted {} chunks for filename={} namespace={}",
                len(ids),
                filename,
                self._namespace,
            )
        return len(ids)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def list_documents(self) -> list[DocumentInfo]:
        """Aggregate stored chunks by filename within the current namespace."""
        result = self._collection.get(
            where={"namespace": self._namespace},
            include=["metadatas"],
        )
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

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespace values across the entire collection."""
        result = self._collection.get(include=["metadatas"])
        namespaces: set[str] = set()
        for meta in result["metadatas"]:
            ns = meta.get("namespace")
            if ns is not None:
                namespaces.add(ns)
        return sorted(namespaces)

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()
