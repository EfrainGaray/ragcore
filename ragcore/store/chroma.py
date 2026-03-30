"""ChromaDB wrapper for ragcore."""

from __future__ import annotations

from datetime import datetime, timezone
from collections import defaultdict

import chromadb
from loguru import logger

from ragcore.config import Settings
from ragcore.models import DocumentInfo
from ragcore.store.bm25 import BM25Index


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
        self._bm25: BM25Index | None = None
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
                "chunk_type": c.get("chunk_type", "document"),
                **{k: v for k, v in c.items() if k.startswith("_meta_")},
            }
            for c in chunks
        ]
        # Merge extra metadata and top-level convenience fields
        for i, c in enumerate(chunks):
            extra = c.get("extra_metadata", {})
            if extra:
                metadatas[i].update(extra)
            # Support parent_id and is_parent as top-level chunk keys
            if "parent_id" in c:
                metadatas[i]["parent_id"] = c["parent_id"]
            if "is_parent" in c:
                metadatas[i]["is_parent"] = c["is_parent"]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Added {} chunks to collection (namespace={})", len(chunks), self._namespace)

        # Rebuild BM25 index with updated corpus
        self._rebuild_bm25()

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from all documents in this namespace."""
        result = self._collection.get(
            where={"namespace": self._namespace},
            include=["documents", "metadatas"],
        )
        ids: list[str] = result["ids"]
        docs: list[str] = result["documents"]
        metas: list[dict] = result["metadatas"]

        filtered_ids = []
        filtered_docs = []
        for doc_id, doc, meta in zip(ids, docs, metas):
            if not meta.get("is_parent", False) and meta.get("chunk_type") != "parent":
                filtered_ids.append(doc_id)
                filtered_docs.append(doc)

        if self._bm25 is None:
            self._bm25 = BM25Index()
        self._bm25.build(filtered_docs, filtered_ids)

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

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        """Hybrid search: dense vector search + BM25, fused with RRF (k=60).

        Returns a merged list sorted by RRF score, deduped by id.
        """
        k = 60  # RRF constant

        # 1. Dense search
        dense_hits = self.search(query_embedding, top_k, filters)

        # 2. BM25 search
        bm25_index = self._bm25
        if bm25_index is not None:
            bm25_raw = bm25_index.search(query_text, top_k)
        else:
            bm25_raw = []

        # Build a lookup of id -> hit dict from dense results
        hits_by_id: dict[str, dict] = {h["id"]: h for h in dense_hits}

        # Collect all unique ids in BM25 results (ids may not be in dense results)
        # We need content/meta for BM25-only hits — fetch from collection
        bm25_only_ids = [bid for bid, _ in bm25_raw if bid not in hits_by_id]
        if bm25_only_ids:
            extra = self._collection.get(
                ids=bm25_only_ids,
                include=["documents", "metadatas"],
            )
            for doc_id, doc, meta in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                hits_by_id[doc_id] = {
                    "id": doc_id,
                    "content": doc,
                    "score": 0.0,
                    "filename": meta.get("filename", ""),
                    "page": meta.get("page", "0"),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "metadata": {
                        k2: v
                        for k2, v in meta.items()
                        if k2 not in {"filename", "page", "chunk_index", "namespace"}
                    },
                }

        # 3. RRF fusion
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            doc_id = hit["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        for rank, (doc_id, _bm25_score) in enumerate(bm25_raw, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # 4. Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        merged = []
        for doc_id in sorted_ids[:top_k]:
            hit = hits_by_id.get(doc_id)
            if hit is not None:
                result = dict(hit)
                result["score"] = rrf_scores[doc_id]
                merged.append(result)

        return merged

    def get_parent_chunk(self, parent_id: str) -> str | None:
        """Retrieve the content of a parent chunk by its ID.

        Tries the namespaced ID first (`{namespace}:{parent_id}`), then the raw ID.
        Returns None if not found.
        """
        candidates = [f"{self._namespace}:{parent_id}", parent_id]
        for id_to_try in candidates:
            try:
                result = self._collection.get(
                    ids=[id_to_try],
                    include=["documents"],
                )
                docs = result.get("documents", [])
                if docs and docs[0]:
                    return docs[0]
            except Exception:
                pass
        return None

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
            # Rebuild BM25 index after deletion
            self._rebuild_bm25()
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
