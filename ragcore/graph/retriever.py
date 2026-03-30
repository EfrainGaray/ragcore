"""GraphRAG retriever."""
from __future__ import annotations

import numpy as np
from loguru import logger
from ragcore.graph.store import GraphStore


class GraphRetriever:
    """Retrieves chunks via knowledge graph traversal."""

    def __init__(self, graph_store: GraphStore, embed_model) -> None:
        self._graph = graph_store
        self._embed_model = embed_model

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Find entities in query, traverse graph, return relevant chunks ranked by embedding sim."""
        if self._graph._graph is None:
            return []

        # Extract query entities
        query_entities = self._graph._extract_entities(query)

        # Collect candidate chunks from graph traversal
        candidate_chunks: list[dict] = []
        seen_ids = set()

        for entity in query_entities:
            # Direct entity chunks
            for chunk in self._graph._entity_chunks.get(entity, []):
                chunk_id = chunk.get("id", "")
                if chunk_id not in seen_ids:
                    candidate_chunks.append(chunk)
                    seen_ids.add(chunk_id)

            # Neighbor chunks (depth=1)
            for neighbor_info in self._graph.get_neighbors(entity, depth=1)[:5]:
                for chunk in neighbor_info.get("chunks", []):
                    chunk_id = chunk.get("id", "")
                    if chunk_id not in seen_ids:
                        candidate_chunks.append(chunk)
                        seen_ids.add(chunk_id)

        if not candidate_chunks:
            logger.debug("GraphRetriever: no candidate chunks for query={!r}", query)
            return []

        # Rank by embedding similarity to query
        q_raw = self._embed_model.encode([query], show_progress_bar=False)
        q_vec = np.array(q_raw[0].tolist() if hasattr(q_raw[0], "tolist") else list(q_raw[0]))
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        scored = []
        for chunk in candidate_chunks:
            content = chunk.get("content", "")
            c_raw = self._embed_model.encode([content], show_progress_bar=False)
            c_vec = np.array(c_raw[0].tolist() if hasattr(c_raw[0], "tolist") else list(c_raw[0]))
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm
            score = float(np.dot(q_vec, c_vec))

            result = dict(chunk)
            result["score"] = score
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
