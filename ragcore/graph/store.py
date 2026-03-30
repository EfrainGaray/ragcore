"""GraphRAG knowledge graph store."""
from __future__ import annotations

import re
from collections import defaultdict
from loguru import logger


class GraphStore:
    """Extracts entities and relations from text chunks, stores as NetworkX graph."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._spacy_model = spacy_model
        self._nlp = None
        self._graph = None  # networkx.Graph
        # entity -> list of chunk dicts that mention it
        self._entity_chunks: dict[str, list[dict]] = defaultdict(list)
        self._initialized = False

    def _init_spacy(self) -> None:
        """Lazy-load spacy. Falls back to regex extraction on failure."""
        if self._nlp is not None:
            return
        try:
            import spacy
            try:
                self._nlp = spacy.load(self._spacy_model)
            except OSError:
                # Model not downloaded — use blank pipeline
                self._nlp = spacy.blank("en")
            logger.debug("GraphStore: spacy loaded ({})", self._spacy_model)
        except ImportError:
            logger.warning("GraphStore: spacy not installed, using regex entity extraction")
            self._nlp = None

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        self._init_spacy()

        if self._nlp is not None:
            try:
                doc = self._nlp(text[:5000])  # Limit text length
                entities = []
                # Use NER labels if available, else noun chunks
                if hasattr(doc, "ents") and list(doc.ents):
                    entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2]
                elif hasattr(doc, "noun_chunks"):
                    entities = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
                return list(set(entities))
            except Exception as e:
                logger.warning("GraphStore: spacy extraction failed ({}), using regex", e)

        # Regex fallback: extract capitalized multi-word phrases
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        return list(set(m for m in matches if len(m) > 3))

    def build(self, chunks: list[dict]) -> None:
        """Build knowledge graph from chunks."""
        try:
            import networkx as nx
        except ImportError:
            logger.warning("GraphStore: networkx not installed, graph will be empty")
            self._graph = None
            return

        self._graph = nx.Graph()
        self._entity_chunks.clear()

        for chunk in chunks:
            content = chunk.get("content", "")
            entities = self._extract_entities(content)

            # Add nodes and link entities to chunk
            for entity in entities:
                if not self._graph.has_node(entity):
                    self._graph.add_node(entity, frequency=0)
                self._graph.nodes[entity]["frequency"] = self._graph.nodes[entity].get("frequency", 0) + 1
                self._entity_chunks[entity].append(chunk)

            # Add co-occurrence edges between entities in same chunk
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if self._graph.has_edge(e1, e2):
                        self._graph[e1][e2]["weight"] += 1
                    else:
                        self._graph.add_edge(e1, e2, weight=1)

        self._initialized = True
        logger.info(
            "GraphStore: built graph with {} nodes, {} edges from {} chunks",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(chunks),
        )

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        """Get neighbor entities and their associated chunks up to given depth."""
        if self._graph is None or not self._graph.has_node(entity):
            return []

        try:
            import networkx as nx  # noqa: F401
            # BFS up to given depth
            nodes_at_depth = set()
            frontier = {entity}
            for _ in range(depth):
                next_frontier = set()
                for node in frontier:
                    neighbors = set(self._graph.neighbors(node)) - nodes_at_depth - {entity}
                    next_frontier.update(neighbors)
                nodes_at_depth.update(next_frontier)
                frontier = next_frontier
        except Exception:
            nodes_at_depth = set()

        results = []
        for node in list(nodes_at_depth)[:20]:  # Cap at 20 neighbors
            chunks = self._entity_chunks.get(node, [])
            results.append({
                "entity": node,
                "chunks": chunks[:3],  # Top 3 chunks per entity
                "edge_weight": self._graph[entity][node]["weight"] if self._graph.has_edge(entity, node) else 0,
            })

        return sorted(results, key=lambda x: x["edge_weight"], reverse=True)
