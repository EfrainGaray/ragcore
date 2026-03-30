"""Tests for ragcore.graph — GraphStore and GraphRetriever."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies before any ragcore.graph import.
# spacy and networkx may not be installed in the test environment.
# ---------------------------------------------------------------------------

# --- spacy stub ---
if "spacy" not in sys.modules:
    _spacy_mod = types.ModuleType("spacy")

    class _FakeNLP:
        """Minimal spacy language model stub."""

        def __call__(self, text: str):
            doc = MagicMock()
            # Return two fake named entities for any non-empty text
            if text.strip():
                ent1 = MagicMock()
                ent1.text = "EntityA"
                ent1.label_ = "ORG"
                ent2 = MagicMock()
                ent2.text = "EntityB"
                ent2.label_ = "PERSON"
                doc.ents = [ent1, ent2]
                # Fake noun chunks
                nc1 = MagicMock()
                nc1.text = "EntityA"
                nc2 = MagicMock()
                nc2.text = "EntityB"
                doc.noun_chunks = [nc1, nc2]
            else:
                doc.ents = []
                doc.noun_chunks = []
            return doc

    _spacy_mod.load = MagicMock(return_value=_FakeNLP())
    _spacy_mod.blank = MagicMock(return_value=_FakeNLP())
    sys.modules["spacy"] = _spacy_mod

# --- networkx stub — only installed if networkx is not already available ---
try:
    import networkx  # noqa: F401
except ModuleNotFoundError:
    _nx_mod = types.ModuleType("networkx")

    class _FakeNodeView:
        """Supports G.nodes[name] and G.nodes[name].get(key, default)."""

        def __init__(self, data: dict):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __contains__(self, key):
            return key in self._data

    class _FakeEdgeAdaptor:
        """Supports G[u][v]['weight'] syntax."""

        def __init__(self, edges: dict, u):
            self._edges = edges
            self._u = u

        def __getitem__(self, v):
            key = (min(self._u, v), max(self._u, v))
            return self._edges.get(key, {})

    class _FakeGraph:
        def __init__(self):
            self._nodes_data: dict = {}
            self._edges_data: dict = {}  # (min,max) -> attrs dict

        @property
        def nodes(self):
            return _FakeNodeView(self._nodes_data)

        def __getitem__(self, u):
            return _FakeEdgeAdaptor(self._edges_data, u)

        def add_node(self, name, **attrs):
            self._nodes_data[name] = dict(attrs)

        def add_edge(self, u, v, **attrs):
            key = (min(u, v), max(u, v))
            self._edges_data[key] = dict(attrs)

        def has_node(self, name):
            return name in self._nodes_data

        def has_edge(self, u, v):
            return (min(u, v), max(u, v)) in self._edges_data

        def neighbors(self, name):
            result = []
            for (a, b) in self._edges_data:
                if a == name:
                    result.append(b)
                elif b == name:
                    result.append(a)
            return result

        def number_of_nodes(self):
            return len(self._nodes_data)

        def number_of_edges(self):
            return len(self._edges_data)

    _nx_mod.Graph = _FakeGraph
    _nx_mod.DiGraph = _FakeGraph
    sys.modules["networkx"] = _nx_mod

from tests.conftest import _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunks(n: int = 4) -> list[dict]:
    return [
        {
            "id": str(i),
            "content": f"EntityA works with EntityB on project number {i}.",
            "filename": "doc.txt",
            "page": 0,
            "chunk_index": i,
        }
        for i in range(n)
    ]


def _make_graph_store():
    from ragcore.graph import GraphStore

    return GraphStore()


def _make_graph_retriever(graph_store):
    from ragcore.graph import GraphRetriever

    return GraphRetriever(graph_store=graph_store, embed_model=_FakeSentenceTransformer())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_store_build_from_chunks():
    """GraphStore.build() must complete without raising for valid chunks."""
    gs = _make_graph_store()
    gs.build(_make_chunks(4))  # must not raise


def test_graph_retriever_search_returns_results():
    """GraphRetriever.search() must return a list (possibly empty) after build."""
    gs = _make_graph_store()
    gs.build(_make_chunks(4))
    gr = _make_graph_retriever(gs)
    results = gr.search("EntityA", top_k=3)
    assert isinstance(results, list)


def test_graph_store_empty_chunks():
    """GraphStore.build([]) must succeed gracefully."""
    gs = _make_graph_store()
    gs.build([])  # must not raise


def test_graph_retriever_top_k():
    """search() must return at most top_k results."""
    gs = _make_graph_store()
    gs.build(_make_chunks(8))
    gr = _make_graph_retriever(gs)
    top_k = 2
    results = gr.search("EntityA EntityB", top_k=top_k)
    assert len(results) <= top_k


def test_graph_store_get_neighbors():
    """get_neighbors() must return a list of dicts after the graph is built."""
    gs = _make_graph_store()
    gs.build(_make_chunks(4))
    neighbors = gs.get_neighbors("EntityA", depth=1)
    assert isinstance(neighbors, list)
    # Each neighbor should be a dict (entity/relation info)
    for n in neighbors:
        assert isinstance(n, dict)
