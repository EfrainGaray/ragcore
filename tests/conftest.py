"""Shared fixtures and sys.modules mocks for ragcore tests.

Heavy ML libraries (sentence_transformers, chromadb) are replaced with
lightweight fakes so tests run instantly without any model downloads.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake sentence_transformers
# ---------------------------------------------------------------------------


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, show_progress_bar=False, **kwargs):
        # Return deterministic unit vectors of dim=8
        n = len(texts) if isinstance(texts, list) else 1
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, 8)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        # Return scores in descending order so the first candidate "wins"
        return np.array([1.0 / (i + 1) for i in range(len(pairs))], dtype=np.float32)


def _make_sentence_transformers_module():
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer
    cross_mod = types.ModuleType("sentence_transformers.cross_encoder")
    cross_mod.CrossEncoder = _FakeCrossEncoder
    mod.CrossEncoder = _FakeCrossEncoder
    # Also patch the submodule path
    sys.modules["sentence_transformers.cross_encoder"] = cross_mod
    return mod


# ---------------------------------------------------------------------------
# Fake chromadb
# ---------------------------------------------------------------------------


class _FakeCollection:
    """In-memory ChromaDB collection."""

    def __init__(self, name: str):
        self.name = name
        self._data: dict[str, dict] = {}  # id -> {document, embedding, metadata}

    def add(self, ids, documents, embeddings, metadatas):
        for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self._data[id_] = {"document": doc, "embedding": emb, "metadata": meta}

    @staticmethod
    def _matches_where(metadata: dict, where: dict) -> bool:
        """Evaluate a ChromaDB where clause (supports plain dict and $and)."""
        if "$and" in where:
            return all(
                _FakeCollection._matches_where(metadata, clause)
                for clause in where["$and"]
            )
        return all(metadata.get(k) == v for k, v in where.items())

    def query(self, query_embeddings, n_results, where=None, include=None):
        items = list(self._data.items())
        # Simple filtering
        if where:
            items = [
                (id_, d)
                for id_, d in items
                if self._matches_where(d["metadata"], where)
            ]
        # Return top n_results (no actual ANN — just first N)
        items = items[:n_results]
        ids = [[i for i, _ in items]]
        docs = [[d["document"] for _, d in items]]
        metas = [[d["metadata"] for _, d in items]]
        distances = [[0.1 * (j + 1) for j in range(len(items))]]
        return {"ids": ids, "documents": docs, "metadatas": metas, "distances": distances}

    def get(self, where=None, include=None, ids=None):
        items = list(self._data.items())
        if ids is not None:
            items = [(id_, d) for id_, d in items if id_ in ids]
        if where:
            items = [
                (id_, d)
                for id_, d in items
                if self._matches_where(d["metadata"], where)
            ]
        return {
            "ids": [i for i, _ in items],
            "documents": [d["document"] for _, d in items],
            "metadatas": [d["metadata"] for _, d in items],
        }

    def delete(self, ids):
        for id_ in ids:
            self._data.pop(id_, None)

    def count(self):
        return len(self._data)


class _FakePersistentClient:
    _collections: dict[str, _FakeCollection] = {}

    def __init__(self, path: str):
        self._path = path
        # Each client instance gets a fresh store so tests don't bleed
        self._collections = {}

    def get_or_create_collection(self, name: str, metadata=None):
        if name not in self._collections:
            self._collections[name] = _FakeCollection(name)
        return self._collections[name]


def _make_chromadb_module():
    mod = types.ModuleType("chromadb")
    mod.PersistentClient = _FakePersistentClient
    return mod


# ---------------------------------------------------------------------------
# Install mocks into sys.modules BEFORE any ragcore import
# ---------------------------------------------------------------------------

sys.modules.setdefault("sentence_transformers", _make_sentence_transformers_module())
sys.modules.setdefault("chromadb", _make_chromadb_module())

# ---------------------------------------------------------------------------
# Fake rank_bm25
# ---------------------------------------------------------------------------


class _FakeBM25Okapi:
    """Minimal BM25Okapi stub — returns zeros for OOV queries, ones otherwise."""

    def __init__(self, tokenized_corpus):
        self._n = len(tokenized_corpus)
        self._vocab: set[str] = set()
        for doc_tokens in tokenized_corpus:
            self._vocab.update(doc_tokens)

    def get_scores(self, query_tokens):
        if not any(t in self._vocab for t in query_tokens):
            return np.zeros(self._n, dtype=np.float32)
        return np.ones(self._n, dtype=np.float32)


def _make_rank_bm25_module():
    mod = types.ModuleType("rank_bm25")
    mod.BM25Okapi = _FakeBM25Okapi
    return mod


# Also stub out heavy optional deps that might be missing in the test env
for _mod in ["pypdf", "docx", "pandas"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

sys.modules.setdefault("rank_bm25", _make_rank_bm25_module())


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_embed_model():
    return _FakeSentenceTransformer()


@pytest.fixture()
def fake_rerank_model():
    return _FakeCrossEncoder()


@pytest.fixture()
def store():
    """A RagStore backed by a fresh in-memory fake ChromaDB."""
    from ragcore.config import Settings
    from ragcore.store.chroma import RagStore

    cfg = Settings(chroma_path="/tmp/test_chroma", chroma_collection="test")
    return RagStore(cfg)


@pytest.fixture()
def retriever(store, fake_embed_model, fake_rerank_model):
    from ragcore.config import Settings
    from ragcore.retrieval import Retriever

    cfg = Settings(top_k=5, top_n=3)
    return Retriever(
        store=store,
        embedding_model=fake_embed_model,
        rerank_model=fake_rerank_model,
        settings=cfg,
    )


@pytest.fixture()
def ingestor(store, fake_embed_model):
    from ragcore.config import Settings
    from ragcore.store.ingest import Ingestor

    cfg = Settings(chunk_size=100, chunk_overlap=10)
    return Ingestor(store=store, embedding_model=fake_embed_model, settings=cfg)


@pytest.fixture()
def client(retriever, ingestor):
    """HTTPX TestClient for the FastAPI app."""
    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport
    from ragcore.server.rest import create_app

    app = create_app(retriever=retriever, ingestor=ingestor)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
