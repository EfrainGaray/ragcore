"""Reranker backends for ragcore.

Providers
---------
- ``local``  — sentence-transformers CrossEncoder (default, fully offline).
- ``cohere`` — Cohere Rerank API (free tier: 1 000 reranks/month).
- ``jina``   — Jina AI Rerank API (free tier available).
- ``voyage`` — Voyage AI Rerank API (free credits on sign-up).
- Any URL you provide that speaks the same ``POST /v1/rerank`` format.

All remote providers share the same request/response shape:

  POST /v1/rerank
  {
    "model": "...",
    "query": "...",
    "documents": ["doc0", "doc1", ...],
    "top_n": N        # optional — we omit it and sort ourselves
  }

  Response: {"results": [{"index": 0, "relevance_score": 0.98}, ...]}

Known aliases (set RERANK_API_URL to the alias or a full base URL):

  Alias    URL
  ───────────────────────────────────────────────────
  cohere   https://api.cohere.com/v1
  jina     https://api.jina.ai/v1
  voyage   https://api.voyageai.com/v1

Usage via env vars
------------------
  RERANK_PROVIDER=cohere
  RERANK_API_URL=cohere            # alias or full URL
  RERANK_API_KEY=your-cohere-key
  RERANK_MODEL=rerank-v3.5

  # Jina example:
  RERANK_PROVIDER=jina
  RERANK_API_KEY=jina_xxxxxxxxxxxx
  RERANK_MODEL=jina-reranker-v2-base-multilingual

  # Voyage example:
  RERANK_PROVIDER=voyage
  RERANK_API_KEY=pa-xxxxxxxxxxxx
  RERANK_MODEL=voyage-rerank-2
"""

from __future__ import annotations

_KNOWN_ALIASES: dict[str, str] = {
    "cohere":  "https://api.cohere.com/v1",
    "jina":    "https://api.jina.ai/v1",
    "voyage":  "https://api.voyageai.com/v1",
}

# Sensible default model per provider alias
_DEFAULT_MODELS: dict[str, str] = {
    "cohere":  "rerank-v3.5",
    "jina":    "jina-reranker-v2-base-multilingual",
    "voyage":  "voyage-rerank-2",
}


class LocalReranker:
    """Wraps ``sentence_transformers.CrossEncoder`` — runs fully offline."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Return one relevance score per (query, doc) pair."""
        raw = self._model.predict(pairs)
        return raw.tolist() if hasattr(raw, "tolist") else list(raw)


class RemoteReranker:
    """Calls any server that exposes ``POST /v1/rerank`` (Cohere / Jina / Voyage style).

    The ``predict(pairs)`` interface mirrors CrossEncoder so ``Retriever``
    needs no changes — it still calls ``predict([[query, doc], ...])``.
    Internally the query is extracted from ``pairs[0][0]`` and docs from
    ``pairs[i][1]``, then the API is called once.  Scores are returned in
    the same order as the input pairs.
    """

    def __init__(self, model_name: str, api_url: str, api_key: str) -> None:
        import httpx

        self.model_name = model_name
        base = _KNOWN_ALIASES.get(api_url.lower(), api_url)
        self._endpoint = base.rstrip("/") + "/rerank"
        self._model_name = model_name
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=30.0)

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Return one relevance score per pair, in original input order."""
        if not pairs:
            return []

        query = pairs[0][0]
        docs = [p[1] for p in pairs]

        response = self._client.post(
            self._endpoint,
            headers=self._headers,
            json={
                "model": self._model_name,
                "query": query,
                "documents": docs,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Build index → score map; APIs return results sorted by score,
        # not necessarily in input order.
        results = data.get("results") or data.get("data") or []
        score_map: dict[int, float] = {
            r["index"]: float(r.get("relevance_score") or r.get("score") or 0.0)
            for r in results
        }

        # Fill in zeros for any index the API omitted (shouldn't happen normally)
        return [score_map.get(i, 0.0) for i in range(len(pairs))]


def build_reranker(settings) -> LocalReranker | RemoteReranker:
    """Factory — pick the right backend from settings."""
    provider = settings.rerank_provider.lower()
    if provider == "local":
        return LocalReranker(settings.rerank_model)

    # Remote providers
    api_url = settings.rerank_api_url or provider  # allow alias as provider name
    if not settings.rerank_api_key:
        raise ValueError(
            f"RERANK_API_KEY must be set when RERANK_PROVIDER={provider!r}"
        )

    # If the user didn't set a custom model, use a sensible default for the provider
    model = settings.rerank_model
    alias = api_url.lower()
    if model == "cross-encoder/ms-marco-MiniLM-L-2-v2" and alias in _DEFAULT_MODELS:
        model = _DEFAULT_MODELS[alias]

    return RemoteReranker(
        model_name=model,
        api_url=api_url,
        api_key=settings.rerank_api_key,
    )
