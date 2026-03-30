"""Embedding backends for ragcore.

Providers
---------
- ``local``  — sentence-transformers running in-process (default, no API key needed).
- ``openai`` — any server that speaks the OpenAI ``POST /v1/embeddings`` format.

OpenAI-compatible providers (set EMBEDDING_API_URL to the alias or a full base URL):

  Alias         URL
  ─────────────────────────────────────────────────────────────────
  openai        https://api.openai.com/v1
  huggingface   https://api-inference.huggingface.co/v1
  nvidia        https://integrate.api.nvidia.com/v1
  together      https://api.together.xyz/v1
  groq          https://api.groq.com/openai/v1
  ollama        http://localhost:11434/v1
  (or any full URL you provide)

Usage via env vars
------------------
  EMBEDDING_PROVIDER=openai
  EMBEDDING_API_URL=huggingface          # alias or full URL
  EMBEDDING_API_KEY=hf_xxxxxxxxxxxx
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

_KNOWN_ALIASES: dict[str, str] = {
    "openai":       "https://api.openai.com/v1",
    "huggingface":  "https://api-inference.huggingface.co/v1",
    "nvidia":       "https://integrate.api.nvidia.com/v1",
    "together":     "https://api.together.xyz/v1",
    "groq":         "https://api.groq.com/openai/v1",
    "ollama":       "http://localhost:11434/v1",
}


class LocalEmbedder:
    """Wraps ``sentence_transformers.SentenceTransformer`` — runs fully offline."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str], **kwargs):
        """Delegate to the underlying SentenceTransformer (returns numpy array)."""
        return self._model.encode(texts, **kwargs)


class OpenAICompatibleEmbedder:
    """Calls any server that exposes ``POST /v1/embeddings`` in OpenAI format.

    The call is synchronous so it can be safely dispatched with
    ``loop.run_in_executor`` (same pattern as LocalEmbedder).
    """

    def __init__(self, model_name: str, api_url: str, api_key: str) -> None:
        import httpx

        self.model_name = model_name
        base = _KNOWN_ALIASES.get(api_url.lower(), api_url)
        self._endpoint = base.rstrip("/") + "/embeddings"
        self._model_name = model_name
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=30.0)

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Return a list of embedding vectors (one per input text)."""
        response = self._client.post(
            self._endpoint,
            headers=self._headers,
            json={"input": texts, "model": self._model_name},
        )
        response.raise_for_status()
        data = response.json()
        # OpenAI format: {"data": [{"index": 0, "embedding": [...]}]}
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


def build_embedder(settings) -> LocalEmbedder | OpenAICompatibleEmbedder:
    """Factory — pick the right backend from settings."""
    if settings.embedding_provider == "openai":
        if not settings.embedding_api_url:
            raise ValueError(
                "EMBEDDING_API_URL must be set when EMBEDDING_PROVIDER=openai.\n"
                f"Known aliases: {', '.join(_KNOWN_ALIASES)}"
            )
        return OpenAICompatibleEmbedder(
            model_name=settings.embedding_model,
            api_url=settings.embedding_api_url,
            api_key=settings.embedding_api_key,
        )
    return LocalEmbedder(settings.embedding_model)
