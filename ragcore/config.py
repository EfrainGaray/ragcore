from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ChromaDB
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "ragcore"
    chroma_namespace: str = "default"

    # Embedding provider — "local" (sentence-transformers) or "openai" (any OpenAI-compatible API)
    embedding_provider: str = "local"
    embedding_api_url: str = ""   # alias (openai/huggingface/nvidia/together/groq/ollama) or full base URL
    embedding_api_key: str = ""   # API key for remote providers

    # Reranker provider — "local" (cross-encoder) or "cohere" / "jina" / "voyage" / custom URL
    rerank_provider: str = "local"
    rerank_api_url: str = ""   # alias (cohere/jina/voyage) or full base URL
    rerank_api_key: str = ""   # API key for remote providers

    # Embedding + rerank models
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    # Retrieval params
    top_k: int = 10       # candidates from vector search
    top_n: int = 5        # results after reranking

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Hybrid search (BM25 + dense vectors + RRF)
    hybrid_search: bool = False   # enable BM25 + dense fusion
    bm25_weight: float = 0.3      # legacy — RRF doesn't use weights, keep for future use

    # HyDE (Hypothetical Document Embeddings)
    hyde_enabled: bool = False
    hyde_llm_url: str = ""    # e.g. "https://api.openai.com/v1" or alias
    hyde_llm_key: str = ""
    hyde_llm_model: str = "gpt-4o-mini"

    # Query Expansion (RAG-Fusion)
    query_expansion_enabled: bool = False
    query_expansion_count: int = 3   # number of alternative phrasings

    # Parent-Child Chunks
    parent_child_chunks: bool = False
    parent_chunk_size: int = 1536

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    mcp_port: int = 8001   # MCP server port (SSE transport)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
