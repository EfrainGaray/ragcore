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

    # Semantic chunking
    semantic_chunking: bool = False
    semantic_chunk_threshold: float = 0.5   # cosine similarity drop threshold
    semantic_chunk_max_size: int = 512

    # RAG Evaluation
    eval_enabled: bool = False
    eval_llm_url: str = ""     # reuses hyde_llm_url if empty
    eval_llm_key: str = ""
    eval_llm_model: str = "gpt-4o-mini"

    # CRAG
    crag_enabled: bool = False
    crag_threshold: float = 0.5    # relevance score below which chunks are dropped
    crag_web_search: bool = False  # supplement with web search if too few chunks survive

    # RAPTOR
    raptor_enabled: bool = False
    raptor_levels: int = 3
    raptor_llm_url: str = ""    # reuses hyde_llm_url if empty
    raptor_llm_key: str = ""
    raptor_llm_model: str = "gpt-4o-mini"

    # Multi-vector retrieval
    multivector_enabled: bool = False

    # GraphRAG
    graphrag_enabled: bool = False
    graphrag_spacy_model: str = "en_core_web_sm"

    # Multi-modal
    multimodal_enabled: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    mcp_port: int = 8001   # MCP server port (SSE transport)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
