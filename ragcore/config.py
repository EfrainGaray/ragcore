from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ChromaDB
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "ragcore"
    chroma_namespace: str = "default"

    # Embedding + rerank models
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    # Retrieval params
    top_k: int = 10       # candidates from vector search
    top_n: int = 5        # results after reranking

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    mcp_port: int = 8001   # MCP server port (SSE transport)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
