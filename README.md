# ragcore

**Pure RAG-as-a-service.** ragcore retrieves relevant document chunks and returns them — it does **not** generate text, call any LLM, or make any AI decisions. Any AI system (Claude, GPT-4, Llama, custom agents) can query ragcore to obtain grounded context from your own documents, then decide how to use that context themselves.

---

## Architecture

```
Any AI / Agent
  │  "I need context about X"
  ▼
ragcore
  ├── embed query  (sentence-transformers all-MiniLM-L6-v2)
  ├── vector search  (ChromaDB — embedded, zero infra)
  ├── rerank  (CrossEncoder ms-marco-MiniLM-L-2-v2)
  └── return ranked chunks with metadata
  │
  ▼
Your documents  (PDF, DOCX, TXT, XLSX, code, markdown …)

Interfaces:
  ┌─────────────────────────────────────────┐
  │  MCP server  (port 8001, SSE transport) │  ← Claude Desktop, Claude Code, Continue.dev
  │  REST API    (port 8000, OpenAI compat) │  ← any tool using the OpenAI SDK
  └─────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourname/ragcore && cd ragcore

# 2. Copy env
cp .env.example .env

# 3. Run (Docker)
docker compose up --build

# REST API → http://localhost:8000
# MCP SSE  → http://localhost:8001/sse
```

Or run locally without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m ragcore.main
```

---

## Integration guides

### Claude Desktop (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragcore": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

### Claude Code (MCP)

```bash
claude mcp add ragcore --transport sse http://localhost:8001/sse
```

### Continue.dev

In `.continue/config.json`:

```json
{
  "contextProviders": [
    {
      "name": "http",
      "params": {
        "url": "http://localhost:8000/search",
        "title": "ragcore",
        "description": "Local knowledge base"
      }
    }
  ]
}
```

### Any OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000", api_key="not-needed")

response = client.chat.completions.create(
    model="ragcore",
    messages=[{"role": "user", "content": "What is retrieval augmented generation?"}],
)

# content is a JSON string of retrieved chunks — NOT an LLM answer
import json
chunks = json.loads(response.choices[0].message.content)
print(chunks["results"])
```

### LangChain custom retriever

```python
from langchain.schema import BaseRetriever, Document
import httpx

class RagcoreRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        r = httpx.post(
            "http://localhost:8000/search",
            json={"query": query, "top_n": 5},
        )
        return [
            Document(page_content=c["content"], metadata={"filename": c["filename"]})
            for c in r.json()["results"]
        ]
```

---

## API reference

### Native endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Main RAG endpoint — `SearchRequest` → `SearchResponse` |
| `POST` | `/documents/upload` | Multipart upload → chunk + embed + store |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{filename}` | Delete all chunks for a file |
| `GET` | `/health` | Liveness + readiness probe |

### OpenAI-compatible endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | Returns `[{id: "ragcore", …}]` |
| `POST` | `/v1/embeddings` | Standard OpenAI embeddings format |
| `POST` | `/v1/chat/completions` | Extracts last user message → runs RAG → returns chunks as JSON in `content` |

### MCP tools (port 8001)

| Tool | Description |
|------|-------------|
| `search_knowledge_base(query, top_n)` | Search and return ranked chunks |
| `list_documents()` | List all indexed documents |
| `get_document_count()` | Total chunks + documents indexed |

---

## Configuration

All settings can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PATH` | `./data/chroma` | ChromaDB storage directory |
| `CHROMA_COLLECTION` | `ragcore` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-2-v2` | CrossEncoder rerank model |
| `TOP_K` | `10` | Vector search candidates |
| `TOP_N` | `5` | Final results after reranking |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | REST API port |
| `MCP_PORT` | `8001` | MCP SSE server port |

---

## Supported file formats

PDF, DOCX, TXT, XLSX/XLS, Markdown, Python, TypeScript, JavaScript, JSON, YAML, TOML, CSV
