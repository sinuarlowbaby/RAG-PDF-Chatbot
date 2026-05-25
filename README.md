# RAG PDF Chatbot

A production-grade Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **Qdrant**, and **Groq**. Upload PDFs and ask questions — the pipeline retrieves relevant passages and generates grounded answers via a streaming API.

## Architecture

```
User Query
     │
Multi-Query Generation (Groq LLM)
     │
Hybrid Retrieval
  ├─ Vector Search  (Qdrant + OpenAI embeddings, MMR)
  └─ Keyword Search (BM25)
     │
Deduplication
     │
Cross-Encoder Reranking
     │
Token-Limited Context Builder
     │
LLM Streaming Generation (Groq — Llama 3.3 70B)
     │
Streamed Answer (SSE)
```

## Project Structure

```
app/
├── app.py                  # FastAPI app entry point, lifespan, middleware
├── routes/v1/
│   ├── chat_router.py      # POST /api/v1/ask  — streaming chat endpoint
│   └── upload_router.py    # POST /api/v1/upload — PDF upload endpoint
├── pipeline/
│   ├── ingest_pipeline.py  # Orchestrates load → chunk → embed → store
│   └── query_pipeline.py   # Orchestrates query → retrieve → rerank → generate
├── ingestion/
│   ├── loader.py           # PDF loading via PyPDFLoader
│   └── chunker.py          # Token-aware recursive text splitting
├── retrieval/
│   ├── hybrid_document_retrieval.py  # BM25 + MMR ensemble retriever
│   ├── reranker.py         # Cross-encoder reranking
│   ├── deduplication.py    # Hash-based chunk deduplication
│   └── build_context.py    # Token-budget context concatenation
├── llm/
│   ├── llm_client.py       # Groq streaming LLM client
│   └── multi_query.py      # LLM-based query expansion
├── vector_store/
│   └── vector_store.py     # Qdrant collection management + ingestion
├── utils/
│   ├── semantic_cache.py   # Redis-backed semantic similarity cache
│   └── time_calculate.py   # Simple wall-clock timer helper
├── schema/
│   └── llm_schemas.py      # Pydantic request/response models
└── templates/
    └── rag_chatbot.html    # Chat UI served at /
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Qdrant + Redis/Valkey)

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd RAG-PDF-Chatbot
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start infrastructure

```bash
docker-compose up vector-db valkey -d
```

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 4. Run the app

```bash
cd app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the chat UI or **http://localhost:8000/docs** for the Swagger API docs.

### 5. Run with Docker (full stack)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI (HTML) |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload PDF files; returns `session_id` |
| `POST` | `/api/v1/ask` | Stream an answer; requires `X-Session-Id` header |

### Example: Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "files=@my_document.pdf"
# → {"session_id": "uuid", "message": "...", "documents": 1}
```

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: <session_id_from_upload>" \
  -d '{"question": "What is this document about?"}' \
  --no-buffer
```

## Environment Variables

See [`.env.example`](.env.example) for all available variables. Key ones:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | For text-embedding-3-small | — |
| `GROQ_API_KEY` | For Llama 3.3 70B generation | — |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `REDIS_URL` | Redis/Valkey URL | `redis://localhost:6379` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `http://localhost:8000` |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | `global_rag_store` |