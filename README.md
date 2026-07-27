<div align="center">

# ⚡ Production RAG PDF Chatbot

**A Production-Grade Retrieval-Augmented Generation (RAG) System with Multi-Query Expansion, Hybrid Retrieval, Redis Semantic Caching, Cross-Encoder Reranking & Full LLM Observability.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-DB-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Observability-000000?style=for-the-badge&logo=langfuse&logoColor=white)](https://langfuse.com/)
[![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## 🌟 Key Features

- **⚡ Redis Dual-Layer Caching**:
  - **Semantic Answer Cache**: Sub-second instant cache hits for semantically similar questions using cosine similarity thresholding.
  - **Multi-Query Cache**: Caches sub-query expansions in Redis (`TTL = 1h`) to eliminate redundant LLM expansion calls.
- **🔍 Hybrid Document Retrieval**: Combines semantic vector search (**Qdrant MMR**) with keyword BM25 search (**EnsembleRetriever**).
- **🧠 Multi-Query Expansion**: Generates semantically diverse sub-queries to maximize retrieval recall from uploaded PDFs.
- **🎯 Cross-Encoder Reranking**: Re-scores deduplicated candidate chunks using `ms-marco-MiniLM-L-6-v2` cross-encoder for precision relevance.
- **📡 Server-Sent Events (SSE) Streaming**: Real-time token-by-token answer streaming powered by **Groq Llama 3.3 70B**.
- **📊 Integrated Web UI & Observability**:
  - Interactive single-page web application featuring dynamic upload states, cache hit indicators, and Redis cache management.
  - Live **RedisInsight** dashboard (`port 5540`) for cache inspection.
  - Full **Langfuse** sub-graph tracing (`port 3000`) for cost, latency, and execution analysis.

---

## 🏗️ Architecture Flow

```text
User Question
      │
┌─────▼─────────────────────────────────────────────────────────────┐
│ ⚡ Redis Semantic Answer Cache Lookup (Cosine Similarity ≥ 0.8)     │
└─────┬───────────────────────────────────────────────┬─────────────┘
      │ Cache Hit (Instant Response)                  │ Cache Miss
      ▼                                               ▼
⚡ Return Cached Answer & Context            Multi-Query Expansion (Groq / Redis Cache)
                                                      │
                                             Hybrid Retrieval
                                           ┌──────────┴──────────┐
                                     Qdrant MMR Vector       BM25 Keyword
                                           └──────────┬──────────┘
                                                      │
                                            Deduplication & Hash Check
                                                      │
                                            Cross-Encoder Reranking (MiniLM)
                                                      │
                                            Token-Budgeted Context Builder
                                                      │
                                            LLM Generation (Llama 3.3 70B)
                                                      │
                                            Streamed Answer (SSE)
```

---

## 📁 Project Structure

```text
RAG-PDF-Chatbot/
├── app/
│   ├── app.py                     # FastAPI entrypoint, middleware, lifespan
│   ├── config.py                  # Centralized Pydantic-settings config
│   ├── llm.py                     # Groq LLM streaming & Multi-Query expansion
│   ├── ingest.py                  # PyPDFLoader & document text chunker
│   ├── vector_store.py            # Qdrant collection management & vector ingestion
│   ├── pipeline/
│   │   ├── ingest_pipeline.py     # Load → Chunk → Embed → Store orchestration
│   │   └── query_pipeline.py      # Multi-query → Retrieve → Rerank → Stream response
│   ├── retrieval/
│   │   ├── hybrid.py              # Qdrant MMR + BM25 Ensemble retriever
│   │   ├── deduplication.py       # Hash-based document chunk deduplication
│   │   ├── reranker.py            # Sentence-Transformers Cross-Encoder scoring
│   │   └── build_context.py       # Token-budget context builder (tiktoken)
│   ├── routes/
│   │   ├── chat_router.py         # POST /api/v1/ask — SSE streaming answer endpoint
│   │   ├── upload_router.py       # POST /api/v1/upload — PDF file ingestion
│   │   └── cache_router.py        # GET/DELETE /api/v1/cache — Redis cache stats & purge
│   ├── utils/
│   │   └── semantic_cache.py      # Redis vector similarity search & cache manager
│   ├── schema/
│   │   └── llm_schemas.py         # Pydantic request & response schemas
│   └── templates/
│       └── rag_chatbot.html       # Web UI with cache toggles & document manager
├── tests/
│   ├── conftest.py                # Pytest mock fixtures (Qdrant, Redis, OpenAI)
│   ├── test_unit.py               # Core unit tests
│   └── test_integration.py        # FastAPI client integration tests
├── docker-compose.yaml            # Main services (App, Qdrant, Valkey, RedisInsight)
├── docker-compose.langfuse.yml    # Observability stack (Langfuse, Postgres, ClickHouse, MinIO)
└── requirements.txt               # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.11+
- **Docker & Docker Compose**: Installed & running
- **API Keys**: Groq API Key & OpenAI API Key

---

### 1️⃣ Clone & Configure Environment

```bash
git clone https://github.com/sinuarlowbaby/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot

# Copy environment template
cp .env.example .env
```

Edit your `.env` file and populate your keys:
```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

---

### 2️⃣ Start Backend Infrastructure (Docker)

Launch Qdrant Vector DB, Valkey (Redis), and RedisInsight Web GUI:

```bash
docker compose up vector-db valkey redis-ui -d
```

---

### 3️⃣ Start Langfuse Tracing (Optional / Observability Stack)

```bash
docker compose -f docker-compose.langfuse.yml up -d
```
> Open **http://localhost:3000** → Create a Project → Generate API Keys → Paste `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` into your `.env` file.

---

### 4️⃣ Run the Application

#### Option A: Local Python Execution
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start FastAPI application
python app/app.py
```

#### Option B: Full Docker Stack Execution
```bash
docker compose up --build -d
```

---

## 🌐 Quick Access Dashboard URLs

| Service / Dashboard | URL | Description |
| :--- | :--- | :--- |
| 💬 **Web Chat Application** | **[http://localhost:8000](http://localhost:8000)** | Interactive RAG Chat UI |
| 📖 **FastAPI Swagger Specs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | OpenAPI Endpoint Documentation |
| ⚡ **RedisInsight Web GUI** | **[http://localhost:5540](http://localhost:5540)** | Visual Redis Cache Inspector |
| 🔍 **Langfuse Observability** | **[http://localhost:3000](http://localhost:3000)** | Execution Traces & Latency Graphs |
| 🎯 **Qdrant Vector Dashboard** | **[http://localhost:6333/dashboard](http://localhost:6333/dashboard)** | Qdrant Vector Collection Inspector |

---

## 📡 API Endpoints Specification

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the web chat single-page application |
| `GET` | `/health` | Application health check endpoint |
| `POST` | `/api/v1/upload` | Upload PDF files for chunking & vector embedding |
| `POST` | `/api/v1/ask` | Stream RAG answer (requires `x-session-id` header) |
| `GET` | `/api/v1/cache/stats` | Returns current Redis cache key counts & connectivity |
| `DELETE` | `/api/v1/cache/clear` | Flushes all semantic and multi-query cache entries |

---

## 🧪 Running Unit & Integration Tests

Execute the automated test suite with `pytest`:

```bash
# Run unit & integration tests
pytest tests/test_unit.py tests/test_integration.py

# Run coverage report
pytest --cov=app --cov-report=term-missing
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
