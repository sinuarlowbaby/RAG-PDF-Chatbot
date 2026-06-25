"""
app/app.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Responsibilities:
  - Bootstrap logging
  - Initialise shared infrastructure (Qdrant, Redis, embeddings, reranker)
    via the FastAPI lifespan context manager
  - Register middleware (CORS)
  - Mount API routers
  - Serve the frontend template
  - Expose /health endpoint
─────────────────────────────────────────────────────────────────────────────
"""

# ── Standard library ─────────────────────────────────────────────────────────
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

# ── Third-party ──────────────────────────────────────────────────────────────
import openai
import redis
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder

# ── Internal ─────────────────────────────────────────────────────────────────
from config import settings, setup_logging
from routes.v1.chat_router import chat_router
from routes.v1.upload_router import upload_router
from schema.llm_schemas import HealthResponse

# ── Bootstrap logging first, before any logger is created ────────────────────
setup_logging(settings)
logger = logging.getLogger(__name__)

# ── Set OpenAI key from centralised settings ──────────────────────────────────
openai.api_key = settings.openai_api_key


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown logic
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down shared infrastructure resources."""
    logger.info("Starting RAG pipeline...")
    try:
        # ── Qdrant vector database ────────────────────────────────────────────
        app.state.client = QdrantClient(url=settings.qdrant_url)

        existing_collections = [
            c.name for c in app.state.client.get_collections().collections
        ]
        if settings.qdrant_collection_name not in existing_collections:
            app.state.client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{settings.qdrant_collection_name}'")

        # ── Cross-encoder reranker ────────────────────────────────────────────
        app.state.reranker = CrossEncoder(settings.reranker_model)

        # ── OpenAI embeddings (shared, not per-request) ───────────────────────
        app.state.embedding_model = OpenAIEmbeddings(
            model=settings.embedding_model,
            chunk_size=settings.embedding_chunk_size,
        )

        # ── Qdrant vector store wrapper ───────────────────────────────────────
        app.state.vector_store = QdrantVectorStore(
            client=app.state.client,
            embedding=app.state.embedding_model,
            collection_name=settings.qdrant_collection_name,
        )

        # ── Redis cache ───────────────────────────────────────────────────────
        app.state.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=0,
            decode_responses=True,
        )

        # ── In-memory session store ───────────────────────────────────────────
        app.state.sessions = {}

        logger.info("All services initialised successfully.")

    except Exception as exc:
        logger.error(f"Failed to initialise server: {exc}")
        raise

    logger.info("FastAPI server is ready!")
    logger.info(f"Swagger UI  -> http://localhost:{settings.port}/docs")
    logger.info(f"Home Page   -> http://localhost:{settings.port}")

    yield  # ← application handles requests here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down RAG pipeline...")
    app.state.client.close()
    logger.info("RAG pipeline shut down successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG PDF Chatbot",
    description="Upload PDFs and ask questions using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Templates ─────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jinja2_env = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(upload_router)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serve the main chatbot UI."""
    return jinja2_env.TemplateResponse(request=request, name="rag_chatbot.html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health-check endpoint for Docker / load-balancer probes."""
    return {
        "status": "ok",
        "pipeline_ready": True,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change to the app directory so module imports resolve correctly
    # when uvicorn's --reload spawns a child process.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,          # let our logging config take precedence
        log_level=settings.log_level,
    )