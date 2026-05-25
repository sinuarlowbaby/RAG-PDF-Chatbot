import os
import dotenv
import redis

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import openai
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.templating import Jinja2Templates
from schema.llm_schemas import HealthResponse
from sentence_transformers import CrossEncoder

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Infrastructure config (read from env so Docker networking works) ──────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "global_rag_store")

# ── CORS — read allowed origins from env (comma-separated list) ───────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG pipeline...")
    try:
        app.state.client = QdrantClient(url=QDRANT_URL)
        app.state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        app.state.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=100
        )
        # Initialize once at startup — not per-request
        app.state.vector_store = QdrantVectorStore(
            client=app.state.client,
            embedding=app.state.embedding_model,
            collection_name=QDRANT_COLLECTION_NAME,
        )

        # Parse REDIS_URL into host/port for redis-py
        _redis_host = REDIS_URL.replace("redis://", "").split(":")[0]
        _redis_port = int(REDIS_URL.replace("redis://", "").split(":")[1]) if ":" in REDIS_URL.replace("redis://", "") else 6379
        app.state.redis = redis.Redis(
            host=_redis_host,
            port=_redis_port,
            db=0,
            decode_responses=True,
        )
        app.state.sessions = {}

        logger.info("Server is ready!")
    except Exception as e:
        logger.error(f"Error initializing Server: {e}")
        raise e

    logger.info("FastAPI server is ready!")
    logger.info("Swagger UI  ->  http://localhost:8000/docs")
    logger.info("Home Page   ->  http://localhost:8000")

    yield  # App handles requests here

    logger.info("Shutting down RAG pipeline...")
    app.state.client.close()
    logger.info("RAG pipeline shut down successfully")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jinja2_env = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

from routes.v1.chat_router import chat_router
from routes.v1.upload_router import upload_router

app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root(request: Request):
    return jinja2_env.TemplateResponse(request=request, name="rag_chatbot.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "pipeline_ready": True, "timestamp": datetime.now().isoformat()}


import uvicorn
if __name__ == "__main__":
    # Change to the app directory so imports like 'schema' work in uvicorn's reload process
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_config=None, log_level="info")