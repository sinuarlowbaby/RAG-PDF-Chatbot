import os
import dotenv
import redis

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG pipeline...")
    try:
        client = QdrantClient(url="http://localhost:6333")
        reranker= CrossEncoder('BAAI/bge-reranker-base')
        app.state.reranker = reranker

        app.state.client = client
        app.state.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=100
        )
        app.state.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        app.state.sessions = {}

        logger.info("✅ Server is ready!")
    except Exception as e:
        logger.error(f"❌ Error initializing Server: {e}")
        raise e

    print("🚀 FastAPI server is ready!")
    print("📖 Swagger UI  →  http://localhost:8000/docs")

    yield  # App handles requests here

    logger.info("🛑 shutting down RAG pipeline...")
    app.state.client.close()
    logger.info("✅ RAG pipeline shut down successfully")


app = FastAPI(title="RAG PDF Chatbot", description="RAG PDF Chatbot", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jinja2_env = Jinja2Templates(directory="templates")

from routes.v1.chat_router import chat_router
from routes.v1.upload import upload_router

app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root(request: Request):
    return jinja2_env.TemplateResponse("rag_chatbot.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "pipeline_ready": True, "timestamp": datetime.now().isoformat()}


import uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_config=None, log_level="info")