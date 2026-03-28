import os
import dotenv

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

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG pipeline...")
    try:
        client = QdrantClient(url="http://localhost:6333")

        app_state["client"] = client
        app_state["embedding_model"] = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=100
        )

        logger.info("✅ Server is ready!")
    except Exception as e:
        logger.error(f"❌ Error initializing Server: {e}")
        raise e

    print("🚀 FastAPI server is ready!")
    print("📖 Swagger UI  →  http://localhost:8000/docs")

    yield  # App handles requests here

    logger.info("🛑 shutting down RAG pipeline...")
    app_state["client"].close()
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
