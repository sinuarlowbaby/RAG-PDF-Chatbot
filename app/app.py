import os
import dotenv

from schema.llm_schemas import QueryRequest,HealthResponse
from retrieval.hybrid_document_retrieval import initialize_retrievers
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline
import openai
import logging
from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime


dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app : FastAPI):
    logger.info("Starting RAG pipeline...")
    try:
        client = QdrantClient(url="http://localhost:6333")
        vector_store,documents = ingest_pipeline(client)
        hybrid_retriever = initialize_retrievers(vector_store,documents,20)

        app_state["client"] = client
        app_state["vector_store"] = vector_store
        app_state["documents"] = documents
        app_state["hybrid_retriever"] = hybrid_retriever

        logger.info("✅ RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing RAG pipeline: {e}")
        raise e
    
    print("🚀 FastAPI server is ready!")
    print("📖 Swagger UI  →  http://localhost:8000/docs")

    yield  # App handles requests here

    logger.info("🛑 shutting down RAG pipeline...")
    app_state["client"].close()
    logger.info("✅ RAG pipeline shut down successfully")    


app =FastAPI(title="RAG PDF Chatbot",description="RAG PDF Chatbot",version="1.0.0",lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask(query:QueryRequest):
    if 'vector_store' not in app_state:
        raise HTTPException(status_code=503,detail="RAG pipeline not initialized")
    
    logger.info(f"Received question: {query.question[:50]}")

    def stream_token():
        try:
            response_generator = query_pipeline(
                app_state["vector_store"],
                query.question,
                app_state["documents"],
                app_state["client"],
            app_state["hybrid_retriever"]
        )

            for chunk in response_generator:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield f"data: [ERROR]: {str(e)}\n\n"

    return StreamingResponse(
        stream_token(),
        media_type="text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health",response_model=HealthResponse)
async def health():
    return {"status":"ok","pipeline_ready":True,"timestamp":datetime.now().isoformat()}

