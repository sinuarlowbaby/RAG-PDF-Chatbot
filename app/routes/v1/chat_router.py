from fastapi import APIRouter
from app import app_state
from schema.llm_schemas import QueryRequest
from pipeline.query_pipeline import query_pipeline
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import logging

chat_router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = logging.getLogger(__name__)


@chat_router.post("/ask")
async def ask(query:QueryRequest):
    if 'vector_store' not in app_state:
        raise HTTPException(status_code=503,detail="Please upload a document first")
    
    logger.info(f"Received question: {query.question[:50]}")

    def stream_token():
        try:
            response_generator = query_pipeline(
                app_state["vector_store"],
                query.question,
                app_state["hybrid_retriever"],
                app_state["session_id"],
                app_state["embedding_model"],
                k=20,
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
