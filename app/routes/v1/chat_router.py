from fastapi import APIRouter,Header
from fastapi.routing import APIRoute
from fastapi import Request
from schema.llm_schemas import QueryRequest
from pipeline.query_pipeline import query_pipeline
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from retrieval.hybrid_document_retrieval import initialize_retrievers
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
import logging

chat_router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = logging.getLogger(__name__)


@chat_router.post("/ask")
async def ask(request: Request, query:QueryRequest, x_session_id: str = Header(...)):
    if not request.app.state.redis.exists(f"session:{x_session_id}"):
        raise HTTPException(status_code=404,detail="Session not found. Please re-upload the document.")

    logger.info(f"Received question: {query.question[:50]}")
    embedding_model = request.app.state.embedding_model
    client = request.app.state.client
    reranker_model = request.app.state.reranker
    # doc_chunks = request.app.state.sessions[x_session_id]["documents"]

    vector_store = QdrantVectorStore(
        client=client,
        embedding=embedding_model,
        collection_name="global_rag_store"
    )
    scroll_result = client.scroll(
        collection_name="global_rag_store",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.session_id",
                    match=models.MatchValue(value=x_session_id),
                )
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    doc_chunks = []
    records, next_page_offset = scroll_result # scroll_result is a tuple of (records, next_page_offset)
    for item in records:
        page_content = item.payload.get("page_content", "")
        metadata = item.payload.get("metadata", {})
        doc_chunks.append(Document(page_content=page_content, metadata=metadata))

    

    hybrid_retriever = initialize_retrievers(vector_store,doc_chunks,x_session_id,k=20)
    
    def stream_token():
        try:
            response_generator = query_pipeline(
                vector_store,
                query.question,
                hybrid_retriever,
                x_session_id,
                embedding_model,
                reranker_model,
                k=20,
                temperature=query.temperature
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
