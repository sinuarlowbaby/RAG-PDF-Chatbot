import logging

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from qdrant_client.http import models

from config import settings
from pipeline.query_pipeline import query_pipeline
from retrieval.hybrid_document_retrieval import initialize_retrievers
from schema.llm_schemas import QueryRequest

chat_router = APIRouter(prefix="/api/v1", tags=["Chat"])
logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.qdrant_collection_name


@chat_router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Purge all Qdrant vectors for a session and remove the Redis key.

    Called by the frontend before uploading a new document so stale data
    from the previous session never leaks into new queries.
    """
    client = request.app.state.client
    redis = request.app.state.redis

    # Delete all Qdrant points that belong to this session
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            )
        ),
    )
    logger.info(f"Deleted Qdrant vectors for session {session_id}")

    # Remove the Redis session key and any semantic cache for this session
    redis.delete(f"session:{session_id}")
    
    cache_keys = redis.keys(f"semantic_cache:{session_id}:*")
    if cache_keys:
        redis.delete(*cache_keys)
        
    logger.info(f"Deleted Redis keys and cache for session {session_id}")

    return {"deleted": True, "session_id": session_id}


def _scroll_all_session_docs(client, session_id: str) -> list[Document]:
    """Paginate through ALL Qdrant records for a session, avoiding the 1000-record hard limit."""
    doc_chunks: list[Document] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for item in records:
            page_content = item.payload.get("page_content", "")
            metadata = item.payload.get("metadata", {})
            doc_chunks.append(Document(page_content=page_content, metadata=metadata))

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Scrolled {len(doc_chunks)} chunks for session {session_id}")
    return doc_chunks


@chat_router.post("/ask")
async def ask(request: Request, query: QueryRequest, x_session_id: str = Header(...)):
    if not request.app.state.redis.exists(f"session:{x_session_id}"):
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please re-upload the document.",
        )

    logger.info(f"Received question: {query.question[:50]!r}")
    embedding_model = request.app.state.embedding_model
    client = request.app.state.client
    reranker_model = request.app.state.reranker
    vector_store = request.app.state.vector_store

    doc_chunks = _scroll_all_session_docs(client, x_session_id)

    if not doc_chunks:
        raise HTTPException(
            status_code=202,
            detail=(
                "Your document is still being processed. "
                "Please wait a few seconds and try again."
            ),
        )

    hybrid_retriever = initialize_retrievers(vector_store, doc_chunks, x_session_id, k=20)

    redis_client = request.app.state.redis

    def stream_token():
        try:
            response_generator = query_pipeline(
                vector_store,
                query.question,
                hybrid_retriever,
                x_session_id,
                embedding_model,
                reranker_model,
                redis_client=redis_client,
                k=20,
                temperature=query.temperature,
            )
            for chunk in response_generator:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR]: {str(e)}\n\n"

    return StreamingResponse(
        stream_token(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
