import logging

from langfuse.decorators import observe

from ingestion.chunker import doc_chunker
from ingestion.loader import load_documents
from utils.time_calculate import time_calculate
from vector_store.vector_store import vector_db

logger = logging.getLogger(__name__)


@observe(name="RAG_Ingest_Pipeline")
def ingest_pipeline(client, embedding_model, saved_files, session_id, redis_client=None):
    # Load documents
    raw_documents = load_documents(saved_files)
    t1 = time_calculate()
    logger.info(f"Documents loaded: {len(raw_documents)} page(s)")

    # Split documents into chunks
    doc_chunks = doc_chunker(raw_documents, session_id)
    t2 = time_calculate()
    logger.info(f"Chunking complete: {len(doc_chunks)} chunk(s) in {t2 - t1:.2f}s")

    # ── Guard: if no chunks were produced, the PDF had no extractable text ────
    # Expire the Redis session immediately so the frontend gets a clear 404
    # instead of an endless 202 "still processing" loop.
    if not doc_chunks:
        logger.error(
            f"Ingestion failed for session {session_id}: "
            "0 chunks produced. The PDF may be scanned/image-based."
        )
        if redis_client is not None:
            redis_client.delete(f"session:{session_id}")
            logger.info(f"Expired Redis session {session_id} due to empty ingestion.")
        return None

    # Ingest chunks into the vector store
    vector_store = vector_db(doc_chunks, embedding_model, client, session_id)
    t3 = time_calculate()
    logger.info(f"Ingestion complete in {t3 - t1:.2f}s — session {session_id}")

    return vector_store
