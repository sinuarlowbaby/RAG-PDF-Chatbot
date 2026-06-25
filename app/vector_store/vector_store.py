import logging
import uuid

from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client.models import Distance, VectorParams

from config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.qdrant_collection_name


@traceable(run_type="tool", name="Ingest_to_Vector_DB")
def vector_db(docs, embedding_model, client, session_id: str) -> QdrantVectorStore:
    """Ensure the Qdrant collection exists and ingest document chunks.

    Args:
        docs: Chunked LangChain Document objects.
        embedding_model: Initialised embedding model.
        client: QdrantClient instance.
        session_id: Used for logging context.

    Returns:
        Configured QdrantVectorStore instance.
    """
    existing_names = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")

    vector_store = QdrantVectorStore(
        client=client,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    vector_store.add_documents(
        documents=docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
        batch_size=256,
    )
    logger.info(f"Ingested {len(docs)} chunk(s) into '{COLLECTION_NAME}' for session {session_id}")

    return vector_store