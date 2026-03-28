from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid

COLLECTION_NAME = "global_rag_store"

def vector_db(docs, embedding_model,client,session_id):

    collections = client.get_collections().collections
    existing_collection = [c.name for c in collections]

    #create vector store if not exists
    if COLLECTION_NAME not in existing_collection:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size = 1536,
                distance = Distance.COSINE,
                )

            )

        vector_store = QdrantVectorStore(
        client=client,
        embedding = embedding_model,
        collection_name = COLLECTION_NAME,
    )

        #add documents to vector store
        vector_store.add_documents(
            documents = docs,
            ids = [str(uuid.uuid4()) for _ in range(len(docs))],
            batch_size = 100
            )
        print(f"✅ successfully ingested documents to vector store {COLLECTION_NAME} for session {session_id}")

    else:
        print(f"collection {COLLECTION_NAME} already exists")
        vector_store = QdrantVectorStore(
            client=client,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
        )

    #create vector store model
    
    return vector_store