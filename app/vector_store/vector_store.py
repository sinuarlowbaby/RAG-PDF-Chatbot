from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid

def vector_db(docs, embedding_model,client):

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    #create vector store if not exists
    if "python_docs" not in collection_names:
        client.create_collection(
            collection_name="python_docs",
            vectors_config=VectorParams(
                size = 1536,
                distance = Distance.COSINE,
                )

            )
    else:
        print("collection already exists")

    #create vector store model
    vector_store = QdrantVectorStore(
        client=client,
        embedding = embedding_model,
        collection_name = "python_docs",
    )

    #add documents to vector store
    vector_store.add_documents(
        documents = docs,
        ids = [str(uuid.uuid4()) for _ in range(len(docs))],
        batch_size = 100
        )
    print("✅ successfully ingested documents to vector store")
    return vector_store