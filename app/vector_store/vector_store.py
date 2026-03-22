from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
import uuid


async def vector_db(docs, embedding_model, client):
    try:
        response = client.get_collections()
        collection_names = [c.name for c in response.collections]

    #create vector store if not exists
        if "python_docs" not in collection_names:
            client.create_collection(
                collection_name="python_docs",
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                )
            )

            vector_store = QdrantVectorStore(
                client=client,
                embedding=embedding_model,
                collection_name="python_docs",
            )

            # Add documents asynchronously
            await vector_store.aadd_documents(
                documents=docs,
                ids=[str(uuid.uuid4()) for _ in range(len(docs))],
                batch_size=100,
            )
            print("✅ Successfully ingested documents to vector store")

        else:
            print("Collection already exists")
            vector_store = QdrantVectorStore(
                client=client,
                embedding=embedding_model,
                collection_name="python_docs",
            )

        return vector_store 

    except Exception as e:
        print(f"❌ vector_db error: {e}")
        raise