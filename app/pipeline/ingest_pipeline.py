from ingestion.loader import doc_loader
from ingestion.chunker import doc_chunker
from utils.time_calculate import time_calculate
from vector_store.vector_store import vector_db
from langchain_openai import OpenAIEmbeddings

async def ingest_pipeline(client):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=100
        )
    #Loading documents
    documents = doc_loader()
    t1 = time_calculate()

    #Splitting documents
    documents = doc_chunker(documents)
    t2 = time_calculate()
    print(f"time taken to split documents: {t2 - t1:.2f}s")

    #Ingesting documents to vector store
    vector_store = await vector_db(documents, embedding_model, client)
    t3 = time_calculate()
    print(f"time taken to ingest documents to vector store: {t3 - t2:.2f}")
    print("✅Ingestion pipeline done")
    
    return vector_store,documents

    
