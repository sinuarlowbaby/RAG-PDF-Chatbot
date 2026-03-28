from ingestion.loader import load_documents
from ingestion.chunker import doc_chunker
from utils.time_calculate import time_calculate
from vector_store.vector_store import vector_db
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import os


def ingest_pipeline(client,embedding_model,saved_files,session_id):
    #Loading documents
    raw_documents = load_documents(saved_files)
    t1 = time_calculate()
    print("✅Documents loaded")


    #Splitting documents
    documents = doc_chunker(raw_documents,session_id)
    t2 = time_calculate()
    print(f"time taken to split documents: {t2 - t1:.2f}s")

    #Ingesting documents to vector store
    vector_store = vector_db(documents, embedding_model,client,session_id)
    t3 = time_calculate()
    print(f"time taken to ingest documents to vector store: {t3 - t2:.2f}")
    print("✅Ingestion pipeline done")
    
    return vector_store

    
