from ingestion.chunker import doc_chunker
from retrival.hybrid_retriever import retrive_documents
from utils.time_calculate import time_calculate
from ingestion.loader import doc_loader
from openai import OpenAI
from llm.llm_client import llm_client
from vector_store.vector_store import vector_db


def rag_pipeline(client, embedding_model,query):
    #Loading documents
    documents = doc_loader()
    t1 = time_calculate()

    #Splitting documents
    docs = doc_chunker(documents)
    t2 = time_calculate()
    print(f"time taken to split documents: {t2 - t1:.2f}s")

    #Ingesting documents to vector store
    vector_store = vector_db(docs, embedding_model,client)
    t3 = time_calculate()
    print(f"time taken to ingest documents to vector store: {t3 - t2:.2f}")

    #Retriving relevent documents from vector store
    retrived_context = retrive_documents(vector_store, query)
    t4 = time_calculate()
    print(f"time taken to retrive documents: {t4 - t3:.2f}s")
    print(f"total time taken: {t4 - t1:.2f}s")

    #Generating response from llm
    response = llm_client(retrived_context, query)
    print(f"📊📊📊 Retrived context : {retrived_context}")
    print(f"🤖🤖🤖response: {response}")

    t5 = time_calculate()
    print(f"time taken to generate response: {t5 - t4:.2f}s")
    print(f"total time taken: {t5 - t1:.2f}s")

    client.close()
    
