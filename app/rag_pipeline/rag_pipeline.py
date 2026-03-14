from ingestion.loader import doc_loader
from ingestion.chunker import doc_chunker
from utils.time_calculate import time_calculate
from llm.llm_client import llm_client
from vector_store.vector_store import vector_db
from retrival.retrieve_documents import retrieve_hybrid_documents


def rag_pipeline(client, embedding_model,query):
    #Loading documents
    documents = doc_loader()
    t1 = time_calculate()

    #Splitting documents
    documents = doc_chunker(documents)
    t2 = time_calculate()
    print(f"time taken to split documents: {t2 - t1:.2f}s")

    #Ingesting documents to vector store
    vector_store = vector_db(documents, embedding_model,client)
    t3 = time_calculate()
    print(f"time taken to ingest documents to vector store: {t3 - t2:.2f}")

    #Retriving relevent documents from vector store
    retrived_context = retrieve_hybrid_documents(vector_store, query,documents)
    t4 = time_calculate()
    print(f"time taken to retrive documents: {t4 - t3:.2f}s")
    print(f"total time taken: {t4 - t1:.2f}s")

    #Generating response from llm
    response = llm_client(retrived_context, query)
    print(f"📊📊📊 Retrived context : {retrived_context}")
    # print(f"🤖🤖🤖response: {response}")

    t5 = time_calculate()
    print(f"time taken to generate response: {t5 - t4:.2f}s")
    print(f"total time taken: {t5 - t1:.2f}s")

    client.close()
    
