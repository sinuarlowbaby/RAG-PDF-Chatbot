from retrival.retrieve_documents import retrieve_hybrid_documents
from llm.llm_client import llm_client
from qdrant_client import QdrantClient
from utils.time_calculate import time_calculate

def query_pipeline(vector_store,query,documents,client):
    t1 = time_calculate()
    #Retriving relevent documents from vector store
    retrived_context = retrieve_hybrid_documents(vector_store, query,documents)
    t2 = time_calculate()
    print(f"time taken to retrive documents: {t2 - t1:.2f}s")
    print(f"total time taken: {t2 - t1:.2f}s")

    #Generating response from llm
    response = llm_client(retrived_context, query)
    # print(f"📊📊📊 Retrived context : {retrived_context}")
    # print(f"🤖🤖🤖response: {response}")

    t3 = time_calculate()
    print(f"time taken to generate response: {t3 - t2:.2f}s")
    print(f"total time taken: {t3 - t1:.2f}s")

    client.close()
    print("✅ response generated \n")

    return response