from retrival.retrieve_documents import retrieve_hybrid_documents,initialize_retrievers,deduplication
from llm.llm_client import llm_client
from qdrant_client import QdrantClient
from utils.time_calculate import time_calculate
from llm.multi_query import generate_queries

def query_pipeline(vector_store,user_query,documents,client):
    t1 = time_calculate()

    #generating multiple queries
    queries = generate_queries(user_query)
    print(f"""💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪\n 
        {queries} \n
         💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪""")

    #Retriving relevent documents from vector store
    hybrid_retriver = initialize_retrievers(vector_store,documents,20)

    

    retrived_context = retrieve_hybrid_documents(hybrid_retriver, queries,user_query)

    print(f"""💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪\n 
        {retrived_context} \n
         💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪""")


    t2 = time_calculate()
    print(f"time taken to retrive documents: {t2 - t1:.2f}s")
    print(f"total time taken: {t2 - t1:.2f}s")

    #Generating response from llm
    response_generator = llm_client(retrived_context, user_query)

    t3 = time_calculate()
    print(f"time taken to generate response: {t3 - t2:.2f}s")
    print(f"total time taken: {t3 - t1:.2f}s")

    print("✅ response generated \n")

    return response_generator