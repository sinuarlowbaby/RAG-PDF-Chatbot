from retrieval.hybrid_document_retrieval import retrieve_hybrid_documents,initialize_retrievers,deduplication
from llm.llm_client import llm_client
from qdrant_client import QdrantClient
from utils.time_calculate import time_calculate
from llm.multi_query import generate_queries
import openai
from retrieval.user_query_embedding import user_query_embedding
from retrieval.reranker import rerank_documents
from retrieval.build_context import build_context
from utils.semantic_cache import semantic_cache_match,store_semantic_cache,redis_available


def query_pipeline(vector_store,user_query,documents,client,hybrid_retriever):
    t1 = time_calculate()

    
    #generating multiple queries
    queries = generate_queries(user_query)

    all_query = ''
    for query in queries:
        all_query += query + ' '

    
    #Embedding user query for caching
    user_query_embeddings = user_query_embedding(all_query)
    

    #checking semantic cache
    cached_response = semantic_cache_match(user_query_embeddings)
    if cached_response:
        print("⚡ Semantic cache hit")
        for chunk in cached_response:
            yield chunk
        return

    # print("⚡ Semantic cache miss")


    # print(f"""💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪\n 
    #     {queries} \n
    #      💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪""")

    # #Retriving relevent documents from vector store


    all_docs = retrieve_hybrid_documents(hybrid_retriever, queries,user_query)

    unique_docs = deduplication(all_docs, k=20)

    
    reranked_docs = rerank_documents(user_query,unique_docs)

    print("➡️reranking documents...\n")
    
    retrived_context = build_context(reranked_docs)

    # print(f"""💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪\n 
    #     {retrived_context} \n
    #      💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪""")


    t2 = time_calculate()
    print(f"time taken to retrive documents: {t2 - t1:.2f}s")
    print(f"total time taken: {t2 - t1:.2f}s")

    #Generating response from llm
    response_generator= llm_client(retrived_context, user_query)
    print("➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️\n ")
    
    full_response = ''
    for chunk in response_generator:
        full_response += chunk
        yield chunk
       
    
    print("✅ response generated \n")

    #storing semantic cache
    if redis_available():
        save_cache = store_semantic_cache(user_query,user_query_embeddings,retrived_context,full_response)

        if save_cache:
            print("⚡ Semantic cache stored for 1 hour")
            save_cache=False
        else:
            print("⚡ Semantic cache not stored")
    t3 = time_calculate()
    print(f"time taken to generate response: {t3 - t2:.2f}s")
    print(f"total time taken: {t3 - t1:.2f}s")

