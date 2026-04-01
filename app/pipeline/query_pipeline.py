from retrieval.hybrid_document_retrieval import retrieve_hybrid_documents,initialize_retrievers,deduplication
from llm.llm_client import llm_client
from qdrant_client import QdrantClient
from utils.time_calculate import time_calculate
from llm.multi_query import generate_queries
import openai
from retrieval.reranker import rerank_documents
from retrieval.build_context import build_context
from utils.semantic_cache import semantic_cache_match,store_semantic_cache,redis_available


def query_pipeline(vector_store,user_query,hybrid_retriever,session_id,embedding_model,reranker_model,k=20,temperature=0.7):
    t1 = time_calculate()

    queries = generate_queries(user_query, n_queries=4)
    all_query = ' '.join(queries)   
    user_query_embeddings = embedding_model.embed_query(all_query)
    
    cached_match = semantic_cache_match(user_query_embeddings)

    if cached_match:
        cached_context, cached_chunks = cached_match
        print("⚡ Semantic cache hit")
        import json
        yield f"[CONTEXT]: {json.dumps(cached_chunks)}"
        
        for chunk in llm_client(cached_context, user_query, temperature=temperature):
            yield chunk
        return


    all_docs = retrieve_hybrid_documents(hybrid_retriever, queries,user_query)
    unique_docs = deduplication(all_docs, k=20)
    reranked_docs = rerank_documents(user_query,unique_docs,reranker=reranker_model)
    retrived_context = build_context(reranked_docs)

    t2 = time_calculate()
    print(f"time taken to retrive documents: {t2 - t1:.2f}s")
    print(f"total time taken: {t2 - t1:.2f}s")
    
    import math, json
    chunk_data = []
    for doc, score in reranked_docs:
        norm_score = 1 / (1 + math.exp(-score)) # sigmoid to bound [0, 1]
        chunk_data.append({
            "text": doc.page_content,
            "score": round(norm_score, 4),
            "source": doc.metadata.get("file_name", "unknown")
        })
    yield f"[CONTEXT]: {json.dumps(chunk_data)}"

    #Generating response from llm
    
    for chunk in llm_client(retrived_context, user_query, temperature=temperature):
        yield chunk
       
    
    print("✅ response generated \n")

    #storing semantic cache
    if redis_available():
        save_cache = store_semantic_cache(user_query,user_query_embeddings,retrived_context,chunk_data)

        if save_cache:
            print("⚡ Semantic cache stored for 1 hour")
            save_cache=False
        else:
            print("⚡ Semantic cache not stored")
    t3 = time_calculate()
    print(f"time taken to generate response: {t3 - t2:.2f}s")
    print(f"total time taken: {t3 - t1:.2f}s")

