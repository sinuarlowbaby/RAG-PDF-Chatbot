import asyncio
from retrieval.hybrid_document_retrieval import retrieve_hybrid_documents, initialize_retrievers, deduplication
from llm.llm_client import llm_client
from utils.time_calculate import time_calculate
from llm.multi_query import generate_queries
from retrieval.user_query_embedding import user_query_embedding
from retrieval.reranker import rerank_documents
from retrieval.build_context import build_context
from utils.semantic_cache import semantic_cache_match, store_semantic_cache, redis_available
import asyncio


async def query_pipeline(vector_store, user_query, documents, client, hybrid_retriever):
    t1 = time_calculate()

    # Embed the original user query once — used as the cache key for
    # semantic_cache_match / store_semantic_cache. The expanded `queries`
    # list is used only for hybrid retrieval, not for cache lookup.
    queries, user_query_embedding_vec = await asyncio.gather(
        generate_queries(user_query),
        user_query_embedding(user_query),
    )
    
    # Checking semantic cache
    cached_response = await semantic_cache_match(user_query_embedding_vec)
    if cached_response:
        print("⚡ Semantic cache hit")
        for chunk in cached_response:
            yield chunk
        return

    all_docs = await retrieve_hybrid_documents(hybrid_retriever, queries, user_query)

    unique_docs = deduplication(all_docs, k=20)

    reranked_docs = await rerank_documents(user_query, unique_docs)

    print("➡️ Reranking documents...\n")

    retrieved_context = build_context(reranked_docs)

    t2 = time_calculate()
    print(f"Time to retrieve documents: {t2 - t1:.2f}s")

    full_response = ''

    async for chunk in llm_client(retrieved_context, user_query):
        full_response += chunk
        yield chunk

    print("✅ Response generated\n")

    if await redis_available():
        saved = await store_semantic_cache(
            user_query, user_query_embedding_vec, retrieved_context, full_response
        )
        if saved:
            print("⚡ Semantic cache stored for 1 hour")
        else:
            print("⚡ Semantic cache not stored")

    t3 = time_calculate()
    print(f"Time to generate response: {t3 - t2:.2f}s")
    print(f"Total time: {t3 - t1:.2f}s")
