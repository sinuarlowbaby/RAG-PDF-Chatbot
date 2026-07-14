import json
import logging
import math

from langfuse.decorators import observe

from llm import llm_client, generate_queries
from retrieval.build_context import build_context
from retrieval.hybrid import retrieve_hybrid_documents
from retrieval.deduplication import deduplication
from retrieval.reranker import rerank_documents
from utils.semantic_cache import semantic_cache_match, store_semantic_cache

logger = logging.getLogger(__name__)


@observe(name="RAG_Query_Pipeline")
def query_pipeline(
    vector_store,
    user_query,
    hybrid_retriever,
    session_id,
    embedding_model,
    reranker_model,
    redis_client=None,
    k: int = 20,
    temperature: float = 0.7,
):
    new_query = generate_queries(user_query)
    all_query = ". ".join(new_query)
    user_query_embeddings = embedding_model.embed_query(all_query)

    # Semantic cache lookup — only if redis is available
    if redis_client is not None:
        cached_match = semantic_cache_match(redis_client, user_query_embeddings, session_id)
        if cached_match:
            cached_context, cached_chunks = cached_match
            logger.info("Semantic cache hit — skipping retrieval")
            yield f"[CONTEXT]: {json.dumps(cached_chunks)}"
            for chunk in llm_client(cached_context, user_query, temperature=temperature):
                yield chunk
            return

    all_docs = retrieve_hybrid_documents(hybrid_retriever, all_query)
    unique_docs = deduplication(all_docs, k=10)
    reranked_docs = rerank_documents(user_query, unique_docs, reranker=reranker_model)
    retrieved_context = build_context(reranked_docs)

    chunk_data = []
    for doc, score in reranked_docs:
        norm_score = 1 / (1 + math.exp(-score))  # sigmoid to bound [0, 1]
        chunk_data.append({
            "text": doc.page_content,
            "score": round(norm_score, 4),
            "source": doc.metadata.get("file_name", "unknown"),
        })
    yield f"[CONTEXT]: {json.dumps(chunk_data)}"

    # Generate streaming response from LLM
    for chunk in llm_client(retrieved_context, user_query, temperature=temperature):
        yield chunk

    logger.info("LLM response generated")

    # Store result in semantic cache for future identical/similar queries
    if redis_client is not None:
        saved = store_semantic_cache(
            redis_client, user_query, new_query, user_query_embeddings, retrieved_context, session_id, chunk_data
        )
        if saved:
            logger.info("Semantic cache stored (TTL=1h)")
        else:
            logger.debug("Semantic cache not stored")
