import hashlib
import json
import logging
import math

from langsmith import traceable

from llm import llm_client, generate_queries
from retrieval.build_context import build_context
from retrieval.hybrid import retrieve_hybrid_documents
from retrieval.deduplication import deduplication
from retrieval.reranker import rerank_documents
from utils.semantic_cache import semantic_cache_match, store_semantic_cache

logger = logging.getLogger(__name__)


@traceable(name="RAG_Query_Pipeline")
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
    use_answer_cache: bool = True,
    use_multi_query_cache: bool = True,
):
    # ── 1. Multi-Query Caching Logic ──────────────────────────────────────────
    new_query = None
    mq_cache_key = None

    if redis_client is not None and use_multi_query_cache:
        query_hash = hashlib.md5(user_query.strip().lower().encode("utf-8")).hexdigest()
        mq_cache_key = f"multi_query_cache:{session_id}:{query_hash}"
        cached_mq_raw = redis_client.get(mq_cache_key)
        if cached_mq_raw:
            try:
                new_query = json.loads(cached_mq_raw)
                logger.info(f"Multi-query cache HIT for query: {user_query!r}")
            except Exception as e:
                logger.warning(f"Failed to parse cached multi-query: {e}")

    if not new_query:
        logger.info("Generating multi-query variants from LLM")
        new_query = generate_queries(user_query)
        if redis_client is not None and use_multi_query_cache and mq_cache_key:
            try:
                redis_client.setex(mq_cache_key, 3600, json.dumps(new_query))
                logger.info("Multi-query variants cached in Redis (TTL=1h)")
            except Exception as e:
                logger.warning(f"Failed to store multi-query cache: {e}")

    all_query = ". ".join(new_query)
    user_query_embeddings = embedding_model.embed_query(all_query)

    # ── 2. Answer / Semantic Cache Lookup ─────────────────────────────────────
    if redis_client is not None and use_answer_cache:
        cached_match = semantic_cache_match(redis_client, user_query_embeddings, session_id)
        if cached_match:
            cached_context, cached_chunks = cached_match
            logger.info("Semantic answer cache HIT — skipping retrieval")
            # Log cache hit metadata (you can use get_current_run_tree() if needed in future)
            yield "[CACHE_HIT]"
            yield f"[CONTEXT]: {json.dumps(cached_chunks)}"
            for chunk in llm_client(cached_context, user_query, temperature=temperature):
                yield chunk
            return

    yield "[CACHE_MISS]"
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

    # Log cache miss metadata (you can use get_current_run_tree() if needed in future)

    yield f"[CONTEXT]: {json.dumps(chunk_data)}"

    # Generate streaming response from LLM
    for chunk in llm_client(retrieved_context, user_query, temperature=temperature):
        yield chunk

    logger.info("LLM response generated")

    # Store result in semantic answer cache (only if enabled)
    if redis_client is not None and use_answer_cache:
        saved = store_semantic_cache(
            redis_client, user_query, new_query, user_query_embeddings, retrieved_context, session_id, chunk_data
        )
        if saved:
            logger.info("Semantic answer cache stored (TTL=1h)")
        else:
            logger.debug("Semantic answer cache not stored")
