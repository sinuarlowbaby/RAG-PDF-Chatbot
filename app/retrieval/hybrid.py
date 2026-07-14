import logging

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langfuse.decorators import observe
from qdrant_client.http import models

from retrieval.deduplication import deduplication
from retrieval.reranker import rerank_documents

logger = logging.getLogger(__name__)


def initialize_retrievers(vector_store, docs, session_id: str, k: int = 20):
    """Build a hybrid retriever combining MMR vector search and BM25 keyword search.

    Args:
        vector_store: Initialised QdrantVectorStore instance.
        docs: All document chunks for this session (used for BM25 corpus).
        session_id: Session identifier used to filter Qdrant results.
        k: Number of results to retrieve from each retriever.

    Returns:
        EnsembleRetriever combining vector MMR and BM25.
    """
    # Semantic / vector search using MMR
    base_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k,
            "fetch_k": 2 * k,
            "filter": models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
        },
        search_type="mmr",
    )

    # Keyword search using BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[base_retriever, bm25_retriever],
        weights=[0.6, 0.4],
        verbose=False,
    )
    return hybrid_retriever


@observe(name="Execute_Hybrid_Retrieval")
def retrieve_hybrid_documents(hybrid_retriever, new_user_query: str, k: int = 10):
    """Invoke the hybrid retriever and return retrieved documents."""

    logger.debug("Retrieving documents via hybrid search...")
    docs = hybrid_retriever.invoke(new_user_query)

    logger.info(f"Hybrid retrieval returned {len(docs)} document(s)")
    return docs
