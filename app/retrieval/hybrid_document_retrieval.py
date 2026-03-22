from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from retrieval.reranker import rerank_documents
from retrieval.deduplication import deduplication
from retrieval.build_context import build_context
import asyncio


def initialize_retrievers(vector_store, docs, k=20):
    # Similarity search using Maximal Marginal Relevance (MMR)
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "fetch_k": 4 * k},
        search_type="mmr",
    )

    # Keyword search using BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
    return hybrid_retriever


async def retrieve_hybrid_documents(hybrid_retriever, queries, user_query, k=20):
    print("➡️ Retrieving documents...\n")

    # Fix 3: run all queries concurrently with asyncio.gather instead of a
    # sequential loop — independent queries were executing one-by-one, wasting time
    results = await asyncio.gather(
        *[hybrid_retriever.ainvoke(query) for query in queries]
    )

    # Flatten results from all queries
    all_docs = [doc for docs in results for doc in docs]

    return all_docs
