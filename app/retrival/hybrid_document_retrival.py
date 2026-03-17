from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from retrival.reranker import rerank_documents
import tiktoken
from retrival.deduplication import deduplication
from retrival.build_context import build_context



def initialize_retrievers(vector_store,docs,k=20):
    # similarity search using mmr
    vector_retriver = vector_store.as_retriever(
        search_kwargs={"k": k, "fetch_k": 4*k},
        search_type="mmr",
    )

    #Keyword search using BM25
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    hybrid_retriver = EnsembleRetriever(
        retrievers=[vector_retriver, bm25_retriever],
        weights=[0.6, 0.4],
    )
    return hybrid_retriver




def retrieve_hybrid_documents(hybrid_retriver, queries,user_query,k=20):

    print("➡️retriving documents...\n")

    #hybrid search
    all_docs = []
    for query in queries:
        docs = hybrid_retriver.invoke(query)
        all_docs.extend(docs)

    #removes duplicate documents
    

    return all_docs


