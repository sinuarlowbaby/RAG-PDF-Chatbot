from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import re

def hybrid_retrievers(vector_store,docs,k=20):
    print("retriving documents...")
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
        weights=[0.7, 0.3],
    )

    return hybrid_retriver

def retrieve_hybrid_documents(vector_store, query,documents,k=20):

    print("➡️retriving documents...\n")
    hybrid_retriver = hybrid_retrievers(vector_store,documents,20)
    #hybrid search
    docs = hybrid_retriver.invoke(query)
    # print(f"retrived {len(docs)} documents")

    #removes duplicate documents
    unique_result = []
    seen = set()
    for doc in docs:
        text = doc.page_content.strip()
        text = re.sub(r"\s+", " ", text).strip()
        if text not in seen:
            unique_result.append(doc)
            seen.add(text)
        if len(unique_result) >= k:
            break

    context = ""
    token_limit = 3000
    for doc in unique_result:
        if len(context) + len(doc.page_content) > token_limit:
            break
        context += doc.page_content + "\n\n"
    
    return context