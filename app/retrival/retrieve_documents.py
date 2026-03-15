from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import re
from retrival.reranker import rerank_documents
import tiktoken


encoding = tiktoken.get_encoding("o200k_base")

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


def deduplication(docs , k=20):
    unique_result = []
    seen = set()
    for doc in docs:
        text = doc.page_content.strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        key = (text, doc.metadata.get("source"))
        if key not in seen:

            unique_result.append(doc)
            seen.add(key)
        if len(unique_result) >= k:
            break
    return unique_result



def build_context(unique_docs,token_limit=5000):
    context = ""
    token_count = 0
    for i,doc in enumerate(unique_docs):
        chunk = f"Document:{i+1} | Source: {doc.metadata.get('source','unknown')}\n{doc.page_content}\n\n"
        token_length = len(encoding.encode(chunk))
        
        if token_count + token_length > token_limit:
            break
        context += chunk
        token_count += token_length
    return context



def retrieve_hybrid_documents(hybrid_retriver, queries,user_query,k=20):

    print("➡️retriving documents...\n")

    #hybrid search
    all_docs = []
    for query in queries:
        docs = hybrid_retriver.invoke(query)
        all_docs.extend(docs)

    #removes duplicate documents
    unique_docs = deduplication(all_docs, k)
    
    for doc in unique_docs:
        print(doc.page_content+"\n") 


    reranked_docs = rerank_documents(user_query,unique_docs)
    
    print("➡️reranking documents...\n")
    
    context = build_context(reranked_docs)

    return context