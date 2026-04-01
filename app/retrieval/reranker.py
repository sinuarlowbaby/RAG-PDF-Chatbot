from sentence_transformers import CrossEncoder


def rerank_documents(user_query,unique_docs,reranker,top_n=5):

    pairs = [(user_query, doc.page_content) for doc in unique_docs]

    scores = reranker.predict(pairs)

    scored_docs = [(doc,score) for doc ,score in zip(unique_docs,scores)]

    scored_docs.sort(key=lambda x: x[1],reverse=True)

    reranked_docs = [(doc,score) for doc,score in scored_docs[:top_n]]

    return reranked_docs
    