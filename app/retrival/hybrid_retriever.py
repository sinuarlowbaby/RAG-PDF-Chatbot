
def retrive_documents(vector_store, query,k=20):
    retriver = vector_store.as_retriever(
        search_kwargs={"k": k, "fetch_k": 4*k},
        search_type="mmr",
    )
    docs = retriver.invoke(query)
    
    #removes duplicate documents
    unique_result = []
    seen = set()
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            unique_result.append(doc)
            seen.add(text)
        if len(unique_result) >= k:
            break
    
    


    context = "\n\n".join([doc.page_content for doc in unique_result])
    return context