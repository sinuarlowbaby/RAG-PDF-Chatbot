import tiktoken

encoding = tiktoken.get_encoding("o200k_base")

def build_context(unique_docs,token_limit=5000):
    context = ""
    token_count = 0
    for i, (doc, score) in enumerate(unique_docs):
        chunk = f"Document:{i+1} | Source: {doc.metadata.get('file_name','unknown')}\n{doc.page_content}\n\n"
        token_length = len(encoding.encode(chunk))
        
        if token_count + token_length > token_limit:
            break
        context += chunk
        token_count += token_length
    return context
