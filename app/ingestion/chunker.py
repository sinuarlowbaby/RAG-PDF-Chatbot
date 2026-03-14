from langchain_text_splitters import RecursiveCharacterTextSplitter

def doc_chunker(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", " "]
    )

    docs= text_splitter.split_documents(documents)
    for doc in docs:
        doc.metadata["source"] = "python_pdf"
        doc.metadata["type"] = "documentation"
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["category"] = "programming"
        doc.metadata["language"] = "python"
        doc.metadata["doc_type"] = "official_documentation"
        
    return docs