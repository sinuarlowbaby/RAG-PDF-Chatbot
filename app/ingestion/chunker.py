from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Python Tutorial, Release.*", "", text)
    text = re.sub(r"\d+ Chapter.*", "", text)
    text = re.sub(". . . . . . . . ", "", text)
    return text.strip()

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
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = "python_pdf"
        doc.metadata["type"] = "documentation"
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["category"] = "programming"
        doc.metadata["language"] = "python"
        doc.metadata["doc_type"] = "official_documentation"
        
    return docs