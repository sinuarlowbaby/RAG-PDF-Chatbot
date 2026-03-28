from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r". . . . . . . . ", "", text)
    return text.strip()

def doc_chunker(raw_documents,session_id):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(raw_documents)
    for i,doc in enumerate(chunks):
        doc.page_content = clean_text(doc.page_content)

        doc.metadata.update({
            "chunk_number":i,
            "session_id":session_id,
            "chunk_id": str(uuid.uuid4()),
            "source":doc.metadata["source"],
            "page":doc.metadata["page"],
            "file_name":Path(doc.metadata["source"]).name,
            "timestamp":datetime.now().isoformat(),
            })
    return chunks