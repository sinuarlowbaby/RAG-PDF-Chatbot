from openai._client import OpenAIWithRawResponse
from openai.types.audio import transcription_text_done_event
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
import re
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r". . . . . . . . ", "", text)
    return text.strip()


from langsmith import traceable

@traceable(run_type="tool", name="Doc_Chunker")
def doc_chunker(raw_documents, session_id):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # text_splitter = SemanticChunker(
    #     embeddings,
    #     breakpoint_threshold_type="percentile",
    #     breakpoint_threshold_amount=85,
    # )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
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