import re
import uuid
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

load_dotenv()

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\. \. \. \. \. \. \. \. ", "", text)
    return text.strip()


@traceable(run_type="tool", name="Doc_Chunker")
def doc_chunker(raw_documents, session_id: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=450,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(raw_documents)

    for i, doc in enumerate(chunks):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({
            "chunk_number": i,
            "session_id": session_id,
            "chunk_id": str(uuid.uuid4()),
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "file_name": Path(doc.metadata["source"]).name,
            "timestamp": datetime.now().isoformat(),
        })

    logger.info(f"Split {len(raw_documents)} document(s) into {len(chunks)} chunk(s)")
    return chunks