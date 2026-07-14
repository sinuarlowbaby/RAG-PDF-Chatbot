import re
import uuid
import logging
from pathlib import Path
from datetime import datetime
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# Minimum number of characters a page must have to be worth chunking.
# Avoids wasting vector slots on scanned / image-only PDF pages.
_MIN_PAGE_CHARS = 50


@observe(name="Load_Documents")
def load_documents(files: list[str]) -> list:
    """Load PDF files from disk into LangChain Document objects.

    Args:
        files: List of absolute file paths to PDF files.

    Returns:
        List of LangChain Document objects, one per page.
    """
    documents = []
    for file_path in files:
        if not file_path.endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {file_path}")
            continue
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        documents.extend(pages)
        logger.info(f"Loaded {len(pages)} page(s) from {os.path.basename(file_path)}")

    return documents

def clean_text(text: str) -> str:
    """Normalise whitespace and strip common PDF artefacts."""
    text = re.sub(r"\s+", " ", text)                    # collapse whitespace
    text = re.sub(r"(\. ){4,}", "", text)               # strip dot leaders
    return text.strip()


@observe(name="Doc_Chunker")
def doc_chunker(raw_documents, session_id: str) -> list:
    # ── 1. Filter out pages that contain no usable text ──────────────────────
    # PyPDFLoader returns empty page_content for scanned / image-only pages.
    usable_docs = [
        doc for doc in raw_documents
        if len(doc.page_content.strip()) >= _MIN_PAGE_CHARS
    ]

    skipped = len(raw_documents) - len(usable_docs)
    if skipped:
        logger.warning(
            f"{skipped} page(s) skipped — no extractable text "
            f"(scanned/image-only PDF?). "
            f"Consider using a PDF with selectable text."
        )

    if not usable_docs:
        logger.error(
            "No usable text found in any page. "
            "The PDF may be entirely image-based and cannot be processed."
        )
        return []

    # ── 2. Split into chunks ──────────────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=450,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(usable_docs)

    # ── 3. Clean text and attach metadata ────────────────────────────────────
    final_chunks = []
    for i, doc in enumerate(chunks):
        cleaned = clean_text(doc.page_content)
        if not cleaned:          # skip chunks that become empty after cleaning
            continue
        doc.page_content = cleaned
        doc.metadata.update({
            "chunk_number": i,
            "session_id": session_id,
            "chunk_id": str(uuid.uuid4()),
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", 0),
            "file_name": Path(doc.metadata.get("source", "unknown")).name,
            "timestamp": datetime.now().isoformat(),
        })
        final_chunks.append(doc)

    logger.info(
        f"Split {len(raw_documents)} page(s) → {len(usable_docs)} usable → "
        f"{len(final_chunks)} chunk(s)"
    )
    return final_chunks