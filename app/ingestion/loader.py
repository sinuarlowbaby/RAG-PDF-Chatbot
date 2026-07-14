import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langfuse.decorators import observe

logger = logging.getLogger(__name__)


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