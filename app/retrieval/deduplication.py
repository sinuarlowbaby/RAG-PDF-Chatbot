import re
import logging

from langfuse.decorators import observe

logger = logging.getLogger(__name__)


@observe(name="Deduplicate_Documents")
def deduplication(docs, k: int = 20) -> list:
    """Remove duplicate document chunks based on normalised content + source.

    Args:
        docs: List of LangChain Document objects.
        k: Maximum number of unique documents to return.

    Returns:
        Deduplicated list of at most k documents.
    """
    unique_result = []
    seen = set()

    for doc in docs:
        text = re.sub(r"\s+", " ", doc.page_content.strip()).lower()
        key = (text, doc.metadata.get("source"))
        if key not in seen:
            unique_result.append(doc)
            seen.add(key)
        if len(unique_result) >= k:
            break

    logger.debug(f"Deduplication: {len(docs)} → {len(unique_result)} unique chunk(s)")
    return unique_result
