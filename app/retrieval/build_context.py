import logging
import os

import tiktoken
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("o200k_base")


@observe(name="Build_Context")
def build_context(reranked_docs, token_limit: int = 5000) -> str:
    """Concatenate reranked document chunks up to a token budget.

    Args:
        reranked_docs: List of (Document, score) tuples sorted by relevance.
        token_limit: Maximum number of tokens to include in the context.

    Returns:
        A single string containing the selected chunks.
    """
    context = ""
    token_count = 0

    for i, (doc, score) in enumerate(reranked_docs):
        chunk = (
            f"Document:{i + 1} | Source: {doc.metadata.get('file_name', 'unknown')}\n"
            f"{doc.page_content}\n\n"
        )
        token_length = len(_encoding.encode(chunk))

        if token_count + token_length > token_limit:
            break

        context += chunk
        token_count += token_length

    logger.debug(f"Built context: {token_count} tokens from {i + 1} chunk(s)")
    return context
