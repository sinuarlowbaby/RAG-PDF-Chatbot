import logging
import os

import tiktoken

logger = logging.getLogger(__name__)

_encoding = None
try:
    _encoding = tiktoken.get_encoding("cl100k_base")
except BaseException:
    _encoding = None


def _count_tokens(text: str) -> int:
    """Count tokens safely using tiktoken or fallback to 4 chars/token heuristic."""
    if _encoding is not None:
        try:
            return len(_encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


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
        token_length = _count_tokens(chunk)

        if token_count + token_length > token_limit:
            break

        context += chunk
        token_count += token_length

    logger.debug(f"Built context: {token_count} tokens from {len(reranked_docs)} chunk(s)")
    return context

