import logging

from langfuse.decorators import observe
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@observe(name="Rerank_Documents")
def rerank_documents(user_query: str, unique_docs: list, reranker: CrossEncoder, top_n: int = 5) -> list:
    """Score and rerank document chunks using a cross-encoder model.

    Args:
        user_query: The original user question.
        unique_docs: Deduplicated list of LangChain Documents.
        reranker: Initialised CrossEncoder model from app.state.
        top_n: Number of top-scoring documents to return.

    Returns:
        List of (Document, score) tuples sorted by descending score.
    """
    pairs = [(user_query, doc.page_content) for doc in unique_docs]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    reranked = list(scored_docs[:top_n])

    logger.debug(f"Reranked {len(unique_docs)} docs → top {len(reranked)} kept")
    return reranked