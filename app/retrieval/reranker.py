import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Lifespan-managed state ────────────────────────────────────────────────────
# Both objects are set by app.py lifespan on startup and cleared on shutdown.
# Never create these at module level — that leaks threads on hot-reload.
_reranker_executor: ThreadPoolExecutor | None = None
_reranker_semaphore: asyncio.Semaphore | None = None


def init_reranker_pool(max_workers: int = 4, max_concurrent: int = 6) -> None:
    """
    Called from app.py lifespan startup.

    Args:
        max_workers:    Thread-pool size. Keep <= vCPU count. Each thread runs
                        CrossEncoder.predict() which uses 1 PyTorch thread
                        (torch.set_num_threads(1)), so 4 workers ≈ 4× throughput
                        on a 4-core box without contention.
        max_concurrent: Semaphore cap. Prevents more than N reranks from being
                        queued at once, avoiding OOM on bursty traffic.
    """
    global _reranker_executor, _reranker_semaphore
    _reranker_executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="reranker",
    )
    _reranker_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(
        f"Reranker thread pool started — "
        f"max_workers={max_workers}, semaphore={max_concurrent}"
    )


def shutdown_reranker_pool() -> None:
    """Called from app.py lifespan shutdown. Waits for in-flight reranks to finish."""
    global _reranker_executor
    if _reranker_executor is not None:
        _reranker_executor.shutdown(wait=True)
        _reranker_executor = None
        logger.info("Reranker thread pool shut down cleanly.")


# ── Public async interface ────────────────────────────────────────────────────

async def rerank_documents_async(
    user_query: str,
    unique_docs: list,
    reranker: CrossEncoder,
    top_n: int = 5,
    timeout_seconds: float = 10.0,
) -> list[tuple]:
    """
    Non-blocking reranker — safe to call from any async FastAPI handler.

    Features:
    - Offloads CPU work to a dedicated thread pool (event loop stays free).
    - Semaphore limits concurrent reranks to prevent memory/CPU thrashing.
    - Hard timeout triggers graceful degradation instead of blocking forever.
    - Input guard truncates absurd doc counts before they reach the thread.

    Args:
        user_query:      The user's original question.
        unique_docs:     Deduplicated list of LangChain Documents (≤ 500 enforced).
        reranker:        CrossEncoder loaded from app.state.reranker.
        top_n:           How many top-scored docs to return.
        timeout_seconds: Wall-clock seconds before giving up and falling back.

    Returns:
        List of (Document, float) tuples, sorted by descending relevance score.
        Falls back to [(doc, 0.0), ...] on timeout.
    """
    if not unique_docs:
        return []

    # Guard: pathological inputs would stall the thread pool indefinitely
    if len(unique_docs) > 500:
        logger.warning(
            f"Truncating {len(unique_docs)} docs to 500 before reranking."
        )
        unique_docs = unique_docs[:500]

    loop = asyncio.get_running_loop()

    # Semaphore: at most N reranks run concurrently across all requests
    async with _reranker_semaphore:
        try:
            reranked = await asyncio.wait_for(
                loop.run_in_executor(
                    _reranker_executor,
                    _rerank_documents_sync,
                    user_query,
                    unique_docs,
                    reranker,
                    top_n,
                ),
                timeout=timeout_seconds,
            )
            return reranked

        except asyncio.TimeoutError:
            logger.error(
                f"Reranking timed out after {timeout_seconds}s — "
                f"returning unranked top-{top_n} as fallback."
            )
            # Graceful degradation: serve something useful instead of a 500
            return [(doc, 0.0) for doc in unique_docs[:top_n]]

        except Exception:
            logger.exception("Reranking raised an unexpected error.")
            raise


# ── Synchronous worker (runs inside thread pool) ──────────────────────────────

def _rerank_documents_sync(
    user_query: str,
    unique_docs: list,
    reranker: CrossEncoder,
    top_n: int,
) -> list[tuple]:
    """
    Synchronous CrossEncoder inference. Runs in a worker thread — never call
    this directly from an async context.

    Returns:
        List of (Document, float) tuples sorted by descending score.
    """
    pairs = [(user_query, doc.page_content) for doc in unique_docs]

    with torch.no_grad():
        scores = reranker.predict(
            pairs,
            batch_size=32,          # 16 for GPU with long docs, 64 for fast CPU
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    scored_docs = sorted(
        zip(unique_docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    reranked = list(scored_docs[:top_n])
    logger.debug(f"Reranked {len(unique_docs)} docs → kept top {len(reranked)}")
    return reranked


# ── Legacy sync alias (kept for tests / non-async callers) ───────────────────

def rerank_documents(
    user_query: str,
    unique_docs: list,
    reranker: CrossEncoder,
    top_n: int = 5,
) -> list[tuple]:
    """
    Synchronous reranker — only for unit tests or non-async contexts.
    In production FastAPI handlers, always use rerank_documents_async().
    """
    return _rerank_documents_sync(user_query, unique_docs, reranker, top_n)