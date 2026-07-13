from __future__ import annotations

import cohere
import logfire

from rag.config import Settings


def rerank_candidates(
    query: str,
    candidates: list[dict],
    cfg: Settings | None = None,
) -> list[dict]:
    """
    Step A: Hosted rerank via Cohere Rerank API.
    Returns the top-N candidates with added 'rerank_score'.
    """
    if cfg is None:
        from rag.config import get_settings
        cfg = get_settings()

    if not candidates:
        return []

    co = cohere.ClientV2(cfg.RERANK_API_KEY)

    documents = [c["text"] for c in candidates]

    with logfire.span("rerank", n_candidates=len(candidates), top_n=cfg.RERANK_TOP_N):
        resp = co.rerank(
            query=query,
            documents=documents,
            top_n=cfg.RERANK_TOP_N,
            model=cfg.RERANK_MODEL,
        )

    reranked: list[dict] = []
    for result in resp.results:
        orig = candidates[result.index]
        reranked.append({**orig, "rerank_score": result.relevance_score})

    logfire.info("rerank_done", n_reranked=len(reranked))
    return reranked
