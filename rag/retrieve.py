from __future__ import annotations

import time
from typing import Any

import cohere
import logfire

from rag.config import Settings
from rag.store import COLLECTION, get_qdrant_client


def _build_query_filter(
    domain: str,
    session_id: str | None = None,
    include_uploads: bool = False,
    cfg: Settings | None = None,
) -> Any:
    """Build a Qdrant filter for domain isolation + optional session uploads."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, Should, MinShould

    must_conditions = [
        FieldCondition(key="domain", match=MatchValue(value=domain)),
        FieldCondition(key="source", match=MatchValue(value="curated")),
    ]

    if include_uploads and session_id and cfg:
        now = int(time.time())
        # Union: curated OR (user_upload + same session + not expired)
        return Filter(
            should=[
                Filter(must=must_conditions),
                Filter(must=[
                    FieldCondition(key="source", match=MatchValue(value="user_upload")),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                    FieldCondition(key="expires_at", range=Range(gt=now)),
                ]),
            ]
        )

    return Filter(must=must_conditions)


def dense_search(
    query: str,
    domain: str,
    session_id: str | None = None,
    include_uploads: bool = False,
    cfg: Settings | None = None,
) -> list[dict]:
    """
    Embed query → Qdrant dense top-K → return candidate dicts with text + metadata.
    """
    if cfg is None:
        from rag.config import get_settings
        cfg = get_settings()

    # Embed query (search_query input type)
    co = cohere.ClientV2(cfg.EMBED_API_KEY)
    resp = co.embed(
        texts=[query],
        model=cfg.EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_vec = resp.embeddings.float_[0]

    # Qdrant search
    client = get_qdrant_client(cfg)
    qdrant_filter = _build_query_filter(domain, session_id, include_uploads, cfg)

    with logfire.span("qdrant_search", domain=domain, k=cfg.DENSE_llTOP_K):
        results = client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=cfg.DENSE_TOP_K,
        )

    candidates = []
    for hit in results.points:
        payload = hit.payload or {}
        candidates.append({
            "id": hit.id,
            "score": hit.score,
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id", ""),
            "page": payload.get("page", 0),
            "section_path": payload.get("section_path", ""),
            "element_type": payload.get("element_type", ""),
            "domain": payload.get("domain", ""),
            "source": payload.get("source", ""),
        })

    logfire.info("dense_search_done", domain=domain, n_candidates=len(candidates))
    return candidates
