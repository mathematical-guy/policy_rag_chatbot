from __future__ import annotations

import threading
import time
from typing import Any

import boto3
import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    VectorParams,
    Distance,
)

from rag.config import Settings

# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

COLLECTION = "policies"


def get_qdrant_client(cfg: Settings) -> QdrantClient:
    return QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY)


def ensure_collection(client: QdrantClient, dim: int) -> None:
    """Create the policies collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # Payload indexes for fast filtering
        for field_name in ("domain", "source", "session_id", "expires_at"):
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field_name,
                field_schema="keyword" if field_name != "expires_at" else "integer",
            )
        logfire.info("created_qdrant_collection", collection=COLLECTION, dim=dim)


# ---------------------------------------------------------------------------
# S3 / MinIO / R2 helpers
# ---------------------------------------------------------------------------

def get_s3_client(cfg: Settings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=cfg.S3_ENDPOINT,
        aws_access_key_id=cfg.S3_KEY,
        aws_secret_access_key=cfg.S3_SECRET,
    )


def upload_raw_pdf(s3: Any, bucket: str, key: str, data: bytes) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/pdf")


# ---------------------------------------------------------------------------
# TTL sweep (background thread)
# ---------------------------------------------------------------------------

def _ttl_sweep_loop(cfg: Settings) -> None:
    """Periodically delete expired upload points and their raw PDFs."""
    client = get_qdrant_client(cfg)
    s3 = get_s3_client(cfg)
    while True:
        try:
            now = int(time.time())
            # Delete expired points from Qdrant
            client.delete(
                collection_name=COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value="user_upload")),
                        FieldCondition(key="expires_at", range=Range(lt=now)),
                    ]
                ),
            )
            logfire.debug("ttl_sweep_tick", deleted_before=now)
        except Exception as exc:
            logfire.error("ttl_sweep_error", error=str(exc))
        time.sleep(300)  # every 5 minutes


_sweep_thread: threading.Thread | None = None


def start_ttl_sweeper(cfg: Settings) -> None:
    global _sweep_thread
    if _sweep_thread is not None and _sweep_thread.is_alive():
        return
    _sweep_thread = threading.Thread(target=_ttl_sweep_loop, args=(cfg,), daemon=True)
    _sweep_thread.start()
