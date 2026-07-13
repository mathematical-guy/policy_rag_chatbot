from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

import cohere
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Settings
from rag.parse import Element, parse_pdf
from rag.store import COLLECTION, ensure_collection, get_qdrant_client, get_s3_client, upload_raw_pdf

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=len,
)


def _chunk_elements(elements: list[Element]) -> list[dict]:
    """Convert parsed elements into embeddable chunks."""
    chunks: list[dict] = []
    for el in elements:
        if el.type == "table":
            # Tables are never split — whole table is one chunk
            text = el.table_md
            if el.table_flat:
                text += "\n\n" + el.table_flat
            chunks.append({
                "text": text,
                "page": el.page,
                "element_type": "table",
                "section_path": "",
            })
        else:
            # Prose: split with LangChain splitter
            splits = _text_splitter.split_text(el.text)
            for i, s in enumerate(splits):
                chunks.append({
                    "text": s,
                    "page": el.page,
                    "element_type": el.type,
                    "section_path": "",
                    "chunk_ord": i,
                })
    return chunks


# ---------------------------------------------------------------------------
# Embedding (Cohere embed API)
# ---------------------------------------------------------------------------

def _embed_batch(co: cohere.ClientV2, texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts via Cohere embed API."""
    resp = co.embed(texts=texts, model=model, input_type="search_document", embedding_types=["float"])
    return resp.embeddings.float_


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    domain: str,
    session_id: str | None = None,
    cfg: Settings | None = None,
    progress_callback=None,
) -> str:
    """
    Parse → chunk → embed → upsert to Qdrant.
    Returns the doc_id (sha256 of bytes).
    """
    if cfg is None:
        from rag.config import get_settings
        cfg = get_settings()

    doc_id = hashlib.sha256(file_bytes).hexdigest()
    source = "user_upload" if session_id else "curated"
    now = int(time.time())
    expires_at = now + cfg.UPLOAD_TTL_SECONDS if source == "user_upload" else None

    if progress_callback:
        progress_callback(0.1, "Parsing PDF...")

    # 1. Parse
    with logfire.span("ingest_parse", doc_id=doc_id):
        elements = parse_pdf(file_bytes)

    if progress_callback:
        progress_callback(0.3, "Chunking...")

    # 2. Chunk
    with logfire.span("ingest_chunk", doc_id=doc_id):
        chunks = _chunk_elements(elements)

    if progress_callback:
        progress_callback(0.5, "Embedding...")

    # 3. Embed (batched)
    co = cohere.ClientV2(cfg.EMBED_API_KEY)
    batch_size = 96
    all_vectors: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i : i + batch_size]]
        vecs = _embed_batch(co, batch_texts, cfg.EMBED_MODEL)
        all_vectors.extend(vecs)
        if progress_callback:
            pct = 0.5 + 0.3 * (min(i + batch_size, len(chunks)) / len(chunks))
            progress_callback(pct, f"Embedding chunks {min(i + batch_size, len(chunks))}/{len(chunks)}...")

    # 4. Upsert to Qdrant
    with logfire.span("ingest_upsert", doc_id=doc_id, n_chunks=len(chunks)):
        client = get_qdrant_client(cfg)
        ensure_collection(client, cfg.EMBED_DIM)

        from qdrant_client.models import PointStruct

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, all_vectors)):
            payload = {
                "domain": domain,
                "source": source,
                "doc_id": doc_id,
                "page": chunk["page"],
                "text": chunk["text"],
                "element_type": chunk.get("element_type", ""),
                "section_path": chunk.get("section_path", ""),
                "chunk_ord": chunk.get("chunk_ord", i),
            }
            if session_id:
                payload["session_id"] = session_id
            if expires_at is not None:
                payload["expires_at"] = expires_at

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            ))

        client.upsert(collection_name=COLLECTION, points=points)
        logfire.info("ingest_upsert_done", doc_id=doc_id, n_points=len(points))

    # 5. Store raw PDF in S3/MinIO
    try:
        s3 = get_s3_client(cfg)
        if source == "curated":
            key = f"curated/{domain}/{doc_id}.pdf"
        else:
            key = f"uploads/{session_id}/{doc_id}.pdf"
        upload_raw_pdf(s3, cfg.S3_BUCKET, key, file_bytes)
    except Exception as exc:
        logfire.warn("s3_upload_failed", error=str(exc))

    if progress_callback:
        progress_callback(1.0, "Done!")

    return doc_id
