from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import streamlit as st


@dataclass(frozen=True)
class Settings:
    # --- Domains ---
    DOMAINS: list[str] = field(default_factory=lambda: ["vehicle", "term", "general"])

    # --- Qdrant ---
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    # --- Embeddings (hosted) ---
    EMBED_API_KEY: str = ""
    EMBED_MODEL: str = "embed-english-v3.0"
    EMBED_DIM: int = 1024

    # --- Rerank (hosted) ---
    RERANK_API_KEY: str = ""
    RERANK_MODEL: str = "rerank-english-v3.0"

    # --- LLM (Groq) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Object storage (MinIO / R2) ---
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_KEY: str = ""
    S3_SECRET: str = ""
    S3_BUCKET: str = "policies"

    # --- Observability ---
    LOGFIRE_TOKEN: str = ""

    # --- Retrieval tuning ---
    DENSE_TOP_K: int = 30
    RERANK_TOP_N: int = 6
    UPLOAD_TTL_SECONDS: int = 7200  # 2 hours


def _resolve(key: str, default: str = "") -> str:
    """Resolve a config value: Streamlit secrets -> env var -> default."""
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except (AttributeError, FileNotFoundError):
        pass
    return os.getenv(key, default)


def get_settings() -> Settings:
    """Build the Settings object, pulling from secrets / env at call time."""
    return Settings(
        QDRANT_URL=_resolve("QDRANT_URL"),
        QDRANT_API_KEY=_resolve("QDRANT_API_KEY"),
        EMBED_API_KEY=_resolve("EMBED_API_KEY"),
        EMBED_MODEL=_resolve("EMBED_MODEL", "embed-english-v3.0"),
        EMBED_DIM=int(_resolve("EMBED_DIM", "1024")),
        RERANK_API_KEY=_resolve("RERANK_API_KEY"),
        RERANK_MODEL=_resolve("RERANK_MODEL", "rerank-english-v3.0"),
        GROQ_API_KEY=_resolve("GROQ_API_KEY"),
        GROQ_MODEL=_resolve("GROQ_MODEL", "llama-3.3-70b-versatile"),
        S3_ENDPOINT=_resolve("S3_ENDPOINT", "http://localhost:9000"),
        S3_KEY=_resolve("S3_KEY"),
        S3_SECRET=_resolve("S3_SECRET"),
        S3_BUCKET=_resolve("S3_BUCKET", "policies"),
        LOGFIRE_TOKEN=_resolve("LOGFIRE_TOKEN"),
        DENSE_TOP_K=int(_resolve("DENSE_TOP_K", "30")),
        RERANK_TOP_N=int(_resolve("RERANK_TOP_N", "6")),
        UPLOAD_TTL_SECONDS=int(_resolve("UPLOAD_TTL_SECONDS", "7200")),
    )
