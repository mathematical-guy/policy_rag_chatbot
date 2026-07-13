from __future__ import annotations

import streamlit as st

from rag.config import get_settings
from rag.graph import RAGState, build_graph
from rag.ingest import ingest_pdf
from rag.obs import setup_observability
from rag.store import ensure_collection, get_qdrant_client, start_ttl_sweeper

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Policy RAG Chatbot", page_icon="📜", layout="wide")
st.title("Policy RAG Chatbot")

# ---------------------------------------------------------------------------
# Init (runs once)
# ---------------------------------------------------------------------------

cfg = get_settings()
setup_observability(cfg)

if "graph" not in st.session_state:cfg = get_settings()
setup_observability(cfg)

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(cfg)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph = build_graph(cfg)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"

if "has_uploads" not in st.session_state:
    st.session_state.has_uploads = False

# Start TTL sweeper
try:
    start_ttl_sweeper(cfg)
except Exception:
    pass  # S3 may not be configured in dev

# ---------------------------------------------------------------------------
# Sidebar — domain selector + file upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    domain = st.selectbox("Policy Domain", cfg.DOMAINS, index=0)
    st.session_state.domain = domain

    st.divider()
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload a policy PDF", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            def _progress(pct, msg):
                st.progress(pct, text=msg)

            doc_id = ingest_pdf(
                file_bytes=uploaded_file.read(),
                domain=domain,
                session_id=st.session_state.session_id,
                cfg=cfg,
                progress_callback=_progress,
            )
            st.session_state.has_uploads = True
            st.success(f"Processed! Doc ID: {doc_id[:12]}...")

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    st.text(f"doc: {c['doc_id'][:12]}…  page: {c['page']}  section: {c['section_path']}")

# Chat input
if query := st.chat_input("Ask a question about your policy..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            graph = st.session_state.graph
            initial_state: RAGState = {
                "question": query,
                "domain": domain,
                "session_id": st.session_state.session_id,
                "use_upload": st.session_state.has_uploads,
                "candidates": [],
                "reranked": [],
                "answer": "",
                "citations": [],
            }
            result = graph.invoke(initial_state)

            st.markdown(result["answer"])

            if result.get("citations"):
                with st.expander("Citations"):
                    for c in result["citations"]:
                        st.text(
                            f"doc: {c['doc_id'][:12]}…  page: {c['page']}  section: {c['section_path']}"
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result.get("citations", []),
    })
