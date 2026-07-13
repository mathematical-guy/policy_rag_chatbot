from __future__ import annotations

from typing import Annotated, TypedDict

import logfire
from langgraph.graph import END, StateGraph

from rag.config import Settings
from rag.generate import generate_answer
from rag.rerank import rerank_candidates
from rag.retrieve import dense_search


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    domain: str            # vehicle | term | general
    session_id: str
    use_upload: bool
    candidates: list       # after retrieve
    reranked: list         # after rerank
    answer: str
    citations: list


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def route_node(state: RAGState, cfg: Settings) -> dict:
    """Route: pick the right domain filter. No user-injectable filter logic."""
    with logfire.span("route", domain=state["domain"]):
        logfire.info("route_decision", domain=state["domain"], use_upload=state["use_upload"])
    return {}


def retrieve_node(state: RAGState, cfg: Settings) -> dict:
    """Dense search against Qdrant with domain isolation filter."""
    with logfire.span("retrieve", domain=state["domain"]):
        candidates = dense_search(
            query=state["question"],
            domain=state["domain"],
            session_id=state.get("session_id"),
            include_uploads=state.get("use_upload", False),
            cfg=cfg,
        )
        logfire.info("retrieve_done", n_candidates=len(candidates))
    return {"candidates": candidates}


def rerank_node(state: RAGState, cfg: Settings) -> dict:
    """Rerank candidates via hosted rerank API (Step A)."""
    if not state["candidates"]:
        return {"reranked": [], "answer": "No relevant policy documents found.", "citations": []}

    with logfire.span("rerank", n_candidates=len(state["candidates"])):
        reranked = rerank_candidates(
            query=state["question"],
            candidates=state["candidates"],
            cfg=cfg,
        )
    return {"reranked": reranked}


def generate_node(state: RAGState, cfg: Settings) -> dict:
    """Generate a grounded answer via Groq."""
    if not state["reranked"]:
        return {"answer": "No relevant policy documents found.", "citations": []}

    with logfire.span("generate", n_chunks=len(state["reranked"])):
        answer = generate_answer(
            query=state["question"],
            context_chunks=state["reranked"],
            cfg=cfg,
        )

    citations = [
        {
            "doc_id": c.get("doc_id", ""),
            "page": c.get("page", 0),
            "section_path": c.get("section_path", ""),
        }
        for c in state["reranked"]
    ]
    return {"answer": answer, "citations": citations}


def fallback_node(state: RAGState, cfg: Settings) -> dict:
    """Handle the no-hits case."""
    return {
        "answer": "This is not covered in the provided policy documents. "
        "Please consult your full policy document or contact your insurer for details.",
        "citations": [],
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def after_retrieve(state: RAGState) -> str:
    return "rerank" if state["candidates"] else "fallback"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(cfg: Settings):
    """Build and return the compiled LangGraph."""
    graph = StateGraph(RAGState)

    # Add nodes (wrap with cfg injection)
    graph.add_node("route", lambda s: route_node(s, cfg))
    graph.add_node("retrieve", lambda s: retrieve_node(s, cfg))
    graph.add_node("rerank", lambda s: rerank_node(s, cfg))
    graph.add_node("generate", lambda s: generate_node(s, cfg))
    graph.add_node("fallback", lambda s: fallback_node(s, cfg))

    # Edges
    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"rerank": "rerank", "fallback": "fallback"})
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()
