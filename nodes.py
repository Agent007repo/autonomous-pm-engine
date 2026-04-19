"""
src/orchestration/nodes.py

Individual LangGraph node functions.
Each node takes PipelineState and returns a partial state dict
with only the fields it modifies.

Node execution order (defined in workflow.py):
  ingest_node -> embed_node -> extract_entities_node
    -> analyze_node -> draft_prd_node -> review_prd_node
    -> (conditional: loop back or) output_node
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.data_analyst_agent import run_data_analyst
from src.agents.engineering_agent import run_engineering_agent
from src.agents.pm_agent import run_pm_agent
from src.config.settings import get_settings
from src.knowledge.document_loader import DocumentLoader
from src.knowledge.graph_store import GraphStore
from src.knowledge.semantic_chunker import SemanticChunker
from src.knowledge.vector_store import VectorStore
from src.orchestration.state import PipelineState
from src.output.prd_generator import PRDGenerator
from src.tools.search_tools import build_search_tools


# ── Dependency singletons (injected at workflow-build time) ──────────────────
# These are set by workflow.py before the graph is compiled.
_vector_store: VectorStore | None = None
_graph_store: GraphStore | None = None
_chunker: SemanticChunker | None = None


def set_dependencies(
    vector_store: VectorStore,
    graph_store: GraphStore,
    chunker: SemanticChunker,
) -> None:
    global _vector_store, _graph_store, _chunker
    _vector_store = vector_store
    _graph_store = graph_store
    _chunker = chunker


# ── Node 1: Ingest ────────────────────────────────────────────────────────────

def ingest_node(state: PipelineState) -> dict[str, Any]:
    """Load raw documents from the input directory."""
    logger.info("[NODE] ingest_node")
    loader = DocumentLoader()
    try:
        documents = loader.load_directory(state["input_dir"])
        return {
            "raw_document_count": len(documents),
            "_raw_documents": documents,  # Transient: not in TypedDict but carried in state
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"ingest_node: {e}"]}


# ── Node 2: Embed ─────────────────────────────────────────────────────────────

def embed_node(state: PipelineState) -> dict[str, Any]:
    """Semantically chunk documents and upsert into ChromaDB."""
    logger.info("[NODE] embed_node")
    documents = state.get("_raw_documents", [])
    if not documents:
        return {"errors": state.get("errors", []) + ["embed_node: no documents to embed"]}

    assert _chunker is not None, "SemanticChunker not initialised"
    assert _vector_store is not None, "VectorStore not initialised"

    chunks = _chunker.chunk_documents(documents)
    upserted = _vector_store.upsert_documents(chunks)

    return {"chunk_count": upserted, "_chunks": chunks}


# ── Node 3: Extract entities ──────────────────────────────────────────────────

def extract_entities_node(state: PipelineState) -> dict[str, Any]:
    """Run LLM-based entity extraction and populate Neo4j graph."""
    logger.info("[NODE] extract_entities_node")
    chunks = state.get("_chunks", [])
    if not chunks:
        return {"errors": state.get("errors", []) + ["extract_entities_node: no chunks"]}

    assert _graph_store is not None, "GraphStore not initialised"

    # Only extract from a subset to control LLM cost:
    # take top 50 chunks ordered by length (longer = more signal)
    subset = sorted(chunks, key=lambda d: len(d.page_content), reverse=True)[:50]
    count = _graph_store.extract_and_store(subset)

    return {"pain_point_count": count}


# ── Node 4: Analyze ───────────────────────────────────────────────────────────

def analyze_node(state: PipelineState) -> dict[str, Any]:
    """Run the Data Analyst Agent to produce an AnalysisReport."""
    logger.info("[NODE] analyze_node")

    assert _vector_store is not None
    assert _graph_store is not None

    tools = build_search_tools(_vector_store, _graph_store)

    report = run_data_analyst(
        tools=tools,
        product_name=state["product_name"],
        product_context=state["product_context"],
    )

    return {"analysis_report": report}


# ── Node 5: Draft PRD ─────────────────────────────────────────────────────────

def draft_prd_node(state: PipelineState) -> dict[str, Any]:
    """Run the PM Agent to produce a PRDDraft."""
    logger.info("[NODE] draft_prd_node")

    assert _vector_store is not None
    assert _graph_store is not None

    if not state.get("analysis_report"):
        return {
            "errors": state.get("errors", []) + ["draft_prd_node: missing analysis_report"]
        }

    tools = build_search_tools(_vector_store, _graph_store)
    cfg = get_settings()

    draft = run_pm_agent(
        tools=tools,
        analysis_report=state["analysis_report"],
        product_name=state["product_name"],
        product_context=state["product_context"],
        prd_version=cfg.prd_version,
    )

    return {"prd_draft": draft, "critique_rounds_completed": 0}


# ── Node 6: Review PRD (self-critique loop) ───────────────────────────────────

def review_prd_node(state: PipelineState) -> dict[str, Any]:
    """Run the Engineering Agent self-critique loop."""
    logger.info("[NODE] review_prd_node")
    cfg = get_settings()

    if not state.get("prd_draft"):
        return {"errors": state.get("errors", []) + ["review_prd_node: missing prd_draft"]}

    updated_prd, history = run_engineering_agent(
        prd_draft=state["prd_draft"],
        product_name=state["product_name"],
        max_rounds=cfg.max_critique_rounds,
        gate_threshold=cfg.engineering_gate_threshold,
    )

    rounds_done = state.get("critique_rounds_completed", 0) + len(history)
    all_history = state.get("critique_history", []) + history

    return {
        "prd_draft": updated_prd,
        "critique_history": all_history,
        "critique_rounds_completed": rounds_done,
    }


# ── Node 7: Output ────────────────────────────────────────────────────────────

def output_node(state: PipelineState) -> dict[str, Any]:
    """Render and save the final PRD, roadmap, and priority matrix."""
    logger.info("[NODE] output_node")

    if not state.get("prd_draft"):
        return {"errors": state.get("errors", []) + ["output_node: no PRD to render"]}

    generator = PRDGenerator()
    paths = generator.generate_all(
        prd_draft=state["prd_draft"],
        analysis_report=state["analysis_report"],
        product_name=state["product_name"],
        critique_history=state.get("critique_history", []),
    )

    return {
        "final_prd_path": paths["prd"],
        "final_roadmap_path": paths["roadmap"],
        "final_matrix_path": paths["matrix"],
        "completed": True,
    }


# ── Conditional edge: should we loop or proceed to output? ────────────────────

def should_continue_critique(state: PipelineState) -> str:
    """
    LangGraph conditional edge function.
    Returns 'output' if we should proceed, 'review_prd' to loop again.

    This is intentionally never used for looping (the Engineering Agent
    runs all critique rounds internally), but is retained here for
    architectural clarity and future extensibility.
    """
    cfg = get_settings()
    history = state.get("critique_history", [])

    if not history:
        return "review_prd"

    last = history[-1]
    if last["passed"]:
        return "output"
    if state.get("critique_rounds_completed", 0) >= cfg.max_critique_rounds:
        return "output"

    return "output"  # Engineering Agent handles its own loop; always proceed
