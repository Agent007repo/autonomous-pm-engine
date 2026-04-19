"""
src/orchestration/workflow.py

Assembles the LangGraph StateGraph and exposes a single `run_pipeline` function.

Graph topology:
  START
    -> ingest
    -> embed
    -> extract_entities
    -> analyze
    -> draft_prd
    -> review_prd
    -> output (conditional, always proceeds — Engineering Agent loops internally)
  END
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.config.settings import get_settings
from src.knowledge.graph_store import GraphStore
from src.knowledge.semantic_chunker import SemanticChunker
from src.knowledge.vector_store import VectorStore
from src.orchestration.nodes import (
    analyze_node,
    draft_prd_node,
    embed_node,
    extract_entities_node,
    ingest_node,
    output_node,
    review_prd_node,
    set_dependencies,
    should_continue_critique,
)
from src.orchestration.state import PipelineState


def build_pipeline(
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
    chunker: SemanticChunker | None = None,
) -> Any:
    """
    Build and compile the LangGraph pipeline.

    Lazily instantiates dependencies if not provided (useful for production).
    Accepts pre-built instances for testing (dependency injection).

    Returns a compiled LangGraph runnable.
    """
    cfg = get_settings()

    # Lazy-init singletons
    vs = vector_store or VectorStore()
    gs = graph_store or GraphStore()
    sc = chunker or SemanticChunker()

    # Inject into nodes module (avoids passing them through state)
    set_dependencies(vector_store=vs, graph_store=gs, chunker=sc)

    # ── Graph definition ─────────────────────────────────────────────────────
    builder = StateGraph(PipelineState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("embed", embed_node)
    builder.add_node("extract_entities", extract_entities_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("draft_prd", draft_prd_node)
    builder.add_node("review_prd", review_prd_node)
    builder.add_node("output", output_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "embed")
    builder.add_edge("embed", "extract_entities")
    builder.add_edge("extract_entities", "analyze")
    builder.add_edge("analyze", "draft_prd")
    builder.add_edge("draft_prd", "review_prd")

    # Conditional edge from review_prd: either loop or proceed to output.
    # In the current implementation this always goes to "output"
    # (the Engineering Agent handles its own critique loop internally).
    # The conditional is retained for extensibility.
    builder.add_conditional_edges(
        "review_prd",
        should_continue_critique,
        {"review_prd": "review_prd", "output": "output"},
    )
    builder.add_edge("output", END)

    graph = builder.compile()
    logger.info("LangGraph pipeline compiled successfully.")
    return graph


def run_pipeline(
    input_dir: str,
    product_name: str,
    product_context: str,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
    chunker: SemanticChunker | None = None,
) -> PipelineState:
    """
    High-level entry point: build and run the full pipeline.

    Args:
        input_dir: Path to directory containing raw feedback documents.
        product_name: Name of the product being analysed.
        product_context: A sentence or two describing the product domain.
        vector_store: Optional pre-built VectorStore (for testing).
        graph_store: Optional pre-built GraphStore (for testing).
        chunker: Optional pre-built SemanticChunker (for testing).

    Returns:
        The final PipelineState after all nodes have executed.
    """
    cfg = get_settings()
    graph = build_pipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        chunker=chunker,
    )

    initial_state: PipelineState = {
        "input_dir": input_dir,
        "product_name": product_name,
        "product_context": product_context,
        "raw_document_count": 0,
        "chunk_count": 0,
        "pain_point_count": 0,
        "analysis_report": None,
        "prd_draft": None,
        "critique_history": [],
        "critique_rounds_completed": 0,
        "final_prd_path": None,
        "final_roadmap_path": None,
        "final_matrix_path": None,
        "completed": False,
        "errors": [],
    }

    runnable_config = RunnableConfig(recursion_limit=25)
    final_state: PipelineState = graph.invoke(initial_state, config=runnable_config)

    if final_state.get("errors"):
        logger.warning(f"Pipeline completed with {len(final_state['errors'])} error(s):")
        for err in final_state["errors"]:
            logger.warning(f"  - {err}")

    if final_state.get("completed"):
        logger.info("Pipeline complete!")
        logger.info(f"  PRD:      {final_state.get('final_prd_path')}")
        logger.info(f"  Roadmap:  {final_state.get('final_roadmap_path')}")
        logger.info(f"  Matrix:   {final_state.get('final_matrix_path')}")
    else:
        logger.error("Pipeline did not reach completion.")

    return final_state
