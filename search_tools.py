"""
src/tools/search_tools.py

LangChain-compatible Tool definitions used by all CrewAI agents.

Tools exposed:
  - hybrid_search_tool       : semantic + keyword retrieval from ChromaDB
  - feature_impact_tool      : ranked feature impact from Neo4j
  - top_pain_points_tool     : raw pain-point frequency from Neo4j
  - theme_summary_tool       : theme clustering from Neo4j
  - co_occurrence_tool       : co-occurring pain points from Neo4j
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.knowledge.vector_store import VectorStore
    from src.knowledge.graph_store import GraphStore


# ── Input schemas (Pydantic v2) ───────────────────────────────────────────────

class HybridSearchInput(BaseModel):
    query: str = Field(..., description="The search query string")
    k: int = Field(10, ge=1, le=50, description="Number of results to return")
    doc_type_filter: str | None = Field(
        None,
        description="Optional filter: 'interview', 'survey', 'research', 'feedback'",
    )


class SimpleQueryInput(BaseModel):
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")


class CoOccurrenceInput(BaseModel):
    min_count: int = Field(2, ge=1, description="Minimum co-occurrence count")


# ── Tool factory ─────────────────────────────────────────────────────────────

def build_search_tools(
    vector_store: "VectorStore",
    graph_store: "GraphStore",
) -> list[StructuredTool]:
    """
    Build and return all search tools, pre-bound to the provided store instances.
    Call this once during pipeline setup and pass the resulting list to each agent.
    """

    # 1. Hybrid vector + keyword search
    def hybrid_search(query: str, k: int = 10, doc_type_filter: str | None = None) -> str:
        where = {"doc_type": doc_type_filter} if doc_type_filter else None
        docs = vector_store.hybrid_search(query=query, k=k, where=where)
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                {
                    "rank": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "doc_type": doc.metadata.get("doc_type", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "text": doc.page_content[:400],
                }
            )
        return json.dumps(results, indent=2)

    hybrid_search_tool = StructuredTool.from_function(
        func=hybrid_search,
        name="hybrid_search",
        description=(
            "Search the customer feedback knowledge base using hybrid semantic + "
            "keyword retrieval. Returns relevant text chunks grounded in real "
            "customer data. Use this to find evidence for specific pain points, "
            "feature requests, or themes. "
            "Args: query (str), k (int, default 10), "
            "doc_type_filter (str | None: 'interview'|'survey'|'research'|'feedback')"
        ),
        args_schema=HybridSearchInput,
    )

    # 2. Feature impact ranking
    def feature_impact(limit: int = 20) -> str:
        data = graph_store.get_feature_impact_summary()
        return json.dumps(data[:limit], indent=2)

    feature_impact_tool = StructuredTool.from_function(
        func=feature_impact,
        name="feature_impact_summary",
        description=(
            "Retrieve a ranked list of product features ordered by customer impact "
            "(number of distinct pain points + total mention frequency). "
            "Use this to identify which features are most critically needed."
        ),
        args_schema=SimpleQueryInput,
    )

    # 3. Top pain points
    def top_pain_points(limit: int = 20) -> str:
        data = graph_store.get_top_pain_points(limit=limit)
        return json.dumps(data, indent=2)

    top_pain_points_tool = StructuredTool.from_function(
        func=top_pain_points,
        name="top_pain_points",
        description=(
            "Retrieve the most frequently mentioned customer pain points, "
            "ordered by frequency. Use this to ground problem statements "
            "and user story justifications in quantitative evidence."
        ),
        args_schema=SimpleQueryInput,
    )

    # 4. Theme summary
    def theme_summary(limit: int = 20) -> str:
        data = graph_store.get_theme_summary()
        return json.dumps(data[:limit], indent=2)

    theme_summary_tool = StructuredTool.from_function(
        func=theme_summary,
        name="theme_summary",
        description=(
            "Get a ranked list of high-level themes (e.g. 'Speed', 'Reliability') "
            "derived from customer feedback, with counts of how many pain points "
            "belong to each theme. Use this to structure the PRD's executive summary."
        ),
        args_schema=SimpleQueryInput,
    )

    # 5. Co-occurrence
    def co_occurrence(min_count: int = 2) -> str:
        data = graph_store.get_co_occurring_pain_points(min_count=min_count)
        return json.dumps(data, indent=2)

    co_occurrence_tool = StructuredTool.from_function(
        func=co_occurrence,
        name="co_occurring_pain_points",
        description=(
            "Find pairs of pain points that are frequently mentioned together "
            "in the same customer feedback chunk. High co-occurrence indicates "
            "tightly coupled problems that may be best addressed by a single feature."
        ),
        args_schema=CoOccurrenceInput,
    )

    return [
        hybrid_search_tool,
        feature_impact_tool,
        top_pain_points_tool,
        theme_summary_tool,
        co_occurrence_tool,
    ]
