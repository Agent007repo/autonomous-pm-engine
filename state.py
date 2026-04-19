"""
src/orchestration/state.py

LangGraph typed state definition.

All workflow nodes read from and write to this single shared state object.
Using TypedDict (instead of a dataclass) is the LangGraph convention.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AnalysisReport(TypedDict):
    """Output of the Data Analyst Agent."""
    top_pain_points: list[dict[str, Any]]
    feature_impact: list[dict[str, Any]]
    theme_summary: list[dict[str, Any]]
    co_occurrences: list[dict[str, Any]]
    narrative_summary: str           # Free-text analyst summary


class PRDDraft(TypedDict):
    """Mutable PRD being built across the PM and Engineering agents."""
    title: str
    version: str
    status: str                     # 'Draft' | 'In Review' | 'Approved'
    executive_summary: str
    problem_statement: str
    goals_and_metrics: str
    user_stories: str
    acceptance_criteria: str
    non_goals: str
    technical_assessment: str       # Populated by Engineering Agent
    open_questions: str
    source_evidence: str            # Cited chunks from vector search


class CritiqueRecord(TypedDict):
    round: int
    score: float                    # 0–10
    feedback: str
    passed: bool


class PipelineState(TypedDict):
    """
    Master state object passed between all LangGraph nodes.

    Lifecycle:
      ingest -> embed -> extract_entities -> analyze -> draft_prd
        -> review_prd [loop] -> output
    """
    # ── Inputs ────────────────────────────────────────────────────────────────
    input_dir: str
    product_name: str
    product_context: str            # Brief description of the product domain

    # ── Ingestion state ───────────────────────────────────────────────────────
    raw_document_count: int
    chunk_count: int
    pain_point_count: int

    # ── Analysis ──────────────────────────────────────────────────────────────
    analysis_report: AnalysisReport | None

    # ── PRD ───────────────────────────────────────────────────────────────────
    prd_draft: PRDDraft | None
    critique_history: list[CritiqueRecord]
    critique_rounds_completed: int

    # ── Output ────────────────────────────────────────────────────────────────
    final_prd_path: str | None
    final_roadmap_path: str | None
    final_matrix_path: str | None
    completed: bool

    # ── Errors (non-fatal, collected for reporting) ───────────────────────────
    errors: list[str]
