"""
tests/test_workflow.py

Integration tests for the LangGraph workflow.
All external dependencies (LLM, ChromaDB, Neo4j) are mocked.
Tests verify that the state machine transitions correctly and
that each node populates the correct state fields.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.state import AnalysisReport, CritiqueRecord, PRDDraft, PipelineState
from src.orchestration.workflow import build_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ANALYSIS: AnalysisReport = {
    "top_pain_points": [
        {"id": "pp_search", "text": "Search is slow", "frequency": 15, "doc_type": "interview"},
    ],
    "feature_impact": [
        {"feature_name": "Search", "category": "Core Feature", "pain_point_count": 8, "total_frequency": 42},
    ],
    "theme_summary": [
        {"theme": "Performance", "pain_point_count": 12, "total_mentions": 58},
    ],
    "co_occurrences": [],
    "narrative_summary": "Search performance is the dominant pain point across all feedback sources.",
}

SAMPLE_PRD: PRDDraft = {
    "title": "PRD: Search Performance",
    "version": "1.0",
    "status": "Draft",
    "executive_summary": "Customers consistently cite slow search as a top pain point.",
    "problem_statement": "42 mentions across 15 customers identify search latency as blocking.",
    "goals_and_metrics": "Objective: Reduce search p95 latency to < 200ms.",
    "user_stories": "As a user, I want search results in under 200ms so that my workflow is not disrupted.",
    "acceptance_criteria": "Given a search query, When submitted, Then results appear within 200ms.",
    "non_goals": "Real-time collaborative search is out of scope for this release.",
    "technical_assessment": "",
    "open_questions": "Should we cache results? What is acceptable staleness?",
    "source_evidence": "Interview #001: 'search is too slow'; Survey R006: 'search is broken'.",
}

SAMPLE_CRITIQUE: CritiqueRecord = {
    "round": 1,
    "score": 8.0,
    "feedback": "Good clarity, NFRs present, AC is testable.",
    "passed": True,
}


@pytest.fixture
def mock_settings():
    with patch("src.orchestration.nodes.get_settings") as m:
        m.return_value = MagicMock(
            max_critique_rounds=1,
            engineering_gate_threshold=7.0,
            prd_version="1.0",
            output_dir="/tmp/pm_engine_test_output/",
        )
        yield m


@pytest.fixture
def mock_dependencies():
    """Return mock VectorStore, GraphStore, and SemanticChunker."""
    vs = MagicMock()
    vs.upsert_documents.return_value = 10
    vs.hybrid_search.return_value = []
    vs.collection_count.return_value = 10

    gs = MagicMock()
    gs.extract_and_store.return_value = 5
    gs.get_top_pain_points.return_value = SAMPLE_ANALYSIS["top_pain_points"]
    gs.get_feature_impact_summary.return_value = SAMPLE_ANALYSIS["feature_impact"]
    gs.get_theme_summary.return_value = SAMPLE_ANALYSIS["theme_summary"]
    gs.get_co_occurring_pain_points.return_value = []

    sc = MagicMock()
    sc.chunk_documents.return_value = [
        MagicMock(page_content="chunk text", metadata={"source": "test.txt", "doc_type": "interview"})
        for _ in range(10)
    ]

    return vs, gs, sc


@pytest.fixture
def sample_input_dir(tmp_path: Path) -> str:
    """Create a temporary input directory with a sample file."""
    (tmp_path / "test_feedback.txt").write_text(
        "Customers find the search feature too slow. "
        "They report that search results take over 5 seconds to load."
    )
    return str(tmp_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPipelineStateTransitions:

    @patch("src.orchestration.nodes.run_data_analyst")
    @patch("src.orchestration.nodes.run_pm_agent")
    @patch("src.orchestration.nodes.run_engineering_agent")
    @patch("src.orchestration.nodes.build_search_tools")
    @patch("src.output.prd_generator.PRDGenerator.generate_all")
    def test_full_pipeline_reaches_completion(
        self,
        mock_generate,
        mock_search_tools,
        mock_engineering,
        mock_pm,
        mock_analyst,
        mock_settings,
        mock_dependencies,
        sample_input_dir,
    ):
        vs, gs, sc = mock_dependencies

        # Set up mocks
        mock_analyst.return_value = SAMPLE_ANALYSIS
        mock_pm.return_value = SAMPLE_PRD
        mock_engineering.return_value = (
            {**SAMPLE_PRD, "technical_assessment": "Feasible. Medium complexity."},
            [SAMPLE_CRITIQUE],
        )
        mock_generate.return_value = {
            "prd": "/tmp/prd.md",
            "roadmap": "/tmp/roadmap.md",
            "matrix": "/tmp/matrix.md",
        }
        mock_search_tools.return_value = []

        pipeline = build_pipeline(vector_store=vs, graph_store=gs, chunker=sc)

        initial_state: PipelineState = {
            "input_dir": sample_input_dir,
            "product_name": "TestProduct",
            "product_context": "A test product.",
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

        final_state = pipeline.invoke(initial_state)

        assert final_state["completed"] is True
        assert final_state["final_prd_path"] == "/tmp/prd.md"
        assert final_state["final_roadmap_path"] == "/tmp/roadmap.md"
        assert final_state["analysis_report"] is not None
        assert final_state["prd_draft"] is not None

    def test_pipeline_tolerates_missing_input_dir(
        self, mock_settings, mock_dependencies
    ):
        vs, gs, sc = mock_dependencies
        pipeline = build_pipeline(vector_store=vs, graph_store=gs, chunker=sc)

        initial_state: PipelineState = {
            "input_dir": "/nonexistent/path/",
            "product_name": "TestProduct",
            "product_context": "A test product.",
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

        # Should not raise; errors should be captured in state
        final_state = pipeline.invoke(initial_state)
        assert len(final_state.get("errors", [])) > 0


class TestConditionalEdge:

    def test_should_continue_returns_output_when_passed(self):
        from src.orchestration.nodes import should_continue_critique

        state: PipelineState = {
            "critique_history": [SAMPLE_CRITIQUE],
            "critique_rounds_completed": 1,
            "input_dir": "",
            "product_name": "",
            "product_context": "",
            "raw_document_count": 0,
            "chunk_count": 0,
            "pain_point_count": 0,
            "analysis_report": None,
            "prd_draft": None,
            "final_prd_path": None,
            "final_roadmap_path": None,
            "final_matrix_path": None,
            "completed": False,
            "errors": [],
        }

        result = should_continue_critique(state)
        assert result == "output"

    def test_should_continue_returns_review_when_no_history(self):
        from src.orchestration.nodes import should_continue_critique

        state: PipelineState = {
            "critique_history": [],
            "critique_rounds_completed": 0,
            "input_dir": "",
            "product_name": "",
            "product_context": "",
            "raw_document_count": 0,
            "chunk_count": 0,
            "pain_point_count": 0,
            "analysis_report": None,
            "prd_draft": None,
            "final_prd_path": None,
            "final_roadmap_path": None,
            "final_matrix_path": None,
            "completed": False,
            "errors": [],
        }

        result = should_continue_critique(state)
        assert result == "review_prd"
