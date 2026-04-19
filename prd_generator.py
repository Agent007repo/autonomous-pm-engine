"""
src/output/prd_generator.py

Renders the final pipeline outputs as Markdown files:
  1. prd_<timestamp>.md        — Full structured PRD
  2. roadmap_<timestamp>.md    — Quarterly engineering roadmap
  3. priority_matrix_<timestamp>.md — Feature priority (RICE scoring)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.orchestration.state import AnalysisReport, CritiqueRecord, PRDDraft
from src.output.templates import (
    PRIORITY_MATRIX_TEMPLATE,
    PRD_TEMPLATE,
    ROADMAP_TEMPLATE,
)


class PRDGenerator:
    """Assembles and writes all pipeline output documents."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._output_dir = Path(self._cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        prd_draft: PRDDraft,
        analysis_report: AnalysisReport | None,
        product_name: str,
        critique_history: list[CritiqueRecord],
    ) -> dict[str, str]:
        """
        Generate all three output documents.
        Returns a dict of {doc_type: file_path}.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = product_name.lower().replace(" ", "_")[:30]

        prd_path = self._write_prd(prd_draft, analysis_report, critique_history, ts, slug)
        roadmap_path = self._write_roadmap(prd_draft, analysis_report, ts, slug)
        matrix_path = self._write_priority_matrix(analysis_report, ts, slug)

        return {"prd": str(prd_path), "roadmap": str(roadmap_path), "matrix": str(matrix_path)}

    # ── PRD ───────────────────────────────────────────────────────────────────

    def _write_prd(
        self,
        draft: PRDDraft,
        report: AnalysisReport | None,
        history: list[CritiqueRecord],
        ts: str,
        slug: str,
    ) -> Path:
        generated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        final_score = history[-1]["score"] if history else "N/A"

        critique_log = ""
        for record in history:
            status = "PASSED" if record["passed"] else "NEEDS WORK"
            critique_log += (
                f"\n#### Round {record['round']} — Score: {record['score']:.1f}/10 [{status}]\n"
                f"{record['feedback']}\n"
            )

        content = PRD_TEMPLATE.format(
            title=draft.get("title", f"PRD: {slug}"),
            version=draft.get("version", "1.0"),
            status=draft.get("status", "Draft"),
            generated_date=generated_date,
            final_score=final_score,
            executive_summary=draft.get("executive_summary", "_Not provided._"),
            problem_statement=draft.get("problem_statement", "_Not provided._"),
            goals_and_metrics=draft.get("goals_and_metrics", "_Not provided._"),
            user_stories=draft.get("user_stories", "_Not provided._"),
            acceptance_criteria=draft.get("acceptance_criteria", "_Not provided._"),
            non_goals=draft.get("non_goals", "_Not provided._"),
            technical_assessment=draft.get("technical_assessment", "_Not provided._"),
            open_questions=draft.get("open_questions", "_Not provided._"),
            source_evidence=draft.get("source_evidence", "_Not provided._"),
            critique_log=critique_log or "_No critique history recorded._",
        )

        path = self._output_dir / f"prd_{slug}_{ts}.md"
        path.write_text(content, encoding="utf-8")
        logger.info(f"PRD written: {path}")
        return path

    # ── Roadmap ───────────────────────────────────────────────────────────────

    def _write_roadmap(
        self,
        draft: PRDDraft,
        report: AnalysisReport | None,
        ts: str,
        slug: str,
    ) -> Path:
        # Build feature list from analysis report
        features_table = ""
        if report and report.get("feature_impact"):
            rows = []
            for i, feat in enumerate(report["feature_impact"][:12], 1):
                rows.append(
                    f"| {i} | {feat.get('feature_name', 'Unknown')} "
                    f"| {feat.get('category', '-')} "
                    f"| {feat.get('total_frequency', 0)} "
                    f"| {feat.get('pain_point_count', 0)} |"
                )
            features_table = "\n".join(rows)
        else:
            features_table = "_No feature data available._"

        # Assign features to quarters based on impact ranking
        q1_features, q2_features, q3_features, q4_features = "TBD", "TBD", "TBD", "TBD"
        if report and report.get("feature_impact"):
            fi = report["feature_impact"]
            q1_features = _format_feature_list(fi[:3])
            q2_features = _format_feature_list(fi[3:6])
            q3_features = _format_feature_list(fi[6:9])
            q4_features = _format_feature_list(fi[9:12])

        current_year = datetime.now().year
        content = ROADMAP_TEMPLATE.format(
            title=draft.get("title", slug),
            generated_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            features_table=features_table,
            year=current_year,
            q1_features=q1_features,
            q2_features=q2_features,
            q3_features=q3_features,
            q4_features=q4_features,
        )

        path = self._output_dir / f"roadmap_{slug}_{ts}.md"
        path.write_text(content, encoding="utf-8")
        logger.info(f"Roadmap written: {path}")
        return path

    # ── Priority matrix ───────────────────────────────────────────────────────

    def _write_priority_matrix(
        self,
        report: AnalysisReport | None,
        ts: str,
        slug: str,
    ) -> Path:
        rows = ""
        if report and report.get("feature_impact"):
            for feat in report["feature_impact"][:15]:
                name = feat.get("feature_name", "Unknown")
                freq = feat.get("total_frequency", 1)
                pp_count = feat.get("pain_point_count", 1)

                # RICE scoring heuristics based on frequency data
                reach = min(freq * 10, 1000)         # Proxy: mentions as reach
                impact = min(pp_count * 2, 10)        # Proxy: pain point count
                confidence = 0.8                      # Default confidence
                effort = max(1, 10 - pp_count)        # Inverse: more pain = simpler fix

                rice = int((reach * impact * confidence) / max(effort, 1))
                rows += (
                    f"| {name} | {feat.get('category', '-')} "
                    f"| {reach} | {impact} | {confidence} "
                    f"| {effort} | **{rice}** |\n"
                )
        else:
            rows = "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"

        content = PRIORITY_MATRIX_TEMPLATE.format(
            generated_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            rows=rows,
        )

        path = self._output_dir / f"priority_matrix_{slug}_{ts}.md"
        path.write_text(content, encoding="utf-8")
        logger.info(f"Priority matrix written: {path}")
        return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_feature_list(features: list[dict[str, Any]]) -> str:
    if not features:
        return "_None scheduled._"
    return "\n".join(
        f"- **{f.get('feature_name', 'Unknown')}** ({f.get('category', '-')})"
        for f in features
    )
