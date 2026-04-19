"""
src/agents/data_analyst_agent.py

CrewAI Data Analyst Agent.

Responsibilities:
  - Query the vector store and graph store for quantitative trends.
  - Produce a structured AnalysisReport that the PM Agent uses as its brief.
  - Cite evidence from actual customer data (no hallucination).

Pattern: ReAct (reason + act) — uses tools iteratively until it has
         enough evidence to write a complete AnalysisReport.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config.settings import get_settings
from src.orchestration.state import AnalysisReport

if TYPE_CHECKING:
    from langchain.tools import StructuredTool


def run_data_analyst(
    tools: list["StructuredTool"],
    product_name: str,
    product_context: str,
) -> AnalysisReport:
    """
    Instantiate and run the Data Analyst Agent.
    Returns a populated AnalysisReport dict.
    """
    cfg = get_settings()
    llm = ChatOpenAI(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        temperature=cfg.openai_temperature,
    )

    analyst = Agent(
        role="Senior Data Analyst",
        goal=(
            "Analyse all customer feedback data and produce a comprehensive, "
            "evidence-based AnalysisReport for the product team. "
            "Every claim must be backed by data retrieved via your tools."
        ),
        backstory=(
            "You are a rigorous quantitative analyst with 10 years of experience "
            "turning messy customer data into actionable product insights. "
            "You distrust anecdotes and always cite frequencies and counts."
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=12,
    )

    task = Task(
        description=f"""
Analyse customer feedback data for: {product_name}
Context: {product_context}

Your deliverable is a JSON object that EXACTLY matches this schema:
{{
  "top_pain_points": [
    {{"id": "...", "text": "...", "frequency": <int>, "doc_type": "..."}}
  ],
  "feature_impact": [
    {{"feature_name": "...", "category": "...", "pain_point_count": <int>, "total_frequency": <int>}}
  ],
  "theme_summary": [
    {{"theme": "...", "pain_point_count": <int>, "total_mentions": <int>}}
  ],
  "co_occurrences": [
    {{"pain_a": "...", "pain_b": "...", "co_count": <int>}}
  ],
  "narrative_summary": "<200-300 word analyst summary of the most important findings>"
}}

Steps:
1. Call `top_pain_points` to get frequency data.
2. Call `feature_impact_summary` to get feature rankings.
3. Call `theme_summary` to get theme clustering.
4. Call `co_occurring_pain_points` to find related issues.
5. Call `hybrid_search` 2-3 times with targeted queries to pull representative
   verbatim quotes for the narrative summary.
6. Write the narrative_summary grounded in the data you retrieved.
7. Return ONLY the JSON object, no markdown fences, no explanation.
""",
        expected_output="A JSON object matching the AnalysisReport schema above.",
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw = str(result).strip()

    # Strip any accidental markdown fences
    import re
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        report: AnalysisReport = {
            "top_pain_points": data.get("top_pain_points", []),
            "feature_impact": data.get("feature_impact", []),
            "theme_summary": data.get("theme_summary", []),
            "co_occurrences": data.get("co_occurrences", []),
            "narrative_summary": data.get("narrative_summary", ""),
        }
        logger.info("Data Analyst Agent complete.")
        return report
    except json.JSONDecodeError as e:
        logger.error(f"Data Analyst returned invalid JSON: {e}")
        logger.debug(f"Raw output: {raw[:500]}")
        raise ValueError(f"Data Analyst Agent returned non-JSON output: {e}") from e
