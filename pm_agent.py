"""
src/agents/pm_agent.py

CrewAI PM Agent — Plan-and-Execute pattern.

The PM Agent receives the AnalysisReport and constructs a PRD by:
  1. Planning: Create a section-by-section execution plan.
  2. Executing: Draft each PRD section, calling hybrid_search to
               ground each section in customer evidence.
  3. Assembling: Return a complete PRDDraft.

This approach produces better output than asking the LLM to write
the entire PRD in one shot, because each section is drafted with
targeted retrieval context.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config.settings import get_settings
from src.orchestration.state import AnalysisReport, PRDDraft

if TYPE_CHECKING:
    from langchain.tools import StructuredTool


_PRD_SYSTEM_CONTEXT = """
You are a Principal Product Manager at a top-tier software company.
You write PRDs that engineering teams love: precise, evidence-backed, 
with clear acceptance criteria and measurable success metrics.

PRD Quality Standards:
- Every problem statement must cite specific data (frequencies, percentages)
- User stories follow: "As a <persona>, I want <goal> so that <benefit>"
- Acceptance criteria follow Gherkin: Given/When/Then
- Goals use OKR format: Objective + 3-5 Key Results with numeric targets
- Non-goals must be explicit to prevent scope creep
"""


def run_pm_agent(
    tools: list["StructuredTool"],
    analysis_report: AnalysisReport,
    product_name: str,
    product_context: str,
    prd_version: str = "1.0",
) -> PRDDraft:
    """
    Run the PM Agent and return a populated PRDDraft.
    """
    cfg = get_settings()
    llm = ChatOpenAI(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        temperature=cfg.openai_temperature,
        max_tokens=cfg.openai_max_tokens,
    )

    pm_agent = Agent(
        role="Principal Product Manager",
        goal=(
            "Write a complete, evidence-backed Product Requirements Document (PRD) "
            "based on the analysis report. Every section must be grounded in "
            "customer data retrieved via your search tools."
        ),
        backstory=_PRD_SYSTEM_CONTEXT,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=20,
    )

    analysis_json = json.dumps(analysis_report, indent=2)

    task = Task(
        description=f"""
Write a complete PRD for: {product_name}
Product context: {product_context}

You have access to this Analysis Report (already computed):
{analysis_json}

EXECUTION PLAN (follow these steps in order):
1. Call hybrid_search 2-3 times with targeted queries to collect supporting evidence
   for the top 3 most impactful features from the report.
2. Draft the executive_summary (100-150 words, cite top 2 themes by name).
3. Draft the problem_statement (200-300 words, cite at least 3 pain points with frequencies).
4. Draft goals_and_metrics in OKR format (1 Objective, 4 Key Results with numeric targets).
5. Draft user_stories: write 3-5 user stories covering the top features.
6. Draft acceptance_criteria: Gherkin-format criteria for each user story.
7. Draft non_goals: list 4-6 explicit out-of-scope items.
8. Draft open_questions: 3-5 unresolved questions requiring stakeholder input.
9. Draft source_evidence: list 5-8 representative customer quotes (from hybrid_search results).

Return ONLY a JSON object matching this exact schema (no markdown, no preamble):
{{
  "title": "PRD: <feature or initiative name>",
  "version": "{prd_version}",
  "status": "Draft",
  "executive_summary": "...",
  "problem_statement": "...",
  "goals_and_metrics": "...",
  "user_stories": "...",
  "acceptance_criteria": "...",
  "non_goals": "...",
  "technical_assessment": "",
  "open_questions": "...",
  "source_evidence": "..."
}}
""",
        expected_output="A JSON object matching the PRDDraft schema.",
        agent=pm_agent,
    )

    crew = Crew(
        agents=[pm_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw = str(result).strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        draft: PRDDraft = {
            "title": data.get("title", f"PRD: {product_name}"),
            "version": data.get("version", prd_version),
            "status": data.get("status", "Draft"),
            "executive_summary": data.get("executive_summary", ""),
            "problem_statement": data.get("problem_statement", ""),
            "goals_and_metrics": data.get("goals_and_metrics", ""),
            "user_stories": data.get("user_stories", ""),
            "acceptance_criteria": data.get("acceptance_criteria", ""),
            "non_goals": data.get("non_goals", ""),
            "technical_assessment": "",
            "open_questions": data.get("open_questions", ""),
            "source_evidence": data.get("source_evidence", ""),
        }
        logger.info("PM Agent complete. PRD draft assembled.")
        return draft
    except json.JSONDecodeError as e:
        logger.error(f"PM Agent returned invalid JSON: {e}")
        raise ValueError(f"PM Agent returned non-JSON output: {e}") from e
