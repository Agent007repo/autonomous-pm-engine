"""
src/agents/engineering_agent.py

CrewAI Engineering Agent — ReAct + Self-Critique loop.

The Engineering Agent performs iterative quality review of the PRD draft:
  1. Read the PRD.
  2. Identify technical feasibility issues, missing NFRs, ambiguous ACs.
  3. Score the PRD on a 0–10 quality rubric.
  4. If score < threshold: generate specific feedback and loop.
  5. If score >= threshold OR max_rounds reached: write the final
     technical assessment and approve.

The self-critique loop is implemented as a Python loop (not LangGraph loop)
because CrewAI manages its own agent execution cycle; the outer loop just
re-runs the crew with the previous critique as additional context.
"""

from __future__ import annotations

import json
import re

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config.settings import get_settings
from src.orchestration.state import CritiqueRecord, PRDDraft


_CRITIQUE_RUBRIC = """
Score the PRD on these dimensions (0-2 points each, total 0-10):

1. CLARITY (0-2): Are acceptance criteria specific, measurable, and unambiguous?
2. FEASIBILITY (0-2): Are the proposed features technically achievable in a typical sprint cycle?
3. COMPLETENESS (0-2): Are NFRs (performance, security, accessibility) addressed?
4. CONSISTENCY (0-2): Do user stories, ACs, and goals align without contradiction?
5. RISK (0-2): Are technical risks and dependencies explicitly identified?

Score 0 = absent/poor, 1 = partial, 2 = excellent.
"""


def run_engineering_agent(
    prd_draft: PRDDraft,
    product_name: str,
    max_rounds: int | None = None,
    gate_threshold: float | None = None,
) -> tuple[PRDDraft, list[CritiqueRecord]]:
    """
    Run the Engineering Agent self-critique loop.

    Returns:
        updated_prd: PRDDraft with technical_assessment populated.
        critique_history: List of CritiqueRecord for each round.
    """
    cfg = get_settings()
    max_rounds = max_rounds or cfg.max_critique_rounds
    gate_threshold = gate_threshold or cfg.engineering_gate_threshold

    llm = ChatOpenAI(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        temperature=0.1,  # Low temperature for consistent scoring
    )

    critique_history: list[CritiqueRecord] = []
    current_prd = prd_draft.copy()
    accumulated_feedback = ""

    for round_num in range(1, max_rounds + 1):
        logger.info(f"Engineering critique round {round_num}/{max_rounds}...")

        eng_agent = Agent(
            role="Principal Software Engineer",
            goal=(
                "Review the PRD for technical feasibility, completeness, and clarity. "
                "Score it rigorously, provide specific actionable feedback, and "
                "write a technical assessment section."
            ),
            backstory=(
                "You are a Staff Engineer with 15 years of experience shipping "
                "complex B2B SaaS products. You have seen PRDs fail because of "
                "vague acceptance criteria and missing NFRs. You are direct, "
                "specific, and constructive."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        prd_text = _prd_to_text(current_prd)
        prior_feedback_section = (
            f"\n\nPRIOR CRITIQUE FEEDBACK (address these in your review):\n{accumulated_feedback}"
            if accumulated_feedback
            else ""
        )

        task = Task(
            description=f"""
Review this PRD for {product_name}:

{prd_text}
{prior_feedback_section}

Scoring rubric:
{_CRITIQUE_RUBRIC}

Return ONLY a JSON object with this exact schema:
{{
  "score": <float 0-10>,
  "passed": <true if score >= {gate_threshold}, false otherwise>,
  "feedback": "<specific, actionable list of issues and suggestions>",
  "technical_assessment": "<200-400 word assessment covering: architecture considerations, estimated complexity (S/M/L/XL t-shirt sizing), key dependencies, performance/security/accessibility NFRs, suggested implementation phases, and any blockers>"
}}
""",
            expected_output="JSON object with score, passed, feedback, and technical_assessment fields.",
            agent=eng_agent,
        )

        crew = Crew(
            agents=[eng_agent],
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
        except json.JSONDecodeError as e:
            logger.warning(f"Engineering Agent round {round_num} returned invalid JSON: {e}")
            data = {
                "score": 5.0,
                "passed": False,
                "feedback": "Unable to parse structured output. Retrying.",
                "technical_assessment": "",
            }

        score = float(data.get("score", 0.0))
        passed = bool(data.get("passed", score >= gate_threshold))
        feedback = str(data.get("feedback", ""))
        tech_assessment = str(data.get("technical_assessment", ""))

        record: CritiqueRecord = {
            "round": round_num,
            "score": score,
            "feedback": feedback,
            "passed": passed,
        }
        critique_history.append(record)

        logger.info(f"Round {round_num} score: {score:.1f}/10 | Passed: {passed}")

        # Update PRD with latest technical assessment
        current_prd["technical_assessment"] = tech_assessment

        if passed:
            logger.info(f"Engineering gate passed at round {round_num}.")
            break

        # Accumulate feedback for the next round
        accumulated_feedback += f"\nRound {round_num} feedback:\n{feedback}\n"

        if round_num == max_rounds:
            logger.warning(
                f"Max critique rounds ({max_rounds}) reached. "
                f"Final score: {score:.1f}. Proceeding with best available draft."
            )

    return current_prd, critique_history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prd_to_text(prd: PRDDraft) -> str:
    """Render a PRDDraft as a readable text block for the agent prompt."""
    sections = [
        ("Title", prd.get("title", "")),
        ("Version", prd.get("version", "")),
        ("Executive Summary", prd.get("executive_summary", "")),
        ("Problem Statement", prd.get("problem_statement", "")),
        ("Goals and Metrics", prd.get("goals_and_metrics", "")),
        ("User Stories", prd.get("user_stories", "")),
        ("Acceptance Criteria", prd.get("acceptance_criteria", "")),
        ("Non-Goals", prd.get("non_goals", "")),
        ("Open Questions", prd.get("open_questions", "")),
    ]
    lines = []
    for heading, content in sections:
        lines.append(f"## {heading}\n{content}\n")
    return "\n".join(lines)
