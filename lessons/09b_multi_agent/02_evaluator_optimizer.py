"""Lesson 09b — evaluator-optimizer loop.

A generator writes a draft, a critic scores it against a structured
rubric and lists specific issues, the generator revises. Loop until
the critic approves or we hit max_iterations.

The critic has a *different* system prompt and a *forced-structured*
output schema — both are anti-rubber-stamp measures. Without those,
loops like this tend to converge in 1 turn with the critic always
saying "looks good."

Run:
    uv run python lessons/09b_multi_agent/02_evaluator_optimizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


# --- Task -----------------------------------------------------------------

BRIEF = (
    "Draft an internal company-wide announcement of an updated PTO carryover "
    "policy: starting Q3 2026 (July 1), employees may carry over up to 10 "
    "unused vacation days into the following year. Previously zero days "
    "carried over. Mention HR can answer questions."
)

REQUIREMENTS = [
    'Must be under 80 words',
    'Must explicitly mention "Q3 2026" or "July 1, 2026"',
    'Must mention the 10-day cap',
    'Must mention HR as the question contact',
    'Tone: professional, brief, no marketing language ("excited", "thrilled", "amazing")',
]


# --- Generator ------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You write concise internal company announcements. Plain prose, no "
    "headings, no bullet points, no marketing voice."
)


def generate(brief: str, prior_draft: str | None, critic_feedback: str | None) -> str:
    if prior_draft is None:
        user = f"Brief:\n{brief}\n\nWrite the announcement."
    else:
        user = (
            f"Brief:\n{brief}\n\n"
            f"Your previous draft:\n{prior_draft}\n\n"
            f"Critic feedback (address every issue):\n{critic_feedback}\n\n"
            "Rewrite the announcement."
        )
    response = chat(
        system=GENERATOR_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return response.text.strip()


# --- Critic ---------------------------------------------------------------


class Critique(BaseModel):
    """Structured output forces the critic to commit to a verdict and
    list specific issues — no waffly "looks pretty good" responses."""
    verdict: Literal["approve", "revise"] = Field(
        description="`approve` if every requirement is met, else `revise`."
    )
    issues: list[str] = Field(
        description=(
            "List of specific, concrete issues with the draft. Empty list "
            "if verdict is `approve`. Each issue must reference a "
            "requirement number or quote the offending text."
        ),
    )


CRITIQUE_TOOL = {
    "name": "submit_critique",
    "description": "Submit your structured critique of the draft.",
    "input_schema": Critique.model_json_schema(),
}

CRITIC_SYSTEM = (
    "You are a strict editorial reviewer. You receive an announcement and a "
    "checklist of requirements. Approve ONLY if every requirement is met. If "
    "any requirement fails — even partially — verdict is `revise` and the "
    "issues list must name what's wrong. Be specific: quote the offending "
    "phrase or cite the missing fact."
)


def critique(draft: str) -> Critique:
    reqs = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(REQUIREMENTS))
    user = (
        f"Requirements:\n{reqs}\n\n"
        f"Draft:\n---\n{draft}\n---\n\n"
        f"Word count: {len(draft.split())}.\n\n"
        f"Critique it now."
    )
    response = chat(
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": user}],
        tools=[CRITIQUE_TOOL],
        tool_choice={"type": "tool", "name": "submit_critique"},
    )
    return Critique.model_validate(response.tool_uses[0]["input"])


# --- The loop -------------------------------------------------------------


def run(brief: str, *, max_iterations: int = 4) -> str:
    draft: str | None = None
    feedback: str | None = None

    for i in range(1, max_iterations + 1):
        print(f"\n--- iteration {i} ---")
        draft = generate(brief, draft, feedback)
        print(f"draft ({len(draft.split())} words):\n  {draft}")

        c = critique(draft)
        print(f"critic: {c.verdict.upper()}")
        if c.verdict == "approve":
            return draft

        for issue in c.issues:
            print(f"  - {issue}")
        feedback = "\n".join(f"- {issue}" for issue in c.issues)

    print(f"\n(hit max_iterations={max_iterations}; returning last draft)")
    return draft or ""


if __name__ == "__main__":
    final = run(BRIEF)
    print(f"\n{'='*72}\nFinal draft:\n{final}")
