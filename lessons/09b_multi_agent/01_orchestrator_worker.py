"""Lesson 09b: orchestrator + parallel workers.

An orchestrator LLM call decomposes a research question into sub-
questions; each is dispatched to a worker (a scoped LLM call) running
in parallel; the orchestrator synthesizes the worker outputs into one
answer. Each worker has its own focused system prompt and never sees
the others' work: context isolation is the whole point.

Run:
    uv run python lessons/09b_multi_agent/01_orchestrator_worker.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def load_handbook() -> str:
    docs: list[str] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        docs.append(f"## {path.name}\n\n{path.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(docs)


HANDBOOK = load_handbook()


# --- Step 1: orchestrator decomposes the question -------------------------


class SubQuestions(BaseModel):
    """Schema the orchestrator must fill on the decomposition turn."""
    sub_questions: list[str] = Field(
        description="2 to 5 focused subquestions whose answers together cover the user's question.",
        min_length=2,
        max_length=5,
    )


ORCHESTRATOR_SYSTEM = (
    "You are a research orchestrator. The user asks one composite question. "
    "Split it into 2 to 5 focused subquestions a worker can answer independently "
    "in parallel. Each subquestion should be self contained: workers don't "
    "see each other's output."
)


def decompose(question: str) -> list[str]:
    response = chat(
        system=ORCHESTRATOR_SYSTEM,
        messages=[{"role": "user", "content": question}],
        output_schema=SubQuestions,  # structured output (Lesson 02)
    )
    if response.parsed is None:
        raise RuntimeError(f"orchestrator gave no plan (stop_reason={response.stop_reason})")
    return response.parsed.sub_questions


# --- Step 2: workers run in parallel --------------------------------------


WORKER_SYSTEM = (
    "You are a focused research worker. Answer the user's subquestion using "
    "ONLY the Acme handbook below. Be concise (1 to 3 sentences) and cite the "
    "doc name when you can. If the handbook doesn't cover it, say so.\n\n"
    f"=== HANDBOOK ===\n\n{HANDBOOK}\n\n=== END HANDBOOK ==="
)


def worker(sub_question: str) -> str:
    response = chat(
        system=WORKER_SYSTEM,
        messages=[{"role": "user", "content": sub_question}],
    )
    return response.text.strip()


# --- Step 3: orchestrator synthesizes -------------------------------------


SYNTHESIZER_SYSTEM = (
    "You are a research synthesizer. Combine the worker findings below into "
    "a single coherent answer to the user's original question. Keep it under "
    "8 sentences. Cite the doc names the workers cited."
)


def synthesize(original_question: str, sub_q_and_a: list[tuple[str, str]]) -> str:
    findings = "\n\n".join(
        f"Subquestion: {q}\nFinding: {a}" for q, a in sub_q_and_a
    )
    response = chat(
        system=SYNTHESIZER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Original question: {original_question}\n\n=== Worker findings ===\n\n{findings}",
        }],
    )
    return response.text.strip()


# --- Glue: the full orchestrator worker flow ------------------------------


def run(question: str) -> str:
    print(f"\nQ: {question}")

    print("\n[1] decompose")
    sub_qs = decompose(question)
    for i, sq in enumerate(sub_qs, start=1):
        print(f"    {i}. {sq}")

    print(f"\n[2] dispatch {len(sub_qs)} workers in parallel")
    with ThreadPoolExecutor(max_workers=len(sub_qs)) as pool:
        # zip preserves the subquestion / answer pairing across the parallel calls.
        answers = list(pool.map(worker, sub_qs))
    for i, (sq, ans) in enumerate(zip(sub_qs, answers), start=1):
        print(f"    [worker {i}] {sq}")
        print(f"               -> {ans[:120]}")

    print("\n[3] synthesize")
    final = synthesize(question, list(zip(sub_qs, answers)))
    return final


if __name__ == "__main__":
    questions = [
        # Naturally decomposes into vacation + parental leave + sick leave.
        "Compare Acme's vacation policy, parental leave, and sick leave.",
        # Naturally decomposes into incident SEV tiers + on call stipend + postmortem.
        "Walk me through how Acme handles a production incident, from page to postmortem.",
    ]
    for q in questions:
        answer = run(q)
        print(f"\nA: {answer}\n{'='*72}")
