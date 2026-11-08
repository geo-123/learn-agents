"""Lesson 05: budget guards.

Every retry layer can run away. The outermost defense is a budget:
caps on steps, wall time, and cumulative input tokens. The loop checks
the budget BEFORE each turn so the worst case is "one extra turn over
budget" rather than "indefinite."

Run:
    uv run python lessons/05_retries_errors/04_budget_guard.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


@dataclass
class Budget:
    """Stop condition aggregator. Any field tripping ends the loop."""
    max_steps: int = 6
    max_seconds: float = 20.0
    max_input_tokens: int = 8_000

    steps_used: int = 0
    seconds_used: float = 0.0
    input_tokens_used: int = 0

    started: float = 0.0

    def start(self) -> None:
        self.started = time.monotonic()

    def step(self, *, input_tokens: int = 0) -> None:
        self.steps_used += 1
        self.seconds_used = time.monotonic() - self.started
        self.input_tokens_used += input_tokens

    def exhausted(self) -> tuple[bool, str]:
        if self.steps_used >= self.max_steps:
            return True, f"max_steps ({self.max_steps})"
        if time.monotonic() - self.started >= self.max_seconds:
            return True, f"max_seconds ({self.max_seconds:.0f})"
        if self.input_tokens_used >= self.max_input_tokens:
            return True, f"max_input_tokens ({self.max_input_tokens})"
        return False, ""

    def report(self) -> str:
        return (
            f"steps={self.steps_used}/{self.max_steps}  "
            f"seconds={self.seconds_used:.1f}/{self.max_seconds:.0f}  "
            f"input_tokens={self.input_tokens_used}/{self.max_input_tokens}"
        )


class BudgetExceeded(RuntimeError):
    pass


def run_agent(task: str, budget: Budget) -> str:
    messages: list[dict] = [{"role": "user", "content": task}]
    budget.start()

    while True:
        # Check BEFORE the call. Cheaper to abort than to make a doomed call.
        done, reason = budget.exhausted()
        if done:
            print(f"\n[budget exhausted] {reason}  |  {budget.report()}")
            return f"(stopped: budget exhausted by {reason})"

        response = chat(messages)
        messages.append({"role": "assistant", "content": response.content})
        budget.step(input_tokens=response.input_tokens)

        print(f"\n--- step {budget.steps_used}  stop={response.stop_reason} ---")
        print(f"text: {response.text}")
        print(f"budget: {budget.report()}")

        if response.stop_reason != "tool_use":
            return response.text

        # If we had tools, we'd dispatch here. For this lesson we just loop
        # text only to keep the focus on budgeting.
        messages.append({"role": "user", "content": "Continue."})


if __name__ == "__main__":
    # Tight budget on purpose: the agent should hit a cap before finishing
    # whatever rambling task we give it.
    budget = Budget(max_steps=3, max_seconds=10, max_input_tokens=3000)
    answer = run_agent(
        "Outline a 10 step plan for migrating a monolith to microservices. "
        "Take your time and be thorough.",
        budget,
    )
    print(f"\n=== final ===\n{answer}")
