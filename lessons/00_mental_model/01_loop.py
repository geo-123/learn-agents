"""Lesson 00: the bare agent loop.

Stop condition: the assistant's text ends with the word "DONE".
No tools, no retries, no memory beyond the messages list.

Run:
    uv run python lessons/00_mental_model/01_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


SYSTEM = (
    "You are a brainstorming agent. RULES: on each turn, propose exactly ONE "
    "item with a one sentence rationale, then stop and wait for the user to "
    "say 'Continue.' Do not list multiple items in a single turn. "
    "When and only when you have produced the number requested, end your "
    "final message with the word DONE on its own line."
)


def run_agent(task: str, *, max_steps: int = 5) -> list[dict]:
    """Run the loop until the assistant says DONE or we hit max_steps."""
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step} ---")
        print(response.text)
        print(f"(in={response.input_tokens} out={response.output_tokens})")

        if response.text.strip().endswith("DONE"):
            print("\n[stop] assistant said DONE")
            return messages

        # Prompt the model to continue. Without this nudge, the model would
        # often stop after one turn: there's no other "user" speaking.
        messages.append({"role": "user", "content": "Continue."})

    print(f"\n[stop] hit max_steps={max_steps}")
    return messages


if __name__ == "__main__":
    # A multi item task with a "one per turn" system rule forces the loop to
    # actually iterate. You should see step 1..4 with a single name each,
    # then a final turn that ends with DONE.
    task = "Propose 4 names for a new note taking app aimed at developers."
    run_agent(task)
