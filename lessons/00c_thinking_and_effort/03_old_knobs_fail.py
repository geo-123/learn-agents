"""Lesson 00c: the old knobs now fail, effort is the one that works.

Code written for 2024 and 2025 models often sends a thinking budget
(`thinking: {"type": "enabled", "budget_tokens": N}`) or a sampling
setting such as `temperature: 0`. On current models both come back as
HTTP 400. This script sends each old knob once, prints the error, then
sends the modern equivalent.

A 400 comes back before the model runs, so the failing calls produce no
output tokens.

`temperature` is no longer a parameter of `messages.create()` in the
SDK, so the script smuggles it in with `extra_body` to show what the
API itself says.

Run:
    uv run python lessons/00c_thinking_and_effort/03_old_knobs_fail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import anthropic
from anthropic.types import MessageParam

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

QUESTION: list[MessageParam] = [{"role": "user", "content": "Classify the mood of 'the build is green again' in one word."}]


def old_thinking_budget() -> None:
    print("\n--- Old: thinking budget ---")
    try:
        client().messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            messages=QUESTION,
        )
        print("  accepted (this model still takes budgets)")
    except anthropic.BadRequestError as e:
        print(f"  400: {e.message}")


def old_temperature() -> None:
    print("\n--- Old: temperature 0 for a 'deterministic' answer ---")
    try:
        client().messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1000,
            messages=QUESTION,
            extra_body={"temperature": 0},
        )
        print("  accepted (this model still takes sampling settings)")
    except anthropic.BadRequestError as e:
        print(f"  400: {e.message}")


def modern() -> None:
    print("\n--- New: adaptive thinking (the default) plus a low effort ---")
    msg = client().messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1000,  # room for any thinking plus the one word answer
        output_config={"effort": "low"},
        messages=QUESTION,
    )
    kinds = [block.type for block in msg.content]
    answer = "".join(block.text for block in msg.content if block.type == "text")
    print(f"  block types : {kinds}")
    print(f"  answer      : {answer.strip()}")
    print(f"  output tokens: {msg.usage.output_tokens}")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")
    old_thinking_budget()
    old_temperature()
    modern()
    print(
        "\nLess thinking: lower the effort. Steadier answers: a tighter prompt "
        "or structured output, not a sampling setting."
    )
