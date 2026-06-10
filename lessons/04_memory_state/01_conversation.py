"""Lesson 04 — naive append-everything conversation memory.

A REPL where every turn appends to the history list. Watch the input
token count grow turn over turn. This is the baseline we're going to
improve on.

Run:
    uv run python lessons/04_memory_state/01_conversation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


SYSTEM = "You are a helpful, concise assistant."


def repl() -> None:
    messages: list[dict] = []
    cumulative_in = 0
    cumulative_out = 0

    print("Conversation REPL. Empty line to quit.")
    while True:
        user = input("\nyou> ").strip()
        if not user:
            break

        messages.append({"role": "user", "content": user})
        response = chat(messages, system=SYSTEM)
        messages.append({"role": "assistant", "content": response.content})

        cumulative_in += response.input_tokens
        cumulative_out += response.output_tokens

        print(f"\nassistant> {response.text}")
        print(
            f"[tokens this turn: in={response.input_tokens} out={response.output_tokens}  "
            f"cumulative: in={cumulative_in} out={cumulative_out}  "
            f"history length: {len(messages)} messages]"
        )


if __name__ == "__main__":
    repl()
