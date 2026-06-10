"""Lesson 04 — windowed conversation memory.

Keep only the last N user+assistant turn pairs. The system prompt is
always preserved (it lives outside the messages list anyway), and the
*first* user message is also preserved as the "anchor task" so the agent
doesn't lose the original goal.

Run:
    uv run python lessons/04_memory_state/02_windowed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


SYSTEM = "You are a helpful, concise assistant."
KEEP_TURNS = 4  # how many recent user+assistant pairs to keep


def trim(messages: list[dict]) -> list[dict]:
    """Keep the first user message + the last KEEP_TURNS turns.

    A "turn" here is one user message + one assistant message.
    The first user message is kept as the anchor task.
    """
    if len(messages) <= 1 + 2 * KEEP_TURNS:
        return messages

    anchor = messages[0]
    tail = messages[-2 * KEEP_TURNS:]
    # Insert a tiny breadcrumb so the model knows it's reading a trimmed view.
    return [anchor, {"role": "user", "content": "(earlier turns trimmed)"}, *tail]


def repl() -> None:
    messages: list[dict] = []
    cumulative_in = 0

    print(f"Windowed REPL (keep last {KEEP_TURNS} turns). Empty line to quit.")
    while True:
        user = input("\nyou> ").strip()
        if not user:
            break

        messages.append({"role": "user", "content": user})

        # Trim *before* sending. The full history stays in memory for
        # display/debugging; only the trimmed view is sent.
        view = trim(messages)
        response = chat(view, system=SYSTEM)
        messages.append({"role": "assistant", "content": response.content})

        cumulative_in += response.input_tokens
        print(f"\nassistant> {response.text}")
        print(
            f"[in={response.input_tokens} (sent {len(view)} of {len(messages)} msgs)  "
            f"cumulative in={cumulative_in}]"
        )


if __name__ == "__main__":
    repl()
