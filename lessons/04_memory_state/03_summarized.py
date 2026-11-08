"""Lesson 04: summarization based memory.

When the conversation exceeds a token threshold, fire ONE extra LLM call
to summarize the older half. The summary replaces the older messages and
lives in a special "summary" system note. Costlier per compaction event,
cheaper per subsequent turn.

Run:
    uv run python lessons/04_memory_state/03_summarized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat, strip_thinking


BASE_SYSTEM = "You are a helpful, concise assistant."

# Compact when the most recent turn used more than this many input tokens.
# In production you'd track cumulative tokens or character counts; this
# threshold on last turn approximation is enough for the lesson.
SUMMARIZE_TRIGGER_TOKENS = 1500


SUMMARIZER_SYSTEM = (
    "You are a memory summarizer for an AI assistant. Given a conversation "
    "transcript, produce a TIGHT summary that captures: the user's goals, "
    "decisions made, open questions, and any facts about the user worth "
    "remembering (preferences, names, dates, constraints). Use 6 bullets "
    "max. Do not include filler."
)


def summarize(messages_to_compact: list[dict]) -> str:
    transcript = []
    for m in messages_to_compact:
        content = m["content"]
        if isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        else:
            text = content
        transcript.append(f"{m['role'].upper()}: {text}")

    response = chat(
        messages=[{"role": "user", "content": "\n\n".join(transcript)}],
        system=SUMMARIZER_SYSTEM,
        max_tokens=400,
    )
    return response.text


def repl() -> None:
    messages: list[dict] = []
    summary: str = ""
    cumulative_in = 0

    print(f"Summarized REPL (trigger > {SUMMARIZE_TRIGGER_TOKENS} input tokens).")
    print("Empty line to quit.")
    while True:
        user = input("\nyou> ").strip()
        if not user:
            break

        messages.append({"role": "user", "content": user})

        # Build the system prompt: base + any accumulated summary.
        system = BASE_SYSTEM
        if summary:
            system = f"{BASE_SYSTEM}\n\nMemory of earlier conversation:\n{summary}"

        response = chat(messages, system=system)
        messages.append({"role": "assistant", "content": response.content})

        cumulative_in += response.input_tokens
        print(f"\nassistant> {response.text}")
        print(
            f"[in={response.input_tokens}  cumulative={cumulative_in}  "
            f"history={len(messages)} msgs  summary={'yes' if summary else 'no'}]"
        )

        # Compact if this turn was expensive. Keep the last 2 turns verbatim;
        # everything before becomes the new summary (merged with any prior summary).
        if response.input_tokens > SUMMARIZE_TRIGGER_TOKENS and len(messages) > 4:
            to_compact = messages[:-4]  # everything except the last 2 turn pairs
            keep = messages[-4:]
            print("\n[compacting older history into summary...]")
            new_summary_pieces = [summary] if summary else []
            new_summary_pieces.append(summarize(to_compact))
            summary = "\n\n".join(new_summary_pieces)
            # The system prompt changes and the older turns go, so the thinking
            # blocks of the kept turns no longer match what came before them.
            # Current models reject such blocks: strip them from the kept turns.
            messages = strip_thinking(keep)
            print(f"[summary now: {summary!r}]")


if __name__ == "__main__":
    repl()
