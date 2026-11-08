"""Lesson 01b: stream text as it is generated.

Part 1 shows why streaming is not optional for long outputs: ask the
SDK for a large `max_tokens` without streaming and it refuses before
sending anything, because such a request could outlive the HTTP
connection. No request leaves your machine in Part 1.

Part 2 streams a few paragraphs with `client().messages.stream(...)`.
`text_stream` yields text pieces as they arrive; `get_final_message()`
then returns the complete Message, the same object `create()` would
have returned, with usage and stop_reason. The script prints the time
to the first text piece next to the total time.

Run:
    uv run python lessons/01b_streaming/01_stream_text.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
HANDBOOK = (CORPUS_DIR / "onboarding.md").read_text(encoding="utf-8")


def part1_big_request_needs_streaming() -> None:
    print("--- Part 1: a large max_tokens without streaming ---")
    try:
        client().messages.create(
            model=DEFAULT_MODEL,
            max_tokens=64000,
            messages=[{"role": "user", "content": "Write a long story."}],
        )
    except ValueError as e:
        print(f"  The SDK refused before sending: {e}")


def part2_stream_text() -> None:
    print("\n--- Part 2: streaming a welcome letter ---\n")
    start = time.perf_counter()
    first_text_at: float | None = None
    pieces = 0

    with client().messages.stream(
        model=DEFAULT_MODEL,
        max_tokens=8000,
        output_config={"effort": "low"},
        system="You write warm, concrete welcome letters from the handbook below.\n\n" + HANDBOOK,
        messages=[{"role": "user", "content": "Write a four paragraph welcome letter for a new engineer."}],
    ) as stream:
        for text in stream.text_stream:
            if first_text_at is None:
                first_text_at = time.perf_counter() - start
            pieces += 1
            print(text, end="", flush=True)  # flush, or the terminal buffers it
        final = stream.get_final_message()

    total = time.perf_counter() - start
    print("\n")
    print(f"time to first text : {first_text_at or 0:5.2f} s")
    print(f"total time         : {total:5.2f} s")
    print(f"text pieces        : {pieces}")
    print(f"block types        : {[block.type for block in final.content]}")
    print(f"stop_reason        : {final.stop_reason}")
    print(f"output tokens      : {final.usage.output_tokens}")
    print(
        "\nThe reader saw words after the first figure, not the second. "
        "Any thinking happens before the first text piece, so a higher effort "
        "moves that first figure too."
    )


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}\n")
    part1_big_request_needs_streaming()
    part2_stream_text()
