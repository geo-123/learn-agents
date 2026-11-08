"""Lesson 01b: the events underneath a stream.

`text_stream` hides the protocol. Iterating the stream itself shows every
server sent event with the time it arrived:

    message_start
      content_block_start  (thinking)    content_block_delta x N   content_block_stop
      content_block_start  (text)        content_block_delta x N   content_block_stop
    message_delta  (stop_reason, final output token count)
    message_stop

The SDK helper also yields convenience events built from the deltas:
`thinking`, `signature`, `text` and `input_json`. They repeat what the
raw deltas carry, so this script counts them separately.

Thinking is requested with display "summarized" so the thinking deltas
carry text you can watch arrive before the answer.

Run:
    uv run python lessons/01b_streaming/02_stream_events.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

CONVENIENCE = {"text", "thinking", "signature", "input_json", "citation"}
QUESTION = (
    "A hotel in Tokyo costs $380 a night and the cap there is $400. A hotel "
    "in Lisbon costs $320 and the cap there is $300. Which stays need "
    "approval? Answer in two sentences."
)


def main() -> None:
    start = time.perf_counter()
    convenience = Counter()
    run_label = ""
    run_count = 0
    run_started = 0.0

    def flush_run() -> None:
        if run_count:
            times = f"x{run_count}" if run_count > 1 else ""
            print(f"  {run_started:6.2f} s  {run_label} {times}")

    with client().messages.stream(
        model=DEFAULT_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": QUESTION}],
    ) as stream:
        for event in stream:
            now = time.perf_counter() - start
            if event.type in CONVENIENCE:
                convenience[event.type] += 1
                continue

            # Build a label that says what this raw event carries.
            if event.type == "message_start":
                label = f"message_start (input tokens {event.message.usage.input_tokens})"
            elif event.type == "content_block_start":
                label = f"content_block_start [{event.index}] {event.content_block.type}"
            elif event.type == "content_block_delta":
                label = f"content_block_delta [{event.index}] {event.delta.type}"
            elif event.type == "content_block_stop":
                label = f"content_block_stop  [{event.index}]"
            elif event.type == "message_delta":
                label = (f"message_delta (stop_reason {event.delta.stop_reason}, "
                         f"output tokens {event.usage.output_tokens})")
            else:
                label = event.type

            # Collapse runs of identical deltas into one line with a count.
            if label == run_label:
                run_count += 1
                continue
            flush_run()
            run_label, run_count, run_started = label, 1, now
        flush_run()
        final = stream.get_final_message()

    print(f"\nConvenience events from the SDK: {dict(convenience)}")
    for block in final.content:
        if block.type == "thinking":
            print(f"\nThinking summary: {block.thinking[:300]}")
        elif block.type == "text":
            print(f"\nAnswer: {block.text}")
    print(
        "\nNotice the gap before the first text delta: that is the thinking. "
        "The stop_reason only arrives in message_delta, near the very end."
    )


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}\n")
    main()
