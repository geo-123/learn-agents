"""Lesson 01c: code execution, a sandbox you do not have to build.

We give Claude the Acme expense policy and one trip's costs, and ask it
to compute what gets reimbursed. With the `code_execution` tool
declared, Claude writes Python, runs it in a container on Anthropic's
side, reads the output and answers. You see each command and its
stdout as blocks in the same response.

Two details this script handles that the helper in src/_llm.py hides:
    * The response carries a `container` id. On `pause_turn` you send it
      back, or the resumed turn starts in a fresh container and loses
      its files.
    * Result blocks hold `return_code` and `stderr`. A failed run is
      still HTTP 200; Claude reads the error and usually tries again.

At the end we compute the same numbers locally, so you can check the
model's arithmetic instead of trusting it.

Cost: tokens, plus container time (5 minute minimum; each organization
gets 1,550 free hours a month).

Run:
    uv run python lessons/01c_server_tools/02_code_execution.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

POLICY = (Path(__file__).resolve().parents[2] / "data" / "corpus" / "expense_policy.md").read_text(encoding="utf-8")

# (city, nights, nightly rate) and one meal total per day, in USD.
HOTELS = [("Tokyo", 3, 420), ("Berlin", 2, 280)]
MEALS = [95, 70, 88, 60, 81]

QUESTION = f"""Here is our expense policy:

{POLICY}

A trip: hotels {HOTELS} as (city, nights, nightly rate in USD), meals per day {MEALS} in USD.
Write and run Python that applies the hotel and meal caps, and print a table with
claimed, reimbursable and over cap amounts per line, plus totals. Then summarize in two sentences."""

TOOLS: list[Any] = [{"type": "code_execution_20260521", "name": "code_execution"}]


def show(block: dict[str, Any]) -> None:
    kind = block["type"]
    if kind == "server_tool_use":
        command = block["input"].get("command") or block["input"].get("code") or block["input"]
        print(f"  [server_tool_use] {block['name']}:\n      {str(command)[:300]}")
    elif kind.endswith("_tool_result"):
        content = block["content"]
        if str(content.get("type", "")).endswith("_error"):
            print(f"  [{kind}] ERROR {content.get('error_code')}")
        elif "stdout" in content:
            print(f"  [{kind}] return_code={content.get('return_code')}")
            for line in (content.get("stdout") or "").splitlines()[:15]:
                print(f"      | {line}")
            if content.get("stderr"):
                print(f"      stderr: {content['stderr'][:200]}")
        else:
            print(f"  [{kind}] {content.get('type')}")
    elif kind == "text":
        print(f"  [text] {block['text'].strip()}")
    else:
        print(f"  [{kind}]")


def run(max_continuations: int = 3) -> None:
    messages: list[Any] = [{"role": "user", "content": QUESTION}]
    container_id: str | None = None
    for _ in range(max_continuations + 1):
        extra: dict[str, Any] = {"container": container_id} if container_id else {}
        msg = client().messages.create(
            model=DEFAULT_MODEL, max_tokens=8000, messages=messages, tools=TOOLS, **extra,
        )
        blocks = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
        for block in blocks:
            show(block)
        if msg.container is not None:
            container_id = msg.container.id
        messages.append({"role": "assistant", "content": blocks})
        print(f"  stop_reason={msg.stop_reason}  container={container_id}  "
              f"tokens in/out={msg.usage.input_tokens}/{msg.usage.output_tokens}")
        if msg.stop_reason != "pause_turn":
            return
        print("  -- pause_turn: resending with the same container --")


def expected() -> None:
    """The same computation, done locally: the answer key."""
    caps = {"NYC": 400, "London": 400, "Tokyo": 400}
    hotel_ok = sum(n * min(rate, caps.get(city, 300)) for city, n, rate in HOTELS)
    hotel_claimed = sum(n * rate for _, n, rate in HOTELS)
    meal_ok = sum(min(m, 80) for m in MEALS)
    total_claimed = hotel_claimed + sum(MEALS)
    print(f"  reimbursable: hotels {hotel_ok}, meals {meal_ok}, total {hotel_ok + meal_ok}")
    print(f"  over cap    : {total_claimed - hotel_ok - meal_ok} (claimed {total_claimed})")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}\n")
    run()
    print("\nAnswer key computed locally:")
    expected()
