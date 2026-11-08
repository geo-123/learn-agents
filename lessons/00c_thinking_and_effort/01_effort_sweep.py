"""Lesson 00c: one hard question at three effort levels.

Effort (`output_config.effort`) is the main quality and cost knob on
current Claude models. It sets how much the model thinks and how
thorough the answer is. This script asks the same multi step expense
question at low, medium and high effort and prints, for each level:

    * wall clock time
    * output tokens (thinking tokens are billed as output and are included)
    * how many thinking blocks came back
    * the first lines of the answer

Notice that the token count, and with it the cost and the latency, grows
with the effort level, while a question this size is often answered
correctly at every level. That is the point: choose the lowest level
that passes your own checks.

Run:
    uv run python lessons/00c_thinking_and_effort/01_effort_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
POLICY = (CORPUS_DIR / "expense_policy.md").read_text(encoding="utf-8")

SYSTEM = (
    "You are Acme's expense checker. Use only the policy below. "
    "Work out the reimbursable amount line by line, then give the total "
    "on the last line as: TOTAL: $<amount>.\n\n" + POLICY
)

# Several caps, one city exception, a receipt rule and an approval rule:
# enough steps that more thinking can matter.
QUESTION = (
    "Four day trip to London (3 nights). Flight 7 hours each way, premium "
    "economy, $1,900 return. Hotel $420 per night. Meals: $95, $70, $88, "
    "$60 (all with itemized receipts, one $70 dinner had no attendee names "
    "in the memo). Client dinner $260 with receipt, no preapproval. Taxi "
    "$22 without receipt. What is reimbursable without VP approval, and "
    "which items need VP approval?"
)

LEVELS = ["low", "medium", "high"]


def run(level: str) -> None:
    start = time.perf_counter()
    response = chat(
        system=SYSTEM,
        messages=[{"role": "user", "content": QUESTION}],
        effort=level,
        max_tokens=16000,  # thinking counts toward max_tokens: leave room
    )
    seconds = time.perf_counter() - start

    thinking_blocks = [b for b in response.content if b.get("type") == "thinking"]
    total_line = next(
        (line for line in response.text.splitlines() if line.startswith("TOTAL")),
        "(no TOTAL line)",
    )
    print(f"\n=== effort={level} ===")
    print(f"  time          : {seconds:5.1f} s")
    print(f"  output tokens : {response.output_tokens}")
    print(f"  thinking blocks: {len(thinking_blocks)}")
    print(f"  stop_reason   : {response.stop_reason}")
    print(f"  {total_line}")
    excerpt = response.text.strip().replace("\n", " ")[:240]
    print(f"  answer starts : {excerpt}...")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")
    print("Same question, three effort levels. Watch tokens and time grow.")
    for level in LEVELS:
        run(level)
    print(
        "\nOutput tokens include the hidden thinking. If every level gets the "
        "same TOTAL, the lowest one is the right setting for this task."
    )
