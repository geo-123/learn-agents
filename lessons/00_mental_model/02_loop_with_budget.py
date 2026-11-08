"""Lesson 00: the loop, with layered stop conditions.

01_loop.py used a single, fabricated stop signal: the word DONE in the
response. That works, but real loops need defense in depth. This file
shows four kinds of stop, in priority order:

  1. Budget (hard stop): max steps / max seconds / max tokens.
                                  The ANTI RUNAWAY defense. Always have one.
  2. stop_reason == "end_turn": the GRACEFUL, model initiated signal.
                                  Universal across providers (see note below).
                                  It means "my turn is over", not "the task
                                  is done". This demo asks for one item per
                                  turn, so every turn ends with end_turn; here
                                  it stops the loop only together with DONE.
  3. Stall detector: same text two turns in a row.
                                  Catches "stuck" loops the model can't
                                  recognize itself.
  4. DONE keyword (kept from 01): a task specific, explicit signal.
                                  The marker that tells "turn over" apart
                                  from "task complete" when, as here, the
                                  loop itself keeps asking for more turns.

Note on cross provider portability of `stop_reason`:
    Anthropic    : stop_reason = "end_turn" | "max_tokens" | "tool_use" | "stop_sequence"
    OpenAI       : finish_reason = "stop"   | "length"     | "tool_calls" | "content_filter"
    Google Gemini: finish_reason = "STOP"   | "MAX_TOKENS" | "SAFETY"     | ...
The *names* differ; the *concept* is universal: every chat completion
API tells you why generation stopped. "end_turn"/"stop"/"STOP" is the
graceful "I'm done speaking for now" signal. "max_tokens"/"length"
means the response was TRUNCATED: don't treat that as done.

Run:
    uv run python lessons/00_mental_model/02_loop_with_budget.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


SYSTEM = (
    "You are a brainstorming agent. RULES: on each turn, propose exactly ONE "
    "item with a one sentence rationale, then stop and wait for the user to "
    "say 'Continue.' Do not list multiple items in a single turn. "
    "When and only when you have produced the number requested, end your "
    "final message with the word DONE on its own line."
)


@dataclass
class Budget:
    """Stop condition aggregator. Any field tripping ends the loop."""
    max_steps: int = 5
    max_input_tokens: int = 20_000
    max_seconds: float = 30.0

    steps_used: int = 0
    input_tokens_used: int = 0
    started_at: float = 0.0

    def start(self) -> None:
        self.started_at = time.monotonic()

    def exhausted(self) -> tuple[bool, str]:
        if self.steps_used >= self.max_steps:
            return True, f"max_steps={self.max_steps}"
        if self.input_tokens_used >= self.max_input_tokens:
            return True, f"max_input_tokens={self.max_input_tokens}"
        if time.monotonic() - self.started_at >= self.max_seconds:
            return True, f"max_seconds={self.max_seconds}"
        return False, ""


def run_agent(task: str, budget: Budget) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": task}]
    budget.start()
    previous_text: str | None = None  # for the stall detector

    while True:
        # --- Stop check 1: budget (hard) -----------------------------------
        # Cheapest to detect; aborts BEFORE spending another API call.
        done, reason = budget.exhausted()
        if done:
            print(f"\n[stop] budget exhausted: {reason}")
            return messages

        response = chat(messages, system=SYSTEM)
        messages.append({"role": "assistant", "content": response.content})

        budget.steps_used += 1
        budget.input_tokens_used += response.input_tokens

        print(f"\n--- step {budget.steps_used}  stop_reason={response.stop_reason} ---")
        print(response.text)
        print(
            f"(this turn: in={response.input_tokens} out={response.output_tokens}  "
            f"cumulative in={budget.input_tokens_used})"
        )

        # --- Stop check 2: max_tokens (response truncated) -----------------
        # NOT a "done" signal. The response was cut off. If we accepted this
        # as graceful, we'd silently return half an answer. Surface it.
        if response.stop_reason == "max_tokens":
            print("\n[warn] response was truncated (max_tokens). Continuing.")
            # In a real agent you'd either raise OR raise max_tokens and
            # retry the same turn. For the lesson we just keep going.

        # --- Stop check 3: end_turn plus DONE (graceful) ------------------
        # end_turn means the model finished its TURN, not the task. In a tool
        # loop (Lesson 01) that difference rarely matters: a turn that ends
        # without a tool call is the answer, so end_turn alone stops the loop.
        # This loop has no tools and the system prompt asks for one item per
        # turn, so EVERY turn ends with end_turn, and this loop answers each
        # one with "Continue.". end_turn alone would stop after the first
        # item; the DONE marker says the task is complete. Both must hold:
        # DONE in a truncated (max_tokens) turn does not count.
        if response.stop_reason == "end_turn" and response.text.strip().endswith("DONE"):
            print("\n[stop] end_turn + assistant said DONE  (clean finish)")
            return messages

        # --- Stop check 4: stall detector ----------------------------------
        # Two identical turns in a row means we're not making progress.
        # Without this, "Continue." can loop a "stuck" model forever
        # (until budget exhaustion, but that's expensive).
        current = response.text.strip()
        if previous_text is not None and current == previous_text:
            print("\n[stop] stall detected: response identical to previous turn")
            return messages
        previous_text = current

        # The task is not complete yet: nudge the model to continue.
        messages.append({"role": "user", "content": "Continue."})


if __name__ == "__main__":
    # The task asks for MORE items than max_steps allows, so the budget's
    # hard cap fires before the model can say DONE. You should see step 1..4,
    # each with one name, then `[stop] budget exhausted: max_steps=4`.
    # To see the graceful end_turn path instead, raise max_steps to 7: the
    # model will produce all 6 names, say DONE, and stop_reason="end_turn"
    # ends the loop before the budget trips.
    task = "Propose 6 names for a new note taking app aimed at developers."
    run_agent(task, Budget(max_steps=4, max_input_tokens=10_000, max_seconds=60))
