"""Lesson 04b: context editing clears old tool results on the server.

A tool heavy loop: the agent reads six long ticket logs, one per turn, and
counts the P1 tickets in each. Every log is about 2,000 tokens and stays
in the history, so each request is bigger than the last.

The script runs the same task twice:
  1. plain: nothing is cleared, input tokens climb every turn;
  2. with `clear_tool_uses_20250919`: once more than 3 tool uses are in
     the prompt, the API replaces the oldest results with a placeholder
     and keeps the last 2. Input tokens level off.

Things to notice:
  * your `messages` list is never changed. The clearing happens on the
    server, on its copy, before the model reads it. Your history stays
    append only (important for thinking blocks, see the README);
  * `response.context_management.applied_edits` reports what was cleared;
  * clearing is not summarizing. A cleared log is gone. The system prompt
    asks the agent to write each count down as text right after reading,
    because text survives and old tool results do not.

Model: claude-haiku-5-5, the cheapest model the context editing docs list.

Run:
    uv run python lessons/04b_long_running_context/02_clear_tool_results.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import client

MODEL = "claude-haiku-5-5"
TEAMS = ["payments", "search", "mobile", "identity", "billing", "infra"]

CLEAR_OLD_RESULTS: dict[str, Any] = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "tool_uses", "value": 3},  # start clearing once 3 tool uses are in the prompt
    "keep": {"type": "tool_uses", "value": 2},     # always keep the 2 most recent
}

TOOLS: Any = [{
    "name": "fetch_ticket_log",
    "description": "Return the full ticket log of one team for last week. Long output.",
    "input_schema": {
        "type": "object",
        "properties": {"team": {"type": "string", "enum": TEAMS}},
        "required": ["team"],
    },
}]

SYSTEM = (
    "You audit ticket logs. Fetch one team's log per turn. Right after each "
    "fetch, write one line 'team: N P1 tickets' before fetching the next. "
    "When all teams are done, name the team with the most P1 tickets."
)


def ticket_log(team: str) -> str:
    """A deterministic fake log, roughly 2,000 tokens."""
    rng = random.Random(team)
    lines = []
    for n in range(80):
        prio = rng.choice(["P1", "P2", "P3", "P3", "P4"])
        lines.append(f"2026-11-0{n % 7 + 1} {team} #{1000 + n} {prio} status={rng.choice(['open', 'closed'])}")
    return "\n".join(lines)


def run(edits: list[dict[str, Any]] | None) -> None:
    label = "with clearing" if edits else "plain"
    print(f"\n=== {label} ===")
    messages: list[Any] = [{"role": "user", "content": f"Audit these teams: {', '.join(TEAMS)}."}]
    extra: dict[str, Any] = {"context_management": {"edits": edits}} if edits else {}
    for turn in range(1, 12):
        response = client().beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=TOOLS,
            betas=["context-management-2025-06-27"],
            output_config={"effort": "low"},
            messages=messages,
            **extra,
        )
        messages.append({"role": "assistant", "content": response.content})
        cleared = ""
        if response.context_management and response.context_management.applied_edits:
            edit = response.context_management.applied_edits[0]
            cleared = f"  cleared {edit.cleared_input_tokens} tokens"
        print(f"turn {turn:2d}  input_tokens={response.usage.input_tokens:6d}{cleared}")

        uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not uses:
            text = "".join(b.text for b in response.content if b.type == "text")
            print(f"answer: {text.strip()[:300]}")
            return
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u.id, "content": ticket_log(str(u.input["team"]))}
            for u in uses
        ]})


if __name__ == "__main__":
    truth = {t: ticket_log(t).count(" P1 ") for t in TEAMS}
    print(f"ground truth P1 counts: {truth}")
    run(None)
    run([CLEAR_OLD_RESULTS])
    print("\nSame task, same answer (hopefully), fewer input tokens on the later turns.")
