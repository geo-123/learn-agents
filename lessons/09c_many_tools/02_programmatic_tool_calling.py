"""Lesson 09c: programmatic tool calling, 30 tool calls in one script.

The task: audit 30 expense reports against the Acme caps (hotel $300 a
night, $400 in NYC, London, Tokyo; meals $80 a day). With ordinary tool
use that is 30 round trips and 30 JSON records in the model's context.

Here the model gets the code execution tool, and `get_expense` carries
`allowed_callers: ["code_execution_20260120"]`. The model writes one
Python script that loops over the ids and awaits `get_expense(...)` for
each. Every call still runs in YOUR process: the container pauses, the
API returns a `tool_use` whose `caller` names the running code, you send
the result back, the code resumes. The results go to the code, not into
the model's context. Only what the script prints comes back to the model.

Things to notice:
  * `caller` on each tool_use: `code_execution_20260120`, not `direct`;
  * the reply to a pending programmatic call holds tool_result blocks and
    nothing else, and it must name the `container` from the paused response;
  * how few model turns the whole audit takes, compared with 30.

Model: claude-haiku-5-5, the cheapest model the programmatic tool calling
docs list (Haiku 4.5 is not supported).

Run:
    uv run python lessons/09c_many_tools/02_programmatic_tool_calling.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import client

MODEL = "claude-haiku-5-5"
CODE_EXEC = "code_execution_20260120"
CITIES = ["Austin", "NYC", "London", "Denver", "Tokyo", "Chicago"]


def make_records() -> dict[str, dict[str, Any]]:
    rng = random.Random(7)
    records = {}
    for n in range(1, 31):
        rid = f"EXP-{n:03d}"
        nights = rng.randint(1, 4)
        records[rid] = {
            "id": rid, "employee": f"E{rng.randint(100, 999)}", "city": rng.choice(CITIES),
            "nights": nights, "hotel_total": rng.randint(220, 450) * nights,
            "days": nights + 1, "meals_total": rng.randint(50, 95) * (nights + 1),
            "notes": "conference travel; receipts attached; " + "x" * 300,  # bulk the model never needs
        }
    return records


RECORDS = make_records()

TOOLS: Any = [
    {"type": CODE_EXEC, "name": "code_execution"},
    {
        "name": "get_expense",
        "description": "Fetch one expense report by id (EXP-001 to EXP-030). Returns a JSON object with "
                       "city, nights, hotel_total, days, meals_total and notes.",
        "input_schema": {"type": "object", "properties": {"report_id": {"type": "string"}},
                         "required": ["report_id"]},
        "allowed_callers": [CODE_EXEC],  # call it from code, not directly
    },
]


def expected_violations() -> list[str]:
    bad = []
    for r in RECORDS.values():
        cap = 400 if r["city"] in ("NYC", "London", "Tokyo") else 300
        if r["hotel_total"] > cap * r["nights"] or r["meals_total"] > 80 * r["days"]:
            bad.append(r["id"])
    return bad


def main() -> None:
    messages: Any = [{"role": "user", "content": (
        "Audit expense reports EXP-001 to EXP-030. Flag every report whose hotel cost per night is over "
        "the cap ($300, or $400 in NYC, London, Tokyo) or whose meals per day are over $80. "
        "Reply with the flagged ids and the reason for each.")}]
    container: str | None = None
    calls = {"code": 0, "direct": 0}
    for turn in range(1, 25):
        extra: dict[str, Any] = {"container": container} if container else {}
        response = client().messages.create(
            model=MODEL, max_tokens=8000, tools=TOOLS, messages=messages,
            output_config={"effort": "low"}, **extra,
        )
        if response.container:
            container = response.container.id
        messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        for u in uses:
            calls["code" if u.caller and u.caller.type != "direct" else "direct"] += 1
        print(f"turn {turn:2d}  stop={str(response.stop_reason):10s} input_tokens={response.usage.input_tokens:6d}"
              f"  tool calls this turn={len(uses)}  blocks={[b.type for b in response.content]}")

        if response.stop_reason == "pause_turn":  # a long server side turn; send it back to continue
            continue
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\nanswer:\n{text.strip()}")
            break
        # Only tool_result blocks: no text allowed next to pending programmatic results.
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u.id,
             "content": json.dumps(RECORDS.get(str(u.input.get("report_id")), {"error": "unknown id"}))}
            for u in uses
        ]})

    print(f"\ntool calls from code: {calls['code']}   direct: {calls['direct']}")
    print(f"expected flags: {expected_violations()}")


if __name__ == "__main__":
    main()
