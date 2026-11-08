"""Lesson 02: the repair loop.

Structured outputs guarantee the *shape* the model sees. They cannot
guarantee rules the schema does not express: business rules, cross field
checks, formats your code needs but the schema leaves loose. When your
validator rejects an answer, don't crash: feed the error back to the model
and ask it to try again. Cap attempts so a stubborn model can't burn the budget.

This is the smallest example of "self correcting" agents. Lesson 05
generalizes the pattern across other failure modes.

Run:
    uv run python lessons/02_structured_outputs/04_repair_loop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class TimeRange(BaseModel):
    """Strict validator. The model never sees these patterns: see LOOSE_SCHEMA below.

    The shapes fight the model's training priors: 4 digit times with no
    separator (vs. the universal HH:MM) and short lowercase tz codes
    (vs. IANA "Europe/Lisbon"). On attempt 1 the model emits the
    conventional shape, fails validation, then corrects from the error.
    """
    start: str = Field(pattern=r"^\d{4}$")
    end: str = Field(pattern=r"^\d{4}$")
    timezone: Literal["utc", "lisbon", "nyc", "tokyo"]


# Loose schema the model is constrained to: types only, no patterns, no enum,
# vague descriptions. The API guarantees this shape; our stricter client side
# validator decides whether the content is acceptable.
LOOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["start", "end", "timezone"],
    "additionalProperties": False,
    "properties": {
        "start": {"type": "string", "description": "Start time."},
        "end": {"type": "string", "description": "End time."},
        "timezone": {"type": "string", "description": "Timezone."},
    },
}


def extract_with_repair(text: str, *, max_attempts: int = 3) -> TimeRange:
    """Try, repair, retry. Up to max_attempts before giving up."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Extract the time range from:\n\n{text}"}
    ]

    last_err: ValidationError | None = None
    for attempt in range(1, max_attempts + 1):
        response = chat(messages=messages, output_schema=LOOSE_SCHEMA, effort="low")
        # Keep the model's previous attempt in history so it sees what it just did.
        # Append, never edit: current models bind their thinking to the exact history.
        messages.append({"role": "assistant", "content": response.content})

        raw = response.parsed or {}
        print(f"  attempt {attempt} produced {raw}")
        try:
            result = TimeRange.model_validate(raw)
            print(f"  attempt {attempt} VALIDATED")
            return result
        except ValidationError as e:
            last_err = e
            # Compact one line per error summary; full Pydantic errors are noisy.
            issues = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            print(f"  attempt {attempt} FAILED: {issues}")

            # No tool_use to answer here: the feedback is a plain user turn.
            messages.append({
                "role": "user",
                "content": (
                    "Your answer failed validation with these errors:\n"
                    f"{json.dumps(e.errors(include_url=False), default=str)}\n\n"
                    "Answer again with corrected values."
                ),
            })

    assert last_err is not None
    raise RuntimeError(
        f"Could not extract a valid TimeRange in {max_attempts} attempts: {last_err}"
    )


if __name__ == "__main__":
    # Inputs chosen to be ambiguous or off format: the model often gets it
    # wrong on the first try, then fixes itself.
    samples = [
        "Office hours: 9 AM to 5 PM, Lisbon time.",
        "Meeting from 14:00 to 15:30 UTC.",
        "From 8 in the morning until noon in New York.",
    ]
    for text in samples:
        print(f"\nINPUT: {text}")
        try:
            r = extract_with_repair(text)
            print(f"  -> {r.model_dump()}")
        except RuntimeError as e:
            print(f"  ! gave up: {e}")
