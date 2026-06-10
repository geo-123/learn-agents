"""Lesson 05 — model-output retries with validation feedback.

When the model's output fails validation, don't crash and don't retry
blindly. Show the model what it got wrong and ask it to fix it. Cap
attempts. The pattern from Lesson 02 lifted out as a reusable helper.

Run:
    uv run python lessons/05_retries_errors/02_schema_retry.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


T = TypeVar("T", bound=BaseModel)


def structured_call(
    prompt: str,
    schema: Type[T],
    *,
    system: str | None = None,
    max_attempts: int = 3,
) -> T:
    """Force a tool call shaped like `schema`; repair on validation errors."""
    tool = {
        "name": "submit",
        "description": f"Submit a valid {schema.__name__}.",
        "input_schema": schema.model_json_schema(),
    }
    messages: list[dict] = [{"role": "user", "content": prompt}]

    last_err: ValidationError | None = None
    for attempt in range(1, max_attempts + 1):
        response = chat(
            messages,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit"},
        )
        # Keep the model's previous output in history so it sees what it tried.
        messages.append({"role": "assistant", "content": response.content})

        if not response.tool_uses:
            # Refusal or weird response. Treat as terminal.
            raise RuntimeError(f"Model declined to call submit: {response.text!r}")

        raw = response.tool_uses[0]["input"]
        try:
            return schema.model_validate(raw)
        except ValidationError as e:
            last_err = e
            print(f"  attempt {attempt}: validation failed — {e.error_count()} errors")
            messages.append({
                "role": "user",
                "content": (
                    "Validation failed with the following errors:\n"
                    f"{e.errors()}\n\n"
                    "Call submit again with corrected arguments. Pay close "
                    "attention to formats, allowed values, and required fields."
                ),
            })

    assert last_err is not None
    raise RuntimeError(f"Could not validate after {max_attempts} attempts: {last_err}")


# --- Demo schema designed to provoke errors -------------------------------


class Event(BaseModel):
    title: str = Field(max_length=60)
    iso_date: str = Field(
        description="Strict ISO 8601 date, YYYY-MM-DD.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    duration_minutes: int = Field(gt=0, le=24 * 60)


if __name__ == "__main__":
    # Inputs that often produce first-pass errors: vague dates, durations
    # in non-integer forms, titles too long.
    samples = [
        "Team lunch on the 15th of next March, an hour and a half.",
        "Quarterly business review with a really long title that goes on and "
        "on and on so that the model will probably truncate it, scheduled "
        "for January 5th 2026, 90 minutes.",
        "Coffee chat next week, 30 min.",
    ]
    for s in samples:
        print(f"\nINPUT: {s}")
        try:
            ev = structured_call(s, Event)
            print(f"  -> {ev.model_dump()}")
        except RuntimeError as e:
            print(f"  ! gave up: {e}")
