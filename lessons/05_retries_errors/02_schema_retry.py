"""Lesson 05: model output retries with validation feedback.

When the model's output fails validation, don't crash and don't retry
blindly. Show the model what it got wrong and ask it to fix it. Cap
attempts. The pattern from Lesson 02 lifted out as a reusable helper.

Structured outputs (Lesson 02) guarantee the schema's *shape*. They do not
enforce what the schema subset cannot express: string patterns, lengths,
numeric ranges. The SDK moves those into the field descriptions as hints,
and Pydantic checks them here. Those are the failures this helper repairs.

Run:
    uv run python lessons/05_retries_errors/02_schema_retry.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Type, TypeVar

from anthropic import transform_schema
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
    """Structured output shaped like `schema`; repair on validation errors."""
    # The API enforces the shape; our full Pydantic validation (patterns,
    # lengths, ranges) runs on the JSON it returns.
    shape = transform_schema(schema.model_json_schema())
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    last_err: ValidationError | None = None
    for attempt in range(1, max_attempts + 1):
        response = chat(messages, system=system, output_schema=shape, effort="low")
        # Keep the model's previous output in history so it sees what it tried.
        messages.append({"role": "assistant", "content": response.content})

        if response.parsed is None:
            # Refusal or a cut off answer: nothing to repair. Treat as terminal.
            raise RuntimeError(f"No structured answer (stop_reason={response.stop_reason})")

        try:
            return schema.model_validate(response.parsed)
        except ValidationError as e:
            last_err = e
            print(f"  attempt {attempt}: validation failed: {e.error_count()} errors")
            messages.append({
                "role": "user",
                "content": (
                    "Validation failed with the following errors:\n"
                    f"{e.errors(include_url=False)}\n\n"
                    "Answer again with corrected values. Pay close "
                    "attention to formats, lengths and allowed ranges."
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
    # Inputs that often produce first pass errors: vague dates, durations
    # in noninteger forms, titles too long.
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
