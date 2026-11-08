"""Lesson 02: nested schemas and schema design.

A real extraction usually isn't one flat record: it's a tree.
A meeting has attendees and agenda items, each with their own fields.

This file exercises three design choices:
  1. Lists of nested objects (attendees, agenda items)
  2. Enums to discipline open ended fields (priority)
  3. Optional fields with explicit nulls when the source is silent

Run:
    uv run python lessons/02_structured_outputs/03_nested_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class Attendee(BaseModel):
    name: str = Field(description="Full name as written in the source.")
    role: str | None = Field(
        default=None,
        description="Role/title if mentioned. null if not.",
    )


class AgendaItem(BaseModel):
    title: str = Field(description="One line item summary.")
    owner: str | None = Field(
        default=None,
        description="Name of the person responsible. null if no owner is named.",
    )
    priority: Literal["high", "medium", "low"] = Field(
        description=(
            "Inferred priority. Use 'high' for items explicitly flagged urgent, "
            "blocker, or critical. 'low' for nice to haves. 'medium' otherwise."
        ),
    )


class Meeting(BaseModel):
    title: str = Field(description="The meeting's title or topic.")
    date: str | None = Field(
        default=None,
        description="ISO 8601 date if mentioned, e.g. 2025-01-15. null otherwise.",
    )
    attendees: list[Attendee]
    agenda: list[AgendaItem]


def extract(text: str) -> Meeting:
    response = chat(
        messages=[{"role": "user", "content": f"Extract a meeting from:\n\n{text}"}],
        output_schema=Meeting,  # nested models and lists work the same way
    )
    if response.parsed is None:
        raise RuntimeError(f"no structured answer (stop_reason={response.stop_reason})")
    return response.parsed


SAMPLE = """\
2025-03-14 product sync. Present: Maria Lopez (CTO), John Smith (Eng),
Priya Rao (Design). Agenda: (1) Auth migration: Maria, blocker for Q2.
(2) Mobile redesign mocks: Priya, nice to have for next sprint.
(3) Cost review: owner TBD, regular cadence.
"""


if __name__ == "__main__":
    print("INPUT:")
    print(SAMPLE)
    meeting = extract(SAMPLE)
    print("OUTPUT:")
    print(meeting.model_dump_json(indent=2))
