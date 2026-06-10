"""Lesson 02 — structured extraction via forced tool call.

Same task as 01_extract_prompt.py, but using a forced tool call. Compare:
  - schema is enforced by the protocol, not by hope
  - typed object out, ready to use
  - the failure mode is now ValidationError (loud) not silent wrong-shape

Run:
    uv run python lessons/02_structured_outputs/02_extract_tool.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class Contact(BaseModel):
    """What we want to extract — same shape as the prompt-based version,
    but now with real constraints. The model reads this schema as a prompt."""
    name: str = Field(description="Full name. Use the most recent name mentioned.")
    company: str | None = Field(
        default=None,
        description="Current employer. null if not mentioned.",
    )
    email: str | None = Field(
        default=None,
        description="Most recent email address mentioned. null if none.",
    )
    seniority: Literal["junior", "mid", "senior", "exec", "unknown"] = Field(
        description=(
            "Inferred seniority. 'exec' for C-level/VP. 'unknown' if the text "
            "gives no signal at all — do NOT guess."
        ),
    )


SUBMIT_TOOL = {
    "name": "submit_contact",
    "description": "Submit the extracted contact information.",
    "input_schema": Contact.model_json_schema(),
}


def extract(text: str) -> Contact:
    response = chat(
        messages=[{"role": "user", "content": f"Extract a contact from:\n\n{text}"}],
        tools=[SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": "submit_contact"},
    )
    assert response.tool_uses, "Forced tool call should always produce a tool_use"
    return Contact.model_validate(response.tool_uses[0]["input"])


if __name__ == "__main__":
    samples = [
        "John Smith joined our team last month from Google. Reach him at "
        "john.smith@acme.com for engineering questions.",
        "Dr. Maria Lopez (CTO) used to write me from m.lopez@old.io but now "
        "uses maria@new.co. She runs the platform team at Northwind.",
        "Anonymous reviewer with no name said the product was good.",
    ]
    for text in samples:
        print(f"\nINPUT: {text}")
        try:
            result = extract(text)
            print(f"OUTPUT: {result.model_dump_json(indent=2)}")
        except Exception as e:
            print(f"  ! extraction failed: {type(e).__name__}: {e}")
