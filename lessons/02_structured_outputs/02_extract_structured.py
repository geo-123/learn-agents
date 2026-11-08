"""Lesson 02: structured extraction with native structured outputs.

Same task as 01_extract_prompt.py, but the API constrains the answer to our
schema. Compare:
  - the schema is enforced by the API (constrained decoding), not by hope
  - typed object out, ready to use (`response.parsed`)
  - no tool, no forced tool choice, no JSON parsing in our code

Under the hood `chat(..., output_schema=Contact)` calls the SDK's
`client.messages.parse(output_format=Contact)`, which sends
`output_config={"format": {"type": "json_schema", "schema": ...}}`.

Run:
    uv run python lessons/02_structured_outputs/02_extract_structured.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class Contact(BaseModel):
    """What we want to extract: same shape as the prompt based version,
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
            "Inferred seniority. 'exec' for C level/VP. 'unknown' if the text "
            "gives no signal at all: do NOT guess."
        ),
    )


def extract(text: str) -> Contact:
    response = chat(
        messages=[{"role": "user", "content": f"Extract a contact from:\n\n{text}"}],
        output_schema=Contact,
        effort="low",  # extraction is easy; low effort is faster and cheaper
    )
    if response.refused or response.parsed is None:
        # The schema guarantee holds for a finished answer. A refusal, or an
        # answer cut off at max_tokens, has nothing to parse: check stop_reason.
        raise RuntimeError(f"no structured answer (stop_reason={response.stop_reason})")
    return response.parsed


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
