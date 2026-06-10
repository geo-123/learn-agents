"""Lesson 02 — naive prompt-and-parse extraction.

Ask the model to return JSON in its text. Parse with json.loads. Watch
some inputs work, some fail. The failure modes are the lesson.

Run:
    uv run python lessons/02_structured_outputs/01_extract_prompt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


PROMPT = """\
Extract the following information from the text below. Return ONLY a JSON
object with these fields:

  name (string), company (string), email (string), seniority (string).

Text:
{text}
"""


def extract(text: str) -> dict | None:
    response = chat([{"role": "user", "content": PROMPT.format(text=text)}])
    raw = response.text.strip()

    # The model often wraps JSON in markdown fences. Strip them.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ! json parse failed: {e}")
        print(f"  raw was: {response.text!r}")
        return None


if __name__ == "__main__":
    samples = [
        # Easy
        "John Smith joined our team last month from Google. Reach him at "
        "john.smith@acme.com for engineering questions.",
        # Tricky: ambiguous seniority, multiple emails
        "Dr. Maria Lopez (CTO) used to write me from m.lopez@old.io but now "
        "uses maria@new.co. She runs the platform team at Northwind.",
        # Adversarial: model may add bonus fields, hallucinate, or refuse
        "Anonymous reviewer with no name said the product was good.",
    ]
    for text in samples:
        print(f"\nINPUT: {text}")
        result = extract(text)
        print(f"OUTPUT: {result}")
