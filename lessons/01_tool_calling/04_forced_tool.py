"""Lesson 01 — forcing a specific tool call.

Sometimes you don't want a chatty agent; you want a structured answer.
By forcing the model to call a specific tool, you guarantee the response
shape — the tool becomes a typed mailbox the model has to drop its answer in.

This is the bridge to Lesson 02 (structured outputs). The tool isn't there
to *do* anything; it's there to give the model a place to put the answer.

Run:
    uv run python lessons/01_tool_calling/04_forced_tool.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class SentimentAnalysis(BaseModel):
    """The schema we want the model to fill in."""
    label: str = Field(description="One of: positive, neutral, negative.")
    confidence: float = Field(description="0.0 to 1.0.", ge=0.0, le=1.0)
    rationale: str = Field(description="One short sentence explaining the label.")


# We expose this as a tool, but its purpose is structured extraction.
# tool_choice will force the model to call it.
SUBMIT_ANALYSIS = {
    "name": "submit_analysis",
    "description": "Submit your sentiment analysis of the given text.",
    "input_schema": SentimentAnalysis.model_json_schema(),
}


def analyze(text: str) -> SentimentAnalysis:
    """Return a typed object — guaranteed shape, no parsing."""
    response = chat(
        messages=[{"role": "user", "content": f"Analyze this text:\n\n{text}"}],
        tools=[SUBMIT_ANALYSIS],
        tool_choice={"type": "tool", "name": "submit_analysis"},
    )

    # With forced tool choice, we know there's exactly one tool_use block.
    assert response.tool_uses, "Forced tool call should produce a tool_use block"
    raw = response.tool_uses[0]["input"]

    # Pydantic validates and gives us a typed object. If the model lied
    # about the shape, this raises immediately. Lesson 02 covers the
    # repair pattern when it does.
    return SentimentAnalysis.model_validate(raw)


if __name__ == "__main__":
    samples = [
        "I loved this product. Fast delivery, exactly as described.",
        "Got the order; haven't tried it yet.",
        "Worst purchase of the year. Save your money.",
    ]
    for text in samples:
        result = analyze(text)
        print(f"\nText: {text!r}")
        print(f"  -> {result.label}  ({result.confidence:.2f})  — {result.rationale}")
