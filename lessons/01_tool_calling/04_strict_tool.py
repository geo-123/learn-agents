"""Lesson 01: steering the model to a specific tool, with a strict schema.

Sometimes you don't want a chatty agent; you want the model to call one
particular tool. Older APIs let you *force* that with
tool_choice={"type": "tool", "name": ...}. The newest Claude models
(Opus 5.5, Sonnet 5.5, Fable 5.1) reject a forced tool choice with a 400,
so the portable pattern is:

  1. tool_choice stays "auto" (the default),
  2. the prompt names the tool to use,
  3. the tool sets `strict: true`, so IF the model calls it, the arguments
     are guaranteed to match the schema exactly,
  4. your code checks that a tool_use came back and reprompts once if not.

When all you want is a structured answer (no tool at all), use structured
outputs instead: that is Lesson 02.

Run:
    uv run python lessons/01_tool_calling/04_strict_tool.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from anthropic import transform_schema
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


class SentimentAnalysis(BaseModel):
    """The schema the tool accepts."""
    label: Literal["positive", "neutral", "negative"]
    confidence: float = Field(description="0.0 to 1.0.")
    rationale: str = Field(description="One short sentence explaining the label.")


# Strict tools accept a subset of JSON Schema (`additionalProperties: false`,
# no numeric ranges). The SDK's transform_schema turns a Pydantic schema into
# that subset, moving constraints it cannot express into the description.
SCHEMA = transform_schema(SentimentAnalysis.model_json_schema())

SUBMIT_ANALYSIS = {
    "name": "submit_analysis",
    "description": "Submit your sentiment analysis of the given text.",
    "input_schema": SCHEMA,
    "strict": True,
}

SYSTEM = "You analyze sentiment. Always answer by calling the submit_analysis tool, never in plain text."


def analyze(text: str) -> SentimentAnalysis:
    """Return a typed object. The schema is guaranteed; the call is steered."""
    messages: list[dict] = [{"role": "user", "content": f"Analyze this text:\n\n{text}"}]
    for attempt in (1, 2):
        response = chat(messages=messages, system=SYSTEM, tools=[SUBMIT_ANALYSIS])
        if response.tool_uses:
            # strict: true means this validates; Pydantic still gives us a typed object
            # and catches anything the schema cannot express (the 0 to 1 range).
            result = SentimentAnalysis.model_validate(response.tool_uses[0]["input"])
            if not 0.0 <= result.confidence <= 1.0:
                raise ValueError(f"confidence out of range: {result.confidence}")
            return result
        # auto does not guarantee a call. Reprompt once, appending (never editing) history.
        print(f"  (attempt {attempt}: no tool call, reprompting)")
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": "Please call submit_analysis with your answer."},
        ]
    raise RuntimeError("model did not call submit_analysis")


if __name__ == "__main__":
    samples = [
        "I loved this product. Fast delivery, exactly as described.",
        "Got the order; haven't tried it yet.",
        "Worst purchase of the year. Save your money.",
    ]
    for text in samples:
        result = analyze(text)
        print(f"\nText: {text!r}")
        print(f"  -> {result.label}  ({result.confidence:.2f}): {result.rationale}")
