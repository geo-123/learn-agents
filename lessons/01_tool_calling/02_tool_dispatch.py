"""Lesson 01 — the load-bearing dispatch pattern.

This file is the heart of the lesson. Read it twice.

The pattern:
    1. Every tool is registered with a pydantic model for its input.
    2. dispatch() validates the model's emitted args against that model.
    3. Validation errors come back to the model as is_error tool_results,
       so the model can recover instead of your process crashing.
    4. The agent loop is short, generic, and works for any tool set.

Run:
    uv run python lessons/01_tool_calling/02_tool_dispatch.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


# --- Tool definitions ------------------------------------------------------
# A tool is a name + a pydantic model for its input + a description + a fn.
# Pydantic's .model_json_schema() gives us the JSON schema for free.


class GetWeatherArgs(BaseModel):
    city: str = Field(
        description="City name. Supported: Lisbon, Tokyo, New York."
    )


def get_weather(args: GetWeatherArgs) -> dict:
    data = {
        "Lisbon": {"temp_c": 18, "conditions": "partly cloudy"},
        "Tokyo": {"temp_c": 22, "conditions": "rainy"},
        "New York": {"temp_c": 8, "conditions": "sunny"},
    }
    if args.city not in data:
        # Returning a structured error is a *feature*. The model will read it.
        return {"error": f"No data for {args.city!r}. Supported: {list(data)}."}
    return data[args.city]


class ListCitiesArgs(BaseModel):
    pass  # No args.


def list_cities(args: ListCitiesArgs) -> dict:
    return {"cities": ["Lisbon", "Tokyo", "New York"]}


# --- Registry --------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def schema(self) -> dict:
        """Convert to the wire format the API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {
    t.name: t for t in [
        Tool(
            name="get_weather",
            description=(
                "Returns the current weather for a supported city. Call this "
                "whenever the user asks about temperature, conditions, or "
                "weather in a known city."
            ),
            args_model=GetWeatherArgs,
            fn=get_weather,
        ),
        Tool(
            name="list_cities",
            description=(
                "Returns the list of cities that get_weather supports. Call "
                "this if you're unsure whether a city is available."
            ),
            args_model=ListCitiesArgs,
            fn=list_cities,
        ),
    ]
}


# --- The dispatcher --------------------------------------------------------


def dispatch(tool_use: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call. Always returns a well-shaped tool_result block.

    Three error paths, each handled:
        - Unknown tool name (model hallucination).
        - Args don't match the schema (validation error).
        - The tool itself raises an exception.
    """
    name = tool_use["name"]
    use_id = tool_use["id"]
    raw_args = tool_use.get("input", {}) or {}

    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"Unknown tool {name!r}. Available: {list(REGISTRY)}.")

    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as e:
        return _err(use_id, f"Invalid args for {name!r}: {e.errors()}")

    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001 — tools are arbitrary code; surface anything
        return _err(use_id, f"Tool {name!r} raised: {type(e).__name__}: {e}")

    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": json.dumps(result),
    }


def _err(use_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": message,
        "is_error": True,
    }


# --- The agent loop --------------------------------------------------------


def run_agent(task: str, *, max_steps: int = 6) -> str:
    """Run until the model stops calling tools (or we hit max_steps)."""
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop_reason={response.stop_reason} ---")
        if response.text:
            print(f"text: {response.text}")
        for tu in response.tool_uses:
            print(f"tool_use: {tu['name']}({tu['input']})")

        # Stop condition: no more tool calls. The text is our answer.
        if response.stop_reason != "tool_use":
            return response.text

        # Dispatch every tool_use in this turn. One tool_result per use.
        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

    print(f"\n[stop] hit max_steps={max_steps}")
    return ""


if __name__ == "__main__":
    # Try changing the question. Try asking about Paris (an unknown city) —
    # watch the model recover by calling list_cities first.
    final = run_agent(
        "What's the weather like in Tokyo and Paris? If Paris isn't supported, "
        "just tell me what is."
    )
    print(f"\n=== final answer ===\n{final}")
