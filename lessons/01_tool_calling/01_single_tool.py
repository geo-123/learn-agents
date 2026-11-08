"""Lesson 01: one tool, one turn.

The simplest possible tool calling demo. We define ONE tool (get_weather),
hand it to the model, and watch the two turn exchange:

    turn 1: user asks -> model emits tool_use
    turn 2: we send tool_result -> model produces text answer

Run:
    uv run python lessons/01_tool_calling/01_single_tool.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


# A "tool" on the wire is just this dict. Name, description, input schema.
# The description is a prompt: write it like one.
GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": (
        "Returns the current weather for a city. Only supports Lisbon, "
        "Tokyo, and New York. Use this whenever the user asks about "
        "temperature, conditions, or weather."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name. Must be one of: Lisbon, Tokyo, New York.",
                "enum": ["Lisbon", "Tokyo", "New York"],
            },
        },
        "required": ["city"],
    },
}


def get_weather(city: str) -> dict:
    """The actual implementation. In real life this would call an API."""
    data = {
        "Lisbon": {"temp_c": 18, "conditions": "partly cloudy"},
        "Tokyo": {"temp_c": 22, "conditions": "rainy"},
        "New York": {"temp_c": 8, "conditions": "sunny"},
    }
    return data[city]


def main() -> None:
    messages: list[dict] = [
        {"role": "user", "content": "What's the weather like in Lisbon right now?"}
    ]

    # Turn 1: model decides whether to call the tool.
    print("--- turn 1: sending user question ---")
    r1 = chat(messages, tools=[GET_WEATHER_TOOL])
    print(f"stop_reason={r1.stop_reason}")
    print(f"content={json.dumps(r1.content, indent=2)}")
    messages.append({"role": "assistant", "content": r1.content})

    # The loop's job: if there's a tool_use, we run it and feed the result back.
    if r1.stop_reason != "tool_use":
        print("\n(Model answered without calling a tool. Done.)")
        return

    tool_use = r1.tool_uses[0]
    print(f"\n--- executing tool: {tool_use['name']}({tool_use['input']}) ---")
    result = get_weather(**tool_use["input"])
    print(f"result: {result}")

    # tool_result MUST reference the tool_use_id. The content is stringified.
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": json.dumps(result),
        }],
    })

    # Turn 2: model sees the result and produces a natural language answer.
    print("\n--- turn 2: sending tool result ---")
    r2 = chat(messages, tools=[GET_WEATHER_TOOL])
    print(f"stop_reason={r2.stop_reason}")
    print(f"\nfinal answer:\n{r2.text}")


if __name__ == "__main__":
    main()
