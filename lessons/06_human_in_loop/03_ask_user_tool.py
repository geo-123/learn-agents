"""Lesson 06: ask_user as a distinct primitive.

ask_user is NOT approval. It's the model requesting missing information.
The agent pauses the loop, prompts the human, and resumes with the
answer as a tool_result. This is different from approval: there's no
proposal to approve, just a question to answer.

Run:
    uv run python lessons/06_human_in_loop/03_ask_user_tool.py
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


class AskUserArgs(BaseModel):
    question: str = Field(
        description=(
            "A question for the human. One question at a time. "
            "Be specific and avoid yes/no when an open answer is more useful."
        ),
    )


def ask_user(args: AskUserArgs) -> dict:
    """The harness pauses for human input. In production this would be
    a UI element / Slack / email."""
    print(f"\n[agent asks] {args.question}")
    answer = input("you> ").strip()
    return {"answer": answer}


class SubmitItineraryArgs(BaseModel):
    destination: str
    days: int = Field(gt=0)
    daily_budget_usd: int = Field(gt=0)
    interests: list[str]


def submit_itinerary(args: SubmitItineraryArgs) -> dict:
    return {"received": args.model_dump()}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("ask_user",
         "Ask the user a single clarifying question and get the answer.",
         AskUserArgs, ask_user),
    Tool("submit_itinerary",
         "Submit the final trip plan once you have all the details.",
         SubmitItineraryArgs, submit_itinerary),
]}


def dispatch(tool_use: dict) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"unknown tool {name!r}")
    try:
        args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
        result = tool.fn(args)
    except ValidationError as e:
        return _err(use_id, f"invalid args: {e.errors()}")
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


SYSTEM = (
    "You are a travel planner. Before submitting an itinerary, ensure you "
    "have: destination, number of days, daily budget in USD, and interests. "
    "Use ask_user ONE clarifying question at a time when missing info. "
    "Once you have enough, call submit_itinerary."
)


def run_agent(task: str, *, max_steps: int = 8) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        for tu in response.tool_uses:
            print(f"  -> {tu['name']}")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

        # If the model called submit_itinerary, we can stop: the agent is done.
        if any(tu["name"] == "submit_itinerary" for tu in response.tool_uses):
            return "itinerary submitted"

    return "(max steps)"


if __name__ == "__main__":
    final = run_agent("Plan me a trip to Portugal.")
    print(f"\n=== final ===\n{final}")
