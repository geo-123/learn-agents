"""Lesson 05 — let the model handle tool errors.

flaky_search() fails 50% of the time. We DON'T auto-retry inside the
dispatcher. We return the error to the model and let it decide whether
to retry, reformulate, or fall back. Watch the trace — the model often
reasons its way out.

Run:
    uv run python lessons/05_retries_errors/03_tool_error_loop.py
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


random.seed(42)  # deterministic flakiness


class SearchArgs(BaseModel):
    query: str = Field(description="The search query.")


def flaky_search(args: SearchArgs) -> dict:
    """Pretends to search but fails 50% of the time. Don't retry inside —
    we want the *model* to see the error and react."""
    if random.random() < 0.5:
        raise RuntimeError("upstream search service returned 503")
    fake_results = {
        "lisbon": ["The Time Out Market in Lisbon is famous."],
        "tokyo": ["Tokyo has the world's busiest train system."],
        "default": [f"Some result about {args.query}"],
    }
    key = args.query.lower().split()[0] if args.query else "default"
    return {"results": fake_results.get(key, fake_results["default"])}


class CachedFactArgs(BaseModel):
    topic: str = Field(description="One word: 'lisbon', 'tokyo', etc.")


def cached_fact(args: CachedFactArgs) -> dict:
    """A reliable fallback the model can use when search keeps failing."""
    facts = {
        "lisbon": "Lisbon is the capital of Portugal.",
        "tokyo": "Tokyo is the capital of Japan.",
    }
    return {"fact": facts.get(args.topic.lower(), "no cached fact")}


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
    Tool("flaky_search",
         "Search the web. May intermittently fail with an upstream error — "
         "if it fails, you can retry once or fall back to cached_fact.",
         SearchArgs, flaky_search),
    Tool("cached_fact",
         "Look up a basic cached fact about a topic. Less rich than search "
         "but always works. Good fallback when search is failing.",
         CachedFactArgs, cached_fact),
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
        # Tool errors come back to the model, NOT raised. The model
        # then decides what to do.
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


SYSTEM = (
    "You are a research assistant. Prefer flaky_search for general queries. "
    "If it fails repeatedly, fall back to cached_fact. Always end with a "
    "one-paragraph answer. Be transparent about which tools succeeded."
)


def run_agent(task: str, *, max_steps: int = 8) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        if response.text:
            print(f"text: {response.text[:200]}")
        for tu in response.tool_uses:
            print(f"  -> {tu['name']}({tu['input']})")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        for r in results:
            if r.get("is_error"):
                print(f"  (tool error returned to model: {r['content']})")
        messages.append({"role": "user", "content": results})

    return "(max steps)"


if __name__ == "__main__":
    answer = run_agent("Tell me one interesting thing about Lisbon.")
    print(f"\n=== final ===\n{answer}")
