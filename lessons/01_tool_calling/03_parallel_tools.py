"""Lesson 01: parallel tool calls.

A capable model batches independent tool calls into one turn. We give it
three tools and ask a question that needs all three. Watch the response:
one assistant turn with three tool_use blocks, followed by one user turn
with three tool_result blocks.

Run:
    uv run python lessons/01_tool_calling/03_parallel_tools.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


# --- Tools -----------------------------------------------------------------


class GetWeatherArgs(BaseModel):
    city: str = Field(description="One of: Lisbon, Tokyo, New York.")


def get_weather(args: GetWeatherArgs) -> dict:
    data = {
        "Lisbon": {"temp_c": 18, "conditions": "partly cloudy"},
        "Tokyo": {"temp_c": 22, "conditions": "rainy"},
        "New York": {"temp_c": 8, "conditions": "sunny"},
    }
    return data.get(args.city, {"error": f"unknown city {args.city!r}"})


class GetTimeArgs(BaseModel):
    timezone: str = Field(description="IANA timezone, e.g. Europe/Lisbon, Asia/Tokyo.")


def get_time(args: GetTimeArgs) -> dict:
    try:
        now = datetime.now(ZoneInfo(args.timezone))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"iso": now.isoformat(timespec="seconds"), "timezone": args.timezone}


class ConvertCurrencyArgs(BaseModel):
    amount: float
    from_: str = Field(alias="from", description="3 letter ISO currency code.")
    to: str = Field(description="3 letter ISO currency code.")


def convert_currency(args: ConvertCurrencyArgs) -> dict:
    # Fake rates; this is a teaching demo, not Bloomberg.
    rates = {("EUR", "JPY"): 165.0, ("EUR", "USD"): 1.07, ("USD", "EUR"): 0.93}
    rate = rates.get((args.from_.upper(), args.to.upper()))
    if rate is None:
        return {"error": f"no rate for {args.from_}->{args.to}"}
    return {"converted": round(args.amount * rate, 2), "rate": rate}


# --- Registry / dispatch / loop (same shape as 02_tool_dispatch.py) --------


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
            "input_schema": self.args_model.model_json_schema(by_alias=True),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("get_weather",
         "Get weather for Lisbon, Tokyo, or New York.",
         GetWeatherArgs, get_weather),
    Tool("get_time",
         "Get the current time in an IANA timezone.",
         GetTimeArgs, get_time),
    Tool("convert_currency",
         "Convert an amount from one currency to another.",
         ConvertCurrencyArgs, convert_currency),
]}


def dispatch(tool_use: dict[str, Any]) -> dict[str, Any]:
    name = tool_use["name"]
    use_id = tool_use["id"]
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


def _err(use_id: str, msg: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


def run_agent(task: str, *, max_steps: int = 4) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop_reason={response.stop_reason} ---")
        if response.text:
            print(f"text: {response.text}")
        for tu in response.tool_uses:
            print(f"  tool_use: {tu['name']}({tu['input']})")

        if response.stop_reason != "tool_use":
            return response.text

        # "Parallel" here is about the *model's* output shape: one turn with
        # N tool_use blocks instead of N turns with one each. That saves N-1
        # model round trips (latency + tokens): the real win.
        # Execution below is still sequential. For true concurrency on
        # I/O bound tools, swap this list comp for ThreadPoolExecutor.map or
        # asyncio.gather. Not done here to keep the lesson focused.
        results = [dispatch(tu) for tu in response.tool_uses]
        print(f"  -> dispatched {len(results)} tool call(s) in one turn")
        messages.append({"role": "user", "content": results})

    return ""


if __name__ == "__main__":
    final = run_agent(
        "Three things: the weather in Lisbon, the current time in Tokyo "
        "(Asia/Tokyo), and how much 100 EUR is in JPY. Bundle them in one reply."
    )
    print(f"\n=== final answer ===\n{final}")
