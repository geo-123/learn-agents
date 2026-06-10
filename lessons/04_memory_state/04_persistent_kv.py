"""Lesson 04 — agent-controlled persistent memory.

The agent has three tools: remember, recall, list_memories. The store is
a JSON file on disk. Run once and tell the agent something memorable.
Run again and ask the agent to recall it.

This is the pattern modern long-running assistants use: the model
decides what's worth remembering, not your harness.

Run:
    uv run python lessons/04_memory_state/04_persistent_kv.py "Hi, I'm Sudeep and I prefer 4-space indents."
    uv run python lessons/04_memory_state/04_persistent_kv.py "What do you remember about me?"
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


STORE_PATH = Path(__file__).resolve().parent / "memory_store.json"


def load_store() -> dict[str, str]:
    if not STORE_PATH.exists():
        return {}
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def save_store(data: dict[str, str]) -> None:
    STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- Tools ----------------------------------------------------------------


class RememberArgs(BaseModel):
    key: str = Field(description="A short slug, e.g. 'name', 'pref_indent'.")
    value: str = Field(description="The value to store.")


def remember(args: RememberArgs) -> dict:
    store = load_store()
    store[args.key] = args.value
    save_store(store)
    return {"stored": {args.key: args.value}}


class RecallArgs(BaseModel):
    key: str


def recall(args: RecallArgs) -> dict:
    store = load_store()
    return {"key": args.key, "value": store.get(args.key)}


class ListMemoriesArgs(BaseModel):
    pass


def list_memories(args: ListMemoriesArgs) -> dict:
    return {"memories": load_store()}


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
    Tool(
        name="remember",
        description=(
            "Save a fact to long-term memory for future conversations. "
            "Use this for user preferences, names, important constraints — "
            "things you'd want to know next time. Don't use it for things "
            "only relevant to this turn."
        ),
        args_model=RememberArgs, fn=remember,
    ),
    Tool(
        name="recall",
        description="Retrieve a previously-remembered fact by key.",
        args_model=RecallArgs, fn=recall,
    ),
    Tool(
        name="list_memories",
        description="List all keys currently in long-term memory.",
        args_model=ListMemoriesArgs, fn=list_memories,
    ),
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
    "You are an assistant with persistent long-term memory. At the start "
    "of any conversation, call list_memories to see what you already know. "
    "When the user shares a fact about themselves or their preferences, "
    "save it with remember(). Use recall() to look things up. "
    "Be transparent — tell the user when you've saved or recalled something."
)


def run_agent(message: str) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": message}]

    for step in range(1, 6):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        if response.text:
            print(f"text: {response.text}")
        for tu in response.tool_uses:
            print(f"  -> {tu['name']}({tu['input']})")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

    return ""


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "What do you remember about me?"
    print(f"\nUSER: {msg}\n")
    answer = run_agent(msg)
    print(f"\nFINAL:\n{answer}")
    print(f"\n[store now at {STORE_PATH}]")
