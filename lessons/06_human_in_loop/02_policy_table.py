"""Lesson 06 — policy-table-driven approvals.

Each tool registers with a policy: 'auto' (just run), 'confirm' (ask the
user), or 'block' (never allowed even if the model asks). The dispatcher
reads the policy at call time. Adding a new tool = one new row.

Also: every decision is appended to an audit log (audit.jsonl).

Run:
    uv run python lessons/06_human_in_loop/02_policy_table.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


AUDIT_LOG = Path(__file__).resolve().parent / "audit.jsonl"

Policy = Literal["auto", "confirm", "block"]


def audit(event: dict) -> None:
    event = {"ts": time.time(), **event}
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# --- Tools ----------------------------------------------------------------


class ReadFileArgs(BaseModel):
    path: str


def read_file(args: ReadFileArgs) -> dict:
    p = Path(args.path)
    if not p.exists():
        return {"error": f"no such file: {args.path}"}
    return {"path": args.path, "content": p.read_text(encoding="utf-8")[:500]}


class SendSlackArgs(BaseModel):
    channel: str
    text: str


def send_slack(args: SendSlackArgs) -> dict:
    return {"posted_to": args.channel, "text": args.text}


class DeleteFileArgs(BaseModel):
    path: str


def delete_file(args: DeleteFileArgs) -> dict:
    """Tool exists in the registry but policy is 'block' — never runs."""
    Path(args.path).unlink(missing_ok=True)
    return {"deleted": args.path}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    policy: Policy

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("read_file",
         "Read up to 500 chars from a local file.",
         ReadFileArgs, read_file, policy="auto"),
    Tool("send_slack",
         "Post a message to a Slack channel.",
         SendSlackArgs, send_slack, policy="confirm"),
    Tool("delete_file",
         "Delete a file. ALWAYS BLOCKED — never executes via the agent.",
         DeleteFileArgs, delete_file, policy="block"),
]}


# --- The dispatcher consults the policy table ----------------------------


def dispatch(tool_use: dict) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"unknown tool {name!r}")
    try:
        args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
    except ValidationError as e:
        return _err(use_id, f"invalid args: {e.errors()}")

    if tool.policy == "block":
        audit({"action": name, "args": args.model_dump(), "decision": "blocked"})
        return _err(use_id, f"Tool {name!r} is blocked by policy.")

    if tool.policy == "confirm":
        print(f"\n>>> {name}({args.model_dump()})")
        if input("approve? [y/n] ").strip().lower() not in ("y", "yes"):
            audit({"action": name, "args": args.model_dump(), "decision": "rejected"})
            return _err(use_id, "User rejected this action.")
        audit({"action": name, "args": args.model_dump(), "decision": "approved"})

    if tool.policy == "auto":
        audit({"action": name, "args": args.model_dump(), "decision": "auto"})

    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


def run_agent(task: str, *, max_steps: int = 6) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        if response.text:
            print(f"assistant: {response.text}")
        for tu in response.tool_uses:
            print(f"  proposes: {tu['name']}({tu['input']})")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

    return ""


if __name__ == "__main__":
    answer = run_agent(
        "Post the message 'shipping in 10' to the #releases Slack channel."
    )
    print(f"\n=== final ===\n{answer}")
    print(f"\n[audit log: {AUDIT_LOG}]")
