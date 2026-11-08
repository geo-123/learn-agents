"""Lesson 06: durable pause and resume.

The agent runs until it needs human approval, then SERIALIZES itself to
disk and exits. A human later approves (by editing the state file or
running --approve). On resume, the agent loads its state and continues
exactly where it stopped.

This is the toy version of the pattern used in async approval workflows
(Slack approvals, email based escalations, multi day reviews).

Run:
    1. First run (will pause and serialize):
       uv run python lessons/06_human_in_loop/04_durable_pause.py "Send 'hi' to alice"

    2. Inspect lessons/06_human_in_loop/pending_state.json

    3. Approve and resume:
       uv run python lessons/06_human_in_loop/04_durable_pause.py --resume approve

    4. Or reject:
       uv run python lessons/06_human_in_loop/04_durable_pause.py --resume reject
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


STATE_PATH = Path(__file__).resolve().parent / "pending_state.json"


# --- Tools (same shape as 01) ---------------------------------------------


class SendEmailArgs(BaseModel):
    to: str
    subject: str
    body: str


def send_email(args: SendEmailArgs) -> dict:
    return {"sent_at": "now", "to": args.to}


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    needs_approval: bool = False

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("send_email",
         "Send an email. Requires human approval.",
         SendEmailArgs, send_email, needs_approval=True),
]}


# --- Pause / resume primitives -------------------------------------------


class PauseForApproval(Exception):
    """Raised by the dispatcher when a tool needs human approval. Caller
    catches and serializes."""
    def __init__(self, tool_use: dict, messages: list[dict]):
        self.tool_use = tool_use
        self.messages = messages


def dispatch(tool_use: dict, messages: list[dict]) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"unknown tool {name!r}")
    try:
        args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
    except ValidationError as e:
        return _err(use_id, f"invalid args: {e.errors()}")

    if tool.needs_approval:
        # Don't decide here: bubble out so the caller can serialize.
        raise PauseForApproval(tool_use=tool_use, messages=messages)

    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


# --- Run loop -------------------------------------------------------------


def run(messages: list[dict], *, max_steps: int = 6) -> str | None:
    """Run until done or until we hit a tool that needs approval.

    Returns the final text if completed, or None if paused (state already
    serialized).
    """
    tools = [t.schema() for t in REGISTRY.values()]

    for step in range(1, max_steps + 1):
        response = chat(messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"--- step {step}  stop={response.stop_reason} ---")
        if response.stop_reason != "tool_use":
            return response.text

        # Try dispatching every tool_use. If one needs approval, pause.
        results: list[dict] = []
        for tu in response.tool_uses:
            try:
                results.append(dispatch(tu, messages))
            except PauseForApproval as pause:
                _save_state({
                    "id": str(uuid.uuid4()),
                    "messages": messages,
                    "pending_tool_use": pause.tool_use,
                })
                print(f"\n[paused] tool {pause.tool_use['name']} needs approval. "
                      f"State serialized to {STATE_PATH}.")
                return None

        messages.append({"role": "user", "content": results})

    return "(max steps)"


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def resume(decision: str) -> str | None:
    """Resume from a serialized pause with an approve/reject decision."""
    state = _load_state()
    pending = state["pending_tool_use"]
    messages: list[dict] = state["messages"]

    if decision == "approve":
        # Run the tool now.
        tool = REGISTRY[pending["name"]]
        args = tool.args_model.model_validate(pending["input"])
        result = tool.fn(args)
        tr = {"type": "tool_result", "tool_use_id": pending["id"],
              "content": json.dumps(result)}
        print(f"[approved] executed {pending['name']}: {result}")
    elif decision == "reject":
        tr = {"type": "tool_result", "tool_use_id": pending["id"],
              "content": "User rejected this action. Tell the user it was not sent.",
              "is_error": True}
        print(f"[rejected] returning error to agent")
    else:
        raise ValueError(f"decision must be 'approve' or 'reject', not {decision!r}")

    messages.append({"role": "user", "content": [tr]})
    STATE_PATH.unlink()  # consume the pending state
    return run(messages)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--resume":
        final = resume(sys.argv[2])
    else:
        task = " ".join(sys.argv[1:]) or "Send an email to alice@example.com saying 'hi'."
        final = run([{"role": "user", "content": task}])
    if final is not None:
        print(f"\n=== final ===\n{final}")
