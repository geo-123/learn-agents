"""Lesson 06: inline approval before risky tools.

A `send_email` tool gets a require_approval wrapper. Before each call,
the harness prints the args and asks y/n/edit. Approved -> tool runs.
Rejected -> tool_result with is_error so the agent can replan. Edited
-> tool runs with the mutated args.

Run:
    uv run python lessons/06_human_in_loop/01_approve_cli.py
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


SENT_EMAILS: list[dict] = []  # in memory side effect log


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str
    body: str


def send_email(args: SendEmailArgs) -> dict:
    """Pretend to send. In production this hits SES/SendGrid/etc."""
    SENT_EMAILS.append(args.model_dump())
    return {"sent": True, "to": args.to}


class GetUserNameArgs(BaseModel):
    pass


def get_user_name(args: GetUserNameArgs) -> dict:
    return {"name": "Sudeep George"}


# --- The approval gate ----------------------------------------------------


def require_approval(name: str, args: BaseModel) -> tuple[bool, BaseModel]:
    """Print the proposal; let the user approve, reject, or edit.

    Returns (approved, possibly_modified_args).
    """
    print(f"\n>>> Approval required for: {name}")
    print(json.dumps(args.model_dump(), indent=2))
    while True:
        choice = input("[y]es / [n]o / [e]dit ? ").strip().lower()
        if choice in ("y", "yes"):
            return True, args
        if choice in ("n", "no"):
            return False, args
        if choice in ("e", "edit"):
            new = {**args.model_dump()}
            for k in new:
                v = input(f"  {k} [{new[k]}]: ").strip()
                if v:
                    new[k] = v
            try:
                edited = args.__class__.model_validate(new)
                print("  -> using edited args")
                return True, edited
            except ValidationError as e:
                print(f"  edited args failed validation: {e}; try again")
                continue
        print("(type y, n, or e)")


# --- Registry / dispatch / loop -------------------------------------------


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
    Tool("get_user_name",
         "Returns the user's full name.",
         GetUserNameArgs, get_user_name, needs_approval=False),
    Tool("send_email",
         "Sends an email. Real side effect: use only when the user has asked.",
         SendEmailArgs, send_email, needs_approval=True),
]}


def dispatch(tool_use: dict) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"unknown tool {name!r}")
    try:
        args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
    except ValidationError as e:
        return _err(use_id, f"invalid args: {e.errors()}")

    if tool.needs_approval:
        approved, args = require_approval(name, args)
        if not approved:
            return _err(use_id, "User rejected this action. Do not retry. Ask the user what to do.")

    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


SYSTEM = (
    "You are an assistant that can send emails on the user's behalf. "
    "Always look up the user's name first if you need it. "
    "If a user approval step rejects your action, do NOT retry the same "
    "action; ask the user what they want to do instead."
)


def run_agent(task: str, *, max_steps: int = 6) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools)
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
        "Send a one line email to alice@example.com inviting her to coffee "
        "tomorrow at 10am. Sign it with my name."
    )
    print(f"\n=== final ===\n{answer}")
    print(f"\nEmails actually sent: {SENT_EMAILS}")
