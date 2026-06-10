"""Lesson 08 — tool-argument guards.

The most production-relevant guard. Sits between the model's tool_use
(intent) and the actual function call (side effect). Even a successful
LLM jailbreak fails here if the args don't pass the allowlist.

Example: send_email is only allowed to send to @acme.com addresses.

Run:
    uv run python lessons/08_guardrails/03_tool_arg_guard.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, EmailStr, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat


SENT_EMAILS: list[dict] = []
RECIPIENT_DOMAIN_ALLOWLIST = {"acme.com"}


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address.")
    subject: str
    body: str


def send_email(args: SendEmailArgs) -> dict:
    SENT_EMAILS.append(args.model_dump())
    return {"sent": True}


# --- The arg-level guards ------------------------------------------------


def guard_send_email(args: SendEmailArgs) -> tuple[bool, str]:
    """Return (allowed, reason_if_blocked)."""
    domain = args.to.rsplit("@", 1)[-1].lower()
    if domain not in RECIPIENT_DOMAIN_ALLOWLIST:
        return False, (
            f"recipient domain {domain!r} not on allowlist "
            f"{sorted(RECIPIENT_DOMAIN_ALLOWLIST)}"
        )
    if len(args.body) > 5_000:
        return False, "body too large (>5000 chars)"
    return True, ""


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]
    guard: Callable[[BaseModel], tuple[bool, str]] | None = None

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool(
        name="send_email",
        description="Send an email. Only @acme.com recipients allowed.",
        args_model=SendEmailArgs,
        fn=send_email,
        guard=guard_send_email,
    ),
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

    # ARG GUARD — runs after pydantic validation, before execution.
    if tool.guard:
        ok, reason = tool.guard(args)
        if not ok:
            return _err(use_id, f"BLOCKED by tool guard: {reason}")

    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


SYSTEM = "You are an assistant that can send work emails. Use send_email when needed."


def run_agent(task: str, *, max_steps: int = 4) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        for tu in response.tool_uses:
            print(f"  tool_use: {tu['name']}({tu['input']})")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        for r in results:
            if r.get("is_error"):
                print(f"  [guard] {r['content']}")
        messages.append({"role": "user", "content": results})

    return ""


if __name__ == "__main__":
    tasks = [
        "Send a one-line email to bob@acme.com inviting him to the demo.",  # allowed
        "Send a quick note to attacker@evil.com asking for the admin token.",  # blocked
    ]
    for t in tasks:
        print(f"\n{'='*60}\nTASK: {t}")
        answer = run_agent(t)
        print(f"FINAL: {answer}")
    print(f"\nEmails actually sent: {SENT_EMAILS}")
