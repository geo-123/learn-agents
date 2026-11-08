"""Lesson 09c: tool search, so 40 tools cost what 4 do.

Builds 40 fake HR tools from a small table (8 areas x 5 actions). Three
common ones stay loaded; the other 37 get `defer_loading: true`. A tool
search tool is added (BM25 by default, `--regex` for the regex variant).
The model starts with only 4 tool definitions in context, searches the
catalog when it needs something, and the API loads just the matches.

Things to notice in the output:
  * the token count of the tool definitions with and without deferral
    (the token counting endpoint is free);
  * the `server_tool_use` block: the search the model ran, executed on
    Anthropic's side. You never answer it with a tool_result;
  * the `tool_search_tool_result` block: the names it found, as
    `tool_reference` entries the API expands into full definitions;
  * the ordinary `tool_use` that follows, which your code runs as usual.

Every deferred tool is still sent in `tools` on every request: the API
needs the definitions to search them. Deferral changes what the model
reads, not what you send.

Model: the course default. The tool search docs list every current
model, Haiku 4.5 and Haiku 5.5 included.

Run:
    uv run python lessons/09c_many_tools/01_tool_search.py
    uv run python lessons/09c_many_tools/01_tool_search.py --regex "I need to cancel leave request LR-1042"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat, client

AREAS = {
    "leave": "vacation, sick days and time off requests",
    "payroll": "salary, payslips and tax withholding",
    "benefits": "health insurance, 401k and wellness stipend",
    "expense": "expense reports, receipts and reimbursements",
    "onboarding": "new hire checklists and first week tasks",
    "training": "courses, certifications and learning budget",
    "equipment": "laptops, monitors and home office gear",
    "review": "performance reviews, goals and feedback",
}
ACTIONS = {
    "get": "Look up one {area} record by id",
    "list": "List the employee's {area} records",
    "submit": "Submit a new {area} request",
    "update": "Change an existing {area} request",
    "cancel": "Cancel or withdraw a {area} request",
}
ALWAYS_LOADED = {"hr_get_leave", "hr_list_leave", "hr_get_payroll"}  # the 3 most used


def build_tools(search_type: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [{"type": search_type, "name": search_type.rsplit("_", 1)[0]}]
    for area, about in AREAS.items():
        for action, template in ACTIONS.items():
            name = f"hr_{action}_{area}"
            tool: dict[str, Any] = {
                "name": name,
                "description": f"{template.format(area=area)}. Covers {about}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string", "description": f"The {area} record id"},
                                   "note": {"type": "string"}},
                },
            }
            if name not in ALWAYS_LOADED:
                tool["defer_loading"] = True  # never on the search tool itself, never on every tool
            tools.append(tool)
    return tools


def definition_tokens(tools: list[dict[str, Any]]) -> int:
    msg: Any = [{"role": "user", "content": "hi"}]
    return client().messages.count_tokens(model=DEFAULT_MODEL, messages=msg, tools=tools).input_tokens  # type: ignore[arg-type]


def main(request: str, search_type: str) -> None:
    tools = build_tools(search_type)
    all_loaded = [{k: v for k, v in t.items() if k != "defer_loading"} for t in tools[1:]]
    print(f"{len(tools) - 1} HR tools, {len(ALWAYS_LOADED)} loaded up front, search: {search_type}")
    print(f"prompt tokens with all tools loaded : {definition_tokens(all_loaded)}")
    print(f"prompt tokens with deferral         : {definition_tokens(tools)}")

    system = "You are an HR assistant. Search for HR tools by area (leave, payroll, benefits, expense, ...) when the loaded ones do not fit."
    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]
    for step in range(1, 6):
        response = chat(messages, system=system, tools=tools, effort="low")
        messages.append({"role": "assistant", "content": response.content})  # search blocks included, unchanged
        for block in response.content:
            if block["type"] == "server_tool_use":
                print(f"\nstep {step}: search {block['input']}")
            elif block["type"] == "tool_search_tool_result":
                found = [r["tool_name"] for r in block["content"].get("tool_references", [])]
                print(f"step {step}: found {found}")
            elif block["type"] == "tool_use":
                print(f"step {step}: CALL {block['name']}({block['input']})")
        if response.stop_reason != "tool_use":
            print(f"\nanswer: {response.text.strip()}")
            print(f"input tokens on the last turn: {response.input_tokens}")
            return
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tu["id"],
             "content": json.dumps({"ok": True, "tool": tu["name"], "input": tu["input"]})}
            for tu in response.tool_uses
        ]})


if __name__ == "__main__":
    args = sys.argv[1:]
    kind = "tool_search_tool_bm25_20251119"
    if args[:1] == ["--regex"]:
        kind, args = "tool_search_tool_regex_20251119", args[1:]
    main(" ".join(args) or "Please cancel my laptop request EQ-77, I got one from my old team.", kind)
