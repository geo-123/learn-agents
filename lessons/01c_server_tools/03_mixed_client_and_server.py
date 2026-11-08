"""Lesson 01c: one client tool and one server tool in the same loop.

`search_handbook` is a client tool: it runs here, on your machine,
over the private Acme handbook in data/corpus. `web_search` is a server
tool: Anthropic runs it. The question needs both, a cap from the
handbook and a current exchange rate from the web.

What to watch for:
    * When Claude asks for both tools in one turn, the API does NOT run
      the search yet. It stops with `stop_reason: "tool_use"` and the
      response holds a `server_tool_use` block with no result. You run
      the client tool, send the tool_result, and the API runs the
      search on the next request.
    * `pause_turn` means "resend to continue"; `tool_use` means "your
      turn to run something". The loop handles the two differently.
    * Only client tools ever reach your dispatcher.

Run:
    uv run python lessons/01c_server_tools/03_mixed_client_and_server.py
"""
from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def search_handbook(query: str) -> list[dict[str, str]]:
    """Return handbook paragraphs that share words with the query."""
    words = set(re.findall(r"[a-z]{4,}", query.lower()))
    hits = []
    for path in sorted(CORPUS.glob("*.md")):
        for para in path.read_text(encoding="utf-8").split("\n\n"):
            score = len(words & set(re.findall(r"[a-z]{4,}", para.lower())))
            if score:
                hits.append((score, path.name, para.strip()))
    hits.sort(reverse=True)
    return [{"source": name, "text": text} for _, name, text in hits[:3]]


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_handbook",
        "description": "Search Acme's internal employee handbook (policies, caps, benefits). "
                       "Call this for any question about Acme rules.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords to look for."}},
            "required": ["query"],
        },
    },
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 2, "allowed_callers": ["direct"]},
]

SYSTEM = (
    f"The current date is {datetime.date.today():%Y-%m-%d}. Use search_handbook for Acme "
    "policy and web_search for live facts such as exchange rates. Be brief."
)


def run_client_tool(tool_use: dict[str, Any]) -> dict[str, Any]:
    if tool_use["name"] != "search_handbook":
        return {"type": "tool_result", "tool_use_id": tool_use["id"],
                "content": f"unknown tool {tool_use['name']!r}", "is_error": True}
    hits = search_handbook(str(tool_use["input"].get("query", "")))
    return {"type": "tool_result", "tool_use_id": tool_use["id"], "content": json.dumps(hits)}


def pending_server_calls(blocks: list[dict[str, Any]]) -> list[str]:
    """server_tool_use ids that have no result block yet in this response."""
    answered = {b.get("tool_use_id") for b in blocks if b["type"].endswith("_tool_result")}
    return [b["name"] for b in blocks if b["type"] == "server_tool_use" and b["id"] not in answered]


def run(question: str, max_steps: int = 8) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=TOOLS, effort="low")
        messages.append({"role": "assistant", "content": response.content})
        kinds = [b["type"] for b in response.content if b["type"] != "thinking"]
        print(f"\n--- step {step}  stop={response.stop_reason}  blocks={kinds}")

        for b in response.content:
            if b["type"] == "server_tool_use":
                print(f"  ran at Anthropic : {b['name']}({b['input']})")
            elif b["type"] == "tool_use":
                print(f"  runs here        : {b['name']}({b['input']})")

        if response.stop_reason == "pause_turn":
            continue  # resend as is; the server loop resumes
        if response.stop_reason == "tool_use":
            waiting = pending_server_calls(response.content)
            if waiting:
                print(f"  deferred server calls (run after our results): {waiting}")
            results = [run_client_tool(tu) for tu in response.tool_uses]
            messages.append({"role": "user", "content": results})
            continue
        print(f"\nAnswer:\n{response.text.strip()}")
        return
    print("(stopped: step limit)")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")
    run("I am booking 4 nights in a Tokyo hotel for work. What is my nightly cap under "
        "the Acme handbook, and roughly what is the 4 night total cap in Japanese yen today?")
