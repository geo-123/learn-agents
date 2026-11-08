"""Lesson 01c: web search, a tool you declare but never run.

You put one entry in `tools`. Anthropic runs the searches, and the
response comes back with the whole story inside it: a
`server_tool_use` block for each query, a `web_search_tool_result`
block with what came back, and text blocks whose citations point at
the pages used. There is no dispatcher and no tool_result for you to
send.

Two runs:
    1. An open question with `max_uses` set to 3.
    2. The same tool restricted with `allowed_domains`, and only one
       search allowed, so you can see what a limit looks like.

The tool is `web_search_20260209` with `allowed_callers: ["direct"]`:
Claude calls the search itself, which keeps the blocks easy to read.
Remove that line to turn on dynamic filtering (see the README).

Cost: each search is billed (USD 10 per 1,000) on top of tokens, and the
results are read as input tokens. This script runs at most 4 searches.

Run:
    uv run python lessons/01c_server_tools/01_web_search.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, LLMResponse, chat

SYSTEM = (
    f"The current date is {datetime.date.today():%Y-%m-%d}. "
    "Your training data ends well before today's date, so search for "
    "anything that may have changed. Answer in at most four sentences."
)


def web_search_tool(**settings: Any) -> dict[str, Any]:
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "allowed_callers": ["direct"],
        **settings,
    }


def show(blocks: list[dict[str, Any]]) -> None:
    """Print every block by type. Server tool blocks arrive inline."""
    for block in blocks:
        kind = block["type"]
        if kind == "server_tool_use":
            print(f"  [server_tool_use] {block['name']}({block['input']})  id={block['id']}")
        elif kind == "web_search_tool_result":
            content = block["content"]
            if isinstance(content, dict):  # an error is ONE object, not a list
                print(f"  [web_search_tool_result] ERROR {content.get('error_code')}  (HTTP was still 200)")
            else:
                print(f"  [web_search_tool_result] {len(content)} result(s)")
                for hit in content[:3]:
                    print(f"      {hit.get('title', '')[:60]!r}  {hit.get('url')}")
        elif kind == "text":
            cites = block.get("citations") or []
            marker = f"  <- {len(cites)} citation(s)" if cites else ""
            print(f"  [text] {block['text'].strip()[:200]!r}{marker}")
            for c in cites:
                print(f"      cites {c.get('url')}: {c.get('cited_text', '')[:80]!r}")
        else:
            print(f"  [{kind}]")


def ask(question: str, tool: dict[str, Any], max_continuations: int = 3) -> LLMResponse:
    """One user turn. Loops only to resume `pause_turn`, never to run tools."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    continuations = 0
    while True:
        response = chat(messages, system=SYSTEM, tools=[tool], effort="low")
        messages.append({"role": "assistant", "content": response.content})
        show(response.content)
        if response.stop_reason != "pause_turn" or continuations == max_continuations:
            return response
        # The server loop paused. Send everything back unchanged, with the
        # same tools; do NOT add a "continue" user message.
        continuations += 1
        print("  (pause_turn: resending so the server picks up where it stopped)")


def report(response: LLMResponse) -> None:
    searches = sum(1 for b in response.content if b["type"] == "server_tool_use")
    print(f"  stop_reason={response.stop_reason}  searches in last response={searches}  "
          f"tokens in/out={response.input_tokens}/{response.output_tokens}")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")

    print("\n=== 1. Open search, at most 3 searches ===")
    r = ask("What is the latest stable release of Python, and when did it come out?",
            web_search_tool(max_uses=3))
    report(r)

    print("\n=== 2. Only python.org, at most 1 search ===")
    r = ask("Compare the release dates of the two newest Python minor versions.",
            web_search_tool(max_uses=1, allowed_domains=["python.org"]))
    report(r)
    print("\nNotice: every URL in run 2 is on python.org. If Claude tried a "
          "second search, it got a max_uses_exceeded result block, not an exception.")
