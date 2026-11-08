"""Lesson 00c: what thinking blocks look like, and how a tool loop keeps them.

Part 1 asks one question twice. With the default display ("omitted") the
response still holds thinking blocks, but their text is empty and only a
signature comes back. With display "summarized" the blocks carry a
readable summary of the reasoning.

Part 2 runs a small tool loop. The assistant turn is appended to the
history exactly as it came back, thinking blocks and signatures
included. Drop or edit them and the next request is rejected or loses
the reasoning that led to the tool call.

Run:
    uv run python lessons/00c_thinking_and_effort/02_see_the_thinking.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import LLMResponse, chat

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
SUMMARIZED = {"type": "adaptive", "display": "summarized"}


def describe(response: LLMResponse) -> None:
    """Print every block by type. Never assume content[0] is the answer."""
    for i, block in enumerate(response.content):
        kind = block.get("type")
        if kind == "thinking":
            text = block.get("thinking", "")
            shown = text[:200].replace("\n", " ") if text else "(empty: omitted)"
            print(f"  [{i}] thinking  signature={len(block.get('signature', ''))} chars  {shown}")
        elif kind == "text":
            print(f"  [{i}] text      {block['text'][:200].replace(chr(10), ' ')}")
        elif kind == "tool_use":
            print(f"  [{i}] tool_use  {block['name']}({block['input']})")


def part1() -> None:
    question = "A meal costs $31 and has a receipt but no attendee names. Is it reimbursable as is?"
    policy = (CORPUS_DIR / "expense_policy.md").read_text(encoding="utf-8")
    messages = [{"role": "user", "content": question}]

    print("\n--- Part 1a: default display (omitted) ---")
    describe(chat(messages, system=policy, effort="high"))

    print("\n--- Part 1b: display summarized ---")
    describe(chat(messages, system=policy, effort="high", thinking=SUMMARIZED))


# --- Part 2: a tool loop that keeps thinking blocks -------------------------

TOOLS = [{
    "name": "read_policy",
    "description": (
        "Return the full text of one Acme handbook document. "
        "Valid names: " + ", ".join(p.stem for p in sorted(CORPUS_DIR.glob("*.md")))
    ),
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}]


def read_policy(args: dict[str, Any]) -> tuple[str, bool]:
    path = CORPUS_DIR / f"{args.get('name', '')}.md"
    if path.parent != CORPUS_DIR or not path.exists():
        return f"No document named {args.get('name')!r}.", True
    return path.read_text(encoding="utf-8"), False


def part2() -> None:
    print("\n--- Part 2: tool loop, thinking blocks passed back unchanged ---")
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": "I work remotely and want a $250 monitor. What do the policies say? Check them.",
    }]
    for turn in range(1, 6):
        response = chat(messages, tools=TOOLS, thinking=SUMMARIZED, effort="medium")
        print(f"\nturn {turn}: stop_reason={response.stop_reason}")
        describe(response)

        # The whole content list goes back as is: thinking, text and tool_use.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break

        results = []
        for call in response.tool_uses:
            output, failed = read_policy(call["input"])
            results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": output,
                "is_error": failed,
            })
        messages.append({"role": "user", "content": results})

    kept = sum(
        1 for m in messages if m["role"] == "assistant"
        for b in m["content"] if b.get("type") == "thinking"
    )
    print(f"\nThinking blocks carried in the history: {kept}")


if __name__ == "__main__":
    part1()
    part2()
