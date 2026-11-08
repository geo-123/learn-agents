"""Lesson 01b: a tool loop that streams every turn.

The loop is the Lesson 01 loop with three changes:

    1. Each turn is streamed. Text is printed as it arrives, and tool
       input fragments (`input_json` events) are printed as they arrive too.
    2. The tools set `eager_input_streaming: True`. The API then streams
       tool input without buffering and validating it first, so a long
       argument (here, the body of a note) shows up at once instead of
       after a silent pause. The price: the input can be truncated or
       malformed, so the loop checks stop_reason and validates every
       input with pydantic before running anything.
    3. After the stream ends, `get_final_message()` gives the complete
       assistant turn, and its full content (thinking, text, tool_use) is
       appended to the history, exactly as in a non streamed loop.

Run:
    uv run python lessons/01b_streaming/04_stream_tool_loop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
NOTES: dict[str, str] = {}  # where save_note writes; a real tool would use a file or a DB


class ReadPolicyArgs(BaseModel):
    name: str


class SaveNoteArgs(BaseModel):
    title: str
    body: str


def read_policy(args: ReadPolicyArgs) -> str:
    path = CORPUS_DIR / f"{args.name}.md"
    if path.parent != CORPUS_DIR or not path.exists():
        return f"No document named {args.name!r}."
    return path.read_text(encoding="utf-8")


def save_note(args: SaveNoteArgs) -> str:
    NOTES[args.title] = args.body
    return f"Saved note {args.title!r} ({len(args.body)} characters)."


REGISTRY: dict[str, tuple[type[BaseModel], Any]] = {
    "read_policy": (ReadPolicyArgs, read_policy),
    "save_note": (SaveNoteArgs, save_note),
}
DOC_NAMES = ", ".join(p.stem for p in sorted(CORPUS_DIR.glob("*.md")))
TOOLS: list[ToolParam] = [
    {
        "name": "read_policy",
        "description": f"Return one Acme handbook document. Valid names: {DOC_NAMES}.",
        "input_schema": ReadPolicyArgs.model_json_schema(),
        "eager_input_streaming": True,
    },
    {
        "name": "save_note",
        "description": "Save a note with a title and a body of plain text.",
        "input_schema": SaveNoteArgs.model_json_schema(),
        "eager_input_streaming": True,
    },
]


def run_tool(name: str, raw_input: object) -> tuple[str, bool]:
    """Validate first, run second. Returns (content, is_error)."""
    if name not in REGISTRY:
        return f"Unknown tool {name!r}.", True
    args_model, fn = REGISTRY[name]
    try:
        args = args_model.model_validate(raw_input)
    except ValidationError as e:
        # Hand the raw input back so the model can see what it sent and retry.
        return f"Invalid input {json.dumps(raw_input)}: {e.errors()[0]['msg']}", True
    return fn(args), False


def main() -> None:
    messages: list[MessageParam] = [{
        "role": "user",
        "content": "Read the security policy, then save a note titled 'Security basics' "
                   "with a ten bullet checklist for new hires.",
    }]
    json_retries = 0
    for turn in range(1, 8):
        print(f"\n=== turn {turn} ===")
        try:
            with client().messages.stream(
                model=DEFAULT_MODEL, max_tokens=8000, tools=TOOLS, messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "text":
                        print(event.text, end="", flush=True)
                    elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                        print(f"\n[tool_use {event.content_block.name}] ", end="", flush=True)
                    elif event.type == "input_json":
                        print(event.partial_json, end="", flush=True)
                response = stream.get_final_message()
            json_retries = 0
        except ValueError as e:
            # Tool input the SDK could not parse at all. No complete tool_use
            # block exists to answer, so rerun the same turn, a bounded number of times.
            json_retries += 1
            print(f"\n[unparseable tool input, retry {json_retries}: {e}]")
            if json_retries > 2:
                raise
            continue

        print(f"\n[stop_reason {response.stop_reason}]")
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "refusal" or not tool_uses:
            break  # a final answer, or a refusal that may have cut a tool call short
        if response.stop_reason == "max_tokens":
            # A truncated input often parses as a valid but partial object. Never run it.
            raise RuntimeError("Tool input cut off at max_tokens; raise max_tokens and retry.")

        # The full assistant turn goes into the history, thinking blocks included.
        messages.append({"role": "assistant", "content": response.content})
        results: list[ToolResultBlockParam] = []
        for block in tool_uses:
            content, failed = run_tool(block.name, block.input)
            print(f"[ran {block.name}: {content[:80]!r}{' (error)' if failed else ''}]")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": failed})
        messages.append({"role": "user", "content": results})

    for title, body in NOTES.items():
        print(f"\n--- saved note: {title} ---\n{body}")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")
    main()
