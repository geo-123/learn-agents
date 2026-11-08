"""Lesson 01b: a stream from the wire up, parsed by hand.

A streamed response is one long HTTP response whose body is plain text in
the server sent events (SSE) format. Each event is a few lines and ends
with a blank line:

    event: content_block_delta
    data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}}

No helper here. The request asks for `stream=True` through
`with_streaming_response`, which hands back the HTTP response unread, so
`iter_lines()` gives you the body line by line, exactly as it arrives.

Part 1 prints every raw line.
Part 2 turns those lines into a message with about forty lines of code:
    * split the lines into events at each blank line, `json.loads` the data;
    * `message_start` gives the message shell (id, model, input usage);
    * `content_block_start` opens a block at its `index`;
    * `content_block_delta` grows it: text, thinking and signature are
      appended, tool input arrives as `partial_json` strings you collect;
    * `content_block_stop` closes it: only now is the collected tool JSON
      complete, so only now do you `json.loads` it;
    * `message_delta` brings `stop_reason` and the final output usage.
Part 3 feeds the very same bytes to the SDK's own stream parser and
checks that `get_final_message()` builds the same message as yours. Same
input, so no second request.

The request offers one tool with `eager_input_streaming` so you see the
tool input arrive in pieces. The tool is never run here; the full loop is
in 04_stream_tool_loop.py.

Run:
    uv run python lessons/01b_streaming/01_sse_by_hand.py
"""
from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import httpx2
from anthropic import NOT_GIVEN, Stream
from anthropic.lib.streaming import MessageStream
from anthropic.types import RawMessageStreamEvent, ToolParam

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, client

TOOLS: list[ToolParam] = [{
    "name": "lookup_hotel_cap",
    "description": "Return the nightly hotel cap in US dollars for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "eager_input_streaming": True,
}]
QUESTION = (
    "Does a $380 a night hotel in Tokyo need approval? Look up the cap with "
    "the tool; say in one sentence what you are about to do first."
)


# --- Step 1: lines to events ---------------------------------------------

def read_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Group raw SSE lines into (event name, parsed data) pairs.

    The format: `event:` names the event, `data:` carries JSON (a long
    payload may be split over several `data:` lines, joined with a newline),
    a line starting with `:` is a comment, and a blank line ends the event.
    """
    name, data_lines = "", []
    for line in lines:
        if line == "":  # a blank line: the event is complete
            if data_lines:
                data = json.loads("\n".join(data_lines))
                if name == "error":  # the server can fail mid stream, with a 200 status
                    raise RuntimeError(f"error event in the stream: {data}")
                if name != "ping":  # keep alive events carry nothing
                    yield name, data
            name, data_lines = "", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


# --- Step 2: events to a message ------------------------------------------

def assemble(events: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build the complete message from the events, the way the SDK does."""
    message: dict[str, Any] = {}
    tool_json: dict[int, str] = {}  # block index -> the partial_json pieces so far
    for name, data in events:
        if name == "message_start":
            message = data["message"]  # content starts empty, stop_reason null
        elif name == "content_block_start":
            message["content"].append(data["content_block"])  # lands at data["index"]
        elif name == "content_block_delta":
            index, delta = data["index"], data["delta"]
            block = message["content"][index]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta["type"] == "input_json_delta":
                # Not valid JSON on its own: '{"ci' then 'ty": "Tok' then 'yo"}'.
                tool_json[index] = tool_json.get(index, "") + delta["partial_json"]
            elif delta["type"] == "citations_delta":
                block.setdefault("citations", []).append(delta["citation"])
        elif name == "content_block_stop":
            pieces = tool_json.pop(data["index"], "")
            if pieces:  # the tool input is complete only now; none at all keeps the {} from the start
                message["content"][data["index"]]["input"] = json.loads(pieces)
        elif name == "message_delta":
            message.update(data["delta"])  # stop_reason, stop_sequence
            # Usage counts here are running totals: overwrite, never add.
            message["usage"].update({k: v for k, v in data["usage"].items() if v is not None})
        elif name == "message_stop":
            break
    return message


def drop_nones(value: Any) -> Any:
    """The same data without null fields, so two dumps compare fairly."""
    if isinstance(value, dict):
        return {k: drop_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nones(v) for v in value]
    return value


def main() -> None:
    print("--- Part 1: the raw lines, as they arrive ---\n")
    raw_lines: list[str] = []
    with client().messages.with_streaming_response.create(
        model=DEFAULT_MODEL,
        max_tokens=4000,
        stream=True,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "low"},
        tools=TOOLS,
        messages=[{"role": "user", "content": QUESTION}],
    ) as response:
        print(f"HTTP {response.status_code}, content-type: {response.headers.get('content-type')}\n")
        for line in response.iter_lines():
            raw_lines.append(line)
            print(f"  |{line}", flush=True)

    print("\n--- Part 2: the message, assembled by hand ---\n")
    mine = assemble(read_sse(raw_lines))
    for i, block in enumerate(mine["content"]):
        if block["type"] == "thinking":
            print(f"[{i}] thinking : {block['thinking'][:200]!r} (signature {len(block['signature'])} chars)")
        elif block["type"] == "text":
            print(f"[{i}] text     : {block['text']!r}")
        elif block["type"] == "tool_use":
            print(f"[{i}] tool_use : {block['name']}({block['input']}) id {block['id']}")
    print(f"stop_reason : {mine['stop_reason']}")
    print(f"usage       : input {mine['usage']['input_tokens']}, output {mine['usage']['output_tokens']}")

    print("\n--- Part 3: the same bytes through the SDK's parser ---\n")
    # Rebuild the body from the lines and hand it to the classes that
    # messages.stream() uses: Stream decodes the SSE, MessageStream accumulates.
    body = ("\n".join(raw_lines) + "\n").encode("utf-8")
    replay = httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    # RawMessageStreamEvent is a union of the event types, not a class: the cast tells the type checker so.
    events: Stream[RawMessageStreamEvent] = Stream(
        cast_to=cast(Any, RawMessageStreamEvent), response=replay, client=client(),
    )
    final = MessageStream(events, output_format=NOT_GIVEN).get_final_message()
    sdk = drop_nones(final.model_dump(mode="json"))

    for field in ("id", "model", "content", "stop_reason", "usage"):
        same = drop_nones(mine[field]) == sdk[field]
        print(f"{field:12}: {'same' if same else 'DIFFERENT'}")
    print(
        "\nThat is all messages.stream() does for you: read the lines, group them "
        "into events, and grow one message from the deltas. stop_reason arrived "
        "in the second to last event; nothing before it says whether the turn is complete."
    )


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}\n")
    main()
