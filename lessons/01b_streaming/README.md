# Lesson 01b: Streaming

Every lesson so far waited for the whole response, then printed it. That works for a two line answer. It stops working when the answer is long, when a person is watching, or when the model is about to write a 20K token tool argument.

> Streaming does not make the model faster. It shows you the work as it happens, and it keeps long requests alive.

This lesson calls the SDK directly through `client()`, because `chat()` in `src/_llm.py` deliberately does not stream.

## Why stream

- **Time to first token.** A user who sees words after one second waits patiently for ten more. A user who sees a spinner for eleven seconds does not. The total time is the same; the experience is not.
- **Long outputs.** A long report or a big file body can take minutes. Streaming lets you print, save or cancel as it goes.
- **HTTP timeouts.** An idle connection that waits minutes for one big body can be dropped by a proxy or a load balancer. A stream sends events the whole time.
- **The SDK insists.** For a large `max_tokens` (roughly above 21K tokens with the default timeout) a non streamed `messages.create()` raises `ValueError` before it sends anything: such a request could run past ten minutes. Stream it, or set your own `timeout`. On current models outputs can reach 128K tokens, so this is not rare.

## The helper: `messages.stream()`

```python
with client().messages.stream(model=DEFAULT_MODEL, max_tokens=8000, messages=messages) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()
```

- `text_stream` yields only text pieces. Thinking and tool input are skipped.
- `get_final_message()` returns the complete `Message`, the same object `create()` returns: `content`, `stop_reason`, `usage`. Use it for everything after the stream; never rebuild the message from pieces yourself.
- The `with` block closes the connection, even if your loop breaks early.

A lower level form, `messages.create(..., stream=True)`, yields the raw events and nothing else: no accumulation, no `get_final_message()`. Use the helper unless memory is tight.

## The events underneath

Iterating the stream itself (`for event in stream`) gives every server sent event:

| Event | When | Carries |
|---|---|---|
| `message_start` | Once, first | Message id, model, input token usage |
| `content_block_start` | Each block begins | `index` and the block `type`: `thinking`, `text`, `tool_use` |
| `content_block_delta` | Many times per block | `thinking_delta`, `signature_delta`, `text_delta` or `input_json_delta` |
| `content_block_stop` | Each block ends | `index` |
| `message_delta` | Once, near the end | `stop_reason`, final output token count |
| `message_stop` | Once, last | Nothing more |

Three things to notice:

1. **Thinking streams too.** On current models a response usually starts with a thinking block. With the default display (`"omitted"`) its deltas carry no text, so you see a pause before the first `text_delta`. With `thinking: {"type": "adaptive", "display": "summarized"}` you get `thinking_delta` events with a readable summary, which makes a good progress indicator.
2. **`stop_reason` arrives last.** You cannot know a turn was cut off at `max_tokens`, or refused, until `message_delta`. Do not act on a turn before the stream ends.
3. **The SDK adds convenience events.** Next to the raw events the helper yields `text`, `thinking`, `signature` and `input_json` events, each with the delta and the snapshot so far. They repeat what the deltas carry; use whichever reads better.

## Streaming tool use

Tool calls stream as a `tool_use` block whose input arrives as `input_json_delta` fragments of JSON. By default the API buffers and validates each argument before sending any of it, so a long argument (a file body, a document) arrives after a silent gap.

Setting `eager_input_streaming: True` on a client tool removes that buffering for that tool: the fragments stream as they are generated. Set it on your own tools whenever you stream with tools. Do not set it on server tools, where it is not a valid field.

What you give up is the validation, so a streamed tool loop needs three guards:

1. **Check `stop_reason` before running anything.** On `"max_tokens"` with a `tool_use` present, the input was cut off, and it usually still parses as a valid looking partial object. On `"refusal"`, a tool call may have been cut short. Run no tools in either case.
2. **Validate every input yourself.** The SDK parses fragments with a tolerant parser, so malformed JSON can come back as a silently truncated object rather than an error. Validate against your schema (a pydantic model, as in Lesson 01) and send failures back as `is_error` tool results.
3. **Catch `ValueError` around the stream.** JSON the SDK cannot parse at all raises `ValueError` from the stream iterator, so wrap the whole `with` block, retry the turn a bounded number of times, and let API errors (rate limits, auth) propagate.

After the stream ends, the loop is the same as before: append `final.content` as the assistant turn, every block unchanged (thinking blocks with their signatures, text, tool use), then a user turn with the tool results. Streaming changes how you watch a turn, not what goes into the history.

## Hands on

### `01_stream_text.py`

Part 1 asks for `max_tokens=64000` without streaming and prints the `ValueError` the SDK raises before sending anything. Part 2 streams a welcome letter from the onboarding handbook with `text_stream`, then prints the time to the first text piece, the total time, and the block types, stop reason and token count from `get_final_message()`.

### `02_stream_events.py`

Streams one short answer with summarized thinking and prints every raw event with its arrival time, collapsing runs of identical deltas into one line with a count. Watch the thinking deltas come first, the text deltas after, and `stop_reason` only in `message_delta`. It also counts the SDK's convenience events.

### `03_stream_tool_loop.py`

A two tool loop over the Acme handbook (`read_policy`, `save_note`) with `eager_input_streaming: True`. Text and tool input fragments print as they arrive; the note body visibly streams into the `save_note` call. Each turn checks `stop_reason`, validates input with pydantic, guards against `ValueError`, and appends the full final message to the history.

## Key takeaways

- Stream whenever a person is watching or the output is long. The SDK refuses non streamed requests with a large `max_tokens`.
- `messages.stream()` plus `get_final_message()` gives you live output and the complete message. Print with `flush=True`.
- Events: `message_start`, then per block `content_block_start`, deltas, `content_block_stop`, then `message_delta` with `stop_reason`, then `message_stop`.
- With tools, set `eager_input_streaming: True` on your own tools, check `stop_reason` before running anything, and validate every input yourself.
- The history gets the full final message, exactly as in a non streamed loop.

## Common misconceptions

- **"Streaming makes responses cheaper or faster."** Same tokens, same price, same total time. It only changes when you see them.
- **"`text_stream` is the whole response."** It skips thinking and tool calls. Read `get_final_message().content` for those.
- **"If it parsed, the tool input is complete."** With eager streaming a truncated input often parses fine. `stop_reason` and your own validation are what catch it.
- **"I can append the text I printed as the assistant turn."** That drops the thinking and tool use blocks. Append `final.content`.

Next: [Lesson 01c: Server tools](../01c_server_tools/README.md).
