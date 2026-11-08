# Lesson 01: Tool / Function Calling

In Lesson 00 the loop was empty: the model talked to itself until it said DONE. Now we give it the ability to *do* things by defining tools.

This is the most mechanically important lesson in the course. Tool calling is the protocol that lets an LLM cause effects in the real world, and almost every "agentic capability" (RAG, code execution, web access, calling internal APIs) is just tool calling with a different tool on the other end.

## What tool calling actually is

You describe to the model what functions are available. Each function gets:

- A **name** (`get_weather`)
- A **description** (a natural language explanation of when to use it)
- A **JSON schema** for its input parameters

When the model decides a tool is appropriate, it doesn't *call* anything. It emits a structured block in its response that looks like:

```
{ "type": "tool_use",
  "id": "toolu_abc123",
  "name": "get_weather",
  "input": { "city": "Lisbon" } }
```

That block is **just text the model produced**. Your runtime is responsible for:

1. **Parsing** the block out of the response.
2. **Validating** that `name` is a real tool and `input` matches the schema. *The model lies sometimes.*
3. **Executing** the actual Python function with those arguments.
4. **Returning** the result to the model as a `tool_result` block in the next user turn.
5. **Looping** until the model produces a response with no more `tool_use` blocks.

That last bullet is the new stop condition. In Lesson 00 the stop was "the assistant said DONE." Now it's "the assistant has nothing more to call."

## The tool schema is a prompt

Writing tool schemas feels like API design but it isn't. Schemas are *prompts*. The model reads them every turn and decides (based on the description) whether and how to use each tool. The two biggest mistakes:

1. **Terse descriptions.** `"Get the weather."` tells the model nothing about *when* to use it, *what* it returns, or *what's not* a valid input.
2. **Overpermissive schemas.** A `city: string` field with no constraints means the model will pass `"the city the user asked about"` as an argument. Use enums, regex, descriptions, and required fields liberally.

Good schemas teach the model two things: **what** the tool does (so it can call it correctly) and **when** to use it (so it doesn't call the wrong tool, or call no tool when it should).

## Tool choice: who decides whether to call?

Most APIs expose a `tool_choice` parameter. Four typical values:

- **`auto`**: the model decides whether to call a tool or just respond. Default.
- **`any`**: model must call *some* tool. No free text answer allowed.
- **`tool: <name>`**: model must call this specific tool.
- **`none`**: no tools allowed this turn.

The two forcing values (`any` and `tool`) are on their way out. The newest Claude models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject them with a 400; older models still accept them. The portable pattern is `auto` plus three things: a prompt that says which tool to use, a tool schema with `strict: true` (so a call, when it comes, is guaranteed to match the schema), and a check in your code that a `tool_use` came back, with one reprompt if it did not. And if you only wanted forcing to get a structured answer, you no longer need a tool at all: structured outputs constrain the answer itself (Lesson 02).

The right choice depends on what the agent's job is. A chat agent uses `auto`. A step that must not touch tools (a planning turn, a final summary) uses `none`.

### Strict tool schemas

Without `strict`, the schema is guidance: the model almost always follows it, and your code validates on receipt. With `"strict": True` on the tool definition, the API guarantees the `tool_use` input matches the schema exactly. Strict schemas must set `"additionalProperties": false` and list every property as required, and they support a subset of JSON Schema (ranges such as `minimum` are still yours to check).

## Parallel tool calls

A capable model can emit *multiple* `tool_use` blocks in one response if they're independent. "What's the weather in Lisbon and the time in Tokyo?" should produce both calls at once, not one after the other. Your dispatcher must handle a *list* of tool_use blocks per turn (dispatching them in parallel or in sequence) and return one `tool_result` per call, all bundled in the next user turn.

If you dispatch sequentially when the calls are independent, you double your latency for no reason.

## The hand roll vs. framework moment

Look at `02_tool_dispatch.py` in this lesson. The whole pattern (registry, validate, dispatch, append results, loop) is about 80 lines of Python. Every "agent framework" wraps essentially this. Once you can write it from memory, frameworks become a question of "do their conveniences outweigh the indirection?", not a magic black box.

## Failure modes (these will bite you)

- **Hallucinated tool names.** The model invents a `search_database` tool you didn't define. Your dispatcher must return an error tool_result and let the model recover, not crash.
- **Hallucinated arg fields.** You defined `city: str`; the model emits `{"city": "Lisbon", "country": "Portugal"}`. Strict pydantic validation catches this. Forgiving validation lets it pass and bites you later.
- **Missing `tool_result`.** The model produced a `tool_use`. You hit an exception in your dispatcher and forgot to send back a tool_result with `is_error=True`. The next API call rejects the conversation because there's a dangling tool_use.
- **Tool result blows up the context.** Your `read_file` tool returned 200 KB. After 3 such calls the model context is full. Truncate, summarize, or return references.
- **No timeout on the tool.** Your tool's HTTP call hangs. The loop hangs. Wrap every tool with a timeout.
- **Same tool called 20 times in a row with the same args.** The model is stuck in a loop. Detect via duplicate detection in the loop and inject a "stop calling that tool, here's what's wrong" message.

## Hands on

Run these in order. Each is standalone.

### `01_single_tool.py`: one tool, one turn

A `get_weather(city)` function with three hard coded cities. Watch the request/response shape. See the `tool_use` and `tool_result` blocks travel across two turns.

### `02_tool_dispatch.py`: the load bearing file

A reusable pattern: a registry of `{name: (callable, pydantic_model)}` tools, a `dispatch()` function that validates arguments, calls the function, and returns a properly shaped `tool_result` block. The agent loop becomes vendor agnostic: adding a new tool is now one entry in a dict.

**Read this file twice.** It's the pattern you'll reuse in every other lesson.

### `03_parallel_tools.py`: multiple calls per turn

Add `get_time(tz)` and `convert_currency(amount, from_, to)`. Ask "What's the weather in Lisbon, the time in Tokyo, and how much is 100 EUR in JPY?": and verify the model emits all three `tool_use` blocks in one turn.

### `04_strict_tool.py`: steering the model to a specific tool

A `submit_analysis` tool with `strict: true`, `tool_choice` left on `auto`, a system prompt that names the tool, and a check that the call came back (one reprompt if not, appended to the history rather than edited into it). The tool is *not* there for side effects: it gives the model a schema shaped place to put its answer. Lesson 02 shows the more direct way to get the same typed answer.

## Key takeaways

- A tool is a function + a JSON schema. The schema is a prompt.
- The model emits *requests* for tool calls; your code executes them.
- The tool using agent's loop stops when there are no more `tool_use` blocks.
- Every `tool_use` must be answered with a `tool_result` (success or error).
- Parallel tool calls cut latency dramatically. Use them.
- `tool_choice` is how you control *when* tools fire; on current models steer with `auto`, the prompt and `strict: true` rather than forcing.

## Common misconceptions

- **"The model runs my function."** It doesn't. It outputs a request for you to run it.
- **"My schema will keep the model honest."** Only if you actually validate on receipt.
- **"Adding more tools makes the agent smarter."** Beyond ~10 tools, the model gets *worse* at choosing. Curate.

Next: [Lesson 01b: Streaming](../01b_streaming/README.md).
