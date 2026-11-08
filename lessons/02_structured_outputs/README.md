# Lesson 02: Structured Outputs

Most nontrivial agents need to **return typed data**, not just free text. A research agent returns a `Report`. An extraction pipeline returns a `Contact`. A planner returns a list of `Step`s. Free text is for humans; structured data is for code.

Lesson 01 ended with a tool used as a typed mailbox for the answer. This lesson shows the direct way: the API constrains the answer itself to your schema. Then it covers what the schema cannot catch, and how to repair it.

## Routes to structured output

### Route 1: prompt and parse (the brittle way)

Ask the model to return JSON in its text response, then parse it.

```python
response = chat([{"role": "user", "content": "Return JSON: {...}"}])
data = json.loads(response.text)  # 🤞
```

This works ~85% of the time. The other 15% (surrounding prose, trailing commas, the word `json` before the code fence, hallucinated fields) will ruin your week in production.

### Route 2: native structured outputs (the reliable way)

Pass the schema with the request. The API constrains decoding so the answer *is* JSON matching your schema: not "usually", by construction.

```python
response = chat(messages, output_schema=Report)   # a Pydantic model class
report = response.parsed                           # a validated Report
```

**How the guarantee works (constrained decoding).** Recall Lesson 00: the model writes one token at a time, each picked from a probability distribution over the whole vocabulary. Route 1 lets it pick any token, so a stray "Sure, here is the JSON:" is always possible. With a schema, the API first compiles your JSON schema into a grammar (a small state machine of what may legally come next: after `{"name": ` only a string may start, after the last required field only `}` may follow). At every step it then *masks* every token the grammar forbids, setting its probability to zero, and samples from what is left. An invalid token can never be chosen, so the finished text always parses and always matches the schema. Two consequences fall straight out of the mechanism: the grammar can only express what the schema subset can express (types, required fields, enums, nesting; not "end after start"), and a cut off answer (`max_tokens`) is still a prefix that never got its closing brace. You could build a crude version yourself (Route 1 plus validation plus the repair loop below is exactly that, done after the fact); the API does it during generation, token by token.

Open `src/_llm.py` and you can see every step the helper takes: it turns your Pydantic class into a JSON schema (`Report.model_json_schema()`), trims it to the subset the API can enforce (the SDK's `transform_schema`), sends it as `output_config={"format": {"type": "json_schema", "schema": ...}}`, then runs `json.loads` on the answer and `Report.model_validate` on the result. The SDK's `client.messages.parse(output_format=Report)` packages exactly these steps. You can also pass a plain JSON schema dict; then `.parsed` holds the parsed JSON.

Two things the guarantee does not cover: an answer that never finished (`stop_reason` is `max_tokens`) and a declined request (`stop_reason` is `refusal`). Check `stop_reason` before trusting `.parsed`. And JSON Schema features outside the supported subset (numeric ranges, string patterns) are not enforced: the SDK's `transform_schema` moves them into the field description as hints, and your validator still has to check them.

### Route 3: a tool as a mailbox (the older way)

Before native structured outputs, the reliable trick was a tool whose `input_schema` is the shape you want, *forced* with `tool_choice={"type": "tool", "name": ...}`. You will see it in a lot of existing code. The newest models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject forced tool choice with a 400, so prefer Route 2. When the answer really should be a tool call (the agent acts, it does not just answer), use a tool with `strict: true` and `auto` tool choice, as in Lesson 01's `04_strict_tool.py`: the arguments are then guaranteed to match the schema. Structured outputs and strict tools can be combined in one request.

## Pydantic is the contract

Use pydantic on both ends:

1. **`Model.model_json_schema()`**: generate the JSON schema you send to the model (the SDK does this for you when you pass the class).
2. **`Model.model_validate(raw_dict)`**: validate the model's response and get a typed object (`parse()` does this for you too).

The same class defines both. You can't have a drift between "what the model thought it was filling in" and "what your code expects."

## Schema design rules

What you put in the schema directly affects what you get back. The model reads it as a prompt.

- **Enums over free strings.** `Literal["positive", "neutral", "negative"]` beats `str` every time.
- **Descriptions on every field.** `Field(description="...")` is read by the model. Don't leave it blank.
- **Flat over nested.** A flat schema with 10 fields is more reliably filled than a tree with 5 levels.
- **Required vs. optional, deliberately.** Required fields get filled with `"unknown"` if the model genuinely doesn't know. Make fields optional when the answer might honestly be "no information available."
- **Numeric units in field names, not values.** `price_usd: float` not `price: str  # "$12.99"`.
- **Examples in the description.** `Field(description="ISO 8601 date, e.g. 2025-01-15")` outperforms `Field(description="The date")`.

## Failure modes

With structured outputs, wrong *shapes* (strings for numbers, missing required fields, extra fields) are gone. What remains:

- **Hallucinated `"N/A"` in required fields.** Required + enum + clear description fixes it; make a field nullable when "unknown" is an honest answer.
- **Rules the schema cannot express.** A pattern, a range, "end after start". Validate them in your code.
- **Silent omission of optional objects.** Model skips an entire nested object. Either make it required, or accept and detect the omission.
- **Successful parse, wrong content.** The schema validates but the answer is just wrong. Schema validation is *necessary* but not *sufficient*. Lesson 07 (evaluation) handles this.
- **Long responses cut off.** `stop_reason` is `max_tokens` and there is nothing to parse. Increase `max_tokens` (thinking counts toward it), lower the effort, or shorten the schema.

## The repair pattern

When your validator rejects an answer, you have one of two responses:

1. **Crash.** Sometimes correct: the upstream caller should know.
2. **Reprompt with the error.** Show the model what it got wrong, ask again, cap retries.

```python
for attempt in range(3):
    response = chat(messages, output_schema=LOOSE_SCHEMA)
    messages.append({"role": "assistant", "content": response.content})
    try:
        return Report.model_validate(response.parsed)
    except ValidationError as e:
        messages.append({
            "role": "user",
            "content": f"That failed validation: {e.errors()}. Try again."
        })
raise RuntimeError("Couldn't get a valid response in 3 attempts")
```

Append to the history; never rewrite earlier turns. Current models bind their thinking blocks to the exact conversation, and an edited history throws that reasoning away (or, on newer accounts, returns an error).

This pattern shows up everywhere in agent code. Lesson 05 generalizes it.

## Hands on

### `01_extract_prompt.py`: the naive way

Ask for JSON in the prompt. Parse with `json.loads`. Run on a few inputs and *watch it fail*. The failures are the lesson: pay attention to *how* they fail.

### `02_extract_structured.py`: the reliable way

Same task, same schema, but with native structured outputs (`output_schema=Contact`, effort `low`). Compare reliability across the same inputs.

### `03_nested_schema.py`: designing for the model

Extract a meeting agenda: `Meeting(title, attendees: list[Attendee], items: list[AgendaItem])`. Demonstrates field descriptions, enums, optional fields, and how schema *shape* changes what you get back.

### `04_repair_loop.py`: automated repair

The model is constrained to a loose schema; a stricter client side validator rejects the conventional answer, and a retry loop feeds the validation errors back. Up to N attempts. The pattern you'll reach for every time your rules go beyond what a schema can say.

## Key takeaways

- Free text JSON parsing is unreliable. Use native structured outputs (or a `strict: true` tool when the answer is an action).
- Pydantic is the contract: same model generates the schema and validates the response.
- Schemas are prompts. Field descriptions, enums, and examples are not decoration.
- Validation passing ≠ content correct. Evaluation (Lesson 07) measures correctness.
- Always have a repair path when validation can fail.

## Common misconceptions

- **"JSON mode = structured output."** JSON mode guarantees *valid JSON*. It does not guarantee *your schema*. Structured outputs do.
- **"Forcing a tool call is how you get structured output."** It was. On current models it is a 400; the answer format is now a request parameter.
- **"Bigger, more thorough schemas are better."** Smaller and opinionated wins. Fewer fields, harder constraints, clear enums.
- **"Once it parses, I'm done."** No: parse passing is the floor, not the ceiling. The content can still be confidently wrong.

Next: [Lesson 02b: Prompt caching](../02b_prompt_caching/README.md), then [Lesson 03a: Retrieval basics](../03a_retrieval_basics/README.md).
