# Lesson 02 — Structured Outputs

Most non-trivial agents need to **return typed data**, not just free text. A research agent returns a `Report`. An extraction pipeline returns a `Contact`. A planner returns a list of `Step`s. Free text is for humans; structured data is for code.

Lesson 01 ended with a hint: a *forced tool call* can be repurposed as a structured-output mechanism. This lesson makes that explicit and shows how to handle the cases where the model doesn't comply.

## Two routes to structured output

### Route 1: prompt-and-parse (the brittle way)

Ask the model to return JSON in its text response, then parse it.

```python
response = chat([{"role": "user", "content": "Return JSON: {...}"}])
data = json.loads(response.text)  # 🤞
```

This works ~85% of the time. The other 15% — surrounding prose, trailing commas, the word `json` before the code fence, hallucinated fields — will ruin your week in production.

### Route 2: forced tool call (the reliable way)

Define a tool whose `input_schema` is the shape you want. Force the model to call it with `tool_choice`. The "tool" doesn't actually do anything — it's a typed mailbox the model has to drop its answer into.

```python
response = chat(messages, tools=[submit_report_tool],
                tool_choice={"type": "tool", "name": "submit_report"})
data = response.tool_uses[0]["input"]  # already a dict matching the schema
report = Report.model_validate(data)
```

The model's training nudges it to fill tool inputs correctly. The wire format enforces the contract. You get back a dict that matches your schema in the overwhelming majority of cases — and when it doesn't, pydantic tells you exactly what went wrong, so you can retry intelligently.

## Pydantic is the contract

Use pydantic on both ends:

1. **`Model.model_json_schema()`** — generate the JSON schema you send to the model.
2. **`Model.model_validate(raw_dict)`** — validate the model's response and get a typed object.

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

- **Hallucinated `"N/A"` in required fields.** Required + enum + clear description fixes it.
- **Numbers as strings.** `"price": "12.99"` instead of `12.99`. Add `examples=[12.99]` and type the field correctly.
- **Silent omission of optional objects.** Model skips an entire nested object. Either make it required, or accept and detect the omission.
- **Successful parse, wrong content.** The schema validates but the answer is just wrong. Schema validation is *necessary* but not *sufficient*. Lesson 07 (evaluation) handles this.
- **Long responses cut off.** Increase `max_tokens` or shorten the schema.

## The repair pattern

When validation fails, you have one of two responses:

1. **Crash.** Sometimes correct — the upstream caller should know.
2. **Re-prompt with the error.** Show the model what it got wrong, ask again, cap retries.

```python
for attempt in range(3):
    response = chat(messages, tools=[tool], tool_choice=forced)
    try:
        return Report.model_validate(response.tool_uses[0]["input"])
    except ValidationError as e:
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": f"That failed validation: {e.errors()}. Try again."
        })
raise RuntimeError("Couldn't get a valid response in 3 attempts")
```

This pattern shows up everywhere in agent code. Lesson 05 generalizes it.

## Hands-on

### `01_extract_prompt.py` — the naive way

Ask for JSON in the prompt. Parse with `json.loads`. Run on a few inputs and *watch it fail*. The failures are the lesson — pay attention to *how* they fail.

### `02_extract_tool.py` — the reliable way

Same task, same schema, but using a forced tool call. Compare reliability across the same inputs.

### `03_nested_schema.py` — designing for the model

Extract a meeting agenda: `Meeting(title, attendees: list[Attendee], items: list[AgendaItem])`. Demonstrates field descriptions, enums, optional fields, and how schema *shape* changes what you get back.

### `04_repair_loop.py` — automated repair

Wrap structured extraction in a retry loop that feeds validation errors back to the model. Up to N attempts. The pattern you'll reach for every time validation matters.

## Key takeaways

- Free-text JSON parsing is unreliable. Use forced tool calls instead.
- Pydantic is the contract — same model generates the schema and validates the response.
- Schemas are prompts. Field descriptions, enums, and examples are not decoration.
- Validation passing ≠ content correct. Evaluation (Lesson 07) measures correctness.
- Always have a repair path when validation can fail.

## Common misconceptions

- **"JSON mode = structured output."** JSON mode guarantees *valid JSON*. It does not guarantee *your schema*.
- **"Bigger, more thorough schemas are better."** Smaller and opinionated wins. Fewer fields, harder constraints, clear enums.
- **"Once it parses, I'm done."** No — parse-passing is the floor, not the ceiling. The content can still be confidently wrong.

Next: [Lesson 02b — Prompt caching](../02b_prompt_caching/README.md), then [Lesson 03a — Retrieval basics](../03a_retrieval_basics/README.md).
