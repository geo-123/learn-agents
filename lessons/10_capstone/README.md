# Lesson 10 — Capstone

Every prior lesson taught one concern in isolation. Real agents combine them. The capstone is a small but complete **research agent** that uses every prior topic in one program:

| Topic | Where it shows up |
|---|---|
| Mental model (00) | The loop is the backbone |
| Tool calling (01) | `search_docs`, `remember`, `recall`, `ask_user`, `submit_report` |
| Structured outputs (02) | `ResearchReport` schema with citations and confidence |
| RAG (03) | `search_docs` against the Acme corpus |
| Memory (04) | `remember` / `recall` for user-specific facts |
| Retries (05) | Transport retry + schema-error repair around `submit_report` |
| HITL (06) | `ask_user` when the question is ambiguous |
| Eval & tracing (07) | JSONL spans + 5 fixture cases |
| Guardrails (08) | Input guard refuses off-topic; tool-arg guards |
| MCP (09) | (Optional) tools served from an MCP subprocess |

The lesson is intentionally **harder to read than any single previous file**. That's the point — production agents are integrations, not single concerns.

## The agent's job

Given a user question:

1. Check input guard. Refuse if off-topic or contains PII.
2. If ambiguous, call `ask_user` to clarify (max once).
3. Search the corpus. May search multiple times with refined queries.
4. Optionally `remember` a user fact for next time.
5. Submit a `ResearchReport` with answer, citations, and confidence.
6. The harness verifies citations actually exist and writes a JSONL trace.

## Failure paths it has to handle

- Question is off-topic → input guard returns refusal; no LLM call made.
- Search returns nothing relevant → agent should say so, not invent.
- `submit_report` validation fails → retry with the error fed back (cap 2 retries).
- Transport error → exponential backoff (cap 4 attempts).
- Budget exhausted → graceful stop with partial answer.
- Citation list references non-existent chunks → mark report as untrustworthy.

## Files

- `capstone_agent.py` — the agent. ~250 lines. Reads like a synthesis of every prior file.
- `fixtures.jsonl` — 5 eval cases (similar to Lesson 07).
- `run_evals.py` — runs the agent against fixtures, writes traces, prints a summary.

## How to run

```powershell
# One question, interactive
uv run python lessons/10_capstone/capstone_agent.py "How many vacation days do I get?"

# Eval suite
uv run python lessons/10_capstone/run_evals.py
```

Traces land in `traces/capstone_<ts>.jsonl`. View with:

```powershell
uv run python lessons/07_eval_tracing/02_trace_viewer.py traces/<file>.jsonl
```

## What to do next

This is the end of the course as a linear path. Suggested extensions to deepen each topic:

- **Tools**: write a tool that calls a real API (e.g., GitHub). Handle auth and rate limits.
- **RAG**: swap the TF-IDF retriever for OpenAI embeddings or sentence-transformers. Measure the quality delta on the fixtures.
- **Memory**: replace the JSON file with SQLite. Add a `forget(key)` tool. Measure retention quality with a multi-session fixture.
- **HITL**: wire the durable-pause harness to a small FastAPI app so a human can approve via HTTP.
- **Tracing**: ship the JSONL spans to a real backend (Phoenix, Langfuse).
- **Guardrails**: add a fine-grained PII classifier. Measure false positive rate on the fixtures.
- **MCP**: write a second MCP server in a different language (Rust, TypeScript) and connect both at once.

You now know enough to read framework code (LangChain, LangGraph, AutoGen, CrewAI) as variations on the patterns in this course. Read one; identify which Lesson concept maps to which abstraction. That exercise makes frameworks tools, not magic.
