# Learn Agents — Agent Engineering from First Principles

A hands-on course that teaches the engineering discipline of building LLM agents. Each lesson is one Markdown explainer + runnable Python code you can read, modify, and break.

**What this is:** A path from "I know Python" to "I can build a production agent" by *implementing* every concept yourself before reaching for a framework.

**What this is not:** A LangChain tutorial. A LangGraph tutorial. An Anthropic SDK tutorial. The lessons use Anthropic's SDK as a thin transport behind `src/_llm.py`, but the *patterns you learn* (tool dispatch, RAG, retries, MCP) apply to any model and any SDK.

## Learning path

| # | Lesson | What you'll build |
|---|---|---|
| 00 | [Mental model](lessons/00_mental_model/README.md) | The agent loop in 30 lines |
| 01 | [Tool / function calling](lessons/01_tool_calling/README.md) | A tool dispatcher with validation |
| 02 | [Structured outputs](lessons/02_structured_outputs/README.md) | Pydantic-typed responses with self-repair |
| 03a | [Retrieval basics](lessons/03a_retrieval_basics/README.md) | A vector "DB" in numpy |
| 03b | [RAG in the loop](lessons/03b_rag_in_loop/README.md) | An agent that searches its own corpus |
| 04 | [Memory and state](lessons/04_memory_state/README.md) | Scratchpad → summarization → persistent KV |
| 05 | [Retries and errors](lessons/05_retries_errors/README.md) | Hand-rolled backoff + self-correcting loops |
| 06 | [Human-in-the-loop](lessons/06_human_in_loop/README.md) | Approval gates and durable pause/resume |
| 07 | [Evaluation and tracing](lessons/07_eval_tracing/README.md) | JSONL traces, eval fixtures, LLM-as-judge |
| 08 | [Guardrails](lessons/08_guardrails/README.md) | Input, output, and tool-arg guards |
| 09 | [MCP and tool servers](lessons/09_mcp/README.md) | A stdio MCP server + client |
| 10 | [Capstone](lessons/10_capstone/README.md) | A research agent that uses every prior topic |

Lessons build on each other; do them in order.

## Setup

This is a `uv` project. Python 3.13.

```powershell
# One-time: copy the example env and add your Anthropic key
Copy-Item .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Sync deps
uv sync

# Run any lesson script
uv run python lessons/00_mental_model/01_loop.py
```

### Corporate-network note

If `uv sync` fails with `invalid peer certificate: UnknownIssuer`, your network is intercepting TLS with a custom CA. Tell uv to trust the OS certificate store:

```powershell
[Environment]::SetEnvironmentVariable("UV_NATIVE_TLS", "true", "User")
# Then restart the shell.
```

## How to read each lesson

1. Read the lesson's `README.md` — that's the concept.
2. Run the numbered Python files in order. Each is standalone.
3. Break something. Re-read. Fix it.
4. Skim "Common misconceptions" and "Failure modes" at the bottom of each lesson.

## Cost

Lessons use `claude-haiku-4-5` by default. A full pass through the course costs well under $1 in API calls. Override the model with `LEARN_AGENTS_MODEL=claude-sonnet-4-5` in `.env` if you want sharper responses.

## Repository layout

```
Learn Agents/
├── README.md              ← you are here
├── .env.example           ← copy to .env
├── pyproject.toml
├── src/
│   └── _llm.py            ← the thin wrapper. The only place SDK calls live.
├── data/
│   └── corpus/            ← small text corpus for RAG lessons
└── lessons/
    ├── 00_mental_model/   ← each lesson is a dir with README.md + .py files
    ├── 01_tool_calling/
    └── ...
```
