"""Lesson 04b: server side compaction, asked for on demand.

Lesson 04 summarized history by hand. Here the API writes the summary.
The loop reads the Acme handbook one document per turn and asks a short
question about each. After every turn it checks how big the next request
will be. Past a (deliberately tiny) limit it sends one extra request with
`compaction={"type": "summarize"}`. The answer is a single `compaction`
block: a summary plus a signature. That block then replaces the whole
history and the conversation carries on from it.

Things to notice in the output:
  * the request size grows every turn, then drops right after a compaction;
  * the compaction block goes back to the API exactly as it came (the
    signature is checked), so the code appends `response.content` and
    never rebuilds the block from its text;
  * the last question is about the FIRST document, which is no longer in
    the history. It is answered from the summary, which is why the
    summarizer instructions say which facts to keep.

Model: claude-haiku-5-5, the cheapest model the compaction docs list.
On demand compaction has no minimum size, so a low limit is fine here.
(Threshold compaction, the other variant, needs a trigger of at least
50,000 input tokens; see the README.)

Run:
    uv run python lessons/04b_long_running_context/01_compaction.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import client

MODEL = "claude-haiku-5-5"
BETAS = ["compact-2026-09-04"]  # needed on the compaction request AND on every request carrying the block
COMPACT_AT_TOKENS = 2500        # tiny on purpose; in production, set it near your real input budget
CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"

SYSTEM = "You are a careful analyst of the Acme employee handbook. Answer in one or two sentences."

# Replaces the default summarization prompt entirely, so say what must survive.
SUMMARY_INSTRUCTIONS = (
    "Summarize this conversation about the Acme handbook. Keep every number, "
    "limit, deadline and named tool exactly as written, grouped by document. "
    "Do not call tools; reply with the summary text only."
)


def ask(history: list[Any], text: str) -> Any:
    history.append({"role": "user", "content": text})
    response = client().beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        betas=BETAS,
        output_config={"effort": "low"},
        messages=history,
    )
    # Append the full content (thinking blocks included), never just the text.
    history.append({"role": "assistant", "content": response.content})
    return response


def compact(history: list[Any]) -> list[Any]:
    """One compaction request. Returns the new history, or the old one if no summary came back."""
    summary = client().beta.messages.create(
        model=MODEL,
        max_tokens=4000,  # caps the summary call, thinking included
        system=SYSTEM,    # same system prompt as the conversation
        betas=BETAS,
        messages=history,
        compaction={"type": "summarize", "instructions": SUMMARY_INSTRUCTIONS},
    )
    if summary.stop_reason != "compaction":
        print(f"  [no summary: stop_reason={summary.stop_reason}; keep going, compact later]")
        return history
    block = next(b for b in summary.content if b.type == "compaction")
    print(f"  [compacted: {len(history)} messages -> 1 compaction block]")
    print(f"  [summary starts: {(block.content or '')[:160]!r}...]")
    # The block goes first, alone; the summarized messages are dropped.
    return [{"role": "assistant", "content": summary.content}]


def main() -> None:
    docs = sorted(CORPUS.glob("*.md"))
    history: list[Any] = []
    for turn, doc in enumerate(docs, start=1):
        text = doc.read_text(encoding="utf-8")
        response = ask(history, f"Here is {doc.name}:\n\n{text}\n\nWhat is the single most important rule in it?")
        answer = "".join(b.text for b in response.content if b.type == "text")
        # The next request resends this reply too, so count both sides.
        next_size = response.usage.input_tokens + response.usage.output_tokens
        print(f"\nturn {turn} ({doc.name})  in={response.usage.input_tokens}  next request about {next_size} tokens")
        print(f"  {answer.strip()[:200]}")
        if next_size > COMPACT_AT_TOKENS:
            history = compact(history)

    first = docs[0].name
    response = ask(history, f"From memory: what exact numbers did {first} contain?")
    answer = "".join(b.text for b in response.content if b.type == "text")
    print(f"\nfinal question about {first}  in={response.usage.input_tokens}")
    print(f"  {answer.strip()}")


if __name__ == "__main__":
    main()
