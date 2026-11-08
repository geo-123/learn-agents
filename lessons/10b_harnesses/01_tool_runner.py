"""Lesson 10b: the Lesson 03b RAG agent, rebuilt with the SDK's Tool Runner.

In 03b you wrote the registry, the schema, the dispatcher, the error wrapper
and the loop: about 150 lines. The Tool Runner (a beta helper inside the
regular `anthropic` package) supplies all of that. You write a typed Python
function; `@beta_tool` turns its signature and docstring into the tool
schema; `client.beta.messages.tool_runner(...)` runs the loop: call the
model, run your function for each tool_use, send the results back, repeat
until the model stops calling tools.

What you still own: the tool itself, the system prompt, the step budget
(`max_iterations`) and anything you want to watch, which you get by
iterating the runner: it yields every assistant message as it arrives.

Run:
    uv run python lessons/10b_harnesses/01_tool_runner.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from anthropic import beta_tool
from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src._llm import DEFAULT_MODEL, client

# The TF IDF index from Lesson 03a, loaded the same way 03b loads it.
spec = importlib.util.spec_from_file_location(
    "_tfidf", HERE.parent / "03a_retrieval_basics" / "02_tfidf_search.py")
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf
spec.loader.exec_module(_tfidf)
INDEX = _tfidf.build_index()


@beta_tool
def search_docs(query: str, k: int = 3) -> str:
    """Search the Acme company handbook. Cite the returned ids in your answer.

    Args:
        query: A natural language query.
        k: How many passages to return, 1 to 8.
    """
    hits = INDEX.search(query, k=max(1, min(k, 8)))
    return json.dumps([{"id": f"{c.doc}#{c.idx}", "text": c.text, "score": round(s, 3)}
                       for c, s in hits])


SYSTEM = (
    "You answer questions about Acme using ONLY the company handbook. Use "
    "search_docs, with separate queries for independent parts of a question. "
    "If the handbook has no answer, say so. Always cite the chunk ids you used."
)


def run_agent(question: str) -> str:
    runner = client().beta.messages.tool_runner(
        model=DEFAULT_MODEL,
        max_tokens=8000,
        max_iterations=6,  # the Lesson 00 step budget, now a parameter
        system=SYSTEM,
        tools=[search_docs],
        messages=[{"role": "user", "content": question}],
    )
    final = ""
    for step, message in enumerate(runner, start=1):  # one assistant message per turn
        for block in message.content:
            if isinstance(block, BetaToolUseBlock):
                print(f"  step {step}: search_docs({block.input})")
            elif isinstance(block, BetaTextBlock):
                final = block.text
        print(f"  step {step}: stop={message.stop_reason} "
              f"in={message.usage.input_tokens} out={message.usage.output_tokens}")
    return final


if __name__ == "__main__":
    print("schema generated from the function signature and docstring:")
    print(json.dumps(search_docs.to_dict(), indent=2))
    for q in ["How many vacation days do I get and do they roll over?",
              "Who pays for my gym membership?"]:
        print(f"\n{'=' * 70}\nQ: {q}")
        print(f"\nA: {run_agent(q)}")
