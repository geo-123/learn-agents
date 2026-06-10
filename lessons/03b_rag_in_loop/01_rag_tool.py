"""Lesson 03b — RAG as a tool the agent calls.

The TfidfIndex from 03a is exposed as `search_docs`. The agent decides
when to search, what to search for, and how to weave the result into an
answer. No preprocessing magic — just a tool.

Run:
    uv run python lessons/03b_rag_in_loop/01_rag_tool.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_tfidf", HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py",
)
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat


# Build the index once at module load.
INDEX = _tfidf.build_index()


# --- Retriever interface --------------------------------------------------
# A protocol both implementations (TF-IDF here, OpenAI in 02_) satisfy.


@dataclass
class Hit:
    doc: str
    idx: int
    text: str
    score: float


def search(query: str, k: int = 3) -> list[Hit]:
    """The Retriever. Currently TF-IDF; swappable per Lesson 03b's lesson."""
    return [
        Hit(c.doc, c.idx, c.text, score)
        for c, score in INDEX.search(query, k=k)
    ]


# --- Tool definition -----------------------------------------------------


class SearchArgs(BaseModel):
    query: str = Field(description="A natural-language query.")
    k: int = Field(default=3, ge=1, le=8, description="How many chunks to return.")


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


def search_docs(args: SearchArgs) -> dict:
    hits = search(args.query, k=args.k)
    return {
        "hits": [
            {"id": f"{h.doc}#{h.idx}", "text": h.text, "score": round(h.score, 3)}
            for h in hits
        ]
    }


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool(
        name="search_docs",
        description=(
            "Search the Acme company handbook for relevant passages. Returns "
            "chunks ranked by lexical similarity. Each hit has an 'id', the "
            "passage 'text', and a relevance 'score'. Cite the 'id' values "
            "you used in your final answer."
        ),
        args_model=SearchArgs,
        fn=search_docs,
    ),
]}


def dispatch(tool_use: dict) -> dict:
    name, use_id = tool_use["name"], tool_use["id"]
    tool = REGISTRY.get(name)
    if tool is None:
        return _err(use_id, f"unknown tool {name!r}")
    try:
        args = tool.args_model.model_validate(tool_use.get("input", {}) or {})
        result = tool.fn(args)
    except ValidationError as e:
        return _err(use_id, f"invalid args: {e.errors()}")
    except Exception as e:  # noqa: BLE001
        return _err(use_id, f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": use_id, "content": json.dumps(result)}


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


SYSTEM = (
    "You answer questions about Acme using ONLY the company handbook. "
    "Use search_docs to look up relevant passages. If multiple aspects of "
    "the question are independent, search for them with separate queries "
    "(in parallel if possible). If the handbook has no answer, say so. "
    "Always cite the chunk ids you used."
)


def run_agent(question: str, *, max_steps: int = 6) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        if response.text:
            print(f"text: {response.text}")
        for tu in response.tool_uses:
            print(f"  search: {tu['input'].get('query')!r}  k={tu['input'].get('k', 3)}")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

    return "(max steps reached)"


if __name__ == "__main__":
    questions = [
        "How many vacation days do I get and do they roll over?",
        "What do I do if I get a phishing email AND if I think I clicked it?",
        "Who pays for my gym membership?",
        "What does Acme make?",
    ]
    for q in questions:
        print(f"\n{'='*70}\nQ: {q}")
        answer = run_agent(q)
        print(f"\nA: {answer}")
