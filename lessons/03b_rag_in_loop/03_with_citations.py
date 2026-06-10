"""Lesson 03b — RAG with verifiable citations.

The agent's final answer must declare the chunk ids it used. Then we
verify, after the fact, that those chunks actually contain phrases
from the answer. A trust-but-verify pattern.

Run:
    uv run python lessons/03b_rag_in_loop/03_with_citations.py
"""
from __future__ import annotations

import importlib.util
import json
import re
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


INDEX = _tfidf.build_index()


# Cache the texts of every chunk by id for citation verification later.
CHUNK_TEXTS: dict[str, str] = {
    f"{c.doc}#{c.idx}": c.text for c in INDEX.chunks
}


# --- Tools ----------------------------------------------------------------


class SearchArgs(BaseModel):
    query: str
    k: int = Field(default=3, ge=1, le=8)


def search_docs(args: SearchArgs) -> dict:
    return {
        "hits": [
            {"id": f"{c.doc}#{c.idx}", "text": c.text, "score": round(s, 3)}
            for c, s in INDEX.search(args.query, k=args.k)
        ]
    }


class FinalAnswer(BaseModel):
    """Schema the model must fill to end the loop."""
    answer: str = Field(description="The answer in plain prose.")
    citations: list[str] = Field(
        description=(
            "List of chunk ids (e.g. 'vacation_policy.md#0') that you used "
            "to construct the answer. MUST be a subset of ids returned by "
            "search_docs in this conversation."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)


def submit_answer(args: FinalAnswer) -> dict:
    # The dispatcher will see this result, but the agent loop also reads
    # the tool_use directly. So this is only here for completeness.
    return {"acknowledged": True}


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


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("search_docs",
         "Search the company handbook. Returns hits with id, text, score.",
         SearchArgs, search_docs),
    Tool("submit_answer",
         "Submit your final answer with citations. Use this exactly once when done.",
         FinalAnswer, submit_answer),
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
    "Answer questions about Acme using ONLY the handbook. Use search_docs to "
    "look things up. When you have the answer, call submit_answer with a "
    "concise answer, the list of chunk ids you actually used, and your "
    "confidence (0-1). Do not invent citations."
)


def run_agent(question: str) -> FinalAnswer | None:
    """Loops until the model calls submit_answer or we hit max_steps."""
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(1, 7):
        response = chat(messages, system=SYSTEM, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        for tu in response.tool_uses:
            print(f"  -> {tu['name']}({list(tu['input'].keys())})")
            if tu["name"] == "submit_answer":
                return FinalAnswer.model_validate(tu["input"])

        if response.stop_reason != "tool_use":
            print(f"(no tool call this turn; text: {response.text!r})")
            break

        results = [dispatch(tu) for tu in response.tool_uses]
        messages.append({"role": "user", "content": results})

    return None


def verify_citations(answer: FinalAnswer) -> tuple[bool, str]:
    """Check that cited chunks exist and that the answer's key terms
    appear in at least one cited chunk. A crude verifier — Lesson 07
    will show better evaluators."""
    for cid in answer.citations:
        if cid not in CHUNK_TEXTS:
            return False, f"cited chunk {cid!r} does not exist"

    if not answer.citations:
        return False, "no citations provided"

    # Take "content words" (4+ chars) from the answer and check at least
    # some appear in the cited chunks.
    answer_words = set(re.findall(r"[a-zA-Z]{4,}", answer.answer.lower()))
    cited_text = " ".join(CHUNK_TEXTS[c].lower() for c in answer.citations)
    cited_words = set(re.findall(r"[a-zA-Z]{4,}", cited_text))
    overlap = answer_words & cited_words
    overlap_ratio = len(overlap) / max(len(answer_words), 1)
    if overlap_ratio < 0.2:
        return False, f"weak overlap ({overlap_ratio:.0%}) between answer and cited chunks"
    return True, f"ok ({len(answer.citations)} citation(s), {overlap_ratio:.0%} overlap)"


if __name__ == "__main__":
    questions = [
        "How many vacation days, and do they roll over?",
        "Who handles a SEV-1 incident, and how fast?",
        "What's the wellness benefit?",
    ]
    for q in questions:
        print(f"\n{'='*70}\nQ: {q}")
        result = run_agent(q)
        if result is None:
            print("(no final answer)")
            continue
        print(f"\nAnswer:    {result.answer}")
        print(f"Citations: {result.citations}")
        print(f"Confidence: {result.confidence:.2f}")
        ok, why = verify_citations(result)
        print(f"Verify:    {'PASS' if ok else 'FAIL'} — {why}")
