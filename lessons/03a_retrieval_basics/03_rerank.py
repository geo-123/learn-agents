"""Lesson 03a — LLM reranking.

Two-stage retrieval: cheap top-N first, then an LLM picks the best K.
Reranking is the highest-leverage quality lever most beginners skip.

We reuse the TfidfIndex from 02_tfidf_search.py, fetch a wider candidate
set, and ask the model to score each candidate's *relevance* to the query.

Run:
    uv run python lessons/03a_retrieval_basics/03_rerank.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Import TfidfIndex from sibling file by path (it isn't a package).
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_tfidf", HERE / "02_tfidf_search.py")
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat


class RerankScore(BaseModel):
    """One scored candidate."""
    chunk_index: int = Field(description="The 0-based index of the chunk in the candidate list.")
    relevance: Literal[0, 1, 2, 3, 4, 5] = Field(
        description="0 = irrelevant, 5 = directly answers the question."
    )
    why: str = Field(description="One short phrase.")


class RerankResult(BaseModel):
    scores: list[RerankScore]


SUBMIT = {
    "name": "submit_ranking",
    "description": "Submit relevance scores for every candidate chunk.",
    "input_schema": RerankResult.model_json_schema(),
}


def rerank(query: str, candidates: list, k: int = 3) -> list:
    """Re-order candidates by LLM-judged relevance; return the top k."""
    listing = "\n\n".join(
        f"[{i}] (doc={c.doc}#{c.idx})\n{c.text}"
        for i, (c, _score) in enumerate(candidates)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Candidate chunks:\n{listing}\n\n"
        "Score each chunk 0-5 for how directly it answers the question. "
        "Be strict: 5 only if the chunk literally contains the answer."
    )
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        tools=[SUBMIT],
        tool_choice={"type": "tool", "name": "submit_ranking"},
    )
    parsed = RerankResult.model_validate(response.tool_uses[0]["input"])

    # Sort by relevance desc; keep original order as the tie-breaker.
    ranked = sorted(
        parsed.scores,
        key=lambda s: (-s.relevance, s.chunk_index),
    )
    return [(candidates[s.chunk_index][0], s.relevance, s.why) for s in ranked[:k]]


if __name__ == "__main__":
    idx = _tfidf.build_index()
    query = "I want to take time off"  # Paraphrase that TF-IDF struggles with.

    raw = idx.search(query, k=8)
    print(f"\nQ: {query}")
    print("\n-- raw TF-IDF top 8 --")
    for c, s in raw:
        print(f"  [{c.doc}#{c.idx}] tfidf={s:.3f}  {c.text.splitlines()[0][:60]}")

    reranked = rerank(query, raw, k=3)
    print("\n-- reranked top 3 --")
    for c, score, why in reranked:
        print(f"  [{c.doc}#{c.idx}] rel={score}  | {why}")
        print(f"    {c.text.splitlines()[0][:80]}")
