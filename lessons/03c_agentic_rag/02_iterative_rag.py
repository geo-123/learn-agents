"""Lesson 03c: Iterative (agentic) RAG.

The agent searches, reflects on whether the retrieved chunks actually
answer the question, and decides between answering and searching again
with a refined query. The reflection is the difference between this and
03b's one shot retrieve then answer.

The trace prints every step so you can see the loop in action: query,
hit ids, the model's reasoning, and ultimately the final answer with
citations.

Run:
    uv run python lessons/03c_agentic_rag/02_iterative_rag.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Reuse the hybrid retriever from 01_hybrid_bm25.py.
_spec = importlib.util.spec_from_file_location(
    "_hybrid", HERE / "01_hybrid_bm25.py",
)
assert _spec and _spec.loader
_hybrid = importlib.util.module_from_spec(_spec)
sys.modules["_hybrid"] = _hybrid
_spec.loader.exec_module(_hybrid)

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat


# Build both indexes once at module load.
CHUNKS = _hybrid.build_corpus()
TFIDF = _hybrid._tfidf.TfidfIndex(CHUNKS)
BM25 = _hybrid.BM25Index(CHUNKS)


def hybrid_search(query: str, k: int = 4) -> list[dict]:
    """Run both rankers, fuse with RRF, return the top k as plain dicts.

    The agent doesn't know or care that two rankers are involved. From
    its perspective there is just `search_docs`. Swap implementations
    freely behind this function.
    """
    tfidf_hits = [c for c, _ in TFIDF.search(query, k=k * 2)]
    bm25_hits = [c for c, _ in BM25.search(query, k=k * 2)]
    fused = _hybrid.rrf_fuse([tfidf_hits, bm25_hits], k=k)
    return [
        {"id": f"{c.doc}#{c.idx}", "text": c.text, "score": round(score, 4)}
        for c, score in fused
    ]


# --- Tool definition ------------------------------------------------------

SEARCH_TOOL = {
    "name": "search_docs",
    "description": (
        "Search the Acme handbook for passages relevant to a query. Returns "
        "ranked chunks, each with an 'id', 'text', and 'score'. Use the id "
        "values when citing in your final answer. Searches are cheap: call "
        "it multiple times with refined queries when a single search isn't "
        "enough."
    ),
    "input_schema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "A focused search query."},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        },
    },
}


SYSTEM = """You answer questions about Acme using ONLY the company handbook.

You have one tool: search_docs(query, k). Follow this loop strictly:

1. Call search_docs with a focused query.
2. Before answering, ask yourself out loud: do the retrieved chunks actually
   contain a complete, grounded answer to the user's question? Name which
   chunk supports which part of the answer.
3. If the answer is incomplete: missing a subquestion, or the retrieved
   chunks use different vocabulary than the question: search again with a
   refined query. Examples of refinement: broaden ("vacation" instead of
   "PTO"), narrow, or aim at a missing subquestion.
4. When you have a complete grounded answer, write it as plain prose with
   inline citations like [security.md#4]. Do not call search_docs again.
5. If two or three searches yield nothing useful, say so honestly: do not
   invent facts. "The handbook does not cover X" is a valid answer.

Keep queries short and specific. Cite every claim."""


def dispatch(tool_use: dict) -> dict:
    """Execute a tool call and return a tool_result block."""
    name, use_id, inp = tool_use["name"], tool_use["id"], tool_use.get("input", {}) or {}
    if name != "search_docs":
        return _err(use_id, f"unknown tool {name!r}")
    query = inp.get("query")
    if not isinstance(query, str) or not query.strip():
        return _err(use_id, "query must be a nonempty string")
    k = inp.get("k", 4)
    hits = hybrid_search(query, k=int(k))
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": json.dumps({"hits": hits}),
    }


def _err(use_id: str, msg: str) -> dict:
    return {"type": "tool_result", "tool_use_id": use_id, "content": msg, "is_error": True}


def run_agent(question: str, *, max_steps: int = 5) -> str:
    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=[SEARCH_TOOL])
        messages.append({"role": "assistant", "content": response.content})

        print(f"\n--- step {step}  stop={response.stop_reason} ---")
        if response.text:
            # The reflection lives in the model's text block. Showing it
            # is the whole point of the lesson.
            print(f"  thought: {response.text.strip()}")
        for tu in response.tool_uses:
            q = tu["input"].get("query", "")
            print(f"  search: {q!r}")

        if response.stop_reason != "tool_use":
            return response.text

        results = [dispatch(tu) for tu in response.tool_uses]
        # Print the chunk ids the agent will see next turn: they're the
        # input to the next reflection.
        for r in results:
            if r.get("is_error"):
                print(f"  -> error: {r['content']}")
                continue
            payload = json.loads(r["content"])
            ids = [h["id"] for h in payload.get("hits", [])]
            print(f"  -> hits: {ids}")

        messages.append({"role": "user", "content": results})

    return "(max steps reached without a final answer)"


if __name__ == "__main__":
    questions = [
        # Multi hop: needs a search for SEV-1 paging *and* one for the
        # postmortem deadline. Both facts live in incident_response.md
        # but in different paragraph chunks.
        "If I'm paged for a SEV-1 after midnight, what stipend do I get and "
        "by when does the postmortem need to be written?",
        # Vocabulary mismatch: the corpus says "vacation" and "roll over",
        # not "PTO" / "carryover". The first search will probably miss;
        # the agent should refine and try again.
        "What's the PTO carryover policy?",
        # Genuine gap: the corpus has no answer. The agent should say so
        # rather than confabulate.
        "What's Acme's policy on emotional support animals in the office?",
    ]
    for q in questions:
        print(f"\n{'='*72}\nQ: {q}")
        answer = run_agent(q)
        print(f"\nA: {answer}")
