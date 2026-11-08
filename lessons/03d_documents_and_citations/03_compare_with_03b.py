"""Lesson 03d: Lesson 03b's citations, rebuilt on the native feature.

Lesson 03b made the model fill a `submit_answer` tool with chunk ids,
then guessed after the fact (word overlap) whether the chunks support
the answer. Here the same TF-IDF index from Lesson 03a picks the
chunks, each chunk becomes its own text document titled with its id,
and the API returns the exact quoted passage per claim.

Part 2 is the rule that bites agent pipelines on API version
2026-10-21: a `text` block with non empty `citations` must not be sent
inside a USER message. Passing one agent's answer to a second agent is
exactly that, so we strip the citations (keep the text) before the
hand off. Sending the answer back as the assistant's own turn is fine
and keeps them.

Run:
    uv run python lessons/03d_documents_and_citations/03_compare_with_03b.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_tfidf", HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py",
)
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import DEFAULT_MODEL, chat

INDEX = _tfidf.build_index()
QUESTION = "How many vacation days do I get, and do unused days roll over?"


def retrieve_as_documents(question: str, k: int = 3) -> tuple[list[dict[str, Any]], list[str]]:
    docs, ids = [], []
    for chunk, _score in INDEX.search(question, k=k):
        ids.append(f"{chunk.doc}#{chunk.idx}")
        docs.append({
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": ids[-1],
            "citations": {"enabled": True},
        })
    return docs, ids


def strip_citations(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Text blocks without `citations`, safe to place in a user message."""
    return [{"type": "text", "text": b["text"]} for b in blocks if b["type"] == "text"]


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}\nQ: {QUESTION}")

    print("\n=== Part 1: retrieved chunks as cited documents ===")
    docs, ids = retrieve_as_documents(QUESTION)
    print(f"retrieved: {ids}")
    first = chat([{"role": "user", "content": [*docs, {"type": "text", "text": QUESTION}]}], effort="low")
    used = set()
    for b in first.content:
        if b["type"] == "text":
            cites = b.get("citations") or []
            used.update(ids[c["document_index"]] for c in cites)
            tag = " ".join(f"<{ids[c['document_index']]}>" for c in cites)
            print(f"  {b['text']!r} {tag}")
    print(f"chunks actually cited: {sorted(used)}  (03b had to ask the model for this list)")

    print("\n=== Part 2: hand the answer to a reviewer agent ===")
    raw = [b for b in first.content if b["type"] == "text"]
    with_cites = sum(1 for b in raw if b.get("citations"))
    print(f"answer has {len(raw)} text blocks, {with_cites} with citations.")
    print("Putting them as they are into a user message would be a 400 on API 2026-10-21.")
    review_input = [
        {"type": "text", "text": "Another assistant wrote this answer. Is it complete and clear? "
                                 "Reply in two sentences.\n\n---\n"},
        *strip_citations(first.content),
    ]
    review = chat([{"role": "user", "content": review_input}], effort="low")
    print(f"reviewer: {review.text.strip()}")

    print("\n=== Part 3: a follow up in the SAME conversation keeps citations ===")
    messages = [
        {"role": "user", "content": [*docs, {"type": "text", "text": QUESTION}]},
        {"role": "assistant", "content": first.content},     # unchanged, citations and all
        {"role": "user", "content": "How far ahead must I request a two week vacation?"},
    ]
    follow = chat(messages, effort="low")
    print(f"follow up: {follow.text.strip()}")
    print(f"tokens: first {first.input_tokens}/{first.output_tokens}, "
          f"follow up {follow.input_tokens}/{follow.output_tokens}")
