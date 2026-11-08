"""Lesson 03d: native citations over plain text documents.

Two Acme handbook files go in as `document` blocks with
`citations: {"enabled": true}`. No retrieval tool, no submit_answer
schema: the API itself splits the answer into text blocks and attaches
to each claim the exact passage it rests on.

For each cited segment we print:
    * which document (`document_index`, `document_title`)
    * where in it (`start_char_index`, `end_char_index`, 0 based, end exclusive)
    * the quoted `cited_text`
and then check the quote against our own copy of the file. That check
is exact, not the word overlap guess Lesson 03b had to make.

Run:
    uv run python lessons/03d_documents_and_citations/01_text_documents_with_citations.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
FILES = ["expense_policy.md", "remote_work.md"]
TEXTS = [(CORPUS / name).read_text(encoding="utf-8") for name in FILES]


def text_document(title: str, text: str) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
        "title": title,                      # shown to the model, never cited
        "citations": {"enabled": True},      # all documents or none
    }


def ask(question: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [text_document(n, t) for n, t in zip(FILES, TEXTS)]
    content.append({"type": "text", "text": question})  # documents first, question last
    response = chat([{"role": "user", "content": content}], effort="low")
    print(f"stop={response.stop_reason}  tokens in/out={response.input_tokens}/{response.output_tokens}")
    return response.content


def print_with_footnotes(blocks: list[dict[str, Any]]) -> None:
    answer, notes = "", []
    for block in blocks:
        if block["type"] != "text":
            continue
        answer += block["text"]
        for c in block.get("citations") or []:
            notes.append(c)
            answer += f"[{len(notes)}]"
    print(f"\nAnswer:\n{answer.strip()}\n")

    for i, c in enumerate(notes, 1):
        source = TEXTS[c["document_index"]]
        if c["type"] == "char_location":
            span = source[c["start_char_index"]:c["end_char_index"]]
            exact = span.strip() == c["cited_text"].strip()
            where = f"chars {c['start_char_index']} to {c['end_char_index']}"
        else:
            exact, where = False, c["type"]
        print(f"[{i}] {c.get('document_title')}  {where}  exact match: {exact}")
        print(f"     {c['cited_text'].strip()!r}")


def summarize_blocks(blocks: list[dict[str, Any]]) -> None:
    texts = [b for b in blocks if b["type"] == "text"]
    cited = [b for b in texts if b.get("citations")]
    print(f"\n{len(texts)} text blocks, {len(cited)} of them carry citations. "
          "One answer, split where the sources change.")


if __name__ == "__main__":
    print(f"Model: {DEFAULT_MODEL}")
    blocks = ask("I work from home. What can I get reimbursed for my home office, "
                 "and how soon do I have to submit expenses?")
    print_with_footnotes(blocks)
    summarize_blocks(blocks)
