"""Lesson 03a — three chunking strategies, side by side.

We chunk every doc in data/corpus/ three different ways and print the
boundaries so you can eyeball what each strategy preserves and what it
mangles.

Run:
    uv run python lessons/03a_retrieval_basics/01_chunking.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass(frozen=True)
class Chunk:
    doc: str          # source filename
    idx: int          # position within doc
    text: str


# --- Strategy 1: fixed-size with overlap ---------------------------------


def chunk_fixed(doc: str, text: str, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Slide a window of `size` chars, stepping by (size - overlap)."""
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(doc=doc, idx=i, text=piece))
    return chunks


# --- Strategy 2: paragraph-aware ----------------------------------------


def chunk_paragraphs(doc: str, text: str) -> list[Chunk]:
    """Split on blank lines. Each non-empty paragraph is one chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Chunk(doc=doc, idx=i, text=p) for i, p in enumerate(paragraphs)]


# --- Strategy 3: Markdown header-aware ----------------------------------


def chunk_markdown(doc: str, text: str) -> list[Chunk]:
    """Split on Markdown headers. The header travels with its section."""
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []
    idx = 0

    def flush() -> None:
        nonlocal idx
        body = "\n".join(current_lines).strip()
        if body:
            full = f"{current_header}\n\n{body}" if current_header else body
            chunks.append(Chunk(doc=doc, idx=idx, text=full))
            idx += 1

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


# --- Helper -------------------------------------------------------------


def load_corpus() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def summarize(name: str, chunks: list[Chunk]) -> None:
    sizes = [len(c.text) for c in chunks]
    print(f"\n{name}: {len(chunks)} chunks, "
          f"size min={min(sizes)} avg={sum(sizes)//len(sizes)} max={max(sizes)}")
    # Show the first chunk verbatim and the first 50 chars of every chunk.
    for c in chunks[:3]:
        snippet = c.text.replace("\n", " ")[:80]
        print(f"  [{c.doc}#{c.idx}] {snippet}...")
    if len(chunks) > 3:
        print(f"  ... ({len(chunks) - 3} more)")


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} docs from {CORPUS_DIR}")

    for strategy_name, fn in [
        ("fixed-size (size=400, overlap=80)", chunk_fixed),
        ("paragraphs", chunk_paragraphs),
        ("markdown headers", chunk_markdown),
    ]:
        all_chunks: list[Chunk] = []
        for doc, text in docs.items():
            all_chunks.extend(fn(doc, text))
        summarize(strategy_name, all_chunks)
