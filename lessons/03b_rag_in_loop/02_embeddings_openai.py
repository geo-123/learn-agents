"""Lesson 03b — swap TF-IDF for real embeddings.

Same `search(query, k) -> list[Hit]` interface. Different implementation.
The agent code from 01_rag_tool.py would work unchanged if you point it
at this Retriever instead.

Cost: ~$0.00002 per query at text-embedding-3-small prices. Pennies for
the whole lesson.

Requires:
  - OPENAI_API_KEY in .env
  - `uv add openai`  (not installed by default to keep base deps minimal)

If you don't want to pay for embeddings, skip this file. The TF-IDF
version answers most queries well enough for learning.

Run:
    uv run python lessons/03b_rag_in_loop/02_embeddings_openai.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Reuse the markdown chunker from 03a.
spec = importlib.util.spec_from_file_location(
    "_tfidf", HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py",
)
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import load_dotenv  # noqa: F401 — triggers .env load

# We import openai lazily so this module imports cleanly without the dep.
try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Run: uv add openai")
    sys.exit(1)


CORPUS_DIR = HERE.parents[1] / "data" / "corpus"
EMBED_MODEL = "text-embedding-3-small"


@dataclass
class Hit:
    doc: str
    idx: int
    text: str
    score: float


class EmbeddingIndex:
    """A retriever backed by OpenAI embeddings + cosine similarity.

    Same public surface as TfidfIndex: build once, search many times.
    """
    def __init__(self, chunks: list):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Add it to .env.")
        self.chunks = chunks
        self.client = OpenAI()

        print(f"embedding {len(chunks)} chunks with {EMBED_MODEL}...")
        embeddings = self._embed_batch([c.text for c in chunks])
        self.matrix = np.array(embeddings)
        # Normalize rows so cosine reduces to a dot product.
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix /= norms

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenAI embeddings supports batches natively.
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = np.array(self._embed_batch([query])[0])
        q /= np.linalg.norm(q) or 1.0
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [
            Hit(self.chunks[i].doc, self.chunks[i].idx, self.chunks[i].text, float(scores[i]))
            for i in top
        ]


def build_index() -> EmbeddingIndex:
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks.extend(_tfidf.chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return EmbeddingIndex(chunks)


if __name__ == "__main__":
    idx = build_index()

    queries = [
        "I want to take time off",          # Paraphrase TF-IDF struggled with.
        "Where do I report a phishing email?",
        "What's the on-call pager stipend?",
        "Can I work abroad?",                # Synonyms for "from another country".
    ]
    for q in queries:
        print(f"\nQ: {q}")
        for hit in idx.search(q, k=3):
            preview = hit.text.split("\n", 1)[0][:80]
            print(f"  [{hit.doc}#{hit.idx}]  score={hit.score:.3f}  | {preview}")
