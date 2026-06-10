"""Lesson 03b (local variant) — real embeddings, no paid API.

Same `search(query, k) -> list[Hit]` interface as the TF-IDF version in
03a and the OpenAI version in 02_embeddings_openai.py. Only the embedding
*implementation* changes — the agent loop in 01_rag_tool.py would work
unchanged against any of the three. That's the whole lesson: a retriever
is an interface, not a vendor.

This variant runs a small open-source model **locally** via
sentence-transformers. No API key, no per-query cost, works offline. The
one cost is a one-time model download (~80 MB) on first run, cached under
your HuggingFace cache dir thereafter.

Why bother over TF-IDF? Dense embeddings capture *meaning*, so paraphrases
like "I want to take time off" or "Can I work abroad?" match the right
doc even when they share almost no literal words with the source text —
the exact queries TF-IDF whiffs on in 03a.

Requires:
  - `uv add sentence-transformers`   (heavy: this also pulls in PyTorch)

Run:
    uv run python lessons/03b_rag_in_loop/02_embeddings_local.py
"""
from __future__ import annotations

import importlib.util
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

# Import lazily so this module imports cleanly without the (heavy) dep.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run: uv add sentence-transformers")
    sys.exit(1)


CORPUS_DIR = HERE.parents[1] / "data" / "corpus"
# Small, fast, 384-dim. Downloads once on first run, then loads from cache.
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Hit:
    doc: str
    idx: int
    text: str
    score: float


class EmbeddingIndex:
    """A retriever backed by a local sentence-transformers model + cosine similarity.

    Same public surface as TfidfIndex and the OpenAI EmbeddingIndex:
    build once, search many times.
    """
    def __init__(self, chunks: list):
        self.chunks = chunks
        print(f"loading {EMBED_MODEL} (first run downloads ~80 MB)...")
        self.model = SentenceTransformer(EMBED_MODEL)

        print(f"embedding {len(chunks)} chunks...")
        self.matrix = self._embed_batch([c.text for c in chunks])
        # Normalize rows so cosine reduces to a dot product.
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix /= norms

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        # sentence-transformers batches internally and returns a numpy array.
        return self.model.encode(texts, convert_to_numpy=True)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self._embed_batch([query])[0]
        q = q / (np.linalg.norm(q) or 1.0)
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
