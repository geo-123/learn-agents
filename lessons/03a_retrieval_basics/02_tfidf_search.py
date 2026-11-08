"""Lesson 03a: TF-IDF retrieval, built from scratch.

A working bag of words + TF-IDF + cosine search engine in ~50 lines.
Same pipeline shape (chunk -> embed -> index -> query -> top k) you'll
use later with real embeddings. The point: a "vector DB" is a matrix.

Run:
    uv run python lessons/03a_retrieval_basics/02_tfidf_search.py
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass(frozen=True)
class Chunk:
    doc: str
    idx: int
    text: str


def chunk_markdown(doc: str, text: str) -> list[Chunk]:
    """Reused from 01_chunking.py: header aware splitting."""
    chunks, header, lines, idx = [], "", [], 0
    def flush() -> None:
        nonlocal idx
        body = "\n".join(lines).strip()
        if body:
            chunks.append(Chunk(doc, idx, f"{header}\n\n{body}" if header else body))
            idx += 1
    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            header, lines = line.strip(), []
        else:
            lines.append(line)
    flush()
    return chunks


_TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfIndex:
    """A minimal TF-IDF + cosine retriever.

    Build once over a list of chunks; query as many times as you want.
    Vocabulary is fixed at build time: out of vocab query terms are dropped.
    """
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        # 1) Vocabulary: every distinct token across the corpus.
        all_tokens = [tokenize(c.text) for c in chunks]
        vocab = sorted({tok for toks in all_tokens for tok in toks})
        self.vocab_index = {tok: i for i, tok in enumerate(vocab)}

        # 2) Document frequency: in how many chunks each token appears.
        n_docs = len(chunks)
        df = Counter()
        for toks in all_tokens:
            for tok in set(toks):
                df[tok] += 1

        # 3) IDF: rare tokens carry more signal. +1 smoothing prevents log(0).
        self.idf = np.zeros(len(vocab))
        for tok, j in self.vocab_index.items():
            self.idf[j] = math.log((n_docs + 1) / (df[tok] + 1)) + 1.0

        # 4) Build the (N × |V|) TF-IDF matrix. Each row is one chunk's vector.
        self.matrix = np.zeros((n_docs, len(vocab)))
        for i, toks in enumerate(all_tokens):
            counts = Counter(toks)
            for tok, n in counts.items():
                self.matrix[i, self.vocab_index[tok]] = n
            self.matrix[i] *= self.idf
        # Normalize rows so cosine becomes a plain dot product.
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix /= norms

    def _vectorize(self, query: str) -> np.ndarray:
        v = np.zeros(len(self.vocab_index))
        for tok in tokenize(query):
            j = self.vocab_index.get(tok)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._vectorize(query)
        if not q.any():
            return []
        scores = self.matrix @ q  # cosine, since both sides are normalized
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_index() -> TfidfIndex:
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    print(f"indexed {len(chunks)} chunks from {len(list(CORPUS_DIR.glob('*.md')))} docs")
    return TfidfIndex(chunks)


if __name__ == "__main__":
    idx = build_index()

    queries = [
        "How many vacation days do I get?",            # Direct lexical match.
        "Can I work from another country?",            # Remote_work doc.
        "What's the on call pager stipend?",           # Incident_response doc.
        "I want to take time off",                     # Paraphrase: watch this miss.
        "Where do I report a phishing email?",         # Security doc.
        "Do reimbursements need receipts?",            # Expense doc.
    ]

    for q in queries:
        print(f"\nQ: {q}")
        results = idx.search(q, k=3)
        if not results:
            print("  (no results: all query tokens were out of vocabulary)")
            continue
        for chunk, score in results:
            preview = chunk.text.split("\n", 1)[0][:80]
            print(f"  [{chunk.doc}#{chunk.idx}]  score={score:.3f}  | {preview}")
