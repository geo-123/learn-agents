"""Lesson 03c: BM25 + Reciprocal Rank Fusion.

Two lexical rankers (BM25 here, TF-IDF from 03a) and the RRF formula
that fuses them. The pedagogical point is the fusion pattern: it works
identically when one channel is a dense embedding retriever: the code
below would not change.

Run:
    uv run python lessons/03c_agentic_rag/01_hybrid_bm25.py
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Reuse the tokenizer + TfidfIndex from 03a. The numeric prefix on the
# file name blocks ordinary `import`: load by path. The module must be
# inserted into sys.modules *before* exec so that @dataclass can resolve
# its own module on Python 3.13+.
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_tfidf", HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py",
)
assert _spec and _spec.loader
_tfidf = importlib.util.module_from_spec(_spec)
sys.modules["_tfidf"] = _tfidf
_spec.loader.exec_module(_tfidf)

Chunk = _tfidf.Chunk
tokenize = _tfidf.tokenize

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def chunk_paragraphs(doc: str, text: str) -> list[Chunk]:
    """Split a doc into paragraphs on blank lines.

    We don't reuse 03a's markdown header chunker here because every doc
    in the demo corpus only has one top level header, so it would yield
    one chunk per doc: leaving the two rankers nothing to disagree on.
    """
    chunks: list[Chunk] = []
    for idx, para in enumerate(re.split(r"\n\s*\n", text)):
        body = para.strip()
        if body:
            chunks.append(Chunk(doc, idx, body))
    return chunks


def build_corpus() -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        out.extend(chunk_paragraphs(path.name, path.read_text(encoding="utf-8")))
    return out


class BM25Index:
    """Okapi BM25, built from scratch.

    Two knobs that distinguish BM25 from plain TF-IDF:
      k1: term frequency saturation. Higher = each extra mention counts more.
      b: length normalization. 0 = ignore doc length; 1 = full normalization.
    Standard defaults (k1=1.5, b=0.75) work well on most corpora.
    """

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b

        tokens = [tokenize(c.text) for c in chunks]
        self.doc_lens = np.array([len(t) for t in tokens], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0

        vocab = sorted({t for ts in tokens for t in ts})
        self.vocab = {t: i for i, t in enumerate(vocab)}

        n = len(chunks)
        df = Counter()
        for ts in tokens:
            for t in set(ts):
                df[t] += 1

        # BM25's IDF dampens common terms harder than plain TF-IDF.
        self.idf = np.zeros(len(vocab))
        for t, j in self.vocab.items():
            self.idf[j] = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)

        # Raw term frequency matrix (no IDF baked in: applied at query time).
        self.tf = np.zeros((n, len(vocab)))
        for i, ts in enumerate(tokens):
            for t, count in Counter(ts).items():
                self.tf[i, self.vocab[t]] = count

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q_terms = [t for t in tokenize(query) if t in self.vocab]
        if not q_terms:
            return []
        scores = np.zeros(len(self.chunks))
        len_norm = 1 - self.b + self.b * self.doc_lens / self.avgdl
        for t in q_terms:
            j = self.vocab[t]
            tf = self.tf[:, j]
            numer = tf * (self.k1 + 1)
            denom = tf + self.k1 * len_norm
            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = np.where(denom > 0, numer / denom, 0.0)
            scores += self.idf[j] * contrib
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


# --- Reciprocal Rank Fusion ----------------------------------------------
#
# Given multiple ranked lists of the same items, RRF assigns each item a
# score of 1/(c + rank) per list and sums. Items that rank consistently
# high across rankers float to the top: without needing to normalize
# scores that live on totally different scales.


def rrf_fuse(
    rankings: list[list[Chunk]],
    *,
    k: int = 5,
    c: int = 60,
) -> list[tuple[Chunk, float]]:
    """Fuse N ranked chunk lists into a single ranked list of length k.

    c = 60 is the value from the original RRF paper. It dampens the
    contribution of low ranked items; smaller c gives top ranks more
    weight, larger c flattens the curve.
    """
    scores: dict[tuple[str, int], float] = {}
    by_id: dict[tuple[str, int], Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = (chunk.doc, chunk.idx)
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
            by_id[key] = chunk
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [(by_id[key], score) for key, score in ordered]


# --- Demo -----------------------------------------------------------------


def _preview(chunk: Chunk) -> str:
    first_line = chunk.text.split("\n", 1)[0]
    return first_line[:70]


def _print_ranking(label: str, ranked: list[tuple[Chunk, float]]) -> None:
    print(f"  {label}")
    if not ranked:
        print("    (no results)")
        return
    for rank, (chunk, score) in enumerate(ranked, start=1):
        print(f"    {rank}. [{chunk.doc}#{chunk.idx}] {score:6.3f} | {_preview(chunk)}")


if __name__ == "__main__":
    chunks = build_corpus()
    print(f"indexed {len(chunks)} paragraph chunks from {len(set(c.doc for c in chunks))} docs")
    tfidf = _tfidf.TfidfIndex(chunks)
    bm25 = BM25Index(chunks)

    # The first three queries land cleanly: both rankers agree on the
    # right chunk, and the score *scales* differ wildly (TF-IDF ~0.3,
    # BM25 ~8). That's the whole point of RRF: it averages ranks, not
    # scores, so incomparable scales don't matter.
    #
    # The last query is a paraphrase: the corpus says "vacation does not
    # roll over" but the question uses "PTO carryover". Both lexical
    # rankers miss: fusion can't rescue them. This is the moment to
    # add a dense embedding channel as a third ranker (Lesson 03b shows
    # the embedding swap; the RRF code below would not change).
    queries = [
        "company holidays and how many days off",
        "what do I do if I clicked a phishing link",
        "stipend for after hours pages",
        "PTO carryover policy",
    ]

    K_RETRIEVE = 5  # depth from each ranker before fusion
    K_FINAL = 3     # top k after fusion

    for q in queries:
        print(f"\n{'='*72}\nQ: {q}")

        tfidf_hits = [(c, s) for c, s in tfidf.search(q, k=K_RETRIEVE)]
        bm25_hits = bm25.search(q, k=K_RETRIEVE)

        _print_ranking("TF-IDF:", tfidf_hits[:K_FINAL])
        _print_ranking("BM25:  ", bm25_hits[:K_FINAL])

        fused = rrf_fuse(
            [[c for c, _ in tfidf_hits], [c for c, _ in bm25_hits]],
            k=K_FINAL,
        )
        _print_ranking("RRF:   ", fused)
