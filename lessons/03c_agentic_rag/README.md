# Lesson 03c: Modern (Agentic) RAG

Lesson 03a built a tiny retriever. Lesson 03b exposed it as a tool. This lesson covers the two patterns production RAG systems converged on between 2024 and 2026:

1. **Hybrid retrieval**: combine a sparse/lexical ranker (BM25) with a second ranker, and fuse them with **Reciprocal Rank Fusion (RRF)**. Neither channel alone is robust; together they're noticeably better.
2. **Iterative / self corrective retrieval**: the agent searches, evaluates whether the retrieved chunks actually answer the question, and either answers or searches again with a refined query.

You'll see these under different names: LangChain's "self-RAG", LlamaIndex's "subquestion query engine", Cohere's rerank then fuse, Anthropic's contextual retrieval. The names vary; the shape is the same. The point of this lesson is the shape, not any one framework.

## Hasn't long context killed RAG?

For a few page corpus, yes: stuff it in the prompt and skip retrieval. But every real corpus exceeds context, and four problems remain even when it fits:

- **Cost**: tokens are billed per call. Stuffing a 200 page handbook on every turn is wasteful.
- **Lost in the middle**: models attend less reliably to the middle of long contexts. Targeted chunks beat stuffed context for quality.
- **Freshness**: an indexable, cacheable corpus is much cheaper than restuffing on every change.
- **Provenance**: citations are easier when chunks are addressable.

The right takeaway is *less* RAG, not *no* RAG. Skip retrieval when the question doesn't need it; iterate when it does.

## Hybrid retrieval and RRF

Lexical retrievers (BM25, TF-IDF) match exact words. Dense retrievers (embeddings) match meaning. Neither alone is robust:

- Lexical misses paraphrases: query "PTO" doesn't match "vacation."
- Dense embeddings hallucinate similarity: query "rollover" can pull in unrelated chunks about IRA rollovers.

**Reciprocal Rank Fusion (RRF)** combines rankings from any number of retrievers *without needing to normalize their scores*:

```
score(d) = sum over rankers of  1 / (c + rank_in_ranker(d))      c = 60
```

The chunk that places consistently high across all rankers wins. Five lines, no tuning beyond `c`, and it almost always beats picking one retriever. The trick: RRF averages **ranks**, not scores. Score scales are usually incomparable across rankers; ranks are not.

This lesson fuses **BM25** (built from scratch) with **TF-IDF** (from 03a): two lexical rankers, because they let you read every line. In production you'd swap one channel for an embedding based ranker (Lesson 03b shows how that's wired). The fusion code wouldn't change.

### Why BM25 specifically?

TF-IDF and BM25 are both lexical, but they disagree more than you'd think. BM25 adds two things TF-IDF doesn't have:

- **Term saturation** (the `k1` parameter): the 10th mention of "vacation" doesn't help much more than the 3rd. TF-IDF treats them linearly.
- **Length normalization** (the `b` parameter): a short, dense chunk shouldn't be penalized vs. a long sprawling one.

These differences make BM25 a complementary signal: a useful demonstration that even two lexical rankers benefit from fusion.

## Iterative / agentic retrieval

A single search is rarely enough for nontrivial questions. The modern pattern is a loop:

1. Agent forms a query and calls `search_docs`.
2. Agent reads the chunks and asks itself: *do these actually answer the question?*
3. If yes, answer with citations. If no, refine the query (broader, narrower, or aimed at a missing subquestion) and search again.

This is what production "agentic RAG" systems do. The reflection step is the thing 03b's one shot tool call doesn't have. It buys you:

- **Multi hop**: "what's the SLA?" and "what's the credit if we miss it?" become two searches the model decides between turns.
- **Vocabulary recovery**: when the model searches "PTO carryover" and gets nothing useful, it retries with "vacation rollover."
- **Honest gaps**: when the corpus genuinely doesn't contain the answer, the agent declares insufficient evidence instead of confabulating.

`02_iterative_rag.py` implements the loop. The system prompt does most of the heavy lifting: it tells the model to evaluate sufficiency between searches and to retry with a different angle when a search comes up short.

## When to skip retrieval entirely

Some questions don't need the corpus:

- "What's 2+2?": never.
- "Hi, are you there?": never.
- "Summarize my last message.": never.

Force retrieving on every turn is a beginner mistake. Make the tool optional; the model is usually right about whether it needs it. If you're worried it'll skip retrieval when it shouldn't, write an eval (Lesson 07) rather than hard coding the call.

## Failure modes

- **Infinite refinement loop.** The agent keeps reformulating queries and never commits. Cap the steps.
- **Reflection rubber stamp.** The agent says "yes, this answers it" every time, even when the chunks are off topic. Mitigation: structure the reflection (e.g., it must name which fact came from which chunk).
- **One ranker dominates RRF.** Fuse three lexical rankers and one dense, and the lexical signals can drown out the dense one. Weight the rankers, or normalize how many you fuse from each family.
- **Stale corpus.** Same as 03b. The index, not the agent, owns this.

## Hands on

### `01_hybrid_bm25.py`: BM25 + RRF fusion

BM25 built from scratch (~40 lines of NumPy), then fused with the TF-IDF index from 03a using RRF. Runs three queries and prints each ranker's top k side by side, plus the fused result, so you can see where the two channels agree, disagree, and where fusion lands the right chunk.

### `02_iterative_rag.py`: search → reflect → search → answer

The agent has one tool, `search_docs`, and a system prompt that forces it to evaluate sufficiency before answering. The demo questions need multi step search; the trace shows the agent searching, finding a partial answer, refining the query, and only then producing a final answer with citations.

## Key takeaways

- RAG is not obsolete; the framing changed. Tool using agents call retrieval *when they need it*, not before every turn.
- Hybrid retrieval is the easy win. Two complementary rankers fused with RRF beats either alone on most real corpora.
- Iterative retrieval is the second easy win. One search → reflect → maybe another search is much more robust than top k and pray.
- "No relevant chunks" is a valid answer. The agent should be allowed to give it.

## Common misconceptions

- **"RRF is just averaging."** It's averaging *ranks*, not scores: that's what lets you fuse rankers with incomparable score scales.
- **"More iterations = better answer."** Past 2 to 3 hops, the model loses the plot. Cap aggressively.
- **"Agentic RAG needs a fancy framework."** It's a loop with a tool. Lesson 01's agent loop is enough.

Next: [Lesson 03d: Documents and citations](../03d_documents_and_citations/README.md).
