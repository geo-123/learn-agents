# Lesson 03b — RAG in the Agent Loop

Lesson 03a built a search engine. This lesson plugs it into the agent loop. The punchline:

> **RAG is just a tool.** The agent decides when to retrieve, formulates the query itself, and incorporates the result into its answer.

This reframing matters. In a lot of beginner code, "RAG" is a hard-coded preprocessing step: every user question gets vectorized, top-k retrieved, then stuffed into the prompt. That works for FAQ-style bots, but it's not really agentic — the LLM has no say.

A tool-based RAG agent is different:

- The model can decide *not* to search (some questions don't need it).
- The model can search multiple times with different queries (decompose, refine).
- The model can search, read, decide to search again with a better query.
- The model can cite the chunks it actually used.

You'll also see a clean "implementation swap": the same `search_docs` tool, backed by either TF-IDF (free, lexical) or real embeddings (better, costs pennies). The agent code doesn't change — only what's behind the tool.

## Three things this lesson teaches

### 1. Retrieval as a tool

`search_docs(query: str, k: int) -> list[Chunk]` is added to the agent's toolbox. The model calls it like any other tool. No special control flow.

### 2. Citations are non-negotiable

A RAG agent without citations is a confabulation machine wearing a costume. We modify the schema so that every answer carries a list of source ids — the chunks the model claims to have used. Downstream code can check that those chunks actually contain what the answer says they do.

### 3. The interface, not the implementation

We define a `Retriever` protocol with one method: `search(query, k) -> list[Chunk]`. TF-IDF and OpenAI embeddings both implement it. The agent doesn't know or care which it's using. This is the same pattern as `src/_llm.py` — isolate the swappable thing behind a thin interface.

## When retrieval helps vs. hurts

Counter-intuitively, RAG can make answers *worse*:

- **Retrieved chunks contradict each other.** The model invents a synthesis. Better: detect contradiction and ask the user.
- **Top-k always returns *something*.** Even when the corpus has no answer, you'll get the 5 least-irrelevant chunks. The model trusts them. Mitigation: relevance threshold + explicit "if nothing matches, say so."
- **Question doesn't need retrieval.** "What's 2+2?" — running RAG over the company handbook is overhead. Let the model skip the tool.

Good RAG agents are conservative: retrieve only when needed, with the right query, and acknowledge gaps.

## "Lost in the middle" and dose control

Even after retrieval, you have to decide how many chunks to send. Putting all 20 retrieved chunks in the prompt isn't free:

- **Cost** scales linearly.
- **Quality** does not. Models attend less reliably to the middle of long contexts ("lost in the middle"). Sending 3 good chunks usually beats 20 mediocre ones.

Default to k=3–5 unless you have a reason to send more.

## Failure modes

- **Hallucinated citations.** The model claims chunk #4 supports its claim; chunk #4 says nothing of the kind. Check by re-grep'ing the cited chunks for key answer phrases.
- **Query is too narrow.** The model searches `"vacation accrual"` and misses the chunk about "PTO." Show the model how to broaden — "if no results, retry with synonyms."
- **Query is too broad.** `search_docs("everything")` returns chaos. Sometimes a system prompt instruction to "be specific in queries" is enough.
- **Stale embeddings.** Source updated; index didn't. Lesson 04 (memory) touches storage; production systems rebuild indexes on a schedule.

## Hands-on

### `01_rag_tool.py` — RAG as a tool

The TfidfIndex from Lesson 03a, exposed as a `search_docs` tool. The agent loop is the same as Lesson 01's. The agent figures out *when* to search, *what* to search for, and *how* to answer.

### `02_embeddings_openai.py` — swap the implementation

A second `Retriever` implementation backed by OpenAI embeddings (`text-embedding-3-small`). Same `Retriever` protocol, completely different quality on paraphrased queries. **Optional**: requires `OPENAI_API_KEY` and `uv add openai`. The lesson works without it — TF-IDF baseline gets you most of the way.

### `03_with_citations.py` — answers with citations

The agent's final answer must include the chunk IDs it used. We define a `submit_answer` tool with a `citations` field, and check after the fact that the cited chunks actually contain the answer's claims.

## Key takeaways

- RAG is a tool the agent decides to use. It is not a hard-coded preprocessor.
- The retriever interface and implementation are separate. Swap implementations, keep the interface.
- Citations are mandatory for trustworthy RAG. Verify them.
- "Lost in the middle" — send fewer, better chunks rather than many.
- Top-k always returns something. A relevance threshold prevents bad answers from bad retrieval.

## Common misconceptions

- **"RAG = preprocessing."** Sometimes — but tool-style RAG is more flexible.
- **"More retrieved chunks = better answer."** Past a small k, returns diminish and can reverse.
- **"My vector DB is the bottleneck."** Usually it's the embeddings or the chunker.

Next: [Lesson 03c — Modern (agentic) RAG](../03c_agentic_rag/README.md).
