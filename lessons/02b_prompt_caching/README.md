# Lesson 02b — Prompt Caching

The single highest-ROI cost lever in agent engineering.

> Cached input tokens are billed at **~10% of the normal input price** and arrive with **dramatically lower latency** (50-85% TTFT reduction in practice). If your agent re-sends the same prefix every turn — and most agents do — caching turns that prefix from a recurring cost into a one-time cost.

This lesson is short because the mechanic is short. But cache-aware prompt design is now a core skill — it's the difference between an agent that costs $0.10 per session and the same agent that costs $1.00.

## What's actually cached

When you mark a span of the prompt with `cache_control`, Anthropic's infrastructure stores the KV-cache for *everything up to and including that span* under a content-addressed hash. The next request whose prefix matches *byte for byte* up to that span gets a **cache hit**: those tokens are billed at the discounted rate and the model doesn't re-compute them.

```python
chat(
    system=[{
        "type": "text",
        "text": LARGE_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }],
    messages=[{"role": "user", "content": "first question"}],
)
# Turn 2: same system, different user message → cache hit on the system block.
```

What you see in the response:

- `cache_creation_input_tokens` — tokens written to a new cache entry on this call.
- `cache_read_input_tokens` — tokens served from an existing cache entry on this call.
- `input_tokens` — the *non-cached* input tokens (deltas after the cached span).

The wrapper in `src/_llm.py` exposes all three on `LLMResponse`.

## What invalidates a cache hit

Anything that changes the bytes of the cached span:

- A timestamp baked into the system prompt.
- A user ID, request ID, or session token.
- Re-ordering tools in the `tools` list.
- A new tool definition added or removed.
- Any whitespace change.

Two of these are silent killers in practice: **inadvertent dynamic content in a "static" system prompt** (today's date, A/B variant id, last-modified time) and **tool catalog churn** (`tools.sort()` matters).

## Cache-aware prompt structure

The prompt is no longer one big string. It's a sequence of spans with deliberate cacheability:

```
[ system prompt — stable, marked cache_control ]   ← cached
[ corpus / persona / examples — stable          ]   ← cached
[ summarized history — slow-changing            ]   ← cache breakpoint here
[ recent turns — fast-changing                  ]   ← not cached
[ this turn's user message                      ]   ← not cached
```

You can declare up to **four** `cache_control` markers per request. Use them to mark "stable enough" boundaries. The most common layout is just one: at the end of the system prompt.

The shape that wastes money: a 50KB system prompt that includes the current timestamp on every call. You write a new cache entry every turn instead of reading one. The metrics will show `cache_creation_input_tokens` > 0 every turn and `cache_read_input_tokens == 0` — the diagnostic for a misdesigned prefix.

## When caching is a no-op

- **Single-turn agents.** No subsequent call to read from. (Caching still helps if your prefix is reused across requests from different users — same hash, same cache entry.)
- **Very short prompts.** There's a minimum span size: 1024 tokens for Sonnet/Opus, 2048 documented for Haiku (empirically closer to 4096 on Haiku 4.5 as of May 2026 — the docs are stale). Below the threshold, `cache_control` is silently ignored — `created` stays at 0 forever. Always measure.
- **Prompts that genuinely change every turn.** Live data, user-specific personalization, no stable structure. Caching helps only the stable portion.

## TTL and pricing nuance

- Ephemeral cache: **5-minute TTL**, refreshed on every read. Read price: ~10% of normal input.
- Extended cache: **1-hour TTL**, write price slightly higher, same read price.
- Cache reads are **always cheaper** than re-paying input tokens. The break-even is reached on the first hit.

## Failure modes

- **No cache hits in the wild.** Check that the cached prefix is actually byte-identical across requests. Diff two real requests; look for hidden dynamic content.
- **Lots of `cache_creation`, no `cache_read`.** Same root cause — the prefix differs every turn.
- **Cache thrashing across users.** If the cached span includes the user id, each user gets their own cache entry (and pays the write price). If the cache prefix is *truly* identical (e.g., a shared system prompt with the user id below the breakpoint), all users share one entry.
- **Tools list re-ordering on each call.** Tool definitions are part of the cacheable prefix. Sort them deterministically.

## Hands-on

### `01_cache_hits.py`

Two LLM calls with an identical, cache-marked system prompt and different user questions. Prints the cache token counts side by side. Turn 1 writes the cache; turn 2 reads it. The math at the end shows the input-token bill that would have been paid without caching.

### `02_cache_invalidation.py`

Same structure, but the "system prompt" contains today's date stamped to the millisecond. Watch `cache_creation_input_tokens` climb every turn while `cache_read_input_tokens` stays at zero. The diagnostic shape of a silently-broken cache.

## Key takeaways

- Cache reads cost ~10% of normal input tokens. Cache writes cost slightly more than normal. The break-even is one hit.
- Mark up to 4 stable boundaries with `cache_control: {"type": "ephemeral"}`. The most common layout is one marker at the end of the system prompt.
- Anything dynamic in the cached span invalidates it. Timestamps, user ids, tool re-ordering are silent killers.
- `cache_read_input_tokens > 0` is the only proof that caching is working. Log it.

## Common misconceptions

- **"Caching needs a special model."** No — same Claude models, no flag, just `cache_control` blocks.
- **"My prefix is too short to cache."** Below ~2K tokens it's a no-op. Above, it's almost always worth it.
- **"Caching means the model remembers across calls."** No — only the *prompt prefix* is cached. The model has no memory; the cache stores the pre-computed KV state for that exact byte sequence.

Next: [Lesson 03a — Retrieval basics](../03a_retrieval_basics/README.md).
