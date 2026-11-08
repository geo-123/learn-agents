# Lesson 02b: Prompt Caching

The single highest ROI cost lever in agent engineering.

> Cached input tokens are billed at **~10% of the normal input price** and arrive with **dramatically lower latency** (50 to 85% TTFT reduction in practice). If your agent resends the same prefix every turn (and most agents do) caching turns that prefix from a recurring cost into a one time cost.

This lesson is short because the mechanic is short. But cache aware prompt design is now a core skill: it's the difference between an agent that costs $0.10 per session and the same agent that costs $1.00.

## What's actually cached

When you mark a span of the prompt with `cache_control`, Anthropic's infrastructure stores the KV cache for *everything up to and including that span* under a content addressed hash. The next request whose prefix matches *byte for byte* up to that span gets a **cache hit**: those tokens are billed at the discounted rate and the model doesn't recompute them.

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

- `cache_creation_input_tokens`: tokens written to a new cache entry on this call.
- `cache_read_input_tokens`: tokens served from an existing cache entry on this call.
- `input_tokens`: the *noncached* input tokens (deltas after the cached span).

The wrapper in `src/_llm.py` exposes all three on `LLMResponse`.

### Automatic caching

The simplest form is one top level marker on the request: `cache_control={"type": "ephemeral"}` (the helper's `cache_control=` argument). The API places the breakpoint on the last cacheable block for you, so the whole stable prefix is cached without annotating individual blocks. Reach for explicit markers (like the example above) when you need the breakpoint somewhere specific.

## What invalidates a cache hit

Anything that changes the bytes of the cached span:

- A timestamp baked into the system prompt.
- A user ID, request ID, or session token.
- Reordering tools in the `tools` list.
- A new tool definition added or removed.
- Any whitespace change.

Two of these are silent killers in practice: **inadvertent dynamic content in a "static" system prompt** (today's date, A/B variant id, last modified time) and **tool catalog churn** (`tools.sort()` matters).

Two more that are easy to miss: **switching models** (caches are per model, so a router that alternates models never reuses an entry) and **changing `effort` or the system prompt mid conversation**.

### Instructions that arrive mid conversation

A mode switch or new context halfway through a session tempts you to edit the top level `system` prompt. That changes the prefix ahead of the whole history, so every cached turn is reprocessed at full price (and, on current models, the earlier thinking blocks stop matching: Lesson 04). Current Claude models accept a `{"role": "system", "content": "..."}` message appended to `messages` instead: it sits *after* the cached history and leaves the prefix intact. It must follow a user message, and it is also the channel an attacker cannot forge from inside user or tool content (Lesson 08).

## Cache aware prompt structure

The prompt is no longer one big string. It's a sequence of spans with deliberate cacheability:

```
[ system prompt: stable, marked cache_control  ]   ← cached
[ corpus / persona / examples: stable          ]   ← cached
[ summarized history: slow changing            ]   ← cache breakpoint here
[ recent turns: fast changing                  ]   ← not cached
[ this turn's user message                      ]   ← not cached
```

You can declare up to **four** `cache_control` markers per request. Use them to mark "stable enough" boundaries. The most common layout is just one: at the end of the system prompt.

The shape that wastes money: a 50KB system prompt that includes the current timestamp on every call. You write a new cache entry every turn instead of reading one. The metrics will show `cache_creation_input_tokens` > 0 every turn and `cache_read_input_tokens == 0`: the diagnostic for a misdesigned prefix.

## When caching is a no op

- **Single turn agents.** No subsequent call to read from. (Caching still helps if your prefix is reused across requests from different users: same hash, same cache entry.)
- **Very short prompts.** There's a minimum prefix size, and it depends on the model: 512 tokens on the current generation (Haiku 5.5, Sonnet 5.5, Opus 5.5), 1024 on some earlier models (Sonnet 4.6, Opus 4.8), up to 4096 on others (Haiku 4.5, Opus 4.6). It is not monotonic across generations, so check it for the model you use. Below the threshold, `cache_control` is silently ignored: `created` stays at 0 forever. Always measure.
- **Prompts that genuinely change every turn.** Live data, user specific personalization, no stable structure. Caching helps only the stable portion.

## TTL and pricing nuance

- Ephemeral cache: **5 minute TTL**, refreshed on every read. Write price 1.25x normal input, read price ~0.1x.
- Extended cache: `{"type": "ephemeral", "ttl": "1h"}`, **1 hour TTL**, write price 2x normal input, same read price. It keeps entries alive across gaps in bursty traffic, but needs more reads to pay for the dearer write.
- With the 5 minute TTL the break even is the first hit (1.25x + 0.1x is less than 2x); with the 1 hour TTL it takes two hits.

## Failure modes

- **No cache hits in the wild.** Check that the cached prefix is actually byte identical across requests. Diff two real requests; look for hidden dynamic content.
- **Lots of `cache_creation`, no `cache_read`.** Same root cause: the prefix differs every turn.
- **Cache thrashing across users.** If the cached span includes the user id, each user gets their own cache entry (and pays the write price). If the cache prefix is *truly* identical (e.g., a shared system prompt with the user id below the breakpoint), all users share one entry.
- **Tools list reordering on each call.** Tool definitions are part of the cacheable prefix. Sort them deterministically.

## Hands on

### `01_cache_hits.py`

Two LLM calls with an identical, cache marked system prompt and different user questions. Prints the cache token counts side by side. Turn 1 writes the cache; turn 2 reads it. The math at the end shows the input token bill that would have been paid without caching.

### `02_cache_invalidation.py`

Same structure, but the "system prompt" contains today's date stamped to the millisecond. Watch `cache_creation_input_tokens` climb every turn while `cache_read_input_tokens` stays at zero. The diagnostic shape of a silently broken cache.

## Key takeaways

- Cache reads cost ~10% of normal input tokens. Cache writes cost 1.25x (5 minutes) or 2x (1 hour). With the 5 minute TTL the break even is one hit.
- Use one top level `cache_control` for automatic placement, or mark up to 4 stable boundaries yourself. The most common layout is one breakpoint at the end of the system prompt.
- Put mid session instructions in an appended `role: "system"` message, never in an edited top level `system`.
- Anything dynamic in the cached span invalidates it. Timestamps, user ids, tool reordering are silent killers.
- `cache_read_input_tokens > 0` is the only proof that caching is working. Log it.

## Common misconceptions

- **"Caching needs a special model."** No: same Claude models, no flag, just `cache_control` blocks.
- **"My prefix is too short to cache."** On current models the minimum is 512 tokens; on some older ones it is 4096. Check the number for your model, then measure.
- **"Caching means the model remembers across calls."** No: only the *prompt prefix* is cached. The model has no memory; the cache stores the precomputed KV state for that exact byte sequence.

Next: [Lesson 03a: Retrieval basics](../03a_retrieval_basics/README.md).
