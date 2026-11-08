# Lesson 00b: Context Engineering

Lesson 00 said: an agent is an LLM + loop + tools + stop condition. This lesson zooms in on **what's actually inside the loop** at each turn (the context window) and how shaping that artifact is now its own discipline.

> **Context engineering**: filling the context window with *just the right information* for the next step.

The term was coined by Tobi Lütke (Shopify) in mid 2025, amplified by Andrej Karpathy, and adopted across the practitioner community within months. It deliberately replaces "prompt engineering" because the prompt is no longer one input: it's a *layered, dynamically assembled artifact*: system prompt, tool definitions, retrieved chunks, summarized history, scratchpad, examples, the user's latest turn.

If Lesson 00 taught you the agent loop, this lesson teaches you the shape of every message that goes into it.

## Why a new name

By 2026, "prompt engineering" sounds like a single human written string of clever instructions. That mental model breaks the moment your agent:

- Resends a 40 turn conversation each step.
- Pulls 8 retrieved chunks into context.
- Maintains a summary of the last 200 turns.
- Carries a tool catalog of 30 functions.
- Inserts a system prompt that references the user's profile.

None of those are "prompting" in the old sense. They're **assembly**: choosing what makes it into the next request and what doesn't. The job has shifted from authoring a string to designing the policy that builds the string.

## Has long context "solved" this?

This is the question people ask in 2026, and the honest answer is *mitigated, not solved*. Two effects persist on frontier 1M token models:

### Lost in the middle
Stanford 2023 finding: replicated repeatedly through 2025. Models attend better to the start and end of long contexts than to the middle. On 1M models the effect is *delayed* (high quality recall to ~600 to 700K tokens) but not eliminated. Mid prompt recall still drops 25 to 40% past that threshold.

### Context rot
Chroma's 2025 study (18 frontier models incl. GPT-4.1, Claude 4, Gemini 2.5, Qwen3) found that **performance degrades as input grows even when the window isn't full**, and that *distractor interference* (semantically similar irrelevant content) compounds the loss. Claude degrades slowest; Gemini fastest.

The takeaway is not "RAG is dead because of long context." It's *the opposite*: even with a 1M window, you should still feed the model focused, relevant content: because more isn't free, even when it fits.

Anthropic's own [Effective harnesses for long running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) post is explicit on this. Their long running agents keep durable state in *files and git* outside the context window, then rederive a focused context each session. The harness, not the window, owns the long horizon.

## The four sources of context

Every turn's context window is some mix of:

| Source | Examples | Owned by |
|---|---|---|
| **Static prefix** | System prompt, tool definitions, persona, examples | You, at build time |
| **Conversation history** | Past user/assistant/tool turns | Accumulates across turns |
| **Retrieved** | RAG chunks, knowledge graph hits, search results | Tool calls (lesson 03) |
| **Stateful** | Summary, scratchpad, persistent memory | Memory subsystem (lesson 04) |

Two of these matter most for cost and quality:

- **The static prefix should be genuinely static.** This unlocks prompt caching (lesson 02b): up to 90% off the input cost of every subsequent turn. Sneaking a timestamp, request ID, or user specific string into the "system" prompt silently invalidates the cache.
- **The dynamic tail should be focused.** Sending 20 retrieved chunks isn't 4x better than 5: it's usually worse (context rot + lost in the middle + 4x cost).

## Interleaved reasoning: the loop shape changed

In 2023 to 2024, "thinking" happened once: the model produced a plan, then executed it. Reasoning models (o3, Claude extended thinking, Gemini Deep Think, DeepSeek R1) shifted this.

> Modern reasoning models think **between** tool calls, not just before the first response.

On current Claude models thinking is *adaptive* and on by default: the model decides when and how much to think, and emits structured `thinking` blocks before tool calls *and* before text responses, adapting to what each tool returned rather than committing to a plan up front. The agent loop's shape changes:

```
old:  plan once → execute many → answer
new:  think → act → think → act → think → answer
```

Practical implications:

- You no longer switch thinking on with a token budget; you set **effort** (`low` to `max`), which controls how much the model thinks and how thorough it is. Raise it for hard multi step tasks; lower it for routine ones. Lesson 00c covers this.
- The reasoning is opaque to you: by default the thinking text comes back empty (a readable summary is opt in), but the blocks are part of the conversation history, so the model sees its own prior reasoning. Send them back unchanged and never rewrite earlier turns.
- It costs output tokens, and thinking counts toward `max_tokens`. Leave room for it.

This makes lesson 02's "repair loop" pattern even more powerful: the model not only sees the validation error, it explicitly reasons about what to change.

## Failure modes specific to context

- **Silent cache invalidation.** A "static" prefix that actually changes (date, user id, feature flag) writes the cache on every turn instead of reading it. Costs go up, you don't notice. Mitigation: keep volatile content out of the cacheable prefix.
- **Distractor interference.** Two chunks both mention "vacation policy" but only one is the right one for this question. The model averages them. Mitigation: relevance threshold + reranking (lesson 03a/03c).
- **Stale summary.** The summary was written 50 turns ago and no longer matches what the user wants. Mitigation: refresh summaries on goal change, not just on turn count.
- **Tool catalog bloat.** 40 tool definitions in every turn. Model accuracy on tool *selection* degrades past ~10 tools. Mitigation: agent side tool routing, or expose tools per task.
- **Hidden state escaping context.** The model "remembers" a decision from 30 turns ago that's no longer in context. It doesn't. Either summarize it forward or persist it.

## Hands on

### `01_context_rot.py`

A minimal needle in haystack: insert one specific fact at three different positions in a ~2K token document, ask the model to retrieve it, observe whether position matters. Then add a *distractor* (a fact that resembles the answer but is wrong) and rerun: observe that distractors hurt more than position does on modern models.

The demo is intentionally small so it costs cents to run. Crank the haystack size to reproduce the published findings yourself.

## Key takeaways

- The job's name changed from prompt engineering to context engineering because the artifact you design is now layered and dynamic, not a single string.
- Long context windows mitigate but don't solve lost in the middle or context rot. Focused context still wins.
- The static prefix is the cache prefix. Keep it actually static.
- Reasoning models think between tool calls, not just before them. The loop is `think → act → think → act → answer`, not `plan → execute`.
- Durable state belongs outside the window. The harness owns the long horizon.

## Common misconceptions

- **"Context windows solved RAG."** No: they shifted the tradeoff. RAG still wins on cost, freshness, provenance, and quality past ~100K tokens.
- **"Context engineering is just prompt engineering with a fancy name."** It's a superset. Prompt engineering writes one string; context engineering designs the policy that assembles every turn's request.
- **"Reasoning models replace agent loops."** They're orthogonal: a reasoning model still calls tools and still needs a loop around it. They change *what happens between calls*, not the outer shape.

Next: [Lesson 00c: Thinking and effort](../00c_thinking_and_effort/README.md).
