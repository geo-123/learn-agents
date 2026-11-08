# Lesson 04b: Long Running Context

Lesson 04 built conversation memory by hand: a window, a summary written by an extra LLM call, a JSON file of facts. Those patterns still matter, because they are what every feature in this lesson does under the hood. But an agent that runs for hours, reads hundreds of files or calls tools thousands of times now has server side help. This lesson covers what the API does for you, what it does not, and how to choose.

The problem in one picture:

```
turn 1   [system][task][tool call][2k token result]
turn 20  [system][task][tool call][2k result][tool call][2k result] ... x20
                        └──────────── mostly stale, still paid for every turn ───┘
```

Three things go wrong as the transcript grows: **cost** (every turn resends everything), **quality** (answers get worse as the context fills with stale material, long before the window is full), and finally **the hard limit** (the request no longer fits). Each tool below attacks a different part of that.

## The rule that comes first: append, never edit

Before any technique, one rule for the harness itself: **only ever append to `messages`.** Do not rewrite, reorder or delete earlier turns, and do not change the top level `system` prompt or the `tools` list mid conversation.

Two reasons:

- **Prompt caching** (Lesson 02b) matches on the prefix. Edit turn 3 and everything after it is a cache miss.
- **Preserved thinking.** Current models return thinking blocks that are bound to the exact conversation that produced them. If you edit an earlier turn and replay a later thinking block, the API either drops that block (the model loses its reasoning) or, on accounts created after 31 August 2026, returns a 400.

So the hand made trimming from Lesson 04 (`messages = messages[-4:]` while keeping later turns with their thinking) is now a trap on current models. The append only replacements:

| You wanted to | Do this instead |
|---|---|
| Drop old tool results | Context editing (server side, below) |
| Shrink a long history | Server side compaction, or replace the *whole* history with one summary message |
| Change instructions mid run | Append a `{"role": "system", ...}` message; leave the top level `system` alone |
| Add a reminder for one turn | Append it as text after the tool results; leave earlier copies in place |

Server side compaction and context editing do not count as edits: the check compares the conversation *as you sent it*, and you sent it unchanged.

## Tool 1: Server side compaction

Compaction is Lesson 04's summarization, run by the API. You get back a `compaction` block: a summary and a signature. From then on that block stands in for the turns it summarizes.

There are two variants.

**On demand** (beta `compact-2026-09-04`): your code decides when. You send the conversation with one extra parameter and get a summary instead of a reply:

```python
summary = client().beta.messages.create(
    model="claude-haiku-5-5", max_tokens=4000, betas=["compact-2026-09-04"],
    system=SYSTEM, messages=history,
    compaction={"type": "summarize", "instructions": "Keep every number and deadline."},
)
if summary.stop_reason == "compaction":
    history = [{"role": "assistant", "content": summary.content}]  # the block goes first, alone
```

Rules that bite: send the block back exactly as returned (the signature is checked), keep it first in `messages`, send exactly one, and keep the beta header on every later request that carries it. Anything the summary cannot carry (images, uploaded files) is gone; restate what you still need. The docs recommend this variant wherever it is available.

**At a token threshold** (beta `compact-2026-01-12`): the API decides. One setting on your ordinary requests:

```python
client().beta.messages.create(
    ..., betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112",
                                   "trigger": {"type": "input_tokens", "value": 50_000}}]},
)
```

The trigger defaults to 150,000 input tokens and **must be at least 50,000**. When it fires, the response starts with a `compaction` block. The critical part: **append the full `response.content`**, not just the text. If you extract the text and append that, the compaction block is lost and the next request resends everything.

Both variants: the summary is all the model has of the compacted work, so if you write `instructions`, say what must survive (they replace the default prompt entirely). Count the cost from `usage.iterations`: the summarization call is billed there, not in the top level token counts.

## Tool 2: Context editing

Context editing (beta `context-management-2025-06-27`) **clears**; it does not summarize. Before the model reads the prompt, the API removes old tool results (or old thinking blocks) and puts a short placeholder where each one was.

By hand, this is a loop over your `messages` that swaps the content of every `tool_result` older than the last few for `"[cleared]"`. Try it and you hit the reason the feature exists: that rewrites history, so the prompt cache misses from the first changed turn on (Lesson 02b) and, on current models, every later thinking block stops matching (Lesson 04). The API applies the same edit on its side, after the cache lookup and with the thinking chain accounted for, while the history you hold stays untouched.

```python
context_management={"edits": [{
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 30_000},   # or {"type": "tool_uses", "value": N}
    "keep": {"type": "tool_uses", "value": 3},              # most recent results stay
    "clear_at_least": {"type": "input_tokens", "value": 5_000},
    "exclude_tools": ["web_search"],                         # never clear these
}]}
```

Defaults: trigger at 100,000 input tokens, keep the last 3 tool uses. `clear_thinking_20251015` does the same for thinking blocks (`keep` counts assistant turns); when you combine both, list the thinking edit first.

Three properties to remember:

- **Your history is untouched.** The clearing happens on the server's copy. You keep appending as usual.
- **Cleared means gone.** If the agent will need a number from an old result, it has to write that number down as text (or into memory, below) while the result is still visible.
- **It costs a cache write.** Each clear changes the prompt from that point on. `clear_at_least` makes sure every clear removes enough to be worth it.

The response tells you what happened in `response.context_management.applied_edits`.

## Tool 3: The memory tool

Compaction and context editing manage *one* conversation. The memory tool (`{"type": "memory_20250818", "name": "memory"}`) is for what must outlive it. It is a client side tool with a file system shape: the model sends `view`, `create`, `str_replace`, `insert`, `delete` and `rename` commands on paths under `/memories`, and your code carries them out on storage you choose.

It is Lesson 04's `remember`/`recall` idea with two upgrades: the model is trained on this exact tool, and the API adds a short protocol to the system prompt ("view your memory directory first, record progress as you go, assume your context may be reset"). An agent with memory can be interrupted, compacted or restarted and pick up from its own notes.

Your side of the contract is security. The model chooses the paths, so every path is untrusted input:

- accept only paths under `/memories`, then resolve them and check the result is still inside your folder (`Path.resolve()` plus `is_relative_to()`);
- refuse encoded or backslash traversal forms outright;
- never let it delete or rename the memory root;
- scope the folder per user in any multi user system, and keep secrets out of it.

The SDK has helpers (`BetaAbstractMemoryTool` to subclass, `BetaLocalFilesystemMemoryTool` ready made). `03_memory_tool.py` writes the backend by hand so every command is visible.

## Tool 4: Task budgets

`max_tokens` is a hard cap per response that the model never sees: it just gets cut off. A **task budget** is the opposite: an advisory token budget for a whole agent loop that the model *does* see, as a countdown, so it can pace itself and wrap up cleanly.

```python
with client().beta.messages.stream(
    model="claude-haiku-5-5", max_tokens=64000, betas=["task-budgets-2026-03-13"],
    output_config={"effort": "medium", "task_budget": {"type": "tokens", "total": 40_000}},
    tools=TOOLS, messages=messages,
) as stream:
    response = stream.get_final_message()
```

The minimum `total` is 20,000. The budget counts what the model generates and the tool results it reads, not the history you resend. Leave `remaining` unset in a normal loop (the server tracks it); pass it only if you rewrote history client side and the server cannot know what was spent. Do not send `remaining` together with on demand compaction: that is a 400. Use a budget to make the agent economical; keep `max_tokens` and your own step counter (Lesson 05) as the hard stops.

## Tool 5: Subagents

The cheapest context is context you never load. When a subtask needs to read a lot but report a little ("search these 40 files and tell me which one defines X"), give it to a subagent with its own fresh context window and take back only its answer. The parent's transcript grows by one short result instead of 40 file reads. Lesson 09b builds this pattern (orchestrator and workers).

## Choosing

| Situation | Reach for | Why |
|---|---|---|
| Old tool outputs pile up, the reasoning around them still matters | Context editing (`clear_tool_uses`) | Cheap, keeps the structure, no summary to go wrong |
| Long chat or agent run will approach the window | Compaction (on demand if you want control, threshold if you want zero code) | Keeps meaning, drops bulk |
| Facts must survive restarts, compaction or new sessions | Memory tool | The only one that persists outside the conversation |
| You want the agent to finish within a token spend, gracefully | Task budget | The model sees it and plans around it |
| A subtask reads far more than it reports | Subagent (Lesson 09b) | The reading never enters the parent context |
| Short conversations | None of the above | Plain append; caching does the rest |

They combine. A long running coding agent typically clears old tool results, compacts when it nears its budget, and keeps a progress file in memory so a compaction never loses the plan.

## Failure modes

- **Appending only the text.** The compaction block disappears and nothing is compacted. Append `response.content`.
- **Rebuilding the compaction block.** Changing its content or dropping the signature is a 400. Send it back verbatim.
- **Trimming history by hand on a thinking model.** Later thinking blocks are dropped or rejected. Use the server side tools, or replace the whole history with one summary.
- **Clearing results the agent still needs.** It fetches them again (cost) or guesses (quality). Have it note key values as text, use `exclude_tools`, or raise `keep`.
- **A summary that forgot the one fact.** Write `instructions` that name what to keep; pair compaction with memory for must keep facts.
- **Memory path traversal.** `/memories/../../.env` is a real attack. Resolve and check every path.

## Hands on

### `01_compaction.py`

Reads the eight handbook documents one per turn. After each turn it checks the next request size against a tiny limit and, when over, asks for an on demand compaction with custom instructions. Watch the size climb, drop to almost nothing after the compaction, and the final question about the first document get answered from the summary.

### `02_clear_tool_results.py`

The same tool heavy audit (six long ticket logs, one per turn) run twice: plain, then with `clear_tool_uses_20250919`. Compare the `input_tokens` column: the plain run climbs every turn, the cleared run levels off. The cleared run still gets the answer right because the agent writes each count down as text before the result is cleared.

### `03_memory_tool.py`

The memory tool with a hand written backend that maps `/memories` onto a `memories/` folder next to the script. Run it once to store facts, then again in a fresh conversation to have the agent find them in its own files. `--selftest` throws hostile paths at the guard without calling the API.

## Key takeaways

- Append only. Never edit earlier turns, the `system` prompt or the `tools` list mid run; it breaks caching and invalidates replayed thinking.
- Compaction summarizes, context editing clears, memory persists. They solve different problems and combine well.
- With compaction, always append the full `response.content` and send the block back verbatim.
- Context editing never touches your copy of the history, and cleared content is gone for the model.
- The memory tool is client side: you own the storage and the security of every path.
- `max_tokens` is a hard cap the model cannot see; a task budget is advice it can.
- The cheapest context is context a subagent read for you.

## Common misconceptions

- **"A 1M token window means I don't need any of this."** Quality drops as context fills with stale material, and every turn pays for all of it. Management pays off long before the limit.
- **"Context editing summarizes old tool results."** It clears them. Only compaction summarizes.
- **"Compaction is memory."** A compaction block lives in one conversation. Start a new one and it is gone. Memory survives.
- **"The memory tool stores data at Anthropic."** It is client side. The model asks; your code reads and writes your storage.
- **"A task budget stops the agent at N tokens."** It is advisory. `max_tokens` and your own step counter are the hard stops.

Next: [Lesson 05: Retries and errors](../05_retries_errors/README.md).
