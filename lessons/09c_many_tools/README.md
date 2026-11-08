# Lesson 09c: Agents with Many Tools

Lesson 01 had two tools. Lesson 09 showed how MCP makes it trivial to connect five servers with a dozen tools each. That is where a new problem starts: an agent with sixty tools is usually a *worse* agent than one with six. This lesson covers why, then builds the two fixes yourself, tool search and skills, with nothing but the Lesson 01 loop and the Lesson 03 retriever. Only then does it show the versions the API runs for you: tool search, Agent Skills and programmatic tool calling. It ends with the design advice that matters more than any of them: fewer, better tools.

## Why many tools hurt

Every tool definition is prompt text: name, description, JSON schema. It is sent on every request and read by the model before it does anything.

- **Context.** A typical setup with five MCP servers (code hosting, chat, error tracking, dashboards, logs) spends around 55,000 tokens on definitions before the first user word. That is cost on every turn and room taken from the actual work.
- **Cache.** Tools sit at the very front of the prompt (Lesson 02b). Add or remove one mid conversation and the whole cache is gone. Tool sets that change per request never cache at all.
- **Selection accuracy.** With 30 to 50 tools in view, the model starts picking the wrong one, especially when names and descriptions overlap (`search_tickets`, `find_issue`, `query_jira`).

The fixes all follow one idea, the same one as retrieval in Lesson 03: keep the fixed context small and **load detail on demand**.

## Build it yourself

Both fixes are small enough to write in an afternoon. Writing them is the fastest way to see what the API versions solve, because you hit the same walls they were built around.

### Tool search by hand

A tool catalog is a corpus. Each tool definition is a chunk: its name, description and argument descriptions are the text. Lesson 03c already built BM25 over chunks, so the search engine is done:

```python
INDEX = BM25Index([Chunk(name, i, search_text(tool)) for i, (name, tool) in enumerate(CATALOG.items())])

def find_tools(query, k=5):
    return [{"name": c.doc, "description": CATALOG[c.doc]["description"]} for c, _ in INDEX.search(query, k=k)]
```

The agent starts with `find_tools` and the three most used tools. The loop is the Lesson 01 dispatch loop with one extra step: when the model calls `find_tools`, the loop adds the full definitions of the matches to `tools`, and the next request carries them.

```
step 1: 4 tools sent, 1245 characters
step 1: find_tools('cancel equipment') -> ['hr_cancel_equipment', 'hr_submit_equipment', ...]
step 1: tools changed (+5): the next request starts a new cache prefix
step 2: 9 tools sent, ...
step 2: CALL hr_cancel_equipment({'record_id': 'EQ-77'})
```

It works. Four tools instead of forty on the first request, a handful more once the model knows what it needs. Now the costs:

- **The cache starts cold after every search.** Tools are the first thing in the prompt. Adding one changes the prefix of everything after it: system prompt, the whole history. Every search turns the next request into a full price cache write (Lesson 02b).
- **Earlier thinking stops matching.** On current models a thinking block is bound to everything sent before it, tools included. After the list changes, the API rejects the earlier blocks. The script keeps its history append only and strips thinking from the turns before the change on the copy it sends (`strip_thinking`, the same escape hatch Lesson 04 uses after trimming history). The request goes through, but the model loses its earlier reasoning.
- **Lexical search misses synonyms.** "cancel laptop" ranks the cancel tools of every area, because the catalog says "laptops", not "laptop", and BM25 does not stem. The model has to guess the catalog's words; a line in the system prompt naming the areas helps it guess right.
- **One extra turn per search.** The search is a full model round trip before the real call.

### Skills by hand

Tool search loads *tool definitions* on demand. A skill loads *instructions* on demand. The format is a folder with a `SKILL.md`: a short header with a name and a one line description, then the full instructions.

```markdown
---
name: expense-audit
description: Audit expense reports against the Acme travel and meal caps. Use when asked to check, flag or approve expenses.
---
# Expense audit

1. Hotel cap: $300 a night, $400 a night in NYC, London and Tokyo.
...
```

Building it takes three functions:

1. `build_index()` reads only the headers. The system prompt lists one line per skill.
2. `load_skill(name)` returns the full `SKILL.md`. The model calls it when a request matches a description.
3. The instructions arrive as a tool result at the end of the history.

That last point is why skills are cheap to build by hand, unlike tool search: the system prompt and the tool list never change. The cache and earlier thinking blocks stay valid; only the history grows, append only.

The one part to get right is the name. `load_skill` takes a string the model wrote and turns it into a file path, so it is untrusted input (Lesson 08). The script accepts only names from the index it built at start, matched against a strict pattern, and checks that the resolved path stays inside the skills folder. `../secrets` gets an error, not a file.

The limits:

- **Loaded instructions stay in context.** Every later turn pays for them. Load late, keep skills short.
- **The model decides.** A vague description means the skill is never loaded. Write descriptions that say *when*, like tool descriptions.
- **No scripts.** A real skill can ship scripts and reference files (level 3). Reading an extra file is the same pattern with the same path check. *Running* a skill's script means running code the model chose, which needs a sandbox (Lesson 08b).

## What the API does for you

Now that you know what each mechanism does, here are the versions the API runs for you, and what each one removes from your list of costs.

### Tool search

The API version keeps the whole list in every request, so the prefix never changes. You send every tool definition, but mark most of them `defer_loading: true`. Add one of the two search tools:

```python
{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}    # natural language queries
{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}  # Python regex patterns
```

The model starts out seeing only the search tool and your non deferred tools. When it needs something else, it searches; the API runs the search on its side (names, descriptions, argument names and argument descriptions), returns `tool_reference` entries for the matches (5 by default), and expands them into full definitions inline in the conversation. The model then calls the found tool with an ordinary `tool_use` that your code runs as usual.

Compared with the version you built: the search is the same idea, the ranking is theirs, and the found definitions are appended to the conversation rather than swapped into `tools`. That is what keeps the cache and the thinking blocks valid. It also saves the extra model turn: the search runs inside the same response.

What you see in the response:

```
server_tool_use          {"query": "cancel equipment request"}     <- ran on the server; never answer it
tool_search_tool_result  tool_references: [hr_cancel_equipment, ...]
tool_use                 hr_cancel_equipment({"record_id": "EQ-77"}) <- yours to run
```

Rules:

- **Never defer everything.** The search tool itself must not be deferred, and at least one tool must stay loaded, or the API returns a 400.
- **Keep your 3 to 5 most used tools loaded** so common requests need no search.
- **Send the full `tools` list every time**, deferred ones included. Deferral changes what the model reads, not what you send.
- **Pass the search blocks back unchanged** in the history. Discovered tools stay available in later turns without a new search, and because they are appended rather than swapped in, the cache survives.
- **Write for search.** Consistent prefixes (`hr_`, `github_`) let one query find a whole family. Put the words users actually say into descriptions. A line in the system prompt naming the tool families helps the model know what to look for.

When it pays: 10 or more tools, definitions over about 10,000 tokens, or a catalog that keeps growing. Under 10 small tools that are all used every time, skip it.

### Agent Skills

The same `SKILL.md` format you used by hand, with a folder of scripts and reference files next to it:

```markdown
---
name: acme-expense-audit
description: Audit Acme expense reports against the travel and meal caps. Use when asked to check, flag or approve expenses.
---
# Acme expense audit

1. Hotel cap: $300 a night, $400 in NYC, London and Tokyo.
2. Meals cap: $80 a day.
3. Run `scripts/audit.py <file.csv>` for a CSV; it prints one line per violation.
```

This is **progressive disclosure** in three levels:

1. Only the `name` and `description` of each skill sit in the prompt (a sentence each).
2. When a request matches, the model reads the full `SKILL.md`.
3. Only if the instructions call for it does it open the extra files or run the scripts.

Twenty skills cost twenty sentences up front, not twenty manuals. Your version did levels 1 and 2. The API adds level 3, because the skill lives in the code execution container: the sandbox you would otherwise have to build is already there.

On the Messages API, you list the skills for the request in `container`:

```python
from anthropic.lib import files_from_dir

skill = client().skills.create(files=files_from_dir("acme-expense-audit"))  # once; keep skill.id

response = client().messages.create(
    model="claude-haiku-5-5", max_tokens=8000,
    container={"skills": [{"type": "custom", "skill_id": skill.id, "version": "latest"},
                          {"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]},
    tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
    messages=[{"role": "user", "content": "Audit these expenses and return a spreadsheet of the violations: ..."}],
)
```

`type: "anthropic"` selects a prebuilt skill (`pptx`, `xlsx`, `docx`, `pdf`); `type: "custom"` one you uploaded. Up to 20 skills per request. Uploading is a one time setup step, not something to do per request; `client().skills.list()` shows what you have. Files the skill writes come back as file ids you download through the Files API.

Skills or tools? A **tool** is an action your code performs, with typed arguments you can validate, gate and log. A **skill** is know how: steps, conventions, scripts that run in the sandbox. "Send an email" is a tool. "How we write incident reports" is a skill.

No script for the API version: running it uploads a skill to your organization, which is a setup step rather than a lesson step. `02_skills_by_hand.py` is the mechanism; the snippet above is the whole shape of the hosted one.

### Programmatic tool calling

Ordinary tool use is one round trip per call: the model calls, the result lands in its context, it reads it, it calls the next. Thirty records means thirty turns and thirty results in context, most of which the model only needed to compare against a number.

With programmatic tool calling, the model writes code that calls your tools:

```python
tools = [
    {"type": "code_execution_20260120", "name": "code_execution"},
    {"name": "get_expense", "description": "...", "input_schema": {...},
     "allowed_callers": ["code_execution_20260120"]},
]
```

The model's script runs in the code execution container. When it awaits `get_expense(...)`, the container pauses and you get a normal `tool_use` block, with one addition: `caller` names the code execution run instead of `direct`. You run the tool and send the result back; it goes to the script, not into the model's context. Loops, filters and sums happen in code. Only what the script prints comes back to the model.

#### What it does under the hood

There is no new idea here, only a new place where the loop runs.

1. **The model writes code.** Instead of one `tool_use`, it emits a Python script, as it would for code execution (Lesson 01c). Your tools appear inside it as ordinary async functions.
2. **The code runs in a sandbox.** Anthropic's container, not your machine.
3. **Each tool call pauses the code.** When the script awaits `get_expense("EXP-007")`, the container stops at that line. The API ends the response with `stop_reason: "tool_use"` and hands you the call as a `tool_use` block whose `caller` names the running script.
4. **Your result goes back into the running code.** You answer with a `tool_result` as always. The API feeds it to the paused script as the return value of that await, and the script continues. The result never becomes part of the model's context.
5. **Only the final output reaches the model.** When the script ends, what it printed comes back as a code execution result, and the model reads that: a short list of flags instead of thirty records.

Building this yourself means building the sandbox. The orchestration is easy: a generator that yields at every tool call is the same pause and resume. The hard part is step 2, running code a model wrote: file system, network, processes, time and memory limits. Lesson 08b shows how much harness one bash tool needs to be safe on your machine, and Lesson 01c shows the hosted container that removes that work. This lesson does not run model written code locally, and neither should a first version of yours. Use the hosted one.

The protocol details:

- The reply to pending programmatic calls holds **only** `tool_result` blocks. No text.
- Pass the `container` id from the paused response on the next request; a continuation without it is rejected.
- Send the same `tools` list again; the code execution tool must still be there.
- A paused call times out after about 4 minutes, so answer promptly.
- Not combinable with `strict: true` tools, `disable_parallel_tool_use`, or forcing a tool through `tool_choice`.
- `allowed_callers` steers the model; it is not a security boundary. Your handler must still validate every call, whoever made it.

When it pays: many calls of the same kind (loops over records, pagination), large intermediate results that only need filtering, or chains where each step just feeds the next. When it does not: a single call, or results the model must actually read and reason about.

## Designing the tool surface

The features above make a big tool set workable. The better fix is a smaller one.

- **Fewer, better tools.** One `hr_leave(action, record_id)` with a clear enum often beats five near duplicates. Merge tools the model confuses; delete tools nobody calls (your traces from Lesson 07 tell you which).
- **Descriptions say when, not just what.** "Call this when the user asks about time off" beats "Returns leave records".
- **Return what the model needs.** A tool that returns 5,000 tokens of JSON when the model needs three fields is a context leak on every call. Filter in the tool.
- **Bash versus dedicated tools.** A `bash` tool gives the model enormous reach with one definition, but your harness only sees an opaque command string. Promote an action to its own tool when you need to **gate** it (irreversible actions: sending, deleting, paying), **check** it (refuse an edit if the file changed since it was read), **render** it in a UI, or **parallelize** it safely (a read only `grep` can run alongside others; an arbitrary shell command cannot be assumed safe). Start broad, promote what needs a hook.
- **Static tool lists.** Decide the tool set at the start of a conversation and keep it. Changing it mid run breaks the cache, and on thinking models it invalidates replayed reasoning (Lesson 04b). You saw both in `01_tool_search_by_hand.py`. Tool search and deferral are the append only way to grow the set.

## Failure modes

- **Swapping `tools` by hand without stripping thinking.** On current models the next request is a 400: the earlier thinking blocks are bound to the old list. Strip them on the copy you send, or use the API's tool search.
- **A skill name used as a path.** `load_skill("../../.env")` reads whatever it points at unless you check. Accept only names from your index and keep the resolved path inside the skills folder.
- **Every tool deferred.** A 400. Keep the search tool and a few common tools loaded.
- **Answering the search.** The `server_tool_use` id belongs to the server; sending a `tool_result` for it is a 400.
- **Dropping deferred definitions on later turns.** A `tool_reference` to a tool missing from `tools` is a 400. Send the full list every time.
- **Vague names, vague descriptions.** Search cannot find what the text does not say, yours or the API's. Name tools by family and write the words users use.
- **Text next to programmatic results.** The continuation is rejected. Only `tool_result` blocks.
- **Lost container id.** The paused script cannot resume. Track `response.container.id` across turns.
- **Using programmatic calling as a sandbox for trust.** The code runs in the container, but every tool call still runs in your process with your permissions. Validate as for any tool call.

## Hands on

### `01_tool_search_by_hand.py`

The 40 HR tools of `03_tool_search.py` (imported from it, so the catalog is identical), searched with the Lesson 03c BM25. The agent starts with `find_tools` and three loaded tools; the loop adds what the search finds. Each step prints the tool count, the definition size in characters, the query, the matches, and when the tool list changed. Try "cancel my laptop request" and watch whether the model's query uses the catalog's words.

### `02_skills_by_hand.py`

Writes three tiny skills into `skills/` (recreated on every run, not tracked by git), lists their names and descriptions in the system prompt, and gives the model one `load_skill` tool. It prints the prompt size against the full skill text, which skill is loaded on which step, and the answer. Try a request no skill covers: the model should answer without loading one.

### `03_tool_search.py`

Generates 40 HR tools from a table of 8 areas and 5 actions, keeps 3 loaded and defers 37, and adds the BM25 search tool (`--regex` switches to the regex variant). It first prints the prompt tokens of the tool definitions with and without deferral, then runs a request and shows the search query, the tools it found and the call that follows. Try requests from different areas and watch which queries the model writes. Compare with `01_tool_search_by_hand.py`: same catalog, no change to `tools`, no stripped thinking.

### `04_programmatic_tool_calling.py`

Thirty fake expense reports and one `get_expense` tool callable only from code. The model writes a script that fetches and checks all thirty against the Acme caps. The trace shows each paused turn with its batch of tool calls (each tagged with its `caller`), how few model turns the audit takes, and the expected flags computed locally so you can check the answer.

## Key takeaways

- Tool definitions are prompt tokens on every turn; past 30 to 50 tools, selection accuracy drops too.
- Tool search is retrieval over a tool catalog. By hand it is BM25 plus a loop that grows `tools`, and growing `tools` costs the cache and the earlier thinking.
- The API's tool search avoids that cost: defer most tools, keep a few loaded, never defer the search tool, send every definition on every request.
- Skills by hand are a header index, a `load_skill` tool and a path check. The prompt prefix never changes, so they are cheap to build yourself.
- Skills load instructions in three levels: description always, `SKILL.md` when relevant, extra files only when needed. The third level needs a sandbox, which is what the hosted version adds.
- Programmatic tool calling moves loops and filtering into code; intermediate results never enter the model's context. Building it yourself means building a sandbox.
- The best fix is still design: fewer tools, sharper descriptions, smaller results, and dedicated tools only where the harness needs a hook.

## Common misconceptions

- **"Tool search needs a vector database."** Forty tool descriptions are a tiny corpus. BM25 over names and descriptions is a fine start; the API's own search is BM25 or regex.
- **"Adding a tool mid conversation is free."** It changes the prompt prefix: the cache restarts, and earlier thinking blocks no longer match.
- **"Deferred tools are not sent."** They are sent on every request. They are just not shown to the model until found.
- **"More tools make the agent more capable."** Past a few dozen they make it less accurate and more expensive.
- **"A skill is a kind of tool."** A tool is an action your code runs. A skill is packaged know how the model reads and follows.
- **"Programmatic tool calling runs my tools in Anthropic's sandbox."** The script runs there; your tools still run in your process, one `tool_use` at a time.
- **"`allowed_callers` stops direct calls."** It guides the model. Validate every call regardless.

Next: [Lesson 10: Capstone](../10_capstone/README.md).
