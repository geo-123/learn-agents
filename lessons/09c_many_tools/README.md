# Lesson 09c: Agents with Many Tools

Lesson 01 had two tools. Lesson 09 showed how MCP makes it trivial to connect five servers with a dozen tools each. That is where a new problem starts: an agent with sixty tools is usually a *worse* agent than one with six. This lesson covers why, and the three things the API now offers to fix it: tool search, Agent Skills and programmatic tool calling. It ends with the design advice that matters more than any of them: fewer, better tools.

## Why many tools hurt

Every tool definition is prompt text: name, description, JSON schema. It is sent on every request and read by the model before it does anything.

- **Context.** A typical setup with five MCP servers (code hosting, chat, error tracking, dashboards, logs) spends around 55,000 tokens on definitions before the first user word. That is cost on every turn and room taken from the actual work.
- **Cache.** Tools sit at the very front of the prompt (Lesson 02b). Add or remove one mid conversation and the whole cache is gone. Tool sets that change per request never cache at all.
- **Selection accuracy.** With 30 to 50 tools in view, the model starts picking the wrong one, especially when names and descriptions overlap (`search_tickets`, `find_issue`, `query_jira`).

The fixes all follow one idea, the same one as retrieval in Lesson 03: keep the fixed context small and **load detail on demand**.

## Tool search

You send every tool definition, but mark most of them `defer_loading: true`. Add one of the two search tools:

```python
{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}    # natural language queries
{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}  # Python regex patterns
```

The model starts out seeing only the search tool and your non deferred tools. When it needs something else, it searches; the API runs the search on its side (names, descriptions, argument names and argument descriptions), returns `tool_reference` entries for the matches (5 by default), and expands them into full definitions. The model then calls the found tool with an ordinary `tool_use` that your code runs as usual.

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

## Agent Skills

Tool search loads *tool definitions* on demand. Skills load *instructions and files* on demand. A skill is a folder with a `SKILL.md` and any scripts or reference files it needs:

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

Twenty skills cost twenty sentences up front, not twenty manuals.

On the Messages API, skills run inside the code execution container. You list the skills for the request in `container`:

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

No script for this section: running it uploads a skill to your organization, which is a setup step rather than a lesson step. The snippet above is the whole shape.

## Programmatic tool calling

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
- **Static tool lists.** Decide the tool set at the start of a conversation and keep it. Changing it mid run breaks the cache, and on thinking models it invalidates replayed reasoning (Lesson 04b). Tool search and deferral are the append only way to grow the set.

## Failure modes

- **Every tool deferred.** A 400. Keep the search tool and a few common tools loaded.
- **Answering the search.** The `server_tool_use` id belongs to the server; sending a `tool_result` for it is a 400.
- **Dropping deferred definitions on later turns.** A `tool_reference` to a tool missing from `tools` is a 400. Send the full list every time.
- **Vague names, vague descriptions.** Search cannot find what the text does not say. Name tools by family and write the words users use.
- **Text next to programmatic results.** The continuation is rejected. Only `tool_result` blocks.
- **Lost container id.** The paused script cannot resume. Track `response.container.id` across turns.
- **Using programmatic calling as a sandbox for trust.** The code runs in the container, but every tool call still runs in your process with your permissions. Validate as for any tool call.

## Hands on

### `01_tool_search.py`

Generates 40 HR tools from a table of 8 areas and 5 actions, keeps 3 loaded and defers 37, and adds the BM25 search tool (`--regex` switches to the regex variant). It first prints the prompt tokens of the tool definitions with and without deferral, then runs a request and shows the search query, the tools it found and the call that follows. Try requests from different areas and watch which queries the model writes.

### `02_programmatic_tool_calling.py`

Thirty fake expense reports and one `get_expense` tool callable only from code. The model writes a script that fetches and checks all thirty against the Acme caps. The trace shows each paused turn with its batch of tool calls (each tagged with its `caller`), how few model turns the audit takes, and the expected flags computed locally so you can check the answer.

## Key takeaways

- Tool definitions are prompt tokens on every turn; past 30 to 50 tools, selection accuracy drops too.
- Tool search: defer most tools, keep a few loaded, never defer the search tool, send every definition on every request.
- Skills load instructions in three levels: description always, `SKILL.md` when relevant, extra files only when needed.
- Programmatic tool calling moves loops and filtering into code; intermediate results never enter the model's context.
- The best fix is still design: fewer tools, sharper descriptions, smaller results, and dedicated tools only where the harness needs a hook.

## Common misconceptions

- **"Deferred tools are not sent."** They are sent on every request. They are just not shown to the model until found.
- **"More tools make the agent more capable."** Past a few dozen they make it less accurate and more expensive.
- **"A skill is a kind of tool."** A tool is an action your code runs. A skill is packaged know how the model reads and follows.
- **"Programmatic tool calling runs my tools in Anthropic's sandbox."** The script runs there; your tools still run in your process, one `tool_use` at a time.
- **"`allowed_callers` stops direct calls."** It guides the model. Validate every call regardless.

Next: [Lesson 10: Capstone](../10_capstone/README.md).
