# Lesson 09: MCP and Tool Servers

In Lesson 01 you hand wired tools into the agent. Every agent application had its own `REGISTRY` of `(name, schema, fn)` tuples. That works for a single app. The moment you have **two** agents that should share a tool (or two teams that build agents in different languages) you have a problem.

**Model Context Protocol (MCP)** is an open protocol that solves this. A tool server publishes tools, resources, and prompts. Any compliant client can discover and invoke them. Your agent code stays the same; the tools become a service.

Anthropic introduced MCP and ships first class support in Claude Desktop and Claude Code. The protocol itself is open and vendor neutral: anyone can write a server, anyone can write a client.

## What MCP actually is

It's a **protocol**, like HTTP or LSP. Three primitives:

| Primitive | What it is | Example |
|---|---|---|
| **Tool** | Callable function | `search_docs(query, k)` |
| **Resource** | Readable data | `acme://handbook/vacation.md` |
| **Prompt** | Reusable parameterized template | `meeting_summary(transcript)` |

A server exposes some combination of these. A client (your agent) connects, lists what's available, and uses them. The wire format is JSON-RPC 2.0 over a transport.

### Transports

Two standard transports in the current spec:

- **stdio**: server runs as a subprocess; messages over stdin/stdout. **Simplest, most common for local tools.** Zero networking, no port conflicts, no TLS. One rule: stdout belongs to the protocol, so the server must never `print` to it (logs go to stderr). `01_mcp_server.py` shows the fix for a library that prints.
- **Streamable HTTP**: server is a long running web service with one endpoint (by convention `/mcp`). The client POSTs JSON-RPC messages; the server answers with plain JSON or, for long work, streams the answer as server sent events on the same response. This is what remote MCP servers use.

The older **HTTP plus SSE** transport (one endpoint to open an event stream, a second one to POST messages) is **deprecated** in the spec and replaced by Streamable HTTP. SDKs still ship it for old clients (`mcp.run(transport="sse")`); do not build new servers on it. WebSocket is not a standard MCP transport.

This lesson uses stdio. Once you understand it, switching to Streamable HTTP is mostly a transport configuration change: `mcp.run(transport="streamable-http", port=8000)` on the server, `Client("http://localhost:8000/mcp")` on the client. We do NOT teach HTTP here because TLS, corporate proxies, auth, and port management are distractions from the protocol itself.

## Why bother?

If MCP is just "RPC for tools," what does it buy you?

1. **Reuse.** Write the tool once; any MCP client can use it. Claude Desktop, Claude Code, your custom agent: same tool.
2. **Process isolation.** The tool server is a subprocess. If it crashes, it doesn't take the agent with it. You can also sandbox it (cgroups, AppArmor, separate user).
3. **Language polyglotism.** Tool server can be Rust; agent can be Python. The protocol is language agnostic.
4. **Composition.** An agent can talk to multiple MCP servers (one for filesystem, one for company API, one for memory) and the agent code doesn't change shape.
5. **Auditability.** A separate process boundary means a separate log boundary. Audit per server is straightforward.

The cost: indirection. For a one off agent with three tools, the hand wired registry is fine. MCP pays off when the tool catalog grows or you want reuse.

## The shape of a stdio MCP server

With the official Python SDK (`mcp` 2.x), a server is just a Python file:

```python
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("my-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Decorators register the tool; the type hints and docstring become its JSON schema and description. The server reads JSON-RPC from stdin, dispatches, writes responses to stdout. Run with `python server.py`: or have a client spawn it as a subprocess.

> **Coming from mcp 1.x?** The high level server class was `FastMCP` (`from mcp.server.fastmcp import FastMCP`). Version 2 renamed it `MCPServer` and moved it to `mcp.server.mcpserver`; the old import now fails with a pointer to the migration guide. The `@tool`, `@resource` and `@prompt` decorators kept their arguments. Other changes you will meet: result fields are snake_case (`input_schema`, `is_error`, `uri_template`, `mime_type`), URIs are plain strings, transport options (host, port) moved from the constructor to `run()`, and a tool error raised as `MCPError` reaches the client as a JSON-RPC error instead of an `is_error` result.

## The shape of a client

The client side is more code, but conceptually:

1. **Spawn** the server subprocess.
2. **Initialize**: client and server agree on a protocol version and exchange capabilities.
3. **List tools**: client asks "what do you have?"; server returns names, descriptions, schemas.
4. **Translate** MCP tool schemas → your LLM's tool format (Anthropic, OpenAI, etc.).
5. **Run the agent loop**: when the model emits a tool_use, RPC it to the server, get back the result, feed to the model.
6. **Close**: send shutdown.

In `mcp` 2.x the `Client` class does steps 1, 2 and 6 for you:

```python
from mcp import Client, StdioServerParameters

params = StdioServerParameters(command="python", args=["server.py"])
async with Client(params) as client:          # spawn + handshake
    tools = (await client.list_tools()).tools  # each has .name, .description, .input_schema
    result = await client.call_tool("add", {"a": 1, "b": 2})
# leaving the block shuts the server down
```

The same `Client` takes a URL for a Streamable HTTP server. The 1.x pair `stdio_client(...)` plus `ClientSession(...)` with an explicit `await session.initialize()` still works.

Steps 4 and 5 are where MCP meets Lesson 01's dispatch pattern. You're still doing tool calling: the dispatch just goes over RPC instead of calling a Python function in process.

Tool results and prompt messages carry a list of **content blocks**, and each block is one of several types (text, image, audio, resource link, embedded resource). Narrow by type (`isinstance(block, TextContent)`) before you read `.text`; resource contents are likewise text or blob.

## Letting the Claude API connect for you: the MCP connector

Bridging tools yourself (the client above) works with any MCP server, including local stdio ones. When the server is **remote** (reachable over Streamable HTTP from the internet), the Claude API can connect to it itself: the API lists the server's tools, calls them during the turn, and returns the calls and results as content blocks. You write no client and no dispatch loop.

It is a beta and needs both halves in one request: the `mcp_servers` parameter with the connection, and an `mcp_toolset` entry in `tools` that names the server. Leaving out the toolset entry is a validation error.

```python
from anthropic import Anthropic

client = Anthropic()
response = client.beta.messages.create(
    model="claude-haiku-5-5",
    max_tokens=4000,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{
        "type": "url",
        "url": "https://handbook.example.com/mcp",
        "name": "acme-handbook",
        "authorization_token": "...",   # optional
    }],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "acme-handbook"}],
    messages=[{"role": "user", "content": "What is Acme's vacation policy?"}],
)
```

`mcp_server_name` must match a `name` in `mcp_servers`, and every server needs exactly one toolset. The toolset takes optional `default_config` (for example `{"enabled": False}` to start from an allowlist) and `configs` (per tool overrides keyed by tool name).

When to use which:

| | Bridge it yourself (`02_mcp_client.py`) | MCP connector |
|---|---|---|
| Server location | anywhere, including local stdio | remote URL the API can reach |
| Resources and prompts | yes | tools only |
| Your code | client, schema translation, dispatch loop | one request parameter |
| Control (timeouts, logging, approval per call) | full | what the toolset config offers |
| Platforms | any | beta on the Claude API, Claude Platform on AWS and Microsoft Foundry; not on Amazon Bedrock or Google Cloud |

A middle way: the SDK's `anthropic[mcp]` extra ships helpers (`async_mcp_tool`, `mcp_message`, `mcp_resource_to_content`) that turn MCP tools, prompts and resources from your own `ClientSession` into API types for the tool runner. They work with `mcp` 1.x and 2.x.

## Resources and prompts (not just tools)

Tools get the most attention, but the spec includes two more primitives:

- **Resources**: readable URIs. Think of them as files: `acme://handbook/vacation.md`. The client can list and read them. Useful for "here's a known set of documents the agent should be able to fetch."
- **Prompts**: parameterized templates. The server defines `summarize_meeting(transcript)`; the client can invoke it to get a ready made prompt back. Useful for sharing prompt design across agents.

In practice, tools are 90% of MCP usage. Resources show up when the server has natural file like content. Prompts are more rare: most teams define prompts in app.

## Sandboxing and trust

MCP servers run with **your privileges**. A malicious or buggy server has the same access to your filesystem and network as the process that started it. Mitigations:

- **Vet the server source.** Treat MCP servers like any other dependency.
- **Sandbox the subprocess.** Linux: `firejail`, AppArmor. macOS: `sandbox-exec`. Containers also work.
- **Allowlist commands.** Only spawn MCP servers from a known directory.
- **Audit tool calls.** Log every dispatch.

The Claude Desktop config file is the most common attack vector: a user adds a tempting looking server URL and gets pwned. Don't add MCP servers from untrusted sources.

## Failure modes

- **Server crashes mid call.** Client hangs waiting for a response. Add a timeout per RPC (`Client(..., read_timeout_seconds=30)`; in 2.x timeouts are float seconds).
- **Schema drift.** Server upgrades the tool's args; client hasn't refreshed its tool list. Always relist on reconnect.
- **No auth on HTTP transport.** Anyone calls your tools. The spec defines OAuth based authorization for HTTP servers, but it is optional: turning it on is your responsibility.
- **Server tool description is poor.** Model doesn't know when to use the tool. The description still matters even when transport is fancy.
- **Subprocess leak.** Client doesn't shut down the server cleanly; orphan processes pile up. Use `async with Client(...)` so the exit path always runs.
- **Stray stdout on stdio.** A `print` in the server (or in a library it imports) lands in the protocol stream and the client fails to parse it. Log to stderr.

## A2A: the interagent counterpart

MCP standardizes how an agent talks to **tools**. By mid 2025, the field needed a parallel standard for how an agent talks to **other agents**: for delegation, handoff, multi team workflows, and cross vendor interop.

That standard is **A2A (Agent to Agent)**, originally introduced by Google in April 2025, donated to the Linux Foundation, and as of May 2026 supported by 150+ organizations including AWS, Cisco, Google, IBM, Microsoft, Salesforce, SAP, and ServiceNow. Five official SDKs ship (Python, JavaScript, Java, Go, .NET). The specification reached **version 1.0**; this lesson's wire examples follow it.

### MCP vs. A2A: the one line distinction

> **MCP**: agent ⟷ tool. *"Run this function for me."*
>
> **A2A**: agent ⟷ agent. *"Take this task. Tell me when you're done."*

They're complementary. A typical 2026 stack uses both: your agent uses MCP to call its own tools, and A2A to delegate subtasks to other agents (potentially belonging to other teams or vendors).

### A2A primitives

| Primitive | What it is |
|---|---|
| **AgentCard** | A JSON document describing what an agent can do: skills, endpoints (`supportedInterfaces`, one per protocol binding), auth (`securitySchemes`), input and output media types. Served at `/.well-known/agent-card.json`. The "business card" of an agent. |
| **Message** | One turn from a client (`ROLE_USER`) or an agent (`ROLE_AGENT`), made of parts. Each part holds exactly one of `text`, `raw` (bytes), `url` or `data` (JSON). The wire shape mirrors LLM chat messages. |
| **Task** | The stateful unit of work an agent opens when a message needs more than a quick reply. Has an id, a context id that groups related tasks, and a state (`TASK_STATE_SUBMITTED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED`, `FAILED`, `CANCELED` or `REJECTED`, with `INPUT_REQUIRED` and `AUTH_REQUIRED` as pauses). |
| **Artifact** | A typed output (a list of parts) the receiving agent attaches to a task. |

Three protocol bindings are standard: JSON-RPC over HTTP (methods such as `SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`), gRPC, and plain HTTP+JSON (REST). Streaming uses server sent events: each event carries a task, a message, a `statusUpdate` or an `artifactUpdate`. Authentication is bring your own: the AgentCard declares the schemes (OAuth, OpenID Connect, bearer tokens, API keys, mutual TLS) and A2A doesn't mandate one.

### Why this isn't just "another HTTP API"

The same reasoning that made MCP worth standardizing applies here:

- **Discoverability**: a single AgentCard format means any A2A client can negotiate with any A2A agent without bespoke integration.
- **Lifecycle**: tasks have a well defined state machine. Long running work doesn't need ad hoc polling conventions.
- **Streaming and async**: results stream back as they're produced; the caller can subscribe instead of polling.
- **Auditability**: the protocol pins what a "delegation" looks like, so logging and policy can hook in uniformly.

### When you reach for A2A vs. just calling another LLM

If the other agent is *yours* and lives in the same process, just call its `run()` function. A2A pays off when:

- The agents belong to different teams (separate deploy lifecycles).
- The agents are written in different languages.
- You need streaming progress for a long running task.
- You want a uniform way to express delegation across many backends.

For a single vendor, single team multi agent system, the orchestration patterns in [Lesson 09b](../09b_multi_agent/README.md) are the cheaper choice.

## Hands on

### `01_mcp_server.py`

A stdio MCP server with three tools (`search_docs`, `remember`, `recall`), one resource (the corpus), and one prompt template. Reuses code from earlier lessons. Run it standalone with `uv run python lessons/09_mcp/01_mcp_server.py`: it'll wait on stdin (that's the protocol).

### `02_mcp_client.py`

A client that spawns `01_mcp_server.py` as a subprocess through the 2.x `Client`, lists tools, translates them to Anthropic's tool format, runs the agent loop dispatching tool calls over MCP (passing MCP tool errors on as `is_error` tool results), and shuts down cleanly.

### `03_resources_and_prompts.py`

Demonstrate the other two primitives: list MCP resource templates from the server and `read` one; invoke the `meeting_summary` prompt template and see how it would be filled in. Runs without the Claude API.

### `04_a2a_concept.py`

Show what A2A 1.0 messages look like on the wire (JSON-RPC binding): an AgentCard, a `SendStreamingMessage` request, the streamed task, status and artifact events, and the final artifact. We construct the payloads as plain dicts (no full A2A SDK dependency) so you can see the protocol's shape end to end.

## Key takeaways

- MCP is a protocol, not a framework. Stdio + JSON-RPC is the wire.
- Three primitives: tools, resources, prompts. Tools dominate in practice.
- Stdio transport is the right starting point: no networking concerns. Streamable HTTP is the remote transport; HTTP plus SSE is deprecated.
- The Python SDK's server class is `MCPServer` in `mcp` 2.x (it was `FastMCP` in 1.x).
- Client side is still Lesson 01's tool dispatch pattern, with RPC in the middle. For a remote server, the Claude API's MCP connector (`mcp_servers` plus an `mcp_toolset` tool) can do that bridging for you.
- Sandboxing is your responsibility. Vet servers like any other dependency.
- A2A is the interagent counterpart to MCP: same JSON-RPC discipline, different problem (delegation rather than function calls). Use both: MCP for your tools, A2A for cross team agent collaboration.

## Common misconceptions

- **"MCP is an Anthropic product."** Anthropic introduced it, but it's an open spec, now governed under the Linux Foundation (Agentic AI Foundation). Multiple clients and servers exist.
- **"MCP replaces my tools."** Your tools become MCP served; the dispatch logic on the client is the same.
- **"MCP is overkill for one tool."** Yes: value emerges with reuse and process isolation.
- **"A2A and MCP overlap."** They don't. MCP exposes tools; A2A exposes agents. A single agent commonly uses both.

Next: [Lesson 09b: Multi agent orchestration](../09b_multi_agent/README.md), then [Capstone](../10_capstone/README.md).
