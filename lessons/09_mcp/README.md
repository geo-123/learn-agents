# Lesson 09 — MCP and Tool Servers

In Lesson 01 you hand-wired tools into the agent. Every agent application had its own `REGISTRY` of `(name, schema, fn)` tuples. That works for a single app. The moment you have **two** agents that should share a tool — or two teams that build agents in different languages — you have a problem.

**Model Context Protocol (MCP)** is an open protocol that solves this. A tool server publishes tools, resources, and prompts. Any compliant client can discover and invoke them. Your agent code stays the same; the tools become a service.

Anthropic introduced MCP and ships first-class support in Claude Desktop and Claude Code. The protocol itself is open and vendor-neutral — anyone can write a server, anyone can write a client.

## What MCP actually is

It's a **protocol**, like HTTP or LSP. Three primitives:

| Primitive | What it is | Example |
|---|---|---|
| **Tool** | Callable function | `search_docs(query, k)` |
| **Resource** | Readable data | `acme://handbook/vacation.md` |
| **Prompt** | Reusable parameterized template | `meeting_summary(transcript)` |

A server exposes some combination of these. A client (your agent) connects, lists what's available, and uses them. The wire format is JSON-RPC 2.0 over a transport.

### Transports

Three transports in the spec:

- **stdio** — server runs as a subprocess; messages over stdin/stdout. **Simplest, most common.** Zero networking, no port conflicts, no TLS.
- **HTTP / SSE** — server is a long-running web service.
- **WebSocket** — bidirectional, persistent connection.

This lesson uses stdio. Once you understand it, switching to HTTP is mostly a transport configuration change. We do NOT teach HTTP here because TLS, corporate proxies, and port management are distractions from the protocol itself.

## Why bother?

If MCP is just "RPC for tools," what does it buy you?

1. **Reuse.** Write the tool once; any MCP client can use it. Claude Desktop, Claude Code, your custom agent — same tool.
2. **Process isolation.** The tool server is a subprocess. If it crashes, it doesn't take the agent with it. You can also sandbox it (cgroups, AppArmor, separate user).
3. **Language polyglotism.** Tool server can be Rust; agent can be Python. The protocol is language-agnostic.
4. **Composition.** An agent can talk to multiple MCP servers (one for filesystem, one for company API, one for memory) and the agent code doesn't change shape.
5. **Auditability.** A separate process boundary means a separate log boundary. Audit per-server is straightforward.

The cost: indirection. For a one-off agent with three tools, the hand-wired registry is fine. MCP pays off when the tool catalog grows or you want reuse.

## The shape of a stdio MCP server

A FastMCP-flavored server is just a Python file:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Decorators register the tool. The server reads JSON-RPC from stdin, dispatches, writes responses to stdout. Run with `python server.py` — or have a client spawn it as a subprocess.

## The shape of a client

The client side is more code, but conceptually:

1. **Spawn** the server subprocess.
2. **Initialize**: client and server exchange version + capabilities.
3. **List tools**: client asks "what do you have?"; server returns names, descriptions, schemas.
4. **Translate** MCP tool schemas → your LLM's tool format (Anthropic, OpenAI, etc.).
5. **Run the agent loop**: when the model emits a tool_use, RPC it to the server, get back the result, feed to the model.
6. **Close**: send shutdown.

Steps 4 and 5 are where MCP meets Lesson 01's dispatch pattern. You're still doing tool calling — the dispatch just goes over RPC instead of calling a Python function in-process.

## Resources and prompts (not just tools)

Tools get the most attention, but the spec includes two more primitives:

- **Resources** — readable URIs. Think of them as files: `acme://handbook/vacation.md`. The client can list and read them. Useful for "here's a known set of documents the agent should be able to fetch."
- **Prompts** — parameterized templates. The server defines `summarize_meeting(transcript)`; the client can invoke it to get a ready-made prompt back. Useful for sharing prompt design across agents.

In practice, tools are 90% of MCP usage. Resources show up when the server has natural file-like content. Prompts are more rare — most teams define prompts in-app.

## Sandboxing and trust

MCP servers run with **your privileges**. A malicious or buggy server has the same access to your filesystem and network as the process that started it. Mitigations:

- **Vet the server source.** Treat MCP servers like any other dependency.
- **Sandbox the subprocess.** Linux: `firejail`, AppArmor. macOS: `sandbox-exec`. Containers also work.
- **Allowlist commands.** Only spawn MCP servers from a known directory.
- **Audit tool calls.** Log every dispatch.

The Claude Desktop config file is the most common attack vector — a user adds a tempting-looking server URL and gets pwned. Don't add MCP servers from untrusted sources.

## Failure modes

- **Server crashes mid-call.** Client hangs waiting for a response. Add a timeout per RPC.
- **Schema drift.** Server upgrades the tool's args; client hasn't refreshed its tool list. Always re-list on reconnect.
- **No auth on HTTP transport.** Anyone calls your tools. Auth is your responsibility — the spec doesn't enforce it.
- **Server tool description is poor.** Model doesn't know when to use the tool. The description still matters even when transport is fancy.
- **Subprocess leak.** Client doesn't shut down the server cleanly; orphan processes pile up.

## A2A — the inter-agent counterpart

MCP standardizes how an agent talks to **tools**. By mid-2025, the field needed a parallel standard for how an agent talks to **other agents** — for delegation, hand-off, multi-team workflows, and cross-vendor interop.

That standard is **A2A (Agent-to-Agent)**, originally introduced by Google in April 2025, donated to the Linux Foundation, and as of May 2026 supported by 150+ organizations including AWS, Cisco, Google, IBM, Microsoft, Salesforce, SAP, and ServiceNow. Five official SDKs ship (Python, JavaScript, Java, Go, .NET).

### MCP vs. A2A — the one-line distinction

> **MCP**: agent ⟷ tool. *"Run this function for me."*
>
> **A2A**: agent ⟷ agent. *"Take this task. Tell me when you're done."*

They're complementary. A typical 2026 stack uses both: your agent uses MCP to call its own tools, and A2A to delegate sub-tasks to other agents (potentially belonging to other teams or vendors).

### A2A primitives

| Primitive | What it is |
|---|---|
| **AgentCard** | A JSON document describing what an agent can do — skills, endpoints, auth, supported message types. Served at `/.well-known/agent.json`. The "business card" of an agent. |
| **Task** | A unit of work one agent asks another to perform. Has an id, state (`submitted` → `working` → `completed`/`failed`/`canceled`), and a list of messages. |
| **Message** | Text and/or structured parts exchanged within a task. The wire shape mirrors LLM chat messages. |
| **Artifact** | A typed output the receiving agent attaches to a completed task. |

Transport is JSON-RPC over HTTP (server-sent events for streaming updates). Authentication is bring-your-own — usually OAuth or signed bearer tokens; A2A doesn't mandate one scheme.

### Why this isn't just "another HTTP API"

The same reasoning that made MCP worth standardizing applies here:

- **Discoverability**: a single AgentCard format means any A2A client can negotiate with any A2A agent without bespoke integration.
- **Lifecycle**: tasks have a well-defined state machine. Long-running work doesn't need ad-hoc polling conventions.
- **Streaming and async**: results stream back as they're produced; the caller can subscribe instead of polling.
- **Auditability**: the protocol pins what a "delegation" looks like, so logging and policy can hook in uniformly.

### When you reach for A2A vs. just calling another LLM

If the other agent is *yours* and lives in the same process, just call its `run()` function. A2A pays off when:

- The agents belong to different teams (separate deploy lifecycles).
- The agents are written in different languages.
- You need streaming progress for a long-running task.
- You want a uniform way to express delegation across many backends.

For a single-vendor, single-team multi-agent system, the orchestration patterns in [Lesson 09b](../09b_multi_agent/README.md) are the cheaper choice.

## Hands-on

### `01_mcp_server.py`

A stdio MCP server with three tools (`search_docs`, `remember`, `recall`), one resource (the corpus), and one prompt template. Reuses code from earlier lessons. Run it standalone with `uv run python lessons/09_mcp/01_mcp_server.py` — it'll wait on stdin (that's the protocol).

### `02_mcp_client.py`

A client that spawns `01_mcp_server.py` as a subprocess, lists tools, translates them to Anthropic's tool format, runs the agent loop dispatching tool calls over MCP, and shuts down cleanly.

### `03_resources_and_prompts.py`

Demonstrate the other two primitives: list MCP resources from the server and `read` one; invoke the `meeting_summary` prompt template and see how it would be filled in.

### `04_a2a_concept.py`

Show what A2A messages look like on the wire — an AgentCard, a task submission, a streamed status update, a completed task with an artifact. We construct the payloads as plain dicts (no full A2A SDK dependency) so you can see the protocol's shape end to end.

## Key takeaways

- MCP is a protocol, not a framework. Stdio + JSON-RPC is the wire.
- Three primitives: tools, resources, prompts. Tools dominate in practice.
- Stdio transport is the right starting point — no networking concerns.
- Client side is still Lesson 01's tool-dispatch pattern, with RPC in the middle.
- Sandboxing is your responsibility. Vet servers like any other dependency.
- A2A is the inter-agent counterpart to MCP — same JSON-RPC discipline, different problem (delegation rather than function calls). Use both: MCP for your tools, A2A for cross-team agent collaboration.

## Common misconceptions

- **"MCP is an Anthropic product."** It's an open spec. Multiple clients and servers exist.
- **"MCP replaces my tools."** Your tools become MCP-served; the dispatch logic on the client is the same.
- **"MCP is overkill for one tool."** Yes — value emerges with reuse and process isolation.
- **"A2A and MCP overlap."** They don't. MCP exposes tools; A2A exposes agents. A single agent commonly uses both.

Next: [Lesson 09b — Multi-agent orchestration](../09b_multi_agent/README.md), then [Capstone](../10_capstone/README.md).
