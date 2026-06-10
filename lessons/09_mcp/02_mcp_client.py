"""Lesson 09 — MCP client that drives the agent loop over RPC.

Spawns 01_mcp_server.py as a subprocess, lists its tools, translates
them into Anthropic's tool format, and runs the agent loop. Each
tool_use the model emits becomes an MCP call to the subprocess.

This is the same agent loop as Lesson 01 — the dispatch just goes over
the protocol instead of calling local Python functions.

Run:
    uv run python lessons/09_mcp/02_mcp_client.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "01_mcp_server.py")],
)


def to_anthropic_tools(mcp_tools) -> list[dict]:
    """Translate MCP tool descriptors to Anthropic's tool schema."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def call_mcp_tool(session: ClientSession, name: str, args: dict) -> str:
    """Invoke a tool on the MCP server and return its result as a string."""
    result = await session.call_tool(name, args)
    # MCP returns a CallToolResult; we flatten the content blocks to text.
    pieces = []
    for block in result.content:
        if hasattr(block, "text"):
            pieces.append(block.text)
        else:
            pieces.append(json.dumps(block.model_dump()))
    return "\n".join(pieces)


SYSTEM = (
    "You are a helpful assistant with access to the Acme handbook and "
    "a persistent memory. Use search_docs for handbook questions. Use "
    "remember/recall for facts about the user. Cite which docs you used."
)


async def run_agent(question: str, *, max_steps: int = 6) -> str:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_list = await session.list_tools()
            tools = to_anthropic_tools(tool_list.tools)
            print(f"\n[mcp] connected; available tools: "
                  f"{[t['name'] for t in tools]}")

            messages: list[dict] = [{"role": "user", "content": question}]

            for step in range(1, max_steps + 1):
                response = chat(messages, system=SYSTEM, tools=tools)
                messages.append({"role": "assistant", "content": response.content})

                print(f"\n--- step {step}  stop={response.stop_reason} ---")
                if response.text:
                    print(f"text: {response.text[:200]}")
                for tu in response.tool_uses:
                    print(f"  -> mcp.call_tool({tu['name']}, {tu['input']})")

                if response.stop_reason != "tool_use":
                    return response.text

                # Dispatch each tool_use over MCP. (Awaited sequentially here
                # for clarity; production code can asyncio.gather for parallel.)
                results = []
                for tu in response.tool_uses:
                    out = await call_mcp_tool(session, tu["name"], tu["input"])
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": out,
                    })
                messages.append({"role": "user", "content": results})

    return "(max steps)"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is Acme's vacation policy?"
    answer = asyncio.run(run_agent(question))
    print(f"\n=== final ===\n{answer}")
