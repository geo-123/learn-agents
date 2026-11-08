"""Lesson 09: MCP client that drives the agent loop over RPC.

Spawns 01_mcp_server.py as a subprocess, lists its tools, translates
them into Anthropic's tool format, and runs the agent loop. Each
tool_use the model emits becomes an MCP call to the subprocess.

This is the same agent loop as Lesson 01: the dispatch just goes over
the protocol instead of calling local Python functions.

mcp 2.x adds `Client`, a high level wrapper: give it StdioServerParameters
and it spawns the server, runs the handshake and closes the subprocess on
exit. (The 1.x pair `stdio_client` plus `ClientSession` still works.)
Result fields are snake_case in 2.x: `input_schema`, `is_error`.

Run:
    uv run python lessons/09_mcp/02_mcp_client.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.types import TextContent, Tool

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parents[1]))
from src._llm import chat


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "01_mcp_server.py")],
)


def to_anthropic_tools(mcp_tools: list[Tool]) -> list[dict]:
    """Translate MCP tool descriptors to Anthropic's tool schema."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.input_schema,
        }
        for t in mcp_tools
    ]


async def call_mcp_tool(client: Client, name: str, args: dict) -> tuple[str, bool]:
    """Invoke a tool on the MCP server. Returns (text, is_error)."""
    result = await client.call_tool(name, args)
    # MCP returns a CallToolResult; we flatten its content blocks to text.
    # `content` is a union (text, image, audio, resource link, embedded
    # resource), so narrow by type before reading `.text`.
    pieces = []
    for block in result.content:
        if isinstance(block, TextContent):
            pieces.append(block.text)
        else:
            pieces.append(json.dumps(block.model_dump(mode="json", by_alias=True)))
    return "\n".join(pieces), result.is_error


SYSTEM = (
    "You are a helpful assistant with access to the Acme handbook and "
    "a persistent memory. Use search_docs for handbook questions. Use "
    "remember/recall for facts about the user. Cite which docs you used."
)


async def run_agent(question: str, *, max_steps: int = 6) -> str:
    # Entering the Client spawns the server and runs the handshake;
    # leaving it shuts the subprocess down.
    async with Client(SERVER_PARAMS) as client:
        tool_list = await client.list_tools()
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
                out, is_error = await call_mcp_tool(client, tu["name"], tu["input"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": out,
                    # An MCP tool error becomes a tool error the model can see.
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})

    return "(max steps)"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is Acme's vacation policy?"
    answer = asyncio.run(run_agent(question))
    print(f"\n=== final ===\n{answer}")
