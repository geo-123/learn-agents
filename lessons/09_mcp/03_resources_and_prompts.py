"""Lesson 09 — MCP resources and prompts (not just tools).

Tools are 90% of MCP usage, but the spec has two more primitives.
This file demonstrates both against the server in 01_mcp_server.py:

  - List resources, then read one.
  - List prompts, then invoke `meeting_summary` with arguments.

Run:
    uv run python lessons/09_mcp/03_resources_and_prompts.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "01_mcp_server.py")],
)


SAMPLE_TRANSCRIPT = """\
Maria: We're slipping on the auth migration. Legal needs it done by Q2.
John:  I can take the SSO piece if we de-scope the audit logging.
Priya: The mocks are blocked on the auth provider config — that's mine.
Maria: OK — John picks up SSO, Priya owns the auth provider config, audit
       logging deferred to Q3. Next sync Friday.
"""


async def main() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Resources ---
            print("== resources ==")
            res_list = await session.list_resource_templates()
            for tmpl in res_list.resourceTemplates:
                print(f"  template: {tmpl.uriTemplate}  ({tmpl.description})")

            # Read a concrete resource via the template URI.
            uri = "acme://handbook/vacation_policy.md"
            print(f"\nreading {uri}...")
            content = await session.read_resource(uri)
            for block in content.contents:
                if hasattr(block, "text"):
                    print(block.text[:300] + "..." if len(block.text) > 300 else block.text)

            # --- Prompts ---
            print("\n== prompts ==")
            prompts = await session.list_prompts()
            for p in prompts.prompts:
                print(f"  prompt: {p.name}  ({p.description})")
                if p.arguments:
                    for arg in p.arguments:
                        print(f"    arg: {arg.name}  required={arg.required}")

            # Invoke meeting_summary with a transcript argument.
            print("\ninvoking meeting_summary(transcript=...)...")
            result = await session.get_prompt(
                "meeting_summary",
                arguments={"transcript": SAMPLE_TRANSCRIPT},
            )
            for msg in result.messages:
                content = msg.content
                text = content.text if hasattr(content, "text") else str(content)
                print(f"\n[role={msg.role}]\n{text}")


if __name__ == "__main__":
    asyncio.run(main())
