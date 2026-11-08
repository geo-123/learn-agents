"""Lesson 09: MCP resources and prompts (not just tools).

Tools are 90% of MCP usage, but the spec has two more primitives.
This file demonstrates both against the server in 01_mcp_server.py:

  - List resource templates, then read one resource.
  - List prompts, then invoke `meeting_summary` with arguments.

Uses the mcp 2.x `Client` (see 02_mcp_client.py). In 2.x, URIs are plain
strings and result fields are snake_case (`uri_template`, `mime_type`).

Run:
    uv run python lessons/09_mcp/03_resources_and_prompts.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.types import TextContent, TextResourceContents

HERE = Path(__file__).resolve().parent


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "01_mcp_server.py")],
)


SAMPLE_TRANSCRIPT = """\
Maria: We're slipping on the auth migration. Legal needs it done by Q2.
John:  I can take the SSO piece if we descope the audit logging.
Priya: The mocks are blocked on the auth provider config: that's mine.
Maria: OK: John picks up SSO, Priya owns the auth provider config, audit
       logging deferred to Q3. Next sync Friday.
"""


async def main() -> None:
    async with Client(SERVER_PARAMS) as client:
        # --- Resources ---
        print("== resources ==")
        res_list = await client.list_resource_templates()
        for tmpl in res_list.resource_templates:
            print(f"  template: {tmpl.uri_template}  ({tmpl.description})")

        # Read a concrete resource via the template URI.
        uri = "acme://handbook/vacation_policy.md"
        print(f"\nreading {uri}...")
        content = await client.read_resource(uri)
        # Contents are text or blob (base64 binary); only text has `.text`.
        for block in content.contents:
            if isinstance(block, TextResourceContents):
                text = block.text
                print(text[:300] + "..." if len(text) > 300 else text)
            else:
                print(f"(binary content, {block.mime_type})")

        # --- Prompts ---
        print("\n== prompts ==")
        prompts = await client.list_prompts()
        for p in prompts.prompts:
            print(f"  prompt: {p.name}  ({p.description})")
            if p.arguments:
                for arg in p.arguments:
                    print(f"    arg: {arg.name}  required={arg.required}")

        # Invoke meeting_summary with a transcript argument.
        print("\ninvoking meeting_summary(transcript=...)...")
        result = await client.get_prompt(
            "meeting_summary",
            arguments={"transcript": SAMPLE_TRANSCRIPT},
        )
        for msg in result.messages:
            # Message content is a union too: narrow before reading `.text`.
            block = msg.content
            text = block.text if isinstance(block, TextContent) else block.model_dump_json()
            print(f"\n[role={msg.role}]\n{text}")


if __name__ == "__main__":
    asyncio.run(main())
