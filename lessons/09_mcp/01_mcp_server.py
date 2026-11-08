"""Lesson 09: a minimal stdio MCP server (mcp 2.x).

Exposes:
  - tools:     search_docs, remember, recall
  - resources: acme://handbook/<doc>       (the corpus files)
  - prompts:   meeting_summary(transcript)

Run standalone (it will wait on stdin: that's the protocol):
    uv run python lessons/09_mcp/01_mcp_server.py

It's meant to be spawned as a subprocess by 02_mcp_client.py.

In mcp 1.x this class was called FastMCP (`mcp.server.fastmcp`). Version 2
renamed it MCPServer; the decorators kept their arguments.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer


HERE = Path(__file__).resolve().parent

# Reuse the TF-IDF index from Lesson 03.
rag_path = HERE.parents[0] / "03a_retrieval_basics" / "02_tfidf_search.py"
spec = importlib.util.spec_from_file_location("_tfidf", rag_path)
assert spec and spec.loader
_tfidf = importlib.util.module_from_spec(spec)
sys.modules["_tfidf"] = _tfidf  # Python 3.13: @dataclass needs the module registered before exec.
spec.loader.exec_module(_tfidf)

CORPUS_DIR = HERE.parents[1] / "data" / "corpus"
MEMORY_PATH = HERE / "mcp_memory.json"
# On stdio, stdout IS the protocol channel: any stray print corrupts it.
# build_index() prints a status line, so send that to stderr instead.
with contextlib.redirect_stdout(sys.stderr):
    INDEX = _tfidf.build_index()


def _load_mem() -> dict:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_mem(d: dict) -> None:
    MEMORY_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")


# --- Server setup ---------------------------------------------------------


mcp = MCPServer("acme-handbook", version="1.0.0")


@mcp.tool()
def search_docs(query: str, k: int = 3) -> list[dict]:
    """Search the Acme company handbook. Returns ranked chunks."""
    return [
        {"id": f"{c.doc}#{c.idx}", "text": c.text, "score": round(s, 3)}
        for c, s in INDEX.search(query, k=k)
    ]


@mcp.tool()
def remember(key: str, value: str) -> dict:
    """Persist a fact to long term memory. Returns the stored pair."""
    store = _load_mem()
    store[key] = value
    _save_mem(store)
    return {"stored": {key: value}}


@mcp.tool()
def recall(key: str) -> dict:
    """Look up a previously remembered fact."""
    store = _load_mem()
    return {"key": key, "value": store.get(key)}


# --- Resources: expose every corpus doc as a URI -------------------------


@mcp.resource("acme://handbook/{name}")
def handbook_doc(name: str) -> str:
    """Read a specific handbook doc by filename, e.g. 'vacation_policy.md'."""
    # The SDK already rejects path traversal values such as "../x" in
    # template variables; we still check, since the server owns its files.
    target = (CORPUS_DIR / name).resolve()
    if not target.is_relative_to(CORPUS_DIR.resolve()) or not target.is_file():
        return f"(no such doc: {name})"
    return target.read_text(encoding="utf-8")


# --- Prompts: a reusable parameterized template --------------------------


@mcp.prompt()
def meeting_summary(transcript: str) -> str:
    """Produce a prompt that summarizes a meeting transcript."""
    return (
        "You are an executive assistant. Summarize the following meeting "
        "transcript in 5 bullets. Capture: decisions made, owners assigned, "
        "open questions.\n\n"
        f"---\n{transcript}\n---"
    )


if __name__ == "__main__":
    # stdio transport: read JSON-RPC from stdin, write to stdout.
    mcp.run(transport="stdio")
