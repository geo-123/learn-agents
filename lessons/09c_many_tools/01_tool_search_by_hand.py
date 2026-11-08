"""Lesson 09c: tool search built by hand, before the API does it for you.

The same 40 HR tools as `03_tool_search.py` (the table and the
definitions are imported from that file, so both scripts see exactly the
same catalog). The agent starts with four tools: the three most used HR
tools and one tool of our own, `find_tools(query)`.

`find_tools` ranks every tool's name, description and argument
descriptions with the course's own BM25 from Lesson 03c. The class is
imported from `lessons/03c_agentic_rag/01_hybrid_bm25.py` (which itself
takes its tokenizer from Lesson 03a), not copied: a tool catalog is just
one more small corpus, and a tool definition is just one more chunk.

The loop is the Lesson 01 dispatch loop with one extra step: when the
model calls `find_tools`, the loop adds the full definitions of the
matches to `tools`, so the model can call them on the next request.

Things to notice in the output, per turn:
  * how many tool definitions were sent and how many characters they
    take, against the 40 tools all loaded;
  * the query the model wrote and the tools it found;
  * "tools changed": the request after a search has a different `tools`
    list. That is the price of doing it by hand, see below.

The cost of doing it by hand:
  * Tools come first in the prompt (Lesson 02b). Changing the list
    changes the prefix of everything after it, system prompt and history
    included, so the prompt cache starts cold after every search.
  * On current models a thinking block is bound to everything sent
    before it: system prompt, tools, earlier messages. After the tools
    change, every earlier thinking block no longer matches and the API
    rejects it. This loop keeps its history append only and strips the
    thinking from the turns before the last change on the copy it sends
    (`strip_thinking` from `src/_llm.py`). The request is accepted, but
    the model loses its earlier reasoning.
  * BM25 is lexical: "laptop" does not match "laptops", "PTO" does not
    match "vacation". The model has to guess the catalog's words.
  * Every search costs one extra model turn.

The API's tool search (`03_tool_search.py`) keeps the full list in every
request with most tools marked `defer_loading` and loads the found
definitions inline in the conversation. The prefix never changes, the
cache survives and the thinking blocks stay valid.

Run:
    uv run python lessons/09c_many_tools/01_tool_search_by_hand.py
    uv run python lessons/09c_many_tools/01_tool_search_by_hand.py "Please cancel my equipment request EQ-77"
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat, strip_thinking

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> ModuleType:
    """Load a lesson file by path: its numeric prefix blocks a plain `import`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, so @dataclass can resolve its module
    spec.loader.exec_module(module)
    return module


_search = _load("_tool_search_api", HERE / "03_tool_search.py")
_bm25 = _load("_hybrid_bm25", HERE.parents[0] / "03c_agentic_rag" / "01_hybrid_bm25.py")

ALWAYS_LOADED: set[str] = _search.ALWAYS_LOADED  # the 3 most used HR tools


def build_catalog() -> dict[str, dict[str, Any]]:
    """Every HR tool definition by name, exactly as 03_tool_search.py builds it, minus the API flags."""
    tools = _search.build_tools("tool_search_tool_bm25_20251119")[1:]  # drop the API's search tool
    return {t["name"]: {k: v for k, v in t.items() if k != "defer_loading"} for t in tools}


CATALOG = build_catalog()


def _search_text(tool: dict[str, Any]) -> str:
    """What the index sees: name, description, argument names and their descriptions."""
    props = tool["input_schema"].get("properties", {})
    args = " ".join(f"{name} {spec.get('description', '')}" for name, spec in props.items())
    return f"{tool['name']} {tool['description']} {args}"


# One chunk per tool. `doc` carries the tool name so a hit maps straight back to its definition.
INDEX = _bm25.BM25Index([_bm25.Chunk(name, i, _search_text(t)) for i, (name, t) in enumerate(CATALOG.items())])


def find_tools(query: str, k: int = 5) -> list[dict[str, str]]:
    """The top k tools for a query: name and description only, ranked by BM25."""
    return [{"name": chunk.doc, "description": CATALOG[chunk.doc]["description"]}
            for chunk, _score in INDEX.search(query, k=k)]


FIND_TOOLS = {
    "name": "find_tools",
    "description": (
        "Search the HR tool catalog. Call this when none of your loaded tools fits the request. "
        "Write a short query with the HR area and the action, for example 'cancel equipment' or "
        "'submit expense'. The matching tools are loaded and you can call them on your next turn."
    ),
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


def run_hr_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A stand in for the real HR system: echo what would have been done."""
    return {"ok": True, "tool": name, "input": args}


def size(tools: list[dict[str, Any]]) -> int:
    """Definition size in characters, a proxy for prompt tokens that needs no API call."""
    return len(json.dumps(tools))


def main(request: str) -> None:
    tools: list[dict[str, Any]] = [FIND_TOOLS] + [CATALOG[n] for n in sorted(ALWAYS_LOADED)]
    print(f"{len(CATALOG)} HR tools in the catalog; all loaded would be {size(list(CATALOG.values()))} characters")

    system = ("You are an HR assistant. Your loaded tools cover only a few HR areas. Use find_tools to load "
              "tools for other areas (leave, payroll, benefits, expense, onboarding, training, equipment, review).")
    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]  # append only, never edited
    tools_changed_at = 0  # messages before this index were produced with an older tools list
    for step in range(1, 7):
        sent = strip_thinking(messages[:tools_changed_at]) + messages[tools_changed_at:]
        dropped = sum(1 for m in messages[:tools_changed_at] if m["role"] == "assistant"
                      for b in m["content"] if b.get("type") in ("thinking", "redacted_thinking"))
        print(f"\nstep {step}: {len(tools)} tools sent, {size(tools)} characters"
              + (f", {dropped} earlier thinking blocks stripped" if dropped else ""))
        response = chat(sent, system=system, tools=tools, effort="low")
        messages.append({"role": "assistant", "content": response.content})
        print(f"step {step}: input tokens {response.input_tokens}")
        if response.stop_reason != "tool_use":
            print(f"\nanswer: {response.text.strip()}")
            return

        results: list[dict[str, Any]] = []
        added: list[str] = []
        loaded = {t["name"] for t in tools}
        for tu in response.tool_uses:
            if tu["name"] == "find_tools":
                query = str(tu["input"].get("query", ""))
                hits = find_tools(query)
                print(f"step {step}: find_tools({query!r}) -> {[h['name'] for h in hits]}")
                for hit in hits:
                    if hit["name"] not in loaded:
                        loaded.add(hit["name"])
                        added.append(hit["name"])
                content = json.dumps({"loaded": hits} if hits else {"loaded": [], "hint": "try other words"})
            elif tu["name"] in loaded and tu["name"] in CATALOG:
                print(f"step {step}: CALL {tu['name']}({tu['input']})")
                content = json.dumps(run_hr_tool(tu["name"], tu["input"]))
            else:
                content = json.dumps({"error": f"{tu['name']!r} is not loaded. Call find_tools first."})
            results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": content})
        messages.append({"role": "user", "content": results})

        if added:
            # Append at the end so the order stays deterministic; any change still changes the prefix.
            tools = tools + [CATALOG[n] for n in added]
            tools_changed_at = len(messages)
            print(f"step {step}: tools changed (+{len(added)}): the next request starts a new cache prefix")
    print("\n[stop] step limit reached")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "Please cancel my laptop request EQ-77, I got one from my old team.")
