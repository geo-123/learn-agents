"""Lesson 04b: the memory tool, a file style memory the model manages.

Lesson 04 gave the agent remember/recall tools that you designed. The
memory tool (`memory_20250818`) is the same idea with a shape the model is
trained on: a directory `/memories` it can view, create, edit, rename and
delete files in. The API defines the tool; YOU run every command, so the
storage is yours. Here it is a real folder next to this script,
`memories/`, and nothing outside it can be touched.

Run it twice. The first run tells the agent a few facts; the second, in a
fresh conversation, asks for them back. The agent finds them by reading
its own files. Watch the commands it sends: it starts with
`view /memories` on its own (the API adds that habit to the system prompt).

`--selftest` runs the path guard against hostile paths, offline, with no
API call: every one of them must be refused.

The SDK ships helpers for this (`BetaAbstractMemoryTool` to subclass,
`BetaLocalFilesystemMemoryTool` ready made). This script writes the
backend by hand so you can see every command.

Model: the course default. The memory tool runs on every current model.

Run:
    uv run python lessons/04b_long_running_context/03_memory_tool.py "Our team standup is at 9:15 and Priya owns the release checklist."
    uv run python lessons/04b_long_running_context/03_memory_tool.py "Who owns the release checklist, and when is standup?"
    uv run python lessons/04b_long_running_context/03_memory_tool.py --selftest
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

ROOT = (Path(__file__).resolve().parent / "memories").resolve()
TOOLS = [{"type": "memory_20250818", "name": "memory"}]


class MemoryError_(Exception):
    """A command that must be refused; its message goes back to the model."""


def resolve(virtual: str) -> Path:
    """Map '/memories/...' onto ROOT. Refuse anything that would land outside it."""
    if "%" in virtual or "\\" in virtual:
        raise MemoryError_(f"Path {virtual} is not allowed")
    if virtual != "/memories" and not virtual.startswith("/memories/"):
        raise MemoryError_(f"Path must start with /memories, got: {virtual}")
    real = (ROOT / virtual.removeprefix("/memories").lstrip("/")).resolve()
    if real != ROOT and not real.is_relative_to(ROOT):
        raise MemoryError_(f"Path {virtual} would escape /memories")
    return real


def run_command(cmd: dict[str, Any]) -> str:
    kind = cmd.get("command")
    if kind == "rename":
        old, new = resolve(cmd["old_path"]), resolve(cmd["new_path"])
        if old == ROOT:
            raise MemoryError_("The /memories directory itself cannot be renamed")
        if not old.exists():
            raise MemoryError_(f"Error: The path {cmd['old_path']} does not exist")
        if new.exists():
            raise MemoryError_(f"Error: The destination {cmd['new_path']} already exists")
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return f"Successfully renamed {cmd['old_path']} to {cmd['new_path']}"

    path, real = cmd["path"], resolve(cmd["path"])
    if kind == "view":
        if real.is_dir():
            items = [f"{p.stat().st_size}\t/memories/{p.relative_to(ROOT).as_posix()}"
                     for p in sorted(real.rglob("*")) if not p.name.startswith(".")]
            return (f"Here're the files and directories up to 2 levels deep in {path}, "
                    "excluding hidden items and node_modules:\n" + "\n".join([f"0\t{path}", *items]))
        if real.is_file():
            lines = real.read_text(encoding="utf-8").split("\n")
            numbered = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, start=1))
            return f"Here's the content of {path} with line numbers:\n{numbered}"
        raise MemoryError_(f"The path {path} does not exist. Please provide a valid path.")
    if kind == "create":
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(cmd["file_text"], encoding="utf-8")  # overwrite is a valid choice
        return f"File created successfully at: {path}"
    if not real.is_file() and kind in ("str_replace", "insert"):
        raise MemoryError_(f"Error: The path {path} does not exist")
    if kind == "str_replace":
        text = real.read_text(encoding="utf-8")
        if text.count(cmd["old_str"]) != 1:
            raise MemoryError_(f"No replacement was performed, old_str `{cmd['old_str']}` must appear exactly once in {path}.")
        real.write_text(text.replace(cmd["old_str"], cmd.get("new_str", "")), encoding="utf-8")
        return "The memory file has been edited."
    if kind == "insert":
        lines = real.read_text(encoding="utf-8").split("\n")
        at = int(cmd["insert_line"])
        if not 0 <= at <= len(lines):
            raise MemoryError_(f"Error: Invalid `insert_line` parameter: {at}. It should be within the range of lines of the file: [0, {len(lines)}]")
        lines.insert(at, cmd["insert_text"].rstrip("\n"))
        real.write_text("\n".join(lines), encoding="utf-8")
        return f"The file {path} has been edited."
    if kind == "delete":
        if real == ROOT:
            raise MemoryError_("The /memories directory itself cannot be deleted")
        if not real.exists():
            raise MemoryError_(f"Error: The path {path} does not exist")
        if real.is_dir():
            shutil.rmtree(real)
        else:
            real.unlink()
        return f"Successfully deleted {path}"
    raise MemoryError_(f"Error: unknown command {kind}")


def handle(tool_use: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use["id"]}
    try:
        result["content"] = run_command(tool_use["input"])
    except (MemoryError_, KeyError, OSError) as e:
        result["content"], result["is_error"] = str(e), True
    return result


def run_agent(message: str) -> str:
    ROOT.mkdir(exist_ok=True)
    messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
    for step in range(1, 9):
        response = chat(messages, tools=TOOLS, effort="low")
        messages.append({"role": "assistant", "content": response.content})
        for tu in response.tool_uses:
            args = {k: v for k, v in tu["input"].items() if k != "command"}
            print(f"  step {step}: memory.{tu['input'].get('command')}({args})")
        if response.stop_reason != "tool_use":
            return response.text
        messages.append({"role": "user", "content": [handle(tu) for tu in response.tool_uses]})
    return "[stopped after 8 steps]"


def selftest() -> None:
    hostile = ["/memories/../../.env", "/memories/..\\..\\x", "/memories/%2e%2e/secret",
               "/etc/passwd", "/memoriesX/notes.md", "/memories/a/../../../x"]
    for p in hostile:
        try:
            resolve(p)
            print(f"  NOT REFUSED: {p}")
        except MemoryError_ as e:
            print(f"  refused {p!r}: {e}")
    print(f"  allowed '/memories/notes.md' -> {resolve('/memories/notes.md')}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        msg = " ".join(sys.argv[1:]) or "What do you remember about our team?"
        print(f"USER: {msg}\n")
        print(f"\nFINAL:\n{run_agent(msg)}")
        print(f"\n[memory files live in {ROOT}]")
