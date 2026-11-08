"""Lesson 08b: a coding agent that fixes a bug with the text editor tool.

The script writes a tiny project into `sandbox/` next to this file (deleted
and recreated on every run): `stats.py` with a bug and `test_stats.py`.
The agent gets one tool, the Anthropic defined text editor tool. You do not
write its schema; you declare its type and name, the model already knows how
to call it, and YOUR code executes every call.

Three things to notice:
  * Every model supplied path goes through `resolve()`, which confines it to
    the sandbox. The self check at the start shows what it rejects.
  * The loop is the Lesson 00 loop: call, run the tool calls, append, repeat.
  * The harness, not the model, verifies the result: it runs the tests with a
    fixed command and prints the diff of what the agent changed.

Run:
    uv run python lessons/08b_coding_agents/01_text_editor_agent.py
"""
from __future__ import annotations

import difflib
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

SANDBOX = Path(__file__).resolve().parent / "sandbox"
VIRTUAL_ROOT = "/project"  # the root the model sees; mapped onto SANDBOX

TEMPLATE = {
    "stats.py": (
        "def mean(values):\n"
        "    \"\"\"Arithmetic mean of a non empty list.\"\"\"\n"
        "    return sum(values) / (len(values) - 1)\n"
    ),
    "test_stats.py": (
        "from stats import mean\n\n"
        "assert mean([2, 4, 6]) == 4, mean([2, 4, 6])\n"
        "assert mean([5]) == 5, mean([5])\n"
        "print('all tests passed')\n"
    ),
}

# Anthropic defined, schema less: type and name only, never an input_schema.
# max_characters caps what one `view` returns (context discipline).
TEXT_EDITOR = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool",
               "max_characters": 10_000}


def make_sandbox() -> None:
    shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX.mkdir(parents=True)
    for name, text in TEMPLATE.items():
        (SANDBOX / name).write_text(text, encoding="utf-8")


def resolve(raw: str) -> Path:
    """Map a model supplied path onto the sandbox, or raise ValueError."""
    if "%" in raw or "\x00" in raw:
        raise ValueError(f"encoded or binary characters in path: {raw!r}")
    path = unquote(raw).replace("\\", "/")
    if path != VIRTUAL_ROOT and not path.startswith(VIRTUAL_ROOT + "/"):
        raise ValueError(f"path outside {VIRTUAL_ROOT}: {raw!r}")
    target = (SANDBOX / path[len(VIRTUAL_ROOT):].lstrip("/")).resolve()
    if not target.is_relative_to(SANDBOX.resolve()):  # catches .. and symlinks
        raise ValueError(f"path escapes the sandbox: {raw!r}")
    return target


def run_editor(args: dict) -> str:
    """Execute one text editor command. Raises on any problem."""
    command, target = args["command"], resolve(args["path"])
    if command == "view":
        if target.is_dir():
            return "\n".join(sorted(p.name for p in target.iterdir()))
        lines = target.read_text(encoding="utf-8").splitlines()
        start, end = args.get("view_range") or [1, len(lines)]
        end = len(lines) if end == -1 else end
        return "\n".join(f"{n}\t{lines[n - 1]}" for n in range(start, end + 1))
    if command == "create":
        target.write_text(args["file_text"], encoding="utf-8")
        return f"created {args['path']}"
    if command == "str_replace":
        text = target.read_text(encoding="utf-8")
        count = text.count(args["old_str"])
        if count != 1:
            raise ValueError(f"old_str found {count} times; it must match exactly once")
        target.write_text(text.replace(args["old_str"], args.get("new_str", "")), encoding="utf-8")
        return f"edited {args['path']}"
    if command == "insert":
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.insert(args["insert_line"], args["insert_text"].rstrip("\n") + "\n")
        target.write_text("".join(lines), encoding="utf-8")
        return f"inserted into {args['path']}"
    raise ValueError(f"unsupported command {command!r}")


def dispatch(tool_use: dict) -> dict:
    result = {"type": "tool_result", "tool_use_id": tool_use["id"]}
    try:
        result["content"] = run_editor(tool_use["input"])
    except (ValueError, KeyError, OSError) as e:  # the model sees the error and can recover
        result.update(content=f"Error: {e}", is_error=True)
    return result


SYSTEM = (
    f"You are a careful coding agent. The project lives at {VIRTUAL_ROOT}. "
    "Look at the directory first, read only the files you need, make the "
    "smallest edit that fixes the bug, then stop and describe the fix in one "
    "sentence. You cannot run commands; the harness runs the tests after you."
)


def run_agent(task: str, *, max_steps: int = 8) -> str:
    messages: list[dict] = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=[TEXT_EDITOR])
        messages.append({"role": "assistant", "content": response.content})
        for tu in response.tool_uses:
            shown = {k: v for k, v in tu["input"].items() if k != "file_text"}
            print(f"  step {step}: {shown}")
        if response.stop_reason != "tool_use":
            return response.text
        messages.append({"role": "user", "content": [dispatch(tu) for tu in response.tool_uses]})
    return "(max steps reached)"


def run_tests() -> tuple[int, str]:
    """A FIXED command chosen by the harness, with a timeout."""
    proc = subprocess.run([sys.executable, "test_stats.py"], cwd=SANDBOX,
                          capture_output=True, text=True, timeout=30)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


if __name__ == "__main__":
    make_sandbox()
    print(f"sandbox: {SANDBOX}  (recreated each run; git ignore it)")

    print("\n--- path self check (no API call) ---")
    for raw in ["/project/stats.py", "/project/../../.env", "/etc/passwd",
                "/project/%2e%2e/secrets", "C:\\Windows\\win.ini"]:
        try:
            print(f"  allowed  {raw!r} -> {resolve(raw).relative_to(SANDBOX.parent)}")
        except ValueError as e:
            print(f"  REJECTED {e}")

    code, out = run_tests()
    print(f"\n--- tests before: exit {code} ---\n{out.splitlines()[-1] if out else ''}")

    print("\n--- agent ---")
    summary = run_agent("test_stats.py fails. Find and fix the bug in the project.")
    print(f"\nagent says: {summary}")

    code, out = run_tests()
    print(f"\n--- tests after: exit {code} ---\n{out}")

    after = (SANDBOX / "stats.py").read_text(encoding="utf-8")
    diff = difflib.unified_diff(TEMPLATE["stats.py"].splitlines(), after.splitlines(),
                                "a/stats.py", "b/stats.py", lineterm="")
    print("\n--- diff to review ---\n" + "\n".join(diff))
