"""Lesson 08b: the bash tool behind an allowlist and an approval prompt.

The bash tool is Anthropic defined and schema less, like the text editor.
The model sends a command string; what happens to it is entirely up to your
harness. This one never hands the string to a shell. It:

  1. rejects shell operators (; && | > ` $( and friends),
  2. splits the string into argv and looks the program up in a policy table
     (Lesson 06): read only programs run, python asks you first, anything
     else is blocked,
  3. runs the program with subprocess, no shell, inside sandbox/, with a
     timeout and a short output cap.

`ls` and `cat` are implemented in Python so the script behaves the same on
Windows and POSIX. The self check at the start shows blocked commands without
any API call.

Run:
    uv run python lessons/08b_coding_agents/02_bash_with_policy.py
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

SANDBOX = Path(__file__).resolve().parent / "sandbox"
BASH = {"type": "bash_20250124", "name": "bash"}  # no input_schema
TIMEOUT_S = 10
MAX_OUTPUT = 4_000

Policy = Literal["auto", "confirm", "block"]
POLICY: dict[str, Policy] = {
    "ls": "auto",        # read only, implemented below
    "cat": "auto",       # read only, implemented below
    "python": "confirm", # can do anything inside the sandbox: ask first
}
OPERATORS = (";", "&", "|", ">", "<", "`", "$(", "\n")


def make_sandbox() -> None:
    shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX.mkdir(parents=True)
    (SANDBOX / "inventory.csv").write_text("item,count\nbolts,40\nnuts,25\nwashers,60\n",
                                           encoding="utf-8")


def inside(raw: str) -> Path:
    target = (SANDBOX / raw).resolve()
    if not target.is_relative_to(SANDBOX.resolve()):
        raise PermissionError(f"path escapes the sandbox: {raw!r}")
    return target


def check(command: str) -> tuple[Policy, list[str], str]:
    """Decide what to do with a command: (policy, argv, reason)."""
    for op in OPERATORS:
        if op in command:
            return "block", [], f"shell operator {op!r} is not allowed"
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return "block", [], f"cannot parse: {e}"
    if not argv:
        return "block", [], "empty command"
    policy = POLICY.get(argv[0], "block")
    reason = "" if policy != "block" else f"program {argv[0]!r} is not on the allowlist"
    return policy, argv, reason


def execute(argv: list[str]) -> str:
    if argv[0] == "ls":
        target = inside(argv[1] if len(argv) > 1 else ".")
        return "\n".join(sorted(p.name for p in target.iterdir()))
    if argv[0] == "cat":
        return "\n".join(inside(a).read_text(encoding="utf-8") for a in argv[1:])
    # python: our interpreter, no shell, cwd in the sandbox, hard timeout.
    proc = subprocess.run([sys.executable, *argv[1:]], cwd=SANDBOX, capture_output=True,
                          text=True, timeout=TIMEOUT_S)
    return f"{proc.stdout}{proc.stderr}exit code {proc.returncode}"


def dispatch(tool_use: dict) -> dict:
    result = {"type": "tool_result", "tool_use_id": tool_use["id"]}
    args = tool_use["input"]
    if args.get("restart"):
        result["content"] = "shell restarted (this harness keeps no shell state)"
        return result
    policy, argv, reason = check(args.get("command", ""))
    if policy == "confirm":
        answer = input(f"\n>>> run {argv}? [y/n] ").strip().lower()
        if answer not in ("y", "yes"):
            policy, reason = "block", "the user declined this command"
    if policy == "block":
        print(f"  [blocked] {reason}")
        result.update(content=f"Blocked: {reason}", is_error=True)
        return result
    try:
        result["content"] = execute(argv)[:MAX_OUTPUT] or "(no output)"
    except (OSError, PermissionError, subprocess.TimeoutExpired) as e:
        result.update(content=f"Error: {type(e).__name__}: {e}", is_error=True)
    return result


SYSTEM = (
    "You work in a sandboxed folder through the bash tool. Allowed programs: "
    "ls, cat, python (python needs the user's approval). Shell operators such "
    "as pipes, redirects and ; are rejected: run one program per call. "
    "Answer in two sentences when done."
)


def run_agent(task: str, *, max_steps: int = 8) -> str:
    messages: list[dict] = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=[BASH])
        messages.append({"role": "assistant", "content": response.content})
        for tu in response.tool_uses:
            print(f"  step {step}: bash {tu['input']}")
        if response.stop_reason != "tool_use":
            return response.text
        messages.append({"role": "user", "content": [dispatch(tu) for tu in response.tool_uses]})
    return "(max steps reached)"


if __name__ == "__main__":
    make_sandbox()
    print(f"sandbox: {SANDBOX}  (recreated each run; git ignore it)")

    print("\n--- policy self check (no API call) ---")
    for cmd in ["ls", "cat inventory.csv", "python -c \"print(1)\"",
                "curl https://example.com", "rm -rf .", "cat inventory.csv | head -2",
                "python -c \"print(1)\"; del *.csv"]:
        policy, _, reason = check(cmd)
        print(f"  {policy:<8} {cmd!r}  {reason}")

    print("\n--- agent ---")
    answer = run_agent("How many items are in inventory.csv in total? "
                       "Compute the sum with a short python one liner.")
    print(f"\nagent says: {answer}")
