"""Lesson 01c: build web fetch and code execution yourself, as client tools.

Before renting Anthropic's server tools, build the two capabilities with
nothing but the Lesson 01 dispatch loop. Each is an ordinary client tool:
a pydantic model for its input, a function, a description. The model asks,
your code runs it, a tool_result goes back.

    fetch_url(url)   an HTTP GET with urllib from the standard library:
                     a domain allowlist (checked again after redirects),
                     a timeout, a byte cap, HTML turned into plain text by
                     a crude tag strip.
    run_python(code) the model's code in a fresh temporary folder, run by
                     `sys.executable -I` in a subprocess with a timeout, an
                     output cap and an almost empty environment. Asks you
                     before every run (Lesson 06), unless you pass --yes.

Every failure (blocked domain, timeout, HTTP error, declined approval)
comes back as an `is_error` tool_result, so the model can read it and
recover, exactly like Lesson 01.

What this shows: a tool is just your code behind a schema. The model part
is trivial; the hard parts are the ones these two functions do badly:
isolation, search quality, limits. Read the docstrings of both tools for
what they do NOT do. The next scripts rent the real versions.

Run:
    uv run python lessons/01c_server_tools/01_build_your_own.py
    uv run python lessons/01c_server_tools/01_build_your_own.py --yes     # no approval prompt
    uv run python lessons/01c_server_tools/01_build_your_own.py --check   # tools only, no API call
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat

POLICY = (Path(__file__).resolve().parents[2] / "data" / "corpus" / "expense_policy.md").read_text(encoding="utf-8")

# --- Limits: every number here is a decision the server tools make for you --

ALLOWED_DOMAINS = ("python.org",)   # subdomains included, like allowed_domains
FETCH_TIMEOUT_S = 10
FETCH_MAX_BYTES = 500_000           # stop reading the body after this
FETCH_MAX_CHARS = 6_000             # what reaches the model (tokens cost money)
RUN_TIMEOUT_S = 10
RUN_MAX_OUTPUT = 4_000

APPROVE_ALL = False                 # set by --yes


# --- fetch_url --------------------------------------------------------------


class FetchUrlArgs(BaseModel):
    url: str = Field(description="An http or https URL on an allowed domain.")


def _allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def _html_to_text(page: str) -> str:
    """A crude tag strip. Real extractors keep headings, tables and links."""
    page = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(page)).strip()


def fetch_url(args: FetchUrlArgs) -> dict[str, Any]:
    """GET one page and return it as plain text.

    Guards: http or https only, the host must be on ALLOWED_DOMAINS (and so
    must the host we land on after redirects), a timeout, a cap on bytes
    read and on characters returned, text content types only.

    What it does NOT do, and a real web search tool does:
      * Find pages. It needs a URL; it has no index of the web to search.
      * Rank. A search engine decides which of a million pages answers the
        question; that ranking is most of the value.
      * Stay fresh. An index is recrawled continuously; this sees one page.
      * Read pages properly: no JavaScript, no PDF, no layout, no caching,
        no robots.txt, no rate limits per site.
    Also: the allowlist is the only thing between the model and any URL a
    prompt injection asks for. Lesson 08 explains why that matters.
    """
    parsed = urlparse(args.url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http and https URLs are allowed, got {parsed.scheme!r}")
    if not _allowed(parsed.hostname or ""):
        raise PermissionError(f"domain {parsed.hostname!r} is not on the allowlist {ALLOWED_DOMAINS}")

    request = urllib.request.Request(args.url, headers={"User-Agent": "learn_agents_lesson_01c"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        final_url = response.url
        final_host = urlparse(final_url).hostname or ""
        if not _allowed(final_host):  # a redirect can leave the allowlist
            raise PermissionError(f"redirected to {final_host!r}, which is not on the allowlist")
        content_type = response.headers.get_content_type()
        if not content_type.startswith("text/"):
            raise ValueError(f"unsupported content type {content_type!r}")
        body = response.read(FETCH_MAX_BYTES + 1)
        charset = response.headers.get_content_charset() or "utf-8"

    text = body[:FETCH_MAX_BYTES].decode(charset, errors="replace")
    if content_type == "text/html":
        text = _html_to_text(text)
    truncated = len(body) > FETCH_MAX_BYTES or len(text) > FETCH_MAX_CHARS
    return {"url": final_url, "truncated": truncated, "text": text[:FETCH_MAX_CHARS]}


# --- run_python -------------------------------------------------------------


class RunPythonArgs(BaseModel):
    code: str = Field(description="A complete Python 3 script. Print the results you need.")


def _minimal_env() -> dict[str, str]:
    """No inherited secrets: no API keys, no tokens, no proxy credentials.

    Windows needs SYSTEMROOT for Python to start (it seeds the random module
    from the system), so that one variable is passed through.
    """
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _approved(code: str) -> bool:
    if APPROVE_ALL:
        return True
    print("\n  >>> the model wants to run this Python:")
    for line in code.splitlines():
        print(f"      | {line}")
    try:
        return input("  >>> run it? [y/n] ").strip().lower() in ("y", "yes")
    except EOFError:  # no terminal to ask: the safe answer is no
        return False


def execute_python(code: str) -> dict[str, Any]:
    """Run code in a fresh temporary folder. No approval: see run_python.

    Guards: a new empty folder per call (deleted afterwards), our own
    interpreter in isolated mode (-I ignores PYTHON* variables, the user
    site folder and the current folder on sys.path), a minimal environment,
    no shell, a timeout, capped output.

    What it does NOT isolate, and a real sandbox does:
      * Filesystem. The code runs as you. It can read your home folder,
        your SSH keys, this repository, and delete anything you can delete.
      * Network. It can open sockets and send whatever it read anywhere.
      * CPU, memory, processes. The timeout stops the main process; a
        child it spawned, a memory bomb or a full disk are not stopped.
    A real sandbox is a container or a virtual machine: its own filesystem,
    no network, quotas, thrown away afterwards. Building and running that
    is exactly what the code execution server tool does for you.
    """
    with tempfile.TemporaryDirectory(prefix="run_python_") as folder:
        script = Path(folder) / "main.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=folder, env=_minimal_env(),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=RUN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"execution_time_exceeded: killed after {RUN_TIMEOUT_S} s") from None
    # The same three fields the code execution server tool returns.
    return {
        "return_code": proc.returncode,
        "stdout": proc.stdout[:RUN_MAX_OUTPUT],
        "stderr": proc.stderr[:RUN_MAX_OUTPUT],
    }


def run_python(args: RunPythonArgs) -> dict[str, Any]:
    """The tool: ask a human, then execute_python. Declining is an error result."""
    if not _approved(args.code):
        raise PermissionError("the user declined to run this code")
    return execute_python(args.code)


# --- Registry and dispatcher: Lesson 01, unchanged --------------------------


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], Any]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": self.args_model.model_json_schema()}


REGISTRY: dict[str, Tool] = {t.name: t for t in [
    Tool("fetch_url",
         f"Fetch one web page and return its text (truncated to {FETCH_MAX_CHARS} characters). "
         f"Only these domains and their subdomains are allowed: {', '.join(ALLOWED_DOMAINS)}. "
         "This is not a search engine: you must already know the URL.",
         FetchUrlArgs, fetch_url),
    Tool("run_python",
         "Run a Python 3 script and return its return_code, stdout and stderr. Each call starts "
         f"in a new empty folder with no state from earlier calls. Standard library only. "
         f"Killed after {RUN_TIMEOUT_S} seconds. Print every result you need.",
         RunPythonArgs, run_python),
]}


def dispatch(tool_use: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call. Always returns a well shaped tool_result block."""
    tool = REGISTRY.get(tool_use["name"])
    if tool is None:
        return _err(tool_use["id"], f"Unknown tool {tool_use['name']!r}. Available: {list(REGISTRY)}.")
    try:
        args = tool.args_model.model_validate(tool_use.get("input") or {})
    except ValidationError as e:
        return _err(tool_use["id"], f"Invalid args for {tool.name!r}: {e.errors()}")
    try:
        result = tool.fn(args)
    except Exception as e:  # noqa: BLE001: blocked, timed out, declined: the model reads it all the same
        return _err(tool_use["id"], f"{type(e).__name__}: {e}")
    return {"type": "tool_result", "tool_use_id": tool_use["id"], "content": json.dumps(result)}


def _err(use_id: str, message: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": use_id, "content": message, "is_error": True}


# --- The agent loop: Lesson 01, unchanged ------------------------------------

SYSTEM = (
    "You have two tools: fetch_url reads a web page on an allowed domain, and run_python "
    "runs a Python script. Compute numbers with run_python instead of by hand. "
    "If a tool returns an error, read it and adapt. Answer in at most three sentences."
)


def run_agent(task: str, *, max_steps: int = 6) -> str:
    tools = [t.schema() for t in REGISTRY.values()]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        response = chat(messages, system=SYSTEM, tools=tools, effort="low")
        messages.append({"role": "assistant", "content": response.content})
        print(f"\n--- step {step}  stop_reason={response.stop_reason}")
        if response.stop_reason != "tool_use":
            return response.text
        results = []
        for tu in response.tool_uses:
            print(f"  runs here : {tu['name']}({str(tu['input'])[:120]})")
            result = dispatch(tu)
            flag = "  is_error" if result.get("is_error") else ""
            print(f"  result{flag}: {result['content'][:200]}")
            results.append(result)
        messages.append({"role": "user", "content": results})
    return "(max steps reached)"


# --- Two questions ------------------------------------------------------------

# The same trip 03_code_execution.py sends to the server tool.
HOTELS = [("Tokyo", 3, 420), ("Berlin", 2, 280)]
MEALS = [95, 70, 88, 60, 81]

EXPENSE_TASK = f"""Here is our expense policy:

{POLICY}

A trip: hotels {HOTELS} as (city, nights, nightly rate in USD), meals per day {MEALS} in USD.
Use run_python to apply the hotel and meal caps and print claimed, reimbursable and over cap
totals. Then summarize."""

FETCH_TASK = ("According to https://www.python.org/ what is the latest Python release? "
              "Also, what does https://example.com/ say?")


def self_check() -> None:
    """Exercise both tools directly, without the model and without an API key."""
    calls = [
        ("run_python", {"code": "print(sum(min(m, 80) for m in [95, 70, 88, 60, 81]))"}),
        ("run_python", {"code": "while True:\n    pass"}),
        ("fetch_url", {"url": "https://example.com/"}),
        ("fetch_url", {"url": "file:///etc/passwd"}),
        ("fetch_url", {"url": "https://www.python.org/"}),
    ]
    for i, (name, args) in enumerate(calls):
        result = dispatch({"type": "tool_use", "id": f"check_{i}", "name": name, "input": args})
        flag = "is_error " if result.get("is_error") else ""
        print(f"  {name}({str(args)[:50]})\n      -> {flag}{result['content'][:160]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web fetch and code execution as client tools.")
    parser.add_argument("--yes", action="store_true", help="run model written code without asking")
    parser.add_argument("--check", action="store_true", help="run the tools directly, no API call")
    cli = parser.parse_args()
    APPROVE_ALL = cli.yes or cli.check  # in the self check you wrote the code yourself

    if cli.check:
        print("--- tool self check (no API call) ---")
        self_check()
        sys.exit(0)

    print(f"Model: {DEFAULT_MODEL}")
    print("\n=== 1. Code execution, built by hand ===")
    print(f"\nAnswer:\n{run_agent(EXPENSE_TASK)}")
    print("\n=== 2. Web fetch, built by hand (one allowed URL, one blocked) ===")
    print(f"\nAnswer:\n{run_agent(FETCH_TASK)}")
    print("\nNotice: both tools went through the same dispatcher as get_weather in "
          "Lesson 01. Nothing about them is special except what they guard against.")
