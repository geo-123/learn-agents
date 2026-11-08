"""Lesson 09c: skills built by hand: progressive disclosure in three functions.

A skill is a folder with a `SKILL.md`: a short header (name and a one
line description) and the full instructions below it. This script writes
three tiny skills into `lessons/09c_many_tools/skills/` (recreated on
every run, ignored by git), then runs an agent that sees them in two
levels:

  1. The system prompt lists only each skill's name and description.
     Three skills cost three lines, whatever the length of the manuals.
  2. A `load_skill(name)` tool returns the full `SKILL.md` when the model
     decides a request needs it. The instructions arrive as a tool
     result, at the end of the history.

That second point is why this one is cheap to build by hand, unlike tool
search (`01_tool_search_by_hand.py`): the system prompt and the tools
never change, so the prompt cache and earlier thinking blocks stay valid.
The only thing that grows is the history, append only.

`load_skill` takes a name the model wrote, so it is untrusted input that
ends up in a file path. It accepts only names from the index it built at
start, and still checks that the resolved path stays inside the skills
folder: `../secrets` is refused, not read.

Things to notice in the output:
  * the size of the skill list in the system prompt against the full text;
  * which skill the model loads, on which step, and how big it was;
  * a request no skill covers: the model should answer without loading one.

Run:
    uv run python lessons/09c_many_tools/02_skills_by_hand.py
    uv run python lessons/09c_many_tools/02_skills_by_hand.py "Draft an incident report: the payroll export failed on Friday."
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import chat

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SKILLS = {
    "expense-audit": """---
name: expense-audit
description: Audit expense reports against the Acme travel and meal caps. Use when asked to check, flag or approve expenses.
---
# Expense audit

1. Hotel cap: $300 a night, $400 a night in NYC, London and Tokyo.
2. Meals cap: $80 a day. A trip of N nights has N + 1 days.
3. Compare totals per night and per day, never the trip total against one cap.
4. Report every violation as: report id, rule, amount over the cap.
5. Approve only when no rule is broken. Never round in the employee's favor.
""",
    "incident-report": """---
name: incident-report
description: How Acme writes incident reports. Use when asked to draft, review or summarize an incident.
---
# Incident report

Sections, in this order, each one short:
1. Summary: one sentence a manager can read alone.
2. Impact: who was affected, for how long, with numbers.
3. Timeline: times in UTC, one line per event.
4. Root cause: the mechanism, not the person. Blameless wording.
5. Actions: each with an owner and a date.
""",
    "leave-request": """---
name: leave-request
description: Acme leave rules. Use when asked whether time off can be taken, carried over or paid out.
---
# Leave rules

1. 25 days of vacation a year, accrued monthly.
2. Up to 5 unused days carry over to March 31 of the next year; the rest expire.
3. Requests over 10 days need approval 4 weeks ahead.
4. Sick days do not count against vacation.
""",
}


def write_skills(root: Path = SKILLS_DIR) -> None:
    """Recreate the skills folder from SKILLS: one subfolder with a SKILL.md per skill."""
    if root.exists():
        shutil.rmtree(root)
    for name, text in SKILLS.items():
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(text, encoding="utf-8")


def read_header(path: Path) -> dict[str, str]:
    """The `key: value` lines between the two `---` markers at the top of a SKILL.md."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    header: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        header[key.strip()] = value.strip()
    return header


def build_index(root: Path = SKILLS_DIR) -> dict[str, str]:
    """Level 1: skill name to description, read from the headers only."""
    index: dict[str, str] = {}
    for path in sorted(root.glob("*/SKILL.md")):
        header = read_header(path)
        if header.get("name") == path.parent.name and header.get("description"):
            index[header["name"]] = header["description"]
    return index


_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


def load_skill(name: str, index: dict[str, str], root: Path = SKILLS_DIR) -> dict[str, Any]:
    """Level 2: the full SKILL.md of one skill. The name is untrusted model output."""
    if not _NAME.fullmatch(name) or name not in index:
        return {"error": f"Unknown skill {name!r}. Available: {sorted(index)}."}
    base = root.resolve()
    path = (base / name / "SKILL.md").resolve()
    if not path.is_relative_to(base):  # a second wall: even a listed name may not leave the folder
        return {"error": f"Refused: {name!r} points outside the skills folder."}
    return {"name": name, "instructions": path.read_text(encoding="utf-8")}


LOAD_SKILL = {
    "name": "load_skill",
    "description": ("Load the full instructions of one skill from the list in the system prompt. "
                    "Call it before working on a request that a skill's description covers."),
    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
}


def system_prompt(index: dict[str, str]) -> str:
    listing = "\n".join(f"- {name}: {description}" for name, description in index.items())
    return ("You are an assistant at Acme. These skills hold Acme's rules. When a request matches a skill, "
            f"load it with load_skill and follow it.\n\nSkills:\n{listing}")


def main(request: str) -> None:
    write_skills()
    index = build_index()
    system = system_prompt(index)
    full = sum(len(t) for t in SKILLS.values())
    print(f"{len(index)} skills; system prompt {len(system)} characters, full skill text {full} characters")

    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]  # append only
    loaded: list[str] = []
    for step in range(1, 6):
        response = chat(messages, system=system, tools=[LOAD_SKILL], effort="low")
        messages.append({"role": "assistant", "content": response.content})
        print(f"step {step}: input tokens {response.input_tokens}, stop {response.stop_reason}")
        if response.stop_reason != "tool_use":
            print(f"\nskills loaded: {loaded or 'none'}")
            print(f"\nanswer:\n{response.text.strip()}")
            return
        results = []
        for tu in response.tool_uses:
            name = str(tu["input"].get("name", ""))
            result = load_skill(name, index) if tu["name"] == "load_skill" else {"error": "unknown tool"}
            if "instructions" in result:
                loaded.append(name)
                print(f"step {step}: loaded skill {name!r} ({len(result['instructions'])} characters)")
            else:
                print(f"step {step}: refused {name!r}: {result['error']}")
            results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(result),
                            **({"is_error": True} if "error" in result else {})})
        messages.append({"role": "user", "content": results})
    print("\n[stop] step limit reached")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or
         "Please check report EXP-014: 3 nights in London, hotel $1,290 in total, meals $300 in total.")
