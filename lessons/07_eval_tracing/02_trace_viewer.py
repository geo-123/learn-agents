"""Lesson 07: pretty print a JSONL trace as a tree.

Reads spans from a .jsonl file, rebuilds the parent/child tree, and
prints it with timing and inline outputs.

Run:
    uv run python lessons/07_eval_tracing/02_trace_viewer.py <path-to-jsonl>

Or with no args: uses the most recent trace in traces/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def latest_trace() -> Path | None:
    traces_dir = Path(__file__).resolve().parents[2] / "traces"
    files = sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_spans(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def render(spans: list[dict]) -> None:
    by_parent: dict[str | None, list[dict]] = defaultdict(list)
    for s in spans:
        by_parent[s["parent_id"]].append(s)

    # The "root" spans are those whose parent_id is None.
    roots = by_parent[None]

    def walk(span: dict, depth: int) -> None:
        prefix = "  " * depth + ("└─ " if depth else "")
        attrs = " ".join(f"{k}={v}" for k, v in span["attrs"].items())
        out_str = ""
        if span["outputs"]:
            preview = {k: v for k, v in span["outputs"].items() if k != "text_preview"}
            if "text_preview" in span["outputs"]:
                preview["text"] = span["outputs"]["text_preview"][:60] + "..."
            out_str = "  ➜ " + ", ".join(f"{k}={v}" for k, v in preview.items())
        status = "" if span["status"] == "ok" else f"  [{span['status']}]"
        print(f"{prefix}{span['kind']}  ({span['ms']} ms)  {attrs}{out_str}{status}")
        # children come from a defaultdict; order by their start order in file.
        for child in by_parent[span["span_id"]]:
            walk(child, depth + 1)

    for r in roots:
        walk(r, 0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = latest_trace()  # type: ignore[assignment]
        if path is None:
            print("no trace files found in traces/")
            sys.exit(1)
    print(f"# {path}\n")
    render(load_spans(path))
