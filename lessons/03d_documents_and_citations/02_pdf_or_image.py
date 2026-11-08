"""Lesson 03d: a PDF or an image as input.

Pass a path to your own file (nothing binary ships with the course):
    * a PDF goes in as a base64 `document` block with citations on, and
      the answer cites page numbers (`page_location`, 1 based, end exclusive);
    * an image (.png, .jpg, .jpeg, .gif, .webp) goes in as an `image`
      block. Citations work on text only, so an image answer has none.

Add --upload to send the file through the Files API instead: upload
once, reference it by `file_id`, and delete it at the end. Useful when
the same file is used in many requests.

Run:
    uv run python lessons/03d_documents_and_citations/02_pdf_or_image.py path/to/file.pdf
    uv run python lessons/03d_documents_and_citations/02_pdf_or_image.py path/to/photo.png "What is shown?"
    uv run python lessons/03d_documents_and_citations/02_pdf_or_image.py path/to/file.pdf --upload
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._llm import DEFAULT_MODEL, chat, client

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}


def inline_source(path: Path, media_type: str) -> dict[str, Any]:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")  # no newlines
    return {"type": "base64", "media_type": media_type, "data": data}


def build_block(path: Path, upload: bool) -> tuple[dict[str, Any], str | None]:
    """Return the content block and, if uploaded, the file id to clean up."""
    suffix = path.suffix.lower()
    is_pdf = suffix == ".pdf"
    if not is_pdf and suffix not in IMAGE_TYPES:
        raise SystemExit(f"Unsupported file type {suffix!r}: use a PDF or {sorted(IMAGE_TYPES)}")
    media_type = "application/pdf" if is_pdf else IMAGE_TYPES[suffix]

    file_id = None
    if upload:
        uploaded = client().files.upload(file=(path.name, path.read_bytes(), media_type))
        file_id = uploaded.id
        print(f"Uploaded {path.name} as {file_id}")
        source: dict[str, Any] = {"type": "file", "file_id": file_id}
    else:
        source = inline_source(path, media_type)

    if is_pdf:
        return {"type": "document", "source": source, "title": path.name,
                "citations": {"enabled": True}}, file_id
    return {"type": "image", "source": source}, file_id


def main(argv: list[str]) -> None:
    upload = "--upload" in argv
    args = [a for a in argv if a != "--upload"]
    if not args:
        print(__doc__)
        return
    path = Path(args[0])
    if not path.is_file():
        raise SystemExit(f"No such file: {path}")
    is_pdf = path.suffix.lower() == ".pdf"
    question = args[1] if len(args) > 1 else (
        "Summarize the three most important points, citing the pages." if is_pdf
        else "Describe what this image shows in three sentences.")

    block, file_id = build_block(path, upload)
    try:
        # The file goes before the question, in a USER message.
        content = [block, {"type": "text", "text": question}]
        response = chat([{"role": "user", "content": content}], effort="low")
    finally:
        if file_id:
            client().files.delete(file_id)
            print(f"Deleted {file_id}")

    print(f"Model: {DEFAULT_MODEL}  stop={response.stop_reason}  "
          f"input tokens={response.input_tokens} (the file is most of these)\n")
    for b in response.content:
        if b["type"] != "text":
            continue
        print(b["text"], end="")
        for c in b.get("citations") or []:
            if c["type"] == "page_location":
                print(f" [p.{c['start_page_number']}]", end="")
    print()
    if not is_pdf:
        print("\n(No citations: they cover text in documents, not images.)")


if __name__ == "__main__":
    main(sys.argv[1:])
