# Lesson 03d: Documents and Citations

Lesson 03b built citations by hand: a `submit_answer` tool with a list of chunk ids, then a crude check that the cited chunks share words with the answer. It works, but the model can still cite a chunk that does not say what the answer claims, and your verifier can only guess.

The Messages API can do this natively. You send your sources as **document** blocks with citations switched on, and the answer comes back split into pieces, each piece carrying the **exact passage** it rests on and where that passage sits in the source. No extra tool, no schema, and a quote you can check character for character.

This lesson covers what you can put in a message besides text (documents and images), how native citations come back, and two rules from API version `2026-10-21` that decide which role a document or a cited answer may sit in.

## What a user message can hold

A user message's `content` is a list of blocks. Besides `text`:

| Block | Source types | Notes |
|---|---|---|
| `document` | `text` (plain text), `base64` PDF, `url` PDF, `file` (Files API), `content` (your own list of text blocks) | Citable. Optional `title` and `context` are shown to the model but never cited |
| `image` | `base64`, `url`, `file` | JPEG, PNG, GIF, WebP. Not citable |

```python
{"type": "document",
 "source": {"type": "text", "media_type": "text/plain", "data": policy_text},
 "title": "expense_policy.md",
 "citations": {"enabled": True}}

{"type": "document",
 "source": {"type": "base64", "media_type": "application/pdf", "data": b64_string}}

{"type": "image",
 "source": {"type": "base64", "media_type": "image/png", "data": b64_string}}
```

Practical rules:

- **Put the files before the question** in the same user message. The model reads them first, then the ask.
- **Base64 with no newlines.** Use `base64.standard_b64encode(...)`, not a line wrapping encoder.
- **PDF limits:** 32 MB per request and 600 pages (100 on models with a 200K context). PDF citations come from the extracted text, so a scanned PDF with no text layer gives little to cite.
- **Formats the block does not take** (Word, Excel): convert to text and send a `text` document.
- **Everything is input tokens.** A 50 page PDF can cost more than the rest of the conversation. Prompt caching (Lesson 02b) helps when you ask several questions about the same file.

### The Files API: upload once, reference by id

When the same file is used in many requests, upload it once and send a small reference instead of the bytes each time:

```python
uploaded = client().files.upload(file=Path("handbook.pdf"))
block = {"type": "document", "source": {"type": "file", "file_id": uploaded.id}}
```

The block type must match the file: a PDF or text file goes in a `document`, an image in an `image`. Upload, list and delete are free; the content is billed as input tokens when a message uses it. Files stay until you delete them (`client().files.delete(file_id)`). The Files API is not available on Amazon Bedrock or Google Vertex AI.

### Documents belong in user messages

On API version `2026-10-21`, a `document` block placed directly in an `assistant` message is a 400. Documents are input you provide, so they go in a user turn at the point the conversation needs them. A document *nested* in Claude's own tool result (a web fetch result from Lesson 01c) is fine: send those back unchanged.

## Native citations

Turn them on per document with `"citations": {"enabled": true}`. It is all or none: every document in the request, or no document.

The answer then arrives as several `text` blocks. Plain connecting text has no citations; each claim that rests on a source has a `citations` list:

```json
[
  {"type": "text", "text": "Hotels are capped at "},
  {"type": "text", "text": "$300 per night, $400 in NYC, London and Tokyo",
   "citations": [{"type": "char_location",
                  "cited_text": "Hotels capped at $300/night ($400 in NYC, London,\nTokyo).",
                  "document_index": 0, "document_title": "expense_policy.md",
                  "start_char_index": 236, "end_char_index": 293}]},
  {"type": "text", "text": "."}
]
```

Every citation has `cited_text`, `document_index` and `document_title`. The location depends on the kind of document:

| Document | Location `type` | Fields | Indexing |
|---|---|---|---|
| Plain text | `char_location` | `start_char_index`, `end_char_index` | 0 based, end exclusive |
| PDF | `page_location` | `start_page_number`, `end_page_number` | 1 based, end exclusive |
| Custom content | `content_block_location` | `start_block_index`, `end_block_index` | 0 based, end exclusive |

`document_index` counts every document block in the request, across all messages, from 0. Plain text and PDFs are split into sentences, so a citation is one sentence or a run of them; with a custom content document you decide the units yourself (one block per bullet, per transcript line, per RAG chunk). Citations cover text only, not images or images inside PDFs.

Cost: switching citations on adds a few input tokens. `cited_text` does not count as output tokens, and does not count as input tokens when you send the answer back later.

### Citations and structured output do not mix

Citations interleave quotes with text, which a strict JSON schema cannot express. A request with citations enabled on any document **and** `output_config.format` (the `output_schema` argument of `chat()`) is a 400. Choose per call: a typed object (Lesson 02) or a cited answer. If you need both, run two calls: one cited answer, then one extraction over it, with the citations stripped.

### Cited text must not go back in a user message

The second rule from API version `2026-10-21`: a `text` block with a non empty `citations` list inside a **user** message is a 400. That includes such a block inside a `tool_result` or a `search_result`.

- **Same conversation:** sending Claude's answer back as the `assistant` turn, unchanged, is fine. Keep the citations there.
- **Hand off:** passing the answer to another agent, a reviewer, a summarizer or a later prompt puts it in a *user* turn. Strip the citations and keep the text:

```python
def strip_citations(blocks):
    return [{"type": "text", "text": b["text"]} for b in blocks if b["type"] == "text"]
```

The SDK types still accept the cited blocks, so this error appears only at run time. Search your code for places that copy `response.content` into a user message.

## Native citations vs Lesson 03b's hand built citations

| | Lesson 03b (`submit_answer` with chunk ids) | This lesson (native citations) |
|---|---|---|
| What you get | A list of chunk ids for the whole answer | A quote and its location for each claim |
| Verification | Word overlap guess after the fact | Exact: compare `cited_text` with your source slice |
| Granularity | Whole chunk | Sentence (or your own blocks) |
| Extra machinery | A tool, a schema, a loop step | One flag per document |
| Typed output in the same call | Yes, the tool is the schema | No, a 400 with `output_config.format` |
| Retrieval | The model searches with a tool | You decide what goes in; or keep the search tool and return chunks as `search_result` blocks |
| Works with | Any model and provider | The Claude API |

They are not rivals. Retrieval still decides *which* text the model sees (Lessons 03a to 03c); native citations replace only the last step, proving *which part* of that text supports each claim. `03_compare_with_03b.py` keeps 03a's TF-IDF index and swaps only that step.

## Hands on

### `01_text_documents_with_citations.py`: two handbook files, cited answer

Sends `expense_policy.md` and `remote_work.md` as text documents with citations on. Prints the answer with numbered footnotes, each with its document, character range, and quote, and checks every quote against the file. Notice how many text blocks one answer becomes.

### `02_pdf_or_image.py`: your own PDF or image

Takes a path (prints usage if you give none). A PDF goes in as a base64 document and the answer cites page numbers; an image goes in as an image block with no citations. Add `--upload` to go through the Files API instead: upload, reference by `file_id`, delete. No binary files ship with the course; bring your own.

### `03_compare_with_03b.py`: the 03b pipeline on native citations, and the hand off rule

Retrieves the top three chunks with Lesson 03a's index, sends each as a document titled with its chunk id, and prints which chunks were actually cited. Then hands the answer to a reviewer call with the citations stripped, and asks a follow up in the original conversation with the assistant turn kept unchanged.

## Key takeaways

- Documents and images are content blocks in a user message, placed before the question.
- `citations: {"enabled": true}` on every document gives you a quote and a location per claim.
- Locations: characters for text (0 based), pages for PDFs (1 based), blocks for custom content; all end exclusive.
- Verify citations exactly by slicing your own copy of the source.
- Citations and `output_config.format` cannot be combined in one call.
- On API `2026-10-21`: documents never directly in assistant messages; cited text never in user messages. Strip citations when you hand an answer on.
- Upload a reused file once with the Files API and reference its `file_id`.

## Common misconceptions

- **"Native citations mean the answer is correct."** They mean the quote exists in your document. Whether the claim follows from it still needs judgment (Lesson 07).
- **"I can cite an image."** Citations cover text only.
- **"The Files API makes a file free to use."** Storage and upload are free; every request that uses the file pays for its tokens.
- **"I can strip `citations` from Claude's assistant turn to save tokens."** No need: `cited_text` is not billed as input, and the assistant turn may keep them.
- **"Copying the response into the next prompt is harmless."** On `2026-10-21` it is a 400 if the copied blocks carry citations.

Next: [Lesson 04: Memory and state](../04_memory_state/README.md).
