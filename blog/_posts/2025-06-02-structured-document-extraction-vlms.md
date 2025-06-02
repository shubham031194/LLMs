---
layout: post
title: "Structured Document Extraction: Moving Beyond OCR with Vision Language Models"
date: 2025-06-02
tags: [vlm, ocr, document-ai, pdf]
read_time: 8
---

Traditional OCR gives you a wall of text. It doesn't understand that this block is a table header, that section is a footnote, or that the number in the top-right corner is a page number you should probably ignore. For months, I patched around this with heuristics — regex for table detection, coordinate-based logic for headers, custom rules for every new document format.

Then I started using Vision Language Models for extraction, and most of those heuristics became unnecessary.

## The Problem with OCR Pipelines

Here's what a typical OCR-based extraction pipeline looks like:

```
PDF → Page Images → Tesseract/PaddleOCR → Raw Text → Regex/Rules → Structured Data
```

Each arrow is a failure point. The OCR might misread characters. The text ordering might be wrong for multi-column layouts. The regex breaks on a slightly different table format. It's fragile engineering layered on fragile engineering.

## VLM-Based Extraction

The shift is conceptual: instead of extracting text and then trying to understand structure, you show the model the page image and ask it to extract structured data directly.

```python
import base64
from openai import OpenAI

def extract_table_from_page(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode()

    client = OpenAI()  # or your local endpoint
    response = client.chat.completions.create(
        model="gpt-4o",  # or a local VLM
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract all tables from this document page.
Return a JSON array where each table is an object with:
- "title": the table caption or heading
- "headers": array of column headers
- "rows": array of arrays (each inner array is one row)

Return ONLY valid JSON, no markdown."""
                    }
                ]
            }
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content)
```

## Where VLMs Actually Win

**Complex table structures.** Merged cells, nested headers, tables that span multiple columns — VLMs handle these because they see the visual layout, not just the text stream.

**Forms and key-value pairs.** Insurance forms, government documents, invoices — the spatial relationship between a label and its value is obvious visually but hard to reconstruct from OCR text coordinates.

**Mixed content pages.** A page with a chart, a table, and paragraphs of text. The VLM understands what each region is and can extract them separately.

## The Practical Pipeline

Here's what I actually run in production:

```python
from pdf2image import convert_from_path

def process_document(pdf_path: str) -> list[dict]:
    pages = convert_from_path(pdf_path, dpi=200)
    results = []

    for i, page in enumerate(pages):
        page_path = f"/tmp/page_{i}.png"
        page.save(page_path, "PNG")

        extraction = extract_structured_content(page_path)
        extraction["page_number"] = i + 1
        results.append(extraction)

    return results
```

The DPI setting matters. Too low and the model can't read small text. Too high and you're burning tokens on unnecessary detail. 200 DPI is the sweet spot for most documents.

## Cost and Latency Considerations

VLM extraction is slower and more expensive than pure OCR. A single page through GPT-4o costs roughly 800-1200 tokens of input (the image) plus output tokens. For high-volume processing, I run a local VLM (LLaVA or Qwen-VL) behind an Ollama endpoint.

The tradeoff is accuracy vs speed. For documents where extraction quality matters — contracts, medical records, financial statements — the VLM approach pays for itself in reduced manual correction time.

## Hybrid Approach

What I actually recommend: use fast OCR for simple, well-formatted documents (like machine-generated PDFs), and route complex or scanned documents to the VLM pipeline. A simple classifier on page complexity can handle the routing.

The key insight is that VLMs don't replace OCR — they replace the fragile post-processing layer that made OCR painful to maintain.
