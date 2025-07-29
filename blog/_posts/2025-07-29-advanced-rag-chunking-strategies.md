---
layout: post
title: "Advanced RAG Chunking Strategies: Semantic Splitting and Contextual Retrieval"
date: 2025-07-29
tags: [rag, chunking, embeddings, retrieval]
read_time: 9
---

The quality of a RAG system lives and dies in its chunking strategy. I've spent more time debugging bad retrieval results that traced back to poor chunking than any other single component.

Fixed-size chunking with overlap is where everyone starts. It's also where most retrieval quality problems start.

## Why Fixed-Size Chunking Breaks

Consider a technical document with this structure:

```
## Database Configuration
### Connection Pooling
The max pool size should be set to...
[200 words of explanation]
### Timeout Settings
Connection timeout should be...
```

With a naive 500-token chunk and 50-token overlap, you might get a chunk that starts mid-paragraph in the Connection Pooling section and ends mid-paragraph in the Timeout Settings section. When someone asks "What should the connection pool size be?", the chunk that matches contains half the answer mixed with unrelated timeout information.

## Strategy 1: Markdown-Aware Splitting

If your documents have any structure at all — markdown headers, HTML tags, numbered sections — use it.

```python
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: dict

def split_by_headers(markdown: str, max_tokens: int = 500) -> list[Chunk]:
    # Split on h2 and h3 headers
    sections = re.split(r'(^#{2,3}\s+.+$)', markdown, flags=re.MULTILINE)
    
    chunks = []
    current_header = ""
    current_content = ""
    
    for section in sections:
        if re.match(r'^#{2,3}\s+', section):
            if current_content.strip():
                chunks.append(Chunk(
                    content=f"{current_header}\n{current_content}".strip(),
                    metadata={"header": current_header.strip('# ')}
                ))
            current_header = section
            current_content = ""
        else:
            current_content += section
    
    # Don't forget the last section
    if current_content.strip():
        chunks.append(Chunk(
            content=f"{current_header}\n{current_content}".strip(),
            metadata={"header": current_header.strip('# ')}
        ))
    
    return chunks
```

This is simple but effective. Each chunk respects the document's own structure. The header is preserved as both content and metadata, so retrieval gets the context it needs.

## Strategy 2: Semantic Splitting

When documents lack clear structural markers, you can split on semantic boundaries — points where the topic shifts.

The idea is to compute embedding similarity between consecutive sentences. When similarity drops below a threshold, that's a split point.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_split(text: str, threshold: float = 0.5) -> list[str]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sentences = text.split('. ')
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        
        if similarity < threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append('. '.join(current_chunk))
    return chunks
```

The threshold parameter needs tuning per domain. Legal documents tend to be more uniform in language (higher threshold needed), while technical docs with mixed topics split more easily.

## Strategy 3: Contextual Retrieval (Prepending Context)

This is the technique that gave me the biggest single improvement in retrieval quality. The idea, popularized by Anthropic, is to prepend a brief contextual summary to each chunk before embedding.

A chunk that says "Set this value to 30 seconds" is almost useless on its own. But "Database Configuration > Timeout Settings: Set this value to 30 seconds" immediately becomes retrievable.

```python
def add_context(chunk: str, full_document: str, llm_client) -> str:
    prompt = f"""Given the full document and a specific chunk from it,
write a 1-2 sentence context that situates this chunk within the 
broader document. Be specific and concise.

Document: {full_document[:3000]}

Chunk: {chunk}

Context:"""
    
    context = llm_client.complete(prompt)
    return f"{context}\n\n{chunk}"
```

Yes, this means an LLM call per chunk during ingestion. It's expensive. But ingestion is a one-time cost, and the retrieval quality improvement is significant — in my benchmarks, contextual retrieval reduced "wrong chunk retrieved" errors by about 40%.

## What I Actually Use

In practice, I combine all three. Markdown-aware splitting first, semantic splitting as a fallback for unstructured sections, and contextual prepending on everything before embedding.

The chunking pipeline runs during ingestion, not at query time, so the additional cost is amortized across all future queries. It's one of those investments where spending more compute upfront saves you from building increasingly complex query-time hacks later.
