---
layout: post
title: "Cross-Encoder Reranking: The Final Step to High-Precision RAG"
date: 2025-09-08
tags: [rag, reranking, cross-encoder, retrieval]
read_time: 7
---

Here's a pattern I keep seeing: someone builds a RAG system, gets decent results from vector search, but the top-ranked chunk isn't always the best one. The answer is often in the retrieved set — it's just not ranked first. The LLM gets a noisy context window and produces a mediocre response.

The fix is a reranking step between retrieval and generation.

## Why Vector Search Alone Isn't Enough

Bi-encoder models (what you use to create embeddings for vector search) encode the query and the document independently. They're fast because you can pre-compute document embeddings, but this independence means they miss nuanced query-document interactions.

A cross-encoder, by contrast, takes the query and document together as a single input and directly outputs a relevance score. It's slower — you can't pre-compute anything — but significantly more accurate.

```
Bi-encoder:  embed(query) · embed(doc) → similarity score
Cross-encoder: model(query + doc) → relevance score
```

## The Two-Stage Pipeline

The practical architecture is retrieve-then-rerank:

1. **Retrieve** — Vector search returns top-k candidates (k=20-50, cast a wide net)
2. **Rerank** — Cross-encoder scores each candidate against the query
3. **Generate** — Top-n reranked results (n=3-5) go to the LLM

```python
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5
    ) -> list[tuple[str, float]]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_n]
```

## Integration with a RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
    
    def query(self, question: str) -> str:
        # Stage 1: Broad retrieval
        candidates = self.retriever.search(question, top_k=30)
        
        # Stage 2: Precise reranking
        reranked = self.reranker.rerank(
            query=question,
            documents=[c.text for c in candidates],
            top_n=5
        )
        
        # Stage 3: Generation with high-quality context
        context = "\n\n---\n\n".join([doc for doc, score in reranked])
        
        prompt = f"""Answer based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate(prompt)
```

## Choosing a Model

For most use cases, `cross-encoder/ms-marco-MiniLM-L-12-v2` is the sweet spot — fast enough for real-time use (scoring 30 documents takes ~50ms on a GPU) and accurate enough to make a noticeable difference.

For domain-specific applications, fine-tuning a cross-encoder on your own relevance judgments gives another significant bump. Even a few hundred labeled examples help.

| Model | Latency (30 docs) | NDCG@10 |
|-------|-------------------|---------|
| ms-marco-MiniLM-L-6 | ~25ms | 0.39 |
| ms-marco-MiniLM-L-12 | ~50ms | 0.41 |
| ms-marco-electra-base | ~80ms | 0.43 |

## The Latency Question

The main objection to reranking is added latency. In practice, the numbers are manageable. Scoring 30 candidates with the L-12 model adds about 50ms on a GPU, 200ms on CPU. Compare that to the LLM generation step which often takes 2-5 seconds — the reranking overhead is negligible.

If latency is truly critical, you can also run the reranker asynchronously with retrieval. Start the vector search, and while results stream back, warm up the reranker. By the time all candidates arrive, the model is ready to score them.

## Impact

In my benchmarks on internal document QA, adding a reranking step improved answer correctness by roughly 15-20%. The effect was most pronounced on ambiguous queries where the top vector search result was topically related but not actually answering the question.

The pattern is clear: vector search for recall, cross-encoder for precision. It's one of the highest ROI improvements you can make to an existing RAG system.
