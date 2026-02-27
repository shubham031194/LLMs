---
layout: post
title: "From Pipelines to Loops: Architecting Agentic RAG"
date: 2026-02-27
tags: [rag, agents, llm, tool-calling]
read_time: 9
---

Standard RAG is a pipeline: query → retrieve → generate. It works for simple questions against a known corpus. But the moment your question requires reasoning across multiple documents, iterative refinement, or deciding *what* to search for — the pipeline model falls apart.

Agentic RAG replaces the fixed pipeline with a loop where the LLM decides its own retrieval strategy.

## The Limitation of Linear RAG

Ask a standard RAG system: "How did our API latency change after the migration to the new database, and what were the main causes?"

A pipeline system does one search, grabs the top-5 chunks, and hopes the answer is in there. But this question actually requires multiple retrieval steps — migration details, latency metrics, and root cause analysis — potentially from different data sources.

## The Agent Loop

```python
from typing import Literal
from pydantic import BaseModel

class SearchAction(BaseModel):
    query: str
    source: Literal["docs", "metrics", "tickets"] = "docs"
    reason: str

class AnswerAction(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

SYSTEM_PROMPT = """You are a research assistant with access to tools.
For each user question, decide whether to:
1. SEARCH - query a knowledge base for more information
2. ANSWER - provide a final answer based on gathered context

You can search multiple times to build a complete picture.
Always explain your reasoning before choosing an action."""

class AgenticRAG:
    def __init__(self, llm, retriever, max_iterations: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
    
    async def query(self, question: str) -> str:
        context_so_far = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        for i in range(self.max_iterations):
            response = await self.llm.generate(
                messages=messages,
                tools=[search_tool, answer_tool]
            )
            
            if response.tool_call.name == "answer":
                return response.tool_call.args
            
            if response.tool_call.name == "search":
                search = SearchAction(**response.tool_call.args)
                results = await self.retriever.search(
                    query=search.query,
                    source=search.source
                )
                
                context_so_far.extend(results)
                
                # Feed results back into the conversation
                messages.append({
                    "role": "assistant",
                    "content": response.text
                })
                messages.append({
                    "role": "user",
                    "content": f"Search results for '{search.query}':\n{format_results(results)}"
                })
        
        # Max iterations reached — force an answer
        return self.force_answer(question, context_so_far)
```

## Tool Design

The tools you give the agent shape its behavior. Overly broad tools lead to unfocused searches. Too many tools cause decision paralysis.

```python
tools = [
    {
        "name": "search_documentation",
        "description": "Search technical docs, API references, and runbooks",
        "parameters": {
            "query": {"type": "string", "description": "Specific search query"},
            "filters": {
                "type": "object",
                "properties": {
                    "doc_type": {"enum": ["api", "runbook", "architecture", "all"]},
                    "date_range": {"type": "string", "description": "e.g., 'last 30 days'"}
                }
            }
        }
    },
    {
        "name": "query_metrics",
        "description": "Query time-series metrics like latency, error rates, throughput",
        "parameters": {
            "metric_name": {"type": "string"},
            "time_range": {"type": "string"},
            "aggregation": {"enum": ["avg", "p50", "p95", "p99", "max"]}
        }
    },
    {
        "name": "answer",
        "description": "Provide the final answer. Use only after sufficient research.",
        "parameters": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "sources": {"type": "array", "items": {"type": "string"}}
        }
    }
]
```

## Controlling the Loop

Unbounded agent loops are dangerous in production. Three controls I always implement:

**Iteration limit.** Hard cap at 5-7 iterations. If the agent hasn't found an answer by then, it should say so rather than spinning.

**Token budget.** Track cumulative token usage across the loop. Once you hit a budget threshold, force the agent to answer with whatever context it has.

**Search deduplication.** Track previous queries and prevent the agent from searching for essentially the same thing twice:

```python
def is_duplicate_search(new_query: str, previous_queries: list[str], threshold: float = 0.85) -> bool:
    new_embedding = embed(new_query)
    for prev in previous_queries:
        prev_embedding = embed(prev)
        if cosine_similarity(new_embedding, prev_embedding) > threshold:
            return True
    return False
```

## When to Use Agentic RAG

Not every query needs an agent. Simple factual lookups — "What is the max connection pool size?" — are better served by a fast pipeline.

I route based on query complexity:

```python
def route_query(question: str, classifier) -> str:
    complexity = classifier.predict(question)
    
    if complexity == "simple":
        return "pipeline"  # Standard retrieve-and-generate
    elif complexity == "complex":
        return "agent"     # Multi-step agentic loop
```

A simple classifier (even keyword-based) that checks for multi-part questions, comparisons, or temporal reasoning works surprisingly well as a router.

The key insight is that agentic RAG isn't a replacement for pipeline RAG — it's a complementary architecture for the queries that pipelines can't handle. Use both, route intelligently, and keep the agent loop tightly controlled.
