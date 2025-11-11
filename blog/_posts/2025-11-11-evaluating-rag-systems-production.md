---
layout: post
title: "Evaluating RAG Systems in Production: Metrics that Actually Matter"
date: 2025-11-11
tags: [rag, evaluation, metrics, production]
read_time: 8
---

Building a RAG system is the easy part. Knowing whether it's actually working well — and catching when it starts to degrade — is where most teams struggle. I've seen systems run for months with silently terrible retrieval because nobody set up proper evaluation.

## The Core Metrics

After experimenting with various evaluation frameworks, I've landed on three metrics that actually tell you something useful:

**Faithfulness** — Does the generated answer only use information present in the retrieved context? This catches hallucination, which is the cardinal sin of RAG. If your LLM is making stuff up instead of saying "I don't know", your faithfulness score will show it.

**Answer Relevance** — Does the generated answer actually address the question? High faithfulness but low relevance means your system is accurately quoting irrelevant passages.

**Context Precision** — Of the chunks retrieved, how many were actually useful for answering the question? Low context precision means you're stuffing the LLM's context window with noise.

## Automated Evaluation with LLM-as-Judge

Manual evaluation doesn't scale. The practical approach is using an LLM to evaluate your RAG system's outputs:

```python
def evaluate_faithfulness(question: str, answer: str, context: str, judge_llm) -> float:
    prompt = f"""Evaluate whether the answer is faithful to the given context.
The answer should ONLY contain information present in the context.

Context: {context}

Question: {question}

Answer: {answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0: Every claim in the answer is supported by the context
- 0.5: Some claims are supported, others are not
- 0.0: The answer contains information not in the context

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""
    
    result = judge_llm.generate(prompt)
    return json.loads(result)
```

## Building a Golden Dataset

You need ground truth to evaluate against. Here's the approach I use:

```python
def generate_synthetic_qa(documents: list[str], llm) -> list[dict]:
    qa_pairs = []
    
    for doc in documents:
        prompt = f"""Given this document, generate 3 question-answer pairs.
Questions should be specific and answerable from the text.
Answers should be concise and factual.

Document: {doc}

Return as JSON array: [{{"question": "...", "answer": "...", "source_chunk": "..."}}]"""
        
        pairs = json.loads(llm.generate(prompt))
        qa_pairs.extend(pairs)
    
    return qa_pairs
```

Then manually review and filter these. You want 50-100 high-quality pairs for a reliable eval set. It takes a few hours but it's a one-time investment that pays off forever.

## The Evaluation Loop

```python
class RAGEvaluator:
    def __init__(self, rag_pipeline, judge_llm):
        self.rag = rag_pipeline
        self.judge = judge_llm
    
    def evaluate(self, test_set: list[dict]) -> dict:
        scores = {
            "faithfulness": [],
            "relevance": [],
            "context_precision": []
        }
        
        for test_case in test_set:
            question = test_case["question"]
            expected = test_case["answer"]
            
            # Run the RAG pipeline
            result = self.rag.query(question)
            
            # Score each metric
            faith = evaluate_faithfulness(
                question, result.answer, result.context, self.judge
            )
            scores["faithfulness"].append(faith["score"])
            
            rel = evaluate_relevance(
                question, result.answer, self.judge
            )
            scores["relevance"].append(rel["score"])
            
            prec = evaluate_context_precision(
                question, expected, result.retrieved_chunks, self.judge
            )
            scores["context_precision"].append(prec["score"])
        
        return {k: sum(v)/len(v) for k, v in scores.items()}
```

## Running Evals in CI

The real value comes when you integrate evaluation into your pipeline:

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation
on:
  push:
    paths:
      - 'src/chunking/**'
      - 'src/retrieval/**'
      - 'config/rag_config.yaml'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG evaluation
        run: python eval/run_eval.py --test-set eval/golden_dataset.json
      - name: Check thresholds
        run: |
          python -c "
          import json
          results = json.load(open('eval/results.json'))
          assert results['faithfulness'] >= 0.85, 'Faithfulness below threshold'
          assert results['relevance'] >= 0.80, 'Relevance below threshold'
          "
```

Every time someone changes the chunking strategy or retrieval configuration, the eval runs automatically. If metrics drop below your thresholds, the PR gets flagged.

## What "Good" Looks Like

From my experience across multiple production RAG systems:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Faithfulness | 0.80 | 0.90 | 0.95+ |
| Answer Relevance | 0.75 | 0.85 | 0.90+ |
| Context Precision | 0.60 | 0.75 | 0.85+ |

These numbers shift based on domain complexity. Legal and medical documents are harder to get right than general knowledge bases.

The point isn't to chase perfect scores — it's to have a baseline, detect regressions, and make data-driven decisions about which pipeline changes actually help.
