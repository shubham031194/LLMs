---
layout: post
title: "Edge AI and Small Language Models (SLMs) in Restricted Environments"
date: 2026-05-02
tags: [edge-ai, slm, privacy, quantization, on-device]
read_time: 7
---

Not every AI workload can call an API. Some environments — air-gapped networks, medical devices, factory floors, government systems — require everything to run locally, with strict data privacy constraints. This is where Small Language Models and edge deployment become essential.

## Why Not Just Use a Big Model?

The constraints are real:

- **No internet access** — air-gapped environments can't reach OpenAI or Anthropic
- **Data cannot leave the device** — regulatory requirements (HIPAA, GDPR) or contractual obligations
- **Latency requirements** — 100ms response times, not 2-second API roundtrips
- **Hardware limits** — 8GB RAM, no GPU, maybe an ARM processor

The good news is that the 1-3B parameter model class has gotten remarkably capable for focused tasks.

## Model Selection

For edge deployment, I evaluate on three axes: task accuracy, inference speed, and memory footprint.

```python
# Benchmarking framework
import time
import psutil

class ModelBenchmark:
    def __init__(self, model, test_cases: list[dict]):
        self.model = model
        self.test_cases = test_cases
    
    def run(self) -> dict:
        results = {
            "accuracy": 0,
            "avg_latency_ms": 0,
            "peak_memory_mb": 0,
            "tokens_per_second": 0
        }
        
        latencies = []
        correct = 0
        
        for case in self.test_cases:
            mem_before = psutil.Process().memory_info().rss
            
            start = time.perf_counter()
            output = self.model.generate(case["input"], max_tokens=100)
            latency = (time.perf_counter() - start) * 1000
            
            mem_after = psutil.Process().memory_info().rss
            
            latencies.append(latency)
            if self.evaluate(output, case["expected"]):
                correct += 1
            
            results["peak_memory_mb"] = max(
                results["peak_memory_mb"],
                (mem_after - mem_before) / 1024 / 1024
            )
        
        results["accuracy"] = correct / len(self.test_cases)
        results["avg_latency_ms"] = sum(latencies) / len(latencies)
        
        return results
```

Models I've had success with on edge hardware:

| Model | Parameters | RAM (Q4) | Best For |
|-------|-----------|----------|----------|
| Phi-3 Mini | 3.8B | ~2.5GB | General reasoning |
| Qwen2 1.5B | 1.5B | ~1.2GB | Multilingual tasks |
| TinyLlama | 1.1B | ~0.8GB | Simple classification |
| Gemma 2B | 2B | ~1.5GB | Instruction following |

## Quantization for Deployment

Full-precision models don't fit on edge devices. Quantization is mandatory:

```bash
# Using llama.cpp for quantization
./quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

The Q4_K_M quantization level is my default — it reduces model size by roughly 4x with minimal quality loss. For extremely constrained devices, Q3_K_S works but expect a noticeable accuracy drop on complex reasoning tasks.

```python
from llama_cpp import Llama

# Load quantized model — runs on CPU, ~2GB RAM
model = Llama(
    model_path="./models/phi-3-mini-q4_k_m.gguf",
    n_ctx=2048,          # Context window
    n_threads=4,         # Match your CPU cores
    n_gpu_layers=0,      # 0 for CPU-only
    verbose=False
)

def classify_document(text: str) -> str:
    response = model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": f"Classify this document into one of: invoice, contract, report, correspondence.\n\nDocument: {text[:500]}\n\nCategory:"
        }],
        max_tokens=10,
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()
```

## Domain Fine-Tuning

A general 2B model won't match GPT-4 on open-ended tasks. But fine-tuned on your specific domain, it can match or exceed larger models on narrow tasks:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments

# LoRA fine-tuning — efficient even on modest hardware
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
model = get_peft_model(model, lora_config)

# Trainable parameters drop from billions to millions
model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 3,824,321,536 || trainable%: 0.089
```

500-1000 domain-specific examples typically get you most of the way there. The model doesn't need to learn language — it already knows that. It just needs to learn your domain's patterns and vocabulary.

## Deployment Architecture

For production edge deployment, I use a simple service wrapper:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    app.state.model = Llama(
        model_path="./models/domain-model-q4.gguf",
        n_ctx=2048,
        n_threads=4
    )
    yield
    # Cleanup
    del app.state.model

app = FastAPI(lifespan=lifespan)

@app.post("/classify")
async def classify(text: str):
    result = app.state.model.create_chat_completion(...)
    return {"category": result}
```

This runs on any Linux box — even a Raspberry Pi 5 can handle a quantized 1.5B model for classification tasks.

The key insight is that edge AI isn't about running the biggest possible model locally. It's about picking the right small model, fine-tuning it on your exact task, and deploying it efficiently. A well-tuned 2B model running in 200ms on a $50 board beats a cloud API that's blocked by your firewall.
