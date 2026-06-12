---
layout: post
title: "Asynchronous AI Processing: Offloading Heavy Document Generation"
date: 2026-06-12
tags: [async, celery, redis, task-queue, python]
read_time: 7
---

Some AI workloads take minutes, not milliseconds. Embedding a 500-page PDF. Generating a 30-page report with charts. Batch-processing a thousand invoices through a VLM. You can't hold an HTTP connection open for that long.

The answer is task queues — decouple the request from the processing and let the user check back when it's done.

## The Architecture

```
Client → FastAPI → Redis (broker) → Celery Worker → Result Store
                                         ↓
                                   GPU / CPU processing
```

The API accepts the job and returns immediately with a task ID. The worker processes it in the background. The client polls or subscribes for completion.

## Setting It Up

```python
# tasks.py
from celery import Celery
from celery.signals import task_prerun, task_postrun
import time

app = Celery(
    'ai_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_time_limit=600,       # Hard kill after 10 minutes
    task_soft_time_limit=540,  # Soft limit — raises exception
    worker_prefetch_multiplier=1,  # Critical for GPU tasks
    worker_concurrency=2,      # Match your GPU capacity
)

@app.task(bind=True)
def process_document(self, document_path: str, options: dict) -> dict:
    self.update_state(state='PROCESSING', meta={
        'stage': 'loading',
        'progress': 0
    })
    
    # Load and chunk the document
    chunks = load_and_chunk(document_path)
    total = len(chunks)
    
    self.update_state(state='PROCESSING', meta={
        'stage': 'embedding',
        'progress': 0,
        'total_chunks': total
    })
    
    # Embed in batches with progress updates
    embeddings = []
    batch_size = 32
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embed_batch(batch)
        embeddings.extend(batch_embeddings)
        
        self.update_state(state='PROCESSING', meta={
            'stage': 'embedding',
            'progress': min(i + batch_size, total) / total * 100,
            'total_chunks': total
        })
    
    # Store in vector DB
    store_embeddings(embeddings, chunks, document_path)
    
    return {
        'status': 'complete',
        'chunks_processed': total,
        'document': document_path
    }
```

The `worker_prefetch_multiplier=1` setting is crucial for GPU workloads. Without it, Celery prefetches multiple tasks per worker, which can cause GPU OOM errors when two large jobs try to run simultaneously.

## The API Layer

```python
from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult

app = FastAPI()

@app.post("/documents/process")
async def submit_document(request: ProcessRequest):
    task = process_document.delay(
        document_path=request.path,
        options=request.options
    )
    return {
        "task_id": task.id,
        "status": "submitted",
        "poll_url": f"/tasks/{task.id}"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    
    if result.status == 'PROCESSING':
        response["meta"] = result.info
    elif result.status == 'SUCCESS':
        response["result"] = result.result
    elif result.status == 'FAILURE':
        response["error"] = str(result.result)
    
    return response
```

## Batch Processing Pattern

For bulk operations — like processing a directory of invoices — I use Celery's group and chord primitives:

```python
from celery import group, chord

@app.task
def process_single_invoice(invoice_path: str) -> dict:
    result = vlm_extract(invoice_path)
    return {"path": invoice_path, "data": result}

@app.task
def aggregate_results(results: list[dict]) -> dict:
    successful = [r for r in results if r.get("data")]
    failed = [r for r in results if not r.get("data")]
    
    # Write to database, generate report, etc.
    save_to_db(successful)
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed)
    }

# Process 100 invoices in parallel, then aggregate
@app.post("/invoices/batch")
async def batch_process(invoice_paths: list[str]):
    workflow = chord(
        group(process_single_invoice.s(path) for path in invoice_paths),
        aggregate_results.s()
    )
    result = workflow.apply_async()
    return {"task_id": result.id}
```

## Error Handling and Retries

AI tasks fail in predictable ways — GPU OOM, model timeout, corrupt input files. Configure retries accordingly:

```python
@app.task(
    bind=True,
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=300,
    autoretry_for=(GPUOutOfMemoryError, TimeoutError),
    retry_jitter=True
)
def process_document(self, document_path: str, options: dict):
    try:
        return do_processing(document_path, options)
    except CorruptFileError:
        # Don't retry on bad input
        return {"status": "failed", "reason": "corrupt_file"}
```

## Monitoring

Flower gives you a real-time dashboard for Celery workers:

```bash
celery -A tasks flower --port=5555
```

For production, I also push task metrics to Prometheus:

```python
from prometheus_client import Counter, Histogram

TASK_COUNTER = Counter('ai_tasks_total', 'Total tasks', ['task_name', 'status'])
TASK_DURATION = Histogram('ai_task_duration_seconds', 'Task duration', ['task_name'])

@task_postrun.connect
def task_completed(sender=None, **kwargs):
    TASK_COUNTER.labels(
        task_name=sender.name,
        status=kwargs['state']
    ).inc()
```

The pattern is simple but it scales well. I've run this setup processing thousands of documents daily with a pool of 4 GPU workers. The key is treating AI inference like any other heavy compute — don't block on it, queue it, monitor it, and let it fail gracefully.
