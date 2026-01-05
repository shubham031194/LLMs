---
layout: post
title: "Serving LLMs at Scale: Building Streaming APIs with FastAPI and Pydantic"
date: 2026-01-05
tags: [fastapi, llm, streaming, sse, python]
read_time: 8
---

Once your LLM-powered feature works locally, the next challenge is serving it to users over HTTP. And users have expectations — they've used ChatGPT, they expect streaming responses, not a 30-second loading spinner followed by a wall of text.

Building a production streaming API taught me a few things the hard way.

## The Basic Streaming Endpoint

FastAPI supports Server-Sent Events (SSE) natively via `StreamingResponse`:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 1024
    temperature: float = 0.7

async def stream_llm_response(request: QueryRequest):
    async for chunk in llm.generate_stream(
        prompt=request.question,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    ):
        data = json.dumps({"token": chunk, "done": False})
        yield f"data: {data}\n\n"
    
    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

@app.post("/v1/chat/stream")
async def chat_stream(request: QueryRequest):
    return StreamingResponse(
        stream_llm_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Prevents nginx from buffering
        }
    )
```

The `X-Accel-Buffering: no` header is critical if you're behind nginx. Without it, nginx buffers the entire response and delivers it all at once, defeating the purpose of streaming.

## Async Generators and Backpressure

The naive implementation above has a problem: if the client disconnects mid-stream, the generator keeps running. You're burning GPU cycles generating tokens nobody will read.

```python
from fastapi import Request

@app.post("/v1/chat/stream")
async def chat_stream(request: QueryRequest, http_request: Request):
    async def generate():
        async for chunk in llm.generate_stream(
            prompt=request.question,
            max_tokens=request.max_tokens
        ):
            if await http_request.is_disconnected():
                logger.info("Client disconnected, stopping generation")
                break
            
            data = json.dumps({"token": chunk, "done": False})
            yield f"data: {data}\n\n"
        
        if not await http_request.is_disconnected():
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

## Request Validation That Works

Pydantic v2 is strict about types by default, which is what you want for an API:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "llama3-8b"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    stream: bool = True
    
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if msg.get("role") not in ("system", "user", "assistant"):
                raise ValueError(f"Invalid role: {msg.get('role')}")
            if not msg.get("content"):
                raise ValueError("message content cannot be empty")
        return v
```

## Concurrent Request Handling

A single LLM inference can take seconds. If you're handling multiple users, you need proper concurrency:

```python
import asyncio
from contextlib import asynccontextmanager

class InferencePool:
    def __init__(self, max_concurrent: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
    
    @asynccontextmanager
    async def acquire(self):
        await self.semaphore.acquire()
        self.active_requests += 1
        try:
            yield
        finally:
            self.active_requests -= 1
            self.semaphore.release()

pool = InferencePool(max_concurrent=4)

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async with pool.acquire():
            async for chunk in llm.generate_stream(
                messages=request.messages,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_requests": pool.active_requests,
        "max_concurrent": 4
    }
```

## Client-Side Consumption

On the frontend, consuming the stream:

```javascript
async function streamChat(question) {
  const response = await fetch('/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        appendToken(data.token);
      }
    }
  }
}
```

## Production Checklist

A few things I always configure before deploying:

- **Request timeouts** — Set a hard timeout on generation (60s is reasonable) to prevent runaway requests
- **Rate limiting** — Use `slowapi` or a reverse proxy to limit per-user request rates
- **Structured logging** — Log request ID, latency, token count, and model used for every request
- **Graceful shutdown** — Handle SIGTERM by finishing active streams before exiting

The streaming API pattern is now table stakes for any LLM-powered application. Getting it right from the start saves you from a painful migration later.
