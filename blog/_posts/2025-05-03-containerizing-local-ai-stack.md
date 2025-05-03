---
layout: post
title: "Containerizing the Local AI Stack: Docker Compose for Vector DBs & Local Inference"
date: 2025-05-03
tags: [docker, devops, vector-db, local-inference]
read_time: 7
---

One of the first roadblocks I hit when building RAG prototypes was the "works on my machine" problem. Qdrant needed one setup, the embedding model server needed another, and the LLM inference endpoint had its own set of dependencies. Every time I switched machines or onboarded someone, it was a half-day of debugging.

The solution was obvious in hindsight — containerize everything with Docker Compose.

## The Stack

Here's what a typical local AI development environment looks like for me:

- **Qdrant** — vector database for storing and querying embeddings
- **Ollama** — local LLM inference (Llama 3, Mistral, etc.)
- **FastAPI app** — the actual RAG application layer

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - qdrant
      - ollama
    volumes:
      - ./src:/app/src

volumes:
  qdrant_data:
  ollama_models:
```

## Key Design Decisions

**Named volumes for persistence.** Without `qdrant_data`, you'd lose your entire vector index every time you restart. Without `ollama_models`, you'd re-download multi-gigabyte model files on every `docker compose up`. This seems obvious but I've seen it missed too many times.

**GPU passthrough for Ollama.** The `deploy.resources` block is critical. Without it, Ollama falls back to CPU inference and a 7B model goes from 30 tokens/sec to 3. If you're on a machine without a GPU, just remove the deploy block — it'll still work, just slower.

**Service discovery via container names.** Inside the Docker network, `qdrant` resolves to the Qdrant container. No need to hardcode IPs or use `host.docker.internal`.

## Model Bootstrapping

One thing Docker Compose doesn't handle well is pulling the initial Ollama model. I added a simple init script:

```bash
#!/bin/bash
# init_models.sh
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done

echo "Pulling models..."
curl -X POST http://localhost:11434/api/pull \
  -d '{"name": "llama3:8b"}'

curl -X POST http://localhost:11434/api/pull \
  -d '{"name": "nomic-embed-text"}'

echo "Models ready."
```

## Health Checks

Adding health checks prevents the app container from starting before dependencies are actually ready:

```yaml
qdrant:
  # ...
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
    interval: 10s
    timeout: 5s
    retries: 5

app:
  # ...
  depends_on:
    qdrant:
      condition: service_healthy
```

## What This Gives You

The entire stack comes up with a single command, tears down cleanly, and works identically across machines. New team member? `git clone && docker compose up`. That's it.

For production, you'd obviously swap Ollama for a proper inference server like vLLM or TGI, and likely use a managed vector DB. But for development and prototyping, this setup has saved me more time than probably anything else I've done this year.
