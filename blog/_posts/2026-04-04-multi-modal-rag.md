---
layout: post
title: "Multi-Modal RAG: Querying Across Text, Images, and Video"
date: 2026-04-04
tags: [rag, multi-modal, embeddings, clip, video]
read_time: 8
---

Most RAG systems only work with text. But production data is messier — technical docs contain diagrams, product catalogs have images, training materials include video. If your retrieval system can only search text, you're ignoring a significant chunk of available knowledge.

Multi-modal RAG extends retrieval to work across content types using unified embedding spaces.

## Unified Embeddings with CLIP

The foundation is a model that maps different modalities into the same vector space. CLIP (and its descendants) can embed both text and images such that semantically similar content clusters together regardless of type.

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

# Text and images land in the same embedding space
text_embedding = model.encode("network architecture diagram with load balancer")
image_embedding = model.encode(Image.open("architecture.png"))

# These can be compared directly
similarity = cosine_similarity(text_embedding, image_embedding)
```

This means a text query like "show me the system architecture" can retrieve a diagram even if that diagram has no alt text or caption.

## The Ingestion Pipeline

Processing multi-modal documents requires type-specific handling during ingestion:

```python
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"

@dataclass
class MultiModalChunk:
    content_type: ContentType
    text: str | None
    image_path: str | None
    embedding: list[float]
    metadata: dict

def ingest_document(doc_path: str) -> list[MultiModalChunk]:
    chunks = []
    
    # Extract text chunks
    text_chunks = extract_text_chunks(doc_path)
    for chunk in text_chunks:
        embedding = text_model.encode(chunk.text)
        chunks.append(MultiModalChunk(
            content_type=ContentType.TEXT,
            text=chunk.text,
            image_path=None,
            embedding=embedding,
            metadata=chunk.metadata
        ))
    
    # Extract and embed images
    images = extract_images(doc_path)
    for img in images:
        embedding = clip_model.encode(Image.open(img.path))
        
        # Also generate a text description for hybrid search
        description = vlm.describe(img.path)
        
        chunks.append(MultiModalChunk(
            content_type=ContentType.IMAGE,
            text=description,
            image_path=img.path,
            embedding=embedding,
            metadata={"page": img.page, "position": img.position}
        ))
    
    return chunks
```

## Video: Frame Sampling Strategy

You can't embed every frame of a video — a 10-minute video at 30fps is 18,000 frames. The approach that works is intelligent sampling:

```python
import cv2
from skimage.metrics import structural_similarity

def extract_key_frames(video_path: str, similarity_threshold: float = 0.85) -> list:
    cap = cv2.VideoCapture(video_path)
    key_frames = []
    prev_frame = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every 30th frame as candidate
        if frame_count % 30 != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            similarity = structural_similarity(prev_frame, gray)
            if similarity < similarity_threshold:
                # Visual change detected — this is a key frame
                key_frames.append({
                    "frame": frame,
                    "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                    "frame_number": frame_count
                })
        else:
            key_frames.append({
                "frame": frame,
                "timestamp": 0,
                "frame_number": 0
            })
        
        prev_frame = gray
    
    cap.release()
    return key_frames
```

This captures frames where something visually changes — a new slide, a scene transition, a diagram appearing — and skips the static portions.

## Retrieval and Response

At query time, the search is modality-agnostic:

```python
class MultiModalRetriever:
    def __init__(self, vector_store, text_model, clip_model):
        self.store = vector_store
        self.text_model = text_model
        self.clip_model = clip_model
    
    def search(self, query: str, top_k: int = 10) -> list[MultiModalChunk]:
        # Embed query in both spaces
        text_emb = self.text_model.encode(query)
        clip_emb = self.clip_model.encode(query)
        
        # Search both and merge results
        text_results = self.store.search(text_emb, top_k=top_k, collection="text")
        visual_results = self.store.search(clip_emb, top_k=top_k, collection="visual")
        
        # Reciprocal rank fusion to merge
        merged = reciprocal_rank_fusion(text_results, visual_results)
        return merged[:top_k]
```

When passing multi-modal results to the LLM, images go in as base64-encoded content alongside text chunks. The VLM can then reference both text and visual content in its response.

## Practical Considerations

**Storage costs increase significantly.** Image embeddings are the same dimensionality as text, but you're storing more chunks. Budget for 3-5x the storage of a text-only system.

**Latency is higher.** CLIP encoding is slower than text embedding. Consider a tiered approach: text-only search for simple queries, multi-modal search only when the query mentions visual content or when text results are insufficient.

The biggest win I've seen from multi-modal RAG is in technical documentation — being able to ask "how do I configure the load balancer" and getting back both the configuration guide *and* the architecture diagram that shows where the load balancer sits. Context that combines both modalities is significantly more useful than either alone.
