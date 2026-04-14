# -*- coding: utf-8 -*-
"""
storage.py - Saves text chunks + vectors to Qdrant.

What is Qdrant?
  A vector database - stores both text AND its vector representation.
  Allows fast similarity search: "find chunks most similar to this question".

How data is organized in Qdrant:
  Collection = like a table in SQL
    Point = one row in that table
      - id: unique identifier (UUID)
      - vector: list of 1536 numbers
      - payload: metadata (text, url, title, hash, chunk_index)

Flow:
  TextChunk -> get_embedding() -> save to Qdrant as Point
"""

import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings
from app.parser.chunker import TextChunk
from app.indexer.embeddings import get_embeddings_batch


# Initialize Qdrant client (connects to local Docker container)
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """
    Get Qdrant client (create once, reuse).
    Connects to Qdrant running in Docker on localhost:6333.
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
    return _client


def ensure_collection_exists() -> None:
    """
    Create Qdrant collection if it doesn't exist yet.
    Called once at startup.

    Collection settings:
    - vectors: 1536 dimensions (matches text-embedding-3-small)
    - distance: Cosine similarity (standard for text embeddings)
    """
    client = get_client()

    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION_NAME not in existing:
        print(f"[Qdrant] Creating collection: {settings.QDRANT_COLLECTION_NAME}")
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Collection created successfully")
    else:
        print(f"[Qdrant] Collection already exists: {settings.QDRANT_COLLECTION_NAME}")


def save_chunks(chunks: List[TextChunk], page_url: str, content_hash: str) -> int:
    """
    Save list of chunks to Qdrant.

    Steps:
    1. Delete old chunks for this URL (if page was re-indexed)
    2. Create embeddings for all chunks in one batch
    3. Save to Qdrant

    Args:
        chunks: List of TextChunk objects from chunker
        page_url: URL of source page (used to delete old chunks)
        content_hash: MD5 hash of page content (for change detection)

    Returns:
        Number of chunks saved
    """
    if not chunks:
        return 0

    client = get_client()

    # ── Step 1: Delete old chunks for this URL ─────────────────
    # This ensures we don't have duplicates if page is re-indexed
    delete_chunks_by_url(page_url)

    # ── Step 2: Create vectors for all chunks at once ──────────
    texts = [chunk.text for chunk in chunks]
    print(f"  [Qdrant] Creating embeddings for {len(texts)} chunks...")
    vectors = get_embeddings_batch(texts)

    if len(vectors) != len(chunks):
        print(f"  [!] Mismatch: {len(chunks)} chunks but {len(vectors)} vectors")
        return 0

    # ── Step 3: Build Qdrant points ────────────────────────────
    points = []
    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            # Unique ID for each point (random UUID)
            id=str(uuid.uuid4()),

            # The vector (1536 numbers representing chunk meaning)
            vector=vector,

            # Metadata stored alongside the vector
            # This is what we return in search results
            payload={
                "text": chunk.text,           # actual chunk text
                "page_url": chunk.page_url,   # source page URL
                "page_title": chunk.page_title, # page title
                "chunk_index": chunk.index,   # position on page
                "content_hash": content_hash, # page hash (for updates)
            },
        )
        points.append(point)

    # ── Step 4: Save to Qdrant in one batch ───────────────────
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points,
    )

    print(f"  [Qdrant] Saved {len(points)} chunks for: {page_url}")
    return len(points)


def delete_chunks_by_url(page_url: str) -> None:
    """
    Delete all chunks belonging to a specific URL.
    Called before re-indexing a page to avoid duplicates.
    """
    client = get_client()

    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="page_url",
                    match=MatchValue(value=page_url),
                )
            ]
        ),
    )


def get_collection_stats() -> dict:
    """
    Get stats about the current collection.
    Used to monitor how many chunks are stored.
    """
    client = get_client()

    try:
        info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
        return {
            "total_chunks": info.points_count,
            "collection": settings.QDRANT_COLLECTION_NAME,
            "status": info.status,
        }
    except Exception:
        return {
            "total_chunks": 0,
            "collection": settings.QDRANT_COLLECTION_NAME,
            "status": "not_found",
        }


def page_needs_update(page_url: str, new_hash: str) -> bool:
    """
    Check if a page needs to be re-indexed.
    Compares the stored hash with the new hash.

    Returns:
        True  - page changed, needs re-indexing
        False - page unchanged, skip it
    """
    client = get_client()

    # Search for any chunk from this URL
    results = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="page_url",
                    match=MatchValue(value=page_url),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    points, _ = results

    if not points:
        # No chunks for this URL - definitely needs indexing
        return True

    # Compare hashes
    stored_hash = points[0].payload.get("content_hash", "")
    return stored_hash != new_hash
