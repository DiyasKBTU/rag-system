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
import json
from pathlib import Path
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

# Файл состояния hot-swap: хранит имя активной коллекции
# Путь: rag_service/hot_swap_state.json
_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "hot_swap_state.json"

# Initialize Qdrant client (connects to local Docker container)
_client: Optional[QdrantClient] = None


# ── State management (hot-swap) ───────────────────────────────────────────────

def get_active_collection() -> str:
    """
    Возвращает имя активной коллекции Qdrant.

    После первого hot-swap читает из hot_swap_state.json.
    До первого — возвращает settings.QDRANT_COLLECTION_NAME.
    """
    try:
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            col = state.get("active_collection", "")
            if col:
                return col
    except Exception:
        pass
    return settings.QDRANT_COLLECTION_NAME


def _set_active_collection(name: str) -> None:
    """
    Атомарно обновляет имя активной коллекции в файле состояния.
    Использует write-to-tmp + rename для атомарности.
    """
    tmp = _STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"active_collection": name}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(_STATE_FILE)


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


def ensure_collection_exists(collection_name: Optional[str] = None) -> None:
    """
    Create Qdrant collection if it doesn't exist yet.
    Called once at startup, and before shadow indexing.

    Collection settings:
    - vectors: 1536 dimensions (matches text-embedding-3-small)
    - distance: Cosine similarity (standard for text embeddings)

    Args:
        collection_name: Name of collection to create. Defaults to settings value.

    Note:
        После первого hot-swap settings.QDRANT_COLLECTION_NAME является алиасом,
        а не реальной коллекцией. В этом случае создание пропускается — алиас уже
        указывает на существующую blue/green коллекцию.
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    client = get_client()

    # Проверяем: если это уже алиас — создавать ничего не нужно
    try:
        aliases = [a.alias_name for a in client.get_aliases().aliases]
        if _col in aliases:
            print(f"[Qdrant] '{_col}' is an alias — skipping collection creation")
            return
    except Exception:
        pass  # Если get_aliases() не поддерживается — идём дальше

    existing = [c.name for c in client.get_collections().collections]

    if _col not in existing:
        print(f"[Qdrant] Creating collection: {_col}")
        client.create_collection(
            collection_name=_col,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Collection created successfully")
    else:
        print(f"[Qdrant] Collection already exists: {_col}")


def save_chunks(chunks: List[TextChunk], page_url: str, content_hash: str,
                collection_name: Optional[str] = None) -> int:
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
        collection_name: Target collection. Defaults to settings value.

    Returns:
        Number of chunks saved
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    if not chunks:
        return 0

    # ── Дедупликация: убираем чанки с одинаковым текстом ──────
    # Некоторые страницы повторяют один и тот же HTML-блок несколько раз
    # (мобильная/десктопная версия, аккордеон-компоненты и т.д.)
    seen_texts: set = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.embed_text or chunk.text).strip()
        if key not in seen_texts:
            seen_texts.add(key)
            unique_chunks.append(chunk)

    duplicates = len(chunks) - len(unique_chunks)
    if duplicates > 0:
        import logging
        logging.getLogger(__name__).info(
            f"[Storage] Removed {duplicates} duplicate chunks for {page_url}"
        )
    chunks = unique_chunks

    client = get_client()

    # ── Step 1: Delete old chunks for this URL ─────────────────
    # This ensures we don't have duplicates if page is re-indexed
    delete_chunks_by_url(page_url, collection_name=_col)

    # ── Step 2: Create vectors for all chunks at once ──────────
    # For question-chunks: embed the question (embed_text), not the body text.
    # This way the question vector is matched at search time,
    # but payload["text"] still holds the original chunk for GPT context.
    texts_for_embedding = [
        chunk.embed_text if chunk.embed_text else chunk.text
        for chunk in chunks
    ]
    print(f"  [Qdrant] Creating embeddings for {len(texts_for_embedding)} chunks...")
    vectors = get_embeddings_batch(texts_for_embedding)

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
                "text": chunk.text,                     # actual chunk text
                "page_url": chunk.page_url,             # source page URL
                "page_title": chunk.page_title,         # page title
                "section_title": chunk.section_title,   # section heading (h2/h3)
                "chunk_index": chunk.index,             # position on page
                "content_hash": content_hash,           # page hash (for updates)
            },
        )
        points.append(point)

    # ── Step 4: Save to Qdrant in one batch ───────────────────
    client.upsert(
        collection_name=_col,
        points=points,
    )

    print(f"  [Qdrant] Saved {len(points)} chunks for: {page_url}")
    return len(points)


def delete_chunks_by_url(page_url: str, collection_name: Optional[str] = None) -> None:
    """
    Delete all chunks belonging to a specific URL.
    Called before re-indexing a page to avoid duplicates.

    Args:
        page_url: URL whose chunks to delete.
        collection_name: Target collection. Defaults to settings value.
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    client = get_client()

    client.delete(
        collection_name=_col,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="page_url",
                    match=MatchValue(value=page_url),
                )
            ]
        ),
    )


def get_collection_stats(collection_name: Optional[str] = None) -> dict:
    """
    Get stats about the current collection.
    Used to monitor how many chunks are stored.

    Args:
        collection_name: Collection to query. Defaults to active collection.
    """
    _col = collection_name or get_active_collection()
    client = get_client()

    try:
        info = client.get_collection(_col)
        return {
            "total_chunks": info.points_count,
            "collection": _col,
            "status": info.status,
        }
    except Exception:
        return {
            "total_chunks": 0,
            "collection": _col,
            "status": "not_found",
        }


def page_needs_update(page_url: str, new_hash: str,
                      collection_name: Optional[str] = None) -> bool:
    """
    Check if a page needs to be re-indexed.
    Compares the stored hash with the new hash.

    Args:
        page_url: URL to check.
        new_hash: Current MD5 hash of page content.
        collection_name: Collection to check. Defaults to settings value.

    Returns:
        True  - page changed, needs re-indexing
        False - page unchanged, skip it
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    client = get_client()

    # Search for any chunk from this URL
    results = client.scroll(
        collection_name=_col,
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
