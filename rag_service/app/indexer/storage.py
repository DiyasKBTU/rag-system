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

import logging
import threading
import uuid
import json
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)

# Файл состояния hot-swap: хранит имя активной коллекции
# Путь: rag_service/hot_swap_state.json
_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "hot_swap_state.json"

# Initialize Qdrant client (connects to local Docker container)
_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


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
    Сохраняет другие поля (last_indexed_at и т.д.) нетронутыми.
    """
    try:
        state: dict = {}
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    state["active_collection"] = name
    tmp = _STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_STATE_FILE)


def get_last_indexed_at() -> Optional[datetime]:
    """
    Возвращает время последней успешной полной индексации.
    Используется pipeline.py для lastmod-фильтрации: страницы не изменившиеся
    с последней индексации можно не скачивать вообще.

    Возвращает datetime без tzinfo (UTC) или None если ещё не индексировалось.
    """
    try:
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            ts = state.get("last_indexed_at", "")
            if ts:
                dt = datetime.fromisoformat(ts)
                # Убираем timezone для сравнения с lastmod (у lastmod нет tz)
                return dt.replace(tzinfo=None)
    except Exception:
        pass
    return None


def set_last_indexed_at(dt: Optional[datetime] = None) -> None:
    """
    Записывает время последней успешной полной индексации.
    Вызывается pipeline.py в конце run_indexing().

    Атомарная запись — не затирает другие поля файла состояния.
    """
    dt = dt or datetime.now(timezone.utc)
    try:
        state: dict = {}
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        state["last_indexed_at"] = dt.isoformat()
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_STATE_FILE)
        logger.info(f"[Storage] last_indexed_at saved: {dt.isoformat()}")
    except Exception as e:
        logger.warning(f"[Storage] Could not save last_indexed_at: {e}")


def get_client() -> QdrantClient:
    """
    Get Qdrant client (create once, reuse, auto-reconnect).

    Почему нужен reconnect:
      Клиент создаётся один раз как синглтон. Если Docker с Qdrant
      рестартовал или временно упал — HTTP-соединение в пуле протухает.
      Следующий вызов .search() бросает исключение, которое в search.py
      молча поглощается → пустые результаты → GPT «не знает».

    Решение: перед возвратом делаем лёгкий ping (get_collections).
      Если ping упал — сбрасываем синглтон и создаём новый клиент.
      Lock гарантирует что два потока не пересоздадут клиент одновременно.
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )
            logger.info("[Qdrant] Client created")
            return _client

        # Лёгкий health check — просто проверяем что соединение живо
        try:
            _client.get_collections()
            return _client
        except Exception as e:
            logger.warning(f"[Qdrant] Ping failed ({e}), reconnecting...")
            try:
                _client.close()
            except Exception:
                pass
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )
            logger.info("[Qdrant] Client reconnected")
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
            logger.info(f"[Qdrant] '{_col}' is an alias — skipping collection creation")
            return
    except Exception:
        pass  # Если get_aliases() не поддерживается — идём дальше

    existing = [c.name for c in client.get_collections().collections]

    if _col not in existing:
        logger.info(f"[Qdrant] Creating collection: {_col}")
        client.create_collection(
            collection_name=_col,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        # Payload index на page_url: ускоряет delete_chunks_by_url() и page_needs_update().
        # Без индекса Qdrant сканирует все ~2000 точек при каждом фильтре по URL.
        # С индексом — O(log n). Особенно важно при параллельной записи в pipeline.
        try:
            client.create_payload_index(
                collection_name=_col,
                field_name="page_url",
                field_schema="keyword",
            )
            logger.info("[Qdrant] Payload index created on 'page_url'")
        except Exception as e:
            logger.warning(f"[Qdrant] Could not create payload index (non-critical): {e}")
        logger.info("[Qdrant] Collection created successfully")
    else:
        logger.debug(f"[Qdrant] Collection already exists: {_col}")


def save_chunks(chunks: List[TextChunk], page_url: str, content_hash: str,
                collection_name: Optional[str] = None) -> int:
    """
    Save list of chunks to Qdrant.

    Steps:
    1. Deduplicate chunks with identical text
    2. Delete old chunks for this URL (if page was re-indexed)
    3. Create embeddings for all chunks in one batch
    4. Save to Qdrant

    Args:
        chunks:          List of TextChunk objects from chunker
        page_url:        URL of source page (used to delete old chunks)
        content_hash:    MD5 hash of page content (for change detection)
        collection_name: Target collection. Defaults to settings value.

    Returns:
        Number of chunks saved
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    if not chunks:
        return 0

    # ── Дедупликация: убираем чанки с одинаковым текстом ──────────
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
        logger.info(f"[Storage] Removed {duplicates} duplicate chunks for {page_url}")
    chunks = unique_chunks

    client = get_client()

    # ── Step 1: Delete old chunks for this URL ─────────────────────
    # This ensures we don't have duplicates if page is re-indexed
    delete_chunks_by_url(page_url, collection_name=_col)

    # ── Step 2: Create vectors for all chunks at once ───────────────
    # For question-chunks: embed the question (embed_text), not the body text.
    # This way the question vector is matched at search time,
    # but payload["text"] still holds the original chunk for GPT context.
    texts_for_embedding = [
        chunk.embed_text if chunk.embed_text else chunk.text
        for chunk in chunks
    ]
    logger.info(f"[Storage] Creating embeddings for {len(texts_for_embedding)} chunks ({page_url})")
    vectors = get_embeddings_batch(texts_for_embedding)

    if len(vectors) != len(chunks):
        logger.error(
            f"[Storage] Mismatch: {len(chunks)} chunks but {len(vectors)} vectors — skipping {page_url}"
        )
        return 0

    # ── Step 3: Build Qdrant points ────────────────────────────────
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":           chunk.text,
                "embed_text":     chunk.embed_text or "",
                "page_url":       chunk.page_url,
                "page_title":     chunk.page_title,
                "section_title":  chunk.section_title,
                "chunk_index":    chunk.index,
                "content_hash":   content_hash,
                "external_links": getattr(chunk, "external_links", []) or [],
            },
        ))

    # ── Step 4: Save to Qdrant in one batch ────────────────────────
    client.upsert(collection_name=_col, points=points)
    logger.info(f"[Storage] Saved {len(points)} chunks for: {page_url}")
    return len(points)


def delete_chunks_by_url(page_url: str, collection_name: Optional[str] = None) -> None:
    """
    Delete all chunks belonging to a specific URL.
    Called before re-indexing a page to avoid duplicates.

    Args:
        page_url:        URL whose chunks to delete.
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
            "collection":   _col,
            "status":       info.status,
        }
    except Exception:
        return {
            "total_chunks": 0,
            "collection":   _col,
            "status":       "not_found",
        }


def url_is_manually_edited(page_url: str,
                           collection_name: Optional[str] = None) -> bool:
    """
    Проверяет, все ли чанки для данного URL созданы вручную через chunk_editor.

    Возвращает True если:
      - Для URL есть чанки в Qdrant
      - ВСЕ из них имеют флаг manually_edited=True

    Возвращает False если:
      - Нет чанков вообще (новая страница → нужна индексация)
      - Хотя бы один чанк без флага (авто-индексированная страница)

    Используется в pipeline.py чтобы не затирать ручные правки при переиндексации.
    """
    _col = collection_name or get_active_collection()
    client = get_client()

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
        limit=200,  # Достаточно для любой страницы (обычно <20 чанков)
        with_payload=True,
        with_vectors=False,
    )

    points, _ = results

    if not points:
        return False  # Нет чанков — новая страница, индексируем

    return all(p.payload.get("manually_edited") is True for p in points)


def page_needs_update(page_url: str, new_hash: str,
                      collection_name: Optional[str] = None) -> bool:
    """
    Check if a page needs to be re-indexed.
    Compares the stored hash with the new hash.

    Args:
        page_url:        URL to check.
        new_hash:        Current MD5 hash of page content.
        collection_name: Collection to check. Defaults to settings value.

    Returns:
        True  - page changed, needs re-indexing
        False - page unchanged, skip it
    """
    _col = collection_name or get_active_collection()
    client = get_client()

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
        return True  # No chunks for this URL — needs indexing

    stored_hash = points[0].payload.get("content_hash", "")
    return stored_hash != new_hash
