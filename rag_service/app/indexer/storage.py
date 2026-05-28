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
import time
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

# Кеш результата ping Qdrant. Раньше get_client() делал get_collections()
# при КАЖДОМ вызове, чтобы убедиться что соединение живо. При пике 40 req/sec
# это +40 round-trip к Qdrant в секунду — заметная нагрузка.
# С кешем: если ping был успешен < _PING_CACHE_TTL сек назад, считаем клиент
# живым без обращения к Qdrant.
_PING_CACHE_TTL = 30.0       # секунды
_last_ping_ok_at: float = 0  # monotonic timestamp последнего успешного ping


# ── State management (hot-swap) ───────────────────────────────────────────────

# Лёгкий кеш чтения hot_swap_state.json. get_active_collection() вызывается
# из get_collection_stats(), get_indexing_status() и других мест — при пиковом
# RPS это становится постоянным disk I/O.
# TTL 5 сек: новая активная коллекция распространится максимум за 5 сек
# после _atomic_swap (а сам swap делает 30+ сек индексации до этого).
_ACTIVE_COLLECTION_TTL = 5.0
_active_collection_cache: tuple[float, str] = (0.0, "")  # (ts, value)
_active_collection_lock = threading.Lock()


def get_active_collection() -> str:
    """
    Возвращает имя активной коллекции Qdrant.

    После первого hot-swap читает из hot_swap_state.json.
    До первого — возвращает settings.QDRANT_COLLECTION_NAME.

    Результат кешируется на _ACTIVE_COLLECTION_TTL секунд чтобы не дёргать
    файл при каждом запросе.
    """
    global _active_collection_cache
    now = time.monotonic()
    with _active_collection_lock:
        ts, cached = _active_collection_cache
        if cached and (now - ts) < _ACTIVE_COLLECTION_TTL:
            return cached

    value = settings.QDRANT_COLLECTION_NAME
    try:
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            col = state.get("active_collection", "")
            if col:
                value = col
    except Exception:
        pass

    with _active_collection_lock:
        _active_collection_cache = (now, value)
    return value


def _invalidate_active_collection_cache() -> None:
    """Сбрасывает кеш — вызывается при set_active_collection после hot-swap."""
    global _active_collection_cache
    with _active_collection_lock:
        _active_collection_cache = (0.0, "")


def set_active_collection(name: str) -> None:
    """
    Атомарно обновляет имя активной коллекции в файле состояния.
    Использует write-to-tmp + rename для атомарности.
    Сохраняет другие поля (last_indexed_at и т.д.) нетронутыми.

    Сбрасывает in-process кеш get_active_collection чтобы следующий вызов
    сразу увидел новое значение (а не ждал TTL).
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
    _invalidate_active_collection_cache()


# Обратная совместимость: приватный алиас для кода который мог импортировать _set_active_collection
_set_active_collection = set_active_collection


def get_last_indexed_at() -> Optional[datetime]:
    """
    Возвращает время последней успешной полной индексации как наивный datetime в UTC.
    Используется pipeline.py для lastmod-фильтрации: страницы не изменившиеся
    с последней индексации можно не скачивать вообще.

    Возвращает datetime без tzinfo (UTC) или None если ещё не индексировалось.

    ВАЖНО про timezone:
      set_last_indexed_at сохраняет datetime.now(timezone.utc).isoformat() —
      это UTC-aware строка вида "2025-06-15T05:00:00+00:00".
      Старый код делал dt.replace(tzinfo=None) — это НЕ конвертировало время,
      а лишь снимало метку зоны. На сервере UTC+5 значение "10:00+05:00" превращалось
      в "10:00" без зоны, а должно было стать "05:00".
      Правильный путь: сначала перевести в UTC, потом снять метку.
    """
    try:
        if _STATE_FILE.exists():
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            ts = state.get("last_indexed_at", "")
            if ts:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is not None:
                    # Aware datetime: конвертируем в UTC, затем снимаем tzinfo
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                # Если naive — считаем что уже UTC (обратная совместимость)
                return dt
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

    Оптимизация ping (двухфазная):
      Фаза 1 — БЕЗ lock'а: читаем _last_ping_ok_at атомарно (float, GIL).
        Если кеш свежий — возвращаем клиент немедленно, не блокируем никого.
      Фаза 2 — ПОД lock'ом: double-check + реальный ping только если нужен.

      Проблема старого кода: ping (get_collections, timeout=10с) выполнялся
      под _client_lock. При 20 одновременных потоках threadpool — 19 из 20
      ждали в очереди до 10 сек, пока первый делал ping. Это сериализовало
      весь поиск при каждом протухании кеша (раз в 30 сек).
    """
    global _client, _last_ping_ok_at

    # ── Фаза 1: быстрая проверка БЕЗ lock'а ─────────────────────
    # float-чтение в CPython атомарно (GIL), поэтому безопасно читать без лока.
    # Большинство вызовов (когда кеш свежий) возвращаются здесь мгновенно.
    now = time.monotonic()
    if _client is not None and (now - _last_ping_ok_at) < _PING_CACHE_TTL:
        return _client

    # ── Фаза 2: требуется создание или ping — берём lock ─────────
    with _client_lock:
        # double-check: пока ждали lock, другой поток мог уже пингануть
        now = time.monotonic()
        if _client is not None and (now - _last_ping_ok_at) < _PING_CACHE_TTL:
            return _client

        if _client is None:
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )
            _last_ping_ok_at = time.monotonic()
            logger.info("[Qdrant] Client created")
            return _client

        # Кеш протух — делаем реальный ping (только ОДИН поток, остальные ждут lock)
        try:
            _client.get_collections()
            _last_ping_ok_at = time.monotonic()
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
            _last_ping_ok_at = time.monotonic()
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
        for field, schema in [("page_url", "keyword"), ("manually_edited", "bool")]:
            try:
                client.create_payload_index(
                    collection_name=_col,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info(f"[Qdrant] Payload index created on '{field}'")
            except Exception as e:
                logger.warning(f"[Qdrant] Could not create payload index on '{field}' (non-critical): {e}")
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


def save_chunks_precomputed(
    chunks: List[TextChunk],
    vectors: List[List[float]],
    page_url: str,
    content_hash: str,
    collection_name: Optional[str] = None,
) -> int:
    """
    Сохраняет чанки с уже вычисленными векторами в Qdrant.

    Используется pipeline.py который вычисляет эмбеддинги для всех страниц
    сразу одним батчем, а потом сохраняет постранично через эту функцию.

    В отличие от save_chunks(), здесь нет вызова get_embeddings_batch() —
    векторы переданы снаружи.

    Args:
        chunks:          Список TextChunk (текст + метаданные).
        vectors:         Список векторов — len(vectors) == len(chunks).
        page_url:        URL источника (для удаления старых чанков).
        content_hash:    MD5 контента (сохраняется в payload для change detection).
        collection_name: Коллекция Qdrant. По умолчанию из settings.

    Returns:
        Количество сохранённых точек.
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    if not chunks:
        return 0

    if len(chunks) != len(vectors):
        logger.error(
            f"[Storage] save_chunks_precomputed: "
            f"{len(chunks)} chunks but {len(vectors)} vectors — skipping {page_url}"
        )
        return 0

    # Дедупликация (та же логика что в save_chunks)
    seen_texts: set = set()
    unique_pairs = []
    for chunk, vec in zip(chunks, vectors):
        key = (chunk.embed_text or chunk.text).strip()
        if key not in seen_texts:
            seen_texts.add(key)
            unique_pairs.append((chunk, vec))

    duplicates = len(chunks) - len(unique_pairs)
    if duplicates > 0:
        logger.info(f"[Storage] Removed {duplicates} duplicate chunks for {page_url}")

    client = get_client()
    delete_chunks_by_url(page_url, collection_name=_col)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
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
        )
        for chunk, vec in unique_pairs
    ]

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
