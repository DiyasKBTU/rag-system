# -*- coding: utf-8 -*-
"""
hot_swap.py — Blue/Green деплой для Qdrant без перезапуска сервиса.

Как работает:
  Qdrant поддерживает алиасы — «виртуальные» имена которые указывают на реальные коллекции.
  Алиас можно переключить атомарно: в один момент он указывает на старую коллекцию,
  в следующий — на новую. Бот продолжает работать через алиас без перебоев.

Две реальные коллекции (blue/green):
  caiu_knowledge_base_blue
  caiu_knowledge_base_green

Один алиас (то что использует поиск):
  caiu_knowledge_base  →  указывает на активную (blue или green)

Пример первого запуска (миграция):
  До:   реальная коллекция  caiu_knowledge_base (старый формат без горячей замены)
  После: алиас             caiu_knowledge_base → caiu_knowledge_base_blue

Пример последующих свопов:
  Активна blue → индексируем green → swap: алиас переключается на green → blue стирается
  Активна green → индексируем blue → swap: алиас переключается на blue → green стирается

Использование:
  from app.indexer.hot_swap import run_shadow_indexing, get_indexing_status

  # Запустить в отдельном потоке:
  import threading
  t = threading.Thread(target=run_shadow_indexing, daemon=True)
  t.start()
"""

import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import portalocker

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    PointStruct,
)

from app.config import settings
from app.indexer.storage import (
    get_client,
    get_active_collection,
    set_active_collection,
    ensure_collection_exists,
    get_collection_stats,
)
from app.indexer.pipeline import run_indexing

logger = logging.getLogger(__name__)

# ── Константы имён коллекций ──────────────────────────────────────────────────
COLLECTION_BLUE  = f"{settings.QDRANT_COLLECTION_NAME}_blue"
COLLECTION_GREEN = f"{settings.QDRANT_COLLECTION_NAME}_green"

# ── Двухуровневая блокировка индексации ──────────────────────────────────────
# Уровень 1 (threading.Lock) — защищает от двух потоков ВНУТРИ одного процесса.
# Уровень 2 (portalocker, файловый lock) — защищает от двух ПРОЦЕССОВ uvicorn
#     (workers=2 в run_api.py запускает два независимых интерпретатора, в каждом
#      из них crontab-loop и /index endpoint).
# Файловый lock работает поверх ОС-вызовов (Windows: LockFileEx; Unix: flock),
# виден всем процессам на этой машине → гарантирует ровно одну индексацию.
_indexing_lock    = threading.Lock()
_INDEXING_LOCK_FILE = Path(__file__).resolve().parent.parent.parent / "indexing.lock"
_indexing_active  = False       # True пока идёт индексация (для текущего процесса)
_last_result: Optional[dict] = None      # Результат последней индексации
_started_at: Optional[str]   = None      # ISO timestamp старта


def _try_acquire_file_lock():
    """
    Пытается захватить межпроцессный файловый lock.
    Возвращает file handle если успешно, None если уже захвачен другим процессом.
    Handle нужно держать открытым на всё время индексации;
    при close() или process exit — lock освобождается автоматически.
    """
    try:
        _INDEXING_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Открываем в режиме "r+" если файл есть, иначе создаём.
        # portalocker.LOCK_EX | LOCK_NB — эксклюзивный non-blocking lock.
        fh = open(_INDEXING_LOCK_FILE, "a+")
        try:
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            return fh
        except portalocker.exceptions.LockException:
            fh.close()
            return None
    except Exception as e:
        # Если файловый lock сломался по какой-то причине (права, FS) —
        # лучше всё равно НЕ запускать индексацию, чем стартовать без защиты.
        logger.error(f"[HotSwap] File lock error: {e}")
        return None


def _release_file_lock(fh) -> None:
    """Освобождает файловый lock. Безопасно вызывать с None."""
    if fh is None:
        return
    try:
        portalocker.unlock(fh)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def get_shadow_collection_name() -> str:
    """
    Возвращает имя теневой (неактивной) коллекции.
    Если сейчас активна blue — возвращает green, и наоборот.
    Если активна оригинальная коллекция (ещё не была горячая замена) — возвращает blue.
    """
    active = get_active_collection()
    if active == COLLECTION_BLUE:
        return COLLECTION_GREEN
    return COLLECTION_BLUE


def _collection_is_alias(collection_name: str) -> bool:
    """
    Проверяет, является ли коллекция алиасом (а не реальной коллекцией).
    Нужно чтобы понять, нужна ли первая миграция.
    """
    client = get_client()
    try:
        aliases_response = client.get_aliases()
        for alias in aliases_response.aliases:
            if alias.alias_name == collection_name:
                return True
        return False
    except Exception:
        return False


def _first_migration(shadow_collection: str) -> None:
    """
    Первая миграция: оригинальная реальная коллекция → blue/green + алиас.

    До:
      caiu_knowledge_base (реальная коллекция, ~2000 чанков)

    После:
      caiu_knowledge_base_blue (новая реальная коллекция со свежими данными)
      caiu_knowledge_base      (алиас → blue)

    Старая реальная коллекция caiu_knowledge_base удаляется.
    Краткое окно недоступности: ~десятки миллисекунд пока удаляем и создаём алиас.
    search.py поймает исключение и вернёт [] — бот ответит "нет данных" один раз.
    """
    client = get_client()
    original = settings.QDRANT_COLLECTION_NAME

    logger.info(f"[HotSwap] First migration: deleting real '{original}', "
                f"creating alias → '{shadow_collection}'")

    try:
        # Удаляем оригинальную реальную коллекцию (освобождаем имя для алиаса)
        existing = [c.name for c in client.get_collections().collections]
        if original in existing:
            client.delete_collection(original)
            logger.info(f"[HotSwap] Deleted original collection '{original}'")

        # Создаём алиас: caiu_knowledge_base → shadow_collection
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=shadow_collection,
                        alias_name=original,
                    )
                )
            ]
        )
        set_active_collection(shadow_collection)
        logger.info(f"[HotSwap] Alias '{original}' → '{shadow_collection}' created")

    except Exception as e:
        logger.error(f"[HotSwap] First migration failed: {e}")
        raise


def _atomic_swap(shadow_collection: str) -> None:
    """
    Атомарный свопа алиаса: старая активная → новая теневая.

    Qdrant принимает оба действия (удалить алиас + создать новый) в одном запросе,
    что гарантирует атомарность — нет окна когда алиас не существует.

    После свопа удаляем старую коллекцию чтобы освободить память.
    """
    client = get_client()
    alias_name    = settings.QDRANT_COLLECTION_NAME
    old_active    = get_active_collection()

    logger.info(f"[HotSwap] Atomic swap: '{alias_name}' → '{shadow_collection}' "
                f"(was '{old_active}')")

    try:
        # Атомарно: удаляем старый алиас + создаём новый в одном запросе
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=alias_name)
                ),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=shadow_collection,
                        alias_name=alias_name,
                    )
                ),
            ]
        )
        set_active_collection(shadow_collection)
        logger.info(f"[HotSwap] Alias now points to '{shadow_collection}'")

        # Удаляем старую коллекцию чтобы не тратить место
        try:
            existing = [c.name for c in client.get_collections().collections]
            if old_active in existing and old_active != shadow_collection:
                client.delete_collection(old_active)
                logger.info(f"[HotSwap] Deleted old collection '{old_active}'")
        except Exception as e:
            logger.warning(f"[HotSwap] Could not delete old collection '{old_active}': {e}")

    except Exception as e:
        logger.error(f"[HotSwap] Atomic swap failed: {e}")
        raise


def _snapshot_active_to_shadow(client, shadow_collection: str) -> int:
    """
    Копирует все точки из активной коллекции в теневую через scroll+upsert.

    Зачем:
      pipeline.page_needs_update() проверяет content_hash в коллекции назначения.
      Если коллекция пустая — все страницы считаются «новыми» и переиндексируются.
      Если предварительно скопировать туда все точки из активной коллекции —
      pipeline пропустит страницы чьи content_hash не изменились.

    Возвращает количество скопированных точек (0 если активная недоступна).
    """
    active = get_active_collection()
    if not active or active == shadow_collection:
        logger.info("[HotSwap] Snapshot skipped: no active collection or same as shadow")
        return 0

    existing = [c.name for c in client.get_collections().collections]
    if active not in existing:
        logger.info(f"[HotSwap] Snapshot skipped: active collection '{active}' not found")
        return 0

    logger.info(f"[HotSwap] Snapshot: copying '{active}' → '{shadow_collection}'...")
    copied = 0
    offset = None

    try:
        while True:
            points, offset = client.scroll(
                collection_name=active,
                limit=256,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            if not points:
                break

            client.upsert(
                collection_name=shadow_collection,
                points=[
                    PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                    for p in points
                ],
            )
            copied += len(points)

            if offset is None:
                break

        logger.info(f"[HotSwap] Snapshot complete: {copied} points copied → '{shadow_collection}'")

    except Exception as e:
        # Не критично: pipeline переиндексирует всё с нуля, это только оптимизация
        logger.warning(f"[HotSwap] Snapshot failed ({e}), will do full reindex")
        copied = 0

    return copied


def run_shadow_indexing() -> dict:
    """
    Полная горячая переиндексация:
    1. Захватить замок (не допустить двойного запуска)
    2. Создать теневую коллекцию
    3. Проиндексировать всё в неё (pipeline.run_indexing)
    4. Переключить алиас атомарно
    5. Освободить замок

    Возвращает словарь с результатами (тот же формат что pipeline.run_indexing).
    Предназначен для запуска в отдельном потоке (threading.Thread).
    """
    global _indexing_active, _last_result, _started_at

    # ── Уровень 1: межпроцессный файловый lock ──────────────────
    # Защищает от второго uvicorn-воркера (workers=2 в run_api.py).
    # Если файл уже залочен другим процессом — выходим без действия.
    file_lock_handle = _try_acquire_file_lock()
    if file_lock_handle is None:
        logger.warning(
            "[HotSwap] Indexing already running in another process — skipping duplicate request"
        )
        return {"status": "skipped", "reason": "already_in_progress_other_process"}

    # ── Уровень 2: внутрипроцессный лок (на случай двух потоков) ─
    if not _indexing_lock.acquire(blocking=False):
        _release_file_lock(file_lock_handle)
        logger.warning("[HotSwap] Indexing already in progress (same process), skipping duplicate request")
        return {"status": "skipped", "reason": "already_in_progress"}

    _indexing_active = True
    _started_at = datetime.now(timezone.utc).isoformat()

    try:
        shadow = get_shadow_collection_name()
        logger.info(f"[HotSwap] Shadow indexing started → '{shadow}'")

        # ── Очищаем теневую коллекцию перед индексацией ──────────
        # Если предыдущий hot-swap упал на середине — там могут быть
        # частичные данные старого прогона. Удаляем и пересоздаём чтобы
        # гарантировать чистый старт. Это безопасно: shadow не активна.
        client = get_client()
        existing = [c.name for c in client.get_collections().collections]
        if shadow in existing:
            logger.info(f"[HotSwap] Clearing stale shadow collection '{shadow}'")
            client.delete_collection(shadow)

        ensure_collection_exists(collection_name=shadow)

        # ── Snapshot: копируем активную коллекцию в теневую ──────
        # Благодаря этому page_needs_update() в pipeline.py найдёт
        # уже существующие content_hash-и → пропустит неизменённые страницы.
        # Без этого каждый /reindex переиндексировал всё с нуля ($0.20-0.30, 30-60 мин).
        # После этого: ~0$ и ~30с если сайт не менялся; только изменённые страницы — GPT.
        snapshot_points = _snapshot_active_to_shadow(client, shadow)

        # ── Индексируем в теневую коллекцию ──────────────────────
        result = run_indexing(collection_name=shadow)

        if result.get("status") != "completed":
            logger.error(f"[HotSwap] Pipeline returned non-completed status: {result}")
            _last_result = result
            return result

        # ── Переключаем алиас ─────────────────────────────────────
        is_alias = _collection_is_alias(settings.QDRANT_COLLECTION_NAME)
        if is_alias:
            _atomic_swap(shadow)
        else:
            _first_migration(shadow)

        result["swapped_to"]      = shadow
        result["swap_type"]       = "atomic" if is_alias else "first_migration"
        result["snapshot_points"] = snapshot_points
        _last_result = result

        # ── Инвалидируем поисковый Redis-кеш ─────────────────────
        # После свопа алиас уже указывает на новую коллекцию, но ключи
        # "caiu:search:*" в Redis ещё хранят результаты из СТАРОГО индекса.
        # Без инвалидации пользователи до 1 часа получают устаревшие данные.
        # Сканируем и удаляем пачками (scan_iter безопасен при большом объёме).
        try:
            from app.redis_client import get_redis
            r = get_redis()
            if r is not None:
                from app.retrieval.search import _REDIS_SEARCH_PREFIX
                keys = list(r.scan_iter(f"{_REDIS_SEARCH_PREFIX}*"))
                if keys:
                    # delete принимает *args — передаём список как *keys
                    r.delete(*keys)
                    logger.info(f"[HotSwap] Search cache invalidated: {len(keys)} keys deleted")
                else:
                    logger.info("[HotSwap] Search cache was empty, nothing to invalidate")
        except Exception as e:
            # Не критично: кеш протухнет сам через TTL (1 час)
            logger.warning(f"[HotSwap] Search cache flush failed (non-critical): {e}")

        logger.info(f"[HotSwap] Complete. New active collection: '{shadow}'")
        return result

    except Exception as e:
        error_result = {
            "status": "failed",
            "error": str(e),
            "collection": get_shadow_collection_name(),
        }
        _last_result = error_result
        logger.error(f"[HotSwap] Shadow indexing failed: {e}", exc_info=True)
        return error_result

    finally:
        _indexing_active = False
        _indexing_lock.release()
        _release_file_lock(file_lock_handle)


def get_indexing_status() -> dict:
    """
    Возвращает текущее состояние индексации.
    Используется эндпоинтом GET /index/status.
    """
    active_collection = get_active_collection()
    try:
        stats = get_collection_stats(collection_name=active_collection)
        total_chunks = stats.get("total_chunks", 0)
    except Exception:
        total_chunks = 0

    return {
        "in_progress":        _indexing_active,
        "started_at":         _started_at,
        "active_collection":  active_collection,
        "total_chunks":       total_chunks,
        "last_result":        _last_result,
    }
