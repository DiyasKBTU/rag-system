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
from typing import Optional

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

from app.config import settings
from app.indexer.storage import (
    get_client,
    get_active_collection,
    _set_active_collection,
    ensure_collection_exists,
    get_collection_stats,
)
from app.indexer.pipeline import run_indexing

logger = logging.getLogger(__name__)

# ── Константы имён коллекций ──────────────────────────────────────────────────
COLLECTION_BLUE  = f"{settings.QDRANT_COLLECTION_NAME}_blue"
COLLECTION_GREEN = f"{settings.QDRANT_COLLECTION_NAME}_green"

# ── Глобальный замок и флаг состояния ────────────────────────────────────────
_indexing_lock    = threading.Lock()
_indexing_active  = False       # True пока идёт индексация
_last_result: Optional[dict] = None      # Результат последней индексации
_started_at: Optional[str]   = None      # ISO timestamp старта


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
        _set_active_collection(shadow_collection)
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
        _set_active_collection(shadow_collection)
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

    if not _indexing_lock.acquire(blocking=False):
        logger.warning("[HotSwap] Indexing already in progress, skipping duplicate request")
        return {"status": "skipped", "reason": "already_in_progress"}

    _indexing_active = True
    _started_at = datetime.now(timezone.utc).isoformat()

    try:
        shadow = get_shadow_collection_name()
        logger.info(f"[HotSwap] Shadow indexing started → '{shadow}'")

        # Убеждаемся что теневая коллекция существует (создаём если надо)
        ensure_collection_exists(collection_name=shadow)

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

        result["swapped_to"] = shadow
        result["swap_type"]  = "atomic" if is_alias else "first_migration"
        _last_result = result

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
