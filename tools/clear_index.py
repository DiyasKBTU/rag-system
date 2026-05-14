#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
clear_index.py — Полностью очистить базу данных Qdrant.

Запуск из корня проекта:
    python tools/clear_index.py

Когда нужно:
    - После изменения chunker.py или extractor.py (структура чанков изменилась)
    - После изменения CHUNK_SIZE в config.py
    - Хотите начать индексацию с нуля

После очистки запустить переиндексацию:
    python tools/reindex.py

Что делает скрипт:
    1. Определяет в каком режиме сейчас Qdrant:
       (a) Pre-hot-swap: одна реальная коллекция `caiu_knowledge_base`.
       (b) Post-hot-swap: алиас `caiu_knowledge_base` → реальная `_blue` или `_green`.
    2. Удаляет алиас (если есть) и ВСЕ реальные коллекции (blue, green
       и оригинальная — на всякий случай).
    3. Сбрасывает hot_swap_state.json — следующая индексация начнётся
       с чистого состояния (первая миграция alias заново создастся).
    4. Удаляет fetch_cache.pkl чтобы переиндексация скачала всё заново.
    5. Создаёт пустую реальную коллекцию `caiu_knowledge_base`.
"""

import os
import sys
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from qdrant_client.models import DeleteAlias, DeleteAliasOperation

from app.indexer.storage import get_client, ensure_collection_exists
from app.config import settings


# Пути к рантайм-файлам (рядом с rag_service/)
_RAG_SERVICE_DIR  = Path(__file__).resolve().parent.parent / "rag_service"
_STATE_FILE       = _RAG_SERVICE_DIR / "hot_swap_state.json"
_FETCH_CACHE_FILE = _RAG_SERVICE_DIR / "fetch_cache.pkl"

# Имена коллекций hot-swap (совпадают с hot_swap.py — НЕ импортируем оттуда,
# чтобы не подтянуть лишние зависимости и не делать запросы к Qdrant на импорте)
_ALIAS_NAME       = settings.QDRANT_COLLECTION_NAME
_COLLECTION_BLUE  = f"{_ALIAS_NAME}_blue"
_COLLECTION_GREEN = f"{_ALIAS_NAME}_green"


def _get_existing_collections(client) -> list:
    """Список реальных коллекций (без алиасов)."""
    try:
        return [c.name for c in client.get_collections().collections]
    except Exception as e:
        print(f"[!] Не удалось получить список коллекций: {e}")
        return []


def _get_existing_aliases(client) -> list:
    """Список существующих алиасов."""
    try:
        return [a.alias_name for a in client.get_aliases().aliases]
    except Exception:
        return []


def _count_points(client, collection: str) -> int:
    """Возвращает количество точек в коллекции, 0 при ошибке."""
    try:
        return client.get_collection(collection).points_count or 0
    except Exception:
        return 0


def main():
    client = get_client()

    existing_collections = _get_existing_collections(client)
    existing_aliases     = _get_existing_aliases(client)

    is_hot_swap_mode = _ALIAS_NAME in existing_aliases

    # Собираем список реальных коллекций к удалению.
    # В hot-swap режиме — blue, green и (на всякий случай) оригинальная.
    # В обычном — только оригинальная.
    real_collections_to_delete = []
    for name in (_COLLECTION_BLUE, _COLLECTION_GREEN, _ALIAS_NAME):
        if name in existing_collections and name not in real_collections_to_delete:
            real_collections_to_delete.append(name)

    if not real_collections_to_delete and not is_hot_swap_mode:
        print(f"[!] Реальных коллекций нет — нечего очищать.")
        print(f"    Алиас '{_ALIAS_NAME}' тоже не найден.")
        # Всё равно создадим пустую коллекцию, чтобы база была готова
        ensure_collection_exists()
        print("[✓] Пустая база создана.")
        return

    # ── Подсчёт записей перед удалением ────────────────────────
    total_points = sum(_count_points(client, c) for c in real_collections_to_delete)

    print("\n" + "=" * 60)
    print("ОЧИСТКА БАЗЫ ZNANIY")
    print("=" * 60)
    if is_hot_swap_mode:
        print(f"Режим: HOT-SWAP (алиас → blue/green)")
        print(f"Алиас:               {_ALIAS_NAME}")
    else:
        print(f"Режим: обычный (одна реальная коллекция)")
    print(f"Реальные коллекции:  {real_collections_to_delete}")
    print(f"Всего записей:       {total_points}")
    print(f"\nБудет также удалено:")
    if _STATE_FILE.exists():
        print(f"  - {_STATE_FILE.name} (состояние hot-swap)")
    if _FETCH_CACHE_FILE.exists():
        print(f"  - {_FETCH_CACHE_FILE.name} (кеш скачанных страниц)")

    confirm = input(f"\nУдалить ВСЁ? Введите 'yes' для подтверждения: ").strip().lower()
    if confirm != "yes":
        print("Отменено.")
        return

    # ── Шаг 1: удалить алиас ───────────────────────────────────
    # ВАЖНО: алиас удаляем ДО реальных коллекций, иначе Qdrant может
    # пожаловаться что не может удалить коллекцию на которую указывает алиас.
    if is_hot_swap_mode:
        try:
            client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=_ALIAS_NAME))
                ]
            )
            print(f"[✓] Алиас '{_ALIAS_NAME}' удалён")
        except Exception as e:
            print(f"[!] Не удалось удалить алиас: {e}")
            print(f"    Продолжаю удаление коллекций...")

    # ── Шаг 2: удалить реальные коллекции ──────────────────────
    for col in real_collections_to_delete:
        try:
            client.delete_collection(col)
            print(f"[✓] Коллекция '{col}' удалена")
        except Exception as e:
            print(f"[!] Не удалось удалить '{col}': {e}")

    # ── Шаг 3: сбросить state-файлы ────────────────────────────
    # hot_swap_state.json — после очистки мы в pre-hot-swap состоянии.
    # Следующий /reindex создаст алиас заново при первой миграции.
    if _STATE_FILE.exists():
        try:
            _STATE_FILE.unlink()
            print(f"[✓] {_STATE_FILE.name} удалён")
        except Exception as e:
            print(f"[!] Не удалось удалить {_STATE_FILE.name}: {e}")

    # fetch_cache.pkl — кеш скачанных страниц. Если очищаем базу,
    # пользователь обычно хочет ПОЛНУЮ переиндексацию, без resume.
    if _FETCH_CACHE_FILE.exists():
        try:
            _FETCH_CACHE_FILE.unlink()
            print(f"[✓] {_FETCH_CACHE_FILE.name} удалён")
        except Exception as e:
            print(f"[!] Не удалось удалить {_FETCH_CACHE_FILE.name}: {e}")

    # ── Шаг 4: создать пустую коллекцию заново ─────────────────
    # ensure_collection_exists() теперь увидит что алиаса нет → создаст
    # реальную коллекцию с именем _ALIAS_NAME (как было до hot-swap).
    ensure_collection_exists()
    print(f"[✓] Пустая коллекция '{_ALIAS_NAME}' создана.")

    print("\n" + "=" * 60)
    print("[✓] БАЗА ПОЛНОСТЬЮ ОЧИЩЕНА")
    print("=" * 60)
    print("\nТеперь запустите переиндексацию:")
    print("    python tools/reindex.py")
    print("\nПри первом запуске hot-swap создаст алиас и blue/green коллекции.")


if __name__ == "__main__":
    main()
