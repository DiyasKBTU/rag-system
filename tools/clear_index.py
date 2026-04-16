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
"""

import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.indexer.storage import get_client, ensure_collection_exists
from app.config import settings


def main():
    client = get_client()

    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION_NAME not in existing:
        print(f"[!] База '{settings.QDRANT_COLLECTION_NAME}' пуста — нечего очищать.")
        return

    info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
    count = info.points_count
    print(f"\nВ базе сейчас: {count} записей")

    confirm = input(f"\nУдалить все {count} записей? Введите 'yes' для подтверждения: ").strip().lower()
    if confirm != "yes":
        print("Отменено.")
        return

    client.delete_collection(settings.QDRANT_COLLECTION_NAME)
    print("[✓] База очищена.")

    ensure_collection_exists()
    print("[✓] Пустая база создана заново.")
    print("\nТеперь запустите переиндексацию:")
    print("    python tools/reindex.py")


if __name__ == "__main__":
    main()
