# -*- coding: utf-8 -*-
"""
clear_index.py - Полностью очищает коллекцию в Qdrant.

Запускай перед переиндексацией если:
  - Изменил настройки парсера (extractor.py, chunker.py)
  - Изменил CHUNK_SIZE или CHUNK_OVERLAP
  - Хочешь начать индексацию с нуля

После очистки запусти:
  python -m app.parser

Запуск:
  python clear_index.py
"""

from app.indexer.storage import get_client, ensure_collection_exists
from app.config import settings


def clear_collection():
    client = get_client()

    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION_NAME not in existing:
        print(f"[!] Коллекция '{settings.QDRANT_COLLECTION_NAME}' не существует — нечего очищать.")
        return

    # Показываем сколько было чанков
    info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
    count = info.points_count
    print(f"[i] В коллекции сейчас: {count} чанков")

    # Подтверждение
    confirm = input(f"\nУдалить все {count} чанков из '{settings.QDRANT_COLLECTION_NAME}'? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("Отменено.")
        return

    # Удаляем коллекцию целиком
    client.delete_collection(settings.QDRANT_COLLECTION_NAME)
    print(f"[✓] Коллекция удалена.")

    # Создаём заново пустую
    ensure_collection_exists()
    print(f"[✓] Пустая коллекция создана заново.")
    print(f"\nТеперь запусти: python -m app.parser")


if __name__ == "__main__":
    clear_collection()
