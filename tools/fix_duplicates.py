# -*- coding: utf-8 -*-
"""
fix_duplicates.py — Удаляет дублирующиеся чанки из Qdrant.

Проблема: страница /faculties-ru/ переиндексировалась несколько раз
и в Qdrant накопилось 5 идентичных копий каждого чанка.
Это происходит когда content_hash страницы меняется при каждом запросе
(динамический контент: счётчики, даты, сессии) и delete_chunks_by_url
не вызывался должным образом.

Что делает скрипт:
1. Читает все чанки из коллекции
2. Находит чанки с одинаковым текстом (payload["text"])
3. Оставляет один, удаляет дубликаты
4. Выводит статистику

Запуск (из папки rag_service/):
    python ../tools/fix_duplicates.py

Опционально — только для конкретного URL:
    python ../tools/fix_duplicates.py https://caiu.edu.kz/faculties-ru/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_service"))

from app.indexer.storage import get_client, get_collection_stats
from app.config import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

target_url = sys.argv[1] if len(sys.argv) > 1 else None

client = get_client()

print("=" * 60)
print("FIX DUPLICATES IN QDRANT")
if target_url:
    print(f"URL фильтр: {target_url}")
else:
    print("Проверяем ВСЮ коллекцию")
print("=" * 60)

stats = get_collection_stats()
print(f"\nЗаписей до очистки: {stats['total_chunks']}")

# ── Читаем все чанки (постранично) ───────────────────────────────────────────
all_points = []
offset = None
batch_size = 500

while True:
    scroll_filter = None
    if target_url:
        scroll_filter = Filter(
            must=[FieldCondition(
                key="page_url",
                match=MatchValue(value=target_url),
            )]
        )

    result, next_offset = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(result)
    print(f"  Читаем: {len(all_points)} записей...")

    if next_offset is None or len(result) == 0:
        break
    offset = next_offset

print(f"\nВсего прочитано: {len(all_points)} записей")

# ── Находим дубликаты по тексту ───────────────────────────────────────────────
seen_texts: dict = {}   # text[:300] → первый id
duplicate_ids: list = []

for point in all_points:
    text = (point.payload or {}).get("text", "")
    key = text[:300].strip()   # первые 300 символов как ключ

    if not key:
        continue

    if key not in seen_texts:
        seen_texts[key] = str(point.id)
    else:
        duplicate_ids.append(str(point.id))

print(f"Уникальных чанков: {len(seen_texts)}")
print(f"Дубликатов найдено: {len(duplicate_ids)}")

if not duplicate_ids:
    print("\n[OK] Дубликатов нет, коллекция чистая!")
    sys.exit(0)

# ── Подтверждение ─────────────────────────────────────────────────────────────
print(f"\nБудет удалено {len(duplicate_ids)} дублирующихся записей.")
answer = input("Продолжить? (yes/no): ").strip().lower()
if answer not in ("yes", "y", "да"):
    print("Отменено.")
    sys.exit(0)

# ── Удаляем батчами по 100 ───────────────────────────────────────────────────
BATCH = 100
deleted = 0
for i in range(0, len(duplicate_ids), BATCH):
    batch = duplicate_ids[i:i + BATCH]
    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=batch,
    )
    deleted += len(batch)
    print(f"  Удалено: {deleted}/{len(duplicate_ids)}")

stats_after = get_collection_stats()
print(f"\nЗаписей после очистки: {stats_after['total_chunks']}")
print(f"[OK] Удалено {deleted} дубликатов.")
