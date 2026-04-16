# -*- coding: utf-8 -*-
"""
rebuild_catalog.py — Быстрое обновление каталог-чанков (факультеты + специальности).

Запускать когда:
- Бот выдаёт неполный список факультетов или специальностей
- После добавления нового факультета / специальности на сайт
- После полной переиндексации (для надёжности)

Время работы: ~3–5 минут (скачивает ~35 страниц).
Стоимость: копейки (только embeddings для ~20 текстов запросов).
Переиндексация НЕ нужна.

Запуск (из папки rag_service/):
    python ../tools/rebuild_catalog.py
"""

import sys
import os

# Добавляем rag_service в путь чтобы импорты работали
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_service"))

from app.logging_setup import setup_logging
setup_logging("catalog")

from app.indexer.storage import ensure_collection_exists, get_collection_stats
from app.indexer.catalog_builder import build_and_save_catalog_chunks

print("=" * 60)
print("REBUILD CATALOG CHUNKS")
print("Факультеты + Специальности → Qdrant")
print("=" * 60)

# Убеждаемся что коллекция существует
ensure_collection_exists()

stats_before = get_collection_stats()
print(f"\nЗаписей в Qdrant до: {stats_before['total_chunks']}")

# Строим каталог-чанки
total = build_and_save_catalog_chunks()

stats_after = get_collection_stats()
print(f"\nЗаписей в Qdrant после: {stats_after['total_chunks']}")
print(f"Добавлено каталог-векторов: {total}")
print("\n[OK] Перезапустите бот и RAG-сервис для применения изменений.")
print("     (изменения в search.py применяются сразу без перезапуска Qdrant)")
