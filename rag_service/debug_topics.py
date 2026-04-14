# -*- coding: utf-8 -*-
"""
debug_topics.py — Проверяет, есть ли в Qdrant чанки по проблемным темам.

Запуск:
    python debug_topics.py

Показывает для каждой темы:
  - лучший score из Qdrant
  - текст найденного чанка
  - вывод: "НАЙДЕНО" / "НЕТ В БАЗЕ" / "НИЖЕ ПОРОГА"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retrieval.search import search
from app.indexer.storage import get_collection_stats
from app.config import settings

# ── Темы для проверки ────────────────────────────────────────────
TOPICS = [
    {
        "name": "История университета",
        "queries": [
            "когда был основан университет",
            "история университета КАИУ",
            "год основания",
        ],
    },
    {
        "name": "Стоимость обучения",
        "queries": [
            "сколько стоит обучение",
            "стоимость обучения в КАИУ",
            "цена контракта",
        ],
    },
    {
        "name": "Количество студентов",
        "queries": [
            "сколько студентов в университете",
            "количество студентов КАИУ",
            "численность обучающихся",
        ],
    },
    {
        "name": "Миссия / О вузе",
        "queries": [
            "миссия университета",
            "о вузе КАИУ",
            "об университете",
        ],
    },
]

CONFIDENT_THRESHOLD = settings.MIN_CONFIDENT_SCORE  # из config.py

# ── Запуск ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ДИАГНОСТИКА ПРОБЛЕМНЫХ ТЕМ")
print("=" * 65)

stats = get_collection_stats()
print(f"\nЧанков в Qdrant: {stats['total_chunks']}")
print(f"Порог MIN_CONFIDENT_SCORE: {CONFIDENT_THRESHOLD}\n")

if stats['total_chunks'] == 0:
    print("[!] Qdrant пустой. Сначала запусти индексацию.")
    sys.exit(1)

for topic in TOPICS:
    print(f"\n{'─' * 65}")
    print(f"  ТЕМА: {topic['name']}")
    print(f"{'─' * 65}")

    best_result = None
    best_query = None

    for query in topic["queries"]:
        # Ищем с очень низким порогом чтобы увидеть что вообще есть
        results = search(query, top_k=3, min_score=0.20)
        if results and (best_result is None or results[0].score > best_result.score):
            best_result = results[0]
            best_query = query

    if best_result is None:
        print(f"  ❌  НЕТ В БАЗЕ — ни один запрос не дал результата (score < 0.20)")
        print(f"      Вероятные причины:")
        print(f"      • Страница не попала в sitemap")
        print(f"      • Содержимое в PDF или рендерится через JavaScript")
        print(f"      • Добавь информацию вручную в Google Docs")
    elif best_result.score < CONFIDENT_THRESHOLD:
        print(f"  ⚠️  НИЖЕ ПОРОГА — найдено, но score {best_result.score:.3f} < {CONFIDENT_THRESHOLD}")
        print(f"      Запрос: «{best_query}»")
        print(f"      Страница: {best_result.page_title}")
        print(f"      Текст: {best_result.text[:200]}...")
        print(f"      → Попробуй снизить MIN_CONFIDENT_SCORE в config.py")
    else:
        print(f"  ✅  НАЙДЕНО — score {best_result.score:.3f}")
        print(f"      Запрос: «{best_query}»")
        print(f"      Страница: {best_result.page_title}")
        print(f"      Текст: {best_result.text[:300]}...")

print(f"\n{'=' * 65}")
print("ИТОГ:")
print("  ❌  НЕТ В БАЗЕ   → добавь в Google Docs или найди URL страницы")
print("  ⚠️  НИЖЕ ПОРОГА  → снизь MIN_CONFIDENT_SCORE или SIMILARITY_THRESHOLD")
print("  ✅  НАЙДЕНО      → тема работает нормально")
print("=" * 65 + "\n")
