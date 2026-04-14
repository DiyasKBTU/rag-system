#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_url.py — Проверяет, запарсился ли конкретный URL в Qdrant.
               С флагом --show выводит текст всех чанков.

Использование:
    python check_url.py https://caiu.edu.kz/ru/contacts/
    python check_url.py https://caiu.edu.kz/ru/contacts/ --show
    python check_url.py  (спросит URL интерактивно)
"""

import sys
import os

# Добавляем корень проекта в путь чтобы импорты работали
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import settings


def get_chunks(url: str) -> list:
    """
    Возвращает все чанки из Qdrant для указанного URL.
    """
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )

    result = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="page_url",
                    match=MatchValue(value=url),
                )
            ]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    return result[0]


def check_url(url: str) -> dict:
    """
    Проверяет наличие URL в Qdrant.
    Возвращает словарь с результатом и списком чанков.
    """
    points = get_chunks(url)

    if not points:
        return {
            "found": False,
            "url": url,
            "chunks_count": 0,
            "title": None,
            "content_hash": None,
            "points": [],
        }

    first_payload = points[0].payload
    title = first_payload.get("title", "—")
    content_hash = first_payload.get("content_hash", "—")

    return {
        "found": True,
        "url": url,
        "chunks_count": len(points),
        "title": title,
        "content_hash": content_hash,
        "points": points,
    }


def print_result(info: dict, show_chunks: bool = False):
    """Выводит результат в консоль."""
    print()
    print("=" * 60)
    if info["found"]:
        print(f"  ✅  URL ЗАПАРСЕН")
        print("=" * 60)
        print(f"  URL       : {info['url']}")
        print(f"  Заголовок : {info['title']}")
        print(f"  Чанков    : {info['chunks_count']}")
        print(f"  Хэш       : {info['content_hash']}")
        print("=" * 60)

        if show_chunks:
            print()
            # Сортируем по chunk_index если есть
            points = sorted(
                info["points"],
                key=lambda p: p.payload.get("chunk_index", 0)
            )
            for i, point in enumerate(points):
                payload = point.payload
                chunk_index = payload.get("chunk_index", i)
                text = payload.get("text", payload.get("content", "—"))

                print(f"  ┌─ Чанк #{chunk_index + 1}  (id: {str(point.id)[:8]}...) ─────────────────")
                print()
                # Выводим текст с отступом
                for line in text.splitlines():
                    print(f"  │  {line}")
                print()
                print(f"  └{'─' * 56}")
                print()
    else:
        print(f"  ❌  URL НЕ НАЙДЕН В БАЗЕ")
        print("=" * 60)
        print(f"  URL       : {info['url']}")
        print()
        print("  Возможные причины:")
        print("  • URL ещё не парсился")
        print("  • URL был исключён фильтрами (EXCLUDE_URL_PATTERNS)")
        print("  • Парсинг завершился с ошибкой")
        print("  • Контент страницы оказался пустым")
        print("=" * 60)
    print()


def main():
    show_chunks = "--show" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--show"]

    if args:
        url = args[0].strip()
    else:
        print()
        url = input("  Введите URL для проверки: ").strip()
        if not show_chunks:
            ans = input("  Показать текст чанков? [y/N]: ").strip().lower()
            show_chunks = ans in ("y", "yes", "да", "д")

    if not url:
        print("  ⚠️  URL не указан. Выход.")
        sys.exit(1)

    # Проверяем оба варианта — со слешом в конце и без
    urls_to_check = [url]
    if url.endswith("/"):
        urls_to_check.append(url.rstrip("/"))
    else:
        urls_to_check.append(url + "/")

    print(f"\n  Подключение к Qdrant ({settings.QDRANT_HOST}:{settings.QDRANT_PORT})...")

    found_info = None
    for u in urls_to_check:
        info = check_url(u)
        if info["found"]:
            found_info = info
            break

    if found_info is None:
        found_info = check_url(url)
        found_info["url"] = url

    print_result(found_info, show_chunks=show_chunks)


if __name__ == "__main__":
    main()
