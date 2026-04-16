#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_chunks.py — Показать статистику по базе знаний.

Запуск:
    python tools/check_chunks.py

Выводит:
    - Сколько всего фрагментов в базе
    - Список проиндексированных страниц
"""

import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.config import settings
from app.indexer.storage import get_client, get_collection_stats

def main():
    client = get_client()
    stats = get_collection_stats()

    print(f"\n{'='*50}")
    print(f"База знаний: {stats['collection']}")
    print(f"Всего фрагментов: {stats['total_chunks']}")
    print(f"Статус: {stats['status']}")
    print(f"{'='*50}")

    # Показать все уникальные страницы
    seen_urls = set()
    offset = None
    pages = []

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            url = point.payload.get("page_url", "")
            title = point.payload.get("page_title", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                pages.append((title, url))

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nПроиндексировано страниц: {len(pages)}\n")
    for title, url in sorted(pages, key=lambda x: x[1]):
        print(f"  {title or '(без заголовка)'}")
        print(f"  {url}\n")

if __name__ == "__main__":
    main()
