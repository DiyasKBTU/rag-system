#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_db.py — Просмотр содержимого базы знаний.

Использование:

    # Показать все проиндексированные URL
    python tools/inspect_db.py

    # Показать чанки конкретной страницы
    python tools/inspect_db.py https://caiu.edu.kz/history-of-the-university-ru/

    # Показать только оригинальные чанки (без вопросов)
    python tools/inspect_db.py https://caiu.edu.kz/obshhezhitie/ --no-questions
"""

import os
import sys
import argparse
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.config import settings
from app.indexer.storage import get_client, get_collection_stats


def list_all_urls():
    """Показать все проиндексированные страницы с количеством чанков."""
    client = get_client()
    stats = get_collection_stats()

    print(f"\n{'='*60}")
    print(f"База: {stats['collection']}")
    print(f"Всего записей в базе: {stats['total_chunks']}")
    print(f"{'='*60}\n")

    # Собираем статистику по URL
    url_stats = defaultdict(lambda: {"title": "", "chunks": 0, "questions": 0})
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            p = point.payload or {}
            url = p.get("page_url", "")
            if not url:
                continue
            url_stats[url]["title"] = p.get("page_title", "")
            # Если embed_text не пустой — это вопрос-чанк
            is_question = bool(p.get("embed_text", ""))
            if is_question:
                url_stats[url]["questions"] += 1
            else:
                url_stats[url]["chunks"] += 1

        if next_offset is None:
            break
        offset = next_offset

    if not url_stats:
        print("[!] База пуста. Запустите индексацию:")
        print("    python tools/reindex.py")
        return

    print(f"Проиндексировано страниц: {len(url_stats)}\n")
    print(f"{'№':<4} {'Чанков':>7} {'Вопросов':>9}  {'Заголовок / URL'}")
    print("-" * 60)

    for i, (url, info) in enumerate(sorted(url_stats.items()), 1):
        title = info["title"] or "(без заголовка)"
        print(f"{i:<4} {info['chunks']:>7} {info['questions']:>9}  {title}")
        print(f"     {'':>7} {'':>9}  {url}")
        print()

    total_chunks = sum(v["chunks"] for v in url_stats.values())
    total_questions = sum(v["questions"] for v in url_stats.values())
    print(f"{'ИТОГО':<4} {total_chunks:>7} {total_questions:>9}")
    print(f"\nДля просмотра чанков конкретной страницы:")
    print(f"    python tools/inspect_db.py <URL>")


def inspect_url(url: str, show_questions: bool = True):
    """Показать все чанки для конкретного URL."""
    client = get_client()

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="page_url", match=MatchValue(value=url))]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        print(f"\n[!] URL не найден в базе: {url}")
        print("\nВозможные причины:")
        print("  1. Страница не была проиндексирована")
        print("  2. URL написан неточно (проверьте слеш в конце, регистр)")
        print("  3. База пуста — запустите: python tools/reindex.py")
        print("\nВсе проиндексированные URL:")
        print("  python tools/inspect_db.py")
        return

    # Разделяем на оригинальные чанки и вопросы
    orig_chunks = []
    question_chunks = []

    for point in results:
        p = point.payload or {}
        embed_text = p.get("embed_text", "")
        if embed_text:
            question_chunks.append(p)
        else:
            orig_chunks.append(p)

    # Сортируем оригинальные по chunk_index
    orig_chunks.sort(key=lambda x: x.get("chunk_index", 0))

    page_title = orig_chunks[0].get("page_title", "") if orig_chunks else ""

    print(f"\n{'='*60}")
    print(f"Страница: {page_title}")
    print(f"URL:      {url}")
    print(f"Чанков:   {len(orig_chunks)}")
    print(f"Вопросов: {len(question_chunks)}")
    print(f"{'='*60}")

    # ── Оригинальные чанки ──────────────────────────────────────
    print(f"\n── ФРАГМЕНТЫ ТЕКСТА ({len(orig_chunks)}) ──────────────────────\n")

    for i, p in enumerate(orig_chunks, 1):
        section = p.get("section_title", "")
        chunk_idx = p.get("chunk_index", 0)
        text = p.get("text", "")

        print(f"[Фрагмент {i}] (индекс: {chunk_idx})")
        if section:
            print(f"  Раздел: {section}")
        print(f"  Текст:")
        # Печатаем текст с отступом
        for line in text.split("\n"):
            print(f"    {line}")
        print()

    # ── Сгенерированные вопросы ──────────────────────────────────
    if show_questions and question_chunks:
        print(f"\n── СГЕНЕРИРОВАННЫЕ ВОПРОСЫ ({len(question_chunks)}) ──────────────\n")
        print("(Эти вопросы используются для поиска — пользователь задаёт похожий")
        print("вопрос, система находит эти векторы и возвращает текст фрагмента)\n")

        # Группируем вопросы по chunk_index чтобы видеть к какому фрагменту они относятся
        questions_by_chunk = defaultdict(list)
        for p in question_chunks:
            questions_by_chunk[p.get("chunk_index", 0)].append(p.get("embed_text", ""))

        for chunk_idx in sorted(questions_by_chunk.keys()):
            questions = questions_by_chunk[chunk_idx]
            print(f"  К фрагменту {chunk_idx + 1}:")
            for q in questions:
                print(f"    • {q}")
            print()

    elif show_questions:
        print(f"\n[i] Вопросы не сгенерированы для этой страницы.")
        print("    Запустите переиндексацию: python tools/reindex.py")


def main():
    parser = argparse.ArgumentParser(
        description="Просмотр базы знаний",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/inspect_db.py
  python tools/inspect_db.py https://caiu.edu.kz/history-of-the-university-ru/
  python tools/inspect_db.py https://caiu.edu.kz/obshhezhitie/ --no-questions
        """,
    )
    parser.add_argument("url", nargs="?", help="URL страницы для просмотра (необязательно)")
    parser.add_argument("--no-questions", action="store_true", help="Не показывать сгенерированные вопросы")
    args = parser.parse_args()

    if args.url:
        inspect_url(args.url, show_questions=not args.no_questions)
    else:
        list_all_urls()


if __name__ == "__main__":
    main()
