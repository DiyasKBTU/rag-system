# -*- coding: utf-8 -*-
"""
index_page.py — Индексировать одну страницу по URL (терминал).

Скачивает страницу, разбивает на чанки, генерирует вопросы и сохраняет в Qdrant.
Старые данные для этого URL удаляются перед сохранением (полная замена).

Использование:
    python tools/index_page.py https://caiu.edu.kz/some-page/

Флаги:
    --no-questions   Пропустить генерацию вопросов GPT (быстро, бесплатно)
    --force          Индексировать даже если содержимое не изменилось
                     ИЛИ страница защищена ручными правками (manually_edited)

Защита ручных правок:
    Если страница ранее была отредактирована вручную через chunk_editor
    (все её чанки имеют флаг manually_edited=True), скрипт ОТКАЖЕТСЯ
    перезаписывать её. Чтобы всё-таки переиндексировать — используйте --force.
    Это сохраняет работу IT-отдела от случайной потери.

Примеры:
    # Переиндексировать страницу с вопросами (стандартно):
    python tools/index_page.py https://caiu.edu.kz/about-ru/

    # Быстро, без GPT:
    python tools/index_page.py https://caiu.edu.kz/contacts/ --no-questions

    # Принудительно переписать ручные правки (опасно!):
    python tools/index_page.py https://caiu.edu.kz/some-page/ --force

    # Проверить страницу (посмотреть что проиндексировано):
    python tools/inspect_db.py https://caiu.edu.kz/some-page/

Запуск из папки rag_service/:
    python ../tools/index_page.py https://caiu.edu.kz/...
"""

import sys
import os
import argparse

# Добавляем rag_service в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_service"))

from app.logging_setup import setup_logging
setup_logging("index_page")

from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks
from app.indexer.storage import (
    ensure_collection_exists,
    save_chunks,
    page_needs_update,
    get_collection_stats,
    url_is_manually_edited,
)
from app.indexer.question_generator import generate_question_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Индексировать одну страницу по URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="URL страницы для индексации")
    parser.add_argument(
        "--no-questions", dest="no_questions", action="store_true",
        help="Не генерировать вопросы GPT (быстрее, дешевле)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Переиндексировать даже если содержимое не изменилось",
    )
    args = parser.parse_args()

    url = args.url.strip()
    if not url.startswith("http"):
        print(f"[!] Некорректный URL: {url}")
        print("    Пример: https://caiu.edu.kz/about-ru/")
        sys.exit(1)

    print("=" * 60)
    print("INDEX SINGLE PAGE")
    print(f"URL: {url}")
    if args.no_questions:
        print("Режим: без генерации вопросов")
    print("=" * 60)

    # Убеждаемся что коллекция существует
    ensure_collection_exists()

    # ── Защита ручных правок ─────────────────────────────────────
    # Если все чанки URL помечены manually_edited=True (через chunk_editor)
    # — отказываемся перезаписывать, чтобы не потерять работу IT-отдела.
    # --force позволяет явно проигнорировать защиту.
    if url_is_manually_edited(url):
        if args.force:
            print("\n[!] ВНИМАНИЕ: страница содержит ручные правки (manually_edited=True).")
            print("    Запущено с --force → ручные правки БУДУТ перезаписаны автопарсингом.")
            print("    Если это случайность — прервите сейчас (Ctrl+C).")
            try:
                # Короткая пауза чтобы пользователь успел прочитать и нажать Ctrl+C
                import time as _time
                _time.sleep(3)
            except KeyboardInterrupt:
                print("\nПрервано пользователем.")
                sys.exit(0)
        else:
            print("\n[!] ОТКАЗ: страница защищена ручными правками (manually_edited=True).")
            print("    Все её чанки были созданы/отредактированы вручную через chunk_editor.")
            print("    Автоматическая переиндексация затёрла бы эту работу.")
            print()
            print("    Что делать:")
            print("      1. Открыть страницу в chunk_editor (http://localhost:8080),")
            print("         внести правки руками — это безопасный путь;")
            print("      2. ИЛИ удалить все чанки страницы через chunk_editor,")
            print("         после чего запустить index_page.py заново;")
            print("      3. ИЛИ запустить эту команду с --force, чтобы")
            print("         явно перезаписать ручные правки автопарсингом.")
            sys.exit(1)

    # ── Шаг 1: Скачать страницу ──────────────────────────────────
    print("\n[1/4] Скачиваю страницу...")
    content = get_page_content(url)

    if not content or not content.text.strip():
        print("[!] Страница пустая или недоступна.")
        print("    Проверьте URL и доступность сайта.")
        sys.exit(1)

    print(f"      Заголовок: {content.title or '(без заголовка)'}")
    print(f"      Текст: {len(content.text)} символов")
    if getattr(content, "external_links", []):
        print(f"      Внешние ссылки: {len(content.external_links)}")

    # ── Шаг 2: Проверить изменения ───────────────────────────────
    if not args.force and not page_needs_update(url, content.content_hash):
        print("\n[=] Страница не изменилась. Переиндексация не нужна.")
        print("    Используйте --force чтобы переиндексировать принудительно.")
        sys.exit(0)

    # ── Шаг 3: Разбить на чанки ──────────────────────────────────
    print("\n[2/4] Разбиваю на чанки...")
    chunks = split_into_chunks(
        text=content.text,
        page_url=content.url,
        page_title=content.title,
        external_links=getattr(content, "external_links", []),
    )

    if not chunks:
        print("[!] Не удалось разбить страницу на чанки (пустой текст?).")
        sys.exit(1)

    print(f"      Получено чанков: {len(chunks)}")
    for i, ch in enumerate(chunks):
        sec = f" [{ch.section_title}]" if ch.section_title else ""
        print(f"      #{i}{sec}: {len(ch.text.split())} слов")

    # ── Шаг 4: Генерация вопросов ────────────────────────────────
    question_chunks = []
    if not args.no_questions:
        print(f"\n[3/4] Генерирую вопросы GPT ({len(chunks)} чанков × 4 вопроса)...")
        try:
            question_chunks = generate_question_chunks(chunks)
            print(f"      Получено вопрос-чанков: {len(question_chunks)}")
        except Exception as e:
            print(f"      [!] Ошибка генерации вопросов: {e}")
            print("          Продолжаю без вопросов...")
    else:
        print("\n[3/4] Генерация вопросов пропущена (--no-questions)")

    # ── Шаг 5: Сохранить в Qdrant ────────────────────────────────
    print("\n[4/4] Сохраняю в Qdrant...")
    all_chunks = chunks + question_chunks
    saved = save_chunks(all_chunks, content.url, content.content_hash)

    stats = get_collection_stats()
    print(f"      Сохранено векторов: {saved}")
    print(f"      Итого в базе: {stats['total_chunks']} записей")

    print()
    print("[OK] Страница проиндексирована.")
    print(f"     {len(chunks)} чанков + {len(question_chunks)} вопросов = {saved} векторов")
    print()
    print("Проверить результат:")
    print(f"    python tools/inspect_db.py {url}")
    print(f"    python tools/debug_search.py \"<вопрос по этой странице>\"")


if __name__ == "__main__":
    main()
