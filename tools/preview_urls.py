#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preview_urls.py — Предпросмотр URL перед индексацией.

Использование:

    # Показать все URL которые будут проиндексированы
    python tools/preview_urls.py

    # Проверить конкретный URL
    python tools/preview_urls.py https://caiu.edu.kz/history-of-the-university-ru/

Статусы в списке:
    [НОВЫЙ]        — будет проиндексирован впервые
    [В БАЗЕ]       — уже проиндексирован, изменений нет → пропустится
    [ОБНОВИТСЯ]    — уже проиндексирован, но страница изменилась → переиндексируется
    [ИСКЛЮЧЁН]     — фильтр EXCLUDE_URL_PATTERNS блокирует этот URL

Для проверки конкретного URL Qdrant запускать не обязательно —
фильтрация по паттернам работает без базы данных.
"""

import os
import sys
import argparse

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.config import settings
from app.parser.crawler import _fetch_sitemap, _filter_urls, SitemapURL
from app.parser.extractor import download_page, extract_text


# ── Цвета для терминала ───────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _qdrant_available() -> bool:
    """Проверяет доступность Qdrant без исключений."""
    try:
        from app.indexer.storage import get_client
        client = get_client()
        client.get_collections()
        return True
    except Exception:
        return False


def _get_indexed_urls() -> dict:
    """
    Возвращает словарь {url: {"chunks": N, "questions": M, "hash": "abc..."}}
    для всех URL которые есть в Qdrant.
    Возвращает {} если Qdrant недоступен.
    """
    if not _qdrant_available():
        return {}

    try:
        from app.indexer.storage import get_client
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = get_client()
        indexed = {}
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
                if url not in indexed:
                    indexed[url] = {"chunks": 0, "questions": 0, "hash": p.get("content_hash", "")}
                if p.get("embed_text"):
                    indexed[url]["questions"] += 1
                else:
                    indexed[url]["chunks"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return indexed
    except Exception:
        return {}


def _check_exclude(url: str) -> str | None:
    """
    Возвращает паттерн который исключает URL, или None если URL проходит.
    """
    for pattern in settings.EXCLUDE_URL_PATTERNS:
        if pattern in url:
            return pattern
    return None


# ── Режим 1: список всех URL ──────────────────────────────────────────────────

def list_all_urls():
    print(f"\n{BOLD}Загружаю sitemap: {settings.SITEMAP_URL}{RESET}")
    print("(это может занять 10–30 секунд)\n")

    # Все URL из sitemap
    all_urls = _fetch_sitemap(settings.SITEMAP_URL)
    if not all_urls:
        print(f"{RED}[!] Не удалось загрузить sitemap. Проверьте интернет-соединение.{RESET}")
        return

    # Отфильтрованные (пройдут индексацию)
    filtered = _filter_urls(all_urls)
    filtered_set = {u.url for u in filtered}

    # Что в Qdrant
    print(f"Проверяю базу данных Qdrant...")
    indexed = _get_indexed_urls()
    qdrant_ok = bool(indexed) or _qdrant_available()

    if not qdrant_ok:
        print(f"{YELLOW}[i] Qdrant недоступен — статус в базе не показывается.{RESET}")
        print(f"    Запустите базу: docker-compose up -d\n")

    # ── Итоговый список ──────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"Всего в sitemap:    {len(all_urls)}")
    print(f"После фильтрации:   {len(filtered)}")
    excluded_count = len(all_urls) - len(filtered)
    print(f"Исключено:          {excluded_count}")
    if qdrant_ok:
        new_count     = sum(1 for u in filtered if u.url not in indexed)
        in_base_count = sum(1 for u in filtered if u.url in indexed)
        print(f"Новых (впервые):    {new_count}")
        print(f"Уже в базе:         {in_base_count}")
    print(f"{'='*65}\n")

    # Сначала показываем URL которые пройдут
    print(f"{BOLD}URL КОТОРЫЕ БУДУТ ПРОИНДЕКСИРОВАНЫ ({len(filtered)}):{RESET}\n")

    for i, item in enumerate(filtered, 1):
        url = item.url
        lastmod = f"  [{item.lastmod.strftime('%Y-%m-%d')}]" if item.lastmod else ""

        if url in indexed:
            info = indexed[url]
            print(f"  {BLUE}[В БАЗЕ]{RESET}    {i:3}.  {url}{GRAY}{lastmod}{RESET}")
            print(f"           {GRAY}чанков: {info['chunks']}, вопросов: {info['questions']}{RESET}")
        else:
            print(f"  {GREEN}[НОВЫЙ]{RESET}     {i:3}.  {url}{GRAY}{lastmod}{RESET}")

    # Показываем исключённые (кратко)
    excluded = [u for u in all_urls if u.url not in filtered_set]
    if excluded:
        print(f"\n{BOLD}ИСКЛЮЧЁННЫЕ URL ({len(excluded)}):{RESET}")
        print(f"{GRAY}(не будут парситься из-за EXCLUDE_URL_PATTERNS в config.py){RESET}\n")
        for item in excluded[:20]:  # показываем первые 20
            pattern = _check_exclude(item.url)
            print(f"  {RED}[ИСКЛЮЧЁН]{RESET}  {item.url}")
            print(f"             {GRAY}причина: паттерн «{pattern}»{RESET}")
        if len(excluded) > 20:
            print(f"\n  {GRAY}... и ещё {len(excluded) - 20} URL (показаны первые 20){RESET}")

    print(f"\n{'='*65}")
    print(f"Для проверки конкретного URL:")
    print(f"  python tools/preview_urls.py <URL>")
    print(f"Для запуска индексации:")
    print(f"  python tools/reindex.py")
    print(f"{'='*65}\n")


# ── Режим 2: проверка конкретного URL ────────────────────────────────────────

def check_single_url(url: str):
    print(f"\n{BOLD}Проверка URL:{RESET} {url}\n")

    # ── Шаг 1: проверяем паттерны исключений ─────────────────────────────────
    excluded_by = _check_exclude(url)
    if excluded_by:
        print(f"  {RED}✗ ИСКЛЮЧЁН из индексации{RESET}")
        print(f"    Причина: паттерн «{excluded_by}» из EXCLUDE_URL_PATTERNS в config.py")
        print(f"\n    Если нужно добавить этот URL — удалите или измените")
        print(f"    соответствующий паттерн в rag_service/app/config.py")
        _check_in_qdrant(url)
        return

    print(f"  {GREEN}✓ Паттерны исключений — ОК{RESET} (URL не попадает под фильтры)")

    # ── Шаг 2: проверяем sitemap ──────────────────────────────────────────────
    print(f"\n  Загружаю sitemap для проверки...")
    all_urls = _fetch_sitemap(settings.SITEMAP_URL)
    in_sitemap = any(u.url == url for u in all_urls)

    if in_sitemap:
        sitemap_item = next(u for u in all_urls if u.url == url)
        lastmod = sitemap_item.lastmod.strftime('%Y-%m-%d') if sitemap_item.lastmod else "не указана"
        print(f"  {GREEN}✓ В sitemap{RESET} (последнее изменение: {lastmod})")
    else:
        print(f"  {YELLOW}! Нет в sitemap{RESET}")
        print(f"    Этот URL не перечислен в sitemap.xml сайта.")
        print(f"    Однако вы можете добавить его вручную через MANUAL_TEST_URLS в config.py")

    # ── Шаг 3: проверяем Qdrant ───────────────────────────────────────────────
    _check_in_qdrant(url)

    # ── Шаг 4: пробуем скачать страницу ──────────────────────────────────────
    print(f"\n  Пробую скачать страницу...")
    html = download_page(url)
    if not html:
        print(f"  {RED}✗ Страница недоступна{RESET} (ошибка загрузки или таймаут)")
        return

    content = extract_text(url, html)
    word_count = len(content.text.split())
    has_headings = "## " in content.text

    print(f"  {GREEN}✓ Страница доступна{RESET}")
    print(f"    Заголовок:    {content.title or '(не определён)'}")
    print(f"    Слов в тексте: {word_count}")
    print(f"    Разделы (h2/h3): {'есть' if has_headings else 'нет'}")

    # Предварительный подсчёт чанков
    from app.parser.chunker import split_into_chunks
    chunks = split_into_chunks(content.text, url, content.title)
    q_count = len(chunks) * 4  # 4 вопроса на чанк

    print(f"\n  {BOLD}Ожидаемый результат при индексации:{RESET}")
    print(f"    Фрагментов:  ~{len(chunks)}")
    print(f"    Вопросов:    ~{q_count}")
    print(f"    Записей в Qdrant: ~{len(chunks) + q_count}")

    # Показываем первые несколько чанков как превью
    if chunks:
        print(f"\n  {BOLD}Превью фрагментов:{RESET}")
        for i, chunk in enumerate(chunks[:3], 1):
            section = f"  [{chunk.section_title}]" if chunk.section_title else ""
            preview = chunk.text[:120].replace("\n", " ").strip()
            print(f"\n    Фрагмент {i}{section}:")
            print(f"    {GRAY}{preview}...{RESET}")
        if len(chunks) > 3:
            print(f"\n    {GRAY}... и ещё {len(chunks) - 3} фрагментов{RESET}")

    print(f"\n  {BOLD}Вывод:{RESET}", end=" ")
    if in_sitemap:
        print(f"{GREEN}URL будет проиндексирован при запуске python tools/reindex.py{RESET}")
    else:
        print(f"{YELLOW}URL можно добавить вручную через MANUAL_TEST_URLS в config.py{RESET}")


def _check_in_qdrant(url: str):
    """Вспомогательная функция: проверяет статус URL в Qdrant."""
    print(f"\n  Проверяю базу данных (Qdrant)...")

    if not _qdrant_available():
        print(f"  {YELLOW}! Qdrant недоступен{RESET}")
        print(f"    Запустите базу: docker-compose up -d")
        return

    indexed = _get_indexed_urls()

    if url in indexed:
        info = indexed[url]
        print(f"  {BLUE}✓ Уже в базе:{RESET}")
        print(f"    Фрагментов: {info['chunks']}, Вопросов: {info['questions']}")
    else:
        print(f"  {YELLOW}! Не проиндексирован{RESET} (в базе Qdrant нет)")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Предпросмотр URL перед индексацией",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/preview_urls.py
  python tools/preview_urls.py https://caiu.edu.kz/history-of-the-university-ru/
  python tools/preview_urls.py https://caiu.edu.kz/news/some-article/
        """,
    )
    parser.add_argument("url", nargs="?", help="URL для проверки (необязательно)")
    args = parser.parse_args()

    if args.url:
        check_single_url(args.url)
    else:
        list_all_urls()


if __name__ == "__main__":
    main()
