# -*- coding: utf-8 -*-
"""
debug_urls.py - Проверяет каждый URL из MANUAL_TEST_URLS
и показывает почему страница парсится или не парсится.

Запуск:
  python debug_urls.py

Показывает для каждой страницы:
  OK   - успешно, сколько символов текста получили
  FAIL - ошибка, по какой причине
"""

import sys
import time

from app.parser.crawler import get_urls_for_indexing
from app.parser.extractor import get_page_content, download_page, extract_text
from app.parser.chunker import split_into_chunks
from app.config import settings

DELAY = 1.0  # пауза между запросами (секунды)


def check_url(url: str) -> dict:
    """Проверяет один URL и возвращает результат."""
    result = {
        "url": url,
        "status": "unknown",
        "title": "",
        "text_len": 0,
        "chunks": 0,
        "error": "",
    }

    # Шаг 1: Скачиваем HTML
    html = download_page(url)
    if not html:
        result["status"] = "FAIL"
        result["error"] = "Could not download (timeout or HTTP error)"
        return result

    result["html_len"] = len(html)

    # Шаг 2: Извлекаем текст
    content = extract_text(url, html)

    result["title"] = content.title or "(no title)"
    result["text_len"] = len(content.text)

    if len(content.text) < 50:
        result["status"] = "FAIL"
        result["error"] = f"Too little text extracted ({len(content.text)} chars). HTML size: {len(html)} chars."
        return result

    # Шаг 3: Разбиваем на чанки
    chunks = split_into_chunks(
        text=content.text,
        page_url=url,
        page_title=content.title,
    )
    result["chunks"] = len(chunks)

    if not chunks:
        result["status"] = "FAIL"
        result["error"] = "Text extracted but no chunks created (too short?)"
        return result

    result["status"] = "OK"
    return result


def main():
    print("\n" + "=" * 70)
    print("URL DIAGNOSTICS")
    print("=" * 70)
    print(f"Checking each URL from MANUAL_TEST_URLS...\n")

    urls = get_urls_for_indexing()

    if not urls:
        print("[!] No URLs in MANUAL_TEST_URLS")
        sys.exit(1)

    print(f"Total URLs to check: {len(urls)}\n")
    print("-" * 70)

    ok_list = []
    fail_list = []

    for i, url_item in enumerate(urls, 1):
        url = url_item.url
        print(f"[{i:2}/{len(urls)}] {url}")

        result = check_url(url)

        if result["status"] == "OK":
            print(f"       ✓ OK | Title: {result['title'][:50]}")
            print(f"            | Text: {result['text_len']} chars | Chunks: {result['chunks']}")
            ok_list.append(result)
        else:
            print(f"       ✗ FAIL | {result['error']}")
            fail_list.append(result)

        print()

        # Пауза между запросами
        if i < len(urls):
            time.sleep(DELAY)

    # Итог
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  OK:   {len(ok_list)} pages")
    print(f"  FAIL: {len(fail_list)} pages")
    print()

    if ok_list:
        print("Pages that parsed successfully:")
        for r in ok_list:
            print(f"  [+] {r['title'][:45]:<45} | {r['text_len']:>5} chars | {r['chunks']} chunks")
            print(f"      {r['url']}")

    print()

    if fail_list:
        print("Pages that FAILED:")
        for r in fail_list:
            print(f"  [-] {r['url']}")
            print(f"      Reason: {r['error']}")

    print()
    print("=" * 70)

    if fail_list:
        print("\nWhat to do with failed pages:")
        print("  - If error is 'Too little text' — page may use unusual HTML structure")
        print("    (open the page in browser, right-click -> View Page Source,")
        print("     look for where the main text is stored)")
        print("  - If error is 'Could not download' — page may be offline or blocked")
        print("  - Google Drive links always fail — remove them from MANUAL_TEST_URLS")


if __name__ == "__main__":
    main()
