#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_langs.py — Найти казахские страницы у которых есть русский аналог.

Запуск из корня проекта:
    python tools/compare_langs.py
"""

import re
import sys
import time
import xml.etree.ElementTree as ET

try:
    import requests
    def fetch(url):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
except ImportError:
    import urllib.request as _req
    def fetch(url):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = _req.Request(url, headers=headers)
        with _req.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8")

SITEMAP_URL = "https://caiu.edu.kz/sitemap.xml"


def parse_urls(xml_text: str) -> list:
    """Извлечь все <loc> из XML."""
    xml_clean = re.sub(r'\s+xmlns(?::\w+)?="[^"]+"', '', xml_text)
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as e:
        print(f"  [XML ошибка] {e}")
        return []
    return [el.text.strip() for el in root.iter("loc") if el.text]


def fetch_all_urls(sitemap_url: str, depth: int = 0) -> list:
    """Рекурсивно обойти sitemap index и собрать все page URL."""
    indent = "  " * depth
    print(f"{indent}-> {sitemap_url}")

    try:
        xml_text = fetch(sitemap_url)
    except Exception as e:
        print(f"{indent}   [ошибка] {e}")
        return []

    locs = parse_urls(xml_text)
    if not locs:
        return []

    # Если loc оканчивается на .xml — это под-сайтмап, идём глубже
    page_urls = []
    for loc in locs:
        if loc.endswith(".xml"):
            time.sleep(0.3)
            page_urls.extend(fetch_all_urls(loc, depth + 1))
        else:
            page_urls.append(loc)

    return page_urls


def normalize(url: str) -> str:
    """Slug без языкового маркера для сравнения пар ru ↔ kk."""
    url = url.rstrip("/")
    url = re.sub(r'https?://[^/]+', '', url)
    url = re.sub(r'^/(ru|kz|kk|en)/', '/', url)
    url = re.sub(r'-(ru|kz|kk|en)(-\d+)?/?$', '', url)
    return url.strip("/")


def main():
    print("Загружаю sitemap (рекурсивно)...\n")
    all_urls = fetch_all_urls(SITEMAP_URL)
    # убрать дубли
    all_urls = list(dict.fromkeys(all_urls))
    print(f"\nВсего страниц: {len(all_urls)}\n")

    ru_urls, kk_urls, other = [], [], []
    for url in all_urls:
        path = url.replace("https://caiu.edu.kz", "")
        if re.search(r'^/ru(/|$)|[-_]ru[-_/]|[-_]ru$', path):
            ru_urls.append(url)
        elif re.search(r'^/kz(/|$)|^/kk(/|$)|[-_]kz[-_/]|[-_]kz$|[-_]kk[-_/]|[-_]kk$', path):
            kk_urls.append(url)
        else:
            other.append(url)

    print(f"Русских страниц:   {len(ru_urls)}")
    print(f"Казахских страниц: {len(kk_urls)}")
    print(f"Прочих:            {len(other)}\n")

    # Строим индекс: slug → ru_url
    ru_index = {normalize(u): u for u in ru_urls}

    matched, unmatched = [], []
    for kk_url in sorted(kk_urls):
        slug = normalize(kk_url)
        if slug in ru_index:
            matched.append((kk_url, ru_index[slug]))
        else:
            unmatched.append(kk_url)

    print("=" * 65)
    print(f"Казахских с русской парой: {len(matched)}")
    print("=" * 65)
    for kk, ru in matched:
        print(f"  KK: {kk}")
        print(f"  RU: {ru}")
        print()

    if unmatched:
        print("=" * 65)
        print(f"Казахских БЕЗ русской пары: {len(unmatched)}")
        print("=" * 65)
        for url in unmatched:
            print(f"  {url}")

    print("\n" + "=" * 65)
    print(f"Казахские URL для индексации ({len(matched)} шт.):")
    print("=" * 65)
    for kk, _ in matched:
        print(kk)


if __name__ == "__main__":
    main()
