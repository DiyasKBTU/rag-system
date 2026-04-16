# -*- coding: utf-8 -*-
"""
debug_catalog.py — Диагностика: показывает что видит экстрактор.

Режимы:
    python ../tools/debug_catalog.py                    # проверяет все кафедры + kaferdra-ru/
    python ../tools/debug_catalog.py <URL>              # проверяет одну страницу подробно

Запуск из папки rag_service/
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_service"))

from app.parser.extractor import get_page_content
from app.indexer.catalog_builder import DEPARTMENT_URLS, FACULTY_URLS, _is_kazakh, _extract_op_lines, _normalize

_KAZAKH_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def check_single_url(url: str):
    """Детальная проверка одной страницы."""
    print("=" * 70)
    print(f"ПОДРОБНАЯ ПРОВЕРКА: {url}")
    print("=" * 70)

    content = get_page_content(url)
    if not content:
        print("[!] Не удалось скачать страницу!")
        return

    print(f"\n  title:      {content.title!r}")
    print(f"  text len:   {len(content.text)} символов")

    headers = [l for l in content.text.split("\n") if l.startswith("## ")]
    print(f"  ## headers: {len(headers)}")
    for h in headers[:10]:
        kaz = " [KAZ]" if _is_kazakh(h) else ""
        print(f"    {h.strip()}{kaz}")

    ops = _extract_op_lines(content.text)
    print(f"\n  ОП-строки ({len(ops)}):")
    for op in ops[:15]:
        print(f"    → {op}")

    print(f"\n  Первые 1500 символов текста:")
    print("-" * 40)
    print(content.text[:1500])
    print("-" * 40)


def check_all_departments():
    """Быстрая проверка всех 10 кафедр."""
    print("=" * 70)
    print("ПРОВЕРКА ВСЕХ КАФЕДР")
    print("=" * 70)

    ok = 0
    failed = []

    for i, url in enumerate(DEPARTMENT_URLS, 1):
        content = get_page_content(url)
        if not content:
            print(f"\n  [{i:2d}] ОШИБКА — {url}")
            failed.append(url)
            continue

        name = _normalize(content.title)
        is_kaz = _is_kazakh(name)
        ops = _extract_op_lines(content.text)
        headers = [l for l in content.text.split("\n") if l.startswith("## ")]

        status = "KAZ!" if is_kaz else "OK "
        print(f"\n  [{i:2d}] [{status}] {name!r}")
        print(f"         text={len(content.text)}ч, ##={len(headers)}, ОП={len(ops)}")
        if ops:
            for op in ops[:3]:
                print(f"         → {op[:70]}")
        if not is_kaz:
            ok += 1

    print(f"\n{'=' * 70}")
    print(f"Итог: {ok}/{len(DEPARTMENT_URLS)} кафедр успешно")
    if failed:
        print(f"Не удалось скачать ({len(failed)}):")
        for u in failed:
            print(f"  - {u}")

    print("\nЧтобы подробно посмотреть страницу:")
    print("  python ../tools/debug_catalog.py <URL>")


# ── Точка входа ───────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    check_single_url(sys.argv[1])
else:
    check_all_departments()
