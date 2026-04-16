#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_search.py — Проверить поиск по конкретному вопросу.

Запуск:
    python tools/debug_search.py "сколько корпусов у университета"
    python tools/debug_search.py "история университета"

Выводит:
    - Найденные фрагменты с оценками релевантности
    - Из каких страниц они взяты
    - Помогает понять почему бот отвечает именно так

Если фрагменты не найдены — возможно нужна переиндексация:
    python tools/reindex.py
"""

import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

def main():
    if len(sys.argv) < 2:
        print("Использование: python tools/debug_search.py \"ваш вопрос\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nВопрос: {question}")
    print("=" * 60)

    from app.retrieval.search import search

    results = search(question)

    if not results:
        print("\n[!] Ничего не найдено.")
        print("    Возможные причины:")
        print("    1. Эта информация не проиндексирована")
        print("    2. Запустите: python tools/reindex.py")
        print("    3. Запустите RAG-сервис: cd rag_service && python run_api.py")
        return

    print(f"\nНайдено фрагментов: {len(results)}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r.score:.4f}")
        print(f"    Страница: {r.page_title}")
        print(f"    URL: {r.page_url}")
        print(f"    Текст: {r.text[:200].strip()}...")
        print()

if __name__ == "__main__":
    main()
