#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_manual_knowledge.py — Пересобрать чанки из manual_knowledge.json.

Запуск из корня проекта:
    python tools/rebuild_manual_knowledge.py

Что делает:
    Читает manual_knowledge.json и для каждой записи пересоздаёт чанки в Qdrant.
    Полная переиндексация сайта НЕ нужна — занимает несколько секунд.

Когда запускать:
    - После добавления или изменения записи в manual_knowledge.json
    - После правки текста, вопросов или ссылки в существующей записи

Как добавить новую запись в manual_knowledge.json:
    1. Откройте файл manual_knowledge.json (корень проекта)
    2. Добавьте объект в массив "entries":

       Для страницы с картинкой / файлом / формой:
       {
         "id": "my-page",
         "title": "Название страницы",
         "link": "https://caiu.edu.kz/my-page/",
         "chunks": [
           {
             "section": "Описание",
             "tags": ["general"],
             "text": "Что изображено / содержится на странице...",
             "questions": ["как найти эту страницу", "что там есть"],
             "skip": false
           }
         ]
       }

       Для факта без привязки к странице:
       {
         "id": "my-fact",
         "title": "ЦАИУ — какой-то факт",
         "link": null,
         "chunks": [
           {
             "section": "Раздел",
             "tags": ["general"],
             "text": "Текст факта...",
             "questions": ["как спросит пользователь"],
             "skip": false
           }
         ]
       }

    3. Запустите этот скрипт

Теги для поля 'tags':
    admission, dormitory, fees, grants, specialties, faculty, department,
    contacts, military, exams, history, management, licenses, general
"""

import sys
import os

# Добавляем rag_service в sys.path но НЕ меняем cwd на уровне модуля —
# os.chdir() как побочный эффект импорта опасен (меняет cwd всего процесса).
# Путь вычисляем один раз и вставляем в sys.path.
_RAG_SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "rag_service")
sys.path.insert(0, os.path.abspath(_RAG_SERVICE_DIR))

from app.logging_setup import setup_logging
from app.indexer.catalog_builder import build_manual_knowledge_catalog
from app.indexer.storage import ensure_collection_exists, get_collection_stats


def main():
    setup_logging("rebuild_manual_knowledge")
    print("\n" + "=" * 60)
    print("REBUILD MANUAL KNOWLEDGE")
    print("Читаю manual_knowledge.json и пересоздаю чанки...")
    print("=" * 60)

    ensure_collection_exists()

    saved = build_manual_knowledge_catalog()
    stats = get_collection_stats()

    print("\n" + "=" * 60)
    if saved > 0:
        print(f"[OK] Сохранено векторов: {saved}")
        print(f"     Итого в базе:       {stats['total_chunks']} записей")
        print("\nПроверить результат:")
        print("    python tools/debug_search.py \"структура управления\"")
        print("    python tools/debug_search.py \"когда основан ЦАИУ\"")
    else:
        print("[!] Ничего не сохранено.")
        print("    Проверьте manual_knowledge.json — возможно, entries пустые")
        print("    или все поля 'skip': true")
    print("=" * 60)


if __name__ == "__main__":
    main()
