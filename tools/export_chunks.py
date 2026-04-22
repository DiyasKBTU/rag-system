# -*- coding: utf-8 -*-
"""
tools/export_chunks.py — Экспорт чанков в JSON для ручного редактирования.

Что делает:
  1. Скачивает все страницы из MANUAL_TEST_URLS
  2. Разбивает их на чанки (тот же алгоритм что при обычной индексации)
  3. Сохраняет в chunks_export.json — файл для ручного редактирования

После редактирования запустите:
  python tools/index_from_chunks.py

Формат chunks_export.json:
  Массив страниц. Каждая страница содержит список чанков.
  Каждый чанк можно:
    - Редактировать текст (поле "text")
    - Разбить на несколько (добавить новый объект в "chunks" страницы)
    - Пропустить ("skip": true)
    - Добавить теги ("tags": ["dormitory", "documents"])
    - Переименовать секцию ("section": "...")

Запуск:
  cd rag_service
  python ../tools/export_chunks.py

  Или с ограничением (первые N страниц для теста):
  python ../tools/export_chunks.py --limit 10
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

# ── Путь к rag_service чтобы импорты работали ──────────────────────────────────
_RAG_SERVICE = Path(__file__).resolve().parent.parent / "rag_service"
sys.path.insert(0, str(_RAG_SERVICE))

from app.config import settings
from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Выходной файл ──────────────────────────────────────────────────────────────
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "chunks_export.json"

# ── Доступные теги (подсказка для редактирования) ─────────────────────────────
AVAILABLE_TAGS = [
    "admission",        # поступление, документы для поступления
    "dormitory",        # общежитие, проживание
    "fees",             # стоимость обучения, контракт
    "grants",           # гранты, скидки
    "specialties",      # специальности, образовательные программы
    "faculty",          # факультеты
    "department",       # кафедры
    "contacts",         # контакты, адрес, телефон
    "military",         # военная кафедра
    "exams",            # экзамены, ЕНТ, пороговые баллы
    "history",          # история университета, о вузе
    "management",       # руководство, ректор, проректоры
    "licenses",         # лицензии, аккредитация
    "general",          # общая информация
]

# ── Автоматическое определение тегов по URL ───────────────────────────────────
# Ключ = подстрока URL → значение = список тегов
_URL_TAG_HINTS = {
    "list-of-documents":        ["admission", "documents"],
    "doc-rus":                   ["admission", "documents"],
    "quzhattar":                 ["admission", "documents"],
    "adminssions":               ["admission"],
    "priem":                     ["admission"],
    "priyomnaya":                ["admission"],
    "postupayushhim":            ["admission"],
    "bachelors-degree-process":  ["admission"],
    "qabyldau":                  ["admission"],
    "obshhezhitie":              ["dormitory"],
    "stud-dom":                  ["dormitory"],
    "svobodnye-mesta":           ["dormitory"],
    "zhataqkhana":               ["dormitory"],
    "discounts-grants":          ["fees", "grants"],
    "grants-and-discounts":      ["fees", "grants"],
    "granttar":                  ["fees", "grants"],
    "granntar":                  ["fees", "grants"],
    "ehreshold-scores":          ["exams", "admission"],
    "shekti-balldar":            ["exams", "admission"],
    "creative-exams":            ["exams"],
    "special-examination":       ["exams"],
    "arnajy-emtihan":            ["exams"],
    "shygharmashylyq":           ["exams"],
    "ent":                       ["exams"],
    "testovyj":                  ["exams"],
    "test-ortalyghy":            ["exams"],
    "bachelor-law":              ["specialties", "department"],
    "bachelor-customs":          ["specialties", "department"],
    "bachelor-kaz-lang":         ["specialties", "department"],
    "bachelor-foreign":          ["specialties", "department"],
    "bachelor-nvp":              ["specialties", "department"],
    "bachelor-sport":            ["specialties", "department"],
    "bachelor-perevod":          ["specialties", "department"],
    "bachelor-gmu":              ["specialties", "department"],
    "bachelor-uchet":            ["specialties", "department"],
    "bachelor-finance":          ["specialties", "department"],
    "bachelor-turism":           ["specialties", "department"],
    "bachelor-report":           ["specialties", "department"],
    "6b":                        ["specialties"],
    "op/":                       ["specialties"],
    "ru-bachelor":               ["specialties"],
    "obrazovatelnye":            ["specialties"],
    "bb-kaz":                    ["specialties"],
    "bakalavriat":               ["specialties"],
    "bilim-beru":                ["specialties"],
    "groups-of-educational":     ["specialties"],
    "faculties":                 ["faculty"],
    "kk-faculty":                ["faculty"],
    "kafedra":                   ["department"],
    "kafedralar":                ["department"],
    "estestvenno-nauchnyi":      ["faculty", "department"],
    "pedagogiki-i-biznesa":      ["faculty", "department"],
    "ru-business-and-law":       ["faculty", "department"],
    "tvorcheskiy":               ["faculty", "department"],
    "contacts":                  ["contacts"],
    "kontakty":                  ["contacts"],
    "bajlanystar":               ["contacts"],
    "call-center":               ["contacts"],
    "military-department":       ["military"],
    "aeskeri-kafedra":           ["military"],
    "history-of-the-university": ["history"],
    "mission-vision":            ["history", "general"],
    "licenses":                  ["licenses"],
    "akkreditaciya":             ["licenses"],
    "naar":                      ["licenses"],
    "rektor":                    ["management"],
    "prorektor":                 ["management"],
    "prorector":                 ["management"],
    "vice-rector":               ["management"],
    "administration":            ["management"],
    "struktura":                 ["general"],
    "putevoditel":               ["general"],
    "kaz-guide":                 ["general"],
}


def _guess_tags(url: str) -> list:
    """Определяет теги автоматически по URL."""
    url_lower = url.lower()
    tags = []
    seen = set()
    for pattern, tag_list in _URL_TAG_HINTS.items():
        if pattern in url_lower:
            for t in tag_list:
                if t not in seen:
                    seen.add(t)
                    tags.append(t)
    return tags


def export_chunks(limit: int = None) -> None:
    """
    Основная функция экспорта.

    Args:
        limit: ограничить количество страниц (для тестирования)
    """
    urls = settings.MANUAL_TEST_URLS
    if limit:
        urls = urls[:limit]

    logger.info(f"Экспортирую чанки из {len(urls)} страниц → {OUTPUT_FILE.name}")
    logger.info("=" * 60)

    pages_data = []
    total_chunks = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i:3}/{len(urls)}] {url}")

        content = get_page_content(url)
        if not content or not content.text.strip():
            logger.warning(f"         Нет контента — пропущено")
            failed += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
        )

        if not chunks:
            logger.warning(f"         Нет чанков — пропущено")
            failed += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        auto_tags = _guess_tags(url)

        page_entry = {
            "page_url":   content.url,
            "page_title": content.title or "",
            "auto_tags":  auto_tags,   # подсказка — не редактировать вручную
            "chunks": [],
        }

        for chunk in chunks:
            page_entry["chunks"].append({
                "index":   chunk.index,
                "section": chunk.section_title or "",
                "tags":    list(auto_tags),  # копия — пользователь редактирует эти
                "skip":    False,
                "text":    chunk.text,
                # ── Ручные вопросы ────────────────────────────────────────────
                # Оставьте пустым [] — GPT сгенерирует автоматически.
                # Или впишите свои вопросы — тогда GPT не вызывается для этого чанка.
                # Пример:
                #   "questions": [
                #     "Какие документы нужны для поступления?",
                #     "Что подать в приёмную комиссию?"
                #   ]
                "questions": [],
            })

        pages_data.append(page_entry)
        total_chunks += len(chunks)
        logger.info(f"         → {len(chunks)} чанков  (теги: {auto_tags or 'нет'})")

        if i < len(urls):
            time.sleep(settings.REQUEST_DELAY)

    # ── Сохраняем ────────────────────────────────────────────────────────────
    output = {
        "_info": {
            "total_pages":  len(pages_data),
            "total_chunks": total_chunks,
            "failed_pages": failed,
            "available_tags": AVAILABLE_TAGS,
            "how_to_edit": (
                "ТЕКСТ: редактируйте поле 'text' свободно — убирайте мусор, исправляйте. "
                "РАЗБИТЬ чанк — добавьте новый объект в массив 'chunks' той же страницы. "
                "ПРОПУСТИТЬ чанк — 'skip': true. "
                "ТЕГИ — поле 'tags', влияет на поиск. "
                "ВОПРОСЫ — поле 'questions': [] пустой = GPT генерирует сам; "
                "заполните своими вопросами — GPT не вызывается для этого чанка. "
                "Количество авто-вопросов задаётся флагом --questions-per-chunk N "
                "при запуске index_from_chunks.py (по умолчанию 4). "
                "После редактирования: python tools/index_from_chunks.py"
            ),
        },
        "pages": pages_data,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"Готово!")
    logger.info(f"  Страниц экспортировано: {len(pages_data)}")
    logger.info(f"  Страниц с ошибками:     {failed}")
    logger.info(f"  Чанков всего:           {total_chunks}")
    logger.info(f"  Файл:                   {OUTPUT_FILE}")
    logger.info("")
    logger.info("Следующий шаг:")
    logger.info("  1. Откройте chunks_export.json в любом редакторе (VS Code)")
    logger.info("  2. Отредактируйте чанки как нужно")
    logger.info("  3. python tools/index_from_chunks.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Экспорт чанков для ручного редактирования"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Обработать только первые N страниц (для тестирования)"
    )
    args = parser.parse_args()
    export_chunks(limit=args.limit)
