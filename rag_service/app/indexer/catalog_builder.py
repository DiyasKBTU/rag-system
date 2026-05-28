# -*- coding: utf-8 -*-
"""
catalog_builder.py — Построитель сводных чанков для каталогов.

Стратегия (v3):
  1. Использует СУЩЕСТВУЮЩИЙ extractor.get_page_content() — не дублирует логику.
  2. Парсит ## заголовки (h2/h3), которые уже инжектирует экстрактор.
  3. Явно знает какой URL — факультет, какой — обзорная страница кафедр.
  4. Фильтрует казахский текст по специальным символам.

Вызов:
    from app.indexer.catalog_builder import build_and_save_catalog_chunks
    build_and_save_catalog_chunks()

    # Или напрямую:
    python tools/rebuild_catalog.py
"""

import json
import logging
import re
import hashlib
import uuid
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Файл ручной базы знаний — в корне проекта (на 4 уровня выше catalog_builder.py)
_MANUAL_KNOWLEDGE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "manual_knowledge.json"

from qdrant_client.models import PointStruct

from app.config import settings
from app.indexer.embeddings import get_embeddings_batch
from app.indexer.storage import get_client, delete_chunks_by_url, page_needs_update
from app.utils import KAZAKH_CHARS

logger = logging.getLogger(__name__)

# ── Виртуальные URL синтетических чанков ─────────────────────────────────────
CATALOG_URL_FACULTIES   = "https://caiu.edu.kz/__catalog__/faculties"
CATALOG_URL_SPECIALTIES = "https://caiu.edu.kz/__catalog__/specialties"
CATALOG_URL_DEPARTMENTS = "https://caiu.edu.kz/__catalog__/departments"


def _is_kazakh(text: str) -> bool:
    """True если в тексте есть казахские буквы (не русские и не латинские)."""
    return any(ch in KAZAKH_CHARS for ch in text)


def _normalize(text: str) -> str:
    """Убирает лишние пробелы, нормализует регистр."""
    text = text.strip()
    text = re.sub(r"\s{2,}", " ", text)
    # Если весь текст в ВЕРХНЕМ регистре — приводим к нормальному
    if text == text.upper() and len(text) > 3 and any(c.isalpha() for c in text):
        text = text.capitalize()
    return text


# ── Заголовки-мусор которые не являются реальными названиями ─────────────────
_SKIP_HEADINGS = {
    "кафедры", "кафедра", "факультеты", "факультет",
    "образовательные программы", "специальности", "бакалавриат",
    "структура университета", "наши кафедры",
    "группы образовательных программ",
    "профильные предметы", "процедура приёма на бакалавриат",
}

_KAZ_SUFFIXES = (
    "кафедрасы", "факультеті", "кафедра болімі",
    "бөлімі", "мамандығы", " және ", "кафедраның",
)


def _is_valid_heading(text: str) -> bool:
    if not text or len(text) < 4:
        return False
    if _is_kazakh(text):
        return False
    if text.lower() in _SKIP_HEADINGS:
        return False
    lower = text.lower()
    if any(s in lower for s in _KAZ_SUFFIXES):
        return False
    return True


def _get_page_content(url: str, pages_cache: Optional[dict] = None):
    """
    Возвращает PageContent для URL.

    Если pages_cache передан (dict {url: PageContent}) и URL уже там —
    берём из кеша без HTTP-запроса. Это устраняет повторное скачивание
    тех же страниц которые pipeline уже скачал в фазе 1 (Bug #3).
    """
    if pages_cache and url in pages_cache:
        cached = pages_cache[url]
        if cached is not None:
            return cached
    try:
        from app.parser.extractor import get_page_content
        return get_page_content(url)
    except Exception as e:
        logger.warning(f"[Catalog] extractor error {url}: {e}")
        return None


def _parse_sections(text: str) -> List[Tuple[str, str]]:
    """
    Разбивает текст на секции по ## маркерам (как делает chunker.py).
    Возвращает список (заголовок_секции, текст_секции).
    Секции с казахским заголовком или мусорным — пропускаются.
    """
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_lines: List[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            body = "\n".join(current_lines).strip()
            if body and _is_valid_heading(current_title):
                sections.append((current_title, body))
            current_title = _normalize(line[3:])
            current_lines = []
        else:
            current_lines.append(line)

    # Последняя секция
    body = "\n".join(current_lines).strip()
    if body and _is_valid_heading(current_title):
        sections.append((current_title, body))

    return sections


def _extract_op_lines(text: str) -> List[str]:
    """
    Извлекает ТОЛЬКО строки с официальными кодами ОП (6В..., 7М...).
    Намеренно строгий фильтр — чтобы не тащить описания, биографии, заголовки.

    Примеры правильных строк:
      6В04107 – «Финансы»
      6В01503-Биология
      6В04201 — «Право»
    """
    ops = []
    seen: set = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line or _is_kazakh(line):
            continue
        if re.search(r"[67][ВBвb]\d{4,}", line):
            # Нормализуем и дедуплицируем по первым 80 символам
            key = line[:80]
            if key not in seen:
                seen.add(key)
                # Обрезаем очень длинные строки (бывают хвосты с описанием)
                ops.append(line[:120] if len(line) > 120 else line)
    return ops


def _save_synthetic_chunk(
    virtual_url: str,
    page_title: str,
    text: str,
    embed_texts: List[str],
    collection_name: Optional[str] = None,
) -> int:
    """
    Сохраняет синтетический чанк в Qdrant (один текст, N векторов).

    Bug #5: если content_hash не изменился с прошлой индексации —
    пропускаем embeddings и upsert. Экономит OpenAI-запросы когда
    каталог не менялся (типичный случай при /reindex без изменений на сайте).
    """
    if not text or not embed_texts:
        return 0

    _col = collection_name or settings.QDRANT_COLLECTION_NAME
    content_hash = hashlib.md5(text.encode()).hexdigest()

    # Skip if already saved with the same content hash (Bug #5)
    if not page_needs_update(virtual_url, content_hash, collection_name=_col):
        logger.info(f"[Catalog] Skipped (not changed): {page_title}")
        return 0

    client = get_client()
    delete_chunks_by_url(virtual_url, collection_name=_col)

    try:
        vectors = get_embeddings_batch(embed_texts)
    except Exception as e:
        logger.error(f"[Catalog] Embedding error: {e}")
        return 0

    if len(vectors) != len(embed_texts):
        return 0

    points = []
    for embed_text, vector in zip(embed_texts, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":             text,
                "page_url":         virtual_url,
                "page_title":       page_title,
                "section_title":    "",
                "chunk_index":      0,
                "content_hash":     content_hash,
                "is_catalog_chunk": True,
                "embed_text":       embed_text,
            },
        ))

    client.upsert(collection_name=_col, points=points)
    logger.info(f"[Catalog] Saved {len(points)} vectors: {page_title}")
    return len(points)


# ─────────────────────────────────────────────────────────────────────────────
# ФАКУЛЬТЕТЫ
# 4 факультета ЦАИУ — явно перечислены URL их страниц
# ─────────────────────────────────────────────────────────────────────────────

FACULTY_URLS = [
    "https://caiu.edu.kz/estestvenno-nauchnyi/",
    "https://caiu.edu.kz/pedagogiki-i-biznesa/",
    "https://caiu.edu.kz/ru-business-and-law/",
    "https://caiu.edu.kz/tvorcheskiy/",
]


def build_faculties_catalog(
    collection_name: Optional[str] = None,
    pages_cache: Optional[dict] = None,
) -> int:
    """
    4 факультета ЦАИУ.
    Скачивает каждую страницу факультета, берёт h1 как название.

    pages_cache: dict {url: PageContent} из pipeline._fetch_all_pages().
    Если URL уже в кеше — не скачиваем повторно (Bug #3).
    """
    logger.info("[Catalog] Building faculties catalog...")
    faculties: List[str] = []

    for url in FACULTY_URLS:
        content = _get_page_content(url, pages_cache=pages_cache)
        if not content:
            logger.warning(f"[Catalog] Could not fetch: {url}")
            if not (pages_cache and url in pages_cache):
                time.sleep(1)
            continue

        title = _normalize(content.title or content.text.split("\n")[0].strip())

        if title and not _is_kazakh(title) and title not in faculties:
            faculties.append(title)
            logger.info(f"[Catalog] Faculty: {title}")
        else:
            logger.warning(f"[Catalog] Bad faculty title '{title}': {url}")

        if not (pages_cache and url in pages_cache):
            time.sleep(1)

    if not faculties:
        logger.warning("[Catalog] No faculties found")
        return 0

    lines = [
        "Факультеты ЦАИУ (Центральноазиатский инновационный университет, г. Шымкент)\n",
        f"Всего факультетов: {len(faculties)}\n",
    ]
    for i, name in enumerate(faculties, 1):
        lines.append(f"{i}. {name}")
    lines.append("\nКонтакты и расписание: caiu.edu.kz/faculties-ru/")

    catalog_text = "\n".join(lines)
    logger.info(f"[Catalog] Faculties list:\n{catalog_text}")

    embed_queries = [
        "какие факультеты есть в ЦАИУ",
        "факультеты университета список",
        "сколько факультетов в ЦАИУ",
        "перечень факультетов ЦАИУ",
        "факультет бизнеса права педагогики",
        "факультеттер тізімі ЦАИУ",
        "what faculties does CAIU have",
        "список факультетов университета",
        "структура вуза факультеты",
        "факультеты и их направления",
    ]

    saved = _save_synthetic_chunk(
        virtual_url=CATALOG_URL_FACULTIES,
        page_title="Факультеты ЦАИУ — полный список",
        text=catalog_text,
        embed_texts=embed_queries,
        collection_name=collection_name,
    )
    logger.info(f"[Catalog] Faculties: {saved} vectors saved")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# КАФЕДРЫ + ОП
#
# ПРОБЛЕМА: kaferdra-ru/ рендерится через JavaScript — статический HTML
#           отдаёт только 3 кафедры из 11.
#
# РЕШЕНИЕ: 10 индивидуальных страниц кафедр скачиваются напрямую.
# ─────────────────────────────────────────────────────────────────────────────

# 10 индивидуальных страниц кафедр ЦАИУ
DEPARTMENT_URLS = [
    "https://caiu.edu.kz/business-and-tourism-ru/",
    "https://caiu.edu.kz/management-and-finance-ru/",
    "https://caiu.edu.kz/prava-ru/",
    "https://caiu.edu.kz/pedagogy-ru/",
    "https://caiu.edu.kz/languages-and-literature-ru/",
    "https://caiu.edu.kz/nvp-fk-ru/",
    "https://caiu.edu.kz/arts-ru/",
    "https://caiu.edu.kz/chemistry-biology-and-ecologyru/",
    "https://caiu.edu.kz/engineering-and-information-technology-rus/",
    "https://caiu.edu.kz/technologies-and-informatization-ru/",
]


def build_departments_catalog(
    collection_name: Optional[str] = None,
    pages_cache: Optional[dict] = None,
) -> int:
    """
    Кафедры и их образовательные программы.

    Сохраняет два типа чанков:
    1. СВОДНЫЙ — список всех 10 кафедр с ОП (вопросы: "какие кафедры", "сколько кафедр")
    2. ОТДЕЛЬНЫЙ на каждую кафедру — её название + ОП-коды
       (вопросы: "какие ОП в кафедре бизнеса", "специальности кафедры права")

    pages_cache: если передан pipeline-кеш уже скачанных страниц — не скачиваем повторно.
    """
    logger.info(f"[Catalog] Building departments catalog ({len(DEPARTMENT_URLS)} pages)...")

    dept_entries: List[tuple] = []   # (name, ops_list, url)
    total_saved = 0

    for url in DEPARTMENT_URLS:
        content = _get_page_content(url, pages_cache=pages_cache)
        if not content:
            logger.warning(f"[Catalog] Could not fetch: {url}")
            if not (pages_cache and url in pages_cache):
                time.sleep(1)
            continue

        name = _normalize(content.title)
        if not name or _is_kazakh(name):
            first_line = content.text.split("\n")[0].strip()
            name = _normalize(first_line)

        if not name or len(name) < 4:
            logger.warning(f"[Catalog] No usable title: {url}")
            if not (pages_cache and url in pages_cache):
                time.sleep(1)
            continue

        ops = _extract_op_lines(content.text)
        dept_entries.append((name, ops, url))
        logger.info(f"[Catalog] Department: {name} → {len(ops)} ОП")

        # ── Отдельный чанк на эту кафедру ──────────────────────────────────
        dept_lines = [f"Кафедра: {name}\n"]
        if ops:
            dept_lines.append("Реализуемые образовательные программы:")
            for op in ops:
                dept_lines.append(f"  - {op}")
        else:
            dept_lines.append("Образовательные программы уточняются на сайте.")
        dept_lines.append(f"\nПодробнее: {url}")
        dept_text = "\n".join(dept_lines)

        short_name = name.lower()
        dept_queries = [
            f"какие специальности в кафедре {short_name}",
            f"образовательные программы кафедры {short_name}",
            f"ОП кафедры {short_name}",
            f"что изучают на кафедре {short_name}",
            f"кафедра {short_name} специальности перечень",
            f"направления обучения кафедра {short_name}",
        ]

        slug = url.rstrip("/").split("/")[-1]
        dept_virtual_url = f"https://caiu.edu.kz/__catalog__/dept/{slug}"

        saved = _save_synthetic_chunk(
            virtual_url=dept_virtual_url,
            page_title=f"Кафедра {name} — образовательные программы",
            text=dept_text,
            embed_texts=dept_queries,
            collection_name=collection_name,
        )
        total_saved += saved
        if not (pages_cache and url in pages_cache):
            time.sleep(0.5)

    if not dept_entries:
        logger.warning("[Catalog] No departments fetched!")
        return 0

    # ── Сводный чанк — все кафедры ──────────────────────────────────────────
    summary_lines = [
        "Кафедры ЦАИУ (Центральноазиатский инновационный университет):\n",
        f"Всего кафедр: {len(dept_entries)}\n",
    ]
    for i, (name, ops, url) in enumerate(dept_entries, 1):
        summary_lines.append(f"{i}. {name}")
        for op in ops[:5]:   # краткий список — первые 5 ОП в сводке
            summary_lines.append(f"   - {op}")

    summary_lines.append("\nПодробнее: caiu.edu.kz/kaferdra-ru/")
    catalog_text = "\n".join(summary_lines)

    if len(catalog_text) > 10000:
        catalog_text = catalog_text[:10000] + "\n...(полный список: caiu.edu.kz/kaferdra-ru/)"

    logger.info(f"[Catalog] Departments summary (first 600 chars):\n{catalog_text[:600]}")

    embed_queries = [
        "кафедры ЦАИУ полный список",
        "какие кафедры есть в университете",
        "образовательные программы кафедр ЦАИУ",
        "специальности по кафедрам",
        "какие ОП на каждой кафедре",
        "кафедра математики физики информатики",
        "кафедра права экономики педагогики",
        "кафедра туризм спорт музыка",
        "departments CAIU programs",
        "кафедралар тізімі ЦАИУ",
    ]

    summary_saved = _save_synthetic_chunk(
        virtual_url=CATALOG_URL_DEPARTMENTS,
        page_title="Кафедры ЦАИУ — полный список",
        text=catalog_text,
        embed_texts=embed_queries,
        collection_name=collection_name,
    )
    total_saved += summary_saved
    logger.info(
        f"[Catalog] Departments total: {total_saved} vectors saved "
        f"({len(dept_entries)} dept chunks + {summary_saved} summary)"
    )
    return total_saved


# ─────────────────────────────────────────────────────────────────────────────
# СПЕЦИАЛЬНОСТИ (ОП) — сводный чанк только с названиями
# ─────────────────────────────────────────────────────────────────────────────

SPECIALTY_URLS = [
    "https://caiu.edu.kz/bachelor-law-ru/",
    "https://caiu.edu.kz/bachelor-customs-ru/",
    "https://caiu.edu.kz/https-caiu-edu-kz-bachelor-kaz-lang-ru/",
    "https://caiu.edu.kz/bachelor-foreign-language-ru/",
    "https://caiu.edu.kz/bachelor-nvp-ru/",
    "https://caiu.edu.kz/bachelor-sport-ru/",
    "https://caiu.edu.kz/bachelor-perevod-delo-ru/",
    "https://caiu.edu.kz/bachelor-gmu-ru/",
    "https://caiu.edu.kz/bachelor-uchet-audit-ru/",
    "https://caiu.edu.kz/bachelor-finance-ru/",
    "https://caiu.edu.kz/bachelor-turism-ru/",
    "https://caiu.edu.kz/bachelor-report-financial-analytics-ru/",
    "https://caiu.edu.kz/6b01409-ru/",
    "https://caiu.edu.kz/6b01501-ru/",
    "https://caiu.edu.kz/6b01509-ru/",
    "https://caiu.edu.kz/6b02101-ru/",
    "https://caiu.edu.kz/6b06103-ru/",
    "https://caiu.edu.kz/6b11104-ru/",
]


def build_specialties_catalog(
    collection_name: Optional[str] = None,
    pages_cache: Optional[dict] = None,
) -> int:
    """
    Список специальностей (образовательных программ) бакалавриата.
    Берём h1 с каждой страницы специальности.

    pages_cache: если передан pipeline-кеш уже скачанных страниц — не скачиваем повторно.
    """
    logger.info("[Catalog] Building specialties catalog...")
    specialties: List[str] = []

    for url in SPECIALTY_URLS:
        content = _get_page_content(url, pages_cache=pages_cache)
        if not content:
            logger.warning(f"[Catalog] Could not fetch: {url}")
            if not (pages_cache and url in pages_cache):
                time.sleep(1)
            continue

        title = _normalize(content.title)
        if title and not _is_kazakh(title) and _is_valid_heading(title) and title not in specialties:
            specialties.append(title)
            logger.info(f"[Catalog] Specialty: {title}")
        else:
            logger.debug(f"[Catalog] Skipped: {title!r} ({url})")

        if not (pages_cache and url in pages_cache):
            time.sleep(1)

    if not specialties:
        logger.warning("[Catalog] No specialties found")
        return 0

    lines = [
        "Образовательные программы (специальности) ЦАИУ — бакалавриат\n",
        f"Всего специальностей: {len(specialties)}\n",
    ]
    for i, name in enumerate(specialties, 1):
        lines.append(f"{i}. {name}")
    lines.append(
        "\nПолный список и подробности: caiu.edu.kz/op/ "
        "или приёмная комиссия: +7 707 510 10 10"
    )

    catalog_text = "\n".join(lines)
    logger.info(f"[Catalog] Specialties ({len(specialties)} total):\n{catalog_text[:600]}")

    embed_queries = [
        "какие специальности есть в ЦАИУ",
        "специальности университета полный список",
        "образовательные программы ЦАИУ бакалавриат",
        "на кого можно поступить в ЦАИУ",
        "направления подготовки перечень",
        "какие программы есть в ЦАИУ",
        "мамандықтар тізімі ЦАИУ",
        "what specialties does CAIU offer",
        "список направлений обучения бакалавр",
        "специальности для поступления",
    ]

    saved = _save_synthetic_chunk(
        virtual_url=CATALOG_URL_SPECIALTIES,
        page_title="Специальности ЦАИУ — полный список",
        text=catalog_text,
        embed_texts=embed_queries,
        collection_name=collection_name,
    )
    logger.info(f"[Catalog] Specialties: {saved} vectors saved")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL KNOWLEDGE — ручная база знаний
#
# Все записи хранятся в manual_knowledge.json (корень проекта).
# Виртуальный URL каждой записи: https://caiu.edu.kz/__manual__/{id}
# Этот URL никогда не показывается пользователю.
#
# Два типа записей:
#   - С полем link: страница с картинкой/файлом/формой.
#     link добавляется в текст (GPT цитирует) и в external_links (бот показывает).
#   - Без link: голый факт — GPT отвечает без ссылки на конкретную страницу.
#
# Добавить новую запись:
#   1. Открыть manual_knowledge.json
#   2. Добавить объект в массив "entries"
#   3. Запустить: python tools/rebuild_manual_knowledge.py
# ─────────────────────────────────────────────────────────────────────────────

def build_manual_knowledge_catalog(collection_name: Optional[str] = None) -> int:
    """
    Создаёт синтетические чанки из manual_knowledge.json.

    Два типа записей:
    - С link: страница-картинка / файл / форма.
      link сохраняется в external_links и добавляется в текст для GPT.
    - Без link: фактические данные без привязки к конкретной странице.

    Виртуальный URL: https://caiu.edu.kz/__manual__/{id}
    """
    if not _MANUAL_KNOWLEDGE_FILE.exists():
        logger.info("[Catalog] manual_knowledge.json not found — skipping manual knowledge")
        return 0

    try:
        data    = json.loads(_MANUAL_KNOWLEDGE_FILE.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
    except Exception as e:
        logger.error(f"[Catalog] Cannot read manual_knowledge.json: {e}")
        return 0

    if not entries:
        logger.info("[Catalog] manual_knowledge.json has no entries — skipping")
        return 0

    logger.info(f"[Catalog] Building manual knowledge ({len(entries)} entries)...")
    _col   = collection_name or settings.QDRANT_COLLECTION_NAME
    client = get_client()
    total_saved = 0

    for entry in entries:
        # Пропускаем технические комментарии (_note и т.д.)
        entry_id = (entry.get("id") or "").strip()
        title    = (entry.get("title") or "").strip()
        link     = (entry.get("link") or "").strip() or None
        chunks   = entry.get("chunks") or []

        if not entry_id or not chunks:
            continue  # _note-записи или незаполненные шаблоны

        virtual_url = f"https://caiu.edu.kz/__manual__/{entry_id}"

        # Удаляем старые чанки перед пересохранением
        delete_chunks_by_url(virtual_url, collection_name=_col)

        all_points  = []
        chunk_idx   = 0

        for chunk in chunks:
            if chunk.get("skip"):
                continue

            text      = (chunk.get("text") or "").strip()
            section   = (chunk.get("section") or "").strip()
            questions = [q.strip() for q in (chunk.get("questions") or []) if q.strip()]
            tags      = chunk.get("tags") or []

            if not text or not questions:
                continue

            # Если есть link — добавляем в текст чтобы GPT мог его процитировать
            chunk_text = text
            if link and link not in chunk_text:
                chunk_text = f"Страница: {link}\n\n{chunk_text}\n\nСсылка для ознакомления: {link}"

            try:
                vectors = get_embeddings_batch(questions)
            except Exception as e:
                logger.error(f"[Catalog] Embedding error for '{entry_id}': {e}")
                continue

            if len(vectors) != len(questions):
                logger.warning(f"[Catalog] Vector count mismatch for '{entry_id}' — skipping chunk")
                continue

            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()

            for question, vector in zip(questions, vectors):
                all_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text":             chunk_text,
                        "embed_text":       question,
                        "page_url":         virtual_url,
                        "page_title":       title,
                        "section_title":    section,
                        "chunk_index":      chunk_idx,
                        "content_hash":     content_hash,
                        "is_catalog_chunk": True,
                        "external_links":   [link] if link else [],
                        "tags":             tags,
                    },
                ))

            chunk_idx += 1

        if all_points:
            client.upsert(collection_name=_col, points=all_points)
            logger.info(
                f"[Catalog] Manual '{title}': {len(all_points)} vectors"
                + (f" → {link}" if link else "")
            )
            total_saved += len(all_points)
        else:
            logger.warning(f"[Catalog] Manual '{title}' ({entry_id}): no chunks saved — check questions/text")

    logger.info(f"[Catalog] Manual knowledge done: {total_saved} vectors total")
    return total_saved


# ─────────────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_catalog_chunks(
    collection_name: Optional[str] = None,
    pages_cache: Optional[dict] = None,
) -> int:
    """
    Строит все каталоги: факультеты, кафедры+ОП, специальности, ручная база знаний.

    Args:
        collection_name: Target collection. Defaults to settings value.
        pages_cache: Dict {url: PageContent} из pipeline — переиспользуем уже скачанные
                     страницы вместо повторного скачивания.
    """
    total = 0
    total += build_faculties_catalog(collection_name=collection_name, pages_cache=pages_cache)
    total += build_departments_catalog(collection_name=collection_name, pages_cache=pages_cache)
    total += build_specialties_catalog(collection_name=collection_name, pages_cache=pages_cache)
    total += build_manual_knowledge_catalog(collection_name=collection_name)
    logger.info(f"[Catalog] All done. Total catalog vectors: {total}")
    return total
