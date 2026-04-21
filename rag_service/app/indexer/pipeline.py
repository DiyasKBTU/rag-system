# -*- coding: utf-8 -*-
"""
pipeline.py — Единая функция полной индексации.

Раньше логика была продублирована в двух местах:
  - app/parser/__main__.py: run_indexing()  (запуск из CLI)
  - app/main.py:            _run_indexing() (фоновая задача FastAPI)

Теперь оба вызывают эту функцию. Дублирование устранено.

Поддерживает hot-swap режим:
  - collection_name=None  → пишет прямо в settings.QDRANT_COLLECTION_NAME
  - collection_name="..." → пишет в указанную (теневую) коллекцию

Возвращает словарь со статистикой индексации.
"""

import time
import hashlib
import logging
from typing import Optional

from app.config import settings
from app.parser.crawler import get_urls_for_indexing
from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks
from app.indexer.storage import (
    ensure_collection_exists,
    save_chunks,
    get_collection_stats,
    page_needs_update,
)
from app.indexer.question_generator import generate_question_chunks
from app.indexer.catalog_builder import build_and_save_catalog_chunks

logger = logging.getLogger(__name__)


def run_indexing(collection_name: Optional[str] = None) -> dict:
    """
    Полный пайплайн индексации сайта + Google Docs + каталог-чанки.

    Args:
        collection_name:
            None  → пишет в активную коллекцию (settings.QDRANT_COLLECTION_NAME).
            str   → пишет в указанную коллекцию (hot-swap теневая индексация).

    Returns:
        dict со статистикой:
          status         — "completed" или "failed"
          collection     — имя коллекции куда писали
          pages_indexed  — страниц успешно проиндексировано
          pages_skipped  — страниц пропущено (нет изменений)
          pages_failed   — страниц не удалось обработать
          chunks         — текстовых чанков сохранено
          questions      — вопрос-чанков сохранено
          catalog        — каталог-чанков сохранено
          total_in_qdrant— итого записей в коллекции
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME

    logger.info(f"[Pipeline] Starting indexing → collection='{_col}'")

    # ── Шаг 1: убеждаемся что коллекция существует ──────────────
    ensure_collection_exists(collection_name=_col)

    # ── Шаг 2: получаем список URL ───────────────────────────────
    urls = get_urls_for_indexing()
    if not urls:
        logger.error("[Pipeline] No URLs to index. Aborting.")
        return {
            "status": "failed",
            "collection": _col,
            "error": "No URLs found",
            "pages_indexed": 0,
            "pages_skipped": 0,
            "pages_failed": 0,
            "chunks": 0,
            "questions": 0,
            "catalog": 0,
            "total_in_qdrant": 0,
        }

    logger.info(f"[Pipeline] Processing {len(urls)} pages...")

    total_pages    = 0
    total_chunks   = 0
    total_questions = 0
    skipped_pages  = 0
    failed_pages   = 0

    # ── Шаг 3: обрабатываем каждую страницу ─────────────────────
    for i, url_item in enumerate(urls, 1):
        url = url_item.url
        logger.info(f"[Pipeline] [{i}/{len(urls)}] {url}")

        content = get_page_content(url)
        if not content:
            logger.warning(f"[Pipeline] Skipped (no content): {url}")
            failed_pages += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        if not page_needs_update(url, content.content_hash, collection_name=_col):
            logger.info(f"[Pipeline] Skipped (not changed): {url}")
            skipped_pages += 1
            continue

        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
        )
        if not chunks:
            logger.warning(f"[Pipeline] Skipped (no chunks): {url}")
            failed_pages += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        question_chunks = generate_question_chunks(chunks)
        all_chunks = chunks + question_chunks

        saved = save_chunks(all_chunks, content.url, content.content_hash,
                            collection_name=_col)
        total_pages     += 1
        total_chunks    += len(chunks)
        total_questions += len(question_chunks)

        logger.info(
            f"[Pipeline] Saved {saved} ({len(chunks)} text + {len(question_chunks)} q) "
            f"for: {content.title or url}"
        )

        if i < len(urls):
            time.sleep(settings.REQUEST_DELAY)

    # ── Шаг 4: Google Docs ───────────────────────────────────────
    all_google_docs = list(settings.GOOGLE_DOC_IDS)
    if settings.GOOGLE_DOC_ID:
        all_google_docs.insert(0, {"id": settings.GOOGLE_DOC_ID, "title": ""})

    if all_google_docs:
        try:
            from app.parser.google_docs_reader import read_google_doc
            logger.info(f"[Pipeline] Indexing {len(all_google_docs)} Google Doc(s)...")

            for doc_cfg in all_google_docs:
                doc_id    = doc_cfg.get("id", "")
                doc_title = doc_cfg.get("title", "")
                if not doc_id:
                    continue

                doc = read_google_doc(
                    doc_id=doc_id,
                    title=doc_title,
                    service_account_file=settings.GOOGLE_SERVICE_ACCOUNT_FILE,
                )
                if not doc:
                    logger.warning(f"[Pipeline] Skipped Google Doc (could not read): {doc_id}")
                    continue

                content_hash = hashlib.md5(doc.text.encode()).hexdigest()

                if not page_needs_update(doc.source_url, content_hash,
                                         collection_name=_col):
                    logger.info(f"[Pipeline] Skipped Google Doc (not changed): {doc.title or doc_id}")
                    continue

                chunks = split_into_chunks(
                    text=doc.text,
                    page_url=doc.source_url,
                    page_title=doc.title,
                )
                if not chunks:
                    logger.warning(f"[Pipeline] Skipped Google Doc (no chunks): {doc.title or doc_id}")
                    continue

                question_chunks = generate_question_chunks(chunks)
                all_doc_chunks  = chunks + question_chunks
                saved = save_chunks(all_doc_chunks, doc.source_url, content_hash,
                                    collection_name=_col)
                total_chunks    += len(chunks)
                total_questions += len(question_chunks)

                logger.info(
                    f"[Pipeline] Google Doc '{doc.title}': "
                    f"chunks={len(chunks)}, questions={len(question_chunks)}, saved={saved}"
                )
        except Exception as e:
            logger.warning(f"[Pipeline] Google Docs step failed (non-critical): {e}")

    # ── Шаг 5: каталог-чанки ─────────────────────────────────────
    catalog_saved = 0
    try:
        catalog_saved = build_and_save_catalog_chunks(collection_name=_col)
        logger.info(f"[Pipeline] Catalog chunks saved: {catalog_saved}")
    except Exception as e:
        logger.warning(f"[Pipeline] Catalog build failed (non-critical): {e}")

    # ── Итог ─────────────────────────────────────────────────────
    stats = get_collection_stats(collection_name=_col)
    result = {
        "status": "completed",
        "collection": _col,
        "pages_indexed": total_pages,
        "pages_skipped": skipped_pages,
        "pages_failed": failed_pages,
        "chunks": total_chunks,
        "questions": total_questions,
        "catalog": catalog_saved,
        "total_in_qdrant": stats["total_chunks"],
    }

    logger.info(
        f"[Pipeline] Done! pages={total_pages}, skipped={skipped_pages}, "
        f"failed={failed_pages}, chunks={total_chunks}, "
        f"questions={total_questions}, catalog={catalog_saved}, "
        f"total_qdrant={stats['total_chunks']}"
    )
    return result
