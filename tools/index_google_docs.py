#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/index_google_docs.py — Переиндексация только Google Docs без перезапуска сайта.

Использовать когда:
  - Изменили содержимое Google Документа и хочется обновить бота сразу
  - Полная переиндексация не нужна (сайт не менялся)

Запуск из корня проекта:
    cd rag_service
    python ../tools/index_google_docs.py

Флаги:
    --force    Переиндексировать даже если содержимое не изменилось
               (нужно если очистили документ — пустой хэш совпадает со старым пустым)

Пример:
    python ../tools/index_google_docs.py --force
"""

import sys
import os
import hashlib
import argparse
import logging

# Добавляем rag_service в sys.path
_RAG_SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "rag_service")
sys.path.insert(0, os.path.abspath(_RAG_SERVICE_DIR))

from app.logging_setup import setup_logging
setup_logging("index_google_docs")

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Переиндексация Google Docs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Переиндексировать даже если content_hash не изменился",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.parser.google_docs_reader import read_google_doc
    from app.parser.chunker import split_into_chunks
    from app.indexer.question_generator import generate_question_chunks
    from app.indexer.embeddings import get_embeddings_batch
    from app.indexer.storage import (
        ensure_collection_exists,
        save_chunks_precomputed,
        page_needs_update,
        delete_chunks_by_url,
        get_collection_stats,
    )

    # Собираем список всех настроенных документов
    all_docs = list(settings.GOOGLE_DOC_IDS)
    if settings.GOOGLE_DOC_ID:
        all_docs.insert(0, {"id": settings.GOOGLE_DOC_ID, "title": ""})

    if not all_docs:
        print("\n[!] В конфиге нет Google Docs.")
        print("    Заполни GOOGLE_DOC_ID или GOOGLE_DOC_IDS в rag_service/app/config.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INDEX GOOGLE DOCS")
    print(f"Документов к обработке: {len(all_docs)}")
    if args.force:
        print("Режим: FORCE (переиндексация независимо от изменений)")
    print("=" * 60)

    ensure_collection_exists()

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for doc_cfg in all_docs:
        doc_id    = doc_cfg.get("id", "").strip()
        doc_title = doc_cfg.get("title", "").strip()

        if not doc_id:
            logger.warning("[GoogleDocs] Пустой doc_id — пропускаем")
            continue

        print(f"\n→ Читаем документ: {doc_title or doc_id}")

        try:
            doc = read_google_doc(
                doc_id=doc_id,
                title=doc_title,
                service_account_file=settings.GOOGLE_SERVICE_ACCOUNT_FILE,
            )
        except Exception as e:
            print(f"  [!] Ошибка чтения документа: {e}")
            logger.error(f"[GoogleDocs] read_google_doc failed for {doc_id}: {e}")
            total_failed += 1
            continue

        if not doc:
            print(f"  [!] Документ не удалось прочитать (проверь права доступа): {doc_id}")
            total_failed += 1
            continue

        print(f"  Заголовок: {doc.title}")
        print(f"  Текст:     {len(doc.text)} символов")

        if not doc.text.strip():
            print("  [!] Документ пустой — удаляем старые чанки из Qdrant")
            logger.warning(f"[GoogleDocs] Empty document: {doc.title or doc_id}")
            delete_chunks_by_url(doc.source_url)
            total_skipped += 1
            continue

        content_hash = hashlib.md5(doc.text.encode()).hexdigest()

        if not args.force and not page_needs_update(doc.source_url, content_hash):
            print("  [=] Содержимое не изменилось — пропускаем")
            print("      (используй --force чтобы переиндексировать принудительно)")
            total_skipped += 1
            continue

        print("  Нарезаем на чанки...")
        chunks = split_into_chunks(
            text=doc.text,
            page_url=doc.source_url,
            page_title=doc.title,
            external_links=[],
        )

        if not chunks:
            print("  [!] Не удалось нарезать на чанки — пропускаем")
            total_failed += 1
            continue

        print(f"  Генерируем вопросы для {len(chunks)} чанков...")
        question_chunks = generate_question_chunks(chunks)
        all_chunks = chunks + question_chunks

        print(f"  Вычисляем эмбеддинги ({len(all_chunks)} векторов)...")
        texts = [c.embed_text if c.embed_text else c.text for c in all_chunks]
        vectors = get_embeddings_batch(texts)

        saved = save_chunks_precomputed(
            all_chunks, vectors, doc.source_url, content_hash
        )
        print(f"  [OK] Сохранено: {len(chunks)} чанков + {len(question_chunks)} вопросов = {saved} векторов")
        total_saved += saved

    # Итог
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print(f"Готово.")
    print(f"  Векторов сохранено: {total_saved}")
    print(f"  Пропущено:          {total_skipped}")
    print(f"  Ошибок:             {total_failed}")
    print(f"  Итого в Qdrant:     {stats['total_chunks']} записей")
    print("=" * 60)

    if total_saved > 0:
        print("\n[!] Redis search-кеш можно сбросить чтобы результаты были свежими:")
        print("    redis-cli -n 0 KEYS 'caiu:search:*' | xargs redis-cli -n 0 DEL")
        print("    (или подождать 1 час — кеш протухнет сам)")


if __name__ == "__main__":
    main()
