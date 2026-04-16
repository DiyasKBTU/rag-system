# -*- coding: utf-8 -*-
"""
__main__.py - Entry point for running the parser from terminal.

Usage:
  python -m app.parser --preview    # show URLs without indexing
  python -m app.parser              # index all pages (or manual URLs)
"""

import sys
import time
import logging
import argparse

from app.logging_setup import setup_logging
from app.parser.crawler import get_urls_for_indexing, preview_urls
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
from app.config import settings


def run_preview():
    """Show all URLs that would be indexed, without indexing."""
    urls = preview_urls()
    print(f"\nTotal: {len(urls)} URLs will be indexed.")
    print("To start indexing run: python -m app.parser")


logger = logging.getLogger(__name__)


def run_indexing():
    """
    Full indexing pipeline:
    1. Get URL list (manual or from sitemap)
    2. For each URL: parse -> chunk -> generate questions -> embed -> save to Qdrant
    3. Print summary

    Question generation: for each chunk, GPT creates 4 questions that the
    chunk answers. These are saved as extra vectors pointing to the same text.
    This dramatically improves search recall — user questions match generated
    questions even when wording differs from the original text.
    """
    setup_logging("indexing")

    print("\n" + "=" * 60)
    print("STARTING INDEXING")
    print("=" * 60)

    # ── Step 1: Ensure Qdrant collection exists ────────────────
    ensure_collection_exists()

    # ── Step 2: Get URLs to index ─────────────────────────────
    urls = get_urls_for_indexing()

    if not urls:
        print("[ERROR] No URLs to index.")
        sys.exit(1)

    print(f"\nWill process: {len(urls)} pages")
    print(f"Question generation: ENABLED (4 questions per chunk, up to 5 parallel)\n")

    # Stats
    total_pages = 0
    total_chunks = 0
    total_questions = 0
    skipped_pages = 0
    failed_pages = 0

    # ── Step 3: Process each page ─────────────────────────────
    for i, url_item in enumerate(urls, 1):
        url = url_item.url
        print(f"[{i}/{len(urls)}] {url}")

        # Download and extract text
        content = get_page_content(url)

        if not content:
            logger.warning(f"[{i}/{len(urls)}] Skipped (no content): {url}")
            print(f"  -> Skipped (no content)")
            failed_pages += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        # Check if page changed (skip if unchanged)
        if not page_needs_update(url, content.content_hash):
            logger.info(f"[{i}/{len(urls)}] Skipped (not changed): {url}")
            print(f"  -> Skipped (not changed)")
            skipped_pages += 1
            continue

        # Split text into chunks (section-aware, ~200 words each)
        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
        )

        if not chunks:
            logger.warning(f"[{i}/{len(urls)}] Skipped (no chunks): {url}")
            print(f"  -> Skipped (no chunks created)")
            failed_pages += 1
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        # Generate question-chunks via GPT (runs in parallel internally)
        # Each question-chunk: embed_text=question, text=original chunk
        question_chunks = generate_question_chunks(chunks)

        # Save originals + question-chunks together in one Qdrant batch
        all_chunks = chunks + question_chunks
        saved = save_chunks(all_chunks, content.url, content.content_hash)

        total_pages += 1
        total_chunks += len(chunks)
        total_questions += len(question_chunks)

        print(f"  -> Title:     {content.title or 'N/A'}")
        print(f"  -> Chunks:    {len(chunks)}  |  Questions: {len(question_chunks)}")
        logger.info(
            f"[{i}/{len(urls)}] OK | chunks={len(chunks)} q={len(question_chunks)} | "
            f"{content.title or url}"
        )

        # Pause between requests (be respectful to the site)
        if i < len(urls):
            time.sleep(settings.REQUEST_DELAY)

    # ── Step 4: Index Google Docs (if configured) ─────────────
    all_google_docs = list(settings.GOOGLE_DOC_IDS)
    if settings.GOOGLE_DOC_ID:
        all_google_docs.insert(0, {"id": settings.GOOGLE_DOC_ID, "title": ""})

    if all_google_docs:
        from app.parser.google_docs_reader import read_google_doc
        import hashlib

        print(f"\n[Google Docs] Indexing {len(all_google_docs)} document(s)...")

        for doc_cfg in all_google_docs:
            doc_id  = doc_cfg.get("id", "")
            doc_title = doc_cfg.get("title", "")
            if not doc_id:
                continue

            print(f"  -> Reading: {doc_title or doc_id}")

            doc = read_google_doc(
                doc_id=doc_id,
                title=doc_title,
                service_account_file=settings.GOOGLE_SERVICE_ACCOUNT_FILE,
            )
            if not doc:
                print(f"  -> Skipped (could not read document)")
                continue

            content_hash = hashlib.md5(doc.text.encode()).hexdigest()

            # Skip if document hasn't changed
            if not page_needs_update(doc.source_url, content_hash):
                print(f"  -> Skipped (not changed): {doc.title}")
                continue

            chunks = split_into_chunks(
                text=doc.text,
                page_url=doc.source_url,
                page_title=doc.title,
            )
            if not chunks:
                print(f"  -> Skipped (no chunks created)")
                continue

            # Generate question-chunks (same as website pages)
            question_chunks = generate_question_chunks(chunks)

            all_chunks = chunks + question_chunks
            save_chunks(all_chunks, doc.source_url, content_hash)

            total_chunks  += len(chunks)
            total_questions += len(question_chunks)

            print(f"  -> Title:     {doc.title}")
            print(f"  -> Chunks:    {len(chunks)}  |  Questions: {len(question_chunks)}")
    else:
        print("\n[Google Docs] Not configured — skipping.")
        print("  To add: set GOOGLE_DOC_ID in rag_service/.env")

    # ── Step 5: Build catalog summary chunks ─────────────────
    # Создаём синтетические «сводные» чанки со ВСЕМИ специальностями
    # и факультетами — решает проблему неполных ответов на вопросы
    # «какие специальности», «какие факультеты».
    print("\n" + "-" * 60)
    print("BUILDING CATALOG CHUNKS (specialties + faculties)")
    print("-" * 60)
    catalog_saved = build_and_save_catalog_chunks()
    print(f"  Catalog vectors saved: {catalog_saved}")

    # ── Step 6: Print final summary ───────────────────────────
    stats = get_collection_stats()

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Pages indexed:    {total_pages}")
    print(f"  Pages skipped:    {skipped_pages} (unchanged)")
    print(f"  Pages failed:     {failed_pages}")
    print(f"  Chunks saved:     {total_chunks}")
    print(f"  Questions saved:  {total_questions}")
    print(f"  Total in Qdrant:  {stats['total_chunks']}")
    print("=" * 60)

    if stats['total_chunks'] > 0:
        print(f"\n[OK] Данные в Qdrant, готовы к поиску!")
        logger.info(
            f"DONE: pages={total_pages}, skipped={skipped_pages}, "
            f"failed={failed_pages}, chunks={total_chunks}, "
            f"questions={total_questions}, total_qdrant={stats['total_chunks']}"
        )
    else:
        print(f"\n[!] Данные не сохранены. Проверьте ошибки выше.")
        logger.error("Indexing finished but no data in Qdrant. Check errors above.")


def main():
    parser = argparse.ArgumentParser(description="CAIU RAG Indexer")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show URLs without indexing",
    )
    args = parser.parse_args()

    if args.preview:
        run_preview()
    else:
        run_indexing()


if __name__ == "__main__":
    main()
