# -*- coding: utf-8 -*-
"""
__main__.py - Entry point for running the parser from terminal.

Usage:
  python -m app.parser --preview    # show URLs without indexing
  python -m app.parser              # index all pages (or manual URLs)
"""

import sys
import time
import argparse

from app.parser.crawler import get_urls_for_indexing, preview_urls
from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks
from app.indexer.storage import (
    ensure_collection_exists,
    save_chunks,
    get_collection_stats,
    page_needs_update,
)
from app.config import settings


def run_preview():
    """Show all URLs that would be indexed, without indexing."""
    urls = preview_urls()
    print(f"\nTotal: {len(urls)} URLs will be indexed.")
    print("To start indexing run: python -m app.parser")


def run_indexing():
    """
    Full indexing pipeline:
    1. Get URL list (manual or from sitemap)
    2. For each URL: parse -> chunk -> embed -> save to Qdrant
    3. Print summary
    """
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

    print(f"\nWill process: {len(urls)} pages\n")

    # Stats
    total_pages = 0
    total_chunks = 0
    skipped_pages = 0  # unchanged pages
    failed_pages = 0

    # ── Step 3: Process each page ─────────────────────────────
    for i, url_item in enumerate(urls, 1):
        url = url_item.url
        print(f"[{i}/{len(urls)}] {url}")

        # Download and extract text
        content = get_page_content(url)

        if not content:
            print(f"  -> Skipped (no content)")
            failed_pages += 1

            # Pause even for failed pages
            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        # Check if page changed (skip if unchanged)
        if not page_needs_update(url, content.content_hash):
            print(f"  -> Skipped (not changed)")
            skipped_pages += 1
            continue

        # Split text into chunks
        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
        )

        if not chunks:
            print(f"  -> Skipped (no chunks created)")
            failed_pages += 1

            if i < len(urls):
                time.sleep(settings.REQUEST_DELAY)
            continue

        # Save to Qdrant (creates embeddings + stores)
        saved = save_chunks(chunks, content.url, content.content_hash)

        total_pages += 1
        total_chunks += saved

        print(f"  -> Title: {content.title or 'N/A'}")
        print(f"  -> Chunks saved: {saved}")

        # Pause between requests (be respectful to the site)
        if i < len(urls):
            time.sleep(settings.REQUEST_DELAY)

    # ── Step 4: Print final summary ───────────────────────────
    stats = get_collection_stats()

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Pages indexed:    {total_pages}")
    print(f"  Pages skipped:    {skipped_pages} (unchanged)")
    print(f"  Pages failed:     {failed_pages}")
    print(f"  Chunks saved:     {total_chunks}")
    print(f"  Total in Qdrant:  {stats['total_chunks']}")
    print("=" * 60)

    if stats['total_chunks'] > 0:
        print(f"\n[OK] Data is in Qdrant and ready for search!")
        print(f"[NEXT] Run the API and test search with Postman")
    else:
        print(f"\n[!] No data saved. Check errors above.")


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
