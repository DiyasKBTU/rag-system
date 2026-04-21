# -*- coding: utf-8 -*-
"""
__main__.py - Entry point for running the parser from terminal.

Usage:
  python -m app.parser --preview    # show URLs without indexing
  python -m app.parser              # index all pages (or manual URLs)
"""

import sys
import logging
import argparse

from app.logging_setup import setup_logging
from app.parser.crawler import preview_urls


def run_preview():
    """Show all URLs that would be indexed, without indexing."""
    urls = preview_urls()
    print(f"\nTotal: {len(urls)} URLs will be indexed.")
    print("To start indexing run: python -m app.parser")


logger = logging.getLogger(__name__)


def run_indexing():
    """
    Full indexing pipeline — delegates to app.indexer.pipeline.run_indexing().

    Logic is centralised there to avoid duplication with the FastAPI hot-swap path.
    """
    setup_logging("indexing")

    print("\n" + "=" * 60)
    print("STARTING INDEXING")
    print("=" * 60)

    from app.indexer.pipeline import run_indexing as _pipeline_run
    result = _pipeline_run()   # writes to active collection (no hot-swap from CLI)

    # ── Print final summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Pages indexed:    {result.get('pages_indexed', 0)}")
    print(f"  Pages skipped:    {result.get('pages_skipped', 0)} (unchanged)")
    print(f"  Pages failed:     {result.get('pages_failed', 0)}")
    print(f"  Chunks saved:     {result.get('chunks', 0)}")
    print(f"  Questions saved:  {result.get('questions', 0)}")
    print(f"  Catalog vectors:  {result.get('catalog', 0)}")
    print(f"  Total in Qdrant:  {result.get('total_in_qdrant', 0)}")
    print("=" * 60)

    if result.get("status") == "completed" and result.get("total_in_qdrant", 0) > 0:
        print(f"\n[OK] Данные в Qdrant, готовы к поиску!")
    else:
        print(f"\n[!] Статус: {result.get('status')}. Проверьте ошибки выше.")
        if result.get("status") == "failed":
            sys.exit(1)


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
