# -*- coding: utf-8 -*-
"""
debug_page.py - Quick test of fixed extractor on one page.
Run: python debug_page.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks

TEST_URL = "https://caiu.edu.kz/history-of-the-university-ru/"

print(f"Testing extractor on: {TEST_URL}")
print("=" * 60)

content = get_page_content(TEST_URL)

if not content:
    print("FAILED: Could not extract content!")
else:
    print(f"Title:        {content.title}")
    print(f"Text length:  {len(content.text)} chars")
    print(f"Hash:         {content.content_hash}")
    print()
    print("=== FIRST 1000 CHARS OF CLEAN TEXT ===")
    print(content.text[:1000])
    print()
    print("=== CHUNKING TEST ===")
    chunks = split_into_chunks(
        text=content.text,
        page_url=content.url,
        page_title=content.title,
    )
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk.text)} chars) ---")
        print(chunk.text[:300])
        print("...")
