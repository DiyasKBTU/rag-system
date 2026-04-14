# -*- coding: utf-8 -*-
"""
debug_search.py - Test semantic search over indexed chunks.
Run: python debug_search.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retrieval.search import search, search_and_format_context
from app.indexer.storage import get_collection_stats

print("=" * 60)
print("TESTING SEMANTIC SEARCH")
print("=" * 60)

# Check how many chunks we have
stats = get_collection_stats()
print(f"\nChunks in Qdrant: {stats['total_chunks']}")

if stats['total_chunks'] == 0:
    print("[!] No data in Qdrant. Run indexing first: python -m app.parser")
    sys.exit(1)

# Test questions - real questions students might ask
test_questions = [
    "Когда был основан университет?",
    "Сколько студентов обучается в КАИУ?",
    "Какие специальности есть в университете?",
    "Как поступить в университет?",
    "Есть ли общежитие?",
]

print("\n" + "=" * 60)

for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)

    results = search(question, top_k=2, min_score=0.3)

    if not results:
        print("  No results found (try lowering min_score)")
    else:
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r.score:.3f} | {r.page_title}")
            print(f"       {r.text[:200]}...")

print("\n" + "=" * 60)
print("CONTEXT FORMAT TEST (what ChatGPT will receive)")
print("=" * 60)

question = "Когда был основан КАИУ?"
print(f"\nQuestion: {question}")
print("\nFormatted context:")
print("-" * 40)
context = search_and_format_context(question)
print(context if context else "No context found")
