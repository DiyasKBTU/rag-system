# -*- coding: utf-8 -*-
"""
debug_embeddings.py - Quick test that OpenAI embeddings work correctly.
Run: python debug_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.indexer.embeddings import get_embedding, get_embeddings_batch, estimate_cost

print("=" * 60)
print("TESTING OPENAI EMBEDDINGS")
print("=" * 60)

# Test 1: Single embedding
print("\n[Test 1] Single text embedding...")
test_text = "История Центрально-Азиатского инновационного университета"

try:
    vector = get_embedding(test_text)
    print(f"  Text:            '{test_text}'")
    print(f"  Vector length:   {len(vector)} numbers")
    print(f"  First 5 values:  {[round(v, 4) for v in vector[:5]]}")
    print(f"  Status: OK")
except Exception as e:
    print(f"  FAILED: {e}")
    print("\n  Check that OPENAI_API_KEY is set correctly in .env file")
    sys.exit(1)

# Test 2: Batch embeddings
print("\n[Test 2] Batch embeddings (3 texts)...")
test_texts = [
    "Когда начинается приём документов?",
    "Какие специальности есть в КАИУ?",
    "Контакты приёмной комиссии",
]

try:
    vectors = get_embeddings_batch(test_texts)
    print(f"  Texts sent:      {len(test_texts)}")
    print(f"  Vectors received: {len(vectors)}")
    print(f"  Each vector has: {len(vectors[0])} numbers")
    print(f"  Status: OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 3: Cost estimate for our 12 chunks
print("\n[Test 3] Cost estimate...")
num_chunks = 12
cost = estimate_cost(num_chunks, avg_chars_per_chunk=2000)
print(f"  Chunks to index:   {num_chunks}")
print(f"  Estimated cost:    ${cost:.6f} ({cost*100:.4f} cents)")
print(f"  (Basically free)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Embeddings working correctly!")
print("=" * 60)
