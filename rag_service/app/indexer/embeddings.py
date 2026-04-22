# -*- coding: utf-8 -*-
"""
embeddings.py - Creates vector representations of text using OpenAI API.

What is a vector (embedding)?
  A vector is a list of ~1536 numbers that represents the MEANING of a text.
  Texts with similar meaning will have similar vectors.
  This allows us to search by meaning, not just by keywords.

Example:
  "When does admission start?" -> [0.123, -0.456, 0.789, ...]
  "Admission dates for bachelors" -> [0.124, -0.451, 0.791, ...]
  These two vectors will be very close to each other.

Cost: ~$0.02 per 1 million tokens (very cheap).
"""

import logging
import time
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI, RateLimitError, APIError
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client once (not on every call)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Maximum number of chunks per single API request
# OpenAI allows up to 2048 texts per batch request
BATCH_SIZE = 100

# Maximum chars per chunk to send to OpenAI
# text-embedding-3-small supports up to ~8000 tokens (~32000 chars)
MAX_CHARS_PER_CHUNK = 8000


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
def get_embedding(text: str) -> List[float]:
    """
    Get vector for a single text.

    Args:
        text: Any text string (question, chunk, etc.)

    Returns:
        List of 1536 floats representing the text meaning.
    """
    if len(text) > MAX_CHARS_PER_CHUNK:
        text = text[:MAX_CHARS_PER_CHUNK]

    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get vectors for a list of texts in batches.
    More efficient than calling get_embedding() one by one.

    Args:
        texts: List of text strings

    Returns:
        List of vectors, same order as input texts.
    """
    if not texts:
        return []

    # Truncate texts that are too long
    texts = [t[:MAX_CHARS_PER_CHUNK] for t in texts]

    all_vectors = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        logger.info(f"[Embeddings] Batch {batch_num}/{total_batches} ({len(batch)} texts)...")

        batch_vectors = _get_batch_with_retry(batch)
        all_vectors.extend(batch_vectors)

        # Small pause between batches to avoid rate limits
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)

    return all_vectors


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
def _get_batch_with_retry(texts: List[str]) -> List[List[float]]:
    """Send one batch to OpenAI API with automatic retry on errors."""
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    # Sort by index to guarantee input order is preserved
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def estimate_cost(num_chunks: int, avg_chars_per_chunk: int = 2000) -> float:
    """
    Estimate cost of indexing N chunks.

    Args:
        num_chunks:           Number of text chunks
        avg_chars_per_chunk:  Average chars per chunk

    Returns:
        Estimated cost in USD

    Example:
        cost = estimate_cost(100, 2000)
        # cost ≈ 0.001 (less than 0.1 cent)
    """
    # Approximate: 1 token ≈ 4 chars for mixed Russian/English text
    avg_tokens   = avg_chars_per_chunk / 4
    total_tokens = num_chunks * avg_tokens
    # text-embedding-3-small: $0.02 per 1M tokens
    return (total_tokens / 1_000_000) * 0.02
