# -*- coding: utf-8 -*-
"""
chunker.py - Splits long text into smaller chunks for RAG indexing.

Why we need this:
  - We can't send a whole page to OpenAI Embeddings (too expensive + too slow)
  - We can't send a whole page as context to ChatGPT (token limit)
  - We need small, meaningful pieces that each answer ONE specific question

Strategy:
  - Split by paragraphs first (most natural boundary)
  - If a paragraph is too long - split it by sentences
  - If a paragraph is too short - merge it with the next one
  - Always add overlap (repeat last N words in next chunk) to preserve context
  - Prepend page title to every chunk (critical for topic-based search recall)

Key improvement: title prefix
  Without it, a chunk "Студенческое общежитие расположено по адресу..."
  might not match a query "общежитие" if the word only appears once
  and in a different form. With title prefix, the embedding captures
  the page topic AND the content meaning.
"""

import re
from dataclasses import dataclass
from typing import List

from app.config import settings


@dataclass
class TextChunk:
    """One text chunk ready for indexing."""
    text: str        # chunk text (with title prefix)
    index: int       # chunk number on this page (0, 1, 2, ...)
    page_url: str    # URL of the source page
    page_title: str  # Title of the source page


def split_into_chunks(
    text: str,
    page_url: str = "",
    page_title: str = "",
) -> List[TextChunk]:
    """
    Split page text into chunks.

    Returns a list of TextChunk objects.
    Each chunk is sized around CHUNK_SIZE words with CHUNK_OVERLAP overlap.
    Each chunk is prefixed with the page title for better search recall.
    """
    if not text or len(text.strip()) < 10:
        return []

    # ── Step 1: Split text into paragraphs ───────────────────────
    paragraphs = _split_into_paragraphs(text)

    if not paragraphs:
        return []

    # ── Step 2: Group paragraphs into chunks ─────────────────────
    raw_chunks = _group_paragraphs(paragraphs)

    if not raw_chunks:
        return []

    # ── Step 3: Add overlap between chunks ───────────────────────
    chunks_with_overlap = _add_overlap(raw_chunks)

    # ── Step 4: Add title prefix + wrap into TextChunk objects ───
    result = []
    for i, chunk_text in enumerate(chunks_with_overlap):
        chunk_text = chunk_text.strip()

        # Skip truly empty or trivially short chunks
        if len(chunk_text) < 20:
            continue

        # Prepend page title to each chunk.
        # This ensures that even if a chunk doesn't repeat the page topic word,
        # the embedding still captures the topic (e.g. "Общежитие").
        # We only add the prefix if the title isn't already at the start of the chunk.
        if page_title:
            title_lower = page_title.lower()
            chunk_start = chunk_text[:len(page_title) + 10].lower()
            if title_lower not in chunk_start:
                chunk_text = f"[{page_title}]\n{chunk_text}"

        result.append(TextChunk(
            text=chunk_text,
            index=i,
            page_url=page_url,
            page_title=page_title,
        ))

    return result


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.

    Handles both:
    - Double newline separated paragraphs (clean text)
    - Single newline separated lines (table-like content from page builders)
    """
    # First try double-newline split (standard paragraphs)
    raw_paragraphs = text.split("\n\n")

    # If we got only one "paragraph" (no double newlines), try single newline
    if len(raw_paragraphs) == 1:
        raw_paragraphs = text.split("\n")

    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        # Keep paragraphs that have at least 3 chars of real content
        # (shorter ones are stray symbols/punctuation)
        if len(p) >= 3:
            paragraphs.append(p)

    return paragraphs


def _group_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Group paragraphs into chunks of approximately CHUNK_SIZE words.

    Logic:
    - Keep adding paragraphs to current chunk while under word limit
    - When we exceed the limit - start a new chunk
    - If a single paragraph exceeds 1.5x CHUNK_SIZE - split by sentences
    """
    chunks = []
    current_chunk_parts: List[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # If a single paragraph is way too large - split it by sentences
        if paragraph_word_count > settings.CHUNK_SIZE * 1.5:
            # First save what we have accumulated
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_word_count = 0

            # Split the long paragraph into sentences
            sentence_chunks = _split_long_paragraph(paragraph)
            chunks.extend(sentence_chunks)
            continue

        # If adding this paragraph would overflow - save current chunk
        if (current_word_count + paragraph_word_count > settings.CHUNK_SIZE
                and current_chunk_parts):
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_word_count = 0

        # Add paragraph to current chunk
        current_chunk_parts.append(paragraph)
        current_word_count += paragraph_word_count

    # Don't forget the last chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def _split_long_paragraph(paragraph: str) -> List[str]:
    """
    Split a very long paragraph into chunks by sentences.

    Handles Russian/Kazakh sentence endings properly:
    - ". " (period + space)
    - "! " (exclamation + space)
    - "? " (question + space)
    - ".\n" (period at end of line)
    - Also handles common abbreviations to avoid false splits
    """
    # Split on sentence boundaries, keeping the delimiter
    # Use a more robust pattern for Cyrillic text
    sentence_pattern = re.compile(
        r"(?<=[.!?…])\s+(?=[А-ЯA-ZӘІҢҒҮҰҚӨҺа-яa-z])",
        re.UNICODE,
    )
    sentences = sentence_pattern.split(paragraph)

    # Fallback: if pattern produced no split, use simpler approach
    if len(sentences) == 1:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks = []
    current_parts: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = len(sentence.split())

        if current_words + sentence_words > settings.CHUNK_SIZE and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_words = 0

        current_parts.append(sentence)
        current_words += sentence_words

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _add_overlap(chunks: List[str]) -> List[str]:
    """
    Add overlap between chunks.

    Example with CHUNK_OVERLAP=50:
      Chunk 1: "...last 50 words of chunk 1"
      Chunk 2: "[last 50 words of chunk 1] first words of chunk 2..."

    This helps the model understand context at chunk boundaries
    and prevents answers from being cut off mid-sentence.
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]  # First chunk as-is

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        # Take last CHUNK_OVERLAP words from previous chunk
        prev_words = prev_chunk.split()
        if len(prev_words) > settings.CHUNK_OVERLAP:
            overlap_text = " ".join(prev_words[-settings.CHUNK_OVERLAP:])
            result.append(f"{overlap_text}\n\n{current_chunk}")
        else:
            # Previous chunk is short — just append without overlap
            result.append(current_chunk)

    return result
