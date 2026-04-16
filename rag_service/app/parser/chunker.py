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
    text: str          # text stored in Qdrant and returned to GPT as context
    index: int         # chunk number on this page (0, 1, 2, ...)
    page_url: str      # URL of the source page
    page_title: str    # Title of the source page
    section_title: str = ""  # Section heading within the page (h2/h3/h4)
    embed_text: str = ""     # Text used for embedding (empty = use text).
                             # For question-chunks: embed_text=question, text=original chunk.


def split_into_chunks(
    text: str,
    page_url: str = "",
    page_title: str = "",
) -> List[TextChunk]:
    """
    Split page text into chunks, respecting section boundaries.

    Strategy:
    1. Split by ## markers (h2/h3/h4 headings injected by extractor)
    2. Within each section, split by word count (CHUNK_SIZE)
    3. Add overlap between chunks within the same section
    4. Prefix: [Page Title > Section] or just [Page Title]

    This way chunks stay topically focused — a chunk about "Учебные корпуса"
    will have that section title in its prefix, making it findable even
    when the chunk body text is just facts and numbers.
    """
    if not text or len(text.strip()) < 10:
        return []

    result = []
    chunk_index = 0

    # ── Step 1: Split text into sections by ## markers ───────────
    sections = _split_into_sections(text)

    for section_title, section_text in sections:
        if not section_text.strip():
            continue

        # ── Step 2: Split section into paragraphs ────────────────
        paragraphs = _split_into_paragraphs(section_text)
        if not paragraphs:
            continue

        # ── Step 3: Group paragraphs into word-count chunks ──────
        raw_chunks = _group_paragraphs(paragraphs)
        if not raw_chunks:
            continue

        # ── Step 4: Add overlap within this section ───────────────
        chunks_with_overlap = _add_overlap(raw_chunks)

        # ── Step 5: Build prefix and wrap into TextChunk ─────────
        for chunk_text in chunks_with_overlap:
            chunk_text = chunk_text.strip()

            if len(chunk_text) < 20:
                continue

            # Build prefix: [Page > Section] or [Page]
            if page_title and section_title:
                prefix = f"[{page_title} > {section_title}]"
            elif page_title:
                prefix = f"[{page_title}]"
            else:
                prefix = ""

            if prefix and not chunk_text.lower().startswith(prefix.lower()):
                chunk_text = f"{prefix}\n{chunk_text}"

            result.append(TextChunk(
                text=chunk_text,
                index=chunk_index,
                page_url=page_url,
                page_title=page_title,
                section_title=section_title,
            ))
            chunk_index += 1

    return result


def _split_into_sections(text: str) -> List[tuple]:
    """
    Split text into (section_title, section_text) pairs.

    Uses ## markers injected by extractor._inject_heading_markers().

    Example input:
        "Intro text\n\n## Учебные корпуса\nКорпус А по адресу...\n## Общежитие\n..."

    Example output:
        [("", "Intro text"), ("Учебные корпуса", "Корпус А по адресу..."), ...]

    If no ## markers exist, returns the whole text as one section with empty title.
    """
    sections: List[tuple] = []
    current_title = ""
    current_lines: List[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            # Save accumulated text as previous section
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_title, section_text))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append((current_title, section_text))

    # If nothing was parsed, return whole text as one unnamed section
    if not sections:
        return [("", text)]

    return sections


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
