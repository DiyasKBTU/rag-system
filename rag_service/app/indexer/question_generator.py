# -*- coding: utf-8 -*-
"""
question_generator.py - Generates questions for each chunk during indexing.

Why this matters:
  Users ask questions. The knowledge base stores answers.
  Without this, "сколько корпусов?" may not find a chunk that says
  "Университет располагает 4 корпусами" because the embedding vectors
  don't perfectly overlap.

  Solution: for each chunk, ask GPT to generate 4 questions that this
  chunk answers. Store those questions as extra vectors pointing back
  to the same original chunk text.

  Now "сколько корпусов?" matches the generated question
  "Сколько учебных корпусов в университете?" — exact semantic hit.

How it works:
  - Original chunk: text="[История > Корпуса]\nКорпус А...", embed_text=""
    → vector is built from the chunk text itself
  - Question chunk: text="[История > Корпуса]\nКорпус А...", embed_text="Сколько корпусов?"
    → vector is built from the QUESTION, but GPT gets the ORIGINAL text

Parallelism:
  Uses asyncio + AsyncOpenAI internally with a semaphore (BATCH_CONCURRENCY).
  The public function generate_question_chunks() is still synchronous —
  __main__.py and other callers don't need to change.
  At 5 concurrent requests, 400 chunks take ~2-3 min instead of 10-15 min.

Storage: original chunks + question chunks are saved together in Qdrant.
"""

import asyncio
import sys
import logging
from typing import List

from openai import AsyncOpenAI

# Windows + Python 3.12: asyncio.run() закрывает event loop до того как httpx
# успевает закрыть соединения → RuntimeError: Event loop is closed.
# WindowsSelectorEventLoopPolicy решает это без изменения логики.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.parser.chunker import TextChunk
from app.config import settings
from app.utils import detect_lang

logger = logging.getLogger(__name__)

# How many questions to generate per chunk.
# 4 is a good balance: covers main angles without too many API calls.
QUESTIONS_PER_CHUNK = 4

# Max parallel GPT requests.
# 10 параллельных запросов безопасны для OpenAI tier-1 (лимит ~60 RPM).
# При большем количестве 429 начинают встречаться чаще — оставляем 10.
# Если будут ошибки 429 — откатите до 5.
BATCH_CONCURRENCY = 10

# GPT prompts for question generation — по языку чанка
_SYSTEM_PROMPT_RU = (
    "Ты помощник для системы поиска информации об университете ЦАИУ (Казахстан). "
    "Генерируй вопросы которые могут задавать абитуриенты и студенты. "
    "Вопросы должны быть краткими, на русском языке. "
    "Отвечай ТОЛЬКО вопросами, каждый на новой строке, без нумерации и пояснений."
)

_SYSTEM_PROMPT_KK = (
    "Сен ЦАИУ университеті (Қазақстан) туралы ақпарат іздеу жүйесінің көмекшісісің. "
    "Абитуриенттер мен студенттер қоюы мүмкін сұрақтарды жаз. "
    "Сұрақтар қысқа, қазақ тілінде болуы керек. "
    "ТЕК сұрақтарды жаз, әр жолда бір сұрақ, нөмірсіз."
)

_USER_PROMPT_TEMPLATE_RU = """Текст из базы знаний:
{chunk_text}

Сгенерируй {n} вопроса на русском языке, на которые отвечает этот текст.
Каждый вопрос с новой строки, без номеров."""

_USER_PROMPT_TEMPLATE_KK = """Білім базасынан мәтін:
{chunk_text}

Осы мәтін жауап беретін {n} сұрақ жаз. Қазақ тілінде.
Әр сұрақ жаңа жолдан, нөмірсіз."""


# detect_lang imported from app.utils — избегаем дублирования константы KAZAKH_CHARS


# ── Public API ────────────────────────────────────────────────────────────────

def generate_question_chunks(
    chunks: List[TextChunk],
    questions_per_chunk: int = QUESTIONS_PER_CHUNK,
) -> List[TextChunk]:
    """
    Generate question-chunks for a list of chunks.

    Runs parallel GPT calls internally (up to BATCH_CONCURRENCY at once).
    The function itself is synchronous — callers don't need to use asyncio.

    For each chunk, calls GPT to get `questions_per_chunk` questions.
    Returns a flat list of new TextChunk objects where:
      - text       = original chunk text (what GPT will receive as context)
      - embed_text = the generated question (what gets embedded for search)
      - All other fields copied from the original chunk

    These are saved alongside original chunks in Qdrant.

    Args:
        chunks:              список чанков для генерации вопросов
        questions_per_chunk: сколько вопросов генерировать на чанк (по умолчанию 4)

    Примечание: используем явно созданный event loop вместо asyncio.run(),
    чтобы избежать конфликтов когда вызываем из потока (threading.Thread).
    asyncio.run() поднимает RuntimeError если в текущем потоке уже есть
    запущенный loop. Явный loop полностью изолирован.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _generate_all_async(chunks, questions_per_chunk=questions_per_chunk)
        )
    finally:
        loop.close()


# ── Internal async implementation ─────────────────────────────────────────────

async def _generate_all_async(
    chunks: List[TextChunk],
    questions_per_chunk: int = QUESTIONS_PER_CHUNK,
) -> List[TextChunk]:
    """Run all chunk question generation in parallel with a semaphore."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

    tasks = [
        _process_chunk(chunk, client, semaphore, n=questions_per_chunk)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_question_chunks: List[TextChunk] = []
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            logger.warning(
                f"[QuestionGen] Failed for chunk {chunk.index} from {chunk.page_url}: {result}"
            )
            continue
        for question in result:
            all_question_chunks.append(TextChunk(
                text=chunk.text,         # original text → returned to GPT
                embed_text=question,     # question → embedded for search
                index=chunk.index,
                page_url=chunk.page_url,
                page_title=chunk.page_title,
                section_title=chunk.section_title,
            ))

    logger.info(
        f"[QuestionGen] Done: {len(chunks)} chunks → {len(all_question_chunks)} question-chunks"
    )
    return all_question_chunks


async def _process_chunk(
    chunk: TextChunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    n: int = QUESTIONS_PER_CHUNK,
) -> List[str]:
    """
    Generate questions for a single chunk, respecting the concurrency semaphore.
    Returns a list of question strings (may be empty on failure or short chunks).
    """
    chunk_text = chunk.text.strip()
    if not chunk_text or len(chunk_text) < 30:
        return []

    # Выбираем промпт по языку чанка
    lang = detect_lang(chunk_text)
    system_prompt = _SYSTEM_PROMPT_KK if lang == "kk" else _SYSTEM_PROMPT_RU
    user_template = _USER_PROMPT_TEMPLATE_KK if lang == "kk" else _USER_PROMPT_TEMPLATE_RU

    prompt = user_template.format(
        chunk_text=chunk_text[:1500],  # limit input to save tokens
        n=n,
    )

    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
            max_tokens=150,
        )

    raw = response.choices[0].message.content.strip()

    # Parse: one question per line, skip empty / too-short lines
    questions = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("-•*0123456789.). ")
        if len(line) >= 10 and "?" in line:
            questions.append(line)

    return questions[:n]
