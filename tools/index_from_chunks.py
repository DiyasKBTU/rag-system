# -*- coding: utf-8 -*-
"""
tools/index_from_chunks.py — Индексация из отредактированного JSON.

Читает chunks_export.json (созданный export_chunks.py и отредактированный вами),
обрабатывает ручные и авто-вопросы и сохраняет всё в Qdrant.

Чем отличается от обычной переиндексации:
  - Использует ваши вручную отредактированные тексты (не парсит сайт заново)
  - Поддерживает ваши теги — они записываются в payload Qdrant
  - Чанки с "skip": true — пропускаются
  - Поле "questions" в чанке:
      []            — GPT генерирует вопросы автоматически (по умолчанию 4)
      ["...", "..."] — используются ВАШИ вопросы, GPT не вызывается для этого чанка

Запуск:
  cd rag_service

  Стандартный (GPT генерирует 4 вопроса на чанк):
  python ../tools/index_from_chunks.py

  Задать другое количество авто-вопросов:
  python ../tools/index_from_chunks.py --questions-per-chunk 6

  Без генерации вопросов вообще (только текстовые чанки, быстро):
  python ../tools/index_from_chunks.py --no-questions

  Очистить базу перед индексацией:
  python ../tools/index_from_chunks.py --clear

  Тест без записи в Qdrant (dry run):
  python ../tools/index_from_chunks.py --dry-run

Примечание: чанки с ручными вопросами в поле "questions" НЕ вызывают GPT,
даже если --no-questions не указан. Ручные вопросы всегда в приоритете.
"""

import sys
import json
import uuid
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# ── Путь к rag_service чтобы импорты работали ──────────────────────────────────
_RAG_SERVICE = Path(__file__).resolve().parent.parent / "rag_service"
sys.path.insert(0, str(_RAG_SERVICE))

from app.config import settings
from app.parser.chunker import TextChunk
from app.indexer.embeddings import get_embeddings_batch
from app.indexer.storage import (
    get_client,
    get_active_collection,
    ensure_collection_exists,
    delete_chunks_by_url,
)
from app.indexer.question_generator import generate_question_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

INPUT_FILE = Path(__file__).resolve().parent.parent / "chunks_export.json"



# ── Сохранение с поддержкой тегов ─────────────────────────────────────────────

def _save_chunks_with_tags(
    chunks: List[TextChunk],
    page_url: str,
    tags: List[str],
    collection_name: str,
    dry_run: bool = False,
) -> int:
    """
    Сохраняет чанки в Qdrant с полем 'tags' в payload.
    При dry_run — только логирует, ничего не пишет.
    """
    if not chunks:
        return 0

    # Дедупликация по тексту
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.embed_text or chunk.text).strip()
        if key not in seen_texts:
            seen_texts.add(key)
            unique_chunks.append(chunk)

    duplicates = len(chunks) - len(unique_chunks)
    if duplicates:
        logger.info(f"  Удалено дублей: {duplicates}")
    chunks = unique_chunks

    if dry_run:
        logger.info(f"  [DRY RUN] Сохранил бы {len(chunks)} чанков для: {page_url}")
        return len(chunks)

    client = get_client()

    # Удаляем старые чанки этой страницы
    delete_chunks_by_url(page_url, collection_name=collection_name)

    # Эмбеддинги
    texts_for_embedding = [
        chunk.embed_text if chunk.embed_text else chunk.text
        for chunk in chunks
    ]
    vectors = get_embeddings_batch(texts_for_embedding)

    if len(vectors) != len(chunks):
        logger.error(f"  Ошибка эмбеддингов: {len(chunks)} чанков, {len(vectors)} векторов")
        return 0

    from qdrant_client.models import PointStruct
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":          chunk.text,
                "page_url":      chunk.page_url,
                "page_title":    chunk.page_title,
                "section_title": chunk.section_title,
                "chunk_index":   chunk.index,
                "content_hash":  "",      # нет хэша — вручную отредактировано
                "tags":          tags,    # ← теги пользователя
                "manually_edited": True,  # маркер ручного редактирования
            },
        ))

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


# ── Основная функция ───────────────────────────────────────────────────────────

def _manual_questions_to_chunks(
    base_chunk: TextChunk,
    questions: List[str],
) -> List[TextChunk]:
    """
    Превращает список ручных вопросов в TextChunk-объекты.

    Каждый вопрос становится отдельным вектором в Qdrant,
    но payload["text"] — оригинальный текст чанка (GPT получит его как контекст).
    Точно такой же формат как у авто-сгенерированных вопросов.
    """
    result = []
    for q in questions:
        q = q.strip()
        if not q:
            continue
        result.append(TextChunk(
            text=base_chunk.text,        # оригинальный текст → идёт в GPT
            embed_text=q,                # вопрос → превращается в вектор
            index=base_chunk.index,
            page_url=base_chunk.page_url,
            page_title=base_chunk.page_title,
            section_title=base_chunk.section_title,
        ))
    return result


def index_from_chunks(
    generate_questions: bool = True,
    questions_per_chunk: int = 4,
    clear_first: bool = False,
    dry_run: bool = False,
    input_file: Optional[Path] = None,
) -> None:
    """
    Читает JSON файл с чанками и индексирует в Qdrant.

    Логика вопросов для каждого чанка:
      1. Если в чанке есть поле "questions": ["...", "..."] — используем их,
         GPT НЕ вызывается (даже если generate_questions=True).
      2. Если "questions": [] и generate_questions=True — GPT генерирует N вопросов.
      3. Если "questions": [] и generate_questions=False — вопросов нет вообще.

    Args:
        generate_questions:  вызывать GPT для чанков без ручных вопросов
        questions_per_chunk: сколько вопросов GPT генерирует на чанк (по умолчанию 4)
        clear_first:         очистить коллекцию перед индексацией
        dry_run:             не писать в Qdrant, только показать что было бы
        input_file:          путь к JSON файлу (по умолчанию chunks_export.json)
    """
    target_file = input_file or INPUT_FILE

    if not target_file.exists():
        logger.error(f"Файл не найден: {target_file}")
        logger.error("Сначала запустите: python tools/export_chunks.py")
        sys.exit(1)

    logger.info(f"Файл:               {target_file.name}")

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if not pages:
        logger.error("В файле нет страниц (поле 'pages' пустое или отсутствует)")
        sys.exit(1)

    collection_name = get_active_collection()
    logger.info(f"Коллекция:          {collection_name}")
    logger.info(f"Страниц в файле:    {len(pages)}")
    logger.info(f"Авто-вопросов/чанк: {questions_per_chunk if generate_questions else 'отключено'}")
    if dry_run:
        logger.info("⚠ DRY RUN — данные в Qdrant НЕ записываются")

    if not dry_run:
        ensure_collection_exists(collection_name)

        if clear_first:
            logger.info("Очищаем коллекцию...")
            client = get_client()
            client.delete_collection(collection_name)
            ensure_collection_exists(collection_name)
            logger.info("Коллекция очищена")

    logger.info("=" * 60)

    total_text_chunks    = 0
    total_manual_q       = 0   # вопросы из JSON (ручные)
    total_auto_q         = 0   # вопросы от GPT
    total_saved          = 0
    pages_skipped        = 0
    pages_processed      = 0

    for i, page in enumerate(pages, 1):
        page_url   = page.get("page_url", "")
        page_title = page.get("page_title", "")
        raw_chunks = page.get("chunks", [])

        if not page_url or not raw_chunks:
            pages_skipped += 1
            continue

        # Фильтруем пропущенные чанки
        active_chunks = [c for c in raw_chunks if not c.get("skip", False)]
        skipped_count = len(raw_chunks) - len(active_chunks)

        if not active_chunks:
            logger.info(f"[{i:3}/{len(pages)}] Все чанки пропущены: {page_url}")
            pages_skipped += 1
            continue

        logger.info(
            f"[{i:3}/{len(pages)}] {page_title or page_url} "
            f"({len(active_chunks)} чанков"
            + (f", {skipped_count} пропущено" if skipped_count else "")
            + ")"
        )

        # ── Собираем TextChunk объекты и разделяем по типу вопросов ──────────
        text_chunks: List[TextChunk] = []          # оригинальные тексты
        manual_q_chunks: List[TextChunk] = []      # из JSON
        chunks_needing_auto_q: List[TextChunk] = [] # нужна генерация GPT

        for c in active_chunks:
            text = c.get("text", "").strip()
            if not text or len(text) < settings.MIN_CHUNK_LENGTH:
                continue

            base = TextChunk(
                text=text,
                index=c.get("index", 0),
                page_url=page_url,
                page_title=page_title,
                section_title=c.get("section", ""),
                embed_text="",
            )
            text_chunks.append(base)

            # Ручные вопросы из JSON?
            manual_qs = [q.strip() for q in c.get("questions", []) if q.strip()]
            if manual_qs:
                # Есть ручные вопросы — конвертируем в TextChunk и больше ничего не делаем
                manual_q_chunks.extend(_manual_questions_to_chunks(base, manual_qs))
            elif generate_questions:
                # Нет ручных, нужна авто-генерация
                chunks_needing_auto_q.append(base)
            # Иначе (нет ручных + generate_questions=False) — вопросов нет

        if not text_chunks:
            logger.warning(f"         Нет валидных чанков (слишком короткие?)")
            pages_skipped += 1
            continue

        # ── Авто-генерация через GPT (только для чанков без ручных вопросов) ─
        auto_q_chunks: List[TextChunk] = []
        if chunks_needing_auto_q:
            try:
                auto_q_chunks = generate_question_chunks(
                    chunks_needing_auto_q,
                    questions_per_chunk=questions_per_chunk,
                )
            except Exception as e:
                logger.warning(f"         Ошибка генерации вопросов: {e}")

        # ── Теги: объединение тегов всех чанков страницы ─────────────────────
        all_tags = set()
        for c in active_chunks:
            for t in c.get("tags", []):
                all_tags.add(t)
        tags = sorted(all_tags)

        # ── Сохраняем: текстовые + ручные вопросы + авто вопросы ─────────────
        all_chunks = text_chunks + manual_q_chunks + auto_q_chunks
        saved = _save_chunks_with_tags(
            chunks=all_chunks,
            page_url=page_url,
            tags=tags,
            collection_name=collection_name,
            dry_run=dry_run,
        )

        total_text_chunks += len(text_chunks)
        total_manual_q    += len(manual_q_chunks)
        total_auto_q      += len(auto_q_chunks)
        total_saved       += saved
        pages_processed   += 1

        # ── Лог с деталями ────────────────────────────────────────────────────
        parts = [f"{len(text_chunks)} текст"]
        if manual_q_chunks:
            parts.append(f"{len(manual_q_chunks)} ручных вопросов")
        if auto_q_chunks:
            parts.append(f"{len(auto_q_chunks)} авто-вопросов")
        if tags:
            logger.info(f"         Теги: {tags}")
        logger.info(f"         Сохранено: {saved} ({', '.join(parts)})")

    logger.info("=" * 60)
    logger.info("Готово!")
    logger.info(f"  Страниц обработано:  {pages_processed}")
    logger.info(f"  Страниц пропущено:   {pages_skipped}")
    logger.info(f"  Текстовых чанков:    {total_text_chunks}")
    logger.info(f"  Ручных вопросов:     {total_manual_q}")
    logger.info(f"  Авто-вопросов (GPT): {total_auto_q}")
    logger.info(f"  Итого в Qdrant:      {total_saved}")
    if dry_run:
        logger.info("  ⚠ DRY RUN — реальной записи не было")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Индексация из отредактированного JSON файла с чанками",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/index_from_chunks.py
  python tools/index_from_chunks.py --file supplement_facts.json
  python tools/index_from_chunks.py --questions-per-chunk 6
  python tools/index_from_chunks.py --no-questions
  python tools/index_from_chunks.py --clear --dry-run

Логика вопросов:
  Чанки с заполненным полем "questions" в JSON → ваши вопросы, GPT не вызывается.
  Чанки с пустым "questions" → GPT генерирует --questions-per-chunk вопросов.
  --no-questions → вопросы не генерируются ВООБЩЕ (ручные из JSON всё равно используются).
        """,
    )
    parser.add_argument(
        "--file", type=str, default=None, metavar="PATH",
        help="Путь к JSON файлу с чанками (по умолчанию chunks_export.json в корне проекта). "
             "Можно передать относительный путь от корня проекта или абсолютный.",
    )
    parser.add_argument(
        "--no-questions", action="store_true",
        help="Отключить авто-генерацию вопросов через GPT. "
             "Ручные вопросы из поля 'questions' в JSON всё равно будут использованы.",
    )
    parser.add_argument(
        "--questions-per-chunk", type=int, default=4, metavar="N",
        help="Сколько вопросов GPT генерирует на чанк (по умолчанию 4). "
             "Не влияет на чанки с ручными вопросами.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Очистить коллекцию Qdrant перед индексацией",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Показать что будет сделано, но не писать в Qdrant",
    )
    args = parser.parse_args()

    # Определяем путь к файлу
    if args.file:
        custom_path = Path(args.file)
        if not custom_path.is_absolute():
            custom_path = Path(__file__).resolve().parent.parent / args.file
        chosen_file = custom_path
    else:
        chosen_file = INPUT_FILE

    index_from_chunks(
        generate_questions=not args.no_questions,
        questions_per_chunk=args.questions_per_chunk,
        clear_first=args.clear,
        dry_run=args.dry_run,
        input_file=chosen_file,
    )
