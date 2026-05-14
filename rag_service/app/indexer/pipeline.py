# -*- coding: utf-8 -*-
"""
pipeline.py — Единая функция полной индексации.

Раньше логика была продублирована в двух местах:
  - app/parser/__main__.py: run_indexing()  (запуск из CLI)
  - app/main.py:            _run_indexing() (фоновая задача FastAPI)

Теперь оба вызывают эту функцию. Дублирование устранено.

Поддерживает hot-swap режим:
  - collection_name=None  → пишет прямо в settings.QDRANT_COLLECTION_NAME
  - collection_name="..." → пишет в указанную (теневую) коллекцию

Возвращает словарь со статистикой индексации.

Оптимизация скачивания:
  Страницы скачиваются параллельно (_FETCH_WORKERS воркеров).
  Фаза 1: все URL → ThreadPoolExecutor → {url: PageContent}
  Фаза 2: последовательная обработка из кеша (hash check → chunk → questions → save)
  Время скачивания 150 страниц: было ~3-4 мин → стало ~30-40 сек.

Устойчивость к обрывам сети:
  После Фазы 1 результат сохраняется в fetch_cache.pkl.
  При следующем запуске кеш загружается и Фаза 1 пропускается.
  После успешного завершения кеш удаляется автоматически.
  Фаза 2 также устойчива: страницы уже сохранённые в Qdrant пропускаются
  через page_needs_update(), так что повторный запуск безопасен.
"""

import time
import pickle
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.parser.crawler import get_urls_for_indexing
from app.parser.extractor import get_page_content
from app.parser.chunker import split_into_chunks
from app.indexer.storage import (
    ensure_collection_exists,
    save_chunks,
    get_collection_stats,
    page_needs_update,
    url_is_manually_edited,
    get_last_indexed_at,
    set_last_indexed_at,
)
from app.indexer.question_generator import generate_question_chunks
from app.indexer.catalog_builder import build_and_save_catalog_chunks

logger = logging.getLogger(__name__)

# Файл кеша скачанных страниц — для resume при обрыве интернета.
# Хранится рядом с hot_swap_state.json (в rag_service/).
_FETCH_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "fetch_cache.pkl"

# Количество параллельных воркеров для скачивания страниц.
# 5 воркеров × 0.3 сек задержка = ~16 req/сек — комфортно для сервера.
# Менять не нужно: при >10 воркерах сервер университета может заблокировать.
_FETCH_WORKERS = 5

# Задержка внутри каждого воркера (сек).
# Итоговая нагрузка: REQUEST_DELAY / _FETCH_WORKERS = 1.5 / 5 = 0.3 сек/воркер.
_WORKER_DELAY = settings.REQUEST_DELAY / _FETCH_WORKERS


def _fetch_one_page(url_item):
    """
    Скачать одну страницу с небольшой задержкой.
    Запускается в ThreadPoolExecutor — не блокирует другие воркеры.
    """
    time.sleep(_WORKER_DELAY)
    return url_item, get_page_content(url_item.url)


def _fetch_all_pages(urls) -> dict:
    """
    Параллельно скачивает все страницы из списка URL.

    Возвращает словарь {url: PageContent | None}.
    None означает что страница не скачалась (недоступна, пустой контент).

    Прогресс выводится в лог по мере завершения воркеров.
    """
    total = len(urls)
    logger.info(f"[Pipeline] Fetching {total} pages ({_FETCH_WORKERS} workers, "
                f"{_WORKER_DELAY:.1f}s delay each)...")

    fetched: dict = {}
    done = 0

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_one_page, ui): ui for ui in urls}

        for future in as_completed(futures):
            url_item = futures[future]
            done += 1
            try:
                _, content = future.result()
            except Exception as e:
                logger.warning(f"[Pipeline] [{done}/{total}] Error fetching {url_item.url}: {e}")
                content = None

            fetched[url_item.url] = content
            status = "✓" if content else "✗"
            logger.info(f"[Pipeline] [{done}/{total}] {status} {url_item.url}")

    ok = sum(1 for v in fetched.values() if v is not None)
    logger.info(f"[Pipeline] Fetch done: {ok}/{total} pages OK")
    return fetched


# ── Кеш скачанных страниц (resume при обрыве) ────────────────────────────────

def _save_fetch_cache(fetched: dict) -> None:
    """
    Сохраняет результат параллельного скачивания на диск.
    При следующем запуске кеш будет загружен вместо повторного скачивания.

    Почему pickle, а не JSON:
      PageContent — dataclass с полем text (может быть очень длинным).
      Pickle быстрее и не требует сериализации вручную.
    """
    try:
        # Сохраняем только успешно скачанные страницы (не None)
        to_save = {url: content for url, content in fetched.items()
                   if content is not None}
        _FETCH_CACHE_FILE.write_bytes(pickle.dumps(to_save))
        logger.info(f"[Pipeline] Fetch cache saved: {len(to_save)} pages → {_FETCH_CACHE_FILE.name}")
    except Exception as e:
        logger.warning(f"[Pipeline] Could not save fetch cache (non-critical): {e}")


def _load_fetch_cache() -> dict:
    """
    Загружает кеш скачанных страниц если он существует.

    Возвращает:
        dict {url: PageContent} если кеш есть, иначе пустой dict.
    """
    if not _FETCH_CACHE_FILE.exists():
        return {}
    try:
        data = pickle.loads(_FETCH_CACHE_FILE.read_bytes())
        logger.info(
            f"[Pipeline] ♻ Loaded fetch cache: {len(data)} pages "
            f"(skipping re-download, delete {_FETCH_CACHE_FILE.name} to force full fetch)"
        )
        return data
    except Exception as e:
        logger.warning(f"[Pipeline] Could not load fetch cache ({e}), will re-fetch")
        try:
            _FETCH_CACHE_FILE.unlink()
        except Exception:
            pass
        return {}


def _clear_fetch_cache() -> None:
    """Удаляет кеш после успешной индексации."""
    try:
        if _FETCH_CACHE_FILE.exists():
            _FETCH_CACHE_FILE.unlink()
            logger.info("[Pipeline] Fetch cache cleared")
    except Exception:
        pass


def run_indexing(collection_name: Optional[str] = None) -> dict:
    """
    Полный пайплайн индексации сайта + Google Docs + каталог-чанки.

    Args:
        collection_name:
            None  → пишет в активную коллекцию (settings.QDRANT_COLLECTION_NAME).
            str   → пишет в указанную коллекцию (hot-swap теневая индексация).

    Returns:
        dict со статистикой:
          status         — "completed" или "failed"
          collection     — имя коллекции куда писали
          pages_indexed  — страниц успешно проиндексировано
          pages_skipped  — страниц пропущено (нет изменений)
          pages_failed   — страниц не удалось обработать
          chunks         — текстовых чанков сохранено
          questions      — вопрос-чанков сохранено
          catalog        — каталог-чанков сохранено
          total_in_qdrant— итого записей в коллекции
    """
    _col = collection_name or settings.QDRANT_COLLECTION_NAME

    logger.info(f"[Pipeline] Starting indexing → collection='{_col}'")

    # ── Шаг 1: убеждаемся что коллекция существует ──────────────
    ensure_collection_exists(collection_name=_col)

    # ── Шаг 2: получаем список URL ───────────────────────────────
    urls = get_urls_for_indexing()
    if not urls:
        logger.error("[Pipeline] No URLs to index. Aborting.")
        return {
            "status": "failed",
            "collection": _col,
            "error": "No URLs found",
            "pages_indexed": 0,
            "pages_skipped": 0,
            "pages_failed": 0,
            "chunks": 0,
            "questions": 0,
            "catalog": 0,
            "total_in_qdrant": 0,
        }

    # ── Шаг 3: lastmod-фильтрация + параллельное скачивание ──────
    #
    # Оптимизация (Bug #7): если у URL есть lastmod (дата изменения из sitemap)
    # и lastmod СТАРШЕ чем время последней индексации — страница не менялась.
    # Мы уже знаем что в коллекции (через snapshot) есть свежие данные → пропускаем.
    #
    # Это первый барьер (до скачивания). Второй барьер — page_needs_update() по content_hash.
    # Без lastmod-фильтра: скачиваем 150 страниц ~30 сек, потом 90% пропускаем по хэшу.
    # С lastmod-фильтром: сразу пропускаем неизменившиеся → скачиваем только реально новые.
    last_indexed = get_last_indexed_at()
    lastmod_skipped_urls: set = set()

    if last_indexed:
        for ui in urls:
            # Только если lastmod точно известен и страница не менялась с прошлой индексации
            if ui.lastmod and ui.lastmod <= last_indexed:
                lastmod_skipped_urls.add(ui.url)

        if lastmod_skipped_urls:
            logger.info(
                f"[Pipeline] lastmod filter: {len(lastmod_skipped_urls)} pages unchanged since "
                f"{last_indexed.strftime('%Y-%m-%d %H:%M')}, skipping download"
            )

    # URLs которые нужно реально скачать (lastmod не позволяет пропустить)
    urls_to_fetch = [ui for ui in urls if ui.url not in lastmod_skipped_urls]

    # Resume-логика: если прошлый запуск прервался после скачивания,
    # кеш уже есть на диске → пропускаем повторное скачивание.
    fetched_pages = _load_fetch_cache()
    if fetched_pages:
        logger.info("[Pipeline] Using cached pages from previous interrupted run")
        # Проверяем что все URL которые нужно скачать — есть в кеше
        missing = [ui.url for ui in urls_to_fetch if ui.url not in fetched_pages]
        if missing:
            logger.info(f"[Pipeline] {len(missing)} new URLs not in cache — fetching them...")
            extra = _fetch_all_pages([ui for ui in urls_to_fetch if ui.url not in fetched_pages])
            fetched_pages.update(extra)
            _save_fetch_cache(fetched_pages)
    elif urls_to_fetch:
        fetched_pages = _fetch_all_pages(urls_to_fetch)
        _save_fetch_cache(fetched_pages)
    else:
        logger.info("[Pipeline] All pages skipped by lastmod — nothing to fetch")
        fetched_pages = {}  # нет страниц для скачивания — пустой кеш

    total_pages    = 0
    total_chunks   = 0
    total_questions = 0
    skipped_pages  = 0
    failed_pages   = 0

    # ── Шаг 4: обрабатываем страницы из кеша ────────────────────
    # Порядок — как в исходном списке URL (для детерминированного лога).
    # time.sleep() убран: задержка уже была в фазе скачивания.
    for i, url_item in enumerate(urls, 1):
        url = url_item.url

        # Bug #7: страница пропущена по lastmod — уже актуальна в коллекции (snapshot)
        if url in lastmod_skipped_urls:
            logger.debug(f"[Pipeline] [{i}/{len(urls)}] Skipped (lastmod): {url}")
            skipped_pages += 1
            continue

        content = fetched_pages.get(url)

        if not content:
            logger.warning(f"[Pipeline] [{i}/{len(urls)}] Skipped (no content): {url}")
            failed_pages += 1
            continue

        # Защита ручных чанков: если URL полностью создан вручную в chunk_editor —
        # пропускаем его при автоиндексации. Ручные правки важнее автопарсинга.
        # Чтобы сбросить защиту: откройте URL в chunk_editor и пересохраните
        # хотя бы один чанк без флага manually_edited, или удалите все чанки URL.
        if url_is_manually_edited(url, collection_name=_col):
            logger.info(f"[Pipeline] [{i}/{len(urls)}] Protected (manually edited): {url}")
            skipped_pages += 1
            continue

        if not page_needs_update(url, content.content_hash, collection_name=_col):
            logger.info(f"[Pipeline] [{i}/{len(urls)}] Skipped (not changed): {url}")
            skipped_pages += 1
            continue

        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
            external_links=getattr(content, "external_links", []),
        )
        if not chunks:
            logger.warning(f"[Pipeline] [{i}/{len(urls)}] Skipped (no chunks): {url}")
            failed_pages += 1
            continue

        question_chunks = generate_question_chunks(chunks)
        all_chunks = chunks + question_chunks

        saved = save_chunks(all_chunks, content.url, content.content_hash,
                            collection_name=_col)
        total_pages     += 1
        total_chunks    += len(chunks)
        total_questions += len(question_chunks)

        logger.info(
            f"[Pipeline] [{i}/{len(urls)}] Saved {saved} "
            f"({len(chunks)} text + {len(question_chunks)} q) "
            f"for: {content.title or url}"
        )

    # ── Шаг 5: Google Docs ───────────────────────────────────────
    all_google_docs = list(settings.GOOGLE_DOC_IDS)
    if settings.GOOGLE_DOC_ID:
        all_google_docs.insert(0, {"id": settings.GOOGLE_DOC_ID, "title": ""})

    if all_google_docs:
        try:
            from app.parser.google_docs_reader import read_google_doc
            logger.info(f"[Pipeline] Indexing {len(all_google_docs)} Google Doc(s)...")

            for doc_cfg in all_google_docs:
                doc_id    = doc_cfg.get("id", "")
                doc_title = doc_cfg.get("title", "")
                if not doc_id:
                    continue

                doc = read_google_doc(
                    doc_id=doc_id,
                    title=doc_title,
                    service_account_file=settings.GOOGLE_SERVICE_ACCOUNT_FILE,
                )
                if not doc:
                    logger.warning(f"[Pipeline] Skipped Google Doc (could not read): {doc_id}")
                    continue

                content_hash = hashlib.md5(doc.text.encode()).hexdigest()

                if not page_needs_update(doc.source_url, content_hash,
                                         collection_name=_col):
                    logger.info(f"[Pipeline] Skipped Google Doc (not changed): {doc.title or doc_id}")
                    continue

                chunks = split_into_chunks(
                    text=doc.text,
                    page_url=doc.source_url,
                    page_title=doc.title,
                    external_links=[],  # Google Docs — внешних ссылок не извлекаем
                )
                if not chunks:
                    logger.warning(f"[Pipeline] Skipped Google Doc (no chunks): {doc.title or doc_id}")
                    continue

                question_chunks = generate_question_chunks(chunks)
                all_doc_chunks  = chunks + question_chunks
                saved = save_chunks(all_doc_chunks, doc.source_url, content_hash,
                                    collection_name=_col)
                total_chunks    += len(chunks)
                total_questions += len(question_chunks)

                logger.info(
                    f"[Pipeline] Google Doc '{doc.title}': "
                    f"chunks={len(chunks)}, questions={len(question_chunks)}, saved={saved}"
                )
        except Exception as e:
            logger.warning(f"[Pipeline] Google Docs step failed (non-critical): {e}")

    # ── Шаг 6: каталог-чанки ─────────────────────────────────────
    catalog_saved = 0
    try:
        catalog_saved = build_and_save_catalog_chunks(
            collection_name=_col,
            pages_cache=fetched_pages,   # Bug #3: переиспользуем уже скачанные страницы
        )
        logger.info(f"[Pipeline] Catalog chunks saved: {catalog_saved}")
    except Exception as e:
        logger.warning(f"[Pipeline] Catalog build failed (non-critical): {e}")

    # ── Итог ─────────────────────────────────────────────────────
    stats = get_collection_stats(collection_name=_col)
    result = {
        "status": "completed",
        "collection": _col,
        "pages_indexed": total_pages,
        "pages_skipped": skipped_pages,
        "pages_failed": failed_pages,
        "chunks": total_chunks,
        "questions": total_questions,
        "catalog": catalog_saved,
        "total_in_qdrant": stats["total_chunks"],
    }

    logger.info(
        f"[Pipeline] Done! pages={total_pages}, skipped={skipped_pages}, "
        f"failed={failed_pages}, chunks={total_chunks}, "
        f"questions={total_questions}, catalog={catalog_saved}, "
        f"total_qdrant={stats['total_chunks']}"
    )

    # Удаляем кеш скачивания — индексация завершена успешно
    _clear_fetch_cache()

    # Bug #7: сохраняем время завершения индексации.
    # Следующий запуск сравнит lastmod страниц с этим временем и пропустит
    # страницы которые не менялись — не нужно их даже скачивать.
    set_last_indexed_at(datetime.now(timezone.utc))

    return result
