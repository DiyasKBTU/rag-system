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

import signal
import threading
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
    save_chunks_precomputed,
    get_collection_stats,
    page_needs_update,
    url_is_manually_edited,
    get_last_indexed_at,
    set_last_indexed_at,
)
from app.indexer.embeddings import get_embeddings_batch
from app.indexer.question_generator import generate_question_chunks
from app.indexer.catalog_builder import build_and_save_catalog_chunks

logger = logging.getLogger(__name__)

# ── Graceful shutdown при активной индексации ─────────────────────────────────
# Если systemctl stop / SIGTERM приходит во время pipeline.run_indexing():
#   - без защиты: indexing.lock остаётся захваченным, fetch_cache.pkl может быть
#     частично записан, следующий старт не сможет начать индексацию.
#   - с защитом: _shutdown_requested ставится в True после текущей страницы,
#     fetch_cache.pkl сохраняется корректно, lock снимается в hot_swap.py.
#
# Важно: SIGTERM handler должен быть зарегистрирован только в основном потоке
# (CPython requirement). hot_swap.run_shadow_indexing() запускается в daemon-thread,
# поэтому регистрацию делаем в run_indexing() с проверкой is_main_thread().
_shutdown_requested = threading.Event()


def _sigterm_handler(signum, frame):
    """Устанавливает флаг завершения — pipeline корректно завершит текущую страницу."""
    logger.warning("[Pipeline] SIGTERM received — will stop after current page")
    _shutdown_requested.set()


def _register_sigterm_handler():
    """
    Регистрирует SIGTERM handler если мы в основном потоке.
    Возвращает старый handler если был установлен, иначе None.
    Сохраняем старый handler ДО регистрации нового, чтобы не потерять SIG_DFL.
    """
    if not isinstance(threading.current_thread(), threading._MainThread):
        return None
    try:
        old = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _sigterm_handler)
        return old
    except Exception:
        return None


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

    # Регистрируем SIGTERM handler (только в main thread; в daemon thread — no-op).
    # _register_sigterm_handler возвращает старый handler (или None если не в main thread).
    _shutdown_requested.clear()
    _old_sigterm = _register_sigterm_handler()
    _sigterm_registered = _old_sigterm is not None

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

    # ── Шаг 4: трёхфазная обработка (сбор → батч-эмбеддинги → сохранение) ────
    #
    # Старый подход: для каждой страницы отдельно → generate_questions → embeddings → save.
    # При 150 страницах × 20 чанков = 150 вызовов OpenAI Embeddings API.
    #
    # Новый подход:
    #   4a. Фаза сбора — проверяем все страницы, собираем чанки тех что изменились.
    #   4b. Батч-вопросы — generate_question_chunks для ВСЕХ чанков разом (1 event loop,
    #       единый AsyncOpenAI клиент и семафор вместо per-page).
    #   4c. Батч-эмбеддинги — get_embeddings_batch для ВСЕХ чанков разом (30 вызовов
    #       вместо 150 при BATCH_SIZE=100 и 3000 чанках).
    #   4d. Сохранение — save_chunks_precomputed с уже готовыми векторами.

    # ── 4a: Фаза сбора ───────────────────────────────────────────
    # pages_to_index: список (url, content_hash, title, text_chunks)
    pages_to_index: list = []

    for i, url_item in enumerate(urls, 1):
        url = url_item.url

        # Graceful shutdown: если SIGTERM пришёл — сохраняем кеш и выходим корректно.
        # hot_swap.py снимет indexing.lock в своём finally-блоке.
        if _shutdown_requested.is_set():
            logger.warning(
                f"[Pipeline] Shutdown requested at page {i}/{len(urls)} — "
                f"saving fetch cache for resume"
            )
            _save_fetch_cache(fetched_pages)
            break

        if url in lastmod_skipped_urls:
            logger.debug(f"[Pipeline] [{i}/{len(urls)}] Skipped (lastmod): {url}")
            skipped_pages += 1
            continue

        content = fetched_pages.get(url)
        if not content:
            logger.warning(f"[Pipeline] [{i}/{len(urls)}] Skipped (no content): {url}")
            failed_pages += 1
            continue

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

        pages_to_index.append((content.url, content.content_hash, content.title, chunks))
        logger.debug(f"[Pipeline] [{i}/{len(urls)}] Queued: {url} ({len(chunks)} chunks)")

    logger.info(
        f"[Pipeline] Collection phase done: {len(pages_to_index)} pages to index, "
        f"{skipped_pages} skipped, {failed_pages} failed"
    )

    if pages_to_index:
        # ── 4b: Батч-генерация вопросов для всех страниц разом ───────────
        # Раньше: N вызовов generate_question_chunks → N event loop'ов.
        # Теперь: 1 вызов с полным списком чанков → 1 event loop, 1 AsyncOpenAI
        # клиент, общий семафор — максимальный параллелизм без дублирования.
        all_text_chunks_flat = [c for _, _, _, chunks in pages_to_index for c in chunks]
        logger.info(
            f"[Pipeline] Generating questions for {len(all_text_chunks_flat)} chunks "
            f"across {len(pages_to_index)} pages..."
        )
        all_question_chunks = generate_question_chunks(all_text_chunks_flat)

        # Группируем вопрос-чанки обратно по URL
        q_by_url: dict = {}
        for qc in all_question_chunks:
            q_by_url.setdefault(qc.page_url, []).append(qc)

        # ── 4c: Батч-эмбеддинги для всех чанков сразу ────────────────────
        # Формируем плоский список всех чанков с метаинформацией о срезах.
        all_chunks_flat: list = []
        # (url, content_hash, title, n_text, n_questions, slice_start, slice_end)
        page_slices: list = []

        for url, content_hash, title, text_chunks in pages_to_index:
            question_chunks = q_by_url.get(url, [])
            combined = text_chunks + question_chunks
            start = len(all_chunks_flat)
            all_chunks_flat.extend(combined)
            end = len(all_chunks_flat)
            page_slices.append((url, content_hash, title,
                                 len(text_chunks), len(question_chunks),
                                 start, end))

        texts_for_embedding = [
            c.embed_text if c.embed_text else c.text
            for c in all_chunks_flat
        ]
        total_texts = len(texts_for_embedding)
        logger.info(
            f"[Pipeline] Computing embeddings for {total_texts} chunks "
            f"({len(pages_to_index)} pages) in one batch..."
        )
        all_vectors = get_embeddings_batch(texts_for_embedding)

        # ── 4d: Сохранение с готовыми векторами ──────────────────────────
        for url, content_hash, title, n_txt, n_q, start, end in page_slices:
            page_chunks  = all_chunks_flat[start:end]
            page_vectors = all_vectors[start:end]
            saved = save_chunks_precomputed(
                page_chunks, page_vectors, url, content_hash, collection_name=_col
            )
            total_pages     += 1
            total_chunks    += n_txt
            total_questions += n_q
            logger.info(
                f"[Pipeline] Saved {saved} ({n_txt} text + {n_q} q) for: {title or url}"
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
                # Батч-эмбеддинги — так же как для страниц сайта (оптимизация)
                doc_texts = [c.embed_text if c.embed_text else c.text for c in all_doc_chunks]
                doc_vectors = get_embeddings_batch(doc_texts)
                saved = save_chunks_precomputed(all_doc_chunks, doc_vectors, doc.source_url,
                                                content_hash, collection_name=_col)
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

    # Если пришёл SIGTERM — помечаем как прерванную, кеш уже сохранён выше.
    status = "interrupted" if _shutdown_requested.is_set() else "completed"

    result = {
        "status": status,
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
        f"[Pipeline] {status.upper()}! pages={total_pages}, skipped={skipped_pages}, "
        f"failed={failed_pages}, chunks={total_chunks}, "
        f"questions={total_questions}, catalog={catalog_saved}, "
        f"total_qdrant={stats['total_chunks']}"
    )

    if status == "completed":
        # Удаляем кеш скачивания — индексация завершена успешно
        _clear_fetch_cache()
        # Bug #7: сохраняем время завершения индексации.
        set_last_indexed_at(datetime.now(timezone.utc))

    # Восстанавливаем старый SIGTERM handler (если мы его заменяли)
    if _sigterm_registered and _old_sigterm is not None:
        try:
            signal.signal(signal.SIGTERM, _old_sigterm)
        except Exception:
            pass

    return result
