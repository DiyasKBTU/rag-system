#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_search_url.py — Диагностика: почему бот не находит чанк по вопросу.

Показывает scores всех кандидатов по шагам и где конкретный чанк отваливается.

Использование:
    python debug_search_url.py
    python debug_search_url.py "кто ректор университета" https://caiu.edu.kz/rektor-czaiu/
"""

import sys
import os
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from app.config import settings
from app.indexer.embeddings import get_embedding, get_embeddings_batch
from app.retrieval.search import _expand_query, _normalize_query, _apply_score_gap_filter, _deduplicate_results, SearchResult

LINE = "─" * 65


def get_embedding_for_query(text: str) -> list:
    return get_embedding(text)


def run_raw_search(query_variants: list, top_k: int = 20, min_score: float = 0.0):
    """Запускает поиск по всем вариантам запроса без фильтров."""
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    print(f"\n  Получаю эмбеддинги для {len(query_variants)} вариантов запроса...")
    vectors = []
    for v in query_variants:
        vectors.append(get_embedding_for_query(v))

    all_results = []
    seen_ids = set()

    for variant, vector in zip(query_variants, vectors):
        hits = client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
        )
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            text_key = text[:200]
            if text_key in seen_ids:
                continue
            seen_ids.add(text_key)
            all_results.append(SearchResult(
                text=text,
                page_url=payload.get("page_url", ""),
                page_title=payload.get("page_title", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=round(hit.score, 4),
            ))

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results


def highlight_url(results, target_url):
    """Возвращает индекс результата с нужным URL, или None."""
    for i, r in enumerate(results):
        if target_url and (r.page_url == target_url or
                           r.page_url == target_url.rstrip("/") or
                           r.page_url == target_url + "/"):
            return i
    return None


def print_results_table(results, target_url=None, label="", max_show=15):
    print(f"\n  {'№':<4} {'Score':<8} {'URL (кратко)':<45} {'Заголовок'}")
    print(f"  {LINE}")
    for i, r in enumerate(results[:max_show]):
        is_target = (target_url and (
            r.page_url == target_url or
            r.page_url == target_url.rstrip("/") or
            r.page_url == target_url + "/"
        ))
        marker = " 👈 ИСКОМЫЙ" if is_target else ""
        url_short = r.page_url.replace("https://caiu.edu.kz", "")[:43]
        print(f"  {i+1:<4} {r.score:<8} {url_short:<45} {r.page_title[:30]}{marker}")
    if len(results) > max_show:
        print(f"  ... и ещё {len(results) - max_show} результатов")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) >= 1:
        question = args[0]
    else:
        print()
        question = input("  Введите вопрос боту: ").strip()

    if len(args) >= 2:
        target_url = args[1]
    else:
        target_url = input("  URL чанка который должен найтись (Enter — пропустить): ").strip() or None

    print(f"\n{'=' * 65}")
    print(f"  ДИАГНОСТИКА ПОИСКА")
    print(f"{'=' * 65}")
    print(f"  Вопрос      : {question}")
    print(f"  Цель URL    : {target_url or '(не указан)'}")
    print(f"{'=' * 65}")
    print(f"\n  Настройки фильтров из config:")
    print(f"  SIMILARITY_THRESHOLD  = {settings.SIMILARITY_THRESHOLD}  (min score при поиске)")
    print(f"  MIN_CONFIDENT_SCORE   = {settings.MIN_CONFIDENT_SCORE}  (если лучший < этого → пустой ответ)")
    print(f"  SCORE_GAP_THRESHOLD   = {settings.SCORE_GAP_THRESHOLD}  (отрезаем хвост: best - gap)")
    print(f"  MAX_CONTEXT_CHUNKS    = {settings.MAX_CONTEXT_CHUNKS}   (максимум чанков в ответ)")
    print(f"  QUERY_EXPANSION       = {settings.QUERY_EXPANSION_ENABLED}")

    # Шаг 1: нормализация и расширение запроса
    norm_q = _normalize_query(question)
    variants = _expand_query(norm_q)
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 1 — Варианты запроса после расширения ({len(variants)} шт.):")
    for i, v in enumerate(variants):
        marker = " (оригинал)" if i == 0 else f" (синоним {i})"
        print(f"    {i+1}. {v}{marker}")

    # Шаг 2: сырой поиск без фильтров
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 2 — Сырой поиск (без фильтров, top_k=20, min_score=0.0):")
    raw = run_raw_search(variants, top_k=20, min_score=0.0)
    print(f"  Найдено кандидатов: {len(raw)}")
    print_results_table(raw, target_url, max_show=15)

    if not raw:
        print("\n  ❌ Qdrant вообще ничего не вернул. Проверь подключение.")
        return

    target_idx_raw = highlight_url(raw, target_url)
    if target_url:
        if target_idx_raw is not None:
            print(f"\n  ✅ Искомый URL есть в сырых результатах (позиция #{target_idx_raw+1}, score={raw[target_idx_raw].score})")
        else:
            print(f"\n  ❌ Искомый URL ОТСУТСТВУЕТ даже в сырых результатах!")
            print(f"     Возможно URL в базе отличается (другой слеш, другое написание)")

    # Шаг 3: применяем SIMILARITY_THRESHOLD
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 3 — После фильтра SIMILARITY_THRESHOLD={settings.SIMILARITY_THRESHOLD}:")
    after_threshold = [r for r in raw if r.score >= settings.SIMILARITY_THRESHOLD]
    print(f"  Осталось: {len(after_threshold)} из {len(raw)}")
    if target_url:
        t_idx = highlight_url(after_threshold, target_url)
        if t_idx is not None:
            print(f"  ✅ Искомый URL прошёл порог (позиция #{t_idx+1}, score={after_threshold[t_idx].score})")
        elif target_idx_raw is not None:
            print(f"  ❌ Искомый URL ОТРЕЗАН на этом шаге! score={raw[target_idx_raw].score} < {settings.SIMILARITY_THRESHOLD}")
            print(f"     → Снизь SIMILARITY_THRESHOLD в config.py")

    # Шаг 4: MIN_CONFIDENT_SCORE
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 4 — Проверка MIN_CONFIDENT_SCORE={settings.MIN_CONFIDENT_SCORE}:")
    if after_threshold:
        best = after_threshold[0].score
        if best >= settings.MIN_CONFIDENT_SCORE:
            print(f"  ✅ Лучший score={best} >= {settings.MIN_CONFIDENT_SCORE} → поиск продолжается")
        else:
            print(f"  ❌ Лучший score={best} < {settings.MIN_CONFIDENT_SCORE} → бот вернёт пустой ответ!")
            print(f"     → Снизь MIN_CONFIDENT_SCORE в config.py")
    else:
        print(f"  ❌ Нет результатов после порога — бот вернёт пустой ответ")

    # Шаг 5: Score gap filter
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 5 — Score gap filter (отрезаем всё ниже best - {settings.SCORE_GAP_THRESHOLD}):")
    after_gap = _apply_score_gap_filter(after_threshold)
    if after_threshold:
        best = after_threshold[0].score
        cutoff = best - settings.SCORE_GAP_THRESHOLD
        print(f"  best={best}, cutoff={cutoff:.4f}")
        print(f"  Осталось: {len(after_gap)} из {len(after_threshold)}")
        if target_url:
            t_idx = highlight_url(after_gap, target_url)
            prev_idx = highlight_url(after_threshold, target_url)
            if t_idx is not None:
                print(f"  ✅ Искомый URL прошёл gap filter (позиция #{t_idx+1})")
            elif prev_idx is not None:
                print(f"  ❌ Искомый URL ОТРЕЗАН gap filter-ом!")
                print(f"     score={after_threshold[prev_idx].score} < cutoff {cutoff:.4f}")
                print(f"     → Увеличь SCORE_GAP_THRESHOLD в config.py")

    # Шаг 6: Дедупликация (max 2 чанка с одной страницы)
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 6 — Дедупликация (max 2 чанка с одной страницы):")
    after_dedup = _deduplicate_results(after_gap, max_per_page=2)
    print(f"  Осталось: {len(after_dedup)} из {len(after_gap)}")
    if target_url:
        t_idx = highlight_url(after_dedup, target_url)
        prev_idx = highlight_url(after_gap, target_url)
        if t_idx is not None:
            print(f"  ✅ Искомый URL прошёл дедупликацию (позиция #{t_idx+1})")
        elif prev_idx is not None:
            print(f"  ❌ Искомый URL ОТРЕЗАН при дедупликации!")
            print(f"     Уже взяли 2 чанка с этой страницы, третий отброшен")

    # Шаг 7: MAX_CONTEXT_CHUNKS
    print(f"\n  {'─'*63}")
    print(f"  ШАГ 7 — Обрезка до MAX_CONTEXT_CHUNKS={settings.MAX_CONTEXT_CHUNKS}:")
    final = after_dedup[:settings.MAX_CONTEXT_CHUNKS]
    print(f"  Финально: {len(final)} чанков")
    if target_url:
        t_idx = highlight_url(final, target_url)
        prev_idx = highlight_url(after_dedup, target_url)
        if t_idx is not None:
            print(f"  ✅ Искомый URL В ФИНАЛЬНОМ КОНТЕКСТЕ (позиция #{t_idx+1})")
        elif prev_idx is not None:
            print(f"  ❌ Искомый URL есть после дедупликации но не вошёл в топ-{settings.MAX_CONTEXT_CHUNKS}")
            print(f"     → Увеличь MAX_CONTEXT_CHUNKS в config.py")

    # Итог
    print(f"\n{'=' * 65}")
    print(f"  ИТОГ — что реально получит ChatGPT:")
    print(f"{'=' * 65}")
    if final:
        for i, r in enumerate(final):
            is_target = (target_url and (
                r.page_url == target_url or
                r.page_url == target_url.rstrip("/") or
                r.page_url == target_url + "/"
            ))
            marker = " 👈 ИСКОМЫЙ" if is_target else ""
            print(f"  #{i+1} score={r.score}  {r.page_url}{marker}")
            print(f"       {r.text[:120].replace(chr(10), ' ')}...")
            print()
        if target_url and highlight_url(final, target_url) is None:
            print(f"  ❗ Искомый чанк НЕ попал в контекст бота.")
            print(f"     Смотри выше — на каком шаге он отвалился.")
    else:
        print(f"  ❌ Финальный список ПУСТОЙ — бот ответит без контекста")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
