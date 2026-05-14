# -*- coding: utf-8 -*-
"""
Юнит-тесты для app.retrieval.search (чистые функции, без Qdrant/Redis)

Запуск:
    cd rag_service
    python -m pytest ../tests/test_search.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_service"))

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List


# ── Минимальный мок settings ──────────────────────────────────────────────────
class FakeSettings:
    QUERY_EXPANSION_ENABLED = True
    SCORE_GAP_THRESHOLD = 0.25
    MIN_CONFIDENT_SCORE = 0.28
    MAX_CONTEXT_CHUNKS = 5
    TOP_K_RESULTS = 8
    QDRANT_COLLECTION_NAME = "test_collection"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    OPENAI_API_KEY = "test-key"


# Патчим до импорта search.py чтобы не нужен .env
@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    import app.config as config_module
    monkeypatch.setattr(config_module, "settings", FakeSettings())

    # Мокаем все внешние зависимости search.py
    mock_qdrant = MagicMock()
    mock_redis = MagicMock(return_value=None)
    mock_embeddings = MagicMock(return_value=[[0.1] * 1536])

    monkeypatch.setattr("app.indexer.storage.get_client", mock_qdrant)
    monkeypatch.setattr("app.indexer.storage._client", None)
    monkeypatch.setattr("app.retrieval.search.get_redis", mock_redis)


# Импортируем функции после патчинга
from app.retrieval.search import (
    _normalize_query,
    _expand_query,
    _deduplicate_results,
    _apply_score_gap_filter,
    SearchResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная фабрика SearchResult
# ─────────────────────────────────────────────────────────────────────────────

def make_result(url: str, chunk_index: int = 0, score: float = 0.8,
                text: str = "текст") -> SearchResult:
    return SearchResult(
        text=text,
        page_url=url,
        page_title="Тестовая страница",
        chunk_index=chunk_index,
        score=score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_query
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeQuery:

    def test_strips_whitespace(self):
        assert _normalize_query("  вопрос  ") == "вопрос"

    def test_collapses_multiple_spaces(self):
        assert _normalize_query("сколько   корпусов") == "сколько корпусов"

    def test_empty_string(self):
        assert _normalize_query("") == ""

    def test_already_clean(self):
        assert _normalize_query("общежитие цаиу") == "общежитие цаиу"


# ─────────────────────────────────────────────────────────────────────────────
# _expand_query
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandQuery:

    def test_always_includes_original(self):
        result = _expand_query("сколько стоит обучение")
        assert "сколько стоит обучение" in result

    def test_expansion_for_known_keyword(self):
        """'общежитие' — ключ в SYNONYM_MAP, должны появиться синонимы."""
        result = _expand_query("есть ли общежитие в цаиу")
        assert len(result) > 1  # нашли синонимы

    def test_expansion_for_specialty(self):
        """'специальности' → должны добавиться варианты."""
        result = _expand_query("какие специальности есть в цаиу")
        assert len(result) > 1

    def test_unknown_query_returns_only_original(self):
        """Запрос без совпадений в SYNONYM_MAP → только оригинал."""
        result = _expand_query("абвгд ёжзий клмнопрст")
        assert result == ["абвгд ёжзий клмнопрст"]

    def test_max_three_queries(self):
        """Не более 3 вариантов запроса."""
        result = _expand_query("стоимость обучения общежитие документы")
        assert len(result) <= 3

    def test_no_duplicate_original_in_expansions(self):
        """Оригинал не должен дублироваться в результате."""
        result = _expand_query("общежитие")
        assert result.count("общежитие") == 1

    def test_disabled_expansion(self, monkeypatch):
        """При QUERY_EXPANSION_ENABLED=False — только оригинал."""
        import app.retrieval.search as search_module
        monkeypatch.setattr(search_module.settings, "QUERY_EXPANSION_ENABLED", False)
        result = _expand_query("общежитие")
        assert result == ["общежитие"]


# ─────────────────────────────────────────────────────────────────────────────
# _deduplicate_results
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplicateResults:

    def test_empty_input(self):
        assert _deduplicate_results([]) == []

    def test_no_duplicates_unchanged(self):
        results = [
            make_result("https://caiu.edu.kz/a/", chunk_index=0),
            make_result("https://caiu.edu.kz/b/", chunk_index=0),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2

    def test_exact_duplicate_removed(self):
        """Два одинаковых (url, chunk_index) → оставить только первый."""
        r1 = make_result("https://caiu.edu.kz/a/", chunk_index=0, score=0.9)
        r2 = make_result("https://caiu.edu.kz/a/", chunk_index=0, score=0.8)
        deduped = _deduplicate_results([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].score == 0.9  # сохранён первый (лучший)

    def test_different_chunk_indexes_same_url_not_duplicate(self):
        """Один URL, разные chunk_index → это разные чанки."""
        r1 = make_result("https://caiu.edu.kz/a/", chunk_index=0)
        r2 = make_result("https://caiu.edu.kz/a/", chunk_index=1)
        r3 = make_result("https://caiu.edu.kz/a/", chunk_index=2)
        # max_per_page=2 → третий должен срезаться
        deduped = _deduplicate_results([r1, r2, r3], max_per_page=2)
        assert len(deduped) == 2

    def test_max_per_page_respected(self):
        """Не более max_per_page чанков с одной страницы."""
        results = [
            make_result("https://caiu.edu.kz/a/", chunk_index=i)
            for i in range(5)
        ]
        deduped = _deduplicate_results(results, max_per_page=2)
        assert len(deduped) == 2

    def test_different_pages_not_limited_by_max_per_page(self):
        """Разные страницы не мешают друг другу."""
        results = [
            make_result(f"https://caiu.edu.kz/page{i}/", chunk_index=0)
            for i in range(10)
        ]
        deduped = _deduplicate_results(results, max_per_page=2)
        assert len(deduped) == 10  # у каждой страницы по 1 чанку


# ─────────────────────────────────────────────────────────────────────────────
# _apply_score_gap_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyScoreGapFilter:

    def test_empty_input(self):
        assert _apply_score_gap_filter([]) == []

    def test_all_close_scores_pass(self):
        """Если все результаты близки — ничего не отрезается."""
        results = [
            make_result("url1", score=0.80),
            make_result("url2", score=0.75),
            make_result("url3", score=0.70),
        ]
        # GAP = 0.25, best=0.80, min_ok = 0.55 → все проходят
        filtered = _apply_score_gap_filter(results)
        assert len(filtered) == 3

    def test_low_score_chunk_removed(self):
        """Чанк слишком далеко от лучшего результата — отрезается."""
        results = [
            make_result("url1", score=0.90),
            make_result("url2", score=0.88),
            make_result("url3", score=0.50),  # 0.90 - 0.50 = 0.40 > GAP=0.25
        ]
        filtered = _apply_score_gap_filter(results)
        assert len(filtered) == 2
        assert all(r.score >= 0.65 for r in filtered)

    def test_single_result_always_passes(self):
        results = [make_result("url1", score=0.3)]
        assert len(_apply_score_gap_filter(results)) == 1

    def test_order_preserved(self):
        """Фильтр не меняет порядок результатов."""
        results = [
            make_result("url1", score=0.90),
            make_result("url2", score=0.85),
            make_result("url3", score=0.80),
        ]
        filtered = _apply_score_gap_filter(results)
        assert [r.score for r in filtered] == [0.90, 0.85, 0.80]
