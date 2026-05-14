# -*- coding: utf-8 -*-
"""
Юнит-тесты для get_last_indexed_at / set_last_indexed_at (storage.py)

Тесты полностью изолированы: вместо реального hot_swap_state.json
используется временный файл (tmp_path), Qdrant не нужен.

Запуск:
    cd rag_service
    python -m pytest ../tests/test_storage.py -v
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_service"))

import pytest
from unittest.mock import MagicMock, patch


# ── Минимальный мок settings ──────────────────────────────────────────────────

class FakeSettings:
    QDRANT_COLLECTION_NAME = "test_collection"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    OPENAI_API_KEY = "test-key"
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 30


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    """Патчим всё что требует реального окружения до импорта storage."""
    import app.config as config_module
    monkeypatch.setattr(config_module, "settings", FakeSettings())

    # Мокаем QdrantClient чтобы не коннектился к Docker
    mock_qdrant_cls = MagicMock()
    monkeypatch.setattr("app.indexer.storage.QdrantClient", mock_qdrant_cls)


# ── Фикстура: перенаправляем _STATE_FILE во временную папку ──────────────────

@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """
    Подменяем _STATE_FILE в модуле storage на временный файл.
    Возвращает Path к нему (файл изначально не существует).
    """
    import app.indexer.storage as storage_module
    fake_path = tmp_path / "hot_swap_state.json"
    monkeypatch.setattr(storage_module, "_STATE_FILE", fake_path)
    return fake_path


# ─────────────────────────────────────────────────────────────────────────────
# get_last_indexed_at
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLastIndexedAt:

    def test_returns_none_when_file_missing(self, state_file):
        from app.indexer.storage import get_last_indexed_at
        assert state_file.exists() is False
        assert get_last_indexed_at() is None

    def test_returns_none_when_key_absent(self, state_file):
        """Файл есть, но last_indexed_at не записан."""
        state_file.write_text(
            json.dumps({"active_collection": "caiu_knowledge_base_blue"}),
            encoding="utf-8",
        )
        from app.indexer.storage import get_last_indexed_at
        assert get_last_indexed_at() is None

    def test_returns_none_when_key_empty_string(self, state_file):
        state_file.write_text(
            json.dumps({"last_indexed_at": ""}),
            encoding="utf-8",
        )
        from app.indexer.storage import get_last_indexed_at
        assert get_last_indexed_at() is None

    def test_returns_datetime_for_valid_iso_timestamp(self, state_file):
        ts = "2026-05-10T03:00:00+00:00"
        state_file.write_text(
            json.dumps({"last_indexed_at": ts}),
            encoding="utf-8",
        )
        from app.indexer.storage import get_last_indexed_at
        result = get_last_indexed_at()
        assert result is not None
        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 10

    def test_result_is_naive_datetime(self, state_file):
        """Возвращаемый datetime не должен иметь tzinfo (naive UTC)."""
        ts = "2026-05-10T03:00:00+00:00"
        state_file.write_text(json.dumps({"last_indexed_at": ts}), encoding="utf-8")
        from app.indexer.storage import get_last_indexed_at
        result = get_last_indexed_at()
        assert result.tzinfo is None

    def test_returns_none_on_corrupt_json(self, state_file):
        """Битый JSON — функция должна вернуть None, не упасть."""
        state_file.write_text("NOT JSON {{{ !!!", encoding="utf-8")
        from app.indexer.storage import get_last_indexed_at
        assert get_last_indexed_at() is None


# ─────────────────────────────────────────────────────────────────────────────
# set_last_indexed_at
# ─────────────────────────────────────────────────────────────────────────────

class TestSetLastIndexedAt:

    def test_creates_file_if_not_exists(self, state_file):
        from app.indexer.storage import set_last_indexed_at
        assert state_file.exists() is False
        set_last_indexed_at(datetime(2026, 5, 12, 3, 0, 0))
        assert state_file.exists()

    def test_saved_value_is_readable_back(self, state_file):
        """set → get должны быть согласованы."""
        from app.indexer.storage import set_last_indexed_at, get_last_indexed_at
        dt = datetime(2026, 5, 12, 3, 0, 0)
        set_last_indexed_at(dt)
        result = get_last_indexed_at()
        assert result is not None
        # Допуск 1 секунда для форматирования ISO
        assert abs((result - dt).total_seconds()) < 1

    def test_preserves_existing_active_collection(self, state_file):
        """Запись last_indexed_at не должна стирать active_collection."""
        state_file.write_text(
            json.dumps({"active_collection": "caiu_knowledge_base_blue"}),
            encoding="utf-8",
        )
        from app.indexer.storage import set_last_indexed_at
        set_last_indexed_at(datetime(2026, 5, 12, 3, 0, 0))
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state.get("active_collection") == "caiu_knowledge_base_blue"
        assert "last_indexed_at" in state

    def test_default_arg_uses_current_time(self, state_file):
        """Без аргумента — сохраняется текущее время (±5 секунд)."""
        from app.indexer.storage import set_last_indexed_at, get_last_indexed_at
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        set_last_indexed_at()
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        result = get_last_indexed_at()
        assert result is not None
        assert before - timedelta(seconds=1) <= result <= after + timedelta(seconds=1)

    def test_overwrites_previous_value(self, state_file):
        """Повторный вызов обновляет время."""
        from app.indexer.storage import set_last_indexed_at, get_last_indexed_at
        dt1 = datetime(2025, 1, 1, 0, 0, 0)
        dt2 = datetime(2026, 5, 12, 3, 0, 0)
        set_last_indexed_at(dt1)
        set_last_indexed_at(dt2)
        result = get_last_indexed_at()
        assert result.year == 2026
