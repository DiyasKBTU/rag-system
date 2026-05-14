# -*- coding: utf-8 -*-
"""
Юнит-тесты для app.parser.chunker

Запуск:
    cd rag_service
    python -m pytest ../tests/test_chunker.py -v
"""
import sys
from pathlib import Path

# Добавляем rag_service в путь — там лежат app.* модули
sys.path.insert(0, str(Path(__file__).parent.parent / "rag_service"))

import pytest
from unittest.mock import patch

# Мокаем settings до импорта чанкера — иначе он требует .env
with patch("app.config.settings") as _mock_settings:
    _mock_settings.CHUNK_SIZE = 200
    _mock_settings.CHUNK_OVERLAP = 30

from app.parser.chunker import (
    _split_into_sections,
    _split_into_paragraphs,
    _add_overlap,
    _group_paragraphs,
    split_into_chunks,
)

# Патчим settings.CHUNK_SIZE / CHUNK_OVERLAP глобально через monkeypatch в фикстурах


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """Заменяем settings на простой объект с нужными полями."""
    import app.parser.chunker as chunker_module
    import app.config as config_module

    class FakeSettings:
        CHUNK_SIZE = 200
        CHUNK_OVERLAP = 30

    monkeypatch.setattr(config_module, "settings", FakeSettings())
    monkeypatch.setattr(chunker_module, "settings", FakeSettings())


# ─────────────────────────────────────────────────────────────────────────────
# _split_into_sections
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitIntoSections:

    def test_no_markers_returns_single_section(self):
        text = "Простой текст без заголовков.\nЕщё одна строка."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ""  # пустой заголовок
        assert "Простой текст" in sections[0][1]

    def test_single_heading(self):
        text = "Введение.\n\n## Корпуса\nКорпус А находится по адресу..."
        sections = _split_into_sections(text)
        assert len(sections) == 2
        assert sections[0] == ("", "Введение.")
        assert sections[1][0] == "Корпуса"
        assert "Корпус А" in sections[1][1]

    def test_multiple_headings(self):
        text = "## История\nОснован в 2021.\n## Корпуса\nЧетыре корпуса.\n## Общежитие\nАдрес: ул. Алдиярова."
        sections = _split_into_sections(text)
        assert len(sections) == 3
        assert sections[0][0] == "История"
        assert sections[1][0] == "Корпуса"
        assert sections[2][0] == "Общежитие"

    def test_heading_without_preceding_text(self):
        text = "## Заголовок\nТекст раздела."
        sections = _split_into_sections(text)
        # Нет текста перед заголовком — только один раздел
        assert len(sections) == 1
        assert sections[0][0] == "Заголовок"

    def test_empty_section_between_headings_is_skipped(self):
        text = "## Раздел 1\n\n## Раздел 2\nТекст 2"
        sections = _split_into_sections(text)
        # Раздел 1 пустой — должен быть пропущен
        assert all(title != "Раздел 1" or text.strip() for title, text in sections)
        assert any(title == "Раздел 2" for title, text in sections)

    def test_heading_strip(self):
        """Лишние пробелы после ## должны обрезаться."""
        text = "##   Заголовок с пробелами   \nТекст"
        sections = _split_into_sections(text)
        assert sections[0][0] == "Заголовок с пробелами"


# ─────────────────────────────────────────────────────────────────────────────
# _add_overlap
# ─────────────────────────────────────────────────────────────────────────────

class TestAddOverlap:

    def test_single_chunk_no_overlap(self):
        chunks = ["Один чанк без перекрытия"]
        result = _add_overlap(chunks)
        assert result == chunks

    def test_empty_list(self):
        assert _add_overlap([]) == []

    def test_two_chunks_overlap_appended(self):
        # CHUNK_OVERLAP = 30, делаем первый чанк >30 слов
        first = " ".join(f"слово{i}" for i in range(50))
        second = "Начало второго чанка"
        result = _add_overlap([first, second])
        assert len(result) == 2
        assert result[0] == first
        # Второй чанк должен начинаться с хвоста первого
        assert "слово49" in result[1]  # последнее слово первого чанка
        assert "Начало второго чанка" in result[1]

    def test_short_prev_chunk_no_overlap(self):
        """Если предыдущий чанк короче CHUNK_OVERLAP — перекрытие не добавляется."""
        first = "Короткий чанк"   # < 30 слов
        second = "Второй чанк"
        result = _add_overlap([first, second])
        assert result[1] == second  # без изменений


# ─────────────────────────────────────────────────────────────────────────────
# split_into_chunks (интеграционный уровень)
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitIntoChunks:

    def test_empty_text_returns_empty(self):
        assert split_into_chunks("") == []
        assert split_into_chunks("   ") == []
        assert split_into_chunks("ab") == []  # < 10 символов

    def test_short_text_returns_one_chunk(self):
        text = "Университет ЦАИУ основан в 2021 году в Шымкенте."
        chunks = split_into_chunks(text, page_url="https://caiu.edu.kz/about/",
                                   page_title="О университете")
        assert len(chunks) >= 1
        assert chunks[0].page_url == "https://caiu.edu.kz/about/"
        assert chunks[0].page_title == "О университете"

    def test_chunk_has_prefix(self):
        text = "Факультет права предлагает три специальности."
        chunks = split_into_chunks(text, page_title="Факультет права",
                                   page_url="https://caiu.edu.kz/law/")
        assert len(chunks) == 1
        assert "[Факультет права]" in chunks[0].text

    def test_section_prefix(self):
        text = "Введение.\n\n## Стоимость\nОбучение стоит 500 000 тенге в год."
        chunks = split_into_chunks(text, page_title="Поступление",
                                   page_url="https://caiu.edu.kz/admission/")
        # Должен быть чанк с section_title="Стоимость"
        stoimost_chunks = [c for c in chunks if c.section_title == "Стоимость"]
        assert len(stoimost_chunks) >= 1
        assert "[Поступление > Стоимость]" in stoimost_chunks[0].text

    def test_chunk_indexes_sequential(self):
        text = "\n\n".join([f"Параграф {i}. " + "слово " * 50 for i in range(10)])
        chunks = split_into_chunks(text, page_url="url", page_title="Тест")
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_external_links_passed_through(self):
        text = "Документы на Google Drive."
        links = ["https://drive.google.com/doc123"]
        chunks = split_into_chunks(text, page_url="url", page_title="Документы",
                                   external_links=links)
        assert len(chunks) >= 1
        assert chunks[0].external_links == links

    def test_no_duplicate_prefix(self):
        """Если текст уже начинается с префикса — не добавлять его снова."""
        text = "[Моя страница]\nТекст который уже имеет префикс."
        chunks = split_into_chunks(text, page_url="url", page_title="Моя страница")
        assert len(chunks) >= 1
        # Префикс не должен дублироваться
        assert chunks[0].text.count("[Моя страница]") == 1
