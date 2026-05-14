# -*- coding: utf-8 -*-
"""
Юнит-тесты для app.parser.extractor (чистые функции, без HTTP)

Покрывает:
  - _inject_heading_markers  — замена h2/h3/h4 на ## маркеры
  - _extract_external_links  — сбор ценных внешних ссылок
  - extract_text (content_hash) — хэш считается ДО добавления title

Запуск:
    cd rag_service
    python -m pytest ../tests/test_extractor.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_service"))

import pytest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup
import hashlib


# ── Мок settings (без .env) ──────────────────────────────────────────────────

class FakeSettings:
    REQUEST_TIMEOUT = 10
    SITE_BASE_URL = "https://caiu.edu.kz"


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    import app.config as config_module
    monkeypatch.setattr(config_module, "settings", FakeSettings())


# Импорт после патчинга
from app.parser.extractor import (
    _inject_heading_markers,
    _extract_external_links,
    extract_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# _inject_heading_markers
# ─────────────────────────────────────────────────────────────────────────────

class TestInjectHeadingMarkers:

    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_h2_replaced_with_marker(self):
        soup = self._soup("<div><h2>История</h2><p>Основан в 2021</p></div>")
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "## История" in text
        assert "<h2>" not in str(soup)

    def test_h3_and_h4_also_replaced(self):
        soup = self._soup("<h3>Факультеты</h3><h4>Кафедры</h4>")
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "## Факультеты" in text
        assert "## Кафедры" in text

    def test_h1_not_touched(self):
        """h1 — заголовок страницы, он НЕ должен превращаться в маркер."""
        soup = self._soup("<h1>Главный заголовок</h1>")
        _inject_heading_markers(soup)
        assert "<h1>" in str(soup)  # h1 остался тегом

    def test_very_long_heading_skipped(self):
        """Заголовок >150 символов — декоративный, не превращаем в маркер."""
        long_heading = "А" * 160
        soup = self._soup(f"<h2>{long_heading}</h2>")
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "##" not in text

    def test_very_short_heading_skipped(self):
        """Заголовок ≤2 символа — слишком короткий, пропускаем."""
        soup = self._soup("<h2>АБ</h2>")
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "##" not in text

    def test_multiple_headings_all_replaced(self):
        html = "<h2>Раздел 1</h2><p>Текст 1</p><h2>Раздел 2</h2><p>Текст 2</p>"
        soup = self._soup(html)
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "## Раздел 1" in text
        assert "## Раздел 2" in text

    def test_empty_heading_skipped(self):
        soup = self._soup("<h2></h2>")
        _inject_heading_markers(soup)
        text = soup.get_text()
        assert "##" not in text


# ─────────────────────────────────────────────────────────────────────────────
# _extract_external_links
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractExternalLinks:

    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_google_drive_link_extracted(self):
        soup = self._soup(
            '<a href="https://drive.google.com/file/d/abc123/view">Документ</a>'
        )
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert any("drive.google.com" in l for l in links)

    def test_google_docs_link_extracted(self):
        soup = self._soup(
            '<a href="https://docs.google.com/document/d/xyz/pub">Правила</a>'
        )
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert len(links) == 1

    def test_pdf_link_extracted(self):
        soup = self._soup(
            '<a href="https://some-server.com/files/regulations.pdf">PDF</a>'
        )
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert len(links) == 1
        assert ".pdf" in links[0]

    def test_internal_caiu_link_ignored(self):
        """Ссылки на caiu.edu.kz не должны попадать в external_links."""
        soup = self._soup(
            '<a href="https://caiu.edu.kz/about-ru/">О нас</a>'
        )
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert links == []

    def test_social_media_links_ignored(self):
        """Ссылки на соцсети игнорируются."""
        html = (
            '<a href="https://instagram.com/caiu_official">Instagram</a>'
            '<a href="https://t.me/caiu_bot">Telegram</a>'
            '<a href="https://youtube.com/watch?v=xxx">YouTube</a>'
        )
        soup = self._soup(html)
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert links == []

    def test_relative_links_ignored(self):
        """Относительные ссылки не относятся к внешним."""
        soup = self._soup('<a href="/admissions/docs/">Документы</a>')
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert links == []

    def test_no_duplicates(self):
        """Одна и та же ссылка дважды → один результат."""
        url = "https://drive.google.com/file/d/abc/view"
        soup = self._soup(f'<a href="{url}">1</a><a href="{url}">2</a>')
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert len(links) == 1

    def test_empty_page_returns_empty(self):
        soup = self._soup("<p>Нет ссылок</p>")
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert links == []

    def test_dropbox_link_extracted(self):
        soup = self._soup(
            '<a href="https://www.dropbox.com/s/abc123/file.docx">Файл</a>'
        )
        links = _extract_external_links(soup, "https://caiu.edu.kz/page/")
        assert len(links) == 1


# ─────────────────────────────────────────────────────────────────────────────
# extract_text — content_hash вычисляется ДО добавления title (Баг #15)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTextContentHash:
    """
    Баг #15: если хэш считать от текста уже с title — любое переименование
    страницы (h1) вызывает ложную переиндексацию всего контента.

    Правильное поведение: хэш от 'clean' (тело без title),
    title добавляется ПОСЛЕ вычисления хэша.
    """

    _BODY_HTML = "<p>Корпуса расположены по улице Байтурсынова.</p>"

    def _make_html(self, title: str, body: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
  <h1>{title}</h1>
  <div class="entry-content">{body}</div>
</body>
</html>"""

    def test_hash_independent_of_title(self):
        """Два вызова с разными title, одинаковым body → один и тот же хэш."""
        html_v1 = self._make_html("Старое название", self._BODY_HTML)
        html_v2 = self._make_html("Новое название страницы", self._BODY_HTML)

        pc1 = extract_text("https://caiu.edu.kz/test/", html_v1)
        pc2 = extract_text("https://caiu.edu.kz/test/", html_v2)

        assert pc1.content_hash == pc2.content_hash, (
            "content_hash изменился при смене title — Баг #15 не исправлен!"
        )

    def test_hash_changes_on_body_change(self):
        """Разный body → разные хэши (хэш не всегда одинаковый)."""
        html_v1 = self._make_html("Страница", "<p>Текст А.</p>")
        html_v2 = self._make_html("Страница", "<p>Текст Б — другое содержимое.</p>")

        pc1 = extract_text("https://caiu.edu.kz/test/", html_v1)
        pc2 = extract_text("https://caiu.edu.kz/test/", html_v2)

        assert pc1.content_hash != pc2.content_hash

    def test_title_injected_into_text(self):
        """Title присутствует в итоговом тексте чанка (для поиска)."""
        html = self._make_html("Общежитие ЦАИУ", self._BODY_HTML)
        pc = extract_text("https://caiu.edu.kz/dormitory/", html)
        assert "Общежитие ЦАИУ" in pc.text

    def test_title_not_duplicated(self):
        """Если body уже начинается с title — не дублировать его."""
        body = "<p>Общежитие ЦАИУ — это место для проживания студентов.</p>"
        html = self._make_html("Общежитие ЦАИУ", body)
        pc = extract_text("https://caiu.edu.kz/dormitory/", html)
        assert pc.text.count("Общежитие ЦАИУ") == 1
