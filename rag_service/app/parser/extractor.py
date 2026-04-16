# -*- coding: utf-8 -*-
"""
extractor.py - Downloads a page and extracts clean text from HTML.

Fixed and improved for WordPress-based sites like caiu.edu.kz.

Key improvements over original:
  - Lower thresholds: pages with little text are no longer silently dropped
  - Title injection: page title is prepended to content for better search recall
  - Content aggregation: for page builders (WPBakery/Elementor), collects ALL
    content blocks instead of just the single largest one
  - Smarter cleaning: shorter lines (3+ chars) are kept if they carry meaning
  - Better fallback: tries body text when all else fails
"""

import re
import hashlib
import httpx
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass
from typing import Optional, List

from app.config import settings


@dataclass
class PageContent:
    """Result of extracting one page."""
    url: str
    title: str
    text: str
    content_hash: str


def download_page(url: str) -> Optional[str]:
    """Download HTML of a page. Returns None on failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,kk;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=settings.REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        if response.status_code == 200:
            return response.text
        else:
            print(f"  [!] HTTP {response.status_code}: {url}")
            return None

    except httpx.TimeoutException:
        print(f"  [!] Timeout: {url}")
        return None
    except Exception as e:
        print(f"  [!] Download error {url}: {e}")
        return None


def extract_text(url: str, html: str) -> PageContent:
    """
    Extract clean text from HTML.

    Strategy for WordPress sites:
    1. Remove all structural garbage (nav, footer, sidebar, etc.)
    2. Try to find main content container
    3. For page builders: aggregate all content blocks
    4. Prepend the page title so topic-based searches work better
    5. Clean and normalize the text
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Get page title ─────────────────────────────────────────
    title = _get_title(soup)

    # ── Remove structural garbage (tags) ───────────────────────
    for tag in soup.find_all([
        "script", "style", "noscript", "iframe",
        "svg", "video", "audio", "img",
        "figure", "figcaption",
    ]):
        tag.decompose()

    # ── Inject heading markers BEFORE removing other tags ──────
    # h2/h3/h4 become "## Heading text" in plain text
    # This lets the chunker split by sections later
    _inject_heading_markers(soup)

    # ── Remove navigation and structural elements by class ─────
    exact_remove_classes = [
        # Navigation
        "header-top", "header-bottom", "header-top_menu",
        "header-bottom_menu", "nav-main", "nav-bottom",
        "mobile-menu", "header-dropdown", "site-header",
        "main-navigation", "primary-menu", "secondary-menu",
        # Footer
        "footer", "site-footer", "footer-widgets",
        # Sidebar / widgets
        "sidebar", "widget-area", "widget",
        # Breadcrumbs
        "breadcrumb", "breadcrumbs", "yoast-breadcrumbs",
        # Social sharing
        "social-share", "sharedaddy", "addtoany_share_save_container",
        # Post navigation (prev/next)
        "post-navigation", "nav-links", "navigation",
        # Comments
        "comments-area", "comment-respond", "comments-list",
        # Search form
        "search-form", "search-widget",
        # Popups and overlays
        "cookie-notice", "popup", "modal",
    ]

    for cls in exact_remove_classes:
        for elem in soup.find_all(class_=cls):
            elem.decompose()

    # Remove by ID
    for id_name in [
        "header", "footer", "sidebar", "nav", "navigation",
        "comments", "respond", "search", "cookie-law-info-bar",
    ]:
        elem = soup.find(id=id_name)
        if elem:
            elem.decompose()

    # ── Find and extract content ────────────────────────────────
    content_text = _extract_content_text(soup)

    if not content_text:
        # Last resort: get all body text
        body = soup.body
        if body:
            content_text = body.get_text(separator="\n", strip=True)

    clean = _clean_text(content_text or "")

    # ── Title injection ─────────────────────────────────────────
    # Prepend title to the content so that queries like "общежитие"
    # match a page titled "Студенческое общежитие" even if the body
    # text doesn't repeat the keyword prominently.
    if title and clean:
        # Check if title is already in the first 100 chars of content
        if title.lower() not in clean[:200].lower():
            clean = f"{title}\n\n{clean}"
    elif title and not clean:
        # Page with almost no body text — at least index the title
        clean = title

    content_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()

    return PageContent(url=url, title=title, text=clean, content_hash=content_hash)


def _extract_content_text(soup: BeautifulSoup) -> str:
    """
    Find and extract the main content text.

    For page builders (WPBakery, Elementor), we aggregate text from
    ALL content blocks, not just the single largest one — this is
    the key fix for pages where content is split across multiple
    small sections that individually fall below the 100-char threshold.
    """

    # ── 1. WordPress standard: .entry-content ──────────────────
    elem = soup.find(class_="entry-content")
    if elem:
        text = elem.get_text(separator="\n", strip=True)
        if len(text) >= 30:
            return text

    # ── 2. Other standard WP content classes ───────────────────
    for cls in [
        "post-content", "page-content", "the-content",
        "article-content", "content-area", "main-content",
        "td-post-content", "entry-the-content", "single-content",
    ]:
        elem = soup.find(class_=cls)
        if elem:
            text = elem.get_text(separator="\n", strip=True)
            if len(text) >= 30:
                return text

    # ── 3. <article> tag ────────────────────────────────────────
    article = soup.find("article")
    if article:
        # Try to get entry-content inside article first
        entry = article.find(class_="entry-content")
        if entry:
            text = entry.get_text(separator="\n", strip=True)
            if len(text) >= 30:
                return text
        text = article.get_text(separator="\n", strip=True)
        if len(text) >= 30:
            return text

    # ── 4. WPBakery Page Builder — AGGREGATE all columns ────────
    # WPBakery splits content into .wpb_text_column elements.
    # We collect ALL of them and join — fixes "too little text" issue.
    for cls in ["wpb_text_column", "wpb_wrapper"]:
        elems = soup.find_all(class_=cls)
        if elems:
            parts = []
            for e in elems:
                t = e.get_text(separator="\n", strip=True)
                if len(t) >= 10:
                    parts.append(t)
            combined = "\n\n".join(parts)
            if len(combined) >= 30:
                return combined

    # Try wider WPBakery containers if text columns weren't enough
    for cls in ["vc_column_inner", "wpb_column", "vc_row"]:
        elems = soup.find_all(class_=cls)
        if elems:
            # Pick the one with the most text (usually main content column)
            best = max(elems, key=lambda e: len(e.get_text(strip=True)))
            text = best.get_text(separator="\n", strip=True)
            if len(text) >= 30:
                return text

    # ── 5. Elementor Page Builder — AGGREGATE all text widgets ──
    for cls in ["elementor-text-editor", "elementor-widget-text-editor"]:
        elems = soup.find_all(class_=cls)
        if elems:
            parts = []
            for e in elems:
                t = e.get_text(separator="\n", strip=True)
                if len(t) >= 10:
                    parts.append(t)
            combined = "\n\n".join(parts)
            if len(combined) >= 30:
                return combined

    # Try wider Elementor containers
    for cls in ["elementor-widget-container", "elementor-section",
                "elementor-widget-wrap"]:
        elems = soup.find_all(class_=cls)
        if elems:
            best = max(elems, key=lambda e: len(e.get_text(strip=True)))
            text = best.get_text(separator="\n", strip=True)
            if len(text) >= 30:
                return text

    # ── 6. id="content" ─────────────────────────────────────────
    elem = soup.find(id="content")
    if elem:
        text = elem.get_text(separator="\n", strip=True)
        if len(text) >= 30:
            return text

    # ── 7. <main> tag ────────────────────────────────────────────
    elem = soup.find("main")
    if elem:
        text = elem.get_text(separator="\n", strip=True)
        if len(text) >= 30:
            return text

    # ── 8. Generic large div/section with content-like classes ──
    for tag in soup.find_all(["div", "section"]):
        classes = " ".join(tag.get("class", []))
        if any(w in classes.lower() for w in ["content", "text", "body", "post", "page"]):
            text = tag.get_text(separator="\n", strip=True)
            if len(text) >= 150:  # Higher threshold for generic fallback
                return text

    return ""


def _inject_heading_markers(soup: BeautifulSoup) -> None:
    """
    Replace h2/h3/h4 tags with ## markers in the soup tree.

    This is done BEFORE get_text() so that section headings are
    preserved in the plain text output. The chunker then uses these
    markers to split text into sections and build rich prefixes like:
      [История университета > Учебные корпуса]

    We skip very long "headings" (>150 chars) — those are usually
    decorative elements, not real section titles.
    """
    for tag in soup.find_all(["h2", "h3", "h4"]):
        heading_text = tag.get_text(strip=True)
        if heading_text and 2 < len(heading_text) < 150:
            tag.replace_with(f"\n## {heading_text}\n")


def _get_title(soup: BeautifulSoup) -> str:
    """Extract clean page title."""
    # Try h1 first (most accurate for WordPress pages)
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
        if title and len(title) > 2:
            return title

    # Try h2 if no h1 (some pages use h2 as the main heading)
    h2 = soup.find("h2")
    if h2:
        title = h2.get_text(strip=True)
        if title and len(title) > 2:
            return title

    # Fall back to <title> tag
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # Remove " - Site Name" or " | Site Name" suffix
        title = re.sub(r"\s*[|\-–—]\s*.+$", "", title).strip()
        return title

    return ""


def _remove_nav_lines(text: str) -> str:
    """
    Удаляет строки которые выглядят как навигационное меню / breadcrumbs.

    Признаки навигационной строки:
    - Короткая (< 60 символов)
    - Нет знаков препинания (.!?:;) и цифр
    - Похожа на заголовок страницы (title case или CAPS)

    Если 4+ подряд идущих строки — все навигационные → весь блок удаляется.
    Одиночные навигационные строки оставляем (могут быть реальными заголовками).
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Проверяем следующие 4 строки — все навигационные?
        nav_block = []
        j = i
        while j < len(lines) and j < i + 8:
            l = lines[j].strip()
            if l and _looks_like_nav_item(l):
                nav_block.append(j)
                j += 1
            elif not l:
                j += 1  # пустые строки пропускаем при проверке блока
            else:
                break

        if len(nav_block) >= 4:
            # Это навигационный блок — пропускаем все строки до j
            i = j
            continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


def _looks_like_nav_item(line: str) -> bool:
    """Возвращает True если строка похожа на пункт навигации."""
    if len(line) > 70:
        return False
    # Есть знаки препинания — скорее всего реальный текст
    if re.search(r"[.!?;]", line):
        return False
    # Есть цифры — может быть телефон, год, код
    if re.search(r"\d{2,}", line):
        return False
    # Очень короткая строка без букв
    if not re.search(r"[а-яёa-zәіңғүұқөһ]{3,}", line, re.IGNORECASE):
        return False
    return True


def _clean_text(text: str) -> str:
    """
    Clean extracted text.

    What we REMOVE:
    - Empty lines
    - Lines shorter than 3 chars (icons, symbols, stray punctuation)
    - Lines that are ONLY special characters (no letters or digits)
    - Repeated whitespace within lines

    What we KEEP:
    - All real text lines (Russian, Kazakh, English)
    - Short but meaningful lines: dates, codes, phone numbers, names
    - Lines with numbers combined with letters
    - Single-digit or single-word lines that aren't pure symbols
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []
    prev_empty = False

    for line in lines:
        line = line.strip()

        # Collapse repeated spaces within line
        line = re.sub(r" {2,}", " ", line)

        # Skip empty lines — but allow ONE empty line as paragraph break
        if not line:
            if not prev_empty and cleaned:
                cleaned.append("")  # preserve paragraph break
            prev_empty = True
            continue
        prev_empty = False

        # Skip very short lines (1-2 chars): symbols, icons
        if len(line) < 3:
            continue

        # Skip lines that contain ONLY special chars and digits (no letters)
        # But KEEP phone numbers, emails, dates, codes
        has_letters = bool(re.search(
            r"[a-zA-Zа-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]", line
        ))
        has_digits = bool(re.search(r"\d", line))

        if not has_letters:
            # Allow if it looks like a phone number
            if re.search(r"\+?\d[\d\s\-\(\)]{5,}", line):
                cleaned.append(line)
                continue
            # Allow if it has digits (could be year, code, etc.) and is not too short
            if has_digits and len(line) >= 4:
                cleaned.append(line)
                continue
            # Otherwise skip (pure symbols/punctuation)
            continue

        cleaned.append(line)

    # Join: single empty lines become paragraph separators
    result = "\n".join(cleaned)

    # Убираем навигационные блоки (breadcrumbs, меню)
    result = _remove_nav_lines(result)

    # Remove triple+ newlines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def get_page_content(url: str) -> Optional[PageContent]:
    """
    Main function: download URL and extract clean text.
    Returns None if page couldn't be loaded.

    Threshold lowered from 20 to 10 chars to index short pages
    (e.g. contact pages, pages that are mostly images with a few lines of text).
    """
    html = download_page(url)
    if not html:
        return None

    content = extract_text(url, html)

    # Accept pages with very little text — they still deserve to be indexed.
    # Even a page title alone is useful for search.
    if len(content.text) < 10:
        print(f"  [!] Too little text ({len(content.text)} chars), skipping: {url}")
        return None

    return content
