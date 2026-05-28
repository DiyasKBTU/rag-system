# -*- coding: utf-8 -*-
"""
crawler.py - Reads sitemap.xml and returns a filtered list of URLs to index.

Why sitemap and not crawling:
  - Crawling means: start from homepage, follow all links recursively
    (slow, may loop, loads the server)
  - Sitemap: the site itself maintains a ready list of ALL its pages
    (fast, always up to date, respectful to the server)

Flow:
  1. Download sitemap.xml (or sitemap_index.xml)
  2. Extract all URLs + their lastmod dates
  3. Filter by EXCLUDE_URL_PATTERNS
  4. If MANUAL_TEST_URLS is set — use only those
  5. Return the final list
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SitemapURL:
    """One URL from the sitemap."""
    url: str
    lastmod: Optional[datetime] = None  # date of last change (if provided)


def get_urls_for_indexing() -> List[SitemapURL]:
    """
    Main function — returns list of URLs to index.

    If MANUAL_TEST_URLS is set in config — returns those only.
    Otherwise reads sitemap and filters.
    """
    if settings.MANUAL_TEST_URLS:
        logger.info(f"[Crawler] Manual URL list: {len(settings.MANUAL_TEST_URLS)} URLs")
        return [SitemapURL(url=url) for url in settings.MANUAL_TEST_URLS]

    logger.info(f"[Crawler] Reading sitemap: {settings.SITEMAP_URL}")
    all_urls = _fetch_sitemap(settings.SITEMAP_URL)

    if not all_urls:
        logger.error("[Crawler] Could not get URLs from sitemap!")
        return []

    logger.info(f"[Crawler] Total URLs in sitemap: {len(all_urls)}")

    filtered = _filter_urls(all_urls)
    logger.info(f"[Crawler] URLs after filtering: {len(filtered)}")

    return filtered


def preview_urls() -> List[SitemapURL]:
    """
    Preview mode — shows all URLs after filtering WITHOUT starting indexing.
    Used before indexing to check what will be indexed.
    """
    urls = get_urls_for_indexing()

    if not urls:
        logger.warning("[Crawler] No URLs found!")
        return []

    logger.info(f"[Crawler] {len(urls)} URLs to index:")
    for i, item in enumerate(urls, 1):
        lastmod_str = ""
        if item.lastmod:
            lastmod_str = f"  [updated: {item.lastmod.strftime('%Y-%m-%d')}]"
        logger.info(f"  {i:3}. {item.url}{lastmod_str}")

    return urls


def _fetch_sitemap(
    sitemap_url: str,
    _seen: Optional[set] = None,
    _depth: int = 0,
) -> List[SitemapURL]:
    """
    Download and parse sitemap.xml.
    Handles both regular sitemap and sitemap_index (which links to other sitemaps).

    _seen:  set of already-visited sitemap URLs — prevents infinite loops
            when a sitemap_index accidentally references itself or a cycle.
    _depth: current recursion depth — hard limit of 5 prevents runaway recursion
            even if _seen somehow misses a cycle (e.g. redirect normalisation).
    """
    _MAX_DEPTH = 5

    if _seen is None:
        _seen = set()

    if sitemap_url in _seen:
        logger.warning(f"[Crawler] Sitemap cycle detected, skipping: {sitemap_url}")
        return []

    if _depth > _MAX_DEPTH:
        logger.warning(
            f"[Crawler] Max recursion depth ({_MAX_DEPTH}) reached, skipping: {sitemap_url}"
        )
        return []

    _seen.add(sitemap_url)

    html = _download_xml(sitemap_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")

    # Check if this is a sitemap_index (contains links to other sitemaps)
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        logger.info(f"[Crawler] Sitemap index found, contains {len(sitemap_tags)} sitemaps")
        all_urls = []
        for sitemap_tag in sitemap_tags:
            loc = sitemap_tag.find("loc")
            if loc:
                child_url = loc.text.strip()
                logger.debug(f"[Crawler] Reading child sitemap: {child_url}")
                child_urls = _fetch_sitemap(child_url, _seen=_seen, _depth=_depth + 1)
                all_urls.extend(child_urls)
                time.sleep(0.5)
        return all_urls

    # Regular sitemap — extract all <url> tags
    url_tags = soup.find_all("url")
    result = []

    for tag in url_tags:
        loc = tag.find("loc")
        if not loc:
            continue

        url     = loc.text.strip()
        lastmod = None

        lastmod_tag = tag.find("lastmod")
        if lastmod_tag:
            try:
                lastmod_str = lastmod_tag.text.strip()[:10]  # take only YYYY-MM-DD
                lastmod = datetime.strptime(lastmod_str, "%Y-%m-%d")
            except Exception:
                pass

        result.append(SitemapURL(url=url, lastmod=lastmod))

    return result


def _filter_urls(urls: List[SitemapURL]) -> List[SitemapURL]:
    """
    Filter URLs by EXCLUDE_URL_PATTERNS from config.
    Also removes duplicates.
    """
    seen: set = set()
    filtered = []

    for item in urls:
        url = item.url

        if url in seen:
            continue
        seen.add(url)

        should_exclude = any(pattern in url for pattern in settings.EXCLUDE_URL_PATTERNS)
        if not should_exclude:
            filtered.append(item)

    return filtered


def _download_xml(url: str) -> Optional[str]:
    """Download XML file. Returns content as string or None on error."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
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
            logger.error(f"[Crawler] Could not download sitemap: HTTP {response.status_code} — {url}")
            return None
    except Exception as e:
        logger.error(f"[Crawler] Failed to download sitemap: {e} — {url}")
        return None
