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
  4. If MANUAL_TEST_URLS is set - use only those
  5. Return the final list
"""

import time
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from app.config import settings


@dataclass
class SitemapURL:
    """One URL from the sitemap."""
    url: str
    lastmod: Optional[datetime] = None  # date of last change (if provided)


def get_urls_for_indexing() -> List[SitemapURL]:
    """
    Main function - returns list of URLs to index.

    If MANUAL_TEST_URLS is set in config - returns those only.
    Otherwise reads sitemap and filters.
    """

    # ── Test mode: manual URLs ────────────────────────────────────
    if settings.MANUAL_TEST_URLS:
        print(f"\n[TEST MODE] Using manually specified URLs: "
              f"{len(settings.MANUAL_TEST_URLS)} pcs.")
        return [SitemapURL(url=url) for url in settings.MANUAL_TEST_URLS]

    # ── Normal mode: read sitemap ─────────────────────────────────
    print(f"\n[INFO] Reading sitemap: {settings.SITEMAP_URL}")
    all_urls = _fetch_sitemap(settings.SITEMAP_URL)

    if not all_urls:
        print("[ERROR] Could not get URLs from sitemap!")
        return []

    print(f"[INFO] Total URLs in sitemap: {len(all_urls)}")

    # ── Filter ────────────────────────────────────────────────────
    filtered = _filter_urls(all_urls)
    print(f"[INFO] URLs after filtering: {len(filtered)}")

    return filtered


def preview_urls() -> List[SitemapURL]:
    """
    Preview mode - shows all URLs after filtering WITHOUT starting indexing.
    Used before indexing to check what will be indexed.
    """
    print("\n" + "="*60)
    print("URL PREVIEW (no indexing started)")
    print("="*60)

    urls = get_urls_for_indexing()

    if not urls:
        print("[!] No URLs found!")
        return []

    print(f"\nFound {len(urls)} URLs to index:\n")
    for i, item in enumerate(urls, 1):
        lastmod_str = ""
        if item.lastmod:
            lastmod_str = f"  [updated: {item.lastmod.strftime('%Y-%m-%d')}]"
        print(f"  {i:3}. {item.url}{lastmod_str}")

    print("\n" + "="*60)
    print("To start indexing: python -m app.parser")
    print("To test on specific pages: add to config.py MANUAL_TEST_URLS")
    print("="*60 + "\n")

    return urls


def _fetch_sitemap(sitemap_url: str) -> List[SitemapURL]:
    """
    Download and parse sitemap.xml.
    Handles both regular sitemap and sitemap_index (which links to other sitemaps).
    """
    html = _download_xml(sitemap_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")

    # Check if this is a sitemap_index (contains links to other sitemaps)
    sitemap_tags = soup.find_all("sitemap")
    if sitemap_tags:
        print(f"[INFO] Sitemap index found, contains {len(sitemap_tags)} sitemaps")
        all_urls = []
        for sitemap_tag in sitemap_tags:
            loc = sitemap_tag.find("loc")
            if loc:
                child_url = loc.text.strip()
                print(f"  -> Reading: {child_url}")
                child_urls = _fetch_sitemap(child_url)
                all_urls.extend(child_urls)
                time.sleep(0.5)  # small pause between requests
        return all_urls

    # Regular sitemap - extract all <url> tags
    url_tags = soup.find_all("url")
    result = []

    for tag in url_tags:
        loc = tag.find("loc")
        if not loc:
            continue

        url = loc.text.strip()
        lastmod = None

        # Try to get the date of last change
        lastmod_tag = tag.find("lastmod")
        if lastmod_tag:
            try:
                # Date can be in format: 2024-01-15 or 2024-01-15T10:30:00+00:00
                lastmod_str = lastmod_tag.text.strip()[:10]  # take only YYYY-MM-DD
                lastmod = datetime.strptime(lastmod_str, "%Y-%m-%d")
            except Exception:
                pass  # if date can't be parsed - skip

        result.append(SitemapURL(url=url, lastmod=lastmod))

    return result


def _filter_urls(urls: List[SitemapURL]) -> List[SitemapURL]:
    """
    Filter URLs by EXCLUDE_URL_PATTERNS from config.
    Also removes duplicates.
    """
    seen = set()
    filtered = []

    for item in urls:
        url = item.url

        # Skip duplicates
        if url in seen:
            continue
        seen.add(url)

        # Check against all exclude patterns
        should_exclude = False
        for pattern in settings.EXCLUDE_URL_PATTERNS:
            if pattern in url:
                should_exclude = True
                break

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
            print(f"[ERROR] Could not download sitemap: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to download sitemap: {e}")
        return None
