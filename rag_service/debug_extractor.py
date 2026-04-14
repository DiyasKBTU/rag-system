# -*- coding: utf-8 -*-
"""
debug_extractor.py - Step by step trace of the extractor to find where text disappears.
Run: python debug_extractor.py
"""

import re
import httpx
from bs4 import BeautifulSoup

URL = "https://caiu.edu.kz/history-of-the-university-ru/"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

print("Downloading page...")
response = httpx.get(URL, headers=headers, timeout=15, follow_redirects=True)
html = response.text
soup = BeautifulSoup(html, "html.parser")

print(f"HTML size: {len(html)} chars")
print(f"Body text (raw): {len(soup.get_text())} chars")
print()

# ── STEP 1: Check article tag BEFORE any cleaning ────────────
article = soup.find("article")
if article:
    article_text = article.get_text(separator="\n", strip=True)
    print(f"=== ARTICLE TAG TEXT (before cleaning): {len(article_text)} chars ===")
    print(article_text[:2000])
    print("...")
else:
    print("No <article> tag found!")

print()

# ── STEP 2: Fresh parse - simulate extractor step by step ────
soup2 = BeautifulSoup(html, "html.parser")

# Show what's inside article BEFORE we remove anything
article2 = soup2.find("article")
if article2:
    print(f"=== ARTICLE TEXT BEFORE GARBAGE REMOVAL: {len(article2.get_text())} chars ===")

# Remove garbage tags
garbage_tags = ["nav", "header", "footer", "aside", "script", "style",
                "noscript", "iframe", "form", "button", "svg", "figure",
                "img", "video", "audio", "ads"]
for tag in soup2.find_all(garbage_tags):
    tag.decompose()

article3 = soup2.find("article")
if article3:
    print(f"After removing garbage tags: {len(article3.get_text())} chars")

# Remove garbage classes
garbage_classes = ["breadcrumb", "pagination", "social", "share", "cookie",
                   "banner", "widget", "sidebar", "menu", "navbar", "topbar",
                   "footer", "copyright", "search-form"]

removed_by_class = []
for css_class in garbage_classes:
    for elem in soup2.find_all(class_=re.compile(css_class, re.I)):
        removed_by_class.append(f".{css_class}: {elem.get_text()[:50]}")
        elem.decompose()

if removed_by_class:
    print(f"\nRemoved {len(removed_by_class)} elements by class:")
    for item in removed_by_class[:10]:
        print(f"  {item}")

article4 = soup2.find("article")
if article4:
    after_class_text = article4.get_text(separator="\n", strip=True)
    print(f"\nAfter removing garbage classes: {len(after_class_text)} chars")
    print("=== ARTICLE CONTENT AFTER ALL CLEANING ===")
    print(after_class_text[:3000])

# ── STEP 3: Check _clean_text filter ─────────────────────────
print()
print("=== STEP 3: Simulating _clean_text filter ===")
if article4:
    raw_text = article4.get_text(separator="\n", strip=True)
    lines = raw_text.split("\n")
    print(f"Total lines: {len(lines)}")

    kept = []
    removed_short = []
    removed_numeric = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 20:
            removed_short.append(line)
            continue
        if re.match(r"^[\d\s\W]+$", line):
            removed_numeric.append(line)
            continue
        kept.append(line)

    print(f"Lines kept: {len(kept)}")
    print(f"Lines removed (too short < 20 chars): {len(removed_short)}")
    print(f"Lines removed (numeric/symbols only): {len(removed_numeric)}")
    print()

    if kept:
        print("=== KEPT LINES (first 20) ===")
        for line in kept[:20]:
            print(f"  [{len(line)}] {line[:100]}")
    else:
        print("!!! ALL LINES WERE REMOVED - this is the bug !!!")
        print()
        print("=== SHOWING SHORT LINES THAT WERE REMOVED ===")
        for line in removed_short[:30]:
            print(f"  [{len(line)}] '{line}'")
