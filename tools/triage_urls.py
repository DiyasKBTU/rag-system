#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
triage_urls.py — Быстрый анализ страниц: нужны ли они абитуриентам.

Запуск из корня проекта:
    python tools/triage_urls.py

Что делает:
    1. Берёт список URL из preview (те же что войдут в индекс)
    2. Для каждой страницы скачивает заголовок + первые 120 слов
    3. Автоматически предлагает: KEEP / REMOVE / CHECK
    4. Сохраняет результат в triage_result.txt для проверки

Критерии автоматической категоризации:
    REMOVE — внутренние документы, нормативы, новости, политический контент
    KEEP   — поступление, специальности, цены, общежитие, контакты, факультеты
    CHECK  — всё остальное (требует ручной проверки)
"""

import os
import sys
import re
import time

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

import httpx
from bs4 import BeautifulSoup
from app.config import settings
from app.parser.crawler import get_urls_for_indexing

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Паттерны для автокатегоризации ───────────────────────────────────────────

# Точно НУЖНЫ абитуриентам
KEEP_PATTERNS = [
    "admission", "priem", "postupay", "postupleni",
    "bachelor", "6b0", "op/", "groups-of-educational",
    "bachelors-profile", "obrazovatelnye-programmy",
    "list-of-documents", "doc-rus", "ehreshold-scores",
    "creative-exams", "special-examination", "ent/",
    "discounts-grants", "grants-and-discounts",
    "obshhezhitie", "svobodnye-mesta", "stud-dom",
    "contacts", "kontakty", "call-center",
    "history-of-the-university", "mission-vision",
    "licenses", "naar", "symbolism",
    "rektor", "prorektor", "pervyj-prorektor",
    "faculties", "kafedra", "struktura-universiteta", "administration",
    "military-department", "question-ru", "putevoditel",
    "prava-ru", "business-and-tourism-ru", "pedagogy-ru",
    "engineering-and-information-technology",
    "management-and-finance-ru", "languages-and-literature",
    "arts-ru", "nvp-fk-ru", "sport-ru", "tvorcheskiy",
    "estestvenno-nauchnyi", "pedagogiki-i-biznesa",
    "chemistry-biology", "technologies-and-informatization",
    "ru-business-and-law", "studentam/",
    "priyomnaya-komissiya", "priemnaya-komissiya",
    "timetable", "testovyj-czentr", "online-services",
    "dualnoe-obuchenie", "otdely/", "ru-bachelor/",
    "kk/zhataqkhana", "kk/bajlanystar", "kk/priem-oaiu",
    "kk/bakalavriatqa", "kk/bilim-beru-baghdarlamalary",
    "kk/shekti-balldar", "kk/quzhattar-tizimi",
    "kk/shygharmashylyq", "kk/arnajy-emtihan",
    "kk/granttar-zhaene", "kk/aeskeri-kafedra",
    "kk/history-of-the-university", "kk/mission-vision",
    "kk/licenses", "kk/rektor", "kk/vice-rector",
    "kk/1prorector", "kk/university-structure",
    "kk/bb-kaz", "kk/kk-faculty", "kk/kafedralar",
    "kk/bejindi-paender", "kk/bakalavriat-2",
    "kk/busines-and-finance", "kk/quqyq", "kk/pedagogy-kz",
    "kk/natural-technical", "kk/creative-kk",
    "kk/nvp-sport", "kk/sport/", "kk/oner/", "kk/filologiya",
    "kk/business-and-tourism", "kk/at-zhaene-dizajn",
    "kk/mathematics-physics", "kk/basqarw-zhaene-qarzhy",
    "kk/kaz-tarikh", "kk/kk-pedagogy", "kk/bajlanystar",
    "kk/akkreditaciya", "kk/zhoghary-oqw",
    "kk/bachelor-", "kk/6b0",
]

# Точно НЕ НУЖНЫ абитуриентам
REMOVE_PATTERNS = [
    "normativnye-document", "normatinye-document",
    "standarts-smk", "basic-smk-document",
    "documented-procedures", "process-maps",
    "regulations-on-divisions", "operating-conditions",
    "rules-codes-models",
    "job-descriptions",
    "otchet-o-finansovoj", "financial-and-economic",
    "poslanie-prezidenta",
    "god-molodej",
    "2023-in-kazakhstan",
    "caiu-brandbook", "oaiu-brandbook",
    "identifikaczionnye-nomera", "zoom",
    "versiya-dlya-slabovidyashhih",
    "elfsight/",
    "account/",
    "materialy-uchebno-metodicheskogo-semi",
    "studenty-cziau-na-sorevnovaniyah",
    "v-czaiu-proshla-mezhdunarodnaya",
    "obzornaya-nedelya",
    "ruhani-jangiru",
    "top-quality-policy",
    "monitoring-and-control-of-the-quality",
    "quality-policy-ru",
    "dokumenty-soiskatelej-uchenogo",
    "ked-na-uchebnyj-god",
    "rector-report",
    "smi-ru",
    "priemnaya-komissiya-2023",
    "uchenyj-sovet", "uchebno-metodicheskij-sovet",
    "nauchno-metodicheskij-sovet",
    "support-for-childrens",
    "/author/", "/category/",
    "kk/rektor-blogi",
    "kk/qr-prezidentinin",
    "kk/ruhani-jangiru",
    "kk/zhastar-zhyly",
    "kk/smzh-standarttary", "kk/smzh-negizgi",
    "kk/quzhattalgan-proceduralar", "kk/process-kartalary",
    "kk/bolimshe-erezheleri", "kk/qyzmettik-nusqaulyq",
    "kk/bb-boiynsha-quzhattama",
    "kk/taerbie-procesi-bojynsha",
    "kk/normative-kuzhattar", "kk/normative-quzhattar",
    "kk/gylymi-process-normative",
    "kk/biliktilik-sipattamasy",
    "kk/moniroting-smk",
    "kk/baq/",
    "kk/ssylki-na-roliki",
    "kk/bb-plan-kz",
    "kk/quality-assurance-commis",
    "kk/itb-qurylymy",
    "kk/innov-usynys-qosu",
    "kk/ghylymi-ataqqa-uemitkerlerding",
    "kk/oqu-zhylyna-arnalghan",
    "kk/sybajlas-zhemqorlyqqa",
]


def auto_categorize(url: str) -> str:
    path = url.replace("https://caiu.edu.kz", "").lower()
    for p in REMOVE_PATTERNS:
        if p in path:
            return "REMOVE"
    for p in KEEP_PATTERNS:
        if p in path:
            return "KEEP"
    return "CHECK"


def fetch_title_and_snippet(url: str, timeout: float = 8.0) -> tuple:
    """Скачать заголовок и первые ~120 слов текста страницы."""
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}", ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Заголовок
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.string or ""
            title = re.sub(r"\s*[|\-–]\s*ЦАИУ.*", "", title).strip()

        # Первый абзац
        snippet = ""
        for tag in soup.find_all(["p", "div"], limit=20):
            text = tag.get_text(separator=" ", strip=True)
            words = text.split()
            if len(words) >= 10:
                snippet = " ".join(words[:120])
                break

        return title or "(нет заголовка)", snippet or "(нет текста)"

    except Exception as e:
        return f"ОШИБКА: {e}", ""


def main():
    print("Загружаю список URL для анализа...")
    urls = get_urls_for_indexing()
    print(f"Всего URL: {len(urls)}\n")

    results = {"KEEP": [], "REMOVE": [], "CHECK": []}

    for i, item in enumerate(urls, 1):
        url = item.url
        category = auto_categorize(url)
        results[category].append((url, item.lastmod))

    # Для CHECK-страниц скачиваем содержимое
    check_urls = results["CHECK"]
    print(f"Автоматически KEEP:   {len(results['KEEP'])}")
    print(f"Автоматически REMOVE: {len(results['REMOVE'])}")
    print(f"Требуют проверки:     {len(check_urls)}\n")

    check_data = []
    if check_urls:
        print(f"Скачиваю содержимое {len(check_urls)} страниц для проверки...")
        for i, (url, lastmod) in enumerate(check_urls, 1):
            print(f"  [{i}/{len(check_urls)}] {url}")
            title, snippet = fetch_title_and_snippet(url)
            check_data.append((url, lastmod, title, snippet))
            time.sleep(0.5)

    # Сохраняем результат
    out_path = os.path.join(os.path.dirname(__file__), "..", "triage_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write(f"АВТОМАТИЧЕСКИ KEEP ({len(results['KEEP'])} страниц)\n")
        f.write("Эти страницы будут проиндексированы — проверка не нужна\n")
        f.write("=" * 70 + "\n")
        for url, _ in results["KEEP"]:
            f.write(f"  {url}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"ТРЕБУЮТ ПРОВЕРКИ — CHECK ({len(check_data)} страниц)\n")
        f.write("Просмотри каждую и реши: KEEP или REMOVE\n")
        f.write("=" * 70 + "\n\n")
        for url, lastmod, title, snippet in check_data:
            f.write(f"URL:     {url}\n")
            f.write(f"Заголовок: {title}\n")
            f.write(f"Текст:   {snippet[:200]}\n")
            f.write(f"Решение: [ KEEP / REMOVE ]\n")
            f.write("-" * 60 + "\n\n")

        f.write("=" * 70 + "\n")
        f.write(f"АВТОМАТИЧЕСКИ REMOVE ({len(results['REMOVE'])} страниц)\n")
        f.write("Эти страницы НЕ будут индексироваться\n")
        f.write("=" * 70 + "\n")
        for url, _ in results["REMOVE"]:
            f.write(f"  {url}\n")

    print(f"\nРезультат сохранён: triage_result.txt")
    print(f"\nСводка:")
    print(f"  KEEP   (будут проиндексированы): {len(results['KEEP'])}")
    print(f"  CHECK  (нужна ручная проверка):  {len(check_data)}")
    print(f"  REMOVE (не нужны):               {len(results['REMOVE'])}")
    print(f"\nОткрой triage_result.txt и для каждой CHECK-страницы")
    print(f"напиши KEEP или REMOVE, потом скинь сюда.")


if __name__ == "__main__":
    main()
