# -*- coding: utf-8 -*-
"""
search.py - Semantic search over indexed chunks in Qdrant.

Senior-level RAG retrieval with:
  - Query preprocessing (normalization + case handling)
  - Query Expansion with university-specific synonym dictionaries
  - Multi-query search (run expanded queries, merge & deduplicate)
  - Relative score gap filtering (drop irrelevant tail chunks)
  - Max context size cap (prevent context bloat to ChatGPT)
  - Source deduplication (don't flood context with same-page chunks)
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz import process, fuzz

from app.indexer.storage import get_client
from app.indexer.embeddings import get_embedding, get_embeddings_batch
from app.config import settings

logger = logging.getLogger(__name__)


# ── Кэш эмбеддингов ───────────────────────────────────────────────────────────
# Хранит уже посчитанные векторы для одинаковых запросов.
# Экономит ~300-500ms на повторных вопросах (частые: "как поступить", "стоимость").
# maxsize=256 — хватит на все типичные запросы, не съест память.
_EMBEDDING_CACHE: Dict[str, List[float]] = {}
_CACHE_MAX_SIZE = 256


def _get_embedding_cached(text: str) -> List[float]:
    """Возвращает вектор из кэша, если он там есть, иначе запрашивает у OpenAI."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _EMBEDDING_CACHE:
        if len(_EMBEDDING_CACHE) >= _CACHE_MAX_SIZE:
            # Удаляем первый (самый старый) ключ
            _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
        _EMBEDDING_CACHE[key] = get_embedding(text)
    return _EMBEDDING_CACHE[key]


def _get_embeddings_batch_cached(texts: List[str]) -> List[List[float]]:
    """
    Пакетно получает эмбеддинги, используя кэш.
    Отправляет к OpenAI ТОЛЬКО те тексты, которых нет в кэше.
    """
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    missing_indices = [i for i, k in enumerate(keys) if k not in _EMBEDDING_CACHE]

    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        # Один батч-запрос вместо N отдельных — главная оптимизация скорости
        new_vectors = get_embeddings_batch(missing_texts)
        for i, vector in zip(missing_indices, new_vectors):
            key = keys[i]
            if len(_EMBEDDING_CACHE) >= _CACHE_MAX_SIZE:
                _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
            _EMBEDDING_CACHE[key] = vector

    return [_EMBEDDING_CACHE[k] for k in keys]


@dataclass
class SearchResult:
    """One search result — a relevant chunk found for the question."""
    text: str           # chunk text
    page_url: str       # source page URL
    page_title: str     # source page title
    chunk_index: int    # position of chunk on page
    score: float        # relevance score (0.0 to 1.0, higher = more relevant)


# ── University-specific synonym dictionary ─────────────────────────────────────
# Key = canonical form, Value = list of synonyms to search additionally.
# Catches case/form variations and domain-specific Russian university vocabulary.
#
# Why needed: OpenAI embeddings handle semantics well, but for very specific
# short queries like "специальности" the embedding might miss related pages
# titled "Образовательные программы" because the cosine distance can be 0.42
# (just below MIN_CONFIDENT_SCORE). Multi-query fixes this.
#
SYNONYM_MAP: Dict[str, List[str]] = {
    # ─── Специальности / Образовательные программы ─────────────
    "специальности": [
        "образовательные программы",
        "направления подготовки",
        "специальность бакалавриат",
        "перечень специальностей",
    ],
    "специальность": [
        "образовательная программа",
        "направление подготовки",
        "бакалавриат специальность",
    ],
    "образовательные программы": [
        "специальности",
        "направления подготовки",
        "перечень специальностей",
    ],

    # ─── Общежитие ──────────────────────────────────────────────
    "общежитие": [
        "студенческое общежитие",
        "проживание студентов",
        "место в общежитии",
        "общежития",
        "заселение в общежитие",
    ],
    "общежития": [
        "общежитие",
        "студенческое общежитие",
        "проживание студентов",
        "заселение в общежитие",
    ],
    "проживание": [
        "общежитие",
        "студенческое общежитие",
        "место в общежитии",
    ],
    # Составной: документы + общежитие
    "документы для общежития": [
        "заявление на место в общежитии",
        "как получить место в общежитии",
        "заселение в общежитие документы",
        "что нужно для общежития",
    ],
    "документы нужны для общежития": [
        "заявление на место в общежитии",
        "список документов для заселения",
        "как заселиться в общежитие",
        "получить место в общежитии",
    ],
    "общежитие документы": [
        "заявление на место в общежитии",
        "документы для заселения",
        "список документов общежитие",
    ],

    # ─── Поступление / Приём ────────────────────────────────────
    "поступление": [
        "приём документов",
        "как поступить",
        "вступительные экзамены",
        "процесс поступления",
        "зачисление",
    ],
    "поступить": [
        "поступление",
        "приём документов",
        "процесс зачисления",
        "требования для поступления",
    ],
    "приёмная комиссия": [
        "поступление",
        "приём документов",
        "контакты приёмной комиссии",
    ],
    "документы": [
        "список документов",
        "перечень документов",
        "необходимые документы для поступления",
    ],

    # ─── Гранты / Скидки ────────────────────────────────────────
    "грант": [
        "гранты на обучение",
        "государственный грант",
        "грантовое финансирование",
        "образовательный грант",
    ],
    "гранты": [
        "грант на обучение",
        "государственные гранты",
        "скидки на обучение",
    ],
    "скидки": [
        "скидка на обучение",
        "льготы",
        "гранты и скидки",
    ],

    # ─── Факультеты / Кафедры ───────────────────────────────────
    "факультет": [
        "факультеты университета",
        "институт",
        "учебные подразделения",
    ],
    "факультеты": [
        "факультет",
        "кафедра",
        "институты",
        "подразделения вуза",
    ],
    "кафедра": [
        "кафедры",
        "кафедра университета",
        "профессора кафедры",
    ],

    # ─── Экзамены / Вступительные ───────────────────────────────
    "экзамены": [
        "вступительные испытания",
        "творческий экзамен",
        "специальный экзамен",
        "ент",
    ],
    "ент": [
        "единое национальное тестирование",
        "вступительные экзамены",
        "результаты ент",
    ],
    "пороговый балл": [
        "проходной балл",
        "минимальный балл",
        "пороговые баллы",
        "баллы для поступления",
    ],

    # ─── Стоимость / Оплата ─────────────────────────────────────
    "стоимость обучения": [
        "цена обучения",
        "оплата за обучение",
        "платное обучение",
        "контрактное обучение",
    ],
    "контракт": [
        "контрактное обучение",
        "стоимость контракта",
        "платное обучение",
    ],

    # ─── Контакты ───────────────────────────────────────────────
    "контакты": [
        "контактная информация",
        "телефон университета",
        "адрес университета",
        "как связаться",
    ],
    "адрес": [
        "местонахождение",
        "как добраться",
        "контакты",
        "где находится",
    ],

    # ─── Лицензии / Аккредитация ────────────────────────────────
    "лицензия": [
        "лицензии",
        "аккредитация",
        "государственная лицензия",
    ],
    "аккредитация": [
        "лицензия",
        "аккредитация университета",
        "институциональная аккредитация",
    ],

    # ─── История / О вузе ───────────────────────────────────────
    "история": [
        "история университета",
        "о вузе",
        "об университете",
        "основание университета",
        "когда основан",
        "год основания",
    ],
    "история университета": [
        "основан в",
        "год основания университета",
        "история цаиу",
        "о вузе",
        "об университете",
    ],
    "когда основан": [
        "история университета",
        "год основания",
        "история цаиу",
    ],
    "сколько лет": [
        "история университета",
        "год основания",
        "основан в",
    ],
    "сколько студентов": [
        "количество студентов",
        "численность студентов",
        "число обучающихся",
        "о вузе",
        "об университете",
    ],
    "количество студентов": [
        "численность студентов",
        "число обучающихся",
        "студентов обучается",
        "о вузе",
    ],
    "миссия": [
        "миссия и видение",
        "цели университета",
        "стратегия университета",
    ],

    # ─── Стоимость / Оплата (расширенный блок) ──────────────────
    "стоимость": [
        "стоимость обучения",
        "оплата за обучение",
        "контрактное обучение",
        "цена обучения",
    ],
    "цена": [
        "стоимость обучения",
        "оплата за обучение",
        "контракт стоимость",
    ],
    "платное обучение": [
        "стоимость обучения",
        "контрактное обучение",
        "оплата за обучение",
    ],
    "сколько стоит": [
        "стоимость обучения",
        "цена обучения",
        "оплата за обучение",
        "контракт стоимость",
    ],
    "оплата": [
        "оплата за обучение",
        "стоимость обучения",
        "контрактное обучение",
    ],

    # ─── Военная кафедра ────────────────────────────────────────
    "военная кафедра": [
        "военный факультет",
        "военная подготовка",
        "военная специальность",
    ],

    # ─── Ректор / Руководство ────────────────────────────────────
    "ректор": [
        "руководитель университета",
        "ректор цаиу",
        "глава университета",
        "руководство университета",
    ],
    "кто ректор": [
        "ректор университета",
        "руководитель вуза",
        "ректор цаиу",
        "глава университета",
    ],
    "руководство": [
        "ректор",
        "проректор",
        "руководство университета",
        "администрация университета",
    ],
    "администрация": [
        "руководство университета",
        "ректор",
        "проректор",
        "администрация цаиу",
    ],
    "проректор": [
        "проректоры университета",
        "руководство цаиу",
        "заместитель ректора",
    ],

    # ════════════════════════════════════════════════════════════
    # КАЗАХСКИЙ ЯЗЫК — синонимы для типичных запросов абитуриентов
    # ════════════════════════════════════════════════════════════

    # ─── Мамандықтар / Специальности ────────────────────────────
    "мамандықтар": [
        "білім беру бағдарламалары",
        "мамандық тізімі",
        "бакалавриат мамандықтары",
        "специальности",
        "образовательные программы",
    ],
    "мамандық": [
        "білім беру бағдарламасы",
        "бакалавриат",
        "мамандықтар",
        "специальность",
    ],
    "білім беру бағдарламалары": [
        "мамандықтар",
        "мамандық тізімі",
        "бакалавриат мамандықтары",
    ],

    # ─── Жатақхана / Общежитие ───────────────────────────────────
    "жатақхана": [
        "студент жатақханасы",
        "жатақханаға орналасу",
        "тұрғын үй",
        "общежитие",
        "студенческое общежитие",
    ],
    "жатақханаға": [
        "жатақхана",
        "студент жатақханасы",
        "жатақханаға орналасу",
    ],
    "жатақхана құжаттары": [
        "жатақханаға орналасу үшін құжаттар",
        "жатақхана тізімі",
        "документы для общежития",
    ],

    # ─── Түсу / Поступление ──────────────────────────────────────
    "түсу": [
        "қабылдау",
        "оқуға түсу",
        "қабылдау комиссиясы",
        "поступление",
        "приём документов",
    ],
    "қабылдау": [
        "түсу",
        "оқуға қабылдау",
        "қабылдау комиссиясы",
        "приём",
    ],
    "қабылдау комиссиясы": [
        "приёмная комиссия",
        "оқуға түсу",
        "қабылдау",
    ],
    "оқуға түсу": [
        "қабылдау",
        "түсу тәртібі",
        "поступление",
        "процесс поступления",
    ],

    # ─── Құжаттар / Документы ───────────────────────────────────
    "құжаттар": [
        "қажетті құжаттар",
        "құжаттар тізімі",
        "документы",
        "список документов",
    ],
    "қажетті құжаттар": [
        "құжаттар тізімі",
        "түсуге қажетті құжаттар",
        "необходимые документы",
    ],

    # ─── Грант / Стипендия ──────────────────────────────────────
    "грант": [
        "мемлекеттік грант",
        "оқу гранты",
        "гранттар",
        "государственный грант",
        "гранты на обучение",
    ],
    "гранттар": [
        "грант",
        "оқу гранты",
        "жеңілдіктер",
        "гранты",
    ],
    "жеңілдіктер": [
        "гранттар",
        "оқу жеңілдіктері",
        "скидки на обучение",
        "льготы",
    ],

    # ─── Факультет / Кафедра ────────────────────────────────────
    "факультет": [
        "факультеттер",
        "кафедра",
        "институт",
        "факультеты",
    ],
    "факультеттер": [
        "факультет",
        "кафедра",
        "факультеты университета",
    ],

    # ─── Емтихандар / Экзамены ──────────────────────────────────
    "емтихандар": [
        "кіру емтихандары",
        "шығармашылық емтихан",
        "арнайы емтихан",
        "вступительные экзамены",
        "экзамены",
    ],
    "ұбт": [
        "ұлттық бірыңғай тестілеу",
        "кіру емтихандары",
        "ент",
        "единое национальное тестирование",
    ],
    "өтпелі балл": [
        "минималды балл",
        "пороговый балл",
        "проходной балл",
        "баллы для поступления",
    ],

    # ─── Оқу құны / Стоимость ───────────────────────────────────
    "оқу құны": [
        "оқу ақысы",
        "контракт бағасы",
        "стоимость обучения",
        "оплата за обучение",
    ],
    "оқу ақысы": [
        "оқу құны",
        "контракт",
        "стоимость обучения",
    ],

    # ─── Байланыс / Контакты ────────────────────────────────────
    "байланыс": [
        "байланыс ақпараты",
        "телефон",
        "мекенжай",
        "контакты",
        "контактная информация",
    ],
    "мекенжай": [
        "байланыс",
        "қалай жетуге болады",
        "адрес",
        "где находится",
    ],

    # ─── Ректор / Басшылық ──────────────────────────────────────
    "ректор": [  # noqa: F811  (перекрывает русский ключ — казахский вариант)
        "университет басшысы",
        "басшылық",
        "ректор цаиу",
        "руководство университета",
    ],
    "басшылық": [
        "ректор",
        "проректор",
        "руководство университета",
        "администрация",
    ],

    # ─── Тарих / История ────────────────────────────────────────
    "тарих": [
        "университет тарихы",
        "университет туралы",
        "история университета",
        "об университете",
    ],
    "миссия": [  # noqa: F811
        "миссия және көзқарас",
        "университет мақсаттары",
        "миссия и видение",
    ],

    # ─── Лицензия / Аккредитация ────────────────────────────────
    "лицензия": [  # noqa: F811
        "аккредитация",
        "мемлекеттік лицензия",
        "государственная лицензия",
    ],
    "аккредитация": [  # noqa: F811
        "лицензия",
        "университет аккредитациясы",
        "аккредитация университета",
    ],

    # ─── Әскери кафедра ─────────────────────────────────────────
    "әскери кафедра": [
        "әскери факультет",
        "әскери дайындық",
        "военная кафедра",
        "военная подготовка",
    ],
}


def _normalize_query(query: str) -> str:
    """
    Normalize query text:
    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    - Convert to lowercase (for synonym lookup only — we pass original to embeddings)
    """
    query = query.strip()
    query = re.sub(r"\s{2,}", " ", query)
    return query


def _fuzzy_match_keywords(query_lower: str) -> Optional[str]:
    """
    Ищет ключи SYNONYM_MAP, схожие со словами в запросе, с помощью нечёткого сравнения.

    Зачем: если пользователь написал "абщежитие" или "общяжитие",
    точная проверка `keyword in query_lower` не сработает.
    Этот метод находит ближайший ключ и возвращает его — как будто
    пользователь написал правильно.

    Порог FUZZY_THRESHOLD=82 подобран эмпирически:
    - 80+ даёт хорошее покрытие опечаток (1-2 буквы)
    - Выше 85 — пропускает некоторые реальные опечатки

    Возвращает первый найденный подходящий ключ или None.
    """
    FUZZY_THRESHOLD = 82
    known_keywords = list(SYNONYM_MAP.keys())

    # Сначала проверяем всю фразу целиком (для многословных ключей типа "военная кафедра")
    match = process.extractOne(
        query_lower,
        known_keywords,
        scorer=fuzz.partial_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if match:
        return match[0]

    # Потом проверяем каждое слово запроса отдельно (для однословных ключей)
    for word in query_lower.split():
        if len(word) < 4:  # Короткие слова пропускаем — слишком много ложных совпадений
            continue
        match = process.extractOne(
            word,
            known_keywords,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if match:
            return match[0]

    return None


def _expand_query(query: str) -> List[str]:
    """
    Return a list of query variants for multi-query search.

    Strategy:
    1. Always include the original query (already preprocessed)
    2. Check if any word or phrase in the query matches a key in SYNONYM_MAP
       — сначала точное совпадение, затем нечёткое (для опечаток)
    3. For each match: add up to 2 synonym queries

    Returns at most 3 queries total to keep API costs low.
    """
    if not settings.QUERY_EXPANSION_ENABLED:
        return [query]

    query_lower = query.lower()
    extra_queries = []
    matched_keyword = None

    # ── Шаг 1: точное совпадение (как было раньше) ────────────────
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in query_lower:
            matched_keyword = keyword
            for syn in synonyms[:2]:
                variant = re.sub(
                    re.escape(keyword),
                    syn,
                    query_lower,
                    flags=re.IGNORECASE,
                )
                if variant not in extra_queries and variant != query_lower:
                    extra_queries.append(variant)
                if len(extra_queries) >= 2:
                    break
            if len(extra_queries) >= 2:
                break

    # ── Шаг 2: нечёткое совпадение (новое — для опечаток) ─────────
    # Запускаем только если точное совпадение ничего не нашло
    if not extra_queries:
        fuzzy_key = _fuzzy_match_keywords(query_lower)
        if fuzzy_key and fuzzy_key != query_lower:
            logger.debug(f"Fuzzy match: '{query_lower}' → '{fuzzy_key}'")
            synonyms = SYNONYM_MAP[fuzzy_key]
            # Добавляем сам исправленный ключ как вариант запроса
            extra_queries.append(fuzzy_key)
            # И первый синоним
            if synonyms:
                extra_queries.append(synonyms[0])

    # Return original + up to 2 expansions
    return [query] + extra_queries[:2]


def _deduplicate_results(
    results: List[SearchResult],
    max_per_page: int = 2,
) -> List[SearchResult]:
    """
    Deduplicate search results:
    1. Remove exact text duplicates (from multi-query overlap)
    2. Limit to max_per_page chunks per source page (prevent one page flooding context)
    """
    seen_texts = set()
    page_count: Dict[str, int] = {}
    deduped = []

    for r in results:
        # Skip exact text duplicate
        text_key = r.text[:200]  # compare first 200 chars to catch near-duplicates
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        # Limit per page
        page_count[r.page_url] = page_count.get(r.page_url, 0) + 1
        if page_count[r.page_url] > max_per_page:
            continue

        deduped.append(r)

    return deduped


def _apply_score_gap_filter(results: List[SearchResult]) -> List[SearchResult]:
    """
    Drop chunks that fall too far below the best result's score.

    Logic: if the best result has score=0.75 and SCORE_GAP_THRESHOLD=0.18,
    we only keep results with score >= 0.75 - 0.18 = 0.57.

    This prevents including marginally-relevant "tail" chunks that confuse ChatGPT.
    """
    if not results:
        return results

    best_score = results[0].score
    min_acceptable = best_score - settings.SCORE_GAP_THRESHOLD

    return [r for r in results if r.score >= min_acceptable]


def search(
    question: str,
    top_k: int = None,
    min_score: float = None,
) -> List[SearchResult]:
    """
    Find most relevant chunks for a user question.

    Improvements over naive similarity search:
    - Query normalization (strips, collapses spaces)
    - Query expansion via synonym dictionary (catches specialist/domain vocabulary)
    - Multi-query: runs expanded queries in parallel, merges results
    - Score gap filter: drops irrelevant tail
    - Deduplication: removes same-text duplicates and limits per-page density
    - Confidence check: returns [] if nothing is truly relevant

    Args:
        question:  User's question in any language
        top_k:     How many chunks to fetch from Qdrant (default from config)
        min_score: Minimum cosine similarity (default from config)

    Returns:
        List of SearchResult ordered by relevance (best first),
        already filtered, deduplicated, capped at MAX_CONTEXT_CHUNKS.
    """
    if top_k is None:
        top_k = settings.TOP_K_RESULTS
    if min_score is None:
        min_score = settings.SIMILARITY_THRESHOLD

    # ── Step 1: Normalize query ───────────────────────────────
    question = _normalize_query(question)
    if not question:
        return []

    # ── Step 2: Expand query into variants ────────────────────
    query_variants = _expand_query(question)
    logger.debug(f"Query variants: {query_variants}")

    # ── Step 3: Multi-query search ────────────────────────────
    # Получаем все эмбеддинги за ОДИН батч-запрос к OpenAI
    # (было: N отдельных вызовов × ~2500ms = медленно)
    # (стало: 1 батч-запрос × ~500ms = быстро)
    t0 = time.perf_counter()
    try:
        vectors = _get_embeddings_batch_cached(query_variants)
    except Exception as e:
        logger.warning(f"Batch embedding failed: {e}, falling back to single")
        vectors = []
        for v in query_variants:
            try:
                vectors.append(_get_embedding_cached(v))
            except Exception as e2:
                logger.warning(f"Single embedding failed for '{v[:40]}': {e2}")
                vectors.append(None)

    logger.debug(f"Embeddings computed in {(time.perf_counter()-t0)*1000:.0f}ms "
                 f"for {len(query_variants)} variants")

    qdrant = get_client()
    all_results: List[SearchResult] = []
    seen_ids: set = set()

    for variant, vector in zip(query_variants, vectors):
        if vector is None:
            continue
        try:
            hits = qdrant.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )

            for hit in hits:
                payload = hit.payload or {}
                # Use text as dedup key (avoid same chunk from multiple queries)
                text = payload.get("text", "")
                text_key = text[:200]
                if text_key in seen_ids:
                    continue
                seen_ids.add(text_key)

                result = SearchResult(
                    text=text,
                    page_url=payload.get("page_url", ""),
                    page_title=payload.get("page_title", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    score=round(hit.score, 4),
                )
                all_results.append(result)

        except Exception as e:
            logger.warning(f"Search failed for variant '{variant[:60]}': {e}")
            continue

    if not all_results:
        return []

    # ── Step 4: Sort merged pool by score (best first) ────────
    all_results.sort(key=lambda r: r.score, reverse=True)

    # ── Step 5: Confidence check ──────────────────────────────
    # If even the best result isn't confident enough, return empty.
    # Better no context than wrong context for ChatGPT.
    if all_results[0].score < settings.MIN_CONFIDENT_SCORE:
        logger.debug(
            f"Best score {all_results[0].score} < MIN_CONFIDENT_SCORE "
            f"{settings.MIN_CONFIDENT_SCORE}, returning empty"
        )
        return []

    # ── Step 6: Score gap filter ──────────────────────────────
    filtered = _apply_score_gap_filter(all_results)

    # ── Step 7: Deduplicate and limit per-page density ────────
    deduped = _deduplicate_results(filtered, max_per_page=2)

    # ── Step 8: Cap at MAX_CONTEXT_CHUNKS ─────────────────────
    final = deduped[:settings.MAX_CONTEXT_CHUNKS]

    logger.info(
        f"Search '{question[:60]}': "
        f"raw={len(all_results)}, after_gap={len(filtered)}, "
        f"after_dedup={len(deduped)}, final={len(final)}"
    )

    return final


def search_and_format_context(question: str) -> str:
    """
    Search for relevant chunks and format them as context for ChatGPT.

    Returns a single string with all relevant chunks combined,
    trimmed to MAX_CONTEXT_CHARS to prevent context bloat.

    Format:
        [Источник: Общежитие]
        Студенческое общежитие КАИУ находится по адресу...

        [Источник: Специальности]
        Университет предлагает следующие специальности...
    """
    results = search(question)

    if not results:
        return ""

    context_parts = []
    total_chars = 0

    for result in results:
        source = result.page_title or result.page_url
        part = f"[Источник: {source}]\n{result.text}"

        # Don't exceed max context size
        if total_chars + len(part) > settings.MAX_CONTEXT_CHARS:
            # If this is the first chunk — include it truncated
            if not context_parts:
                truncated = part[:settings.MAX_CONTEXT_CHARS]
                context_parts.append(truncated)
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)
