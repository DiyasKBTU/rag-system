# -*- coding: utf-8 -*-
"""
search.py - Semantic search over indexed chunks in Qdrant.

RAG retrieval:
  - Query preprocessing (normalization + case handling)
  - Query Expansion with university-specific synonym dictionaries
  - Multi-query semantic search (run expanded queries, merge & deduplicate)
  - Score gap filtering (drop irrelevant tail chunks)
  - Max context size cap (prevent context bloat to ChatGPT)
  - Source deduplication (don't flood context with same-page chunks)

Note: BM25 keyword search was removed — question-chunk indexing at ingestion
time provides better recall than runtime keyword matching.
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

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
    tags: list = None   # пользовательские теги (из ручного редактирования)


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
        "мемлекеттік грант",
        "оқу гранты",
        "гранттар",
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
        "факультеттер",
        "кафедра",
        "факультеты",
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
        "где находится университет",
        "адрес университета",
    ],

    # ─── Лицензии / Аккредитация ────────────────────────────────
    "лицензия": [
        "лицензии",
        "аккредитация",
        "государственная лицензия",
        "мемлекеттік лицензия",
    ],
    "аккредитация": [
        "лицензия",
        "аккредитация университета",
        "институциональная аккредитация",
        "университет аккредитациясы",
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
        "миссия және көзқарас",
        "университет мақсаттары",
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
        "ректор цаиу фамилия имя",
        "руководитель университета биография",
        "глава университета фио",
        "руководство университета",
        "университет басшысы",
        "басшылық",
    ],
    "кто ректор": [
        "ректор университета биография",
        "ректор цаиу фио",
        "руководитель вуза фамилия имя",
        "глава университета",
        "ректор цаиу",
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

    # ─── Дистанционное обучение ─────────────────────────────────
    "дистанционно": [
        "очно-дистанционная форма",
        "онлайн обучение",
        "дистанционное обучение",
    ],
    "дистанционное обучение": [
        "очно-дистанционная",
        "онлайн обучение",
    ],

    # ─── Юридический факультет ──────────────────────────────────
    "юридический факультет": [
        "факультет бизнеса и права",
        "кафедра права",
        "юриспруденция",
    ],
    "юриспруденция": [
        "кафедра права",
        "факультет бизнеса и права",
    ],

    # ─── Пороговые баллы ────────────────────────────────────────
    "проходной балл": [
        "пороговый балл",
        "пороговые баллы",
        "минимальный балл ент",
    ],
    "балл ент": [
        "пороговые баллы",
        "проходной балл",
    ],

    # ─── Отличия / Преимущества ─────────────────────────────────
    "чем отличается": [
        "преимущества университета",
        "миссия университета",
        "история университета",
    ],
    "преимущества": [
        "миссия и видение",
        "преимущества цаиу",
        "о вузе",
    ],

    # ─── Корпуса / Здания ───────────────────────────────────────
    # "корпус" — часто задаваемый вопрос: "сколько корпусов", "где находится"
    "корпус": [
        "учебный корпус",
        "здание университета",
        "учебные корпуса",
        "campus",
    ],
    "корпуса": [
        "учебные корпуса",
        "здания университета",
        "учебный корпус",
    ],
    "здание": [
        "учебный корпус",
        "здание университета",
        "где находится",
    ],

    # ─── Лицензия / Аккредитация ────────────────────────────────

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
    1. Always include the original query
    2. Check SYNONYM_MAP for exact substring matches
    3. If no match — try fuzzy matching (catches typos like "абщежитие")
    4. Return at most 3 queries total to keep API costs low.
    """
    if not settings.QUERY_EXPANSION_ENABLED:
        return [query]

    query_lower = query.lower()
    extra_queries = []

    # ── Шаг 1: точное подстроковое совпадение ────────────────────
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in query_lower:
            for syn in synonyms[:2]:
                if syn not in extra_queries and syn != query_lower:
                    extra_queries.append(syn)
                if len(extra_queries) >= 2:
                    break
            if extra_queries:
                break

    # ── Шаг 2: нечёткое совпадение (опечатки) ─────────────────────
    if not extra_queries:
        fuzzy_key = _fuzzy_match_keywords(query_lower)
        if fuzzy_key and fuzzy_key != query_lower:
            logger.debug(f"Fuzzy expansion: '{query_lower[:50]}' → '{fuzzy_key}'")
            synonyms = SYNONYM_MAP[fuzzy_key]
            extra_queries.append(fuzzy_key)
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


# ── Ключевые слова для «списочных» запросов ───────────────────────────────────
# Если вопрос касается полного перечня специальностей / факультетов,
# обычных лимитов (5 чанков, 2 с одной страницы) катастрофически мало.
# Для таких запросов переключаемся в «режим каталога»:
#   top_k          → 25   (больше кандидатов из Qdrant)
#   MAX_CONTEXT_CHUNKS → 15  (больше чанков в контекст GPT)
#   max_per_page   → 5   (больше чанков с одной страницы)
#   SCORE_GAP_THRESHOLD → 0.40  (не обрезаем релевантный хвост)
LIST_QUERY_KEYWORDS = [
    # Russian
    "специальност", "направлени", "образовательн", "программ",
    "факультет", "кафедр", "перечень", "список", "все направлени",
    "какие специальност", "какие факультет", "какие направлени",
    "есть специальност", "есть факультет",
    # Kazakh
    "мамандық", "мамандықтар", "факультеттер", "білім беру",
    "бағдарлама", "тізімі",
    # English
    "specialt", "facult", "program", "department", "list of",
]

LIST_TOP_K              = 25
LIST_MAX_CHUNKS         = 15
LIST_MAX_PER_PAGE       = 5
LIST_SCORE_GAP          = 0.40   # шире окно — не обрезаем релевантные хвосты
LIST_QDRANT_THRESHOLD   = 0.15   # ниже порог Qdrant-запроса — пускаем больше кандидатов
LIST_MIN_CONFIDENT      = 0.15   # ниже порог уверенности — факультетные страницы могут давать 0.20-0.25


def _is_list_query(question: str) -> bool:
    """
    Определяет, является ли вопрос «каталожным» — требует перечисления.

    Примеры:
      «Какие специальности есть в ЦАИУ?»   → True
      «Сколько стоит обучение?»             → False
      «Перечисли все факультеты»            → True
    """
    q = question.lower()
    return any(kw in q for kw in LIST_QUERY_KEYWORDS)


def search(
    question: str,
    top_k: int = None,
    min_score: float = None,
) -> List[SearchResult]:
    """
    Find most relevant chunks for a user question.

    Pipeline:
    1. Нормализация запроса
    2. Определение режима: обычный vs «каталог» (специальности/факультеты)
    3. Расширение запроса через SYNONYM_MAP (query expansion)
    4. Семантический поиск (batch embeddings → Qdrant cosine similarity)
    5. Проверка уверенности (confidence check)
    6. Score gap filter — отсекаем нерелевантный хвост
    7. Дедупликация + лимит по страницам
    8. Обрезка до MAX_CONTEXT_CHUNKS

    Question-chunk indexing (at ingest time) replaced BM25 runtime search:
    for each chunk, GPT generated 4 questions stored as extra vectors.
    User queries now match these question-vectors with high precision.

    Args:
        question:  User's question in any language
        top_k:     How many chunks to fetch from Qdrant (default from config)
        min_score: Minimum cosine similarity (default from config)

    Returns:
        List of SearchResult ordered by relevance (best first),
        already filtered, deduplicated, capped at MAX_CONTEXT_CHUNKS.
        Поле score содержит cosine similarity (0.0–1.0).
    """
    if top_k is None:
        top_k = settings.TOP_K_RESULTS
    if min_score is None:
        min_score = settings.SIMILARITY_THRESHOLD

    # ── Step 1: Normalize query ───────────────────────────────
    question = _normalize_query(question)
    if not question:
        return []

    # ── Step 2: Detect «catalog» mode ────────────────────────
    # Для вопросов о специальностях / факультетах увеличиваем все лимиты,
    # чтобы GPT видел полный список, а не первые 5 совпадений.
    catalog_mode = _is_list_query(question)
    if catalog_mode:
        top_k           = max(top_k, LIST_TOP_K)
        max_chunks      = LIST_MAX_CHUNKS
        max_per_pg      = LIST_MAX_PER_PAGE
        score_gap       = LIST_SCORE_GAP
        # Ключевое: снижаем score_threshold для Qdrant-запроса — иначе
        # страницы факультетов (score ~0.20-0.25) не дойдут до нашего кода
        qdrant_threshold    = LIST_QDRANT_THRESHOLD
        min_confident_score = LIST_MIN_CONFIDENT
        logger.info(f"Catalog mode enabled for: '{question[:60]}'")
    else:
        max_chunks          = settings.MAX_CONTEXT_CHUNKS
        max_per_pg          = 2
        score_gap           = settings.SCORE_GAP_THRESHOLD
        qdrant_threshold    = min_score          # стандартный порог из config
        min_confident_score = settings.MIN_CONFIDENT_SCORE

    # ── Step 3: Expand query into variants ────────────────────
    query_variants = _expand_query(question)
    logger.debug(f"Query variants: {query_variants}")

    # ── Step 3: Semantic search (Qdrant) ──────────────────────
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
    semantic_results: List[SearchResult] = []
    seen_ids: set = set()

    for variant, vector in zip(query_variants, vectors):
        if vector is None:
            continue
        try:
            hits = qdrant.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                score_threshold=qdrant_threshold,   # в catalog mode = 0.15
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

                # ── Фильтр мусорных чанков ─────────────────────────────────
                # Пропускаем чанки короче 80 символов — это почти всегда
                # заголовки страниц или навигационные блоки без реального содержания.
                # Пример: "[вопросы и ответы (блог ректора)]\nвопросы и ответы..." —
                # такой чанк содержит только название, а не ответ на вопрос.
                if len(text.strip()) < 80:
                    logger.debug(f"Filtered short chunk ({len(text)} chars): {text[:60]!r}")
                    continue

                result = SearchResult(
                    text=text,
                    page_url=payload.get("page_url", ""),
                    page_title=payload.get("page_title", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    score=round(hit.score, 4),
                    tags=payload.get("tags", []),
                )
                semantic_results.append(result)

        except Exception as e:
            logger.warning(f"Search failed for variant '{variant[:60]}': {e}")
            continue

    # Сортируем семантику по score (best first)
    semantic_results.sort(key=lambda r: r.score, reverse=True)

    semantic_best = semantic_results[0].score if semantic_results else 0.0

    # ── Step 4: Confidence check ──────────────────────────────────
    # Если лучший результат ниже порога — вопрос не по теме, возвращаем пустой.
    # В catalog_mode порог снижен до 0.15: страницы факультетов/специальностей
    # могут давать score ~0.20-0.25 (разреженный контент), и их нельзя отбрасывать.
    if semantic_best < min_confident_score:
        logger.debug(
            f"Not confident: best={semantic_best:.4f} < {min_confident_score} "
            f"(catalog={catalog_mode}) — returning empty"
        )
        return []

    # ── Step 5: Score gap filter ──────────────────────────────────
    # В catalog_mode используем более широкий score_gap (0.40 вместо 0.25),
    # чтобы не отсекать страницы с отдельными специальностями/факультетами.
    if not semantic_results:
        return []

    best_score = semantic_results[0].score
    min_acceptable = best_score - score_gap
    filtered = [r for r in semantic_results if r.score >= min_acceptable]

    if not filtered:
        return []

    # ── Step 6: Deduplicate and limit per-page density ───────────
    # В catalog_mode max_per_pg=5, чтобы взять несколько чанков со страниц
    # с длинными списками специальностей (они часто разбиты на 3-4 чанка).
    deduped = _deduplicate_results(filtered, max_per_page=max_per_pg)

    # ── Step 7: Cap at MAX_CONTEXT_CHUNKS ────────────────────────
    final = deduped[:max_chunks]

    logger.info(
        f"Search '{question[:60]}': "
        f"catalog={catalog_mode}, semantic={len(semantic_results)}, best={semantic_best:.4f}, "
        f"after_gap={len(filtered)}, after_dedup={len(deduped)}, final={len(final)}"
    )

    return final


def format_results_as_context(question: str, results: List[SearchResult]) -> str:
    """
    Форматирует уже найденные результаты в строку контекста для ChatGPT.

    Принимает ГОТОВЫЕ результаты — НЕ вызывает search() повторно.
    Используется в main.py чтобы не делать два запроса к OpenAI Embeddings.

    Returns a single string with all relevant chunks combined,
    trimmed to MAX_CONTEXT_CHARS (или вдвое больше для каталожных запросов).

    Format:
        [Источник: Общежитие]
        Студенческое общежитие КАИУ находится по адресу...

        [Источник: Специальности]
        Университет предлагает следующие специальности...
    """
    if not results:
        return ""

    # Для каталожных запросов разрешаем вдвое больше символов,
    # чтобы все специальности/факультеты поместились в контекст GPT.
    if _is_list_query(question):
        max_chars = settings.MAX_CONTEXT_CHARS * 2  # ~12 000 символов
    else:
        max_chars = settings.MAX_CONTEXT_CHARS

    context_parts = []
    total_chars = 0

    for result in results:
        source = result.page_title or result.page_url
        part = f"[Источник: {source}]\n{result.text}"

        # Don't exceed max context size
        if total_chars + len(part) > max_chars:
            # If this is the first chunk — include it truncated
            if not context_parts:
                truncated = part[:max_chars]
                context_parts.append(truncated)
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)


def search_and_format_context(question: str) -> str:
    """
    Search for relevant chunks and format them as context for ChatGPT.

    Вызывает search() и format_results_as_context() последовательно.
    Используется только там где нет готовых результатов.
    Если результаты уже есть — используй format_results_as_context() напрямую.
    """
    results = search(question)
    return format_results_as_context(question, results)
