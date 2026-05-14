# TESTS.md — Полный гид по тестам ЦАИУ RAG-системы

> Читать вместе с `CLAUDE.md`. Этот файл описывает **все тесты**: что проверяют, какие ресурсы тратят, как запускать.

---

## Карта тестов

| Файл | Тип | Интернет | Токены OpenAI | Docker | Время |
|---|---|---|---|---|---|
| `tests/test_chunker.py` | Unit | ❌ нет | ❌ нет | ❌ нет | ~2 сек |
| `tests/test_extractor.py` | Unit | ❌ нет | ❌ нет | ❌ нет | ~2 сек |
| `tests/test_search.py` | Unit | ❌ нет | ❌ нет | ❌ нет | ~2 сек |
| `tests/test_storage.py` | Unit | ❌ нет | ❌ нет | ❌ нет | ~2 сек |
| `tests/test_concurrency.py` | Unit | ❌ нет | ❌ нет | ❌ нет | ~3 сек |
| `tools/load_test.py` | Integration | ✅ нужен | ✅ тратит | ✅ нужен | 1–5 мин |

**Вывод:** все файлы в папке `tests/` — бесплатные. `load_test.py` — реальный тест, тратит деньги.

---

## Быстрый старт

```bash
# Перейти в нужную папку
cd rag_service

# Запустить ВСЕ unit-тесты (бесплатно, без Docker)
pytest ../tests/ -v

# Запустить конкретный файл
pytest ../tests/test_chunker.py -v

# Запустить только один тест
pytest ../tests/test_search.py::TestExpandQuery::test_synonym_expansion -v

# Запустить с отчётом покрытия
pytest ../tests/ -v --tb=short
```

---

## Unit-тесты (папка `tests/`)

### Ресурсы

Все unit-тесты:
- **Интернет**: не нужен
- **OpenAI токены**: не тратятся (все вызовы замоканы)
- **Docker / Qdrant / Redis**: не нужны
- **Деньги**: $0.00
- **Время**: ~10 секунд на все 5 файлов вместе

---

### `tests/test_chunker.py` — разбивка текста на чанки

**Что тестирует:** логику `app.parser.chunker` — как текст делится на фрагменты по ~200 слов с учётом заголовков h2/h3.

**Ключевые сценарии:**
- Heading-aware splitting: `## Заголовок` становится границей чанка
- Чанк не превышает `CHUNK_SIZE` слов
- Overlap (перекрытие): последние N слов предыдущего чанка начинают следующий
- Короткие секции не создают пустых чанков
- Каждый чанк получает правильный prefix `[Страница > Раздел]`
- Граничные случаи: пустой текст, один абзац, только заголовки

**Запуск:**
```bash
cd rag_service && pytest ../tests/test_chunker.py -v
```

---

### `tests/test_extractor.py` — парсинг HTML страниц

**Что тестирует:** логику `app.parser.extractor` — как HTML превращается в чистый текст с маркерами заголовков.

**Ключевые сценарии:**
- `h2`/`h3`/`h4` теги → `## Заголовок` маркеры
- Удаление навигации, футеров, скриптов
- Сохранение полезного контента
- Извлечение `external_links` (Google Drive, Docs, PDF, Dropbox)
- Обработка страниц без заголовков
- Страницы с пустым или минимальным контентом

**Запуск:**
```bash
cd rag_service && pytest ../tests/test_extractor.py -v
```

---

### `tests/test_search.py` — поиск по базе знаний

**Что тестирует:** чистые функции в `app.retrieval.search` без обращения к Qdrant или OpenAI.

**Ключевые сценарии:**
- `_normalize_query()`: нормализация регистра и пробелов
- `_expand_query()`: синонимы ("общага" → "общежитие", "бакалавр" → "бакалавриат")
- Сортировка ключей SYNONYM_MAP по длине (составные ключи приоритетнее коротких)
- `_apply_score_gap_filter()`: отсечение нерелевантного хвоста по разрыву score
- `_deduplicate_results()`: дедупликация по `(page_url, chunk_index)`
- `get_candidate_urls()` с заниженным порогом (0.10 вместо 0.28)

**Запуск:**
```bash
cd rag_service && pytest ../tests/test_search.py -v
```

---

### `tests/test_storage.py` — работа с Qdrant state

**Что тестирует:** функции управления состоянием (`hot_swap_state.json`) в `app.indexer.storage`.

**Ключевые сценарии:**
- `get_last_indexed_at()`: читает время последней индексации
- `set_last_indexed_at()`: атомарная запись через `.tmp` файл
- Корректная работа с отсутствующим файлом состояния
- `get_active_collection()`: возвращает blue/green или базовое имя
- Изоляция через `tmp_path` — не пишет в реальную папку проекта

**Запуск:**
```bash
cd rag_service && pytest ../tests/test_storage.py -v
```

---

### `tests/test_concurrency.py` — поведение под нагрузкой

**Что тестирует:** корректность работы при одновременных запросах от многих пользователей.

**Ключевые сценарии:**

| Класс | Что проверяет |
|---|---|
| `TestRateLimiter` | 10 сообщений — проходят, 11-е — блокируется |
| `TestRateLimiter` | Лимит user1 не влияет на user2 |
| `TestRateLimiter` | 20 потоков одновременно — нет race condition |
| `TestSearchCache` | Второй вызов с тем же вопросом — не идёт в OpenAI |
| `TestSearchCache` | 50 параллельных чтений из кеша — без ошибок |
| `TestSearchSemaphore` | Не более N поисков одновременно (семафор работает) |
| `TestSearchSemaphore` | Семафор освобождается даже при исключении |
| `TestSearchSemaphore` | При перегрузке — мгновенный 503, не зависание |
| `TestTranslationCache` | LRU: старые записи вытесняются при MAX записях |
| `TestBotHttpxPool` | httpx.Limits создаются с правильными значениями |

**Запуск:**
```bash
cd rag_service && pytest ../tests/test_concurrency.py -v

# Только rate limiter
pytest ../tests/test_concurrency.py -v -k "RateLimit"

# Только семафор
pytest ../tests/test_concurrency.py -v -k "Semaphore"
```

---

## Нагрузочный тест (`tools/load_test.py`)

### ⚠️ Тратит реальные ресурсы

| Ресурс | Расход |
|---|---|
| **Интернет** | Нужен (запросы к OpenAI) |
| **OpenAI токены** | ~0.001$ за embedding × кол-во уникальных вопросов |
| **Docker / Qdrant / Redis** | Нужны (RAG-сервис должен быть запущен) |
| **Время** | 30 сек – 5 мин в зависимости от числа пользователей |

**Сколько тратит:** при 20 уникальных вопросах — ~$0.002 (очень мало). При повторных раундах большинство вопросов берутся из Redis-кеша → 0 токенов.

### Требования

```bash
# Перед запуском — убедиться что работает:
docker compose up -d            # Qdrant + Redis
cd rag_service && python run_api.py  # RAG-сервис (в отдельном терминале)
```

### Команды

```bash
# Из корня проекта:

# Базовый тест — 10 одновременных пользователей
python tools/load_test.py

# Пиковая нагрузка периода приёма
python tools/load_test.py --users 30

# Стресс: 50 пользователей, 3 волны
python tools/load_test.py --users 50 --rounds 3

# Тест продакшн-сервера с другого компьютера
python tools/load_test.py --url http://123.45.67.89:8001 --users 20

# Сохранить детальный JSON-отчёт
python tools/load_test.py --users 30 --save
```

### Чтение результатов

```
РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТА
Пользователей:   30
Всего запросов:  30
Успешных:        30  (100.0%)
Ошибок:          0   (0.0%)

── Время ответа (успешные) ──────────────────
  Минимум:   312 мс        ← вопрос из Redis-кеша
  Медиана:   1840 мс       ← типичный ответ с OpenAI
  Среднее:   1923 мс
  Максимум:  4210 мс       ← нагрузка на OpenAI в пике
  P95:       3800 мс       ← 95% пользователей ждут не дольше

── Вердикт ─────────────────────────────────
  ✅ ОТЛИЧНО — сервис справляется с нагрузкой
```

**На что смотреть:**

| Метрика | Хорошо | Проблема |
|---|---|---|
| Успешных | > 99% | < 95% — ошибки под нагрузкой |
| P95 | < 5000 мс | > 10000 мс — пользователи ждут слишком долго |
| Ошибки 503 | 0 | > 0 — нужно увеличить `MAX_CONCURRENT_SEARCHES` |
| Ошибки 429 | 0 | > 0 — превышен лимит OpenAI (снизить `MAX_CONCURRENT_SEARCHES`) |
| Таймауты | 0 | > 0 — сервис не справляется, нужно больше workers |

---

## Как запускать все тесты вместе

```bash
cd rag_service

# Все unit-тесты (5 файлов, ~10 сек, $0)
pytest ../tests/ -v

# С кратким выводом (только что упало)
pytest ../tests/ -v --tb=short

# Только конкретный класс тестов
pytest ../tests/test_concurrency.py::TestRateLimiter -v

# Только тесты с определённым именем
pytest ../tests/ -v -k "cache"

# Показать самые медленные тесты
pytest ../tests/ -v --durations=10
```

---

## Ожидаемый результат при `pytest ../tests/ -v`

```
tests/test_chunker.py::test_empty_text PASSED
tests/test_chunker.py::test_single_paragraph PASSED
tests/test_chunker.py::test_heading_splits PASSED
...
tests/test_extractor.py::test_heading_markers PASSED
tests/test_extractor.py::test_external_links_google_drive PASSED
...
tests/test_search.py::TestNormalizeQuery::test_lowercase PASSED
tests/test_search.py::TestExpandQuery::test_synonym_expansion PASSED
...
tests/test_storage.py::test_get_last_indexed_at_missing PASSED
tests/test_storage.py::test_set_last_indexed_at PASSED
...
tests/test_concurrency.py::TestRateLimiter::test_allows_up_to_limit PASSED
tests/test_concurrency.py::TestRateLimiter::test_blocks_after_limit PASSED
tests/test_concurrency.py::TestSearchSemaphore::test_semaphore_limits_concurrency PASSED
...

====== X passed in ~10s ======
```

Если какой-то тест упал — значит был сделан баг в соответствующем модуле.

---

## Добавление новых тестов

Новые test-файлы кладутся в `tests/`. Правила:
- Имя файла: `test_*.py`
- Все внешние зависимости (OpenAI, Qdrant, Redis) мокируются через `unittest.mock.patch`
- Тест должен работать без Docker и интернета
- Используй `tmp_path` (pytest fixture) для временных файлов вместо реальных путей

Пример:
```python
# tests/test_my_feature.py
import pytest
from unittest.mock import patch, MagicMock

def test_my_function():
    with patch("app.my_module.openai_client") as mock_openai:
        mock_openai.return_value = MagicMock(...)
        result = my_function("input")
        assert result == "expected"
```
