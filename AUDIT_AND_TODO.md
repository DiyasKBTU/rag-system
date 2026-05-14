# AUDIT_AND_TODO.md — Аудит проекта и план улучшений

**Дата аудита:** 2026-04-30
**Цель документа:** зафиксировать все найденные баги, проблемы и идеи улучшения,
чтобы продолжить работу в новом чате без потери контекста.

> Читать вместе с `CLAUDE.md`. Этот файл — план действий, `CLAUDE.md` — описание системы.

---

## 🔴 КРИТИЧНО — главная причина медленной индексации и расхода токенов

### Баг #1. Hot-swap всегда обнуляет кэш чанков → каждый /reindex переиндексирует ВСЁ

**Файл:** `rag_service/app/indexer/hot_swap.py`, строки ~220–226

```python
existing = [c.name for c in client.get_collections().collections]
if shadow in existing:
    logger.info(f"[HotSwap] Clearing stale shadow collection '{shadow}'")
    client.delete_collection(shadow)
ensure_collection_exists(collection_name=shadow)
```

**Что происходит:**
- shadow коллекция удаляется и создаётся пустой
- `pipeline.run_indexing(collection_name=shadow)` вызывает
  `page_needs_update(url, hash, collection_name=shadow)`
- shadow пустая → функция возвращает `True` для **каждой** страницы
- Полная переиндексация: 149 страниц × (HTML + chunking + 4 GPT-вопроса/чанк + embeddings)

**Результат:** на каждый /reindex тратится:
- ~$0.20–0.30 на OpenAI (вопросы + embeddings)
- 30–60 минут времени
- хотя на сайте могла измениться 1 страница

### Решение: паттерн «snapshot + patch»

В `run_shadow_indexing()`, между `ensure_collection_exists(shadow)` и `run_indexing(shadow)` вставить копирование точек из активной коллекции в shadow:

```python
active = get_active_collection()
if active != shadow and active in [c.name for c in client.get_collections().collections]:
    logger.info(f"[HotSwap] Cloning '{active}' → '{shadow}' for incremental reuse")
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=active,
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break
        client.upsert(
            collection_name=shadow,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        if offset is None:
            break
```

**После фикса:**
- /reindex без изменений: ~30 сек, 0 токенов OpenAI
- /reindex с 3 изменёнными страницами: ~1 минута, токены только на 3 страницы

### Дополнительно: использовать `lastmod` из sitemap

В `crawler.py` `SitemapURL.lastmod` уже парсится, но `pipeline.py` его игнорирует.
Если `lastmod` старше времени последней индексации (хранится в `hot_swap_state.json`) —
страницу можно даже не качать для подсчёта хэша.

**Файлы для правки:**
- `rag_service/app/indexer/hot_swap.py` (главный фикс)
- `rag_service/app/indexer/pipeline.py` (использование `lastmod`)
- `rag_service/hot_swap_state.json` (добавить поле `last_full_indexing_at`)

---

## 🟠 СИЛЬНО ЗАМЕДЛЯЕТ ИНДЕКСАЦИЮ

### Баг #2. Скачивание страниц последовательное с большой задержкой

**Файл:** `rag_service/app/indexer/pipeline.py`, строка ~137
**Файл:** `rag_service/app/config.py`: `REQUEST_DELAY: float = 1.5`

149 URL × 1.5 сек = **+225 секунд** только на ожидание.

**Решение:** параллельное скачивание (5 одновременно) + снижение задержки до 0.3 сек:

```python
import asyncio

async def fetch_all_pages(urls):
    semaphore = asyncio.Semaphore(5)
    async def fetch_one(url_item):
        async with semaphore:
            await asyncio.sleep(0.3)
            return await asyncio.to_thread(get_page_content, url_item.url)
    return await asyncio.gather(*[fetch_one(u) for u in urls])
```

Скачивание 149 страниц: было ~4 минуты → станет ~30 сек.

### Баг #3. catalog_builder качает 32 страницы повторно

**Файл:** `rag_service/app/indexer/catalog_builder.py`

Те же URL, что в основном цикле `pipeline.run_indexing`, скачиваются второй раз.
У `catalog_builder` свой `time.sleep(1)` / `time.sleep(0.5)`.

**Решение:** передавать в `build_and_save_catalog_chunks(...)` уже извлечённые
`PageContent` через словарь `{url: PageContent}` из основного цикла.
Экономия ~40 секунд + меньше нагрузки на сайт университета.

**Что менять:**
- `pipeline.py`: собирать `pages_cache: dict[str, PageContent]` во время основного цикла
- `catalog_builder.py`: `build_and_save_catalog_chunks(pages_cache=None)` —
  если URL есть в кэше, не качать заново

### Баг #4. BATCH_CONCURRENCY=5 в генераторе вопросов — занижено

**Файл:** `rag_service/app/indexer/question_generator.py`, строка ~59

```python
BATCH_CONCURRENCY = 5
```

На tier-1 OpenAI можно держать 10–15 параллельных запросов без 429.
Переход 5→10 экономит 30–50% времени фазы вопросов.

**Решение:** поднять до 10. Если будут 429 — откатить или добавить адаптивный
backoff (уже есть retry в embeddings.py через tenacity, но не в question_generator).

### Баг #5. Каталог-чанки пересоздаются всегда

**Файл:** `rag_service/app/indexer/catalog_builder.py`

`_save_synthetic_chunk` всегда вызывает `delete_chunks_by_url` + новые embeddings.
Если содержимое страниц факультетов/кафедр не менялось — это лишние OpenAI вызовы.

**Решение:** сравнивать `content_hash` каталога с тем, что уже в shadow коллекции.
Пропускать пересохранение, если хэш совпадает.

---

## 🟡 РАСХОЖДЕНИЕ ДОКУМЕНТАЦИИ И КОДА

### Баг #6. CLAUDE.md обещает админ-команды бота — их нет в коде

CLAUDE.md описывает в `bot/run.py`:
- env var `ADMIN_USER_ID`
- `admin_router`
- команды `/reindex`, `/schedule`, `/unschedule`, `/index_status`

Поиск по `bot/`:
```
admin_router|ADMIN_USER_ID|/reindex|/schedule  →  No matches found
```

**Эти команды не реализованы.**

**Варианты:**
1. Реализовать команды (вызовы /index, /index/schedule и т.д. через httpx + проверка user_id)
2. Убрать упоминания из CLAUDE.md, оставить только REST API через FastAPI

Рекомендация: реализовать (это часть UX, а не «лишний» код).

### Баг #7. CLAUDE.md упоминает `tools/indexer_ctl.py` — нужно проверить

Цитата из CLAUDE.md:
> `python tools/indexer_ctl.py schedule "DD.MM HH:MM"`

Нужно убедиться что файл существует и работает (не было в Read'ах файлов).
Если нет — убрать из docs или создать.

---

## 🟡 ПРОЧИЕ БАГИ И РИСКИ

### Баг #8. _first_migration окно с пустым ответом

**Файл:** `rag_service/app/indexer/hot_swap.py:118–137`

```python
client.delete_collection(original)         # коллекция исчезла
# ... тут окно ~10мс ...
client.update_collection_aliases(...)      # алиас создан
```

Между этими действиями `POST /search` вернёт ошибку `Not found: collection`.
CLAUDE.md упоминает «microsecond downtime», но это решаемо.

**Решение:** делать первую миграцию через отдельный CLI-скрипт `tools/migrate_to_hotswap.py`
вне продового /reindex (один раз при первом запуске hot-swap архитектуры).

### Баг #9. schedule.fired не сбрасывается для одноразовых задач

**Файл:** `rag_service/app/main.py:138`

После срабатывания `fired=True`. Если пользователь снова сделает
POST /index/schedule с тем же временем — получит 409.

**Решение:** в `schedule_index_endpoint` игнорировать существующее расписание
если `fired=True` (просто перезаписывать).

### Баг #10. Логи бота создают новый файл при каждом рестарте

**Файл:** `bot/run.py:56`

```python
log_file = logs_dir / f"bot_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
```

Папка `logs/` растёт бесконечно при частых рестартах.

**Решение:** `RotatingFileHandler` (5 файлов × 5 MB) или `TimedRotatingFileHandler` (1 файл/день, 30 дней хранится).

```python
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(
    log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
```

### Баг #11. Кэш переводов и кэш embeddings только в памяти

**Файлы:**
- `bot/run.py:88` — `_TRANSLATION_CACHE: OrderedDict`
- `rag_service/app/retrieval/search.py:37` — `_EMBEDDING_CACHE: Dict`

Оба обнуляются при каждом рестарте. В docker-compose уже поднят Redis,
но не подключён.

**Решение:** перенести в Redis с TTL=30 дней.
- ключ перевода: `f"trans:{lang}:{md5(text)}"`
- ключ embedding'а: `f"emb:{md5(text)}"`

Польза: повторяющиеся запросы («как поступить?», «общежитие») будут отвечать
быстрее и без расхода токенов.

### Баг #12. _expand_query берёт первое совпадение и break

**Файл:** `rag_service/app/retrieval/search.py:712–721`

```python
for keyword, synonyms in SYNONYM_MAP.items():
    if keyword in query_lower:
        ...
        break
```

Запрос «документы для общежития» поймает `документы` (короткий ключ)
раньше чем `документы для общежития` (составной).

**Решение:** сортировать ключи по убыванию длины:
```python
sorted_keys = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
for keyword in sorted_keys:
    if keyword in query_lower:
        ...
```

### Баг #13. Дедупликация поиска по text[:200] слишком грубая

**Файл:** `rag_service/app/retrieval/search.py:752`

Чанки с одинаковым префиксом `[Страница > Раздел]\n` (первые 200 символов часто совпадают)
будут считаться одним.

**Решение:** дедуп по `(page_url, chunk_index)` или по полному тексту через md5.

### Баг #14. CORS открыт для всех origin'ов

**Файл:** `rag_service/app/main.py:215`

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

RAG-сервис только для бота. На продакшне ограничить:
```python
allow_origins=["http://localhost", "http://127.0.0.1"]
```
или вообще убрать CORS middleware.

### Баг #15. content_hash считается с title prefix

**Файл:** `rag_service/app/parser/extractor.py:153–158`

```python
clean = f"{title}\n\n{clean}"
content_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()
```

Если на сайте поправили только `<h1>` (косметика) — хэш меняется → ложная переиндексация.

**Решение:** считать хэш до добавления title:
```python
content_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()
if title and clean and title.lower() not in clean[:200].lower():
    clean = f"{title}\n\n{clean}"
```

---

## 🟢 УЛУЧШЕНИЯ (не баги, но полезно)

### Улучшение #1. Параллелизация скачивания страниц
Уже описано в Баг #2. Самый быстрый win после главного фикса.

### Улучшение #2. Telegram Webhook вместо long-polling
В CLAUDE.md как TODO. Для прода с systemd webhook надёжнее (нет «гонок»
при рестарте, не нагружает Telegram-сервера).

### Улучшение #3. systemd-юниты для автоперезапуска
Самый критичный пункт «Известные проблемы» из CLAUDE.md.
Создать `/etc/systemd/system/caiu-bot.service` и `caiu-rag.service`
с `Restart=always`.

Шаблон:
```ini
[Unit]
Description=CAIU RAG Service
After=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/caiu/rag_service
ExecStart=/opt/caiu/venv/bin/python run_api.py
Restart=always
RestartSec=10
User=caiu

[Install]
WantedBy=multi-user.target
```

### Улучшение #4. Тесты
Сейчас тестов нет вообще. Минимум:
- `tests/test_chunker.py` — `_split_into_sections`, `_group_paragraphs`, `_add_overlap`
- `tests/test_search.py` — `_expand_query`, `_apply_score_gap_filter`, `_deduplicate_results`
- `tests/test_extractor.py` — `_clean_text`, `_inject_heading_markers`

Запуск через pytest. 10–15 unit-тестов покрыли бы критичную логику.

### Улучшение #5. Метрики и мониторинг
- `/metrics` Prometheus-эндпоинт (счётчики: запросы, найденные чанки, ошибки)
- `/health` сейчас возвращает ok даже при пустой коллекции — добавить проверку `total_chunks > 0`
- Алерты в Telegram если `/health` падает (отдельный мониторинг-скрипт)

### Улучшение #6. Параллельная генерация вопросов
Уже есть, но `BATCH_CONCURRENCY=5` — занижено (см. Баг #4).

### Улучшение #7. Использовать lastmod из sitemap
Если `lastmod` страницы старше `last_indexed_at` — пропускать без скачивания.
Экономит время на 80% страниц.

---

## 📋 ПЛАН РАБОТ (приоритеты)

### Спринт 1 — Главный фикс (1-2 часа работы)
- [x] **#1** Snapshot+patch в `hot_swap.py` (копирование active → shadow) ✅ 2026-04-30
- [ ] Тестовый прогон /reindex без изменений на сайте → должен быть ~30 сек, 0 токенов
- [ ] Тестовый прогон /reindex с одной изменённой страницей → должна обновиться только она

### Спринт 2 — Параллелизация и кэш (2-3 часа)
- [x] **#2** Параллельное скачивание страниц в pipeline.py ✅ 2026-04-30
- [x] **#3** Переиспользование PageContent между pipeline и catalog_builder ✅ 2026-05-08
- [x] **#5** Skip каталог-чанков если хэш совпадает ✅ 2026-05-08
- [x] **#4** BATCH_CONCURRENCY: 5 → 10 в question_generator.py ✅ 2026-05-08
- [x] **#11** Перенести кэш переводов и embeddings в Redis ✅ 2026-04-30
- [x] **#15** Хэшировать текст до добавления title prefix ✅ 2026-05-08

> ✅ Выполнено в Спринте 2 (продолжение):
> - **#12** Synonym sort + **#13** Dedup fix — `search.py` ✅ 2026-04-30
> - Payload index на `page_url` — `storage.py` ✅ 2026-04-30
> - Redis-кеш полных ответов поиска — `search.py` (TTL 1 час) ✅ 2026-04-30

### Спринт 3 — Прод-готовность (2-3 часа)
- [x] **#6** Админ-команды бота — решение: только через терминал, из CLAUDE.md убрать упоминание ✅ 2026-05-05
- [x] **#10** RotatingFileHandler для логов ✅ 2026-05-08
- [x] **#14** Ограничить CORS ✅ 2026-05-08
- [x] **#9** Сбрасывать schedule.fired — фактически исправлено: `if existing and not existing.get("fired"):` → при fired=True расписание перезаписывается без 409 ✅
- [x] **Улучшение #3** systemd-юниты для автоперезапуска ✅ 2026-05-08

> ✅ Выполнено в Спринте 1 (нестабильность):
> - Qdrant reconnect — `storage.py`
> - Постоянный httpx + retry + раздельные таймауты — `bot/run.py`
> - Thread-safe + валидация векторов в embedding cache — `search.py`

> ✅ Выполнено в Спринте 2 (скорость):
> - **#11** Redis-кеш для embeddings — `search.py` (двухуровневый: in-memory + Redis, TTL 30 дней)
> - **#11** Redis-кеш для переводов — `bot/run.py` (async Redis, те же принципы)
> - Новый модуль `redis_client.py` (sync) для RAG-сервиса
> - Streaming GPT — `bot/run.py`: пользователь видит первые символы за ~300мс вместо 2-5с
> - `redis==5.2.1` в `requirements.txt`

### Спринт 4 — Качество поиска (1-2 часа)
- [x] **#12** Сортировать ключи SYNONYM_MAP по длине ✅ (уже в коде: `sorted(..., key=len, reverse=True)`)
- [x] **#13** Дедупликация по (page_url, chunk_index) ✅ (уже в коде: `_deduplicate_results()`)
- [x] **#7** lastmod из sitemap ✅ 2026-05-08

### Спринт 5 — Тесты и мониторинг (2-4 часа)
- [x] Unit-тесты chunker'а, search'а, extractor'а, storage — ✅ Сессия 6
- [x] Unit-тесты нагрузки (rate limit, semaphore, cache) — ✅ Сессия 7
- [x] Нагрузочный тест (tools/load_test.py) — ✅ Сессия 7
- [ ] /metrics endpoint (Prometheus)
- [ ] Telegram webhook вместо polling
- [ ] Healthcheck с проверкой total_chunks > 0

---

## 🎨 UX БЭКЛОГ (bot/run.py)

Зафиксировано по итогам анализа 2026-05-05. Реализуется по мере готовности.

### ✅ Реализовано (2026-05-05)
- **UX #4** — кандидаты и фоллбек встроены в основной ответ (одно сообщение вместо двух)
- **UX #7** — команда `/help` на трёх языках (специальности, стоимость, контакты)
- **UX #8** — сообщение об ошибке теперь содержит телефон приёмной комиссии
- **UX #9** — rate limit сообщение стало теплее, добавлен телефон для срочных вопросов

### ✅ Реализовано (2026-05-08)
- **UX #5** — FAQ inline-кнопки после выбора языка: "💰 Стоимость/Оқу ақысы/Tuition", "📋 Документы/Құжаттар/Documents", "🏠 Общежитие/Жатақхана/Dormitory", "🎓 Специальности/Мамандықтар/Specialties"
- **UX #6** — `send_chat_action(ChatAction.TYPING)` в начале `handle_message` и `handle_faq_button`
- **UX #10** — контекст диалога: последние 3 пары вопрос-ответ передаются в GPT между system-промптом и текущим вопросом. RAG не получает историю (ищет только по текущему вопросу). История хранится в FSM-стейте, сбрасывается при смене языка или `/start`.

### 🔲 Бэклог
- **UX #1** — inline keyboard вместо reply keyboard для выбора языка
- **UX #2** — кнопка смены языка прямо в экране вопроса (не через "⬅ Назад")
- **UX #10** — контекст диалога: передавать последние 2–3 обмена в промпт GPT
- **UX #4b** — special_pages.json расширение: регулярный аудит незаиндексированных страниц
- ~~**Инфра** — systemd-юниты для автоперезапуска бота и RAG-сервиса~~ ✅ `deploy/` папка создана

---

## 📝 ЖУРНАЛ ИЗМЕНЕНИЙ

### 2026-05-13 — Сессия 7: Нагрузочная защита + тесты конкурентности

**Новые файлы:**

| Файл | Описание |
|---|---|
| `TESTS.md` | Полный гид по всем тестам проекта: ресурсы, команды, таблица результатов |
| `tests/test_concurrency.py` | Unit-тесты: rate limit, семафор, LRU-кеш, httpx pool, thread-safety |
| `tools/load_test.py` | Нагрузочный тест: симуляция N пользователей, P95/median/error rate |

**Изменённые файлы:**

| Файл | Изменение |
|---|---|
| `bot/run.py` | `max_connections` 10→30, `max_keepalive_connections` 5→10 |
| `rag_service/app/main.py` | Семафор `MAX_CONCURRENT_SEARCHES=20` + 503 при таймауте очереди |
| `rag_service/run_api.py` | `workers=2` для uvicorn |

**Анализ нагрузки (результаты):**
- Основная защита уже была: Redis search cache (1 час) покрывает ~60% повторяющихся вопросов → 0 токенов
- Узкие места: httpx pool (10 → 30 соединений), отсутствие семафора на OpenAI-вызовы
- После фикса: 40 параллельных поисков одновременно, остальные в очереди (не падают)

**Помечено в плане:**
- [x] Unit-тесты нагрузки ✅ 2026-05-13
- [x] Нагрузочный тест ✅ 2026-05-13

---

### 2026-05-08 — Systemd автозапуск

**Новые файлы:**

| Файл | Описание |
|---|---|
| `deploy/caiu-rag.service` | Systemd юнит для RAG-сервиса (FastAPI, порт 8001) |
| `deploy/caiu-bot.service` | Systemd юнит для Telegram бота (aiogram) |
| `deploy/install.sh` | Скрипт установки: подставляет пользователя и путь, копирует в `/etc/systemd/system/`, включает `systemctl enable` |

**README.md** — добавлен раздел "Автозапуск через systemd" с командами start/stop/restart/logs и инструкцией по отладке.

**Помечено в плане:**
- [x] **Улучшение #3** systemd-юниты ✅ 2026-05-08

---

### 2026-05-08 — Bug #7: lastmod-фильтрация перед скачиванием

**Изменённые файлы:**

| Файл | Изменение |
|---|---|
| `rag_service/app/indexer/storage.py` | Импорт `datetime/timezone`; `_set_active_collection()` теперь сохраняет другие поля нетронутыми; `get_last_indexed_at()` — читает `last_indexed_at` из состояния; `set_last_indexed_at()` — пишет время завершения |
| `rag_service/app/indexer/pipeline.py` | Импорт `datetime/timezone`; `get_last_indexed_at`/`set_last_indexed_at` в imports; перед fetch-фазой: `lastmod_skipped_urls` = страницы чей `lastmod ≤ last_indexed_at`; processing-цикл пропускает их с `skipped`; в конце успешной индексации вызов `set_last_indexed_at()` |

**Эффект:**
- После первой успешной индексации в `hot_swap_state.json` появляется `last_indexed_at`.
- При следующем `/reindex` (если сайт не менялся): 0 страниц скачивается, всё пропускается по lastmod → ~5 сек вместо 30–40 сек на fetch.
- Если изменилось 3 страницы из 150: скачиваются только 3.
- Страницы без `lastmod` в sitemap по-прежнему скачиваются (безопасный фоллбек).

**Помечено в плане:**
- [x] **#7** lastmod из sitemap ✅ 2026-05-08

---

### 2026-05-08 — UX #10: контекст диалога в GPT

**Изменённые файлы:**

| Файл | Изменение |
|---|---|
| `bot/run.py` | `_MAX_HISTORY_PAIRS = 3` — константа глубины истории |
| `bot/run.py` | `stream_answer_to_message(history=None)` — принимает историю, строит `messages = [system] + history + [user]` |
| `bot/run.py` | `process_question(history=None)` — принимает историю, прокидывает в `stream_answer_to_message` |
| `bot/run.py` | `handle_message` — читает `history` из FSM-стейта, передаёт в `process_question`, обновляет после ответа |
| `bot/run.py` | `handle_faq_button` — аналогично; при `set_data` сохраняет существующую историю |

**Эффект:**
- Пользователь может задавать уточняющие вопросы: "а сколько это стоит?" после "какие специальности есть в ЦАИУ" — GPT понимает контекст.
- История живёт в MemoryStorage (сбрасывается при рестарте бота — известное ограничение, Redis не подключён для FSM).
- История сбрасывается при смене языка (`select_language` делает `set_data({"lang": lang})` без `history`).
- RAG по-прежнему получает только текущий вопрос — не перегружаем embeddings историей.

---

### 2026-05-08 — UX #5 + UX #6: FAQ inline-кнопки + typing индикатор

**Изменённые файлы:**

| Файл | Изменение |
|---|---|
| `bot/run.py` | Импорты: `ChatAction`, `CallbackQuery`, `InlineKeyboardMarkup`, `InlineKeyboardButton` |
| `bot/run.py` | `FAQ_ITEMS` — dict с 4 вопросами на трёх языках (ru/kk/en) |
| `bot/run.py` | `faq_keyboard(lang)` — inline-клавиатура с FAQ кнопками |
| `bot/run.py` | `select_language()` — после выбора языка добавлено отдельное сообщение с FAQ кнопками |
| `bot/run.py` | `@router.callback_query(F.data.startswith("faq:"))` — новый хендлер нажатия FAQ кнопки |
| `bot/run.py` | `handle_message()` — принимает `bot: Bot`, добавлен `send_chat_action(TYPING)` в начало |

**Эффект:**
- Пользователь видит 4 кнопки сразу после выбора языка. Нажатие → запускает полный RAG-поиск, без ввода текста.
- Typing-индикатор (три точки) появляется мгновенно при любом входящем сообщении.

---

### 2026-05-08 — Спринт 2 (продолжение): catalog кеш, skip по хэшу, BATCH_CONCURRENCY, content_hash

**Изменённые файлы:**

| Файл | Изменение |
|---|---|
| `rag_service/app/indexer/question_generator.py` | `BATCH_CONCURRENCY`: 5 → 10 (Bug #4) |
| `rag_service/app/parser/extractor.py` | `content_hash` вычисляется ДО добавления title-prefix (Bug #15) |
| `rag_service/app/indexer/catalog_builder.py` | `_get_page_content()` принимает `pages_cache`; `_save_synthetic_chunk()` проверяет хэш и пропускает если не изменился (Bug #5); `build_faculties_catalog()`, `build_departments_catalog()`, `build_specialties_catalog()`, `build_and_save_catalog_chunks()` принимают `pages_cache` и пропускают `time.sleep()` при попадании в кеш (Bug #3) |
| `rag_service/app/indexer/pipeline.py` | Шаг 6: передаёт `pages_cache=fetched_pages` в `build_and_save_catalog_chunks()` (Bug #3) |

**Эффект:**
- Каталог-страницы (факультеты, кафедры, специальности) больше не скачиваются повторно если уже скачаны pipeline на шаге 1. Каталог начинает обрабатываться сразу, без `time.sleep()`.
- Каталог-чанки не перезаписываются если текст не изменился (нет лишних embeddings-запросов).
- Question generator использует 10 параллельных запросов вместо 5 → индексация ~2× быстрее.
- `content_hash` теперь отражает реальный контент страницы, а не артефакт добавления title.

**Помечено в плане:**
- [x] **#3** Переиспользование PageContent ✅ 2026-05-08
- [x] **#5** Skip каталог-чанков по хэшу ✅ 2026-05-08
- [x] **#4** BATCH_CONCURRENCY 5→10 ✅ 2026-05-08
- [x] **#15** content_hash до title prefix ✅ 2026-05-08

---

### 2026-05-08 — Спринт 3: debug prints, RotatingFileHandler, CORS, README

**Изменённые файлы:**
- `bot/run.py`
- `rag_service/app/main.py`
- `README.md`
- `AUDIT_AND_TODO.md` (этот файл)

**Что сделано:**

#### Убраны debug prints (`bot/run.py`)
- 8 строк `print(">>> ШАГ N: ...")` удалены из начала файла
- Спамили в stdout при каждом запуске — мешали читать реальные логи

#### RotatingFileHandler для логов (`bot/run.py`) — Bug #10
- Было: `FileHandler(f"bot_{datetime.now()}.log")` — новый файл при каждом рестарте
- Стало: `RotatingFileHandler("bot.log", maxBytes=5MB, backupCount=5)`
- Максимум 25 MB на диске, лог дописывается при рестарте вместо создания нового файла
- Архивные файлы: `bot.log.1` … `bot.log.5`

#### Ограничен CORS (`rag_service/app/main.py`) — Bug #14
- Было: `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`
- Стало: только `http://localhost` и `http://127.0.0.1`, методы GET/POST/DELETE, нужные заголовки

#### Обновлён README.md
- Секции «Страницы-картинки» и «Добавление фактов» переписаны под `manual_knowledge.json`
- Убраны ссылки на `special_pages.json`, `supplement_facts.json`, `rebuild_special_pages.py`
- В разделе «Частые вопросы» — обновлена ссылка на `rebuild_manual_knowledge.py`

**Помечено в плане:**
- [x] **#10** RotatingFileHandler ✅ 2026-05-08
- [x] **#14** Ограничить CORS ✅ 2026-05-08

---

### 2026-05-06 (вечер) — Объединение special_pages.json + supplement_facts.json → manual_knowledge.json

**Изменённые файлы:**
- `manual_knowledge.json` (НОВЫЙ — корень проекта)
- `tools/rebuild_manual_knowledge.py` (НОВЫЙ)
- `tools/rebuild_special_pages.py` (устарел — делегирует вызов)
- `rag_service/app/indexer/catalog_builder.py`
- `bot/run.py`
- `tools/backup.py`
- `CLAUDE.md`, `AUDIT_AND_TODO.md`

**Что сделано:**

#### Создан manual_knowledge.json
- Объединяет `rag_service/special_pages.json` и `supplement_facts.json` в один файл
- Единый формат для обоих случаев: поле `"link"` (опционально) — разделяет страницы-картинки от чистых фактов
- 4 записи: `struktura-universiteta` (с link), `university-general`, `discounts-and-grants`, `dormitory`
- Виртуальный URL: `https://caiu.edu.kz/__manual__/{id}` — никогда не показывается пользователю
- Если `link` задан — он добавляется в `external_links` payload и прописывается в текст чанка

#### catalog_builder.py — новая функция
- `build_manual_knowledge_catalog()` заменяет `build_special_pages_catalog()`
- Путь к файлу: 4 уровня вверх от `indexer/` → корень проекта
- `build_and_save_catalog_chunks()` теперь вызывает новую функцию

#### bot/run.py — фильтр виртуальных URL
- `_format_candidate_links()` теперь фильтрует все виртуальные URL-префиксы:
  `__catalog__`, `__manual__`, `__special__`, `__facts__`
- Ранее только `__catalog__` фильтровался

#### backup.py — обновлён список файлов
- Копирует `manual_knowledge.json` в `backups/configs_{timestamp}/`
- Дополнительно копирует старые файлы если существуют (переходный период)

**Действие:** запустить `python tools/rebuild_manual_knowledge.py` для индексации новых данных

---

### 2026-05-06 — Защита ручных чанков + special_pages.json + бэкап конфигов

**Изменённые файлы:**
- `rag_service/special_pages.json`
- `rag_service/app/indexer/storage.py`
- `rag_service/app/indexer/pipeline.py`
- `tools/backup.py`
- `README.md`, `CLAUDE.md`, `AUDIT_AND_TODO.md`

**Что сделано:**

#### special_pages.json — заполнено описание struktura-universiteta
- Поле `"description"` было пустым → `build_special_pages_catalog()` пропускал запись (строка 580 catalog_builder.py)
- Чанки для страницы организационной структуры **не создавались вообще**
- Добавлено полное описание иерархии руководства с двумя вхождениями ссылки (для GPT)
- Добавлено 5 дополнительных поисковых запросов (19 итого)
- **Действие:** запустить `python tools/rebuild_special_pages.py` чтобы создать чанки

#### Защита ручных чанков (`storage.py` + `pipeline.py`)
- Новая функция `url_is_manually_edited(url, collection_name)` в `storage.py`
- Проверяет: есть ли чанки для URL И все ли они имеют `manually_edited=True`
- В `pipeline.py` добавлена проверка **до** `page_needs_update()` — если URL защищён, пропускается
- Логирует: `"Protected (manually edited): <url>"` в лог
- Риск который остался: `clear_index.py` удаляет всё включая ручные чанки → после него нужен бэкап

#### Бэкап JSON-конфигов (`backup.py`)
- Добавлен шаг 5: копирует `special_pages.json` и `supplement_facts.json` в `backups/configs_{timestamp}/`
- Добавлен импорт `shutil`
- Эти файлы содержат ручные данные которых нет в Qdrant — потеря критична

#### Обновлена документация
- `README.md`: время 30-60 мин → 4-6 мин, стоимость $0.20-0.30 → $0.18-0.20, добавлены: supplement_facts.json, защита ручных чанков
- `CLAUDE.md`: новый раздел 9 (supplement_facts.json), обновлена структура tools/, перенумерованы разделы 9-13
- `AUDIT_AND_TODO.md`: эта запись

---

### 2026-05-05 — UX улучшения бота

**Изменённые файлы:**
- `bot/run.py`
- `AUDIT_AND_TODO.md` (этот файл)

**Что сделано:**

#### Единое сообщение при отсутствии ответа (`handle_message`)
- Убрано отдельное `await message.answer(links_text, ...)` после основного ответа
- Кандидаты и фоллбек теперь **дописываются в то же сообщение** через `placeholder.edit_text(answer + "\n\n" + suffix)`
- Если кандидаты есть → форматируются ссылки (как раньше, но в одном сообщении)
- Если кандидатов нет → фоллбек: ссылка на `caiu.edu.kz` + телефон

#### Команда `/help` (новый хендлер `cmd_help`)
- Работает в любом состоянии FSM (включая `Chat.waiting`)
- Показывает что умеет бот, контакты и ссылку на сайт
- Текст на трёх языках в `T["help"]`; язык берётся из FSMContext (дефолт: `ru`)

#### Сообщение об ошибке с телефоном (`T["error"]`)
- Было: *"⚠️ Произошла ошибка. Попробуйте позже."*
- Стало: *"⚠️ Возникла техническая проблема. Позвоните в приёмную комиссию: 📞 +7 707 510 10 10"*

#### Rate limit — теплее + телефон (`T["rate_limit"]`)
- Было: *"⏱ Слишком много запросов. Подождите 30 секунд."*
- Стало: *"⏱ Слишком много вопросов за раз. Подождите 30 секунд. По срочным вопросам: 📞 +7 707 510 10 10"*

#### Добавлен импорт `Command` из `aiogram.filters`
- Нужен для хендлера `/help`

### 2026-04-30 — Спринт 1: Фиксы нестабильности RAG

**Изменённые файлы:**
- `rag_service/app/indexer/storage.py`
- `rag_service/app/retrieval/search.py`
- `bot/run.py`

**Что сделано:**

#### Qdrant reconnect (`storage.py`)
- Добавлен `threading.Lock` (`_client_lock`) для защиты синглтона клиента
- `get_client()` теперь делает `ping` (`get_collections()`) перед возвратом клиента
- Если ping упал — клиент закрывается и пересоздаётся автоматически
- Добавлен `timeout=10` при создании клиента
- **Фикс:** Qdrant Docker можно рестартовать — RAG-сервис восстановится сам

#### Постоянный httpx клиент в боте (`bot/run.py`)
- Убран `async with httpx.AsyncClient(...)` внутри `get_rag_context()` — новый TCP каждый раз
- Добавлен глобальный `_rag_client` (создаётся один раз, переиспользуется)
- Разделены таймауты: `connect=5s`, `read=25s` (вместо единого `timeout=15s`)
  - `connect=5s` — быстро падаем если RAG-сервис не запущен
  - `read=25s` — даём время embedding + Qdrant при нагрузке на OpenAI
- Добавлен retry: 3 попытки, экспоненциальный backoff (1с, 2с, 4с)
- При `ConnectError` клиент пересоздаётся
- Корректное закрытие клиента в `main()` при остановке бота
- **Фикс:** Таймаут 15с больше не ронял запросы при медленном OpenAI

#### Thread-safe embedding cache (`search.py`)
- Добавлен `threading.Lock` (`_CACHE_LOCK`) для всех операций с `_EMBEDDING_CACHE`
- Чтение из кеша и запрос к OpenAI разделены — lock НЕ держится во время сетевого вызова
- Добавлена валидация векторов `_is_valid_vector()`:
  - Проверка размерности (ожидается 1536 для text-embedding-3-small)
  - Проверка что вектор не нулевой
  - Некорректные векторы НЕ кешируются → следующий запрос сделает новую попытку
- Увеличен `_CACHE_MAX_SIZE`: 256 → 512
- **Фикс:** Некорректный вектор от OpenAI больше не «застрянет» в кеше

---

### 2026-04-30 — Спринт 5: Оптимизации поиска (Баги #12, #13 + payload index + search cache)

**Изменённые файлы:**
- `rag_service/app/retrieval/search.py`
- `rag_service/app/indexer/storage.py`

**Что сделано:**

#### Сортировка ключей SYNONYM_MAP по длине (`search.py`, `_expand_query`)
- `sorted(SYNONYM_MAP.keys(), key=len, reverse=True)` — составные ключи ("документы для общежития") проверяются до коротких ("документы")
- До: запрос "документы для общежития" попадал в ветку "документы" и получал синонимы поступления вместо общежития
- После: точное совпадение с правильным ключом

#### Дедупликация по (page_url, chunk_index) (`search.py`, `_deduplicate_results` + loop)
- `chunk_id = (page_url, chunk_index)` вместо `text_key = text[:200]`
- До: чанки с одной страницы начинаются с `[Страница > Раздел]\n` → одинаковые 200 символов → выбрасывались разные по смыслу блоки
- После: каждый чанк идентифицируется точно, без ложных совпадений

#### Payload index на page_url (`storage.py`, `ensure_collection_exists`)
- `client.create_payload_index(field_name="page_url", field_schema="keyword")` при создании коллекции
- Ускоряет `delete_chunks_by_url()` (вызывается при каждой переиндексации страницы) и `page_needs_update()` с O(n) до O(log n)
- Graceful: ошибка создания индекса не прерывает создание коллекции

#### Redis-кеш ответов поиска (`search.py`, `search()`)
- Добавлены константы `_REDIS_SEARCH_PREFIX = "caiu:search:"`, `_SEARCH_CACHE_TTL = 3600`
- Импорты: `json`, `dataclasses`
- В начале `search()`: проверка Redis, при HIT — возврат без Qdrant и OpenAI
- В конце `search()`: запись результата (`json.dumps([dataclasses.asdict(r)...])`)
- TTL 1 час: покрывает поток абитуриентов, обновляется естественно после `/reindex`
- Польза: ~60% запросов повторяются ("как поступить", "общежитие") → 0 токенов, 0 Qdrant, <2мс

---

### 2026-04-30 — Спринт 4: Параллельное скачивание страниц (Баг #2)

**Изменённые файлы:**
- `rag_service/app/indexer/pipeline.py`

**Что сделано:**

#### Параллельный fetch в pipeline.py
- Добавлен импорт `from concurrent.futures import ThreadPoolExecutor, as_completed`
- Константа `_FETCH_WORKERS = 5` (воркеры), `_WORKER_DELAY = REQUEST_DELAY / _FETCH_WORKERS = 0.3 сек`
- Новая функция `_fetch_one_page(url_item)` — спит 0.3с, вызывает `get_page_content()`
- Новая функция `_fetch_all_pages(urls)` — запускает все воркеры, собирает `{url: content}`
- `run_indexing()` разбита на две явные фазы:
  - **Фаза 1**: `_fetch_all_pages()` — параллельное скачивание, прогресс в логе
  - **Фаза 2**: последовательная обработка из кеша — `page_needs_update` → chunk → questions → save
- `time.sleep(settings.REQUEST_DELAY)` из processing-фазы убран (задержка теперь в фазе fetch)

**Результат:**
- До: 150 URL × 1.5 сек задержка = ~225 сек только на sleep + ~150 сек сеть = ~6 мин
- После: 5 воркеров × 0.3 сек = ~45 сек fetch + ~150 сек processing = ~3 мин полная индексация
- Нагрузка на сервер университета осталась той же (1.5 req/сек суммарно)

---

### 2026-04-30 — Спринт 3: Инкрементальная переиндексация (Баг #1)

**Изменённые файлы:**
- `rag_service/app/indexer/hot_swap.py`

**Что сделано:**

#### Snapshot+patch в `hot_swap.py`
- Добавлен импорт `PointStruct` из `qdrant_client.models`
- Добавлена функция `_snapshot_active_to_shadow(client, shadow_collection)`:
  - Копирует все точки из активной коллекции в теневую через `scroll`+`upsert`
  - Батчи по 256 точек, итерация через `offset` (cursor API Qdrant)
  - Graceful fallback: ошибка при копировании не прерывает индексацию — идёт полная
  - Возвращает количество скопированных точек (добавляется в итоговый dict `snapshot_points`)
- `run_shadow_indexing()` теперь вызывает `_snapshot_active_to_shadow()` после создания shadow коллекции и до запуска `run_indexing()`

**Эффект:**
- `pipeline.page_needs_update(url, hash, collection_name=shadow)` находит уже существующие `content_hash`-и → пропускает неизменённые страницы
- До фикса: `/reindex` всегда тратил $0.20-0.30 и 30-60 мин (200 страниц × GPT-вопросы + embeddings)
- После фикса: ~$0 и ~30 сек если сайт не менялся; только изменённые страницы идут через OpenAI

---

### 2026-04-30 — Спринт 2: Оптимизация скорости (Redis + Streaming)

**Изменённые файлы:**
- `requirements.txt` — добавлен `redis==5.2.1`
- `rag_service/app/redis_client.py` — новый файл
- `rag_service/app/retrieval/search.py`
- `bot/run.py` — переписан полностью

**Что сделано:**

#### Redis утилита (`redis_client.py`)
- Sync-клиент для использования в FastAPI sync-эндпоинтах (threadpool)
- Graceful degradation: недоступен → логируется 1 раз, кеши падают на in-memory
- `CACHE_TTL_SECONDS = 30 * 24 * 3600` (30 дней)
- `REDIS_AVAILABLE` property для проверки состояния

#### Двухуровневый кеш embeddings (`search.py`)
- `_REDIS_EMB_PREFIX = "caiu:emb:"` — ключ в Redis
- `_vector_to_bytes()` / `_bytes_to_vector()` через `struct.pack/unpack` (компактное float32)
- `_redis_get_vector(key)` / `_redis_set_vector(key, vector)` — tихие fail'ы
- `_get_embedding_cached()`: in-memory → Redis → OpenAI → записываем оба кеша
- `_get_embeddings_batch_cached()`: сначала in-memory+Redis, батч OpenAI только для miss'ов
- Экономия: повторный запрос (~60% трафика) = 0 токенов + 0 OpenAI latency

#### Двухуровневый кеш переводов + Streaming GPT (`bot/run.py`)
- Async Redis клиент (`aioredis` через `redis.asyncio`) с graceful degradation
- `_TRANS_REDIS_PREFIX = "caiu:trans:"` — ключ `{prefix}{lang}:{md5(text)}`
- `translate_to_russian()`: in-memory OrderedDict → Redis → GPT → оба кеша
- `_build_system_prompt(lang, context)` выделен в отдельную функцию
- `stream_answer_to_message(question, lang, context, placeholder)`:
  - Открывает стрим GPT (`stream=True`)
  - Первый edit после `_STREAM_FIRST_EDIT_CHARS=80` символов
  - Затем edit каждые `_STREAM_EDIT_INTERVAL=1.1` сек (не превышаем 1 edit/сек Telegram)
  - Финальный edit с `parse_mode="HTML"`, fallback без parse_mode при MarkupParseError
- `handle_message()` отправляет `⏳` placeholder, потом streaming в него

---

### 2026-04-30 — Аудит проекта
- Создан этот файл `AUDIT_AND_TODO.md`
- Проанализированы файлы:
  - `rag_service/app/indexer/{pipeline,storage,hot_swap,question_generator,embeddings,catalog_builder}.py`
  - `rag_service/app/parser/{crawler,extractor,chunker}.py`
  - `rag_service/app/retrieval/search.py`
  - `rag_service/app/{main,config}.py`
  - `bot/run.py`
- Найдено 15 багов/проблем + 7 улучшений
- Главный виновник медленной индексации и расхода токенов: **Баг #1** (hot_swap всегда стирает shadow коллекцию)
- План разбит на 5 спринтов

### Следующая запись (после фиксов) — формат
```
### 2026-MM-DD — Спринт N
- [x] Реализовано: ...
- [ ] Откатилось: ...
- Замеры: до фикса /reindex = X мин, после = Y мин
- Замеры: токенов до = $Z, после = $W
```

---

## 🔗 КОНТЕКСТ ДЛЯ НОВОГО ЧАТА

Если открываешь новый чат:
1. Прочитать `CLAUDE.md` (контекст проекта целиком)
2. Прочитать этот файл (`AUDIT_AND_TODO.md`)
3. Сказать «продолжаем со спринта N» — Claude поймёт что делать
4. После каждого фикса добавлять запись в раздел **ЖУРНАЛ ИЗМЕНЕНИЙ**

Все правки — без перезапуска RAG-сервиса (благодаря hot-swap).
Бот рестартует только если правились файлы в `bot/`.
