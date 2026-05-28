# Анализ уязвимостей и багов RAG-системы ЦАИУ

Дата анализа: 2026-05-21  
Охват: `bot/run.py`, `rag_service/app/main.py`, `retrieval/search.py`, `indexer/hot_swap.py`, `indexer/storage.py`

---

## 🔴 КРИТИЧЕСКИЕ — могут вызвать реальные сбои в продакшне

---

### 1. Поисковый Redis-кеш не инвалидируется после hot_swap

**Файл:** `rag_service/app/retrieval/search.py`, `indexer/hot_swap.py`

**Проблема.** После успешной переиндексации (`hot_swap`) алиас Qdrant переключается на новую коллекцию, но ключи `caiu:search:*` в Redis не сбрасываются. TTL = 1 час. Все вопросы, заданные в течение последнего часа до reindex, продолжат отдавать **старые результаты** из кеша ещё до 60 минут после переключения.

**Сценарий:** Исправили ошибочную информацию о стоимости обучения, переиндексировали — пользователи ещё час получают старые неверные данные.

**Где исправить:** `hot_swap.py`, функция `run_shadow_indexing()`, после строки `_atomic_swap(shadow)` / `_first_migration(shadow)`:

```python
# После успешного свопа — сбрасываем search cache
try:
    from app.redis_client import get_redis
    r = get_redis()
    if r:
        keys = list(r.scan_iter("caiu:search:*"))
        if keys:
            r.delete(*keys)
            logger.info(f"[HotSwap] Invalidated {len(keys)} search cache entries")
except Exception as e:
    logger.warning(f"[HotSwap] Search cache flush failed (non-critical): {e}")
```

---

### 2. Нет ограничения длины входящего сообщения

**Файл:** `bot/run.py`, `handle_message()`

**Проблема.** Telegram позволяет отправить до 4096 символов. Сообщение без проверки уходит:
1. В `translate_to_russian()` → GPT-запрос с очень длинным текстом
2. В `get_rag_context()` → embedding очень длинного текста (API OpenAI режет по токенам сам, но стоит $)
3. В промпт GPT как вопрос

Злоумышленник может спамить длинными сообщениями, перегружая OpenAI-квоту и увеличивая стоимость.

**Исправление:** Добавить проверку в `handle_message` и `handle_faq_button` (хотя FAQ-кнопки ограничены):

```python
MAX_QUESTION_LENGTH = 1000

@router.message(Chat.waiting)
async def handle_message(message: Message, state: FSMContext, bot: Bot) -> None:
    text = (message.text or "").strip()
    if len(text) > MAX_QUESTION_LENGTH:
        await message.answer(T["error"][lang])  # или отдельное сообщение
        return
    ...
```

---

### 3. Thundering herd при перегрузке RAG-сервиса

**Файл:** `bot/run.py`, `get_rag_context()`

**Проблема.** RAG-сервис возвращает 503 при переполнении семафора (>20 одновременных). Бот делает 3 попытки с backoff 1→2→4 сек. При 30 одновременных пользователях, все получивших 503, RAG-сервис получает **90 повторных запросов** вместо 30. Это усугубляет перегрузку и создаёт автоматический DDoS на свой же сервис.

**Исправление:** Добавить jitter к backoff, и при 503 не ретраить немедленно:

```python
import random

if attempt < _RAG_RETRY_ATTEMPTS:
    delay = _RAG_RETRY_BASE_DELAY * (2 ** (attempt - 1))
    jitter = random.uniform(0, delay * 0.3)   # ±30% jitter
    # При 503 — ждём дольше (сервис явно перегружен)
    if 'resp' in dir() and resp.status_code == 503:
        delay = delay * 2
    await asyncio.sleep(delay + jitter)
```

---

### 4. `get_client()` сериализует все потоки на время пинга Qdrant

**Файл:** `rag_service/app/indexer/storage.py`, `get_client()`

**Проблема.** `_client_lock` — это `threading.Lock`. Когда кеш пинга протухает (каждые 30 сек), и первый поток входит в `with _client_lock:` чтобы сделать `get_collections()` (timeout=10 сек), **все остальные потоки** из ThreadPoolExecutor стоят в очереди на этот lock. На пиковой нагрузке 20 одновременных поисков — 19 из 20 ждут до 10 секунд просто чтобы проверить живость клиента.

**Исправление:** Делать ping оптимистично (снаружи lock'а) или использовать флаг «ping in progress»:

```python
def get_client() -> QdrantClient:
    global _client, _last_ping_ok_at
    now = time.monotonic()
    
    # Быстрая проверка БЕЗ лока: если кеш свежий — возвращаем сразу
    if _client is not None and (now - _last_ping_ok_at) < _PING_CACHE_TTL:
        return _client
    
    with _client_lock:
        # double-check под локом
        if _client is not None and (now - _last_ping_ok_at) < _PING_CACHE_TTL:
            return _client
        
        if _client is None:
            _client = QdrantClient(...)
            _last_ping_ok_at = now
            return _client
        
        try:
            _client.get_collections()
            _last_ping_ok_at = now
            return _client
        except Exception:
            # переподключение...
```

---

## 🟡 СРЕДНИЕ — влияют на качество работы под нагрузкой

---

### 5. Стальная локальная переменная `fire_indexing` в `_scheduler_loop`

**Файл:** `rag_service/app/main.py`, `_scheduler_loop()`

**Проблема.** В Python переменные, объявленные внутри `while True`, персистентны между итерациями. После первого срабатывания `fire_indexing = True` переменная существует в локальном пространстве. В редком сценарии (исключение внутри `if should_try_fire:` до присвоения `fire_indexing`) в следующей итерации `'fire_indexing' in locals()` вернёт `True` со значением предыдущей итерации. Конкретный случай — маловероятен из-за защиты внешним `try/except`, но паттерн `in locals()` хрупкий.

**Исправление:**
```python
async def _scheduler_loop() -> None:
    while True:
        fire_indexing = False   # ← инициализируем в начале каждой итерации
        try:
            ...
```

---

### 6. `_EMBEDDING_CACHE` использует FIFO-вытеснение вместо LRU

**Файл:** `rag_service/app/retrieval/search.py`, `_cache_put()`

**Проблема.** `_EMBEDDING_CACHE` — обычный `dict` (не `OrderedDict`). Вытеснение: `_EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))` удаляет **первый добавленный** ключ, а не самый давно используемый. Популярные запросы (напр. «специальности») могут быть вытеснены, и каждый раз заново считаются через OpenAI.

**Исправление:** Использовать `collections.OrderedDict` с LRU-логикой, аналогично `_TRANSLATION_CACHE` в боте:

```python
from collections import OrderedDict
_EMBEDDING_CACHE: OrderedDict = OrderedDict()

def _cache_put(key: str, vector: List[float]) -> None:
    with _CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
            return
        _EMBEDDING_CACHE[key] = vector
        _EMBEDDING_CACHE.move_to_end(key)
        while len(_EMBEDDING_CACHE) > _CACHE_MAX_SIZE:
            _EMBEDDING_CACHE.popitem(last=False)
```

---

### 7. Некорректное сравнение `last_indexed_at` из-за смешения timezone

**Файл:** `rag_service/app/indexer/storage.py`, `get_last_indexed_at()`

**Проблема.** `set_last_indexed_at` сохраняет время через `datetime.now(timezone.utc).isoformat()` — это UTC-aware строка, например `"2025-06-15T05:00:00+00:00"`. При чтении:

```python
dt = datetime.fromisoformat(ts)
return dt.replace(tzinfo=None)  # strip timezone, НО значение остаётся UTC
```

`replace(tzinfo=None)` **не конвертирует** время — просто снимает метку зоны. Если сервер в UTC+5, а `lastmod` в sitemap указан как местное время (без зоны), то `pipeline.py` сравнивает "UTC-значение без метки" против "местного времени без метки" — расхождение **5 часов**. Страницы, изменённые после полуночи по UTC, но до 5 утра по серверному времени, будут пропущены.

**Исправление:** Сохранять в UTC и при чтении явно обрабатывать:
```python
def get_last_indexed_at() -> Optional[datetime]:
    ...
    dt = datetime.fromisoformat(ts)
    # Если aware — конвертируем в UTC и снимаем tzinfo
    if dt.tzinfo is not None:
        from datetime import timezone
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt
```

---

### 8. Потеря recurring-расписания при краше процесса

**Файл:** `rag_service/app/main.py`, `_scheduler_loop()`

**Проблема.** При `recurring=True` scheduler делает два отдельных `_write_schedule()`:
1. Пишет `fired=True` для текущего
2. Пишет новое расписание на следующий день

Если процесс крашится между этими двумя записями — следующий запуск будет потерян. При ежедневной переиндексации это значит пропуск без алерта.

**Исправление:** Записывать следующий запуск **перед** тем как помечать текущий как fired (или в одной атомарной записи):
```python
if is_recurring:
    next_schedule = {..., "fired": False}
    _write_schedule(next_schedule)   # сначала следующий
schedule["fired"] = True
_write_schedule(schedule)             # потом помечаем текущий
```

---

### 9. `_is_list_query` — ложные срабатывания на частичные совпадения

**Файл:** `rag_service/app/retrieval/search.py`

**Проблема.** Подстроковые ключевые слова слишком широкие:
- `"программ"` матчит «**программ**ирование» → каталожный режим для вопроса о ИТ-специальности  
- `"направлени"` матчит «в каком **направлени**и»  
- `"образовательн"` матчит «**образовательн**ая система Казахстана»

При ложном срабатывании отправляется **в 3 раза больше чанков** в GPT (15 вместо 5), стоимость одного запроса растёт.

**Исправление:** Добавить пограничные условия — требовать наличие слова-триггера («какие», «перечисли», «список», «все») вместе с ключевым словом:

```python
LIST_QUERY_TRIGGER_WORDS = ["какие", "перечисли", "список", "все ", "тізімі", "list of"]

def _is_list_query(question: str) -> bool:
    q = question.lower()
    has_list_keyword = any(kw in q for kw in LIST_QUERY_KEYWORDS)
    has_trigger = any(t in q for t in LIST_QUERY_TRIGGER_WORDS)
    # Только если есть и ключевое слово, и триггер перечисления
    return has_list_keyword and has_trigger
```

---

## 🟢 НИЗКИЕ — качество кода и небольшие улучшения

---

### 10. `/index/status` показывает `in_progress=False` на воркере B при workers>1

**Файл:** `hot_swap.py`, модульная переменная `_indexing_active`

При `workers=2` в `run_api.py`: воркер A запустил индексацию (`_indexing_active=True`), воркер B при запросе `/index/status` отдаёт `in_progress=False`. Файловый lock предотвращает двойную индексацию, но мониторинг покажет некорректный статус. При `workers=1` (текущий дефолт) — не проблема.

**Исправление для будущего масштабирования:** Хранить `in_progress` в `hot_swap_state.json`, аналогично `active_collection`.

---

### 11. Двойной `html.escape` в fallback-ветке

**Файл:** `bot/run.py`, `handle_message()` (строка ~1240)

```python
# Fallback без HTML
try:
    await placeholder.edit_text(
        answer + "\n\n" + suffix, disable_web_page_preview=True  # suffix содержит HTML-теги как текст
    )
```

Если основной `edit_text` с `parse_mode="HTML"` упал, fallback отправляет `answer + suffix` без parse_mode. `suffix` содержит сырые `<a href=...>` теги — они отобразятся как текст, что лучше чем ничего, но некрасиво. Это приемлемо как fallback.

---

### 12. Кеш переводов `_TRANSLATION_CACHE` не защищён локом

**Файл:** `bot/run.py`, `_cache_get()` / `_cache_set()`

Функции работают с `OrderedDict` без `asyncio.Lock`. В текущей asyncio-архитектуре (один поток) это безопасно, потому что нет `await` внутри этих функций — вытеснения не будет. Но если в будущем эти функции вызовут из `asyncio.to_thread()` или из background thread — будет race condition.

---

## Итоговая таблица приоритетов

| # | Уязвимость | Критичность | Файл | Сложность исправления |
|---|---|---|---|---|
| 1 | Search cache не инвалидируется после hot_swap | 🔴 Критическая | `hot_swap.py` | Малая — 10 строк |
| 2 | Нет лимита длины сообщения | 🔴 Критическая | `bot/run.py` | Малая — 5 строк |
| 3 | Thundering herd при 503 от RAG | 🔴 Высокая | `bot/run.py` | Малая — jitter |
| 4 | `get_client()` блокирует все потоки на пинге | 🔴 Высокая | `storage.py` | Средняя |
| 5 | Стальная переменная `fire_indexing` | 🟡 Средняя | `main.py` | Минимальная — 1 строка |
| 6 | FIFO вместо LRU в embedding cache | 🟡 Средняя | `search.py` | Малая |
| 7 | Timezone-баг в `last_indexed_at` | 🟡 Средняя | `storage.py` | Малая |
| 8 | Потеря recurring-расписания при краше | 🟡 Средняя | `main.py` | Малая |
| 9 | Ложные срабатывания `_is_list_query` | 🟡 Средняя | `search.py` | Малая |
| 10 | `in_progress` неверен при workers>1 | 🟢 Низкая | `hot_swap.py` | Средняя |
| 11 | Двойной escape в fallback | 🟢 Низкая | `bot/run.py` | Нет (OK как fallback) |
| 12 | Translation cache без лока | 🟢 Низкая | `bot/run.py` | Малая (превентивная) |

## Рекомендуемый порядок исправлений

**Немедленно (до следующего reindex):**
- [#1] Сбрасывать search cache после hot_swap
- [#2] Добавить `MAX_QUESTION_LENGTH = 1000`

**При следующем деплое:**
- [#3] Jitter к retry в `get_rag_context`
- [#4] Оптимизировать `get_client()` ping-lock
- [#5] `fire_indexing = False` в начале итерации
- [#6] OrderedDict + LRU для `_EMBEDDING_CACHE`
- [#7] Исправить timezone в `get_last_indexed_at`
- [#8] Swap порядок записи для recurring schedule
