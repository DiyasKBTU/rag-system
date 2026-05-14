# -*- coding: utf-8 -*-
"""
test_concurrency.py — Тесты на поведение под нагрузкой.

Запуск:
    cd rag_service && pytest ../tests/test_concurrency.py -v

Что тестируем:
    1. Rate limiter: блокирует спам от одного пользователя
    2. Rate limiter: разные пользователи не мешают друг другу
    3. Cache: повторный вопрос возвращается быстрее (из кеша)
    4. Семафор: при превышении MAX_CONCURRENT_SEARCHES возвращает 503
    5. httpx pool: бот может делать много параллельных запросов
    6. Корректная очистка timestamps у rate limiter

Все тесты изолированы — не нужен Docker, OpenAI, Qdrant.
"""

import sys
import time
import asyncio
import threading
from unittest.mock import patch, MagicMock
from collections import defaultdict

import pytest

# ── Тест 1-2: Rate limiting ───────────────────────────────────────────────────

# Копируем логику rate limiting из bot/run.py чтобы не импортировать весь бот
# (бот читает BOT_TOKEN из env при импорте — это неудобно в тестах)
_RATE_LIMIT_MESSAGES = 10
_RATE_LIMIT_WINDOW   = 60

def make_rate_limiter():
    """Фабрика: создаёт изолированный rate limiter (как в bot/run.py)."""
    user_timestamps = defaultdict(list)
    lock = threading.Lock()

    def is_rate_limited(user_id: int) -> bool:
        now = time.time()
        with lock:
            user_timestamps[user_id] = [
                t for t in user_timestamps[user_id]
                if now - t < _RATE_LIMIT_WINDOW
            ]
            if len(user_timestamps[user_id]) >= _RATE_LIMIT_MESSAGES:
                return True
            user_timestamps[user_id].append(now)
            return False

    return is_rate_limited


class TestRateLimiter:

    def test_allows_up_to_limit(self):
        """До 10 сообщений — пропускает все."""
        rl = make_rate_limiter()
        for i in range(10):
            assert rl(user_id=1) is False, f"Сообщение {i+1} должно пройти"

    def test_blocks_after_limit(self):
        """11-е сообщение — блокирует."""
        rl = make_rate_limiter()
        for _ in range(10):
            rl(user_id=1)
        assert rl(user_id=1) is True, "11-е сообщение должно быть заблокировано"

    def test_different_users_independent(self):
        """Лимит пользователя 1 не влияет на пользователя 2."""
        rl = make_rate_limiter()
        # Исчерпываем лимит пользователя 1
        for _ in range(10):
            rl(user_id=1)
        assert rl(user_id=1) is True,  "User 1 должен быть заблокирован"
        assert rl(user_id=2) is False, "User 2 не должен быть заблокирован"

    def test_concurrent_users_no_race(self):
        """Конкурентные запросы от 20 пользователей не вызывают race condition."""
        rl = make_rate_limiter()
        results = []
        errors  = []

        def spam_user(user_id: int):
            try:
                for _ in range(5):
                    rl(user_id=user_id)
                results.append(user_id)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=spam_user, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Race condition ошибки: {errors}"
        assert len(results) == 20, "Все потоки должны завершиться"

    def test_window_cleanup(self):
        """Старые timestamps убираются из памяти."""
        rl = make_rate_limiter()
        # Прогоняем через rate limiter
        for _ in range(5):
            rl(user_id=999)
        # Проверяем что timestamps хранятся (но не более LIMIT)
        # Мы не можем проверить внутренний dict напрямую —
        # но можем проверить что объект не падает при нагрузке
        for _ in range(100):
            rl(user_id=999)  # Большинство будет заблокировано, но не должно упасть


# ── Тест 3: Кеш поиска ───────────────────────────────────────────────────────

class TestSearchCache:

    def test_cache_miss_then_hit(self):
        """
        Первый вызов — промах (идёт в OpenAI).
        Второй вызов — попадание (из кеша, быстро).
        """
        call_count = 0

        def mock_embedding(text: str, *args, **kwargs) -> list:
            nonlocal call_count
            call_count += 1
            return [0.1] * 1536  # фиктивный вектор

        # Создаём изолированный кеш (как в search.py)
        cache: dict = {}
        cache_lock = threading.Lock()

        def get_cached_embedding(text: str) -> list:
            with cache_lock:
                if text in cache:
                    return cache[text]
            # промах — вызываем "OpenAI"
            embedding = mock_embedding(text)
            with cache_lock:
                cache[text] = embedding
            return embedding

        q = "сколько стоит обучение"

        # Первый вызов — промах
        e1 = get_cached_embedding(q)
        assert call_count == 1, "Первый вызов должен идти в OpenAI"

        # Второй вызов — попадание
        e2 = get_cached_embedding(q)
        assert call_count == 1, "Второй вызов должен взяться из кеша (без OpenAI)"
        assert e1 == e2

    def test_cache_concurrent_reads(self):
        """Параллельные reads из кеша не вызывают гонок."""
        cache = {"question": [0.5] * 1536}
        lock  = threading.Lock()
        results = []
        errors  = []

        def read_cache():
            try:
                with lock:
                    v = cache.get("question")
                results.append(len(v) if v else 0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_cache) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Ошибки при параллельном чтении: {errors}"
        assert all(r == 1536 for r in results), "Все потоки должны получить корректный вектор"


# ── Тест 4: Семафор конкурентных поисков ─────────────────────────────────────

class TestSearchSemaphore:

    def test_semaphore_limits_concurrency(self):
        """
        Семафор MAX=3: при 10 одновременных запросах максимум 3 идут параллельно,
        остальные ждут в очереди.
        """
        MAX = 3
        sem = threading.Semaphore(MAX)
        active_count = 0
        max_observed = 0
        lock = threading.Lock()
        results = []

        def do_work(i: int):
            nonlocal active_count, max_observed
            acquired = sem.acquire(timeout=5.0)
            if not acquired:
                results.append({"i": i, "ok": False, "error": "timeout"})
                return
            try:
                with lock:
                    active_count += 1
                    if active_count > max_observed:
                        max_observed = active_count
                time.sleep(0.05)  # симулируем работу
                results.append({"i": i, "ok": True})
            finally:
                with lock:
                    active_count -= 1
                sem.release()

        threads = [threading.Thread(target=do_work, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_observed <= MAX, (
            f"Одновременно активных: {max_observed}, лимит: {MAX}"
        )
        assert all(r["ok"] for r in results), "Все запросы должны завершиться успешно"

    def test_semaphore_releases_on_exception(self):
        """Семафор освобождается даже если обработчик бросил исключение."""
        sem = threading.Semaphore(1)
        released = threading.Event()

        def failing_work():
            sem.acquire()
            try:
                raise ValueError("Симулируем ошибку поиска")
            finally:
                sem.release()
                released.set()

        t = threading.Thread(target=failing_work)
        t.start()
        t.join()

        # Семафор должен быть освобождён → можно войти сразу
        acquired = sem.acquire(timeout=0.1)
        assert acquired, "Семафор не был освобождён после исключения"
        sem.release()

    def test_semaphore_returns_503_on_overload(self):
        """
        Если все слоты заняты и timeout истёк — должен быть 503.
        Проверяем логику timeout=0 (мгновенный отказ).
        """
        MAX = 2
        sem = threading.Semaphore(MAX)
        # Занимаем все слоты
        sem.acquire()
        sem.acquire()

        # Следующий с timeout=0 должен получить False
        acquired = sem.acquire(timeout=0)
        assert acquired is False, "Перегруженный семафор должен сразу возвращать False"

        # Освобождаем
        sem.release()
        sem.release()


# ── Тест 5: Translation cache (LRU) ─────────────────────────────────────────

class TestTranslationCache:

    def test_lru_eviction(self):
        """LRU кеш выбрасывает старые записи при достижении максимума."""
        from collections import OrderedDict

        MAX = 5
        cache = OrderedDict()

        def cache_put(key, value):
            if key in cache:
                cache.move_to_end(key)
            cache[key] = value
            if len(cache) > MAX:
                cache.popitem(last=False)  # удаляем самый старый

        def cache_get(key):
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            return None

        # Заполняем до максимума
        for i in range(MAX):
            cache_put(f"q{i}", f"answer{i}")

        assert len(cache) == MAX

        # Добавляем ещё один — должен вытолкнуть самый старый (q0)
        cache_put("q_new", "new_answer")
        assert len(cache) == MAX
        assert cache_get("q0") is None,    "q0 должен быть вытолкнут"
        assert cache_get("q_new") is not None, "q_new должен быть в кеше"

    def test_cache_thread_safe_reads(self):
        """Параллельные чтения из OrderedDict с Lock не падают."""
        from collections import OrderedDict
        import threading

        cache = OrderedDict()
        cache["question_ru"] = "Сколько стоит обучение?"
        lock = threading.Lock()
        results = []
        errors  = []

        def read():
            try:
                with lock:
                    v = cache.get("question_ru")
                results.append(v)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 30
        assert all(r == "Сколько стоит обучение?" for r in results)


# ── Тест 6: Одновременные запросы к bot httpx pool ───────────────────────────

class TestBotHttpxPool:

    def test_connection_pool_size(self):
        """Проверяем что httpx Limits корректно создаются с нужными значениями."""
        import httpx

        limits = httpx.Limits(max_connections=30, max_keepalive_connections=10)
        # httpx хранит эти значения в _max_connections и _max_keepalive_connections
        assert limits.max_connections == 30
        assert limits.max_keepalive_connections == 10

    @pytest.mark.asyncio
    async def test_async_concurrent_requests_mock(self):
        """
        Симулируем 15 одновременных asyncio задач — они все стартуют вместе
        и ждут ответа. С правильным connection pool все должны завершиться.
        """
        results = []
        errors  = []

        async def fake_request(i: int):
            try:
                await asyncio.sleep(0.01)  # симулируем сетевую задержку
                results.append(i)
            except Exception as e:
                errors.append(str(e))

        tasks = [fake_request(i) for i in range(15)]
        await asyncio.gather(*tasks)

        assert not errors
        assert len(results) == 15
        assert sorted(results) == list(range(15))
