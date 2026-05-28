# -*- coding: utf-8 -*-
"""
main.py - FastAPI сервер RAG-сервиса.

Эндпоинты:
  POST /search  — семантический поиск по базе знаний (вызывает бот)
  POST /index   — запуск переиндексации сайта
  GET  /health  — проверка что сервер работает
  GET  /stats   — сколько записей в Qdrant

Как работает:
  1. Пользователь пишет вопрос в Telegram
  2. bot/run.py делает POST /search с вопросом
  3. RAG-сервис возвращает релевантные фрагменты текста
  4. bot/run.py передаёт вопрос + фрагменты в ChatGPT
  5. ChatGPT формирует ответ → бот отправляет пользователю

Безопасность:
  Все эндпоинты кроме /health требуют заголовок:
  X-API-Key: your_secret_key_from_env
"""

import asyncio
import json
import logging
import secrets
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

import portalocker

# ── Ограничитель конкурентных поисков ────────────────────────────────────────
# Каждый поиск делает вызов к OpenAI Embeddings API.
# При 50+ одновременных запросах OpenAI может вернуть 429 (rate limit).
# Семафор ограничивает: максимум MAX_CONCURRENT_SEARCHES запросов к /search
# одновременно могут идти в OpenAI; остальные ждут в очереди (не падают).
#
# Значение 20 выбрано исходя из:
#   - OpenAI RPM лимит для text-embedding-3-small: обычно 3000 RPM = 50/сек
#   - Средняя задержка поиска: ~1-2 сек → 20 * (1/1.5) ≈ 13 запросов/сек
#   - Это с большим запасом для пикового потока абитуриентов
#
# ВАЖНО: asyncio.Semaphore (а не threading.Semaphore) — endpoint async,
# ожидание слота происходит в event loop и НЕ занимает поток threadpool.
# Это означает что при 100 одновременных запросах:
#   - 20 реально работают (зашли в семафор → search() в to_thread)
#   - 80 висят в семафоре как лёгкие корутины (не съедают потоки)
#   - /health и другие endpoints отзывчивы (event loop свободен).
MAX_CONCURRENT_SEARCHES = 20
_search_semaphore: "asyncio.Semaphore | None" = None  # создаётся лениво в lifespan

# Размер default-executor для asyncio.to_thread.
# Зачем не дефолт:
#   asyncio по умолчанию: min(32, os.cpu_count() + 4) → на 2-ядерной VPS = 6 потоков.
#   В search.search() мы делаем 2 to_thread (search + get_candidate_urls).
#   При семафоре в 20 слотов и 6 потоках threadpool становится бутылочным горлом:
#   часть запросов ждёт поток вместо того чтобы реально работать.
# 32 потока спокойно укладываются в RAM (~8 МБ stack каждый), позволяя
# реализовать заявленную параллельность семафора без неявных ограничений.
_THREADPOOL_SIZE = 32

from app.logging_setup import setup_logging
setup_logging("api")
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Prometheus метрики ────────────────────────────────────────────────────────
# prometheus_client — стандартная библиотека для экспорта метрик в Prometheus.
# Если она не установлена — метрики недоступны, но сервис работает нормально.
#
# Доступные метрики:
#   caiu_search_total          — счётчик поисков (labels: status=ok/empty/error)
#   caiu_search_latency_seconds — гистограмма латентности поиска
#   caiu_semaphore_waiting     — gauge текущего числа запросов в очереди на семафор
#   caiu_indexing_runs_total   — счётчик запусков индексации (labels: trigger=api/scheduler)
#   caiu_indexing_errors_total — счётчик ошибок индексации
#
# Установка: pip install prometheus_client
# Метрики доступны на GET /metrics (без авторизации — только для localhost).
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True

    _search_total = Counter(
        "caiu_search_total",
        "Total number of /search requests",
        ["status"],  # ok | empty | error
    )
    _search_latency = Histogram(
        "caiu_search_latency_seconds",
        "Search request latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    _semaphore_waiting = Gauge(
        "caiu_semaphore_waiting",
        "Number of /search requests waiting for semaphore slot",
    )
    _indexing_runs = Counter(
        "caiu_indexing_runs_total",
        "Total number of hot-swap indexing runs started",
        ["trigger"],  # api | scheduler
    )
    _indexing_errors = Counter(
        "caiu_indexing_errors_total",
        "Total number of indexing errors",
    )
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger_bootstrap = __import__("logging").getLogger(__name__)
    logger_bootstrap.warning(
        "[Metrics] prometheus_client not installed — /metrics endpoint unavailable. "
        "Install: pip install prometheus_client"
    )

from app.config import settings
from app.retrieval.search import search, format_results_as_context, get_candidate_urls
from app.indexer.storage import get_collection_stats, ensure_collection_exists


logger = logging.getLogger(__name__)

# ── Файл расписания горячей замены ───────────────────────────────────────────
# Хранит: {"scheduled_time": "03:00", "scheduled_by": "admin_id", ...}
_SCHEDULE_FILE = Path(__file__).resolve().parent.parent / "schedule_state.json"


def _read_schedule() -> Optional[dict]:
    """Читает расписание из файла. Возвращает None если файл не существует."""
    try:
        if _SCHEDULE_FILE.exists():
            return json.loads(_SCHEDULE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _write_schedule(data: Optional[dict]) -> None:
    """Записывает расписание. data=None удаляет файл (отмена расписания)."""
    if data is None:
        try:
            _SCHEDULE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        tmp = _SCHEDULE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_SCHEDULE_FILE)


def _parse_schedule_datetime(dt_str: str) -> Optional[datetime]:
    """
    Парсит строку даты+времени в объект datetime.

    Поддерживаемые форматы:
      "HH:MM"               — сегодня в это время (или завтра если уже прошло)
      "DD.MM HH:MM"         — конкретная дата текущего года
      "DD.MM.YYYY HH:MM"    — конкретная дата с годом

    Возвращает None если формат не распознан.
    """
    import re
    dt_str = dt_str.strip()
    now = datetime.now()

    # Формат 1: "HH:MM"
    if re.match(r"^\d{2}:\d{2}$", dt_str):
        t = datetime.strptime(dt_str, "%H:%M")
        candidate = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        # Если время уже прошло сегодня — переносим на завтра.
        # Используем timedelta вместо replace(day=...) — иначе крэш в конце месяца (day=32).
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    # Формат 2: "DD.MM HH:MM"
    if re.match(r"^\d{2}\.\d{2} \d{2}:\d{2}$", dt_str):
        return datetime.strptime(f"{dt_str}.{now.year}", "%d.%m %H:%M.%Y")

    # Формат 3: "DD.MM.YYYY HH:MM"
    if re.match(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}$", dt_str):
        return datetime.strptime(dt_str, "%d.%m.%Y %H:%M")

    return None


# Отдельный файловый lock для решения "пора ли стартовать".
# НЕ совпадает с indexing.lock — иначе scheduler бы конфликтовал с уже идущей
# индексацией. Этот lock защищает только короткую критическую секцию:
# read schedule → check time → mark fired → write schedule. Сама индексация
# запускается уже ПОСЛЕ освобождения этого lock — она возьмёт свой собственный
# межпроцессный lock в hot_swap.run_shadow_indexing.
_SCHEDULER_DECISION_LOCK_FILE = (
    Path(__file__).resolve().parent.parent / "scheduler.lock"
)


def _try_acquire_scheduler_lock():
    """
    Пытается захватить файловый lock для compare-and-swap расписания.
    Возвращает file handle если успешно, None если уже захвачен.
    Lock держится КОРОТКО — только пока читаем/пишем schedule_state.json.
    """
    try:
        _SCHEDULER_DECISION_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = open(_SCHEDULER_DECISION_LOCK_FILE, "a+")
        try:
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            return fh
        except portalocker.exceptions.LockException:
            fh.close()
            return None
    except Exception as e:
        logger.error(f"[Scheduler] Decision lock error: {e}")
        return None


def _release_scheduler_lock(fh) -> None:
    """Освобождает scheduler decision lock."""
    if fh is None:
        return
    try:
        portalocker.unlock(fh)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


async def _scheduler_loop() -> None:
    """
    Фоновая корутина: проверяет расписание каждые 30 секунд.
    Если текущее время совпадает с запланированным — запускает горячую переиндексацию.
    Запускается один раз при старте FastAPI (в lifespan).

    Поддерживает recurring (ежедневное повторение):
      Если в расписании есть поле "recurring": true — после срабатывания
      автоматически создаётся новое расписание на завтра в то же время.

    Безопасно для workers=2 в uvicorn:
      Compare-and-swap (read fired → mark fired=True) выполняется под файловым
      межпроцессным lock'ом (scheduler.lock). Только один из двух воркеров
      пройдёт критическую секцию; второй увидит fired=True и пропустит запуск.
    """
    logger.info("[Scheduler] Background scheduler started (checks every 30s)")
    while True:
        # Инициализируем в начале каждой итерации — переменные в Python
        # персистентны между итерациями while True, поэтому без сброса
        # fire_indexing от предыдущей итерации может остаться True.
        fire_indexing = False
        try:
            # Дешёвая предпроверка БЕЗ lock'a — большую часть времени
            # расписания нет или время не пришло. Lock берём только когда
            # реально надо что-то менять.
            pre = _read_schedule()
            should_try_fire = False
            if pre and not pre.get("fired"):
                pre_dt_str = pre.get("scheduled_datetime", "")
                if pre_dt_str:
                    try:
                        pre_dt = datetime.fromisoformat(pre_dt_str)
                        now_pre = datetime.now()
                        if now_pre >= pre_dt and (now_pre - pre_dt).total_seconds() <= 300:
                            should_try_fire = True
                    except Exception:
                        pass

            if should_try_fire:
                # ── Критическая секция: read-modify-write под межпроцессным lock'ом ─
                lock_fh = _try_acquire_scheduler_lock()
                if lock_fh is None:
                    # Другой воркер уже принимает решение — мы выходим.
                    # На следующей итерации (через 30с) мы увидим fired=True
                    # либо обновлённое recurring-расписание.
                    logger.debug("[Scheduler] Decision lock held by another worker, skipping")
                else:
                    try:
                        # Перечитываем под lock'ом — состояние могло измениться.
                        schedule = _read_schedule()
                        if schedule and not schedule.get("fired"):
                            scheduled_dt_str = schedule.get("scheduled_datetime", "")
                            if scheduled_dt_str:
                                scheduled_dt = datetime.fromisoformat(scheduled_dt_str)
                                now = datetime.now()
                                if now >= scheduled_dt and (now - scheduled_dt).total_seconds() <= 300:
                                    logger.info(
                                        f"[Scheduler] Firing scheduled indexing at {scheduled_dt_str}"
                                    )

                                    is_recurring = schedule.get("recurring", False)

                                    # Для recurring: сначала пишем СЛЕДУЮЩИЙ запуск,
                                    # потом помечаем текущий как fired.
                                    # Порядок важен: если процесс крашнется между
                                    # двумя записями — лучше потерять пометку fired
                                    # (scheduler срабатывает ещё раз, что безопасно —
                                    # file lock не даст двойной индексации), чем
                                    # потерять следующий запуск (ежедневная индексация
                                    # просто прекратится без алерта).
                                    if is_recurring:
                                        next_dt = scheduled_dt + timedelta(days=1)
                                        if next_dt <= now:
                                            next_dt = now.replace(
                                                hour=scheduled_dt.hour,
                                                minute=scheduled_dt.minute,
                                                second=0,
                                                microsecond=0,
                                            ) + timedelta(days=1)

                                        next_schedule = {
                                            "scheduled_datetime": next_dt.isoformat(),
                                            "created_at": datetime.now(timezone.utc).isoformat(),
                                            "fired": False,
                                            "recurring": True,
                                        }
                                        _write_schedule(next_schedule)
                                        logger.info(
                                            f"[Scheduler] Recurring: next run scheduled for "
                                            f"{next_dt.strftime('%d.%m.%Y %H:%M')}"
                                        )

                                    # Пишем fired=True ПОСЛЕ записи следующего расписания
                                    schedule["fired"] = True
                                    schedule["fired_at"] = datetime.now(timezone.utc).isoformat()
                                    _write_schedule(schedule)

                                    # Запускаем индексацию ПОСЛЕ освобождения lock'а
                                    # (см. ниже). Сама индексация возьмёт свой собственный
                                    # межпроцессный lock в hot_swap.
                                    if _PROMETHEUS_AVAILABLE:
                                        _indexing_runs.labels(trigger="scheduler").inc()
                                    fire_indexing = True
                                else:
                                    fire_indexing = False
                            else:
                                fire_indexing = False
                        else:
                            # Кто-то уже пометил fired=True (другой воркер успел раньше)
                            fire_indexing = False
                    finally:
                        _release_scheduler_lock(lock_fh)

                    # Стартуем индексацию ВНЕ lock'а
                    if fire_indexing:
                        from app.indexer.hot_swap import run_shadow_indexing
                        t = threading.Thread(
                            target=run_shadow_indexing,
                            daemon=True,
                            name="hot_swap_indexing",
                        )
                        t.start()

        except Exception as e:
            logger.error(f"[Scheduler] Error in scheduler loop: {e}")

        await asyncio.sleep(30)


# ── Startup / Shutdown ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called once when server starts.
    Connects to Qdrant, ensures collection exists, starts background scheduler.
    """
    global _search_semaphore

    logger.info("Starting RAG service...")

    # asyncio.Semaphore нужно создавать когда event loop уже работает.
    # Module-level создание может привязать его к другому loop'у при workers>1.
    _search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    logger.info(f"Search semaphore initialised: {MAX_CONCURRENT_SEARCHES} slots")

    # Явный executor для asyncio.to_thread. См. _THREADPOOL_SIZE выше:
    # без этого дефолтный пул может стать узким местом при пиковой нагрузке.
    loop = asyncio.get_running_loop()
    custom_executor = ThreadPoolExecutor(
        max_workers=_THREADPOOL_SIZE,
        thread_name_prefix="rag-worker",
    )
    loop.set_default_executor(custom_executor)
    logger.info(f"Default executor: ThreadPoolExecutor(max_workers={_THREADPOOL_SIZE})")

    try:
        ensure_collection_exists()
        stats = get_collection_stats()
        logger.info(f"Qdrant ready. Chunks in collection: {stats['total_chunks']}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.warning("Make sure Docker is running: docker-compose up -d")

    # Запускаем планировщик в фоне
    scheduler_task = asyncio.create_task(_scheduler_loop())

    yield  # Server is running

    scheduler_task.cancel()
    # Аккуратно гасим executor (ждём активные таски, не принимаем новые).
    # wait=False — uvicorn уже идёт на выход, не блокируем shutdown.
    try:
        custom_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    logger.info("RAG service shutting down.")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="CAIU RAG Service",
    description="Semantic search API for CAIU Telegram bot",
    version="1.0.0",
    lifespan=lifespan,
)

# RAG-сервис вызывается только ботом с того же хоста → ограничиваем CORS.
# Если понадобится доступ с другого хоста (например, веб-панель на отдельном порту)
# — добавьте нужный origin явно вместо "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1",
                   "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ── Security: API Key check ────────────────────────────────────
def verify_api_key(x_api_key: str = Header(..., description="Secret API key")):
    """
    Check that the request has a valid API key.
    Bot sends this key in every request header.

    Usage: add  X-API-Key: your_secret_key  to request headers.

    Сравнение через secrets.compare_digest — constant-time, защита от timing
    attack (атакующий, замеряя время ответа, может постепенно угадать ключ
    байт за байтом, если использовать обычное ==).
    """
    expected = settings.API_SECRET_KEY or ""
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Request / Response models ─────────────────────────────────
class SearchRequest(BaseModel):
    """What the bot sends to /search"""
    question: str                      # User's question
    top_k: Optional[int] = None        # How many chunks to return (default from config)
    min_score: Optional[float] = None  # Minimum relevance score (default from config)
    format_as_context: bool = True     # True = return formatted string; False = return raw list

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Как поступить в КАИУ?",
                "top_k": 5,
                "min_score": 0.3,
                "format_as_context": True,
            }
        }


class SearchResultItem(BaseModel):
    """One chunk returned by /search"""
    text: str          # Chunk text
    page_url: str      # Source URL
    page_title: str    # Source page title
    chunk_index: int   # Position on page (0 = first chunk)
    score: float       # Relevance score (0.0 to 1.0)


class CandidateUrl(BaseModel):
    """Страница-кандидат для показа когда бот не нашёл уверенного ответа."""
    url: str
    title: str
    external_links: List[str] = []
    # external_links — Google Docs, PDF и т.д. найденные на этой странице.
    # Бот показывает их как прямые ссылки на документы.


class SearchResponse(BaseModel):
    """Ответ /search — возвращается боту"""
    question: str
    context: str                        # Formatted context string for ChatGPT
    results: List[SearchResultItem]     # Raw chunks (for debugging)
    total_found: int                    # How many chunks found
    search_time_ms: int                 # How long search took
    candidate_urls: List[CandidateUrl] = []  # Страницы-кандидаты (когда context пустой)


class IndexRequest(BaseModel):
    """Request body for /index endpoint"""
    # Optional: run in background (default True)
    # If False, wait for indexing to complete (can take minutes!)
    background: bool = True


class IndexResponse(BaseModel):
    """Response from /index"""
    message: str
    status: str  # "started" or "completed"


class StatsResponse(BaseModel):
    """Response from /stats"""
    total_chunks: int
    collection: str
    status: str


class IndexStatusResponse(BaseModel):
    """Response from GET /index/status"""
    in_progress: bool
    started_at: Optional[str]
    active_collection: str
    total_chunks: int
    last_result: Optional[dict]
    schedule: Optional[dict]


class ScheduleRequest(BaseModel):
    """
    Request body for POST /index/schedule.

    Поддерживаемые форматы поля `datetime`:
      "HH:MM"            — сегодня в это время (если прошло — завтра)
      "DD.MM HH:MM"      — конкретный день текущего года
      "DD.MM.YYYY HH:MM" — конкретный день с годом

    Поле `recurring`:
      True  — ежедневное повторение (автоматически планирует следующий запуск)
      False — одноразово (по умолчанию)
    """
    datetime: str           # например "03:00", "15.06 03:00", "15.06.2025 03:00"
    recurring: bool = False # True = ежедневно в это время

    class Config:
        json_schema_extra = {
            "example": {"datetime": "03:00", "recurring": True}
        }


class ScheduleResponse(BaseModel):
    """Response from /index/schedule"""
    message: str
    scheduled_datetime: str


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check — no auth required.
    Проверяет Qdrant и Redis. Возвращает 200 если сервис работает,
    200 с degraded=True если один из компонентов недоступен,
    503 если Qdrant (критический) недоступен.

    Используется systemd-watchdog и внешним мониторингом.
    """
    from app.indexer.storage import get_client
    from app.redis_client import get_redis, is_redis_available

    components: dict = {}
    overall_ok = True

    # ── Qdrant (критический компонент — без него поиск невозможен) ────────────
    try:
        client = get_client()
        client.get_collections()  # лёгкий запрос, не трогает данные
        components["qdrant"] = "ok"
    except Exception as e:
        components["qdrant"] = f"error: {e}"
        overall_ok = False

    # ── Redis (некритический — без него кеши просто не работают) ─────────────
    try:
        r = get_redis()
        if r is not None:
            r.ping()
            components["redis"] = "ok"
        else:
            components["redis"] = "unavailable"
    except Exception as e:
        components["redis"] = f"error: {e}"

    status_code = 200 if overall_ok else 503
    body = {
        "status": "ok" if overall_ok else "degraded",
        "service": "CAIU RAG Service",
        "components": components,
    }

    from fastapi.responses import JSONResponse
    return JSONResponse(content=body, status_code=status_code)


@app.get("/metrics")
def metrics_endpoint():
    """
    Prometheus metrics endpoint — no auth required.
    Возвращает метрики в формате Prometheus text exposition.

    Подключение к Prometheus (prometheus.yml):
        scrape_configs:
          - job_name: 'caiu-rag'
            static_configs:
              - targets: ['localhost:8001']

    Метрики:
      caiu_search_total{status="ok|empty|error"}
      caiu_search_latency_seconds (histogram)
      caiu_semaphore_waiting (gauge)
      caiu_indexing_runs_total{trigger="api|scheduler"}
      caiu_indexing_errors_total
    """
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="prometheus_client not installed. Run: pip install prometheus_client",
        )
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", response_model=StatsResponse)
def get_stats(api_key: str = Depends(verify_api_key)):
    """
    Get collection statistics.
    How many chunks are in Qdrant?
    """
    stats = get_collection_stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        collection=stats["collection"],
        status=str(stats["status"]),
    )


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Главный эндпоинт — семантический поиск по базе знаний.

    Бот вызывает его для каждого вопроса пользователя.
    Возвращает релевантные фрагменты текста для передачи в ChatGPT.

    Архитектура (async):
      1. Ожидание слота в семафоре — happens в event loop, поток НЕ занят.
         При 100 одновременных запросах 80 ждут как лёгкие корутины (~1 KB),
         а не как заблокированные потоки threadpool.
      2. Сам поиск (sync OpenAI + sync Qdrant) запускается через asyncio.to_thread —
         блокирующая работа уезжает в threadpool, event loop остаётся свободным
         для других endpoints (/health, /index/status и т.д.).
      3. После поиска release семафора → следующая корутина выходит из ожидания.

    Пример запроса:
        POST /search
        Headers: X-API-Key: your_secret_key
        Body: {"question": "Как поступить в КАИУ?"}
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question.strip()
    logger.info(
        f"Search request: '{question[:80]}...'" if len(question) > 80
        else f"Search request: '{question}'"
    )

    if _search_semaphore is None:
        # Не должно произойти — lifespan инициализирует семафор до приёма запросов.
        logger.error("Search semaphore not initialised — lifespan never ran?")
        raise HTTPException(status_code=500, detail="Service not initialised")

    start_time = time.time()

    # Ждём слота в семафоре — без блокировки потока threadpool.
    # При перегрузке (все 20 слотов заняты дольше 20 сек) — отказ 503,
    # чтобы клиент мог отступить и попробовать снова.
    if _PROMETHEUS_AVAILABLE:
        _semaphore_waiting.inc()
    try:
        await asyncio.wait_for(_search_semaphore.acquire(), timeout=20.0)
    except asyncio.TimeoutError:
        logger.warning("Search semaphore timeout — too many concurrent requests")
        if _PROMETHEUS_AVAILABLE:
            _semaphore_waiting.dec()
            _search_total.labels(status="error").inc()
        raise HTTPException(
            status_code=503,
            detail="Сервис перегружен, попробуйте через несколько секунд.",
        )
    finally:
        if _PROMETHEUS_AVAILABLE:
            _semaphore_waiting.dec()

    try:
        # search() — блокирующая (sync OpenAI, sync Qdrant), запускаем в threadpool.
        # to_thread возвращает корутину — мы её awaiт'им, освобождая event loop
        # для других одновременных запросов (включая ожидающих в семафоре).
        results = await asyncio.to_thread(
            search,
            question,
            request.top_k,
            request.min_score,
        )

        # format_results_as_context — быстрая строковая работа, оставляем в event loop.
        context = format_results_as_context(question, results) if results else ""

        # Когда уверенного ответа нет — получаем страницы-кандидаты для показа в боте.
        # Эмбеддинг уже в кеше после search() → дополнительного вызова OpenAI нет.
        # Тоже sync — через to_thread.
        candidates = []
        if not results:
            raw_candidates = await asyncio.to_thread(get_candidate_urls, question, 3)
            candidates = [
                CandidateUrl(
                    url=c["url"],
                    title=c["title"],
                    external_links=c.get("external_links", []),
                )
                for c in raw_candidates
            ]
            if candidates:
                logger.info(f"No confident results, returning {len(candidates)} candidate URLs")

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Found {len(results)} chunks in {elapsed_ms}ms")

        # ── Prometheus инструментация ──────────────────────────────
        if _PROMETHEUS_AVAILABLE:
            _search_latency.observe(elapsed_ms / 1000)
            _search_total.labels(status="ok" if results else "empty").inc()

        return SearchResponse(
            question=question,
            context=context,
            results=[
                SearchResultItem(
                    text=r.text,
                    page_url=r.page_url,
                    page_title=r.page_title,
                    chunk_index=r.chunk_index,
                    score=r.score,
                )
                for r in results
            ],
            total_found=len(results),
            search_time_ms=elapsed_ms,
            candidate_urls=candidates,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        if _PROMETHEUS_AVAILABLE:
            _search_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        _search_semaphore.release()


def _run_hot_swap_indexing():
    """
    Запускает горячую переиндексацию (shadow indexing + atomic alias swap).
    Используется как фоновая задача FastAPI.
    """
    from app.indexer.hot_swap import run_shadow_indexing
    run_shadow_indexing()


@app.post("/index", response_model=IndexResponse)
def index_endpoint(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Запускает полную переиндексацию с горячей заменой базы знаний.

    Индексирует в теневую коллекцию, затем атомарно переключает алиас.
    Бот продолжает работать во время индексации — переключение происходит
    мгновенно когда новые данные готовы.

    background=True (по умолчанию): возвращает сразу, индексация идёт фоном.
    background=False: ждёт завершения (ВНИМАНИЕ: 30-60 минут!).

    Пример:
        POST /index
        Headers: X-API-Key: your_secret_key
        Body: {"background": true}
    """
    from app.indexer.hot_swap import get_indexing_status
    status = get_indexing_status()
    if status["in_progress"]:
        raise HTTPException(
            status_code=409,
            detail="Indexing already in progress. Check /index/status for details.",
        )

    logger.info(f"Index request received. background={request.background}")

    if _PROMETHEUS_AVAILABLE:
        _indexing_runs.labels(trigger="api").inc()

    if request.background:
        background_tasks.add_task(_run_hot_swap_indexing)
        return IndexResponse(
            message="Hot-swap indexing started in background. "
                    "Bot continues working. Check /index/status to monitor progress.",
            status="started",
        )
    else:
        # Синхронно — ждём завершения (блокирует запрос!)
        from app.indexer.hot_swap import run_shadow_indexing
        result = run_shadow_indexing()
        stats = get_collection_stats()
        return IndexResponse(
            message=f"Indexing complete. Active: '{result.get('swapped_to', '?')}'. "
                    f"Total chunks: {stats['total_chunks']}",
            status="completed",
        )


@app.get("/index/status", response_model=IndexStatusResponse)
def index_status_endpoint(api_key: str = Depends(verify_api_key)):
    """
    Возвращает текущее состояние индексации и активную коллекцию.

    Пример:
        GET /index/status
        Headers: X-API-Key: your_secret_key
    """
    from app.indexer.hot_swap import get_indexing_status
    status = get_indexing_status()
    schedule = _read_schedule()

    # Добавляем читаемое время в расписание для удобства
    if schedule and schedule.get("scheduled_datetime"):
        try:
            dt = datetime.fromisoformat(schedule["scheduled_datetime"])
            schedule["scheduled_datetime_human"] = dt.strftime("%d.%m.%Y %H:%M")
        except Exception:
            pass

    return IndexStatusResponse(
        in_progress=status["in_progress"],
        started_at=status.get("started_at"),
        active_collection=status["active_collection"],
        total_chunks=status["total_chunks"],
        last_result=status.get("last_result"),
        schedule=schedule,
    )


@app.post("/index/schedule", response_model=ScheduleResponse)
def schedule_index_endpoint(
    request: ScheduleRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Запланировать горячую переиндексацию на указанную дату и время.

    Форматы поля datetime:
      "HH:MM"            — сегодня/завтра в это время
      "DD.MM HH:MM"      — конкретный день текущего года
      "DD.MM.YYYY HH:MM" — конкретный день с годом

    Если расписание уже существует — возвращает 409. Сначала отмени через DELETE /index/schedule.

    Пример:
        POST /index/schedule
        Headers: X-API-Key: your_secret_key
        Body: {"datetime": "15.06 03:00"}
    """
    # Проверяем нет ли уже активного расписания
    existing = _read_schedule()
    if existing and not existing.get("fired"):
        raise HTTPException(
            status_code=409,
            detail=f"Indexing already scheduled for {existing.get('scheduled_datetime', '?')}. "
                   f"Cancel it first via DELETE /index/schedule.",
        )

    # Парсим дату/время
    scheduled_dt = _parse_schedule_datetime(request.datetime)
    if scheduled_dt is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid datetime format. Supported formats: "
                "'HH:MM', 'DD.MM HH:MM', 'DD.MM.YYYY HH:MM'. "
                "Example: '03:00' or '15.06 03:00'"
            ),
        )

    if scheduled_dt <= datetime.now():
        raise HTTPException(
            status_code=400,
            detail=f"Scheduled time {scheduled_dt.strftime('%d.%m.%Y %H:%M')} is in the past.",
        )

    schedule_data = {
        "scheduled_datetime": scheduled_dt.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fired": False,
        "recurring": request.recurring,
    }
    _write_schedule(schedule_data)

    human_dt = scheduled_dt.strftime("%d.%m.%Y %H:%M")
    recurring_note = " (ежедневно)" if request.recurring else ""
    logger.info(f"Indexing scheduled for {human_dt}{recurring_note}")

    return ScheduleResponse(
        message=f"Hot-swap indexing scheduled for {human_dt} (server local time){recurring_note}. "
                f"Cancel via DELETE /index/schedule.",
        scheduled_datetime=human_dt,
    )


@app.delete("/index/schedule", response_model=IndexResponse)
def unschedule_index_endpoint(api_key: str = Depends(verify_api_key)):
    """
    Отменить запланированную переиндексацию.

    Пример:
        DELETE /index/schedule
        Headers: X-API-Key: your_secret_key
    """
    schedule = _read_schedule()
    if not schedule:
        raise HTTPException(status_code=404, detail="No indexing is scheduled.")

    scheduled_dt_str = schedule.get("scheduled_datetime", "?")
    # Форматируем для человека если возможно
    try:
        scheduled_human = datetime.fromisoformat(scheduled_dt_str).strftime("%d.%m.%Y %H:%M")
    except Exception:
        scheduled_human = scheduled_dt_str

    _write_schedule(None)  # удаляем файл
    logger.info(f"Scheduled indexing at {scheduled_human} was cancelled")

    return IndexResponse(
        message=f"Scheduled indexing at {scheduled_human} has been cancelled.",
        status="cancelled",
    )
