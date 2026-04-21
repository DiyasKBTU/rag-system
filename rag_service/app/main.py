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
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.logging_setup import setup_logging
setup_logging("api")
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.retrieval.search import search, format_results_as_context
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


async def _scheduler_loop() -> None:
    """
    Фоновая корутина: проверяет расписание каждые 30 секунд.
    Если текущее время совпадает с запланированным — запускает горячую переиндексацию.
    Запускается один раз при старте FastAPI (в lifespan).
    """
    logger.info("[Scheduler] Background scheduler started (checks every 30s)")
    while True:
        try:
            schedule = _read_schedule()
            if schedule and not schedule.get("fired"):
                scheduled_dt_str = schedule.get("scheduled_datetime", "")
                if scheduled_dt_str:
                    scheduled_dt = datetime.fromisoformat(scheduled_dt_str)
                    now = datetime.now()
                    # Запускаем если время пришло.
                    # Допуск 5 минут — на случай если сервис рестартовал чуть после расписания.
                    # fired=True предотвращает повторный запуск при следующей проверке.
                    if now >= scheduled_dt and (now - scheduled_dt).total_seconds() <= 300:
                        logger.info(f"[Scheduler] Firing scheduled indexing at {scheduled_dt_str}")
                        schedule["fired"] = True
                        schedule["fired_at"] = datetime.now(timezone.utc).isoformat()
                        _write_schedule(schedule)

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
    logger.info("Starting RAG service...")
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
    logger.info("RAG service shutting down.")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="CAIU RAG Service",
    description="Semantic search API for CAIU Telegram bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security: API Key check ────────────────────────────────────
def verify_api_key(x_api_key: str = Header(..., description="Secret API key")):
    """
    Check that the request has a valid API key.
    Bot sends this key in every request header.

    Usage: add  X-API-Key: your_secret_key  to request headers.
    """
    if x_api_key != settings.API_SECRET_KEY:
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


class SearchResponse(BaseModel):
    """Ответ /search — возвращается боту"""
    question: str
    context: str                        # Formatted context string for ChatGPT
    results: List[SearchResultItem]     # Raw chunks (for debugging)
    total_found: int                    # How many chunks found
    search_time_ms: int                 # How long search took


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
    """
    datetime: str  # например "03:00", "15.06 03:00", "15.06.2025 03:00"

    class Config:
        json_schema_extra = {
            "example": {"datetime": "15.06 03:00"}
        }


class ScheduleResponse(BaseModel):
    """Response from /index/schedule"""
    message: str
    scheduled_datetime: str


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check - no auth required.
    Returns 200 if server is alive.
    Used by monitoring tools, load balancers, etc.
    """
    return {"status": "ok", "service": "CAIU RAG Service"}


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
def search_endpoint(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Главный эндпоинт — семантический поиск по базе знаний.

    Бот вызывает его для каждого вопроса пользователя.
    Возвращает релевантные фрагменты текста для передачи в ChatGPT.

    Пример запроса:
        POST /search
        Headers: X-API-Key: your_secret_key
        Body: {"question": "Как поступить в КАИУ?"}
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question.strip()
    logger.info(f"Search request: '{question[:80]}...' " if len(question) > 80 else f"Search request: '{question}'")

    start_time = time.time()

    try:
        # Один вызов search() → форматируем готовые результаты.
        # format_results_as_context не вызывает search() повторно — нет двойных расходов.
        results = search(
            question=question,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        context = format_results_as_context(question, results) if results else ""

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Found {len(results)} chunks in {elapsed_ms}ms")

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
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


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
    }
    _write_schedule(schedule_data)

    human_dt = scheduled_dt.strftime("%d.%m.%Y %H:%M")
    logger.info(f"Indexing scheduled for {human_dt}")

    return ScheduleResponse(
        message=f"Hot-swap indexing scheduled for {human_dt} (server local time). "
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
