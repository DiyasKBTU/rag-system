# -*- coding: utf-8 -*-
"""
bot/run.py — Telegram-бот приёмной комиссии ЦАИУ.

Запуск:
    python bot/run.py

Что делает:
    1. Принимает вопрос от пользователя в Telegram
    2. Переводит на русский если вопрос на казахском или английском
    3. Ищет ответ в RAG-сервисе (Qdrant + embeddings)
    4. Формирует ответ через GPT-4.1-mini (streaming — текст появляется по мере генерации)
    5. Отправляет ответ пользователю на его языке

Переменные окружения (.env):
    BOT_TOKEN       — токен Telegram-бота (от @BotFather)
    OPENAI_API_KEY  — ключ OpenAI
    RAG_SERVICE_URL — адрес RAG-сервиса (по умолчанию http://localhost:8001)
    RAG_API_KEY     — секретный ключ RAG-сервиса (API_SECRET_KEY из rag_service/.env)
"""

import asyncio
import hashlib
import html
import logging
import os
import random
import sys
import threading
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import httpx
from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ChatAction
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
# RedisStorage импортируем «мягко» — если redis недоступен (например, пакета нет
# или Docker не запущен), используем MemoryStorage. Так бот всё равно стартует.
try:
    from aiogram.fsm.storage.redis import RedisStorage
    _REDIS_STORAGE_AVAILABLE = True
except Exception:
    _REDIS_STORAGE_AVAILABLE = False
from aiogram.types import (
    Message, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Загрузка конфига ──────────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN        = os.environ["BOT_TOKEN"]
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
RAG_SERVICE_URL  = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
RAG_API_KEY      = os.environ["RAG_API_KEY"]
# Прокси для Telegram API (если api.telegram.org недоступен напрямую).
# Примеры: http://user:pass@proxy:port  или  socks5://proxy:port
# Если не нужен — оставьте пустым или не добавляйте в .env.
TELEGRAM_PROXY   = os.getenv("TELEGRAM_PROXY", "")


def _setup_logging() -> None:
    """
    Логирование в stdout и в файл logs/bot.log с автоматической ротацией.

    Ротация: 5 файлов × 5 MB = максимум 25 MB на диске.
    При каждом рестарте бот дописывает в тот же файл (не создаёт новый).
    """
    from logging.handlers import RotatingFileHandler

    logs_dir = Path(__file__).resolve().parent.parent / "rag_service" / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "bot.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    rotating = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,   # 5 MB на файл
        backupCount=5,        # хранить 5 архивных файлов (bot.log.1 … bot.log.5)
        encoding="utf-8",
    )
    rotating.setFormatter(fmt)
    root.addHandler(rotating)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    for noisy in ("httpx", "httpcore", "openai", "aiogram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(f"Bot logging started → {log_file} (rotating, 5×5MB)")


_setup_logging()
logger = logging.getLogger(__name__)

# ── OpenAI клиент ─────────────────────────────────────────────────────────────
# Таймауты важны: без них зависший запрос к OpenAI держит корутину бесконечно
# и съедает слот httpx connection pool. При пиковой нагрузке (десятки
# одновременных пользователей) накопление зависших корутин = деградация бота.
#
#   connect=5  — TCP-handshake до OpenAI обычно <1с, 5с с большим запасом
#   read=45    — streaming-ответ GPT длинный: 700 max_tokens × ~25 ms/token
#                + сетевая задержка. 30с может не хватить для длинных ответов.
#   write=10   — отправка system prompt + истории (несколько КБ)
#   pool=5     — ожидание свободного коннекта в httpx pool
#
# max_retries=2: библиотека OpenAI сама ретраит на 429/5xx с exponential backoff.
# Streaming ретраит только до получения первого токена.
_openai_timeout = httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0)
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=_openai_timeout,
    max_retries=2,
)

# ── Async Redis клиент для бота ───────────────────────────────────────────────
# Используется для персистентного кеша переводов (TTL 30 дней).
# Если Redis недоступен — работаем только с in-memory OrderedDict (graceful fallback).
_REDIS_HOST         = "localhost"
_REDIS_PORT         = 6379
_REDIS_TTL          = 30 * 24 * 3600     # 30 дней
_REDIS_TRANS_PREFIX = "caiu:trans:"

_async_redis      = None
_redis_available  = None   # None = ещё не проверяли
# Lock защищает от двойной инициализации Redis при первом параллельном запросе.
# Без него N одновременных корутин могут попасть в "if _async_redis is None" и
# создать N клиентов (каждый со своим TCP-соединением).
# Lock создаётся лениво — нельзя на module-level, иначе он привяжется к чужому
# event loop.
_redis_init_lock: asyncio.Lock | None = None


def _get_redis_init_lock() -> asyncio.Lock:
    global _redis_init_lock
    if _redis_init_lock is None:
        _redis_init_lock = asyncio.Lock()
    return _redis_init_lock


async def _get_async_redis():
    """Ленивая инициализация async Redis. Возвращает клиент или None."""
    global _async_redis, _redis_available
    if _redis_available is False:
        return None
    if _async_redis is not None:
        return _async_redis

    async with _get_redis_init_lock():
        # double-check: пока ждали lock, другая корутина могла уже инициализировать
        if _async_redis is not None:
            return _async_redis
        if _redis_available is False:
            return None
        try:
            import redis.asyncio as aioredis
            r = aioredis.Redis(
                host=_REDIS_HOST, port=_REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await r.ping()
            _async_redis = r
            _redis_available = True
            logger.info(f"[Redis] Async клиент подключён: {_REDIS_HOST}:{_REDIS_PORT}")
            return _async_redis
        except Exception as e:
            _redis_available = False
            logger.warning(f"[Redis] Async недоступен ({e}) — кеш переводов только in-memory")
            return None


# ── HTTP клиент для RAG-сервиса ───────────────────────────────────────────────
# Создаём ОДИН раз на весь lifecycle бота:
#   - нет накладных расходов на TCP handshake при каждом запросе (-50..150мс)
#   - connection pool переиспользует уже открытые соединения
#   - timeout разделён: 5с на подключение, 25с на чтение ответа
#     (embedding + Qdrant иногда занимает 10-15с при нагрузке на OpenAI)
_RAG_TIMEOUT = httpx.Timeout(connect=5.0, read=25.0, write=5.0, pool=5.0)
_rag_client: httpx.AsyncClient | None = None
# Lock на пересоздание клиента — иначе при ConnectError несколько корутин
# параллельно обнулят _rag_client и создадут несколько новых.
_rag_client_lock: asyncio.Lock | None = None


def _get_rag_client_lock() -> asyncio.Lock:
    global _rag_client_lock
    if _rag_client_lock is None:
        _rag_client_lock = asyncio.Lock()
    return _rag_client_lock


def _get_rag_client() -> httpx.AsyncClient:
    """Возвращает постоянный httpx клиент, создаёт при первом вызове.

    Лимиты выбраны с запасом на пиковую нагрузку:
      max_connections=30       — до 30 одновременных запросов к RAG-сервису
      max_keepalive_connections=10 — держим 10 соединений открытыми между запросами
    RAG-сервис ограничивает параллельность сам (семафор MAX_CONCURRENT_SEARCHES),
    поэтому 30 соединений здесь не приведут к перегрузке OpenAI.

    Безопасно вызывать конкурентно: создание клиента не требует await, поэтому
    GIL гарантирует, что только один из конкурентных вызовов попадёт в ветку
    создания. Lock используется в _reset_rag_client_locked() ниже — там
    важна именно сериализация relinquish + recreate.
    """
    global _rag_client
    if _rag_client is None or _rag_client.is_closed:
        _rag_client = httpx.AsyncClient(
            timeout=_RAG_TIMEOUT,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        )
    return _rag_client


async def _reset_rag_client_locked() -> httpx.AsyncClient:
    """
    Закрывает текущий httpx клиент и создаёт новый. Используется при ConnectError
    к RAG-сервису (например, после рестарта RAG). Под asyncio.Lock — чтобы
    при 30 параллельных запросах с ошибкой не было N одновременных пересозданий.
    """
    global _rag_client
    async with _get_rag_client_lock():
        # double-check: пока ждали lock, кто-то мог уже пересоздать
        if _rag_client is not None and not _rag_client.is_closed:
            try:
                # Проверим живой ли — если да, возвращаем как есть
                return _rag_client
            except Exception:
                pass
        if _rag_client is not None:
            try:
                await _rag_client.aclose()
            except Exception:
                pass
        _rag_client = httpx.AsyncClient(
            timeout=_RAG_TIMEOUT,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        )
        return _rag_client


# ── Казахские буквы которых нет в русском ────────────────────────────────────
# Если пользователь выбрал "kk" но написал по-русски — не переводим зря.
_KAZAKH_CHARS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# ── Кеш переводов {(вопрос, lang): перевод_на_русском} ───────────────────────
# OrderedDict + ограничение по размеру — простой LRU без зависимостей.
# Максимум 500 записей: хватит на все частые вопросы, не съест память.
_TRANSLATION_CACHE: OrderedDict = OrderedDict()
_TRANSLATION_CACHE_MAX = 500

# Вспомогательный кеш прогрева: {redis_key → translated_text}.
# Заполняется при старте бота из Redis scan (translate_to_russian записей).
# Проверяется в translate_to_russian после вычисления redis_key — до обращения
# к Redis. Записи вытесняются в _TRANSLATION_CACHE при первом реальном обращении.
_TRANSLATION_WARMUP_CACHE: dict = {}

# ── Ограничение длины вопроса ─────────────────────────────────────────────────
# Telegram позволяет до 4096 симв. Без ограничения злоумышленник может отправлять
# длинные тексты, сжигая квоту OpenAI (embedding + перевод + GPT).
# 1000 симв. ≈ 200 слов — с запасом для самых длинных реальных вопросов.
_MAX_QUESTION_LENGTH = 1000

_TOO_LONG_MSG = {
    "kk": "⚠️ Сұрақ тым ұзын. 1000 таңбадан аз жазыңыз.",
    "ru": "⚠️ Вопрос слишком длинный. Пожалуйста, сократите до 1000 символов.",
    "en": "⚠️ Question is too long. Please shorten it to 1000 characters.",
}

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Защита от спама и случайного сжигания бюджета OpenAI.
_RATE_LIMIT_MESSAGES = 10   # максимум сообщений
_RATE_LIMIT_WINDOW   = 60   # за сколько секунд
_RATE_LIMIT_COOLDOWN = 30   # сколько секунд ждать после превышения

_user_timestamps: dict           = defaultdict(list)  # {user_id: [timestamp, ...]}
_rate_limit_lock: threading.Lock = threading.Lock()   # защита от race condition

# ── Отслеживание первого входа ────────────────────────────────────────────────
# Пользователи которые уже видели приветствие.
# Хранится как {user_id: timestamp последней активности} — для TTL-очистки.
# Через сутки запись «протухает»; пользователь снова увидит полное приветствие.
# Это нормально (UX не страдает) и предотвращает утечку памяти на длинной дистанции.
_GREETED_TTL_SECONDS = 24 * 3600
_greeted_users: dict = {}      # {user_id: last_seen_ts}
_greeted_lock: threading.Lock = threading.Lock()

# Период (сек) фоновой очистки in-memory словарей от мёртвых ключей.
# При тысячах разных пользователей за месяцы defaultdict разрастается даже
# когда списки timestamps пустые — каждая запись съедает ~200 байт.
_MEMORY_CLEANUP_INTERVAL = 600   # раз в 10 минут

# ── Тексты интерфейса ─────────────────────────────────────────────────────────
LANGS = {
    "🇰🇿 Қазақша": "kk",
    "🇷🇺 Русский":  "ru",
    "🇬🇧 English":  "en",
}

BACK_LABELS = {"kk": "⬅ Артқа", "ru": "⬅ Назад", "en": "⬅ Back"}

T = {
    # ── Экран приветствия (первый /start) ─────────────────────────────────────
    "greet": (
        "🎓 <b>ЦАИУ · ОАИУ · CAIU</b>\n"
        "📍 Шымкент, Казахстан\n\n"
        "Добро пожаловать! Приёмная комиссия готова ответить на ваши вопросы.\n"
        "Қош келдіңіз! Қабылдау комиссиясы сұрақтарыңызға жауап беруге дайын.\n"
        "Welcome! The admissions office is ready to answer your questions.\n\n"
        "✅ Специальности · Мамандықтар · Programs\n"
        "✅ Стоимость · Оқу ақысы · Tuition\n"
        "✅ Гранты · Гранттар · Grants\n"
        "✅ Документы · Құжаттар · Documents"
    ),
    # Короткий баннер для повторных /start
    "greet_short": (
        "🎓 <b>ЦАИУ · ОАИУ · CAIU</b> — Приёмная комиссия\n"
        "Выберите язык / Тілді таңдаңыз / Choose language 👇"
    ),
    "choose_lang": "🌐 <b>Тілді таңдаңыз / Выберите язык / Choose a language:</b>",
    "ask_prompt": {
        "kk": "✍️ <b>Сұрағыңызды жазыңыз:</b>",
        "ru": "✍️ <b>Напишите ваш вопрос:</b>",
        "en": "✍️ <b>Write your question:</b>",
    },
    "typing": {
        "kk": "⏳",
        "ru": "⏳",
        "en": "⏳",
    },
    "ask_more": {
        "kk": "💬 Тағы сұрақ қойыңыз немесе артқа қайтыңыз:",
        "ru": "💬 Задайте ещё вопрос или вернитесь назад:",
        "en": "💬 Ask another question or go back:",
    },
    "error": {
        "kk": "⚠️ Техникалық мәселе туындады. Қабылдау комиссиясына хабарласыңыз: 📞 +7 707 510 10 10",
        "ru": "⚠️ Возникла техническая проблема. Позвоните в приёмную комиссию: 📞 +7 707 510 10 10",
        "en": "⚠️ A technical issue occurred. Please call the admissions office: 📞 +7 707 510 10 10",
    },
    "rate_limit": {
        "kk": f"⏱ Бір мезетте тым көп сұраныс. {_RATE_LIMIT_COOLDOWN} секунд кейін қайталаңыз. Жедел сұрақ болса: 📞 +7 707 510 10 10",
        "ru": f"⏱ Слишком много вопросов за раз. Подождите {_RATE_LIMIT_COOLDOWN} секунд. По срочным вопросам: 📞 +7 707 510 10 10",
        "en": f"⏱ Too many questions at once. Please wait {_RATE_LIMIT_COOLDOWN} seconds. For urgent questions: 📞 +7 707 510 10 10",
    },
    "help": {
        "kk": (
            "🎓 <b>ОАИУ Қабылдау комиссиясының чат-боты</b>\n\n"
            "Мен мына сұрақтарға жауап бере аламын:\n"
            "• Мамандықтар мен факультеттер\n"
            "• Оқу ақысы және гранттар\n"
            "• Қабылдауға қажетті құжаттар\n"
            "• Жатақхана және кампус\n"
            "• Байланыс және мекенжай\n\n"
            "📞 Тікелей байланыс: +7 707 510 10 10\n"
            "🌐 Сайт: caiu.edu.kz\n\n"
            "Тілді өзгерту үшін /start жазыңыз."
        ),
        "ru": (
            "🎓 <b>Чат-бот приёмной комиссии ЦАИУ</b>\n\n"
            "Я отвечу на вопросы о:\n"
            "• Специальностях и факультетах\n"
            "• Стоимости обучения и грантах\n"
            "• Документах для поступления\n"
            "• Общежитии и кампусе\n"
            "• Контактах и адресе\n\n"
            "📞 Прямая связь: +7 707 510 10 10\n"
            "🌐 Сайт: caiu.edu.kz\n\n"
            "Чтобы сменить язык — напишите /start."
        ),
        "en": (
            "🎓 <b>CAIU Admissions Office Chatbot</b>\n\n"
            "I can answer questions about:\n"
            "• Programs and faculties\n"
            "• Tuition fees and grants\n"
            "• Admission documents\n"
            "• Dormitory and campus\n"
            "• Contacts and address\n\n"
            "📞 Direct line: +7 707 510 10 10\n"
            "🌐 Website: caiu.edu.kz\n\n"
            "To change language, type /start."
        ),
    },
}

# UX #5: FAQ inline-кнопки. Ключ = callback_data суффикс, значение = (label, вопрос для поиска).
FAQ_ITEMS = {
    "kk": [
        ("💰 Оқу ақысы",    "ЦАИУ-да оқу ақысы қанша"),
        ("📋 Құжаттар",      "қабылдауға қандай құжаттар керек"),
        ("🏠 Жатақхана",    "ЦАИУ-да жатақхана бар ма"),
        ("🎓 Мамандықтар",  "ЦАИУ-да қандай мамандықтар бар"),
    ],
    "ru": [
        ("💰 Стоимость",    "сколько стоит обучение в ЦАИУ"),
        ("📋 Документы",    "какие документы нужны для поступления"),
        ("🏠 Общежитие",    "есть ли общежитие в ЦАИУ"),
        ("🎓 Специальности","какие специальности есть в ЦАИУ"),
    ],
    "en": [
        ("💰 Tuition",      "how much does CAIU tuition cost"),
        ("📋 Documents",    "what documents are needed for admission"),
        ("🏠 Dormitory",    "is there a dormitory at CAIU"),
        ("🎓 Specialties",  "what specialties does CAIU offer"),
    ],
}

NO_INFO_SIGNALS = [
    "нет точной информации",
    "нақты ақпарат жоқ",
    "don't have exact information",
]

_LANG_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}

# Название университета на каждом языке
UNI = {
    "ru": {
        "short": "ЦАИУ",
        "full":  "Центральноазиатский инновационный университет",
        "city":  "Шымкент, Казахстан",
    },
    "kk": {
        "short": "ОАИУ",
        "full":  "Орталық Азиялық инновациялық университет",
        "city":  "Шымкент, Қазақстан",
    },
    "en": {
        "short": "CAIU",
        "full":  "Central Asian Innovation University",
        "city":  "Shymkent, Kazakhstan",
    },
}

# ── FSM ───────────────────────────────────────────────────────────────────────
class Chat(StatesGroup):
    waiting = State()


# ── Клавиатуры ────────────────────────────────────────────────────────────────
def lang_keyboard() -> ReplyKeyboardMarkup:
    """Три кнопки языков — по одной в ряд для удобства нажатия."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=label)] for label in LANGS],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def chat_keyboard(lang: str) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_LABELS[lang])]],
        resize_keyboard=True,
    )


def faq_keyboard(lang: str) -> InlineKeyboardMarkup:
    """
    Inline-кнопки частых вопросов — появляются после выбора языка.
    Нажатие = callback_data "faq:ru:0", "faq:ru:1" и т.д.
    """
    items = FAQ_ITEMS.get(lang, FAQ_ITEMS["ru"])
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"faq:{lang}:{i}")]
            for i, (label, _) in enumerate(items)
        ]
    )


# ── Rate limiting ─────────────────────────────────────────────────────────────
def _is_rate_limited(user_id: int) -> bool:
    """
    Проверяет не превысил ли пользователь лимит запросов.
    Возвращает True если нужно заблокировать сообщение.

    Примечание: после очистки старых timestamps, если список пустой,
    ключ удаляется. Это предотвращает рост _user_timestamps в памяти
    при тысячах одноразовых пользователей — без удаления defaultdict
    хранил бы пустой список для каждого user_id навсегда.
    """
    now = time.time()
    with _rate_limit_lock:
        fresh = [t for t in _user_timestamps[user_id] if now - t < _RATE_LIMIT_WINDOW]
        if len(fresh) >= _RATE_LIMIT_MESSAGES:
            _user_timestamps[user_id] = fresh
            return True
        fresh.append(now)
        _user_timestamps[user_id] = fresh
        return False


def _cleanup_in_memory_state() -> None:
    """
    Удаляет из _user_timestamps и _greeted_users записи, по которым давно
    не было активности. Запускается из фонового asyncio-task раз в
    _MEMORY_CLEANUP_INTERVAL секунд.

    Без этого defaultdict + set росли бы навсегда: каждый зашедший хоть раз
    юзер занимал бы память до перезапуска бота.
    """
    now = time.time()
    # _user_timestamps: убираем тех, у кого нет недавних запросов
    with _rate_limit_lock:
        stale = [uid for uid, ts_list in _user_timestamps.items()
                 if not ts_list or all(now - t >= _RATE_LIMIT_WINDOW for t in ts_list)]
        for uid in stale:
            _user_timestamps.pop(uid, None)
        rl_removed = len(stale)

    # _greeted_users: TTL по timestamp
    with _greeted_lock:
        stale_g = [uid for uid, ts in _greeted_users.items()
                   if now - ts >= _GREETED_TTL_SECONDS]
        for uid in stale_g:
            _greeted_users.pop(uid, None)
        g_removed = len(stale_g)

    if rl_removed or g_removed:
        logger.info(
            f"[Cleanup] Removed {rl_removed} stale rate-limit entries, "
            f"{g_removed} stale greeted-user entries "
            f"(rl_size={len(_user_timestamps)}, greeted_size={len(_greeted_users)})"
        )


async def _memory_cleanup_loop() -> None:
    """Фоновый цикл очистки in-memory словарей. Стартует в main()."""
    while True:
        try:
            await asyncio.sleep(_MEMORY_CLEANUP_INTERVAL)
            _cleanup_in_memory_state()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[Cleanup] loop error: {e}")


# ── Кеш переводов (LRU) ───────────────────────────────────────────────────────
def _cache_get(key: tuple) -> str | None:
    if key in _TRANSLATION_CACHE:
        _TRANSLATION_CACHE.move_to_end(key)
        return _TRANSLATION_CACHE[key]
    return None


def _cache_set(key: tuple, value: str) -> None:
    _TRANSLATION_CACHE[key] = value
    _TRANSLATION_CACHE.move_to_end(key)
    while len(_TRANSLATION_CACHE) > _TRANSLATION_CACHE_MAX:
        _TRANSLATION_CACHE.popitem(last=False)


# ── RAG поиск ─────────────────────────────────────────────────────────────────
_RAG_RETRY_ATTEMPTS  = 3
_RAG_RETRY_BASE_DELAY = 1.0


async def get_rag_context(question: str) -> tuple[str, list[dict]]:
    """
    Запрашивает контекст из RAG-сервиса.

    Использует постоянный httpx клиент (нет TCP handshake каждый раз).
    При ConnectError или Timeout — повторяет до 3 раз с экспоненциальным backoff.

    Возвращает:
        (context, candidate_urls)
        context        — строка для GPT, пустая если ответ не найден
        candidate_urls — list of {"url": str, "title": str} для показа пользователю
                         когда context пустой (страницы-кандидаты)
    """
    client = _get_rag_client()

    for attempt in range(1, _RAG_RETRY_ATTEMPTS + 1):
        try:
            resp = await client.post(
                f"{RAG_SERVICE_URL}/search",
                json={"question": question},
                headers={"X-API-Key": RAG_API_KEY},
            )
            if resp.status_code == 200:
                data           = resp.json()
                context        = data.get("context", "")
                found          = data.get("total_found", 0)
                candidate_urls = data.get("candidate_urls", [])
                logger.info(f"[RAG] OK — найдено чанков: {found}, контекст: {len(context)} симв., "
                            f"кандидатов: {len(candidate_urls)}")
                return context, candidate_urls
            elif resp.status_code == 401:
                logger.error(
                    "[RAG] 401 Unauthorized — неверный API ключ. "
                    "Проверь RAG_API_KEY в .env и API_SECRET_KEY в rag_service/.env."
                )
                return "", []
            elif resp.status_code == 500:
                logger.error(f"[RAG] 500 Server Error (попытка {attempt}): {resp.text[:200]}")
            else:
                logger.warning(f"[RAG] Статус {resp.status_code} (попытка {attempt}): {resp.text[:100]}")

        except httpx.ConnectError as e:
            logger.error(f"[RAG] Нет подключения к {RAG_SERVICE_URL} (попытка {attempt}): {e}")
            # Пересоздание клиента — под asyncio.Lock'ом, чтобы 30 параллельных
            # запросов с ошибкой не пересоздали клиента 30 раз.
            client = await _reset_rag_client_locked()

        except httpx.TimeoutException:
            logger.warning(f"[RAG] Таймаут (попытка {attempt}/{_RAG_RETRY_ATTEMPTS})")

        except Exception as e:
            logger.warning(f"[RAG] Ошибка (попытка {attempt}): {e}")

        if attempt < _RAG_RETRY_ATTEMPTS:
            delay = _RAG_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            # Jitter ±30%: при массовых 503 (семафор RAG переполнен) все боты
            # без jitter повторяют запросы синхронно → thundering herd усугубляется.
            # С jitter повторные запросы размазываются по времени.
            jitter = random.uniform(-delay * 0.3, delay * 0.3)
            # При явной перегрузке сервиса (503) ждём вдвое дольше обычного
            is_503 = 'resp' in locals() and hasattr(resp, 'status_code') and resp.status_code == 503
            if is_503:
                delay *= 2
            total_delay = max(0.5, delay + jitter)
            logger.info(f"[RAG] Повтор через {total_delay:.1f}с (503={is_503}, jitter={jitter:+.1f})...")
            await asyncio.sleep(total_delay)

    logger.error(f"[RAG] Все {_RAG_RETRY_ATTEMPTS} попытки исчерпаны")
    return "", []


# ── Перевод на русский ────────────────────────────────────────────────────────
async def translate_to_russian(text: str, source_lang: str) -> str:
    """
    Переводит текст на русский через GPT.

    Двухуровневый кеш:
      1. In-memory LRU (OrderedDict, max 500) — мгновенно, теряется при рестарте
      2. Redis (TTL 30 дней) — персистентный, выживает рестарт бота
    """
    cache_key = (text, source_lang)

    # Уровень 1: in-memory
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Уровень 2: warmup-кеш (redis_key → value, заполнен при старте)
    redis_key = _REDIS_TRANS_PREFIX + source_lang + ":" + hashlib.md5(text.encode()).hexdigest()
    warmed = _TRANSLATION_WARMUP_CACHE.pop(redis_key, None)
    if warmed:
        _cache_set(cache_key, warmed)
        logger.debug(f"[Translate] Warmup hit [{source_lang}]: '{text[:40]}'")
        return warmed

    # Уровень 3: Redis
    r = await _get_async_redis()
    if r:
        try:
            redis_val = await r.get(redis_key)
            if redis_val:
                _cache_set(cache_key, redis_val)
                logger.debug(f"[Translate] Redis hit [{source_lang}]: '{text[:40]}'")
                return redis_val
        except Exception as e:
            logger.debug(f"[Redis] Translate get failed: {e}")

    # Уровень 4: GPT
    lang_name = "казахского" if source_lang == "kk" else "английского"
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"Переведи с {lang_name} на русский. Верни ТОЛЬКО перевод."},
                {"role": "user",   "content": text},
            ],
            temperature=0.0,
            # 800 токенов с запасом на длинные вопросы абитуриентов
            # (часто пишут в 2-3 предложения по-казахски). Раньше было 200 —
            # обрезалось на длинных вопросах, RAG потом искал по обрывку.
            # Стоимость прироста минимальна — это input-токены × дешёвая модель.
            max_tokens=800,
        )
        translated = resp.choices[0].message.content.strip()
        _cache_set(cache_key, translated)
        logger.info(f"[Translate] GPT [{source_lang}→ru]: '{text[:60]}' → '{translated[:60]}'")

        if r:
            try:
                await r.set(redis_key, translated, ex=_REDIS_TTL)
            except Exception as e:
                logger.debug(f"[Redis] Translate set failed: {e}")

        return translated
    except Exception as e:
        logger.warning(f"[Translate] Ошибка перевода: {e}")
        return text


# ── Промпт Айданы ─────────────────────────────────────────────────────────────
def _build_system_prompt(lang: str, context: str) -> str:
    """
    Собирает системный промпт Айданы.
    Переиспользуется в stream_answer (streaming) и generate_answer (fallback).
    """
    lang_name = _LANG_NAMES.get(lang, "Russian")
    u = UNI.get(lang, UNI["ru"])

    lang_extra = ""
    if lang == "kk":
        lang_extra = f"""
═══ КАЗАХСКИЙ ЯЗЫК — СТРОГИЕ ПРАВИЛА ═══

— Отвечай ИСКЛЮЧИТЕЛЬНО на казахском языке — ни одного русского слова в ответе
— База знаний на русском: переводи всю информацию на казахский при ответе
— Университет называй ТОЛЬКО: «{u["short"]}» или «{u["full"]}»
  НИКОГДА не используй «ЦАИУ», «CAIU» или русское полное название
— Используй правильную казахскую грамматику и естественный стиль речи

ТЕРМИНОЛОГИЯ (русский → казахский):
мамандық = специальность | жатақхана = общежитие | оқу ақысы = стоимость обучения
қабылдау комиссиясы = приёмная комиссия | құжаттар = документы | грант = грант
оқуға түсу = поступление | бакалавриат = бакалавриат | факультет = факультет
емтихан = экзамен | зертхана = лаборатория | кітапхана = библиотека
ҰБТ = ЕНТ (единое национальное тестирование) | өтпелі балл = проходной балл
жеңілдік = скидка | шарт = контракт | оқу жылы = учебный год

СТИЛЬ:
— Говори тепло и естественно, как казахскоязычный консультант
— Не смешивай казахский с русским в одном предложении
— Если термин не переводится (например, "Wi-Fi") — используй его как есть
"""
    elif lang == "en":
        lang_extra = f"""
═══ ENGLISH LANGUAGE — STRICT RULES ═══

— Reply EXCLUSIVELY in English — no Russian or Kazakh words in the answer
— The knowledge base is in Russian: translate all information into English
— Always refer to the university as «{u["short"]}» or «{u["full"]}»
  NEVER use «ЦАИУ», «ОАИУ», or the Russian name
— Use natural, professional English appropriate for an admissions consultant

TERMINOLOGY (Russian → English):
specialty / program = специальность | dormitory = общежитие | tuition fee = стоимость обучения
admissions office = приёмная комиссия | documents = документы | grant = грант
enrollment = поступление | bachelor's degree = бакалавриат | faculty = факультет
exam = экзамен | laboratory = зертхана | library = библиотека
UNT (Unified National Testing) = ЕНТ | passing score = проходной балл
discount = жеңілдік | contract = контракт | academic year = учебный год
"""

    no_info = {
        "ru": "К сожалению, у меня нет точной информации по этому вопросу 🙏 Уточните в приёмной комиссии: 📞 +7 707 510 10 10",
        "kk": "Өкінішке орай, бұл сұрақ бойынша нақты ақпарат жоқ 🙏 Қабылдау комиссиясына хабарласыңыз: 📞 +7 707 510 10 10",
        "en": "Unfortunately, I don't have exact information on this 🙏 Please contact the admissions office: 📞 +7 707 510 10 10",
    }

    return f"""Ты — Айдана, живой консультант приёмной комиссии {u["short"]} ({u["full"]}, {u["city"]}).

КТО ТЫ:
Опытный консультант приёмной комиссии с многолетним опытом. Помогаешь абитуриентам
и их родителям разобраться с поступлением. Говоришь как человек, а не инструкция.

═══ ГЛАВНОЕ ПРАВИЛО: ИНФОРМАЦИЯ ═══

Конкретные ФАКТЫ (цифры, даты, баллы, стоимость, телефоны, списки документов,
названия специальностей, имена сотрудников) — бери ТОЛЬКО из базы знаний ниже.
НИКОГДА не выдумывай факты, даже если они "звучат правдоподобно".

ОБЩУЮ ИНФОРМАЦИЮ о системе образования Казахстана, как устроено поступление
в вузы в целом, что такое ЕНТ, как работает грант от государства — можешь
объяснять своими словами опираясь на общие знания, но БЕЗ конкретных цифр.

═══ ЕСЛИ В БАЗЕ НЕТ ТОЧНОГО ОТВЕТА ═══

Веди себя как живой консультант:
1. Если вопрос РЯДОМ с темой в базе — ответь что знаешь из базы, и предложи
   уточнить детали у приёмной комиссии.

2. Если вопрос общий (как поступить, что такое ЕНТ) — можешь объяснить в общем,
   но в конце направь в приёмную комиссию для деталей.

3. Если вопрос вообще не по теме университета — мягко верни к теме:
   "Я помогаю с вопросами о поступлении в ЦАИУ. Что вас интересует?"

4. Если нужна конкретика которой нет — скажи:
   «{no_info.get(lang, no_info["ru"])}»

═══ ВАЖНО: СПИСКИ СПЕЦИАЛЬНОСТЕЙ И ФАКУЛЬТЕТОВ ═══

Когда пользователь спрашивает «какие специальности», «какие факультеты»,
«перечисли направления» — перечисляй ТОЛЬКО то, что есть в базе знаний ниже.
Не добавляй специальности или факультеты которых нет в базе.

В конце такого ответа ОБЯЗАТЕЛЬНО добавь:
«Полный актуальный список — на сайте caiu.edu.kz или по телефону 📞 +7 707 510 10 10»

═══ ЧТО ЗАПРЕЩЕНО ═══

— Выдумывать цифры, даты, баллы, цены, имена
— Выдумывать названия специальностей или факультетов которых нет в базе
— Обещать поступление, грант, скидку, место в общежитии
— Упоминать что ты ИИ / бот / программа
— Писать "согласно базе знаний", "из источников" — говори как человек
— Игнорировать вопрос если база не на 100% совпадает — думай, импровизируй
— Давать медицинские, юридические, политические советы

═══ СТИЛЬ ═══

— Короткие ясные ответы, без "воды"
— 1-2 уместных эмодзи максимум (👋 🎓 📚 📞 🙏 ✅)
— Списки когда есть несколько пунктов, иначе прозой
— Отвечай на языке вопроса: {lang_name}
— Тон: тёплый, живой, уверенный, но без приторности
— Университет в ответах называй: «{u["short"]}»
{lang_extra}
═══ БАЗА ЗНАНИЙ ═══

{context if context else "[для этого вопроса база не вернула релевантных данных]"}""".strip()


# ── Streaming GPT-ответ ───────────────────────────────────────────────────────
# Минимальное количество накопленных символов перед первым редактированием.
# Даём GPT чуть разогнаться — короткие обновления выглядят дёргано.
_STREAM_FIRST_EDIT_CHARS = 80
# Пауза между редактированиями сообщения (секунды).
# Telegram Rate Limit: ~1 правка/сек на чат — больше не нужно.
_STREAM_EDIT_INTERVAL    = 1.1

_GPT_PARAMS = dict(
    model="gpt-4.1-mini",
    temperature=0.3,
    max_tokens=700,
    top_p=0.9,
    frequency_penalty=0.3,
    presence_penalty=0.1,
)

# UX #10: контекст диалога. Храним последние N пар вопрос-ответ в FSM-стейте,
# передаём в GPT между system-промптом и текущим вопросом.
# Значение 3: ~600-900 токенов истории — компромисс между памятью и стоимостью.
_MAX_HISTORY_PAIRS = 3


async def stream_answer_to_message(
    question: str,
    lang: str,
    context: str,
    placeholder: Message,
    history: list | None = None,
) -> tuple[str, bool]:
    """
    Стримит ответ GPT прямо в уже отправленное сообщение Telegram.

    Алгоритм:
      1. Открываем GPT stream
      2. Собираем токены в буфер
      3. Как только накопится _STREAM_FIRST_EDIT_CHARS символов — первое редактирование
      4. Дальше редактируем каждые _STREAM_EDIT_INTERVAL секунд (rate limit Telegram)
      5. По окончании — финальное редактирование с полным текстом

    history: список предыдущих обменов (UX #10).
      Формат: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
      Вставляется между system-промптом и текущим вопросом.

    Преимущество перед обычным вызовом:
      Без стриминга: пользователь ждёт 2-5с полного ответа.
      Со стримингом: первый текст появляется через ~300-500мс после RAG.

    Возвращает (полный_ответ, used_rag).
    """
    system_prompt = _build_system_prompt(lang, context)
    accumulated   = ""
    last_edit_at  = 0.0
    first_edit_done = False

    # Строим список сообщений: system → история → текущий вопрос
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        async with await openai_client.chat.completions.create(
            **_GPT_PARAMS,
            messages=messages,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                accumulated += delta

                now = time.monotonic()

                # Первое редактирование: ждём минимум _STREAM_FIRST_EDIT_CHARS символов
                if not first_edit_done and len(accumulated) >= _STREAM_FIRST_EDIT_CHARS:
                    try:
                        await placeholder.edit_text(accumulated)
                        first_edit_done = True
                        last_edit_at    = now
                    except Exception:
                        pass  # Telegram иногда отклоняет слишком быстрые правки

                # Последующие редактирования: раз в _STREAM_EDIT_INTERVAL
                elif first_edit_done and (now - last_edit_at) >= _STREAM_EDIT_INTERVAL:
                    try:
                        await placeholder.edit_text(accumulated)
                        last_edit_at = now
                    except Exception:
                        pass

        # Финальное редактирование — полный текст с parse_mode.
        # GPT не использует HTML-разметку (в системном промпте мы её не разрешали),
        # поэтому экранируем спецсимволы — иначе случайные '<', '>', '&'
        # в ответе модели сломают HTML-парсер Telegram и сообщение откатится
        # к промежуточной версии (или вообще к "⏳").
        if accumulated:
            try:
                await placeholder.edit_text(html.escape(accumulated), parse_mode="HTML")
            except Exception:
                # Если даже escape-версия не прошла (например, длиннее 4096 симв.)
                # — пробуем без parse_mode
                try:
                    await placeholder.edit_text(accumulated)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"[Stream] GPT streaming failed: {e}", exc_info=True)
        # Fallback — отправляем то что успели набрать
        if accumulated:
            try:
                await placeholder.edit_text(accumulated)
            except Exception:
                pass
        else:
            raise  # если совсем ничего нет — пусть handle_message поймает

    is_no_info = any(signal in accumulated for signal in NO_INFO_SIGNALS)
    used_rag   = bool(context) and not is_no_info
    return accumulated, used_rag


# ── Основная логика обработки вопроса ─────────────────────────────────────────
async def process_question(
    question: str,
    lang: str,
    placeholder: Message,
    history: list | None = None,
) -> tuple[str, bool, list[dict]]:
    """
    Полный пайплайн:
    1. Перевод на русский (если kk с казахскими буквами, или en)
    2. RAG-поиск
    3. Streaming GPT-ответ прямо в placeholder-сообщение

    history: список предыдущих обменов для контекста GPT (UX #10).
      Формат: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
      RAG не получает историю — ищет только по текущему вопросу.

    Возвращает (ответ, used_rag, candidate_urls).
    candidate_urls — страницы-кандидаты для показа когда ответ не найден.
    """
    rag_question = question
    needs_translation = (
        lang == "en"
        or (lang == "kk" and any(ch in _KAZAKH_CHARS for ch in question))
    )
    if needs_translation:
        rag_question = await translate_to_russian(question, lang)

    context, candidate_urls = await get_rag_context(rag_question)

    if context:
        logger.info(f"[RAG] Контекст для GPT ({len(context)} симв.): {context[:300]}...")
    else:
        logger.warning(f"[RAG] Контекст пустой для: '{rag_question[:80]}'")

    answer, used_rag = await stream_answer_to_message(
        question, lang, context, placeholder, history=history
    )
    return answer, used_rag, candidate_urls


# ── Ссылки-кандидаты когда бот не знает ответа ───────────────────────────────
# Когда search() не нашёл уверенного ответа, RAG-сервис возвращает top-3 URL.
# Бот показывает их как подсказки — «возможно, там есть информация».

_CANDIDATE_LINKS_HEADER = {
    "ru": "🔗 <b>Возможно, здесь найдёте ответ:</b>",
    "kk": "🔗 <b>Мүмкін, мұнда жауап табасыз:</b>",
    "en": "🔗 <b>You may find the answer here:</b>",
}


def _format_candidate_links(candidate_urls: list[dict], lang: str) -> str:
    """
    Форматирует список кандидатов в HTML-сообщение.

    Два уровня ссылок:
    1. Страницы-кандидаты (caiu.edu.kz) — «возможно там есть информация»
    2. Внешние документы (Google Docs, PDF) найденные на этих страницах — прямые ссылки

    Пример вывода:
        🔗 Возможно, здесь найдёте ответ:
        • <a href="https://caiu.edu.kz/doc-rus/">Документы для поступления</a>
          📄 <a href="https://docs.google.com/...">Открыть документ</a>
        • <a href="https://caiu.edu.kz/adminssions-ru/">Правила приёма</a>
    """
    if not candidate_urls:
        return ""

    header = _CANDIDATE_LINKS_HEADER.get(lang, _CANDIDATE_LINKS_HEADER["ru"])
    lines  = [header]

    ext_label = {
        "ru": "📄 Документ на этой странице:",
        "kk": "📄 Осы беттегі құжат:",
        "en": "📄 Document on this page:",
    }.get(lang, "📄 Документ:")

    for c in candidate_urls:
        url   = c.get("url", "")
        title = c.get("title", "") or url
        ext   = c.get("external_links", []) or []

        if not url:
            continue

        # Фильтруем все виртуальные URL — они не открываются в браузере
        if any(prefix in url for prefix in ("__catalog__", "__manual__", "__special__", "__facts__")):
            continue

        if len(title) > 60:
            title = title[:57] + "…"

        # Экранируем title и url: '&', '<', '>' в реальных URL/заголовках страниц
        # ломают HTML-парсер Telegram → сообщение полностью отклоняется.
        # quote=True эскейпит и кавычки — обязательно внутри href="...".
        url_safe   = html.escape(url, quote=True)
        title_safe = html.escape(title, quote=False)
        lines.append(f'• <a href="{url_safe}">{title_safe}</a>')

        # Показываем внешние документы найденные на этой странице
        for ext_url in ext[:2]:   # максимум 2 внешних ссылки на страницу
            ext_url = ext_url.strip()
            if not ext_url:
                continue
            # Короткий отображаемый текст для ссылки
            if "docs.google.com" in ext_url or "drive.google.com" in ext_url:
                link_text = "Google Doc / Drive"
            elif ext_url.endswith(".pdf"):
                link_text = "PDF-документ"
            elif ext_url.endswith((".docx", ".xlsx")):
                link_text = "Файл документа"
            else:
                link_text = "Внешний документ"
            ext_url_safe = html.escape(ext_url, quote=True)
            lines.append(f'  {ext_label} <a href="{ext_url_safe}">{link_text}</a>')

    if len(lines) == 1:
        return ""  # нет ни одной валидной ссылки

    return "\n".join(lines)


# ── Telegram handlers ─────────────────────────────────────────────────────────
router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    now = time.time()
    with _greeted_lock:
        last_seen = _greeted_users.get(user_id)
        # «Уже здоровались» — если запись есть и не протухла по TTL
        already_greeted = last_seen is not None and (now - last_seen) < _GREETED_TTL_SECONDS
        _greeted_users[user_id] = now   # обновляем активность в любом случае

    if not already_greeted:
        await message.answer(T["greet"], parse_mode="HTML")
        await message.answer(T["choose_lang"], parse_mode="HTML", reply_markup=lang_keyboard())
    else:
        await message.answer(T["greet_short"], parse_mode="HTML", reply_markup=lang_keyboard())


@router.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    lang = data.get("lang", "ru")
    if lang not in ("kk", "ru", "en"):
        lang = "ru"
    await message.answer(T["help"][lang], parse_mode="HTML", disable_web_page_preview=True)


@router.message(StateFilter(None), F.text.in_(LANGS))
async def select_language(message: Message, state: FSMContext) -> None:
    lang = LANGS[message.text]
    await state.set_data({"lang": lang})
    await state.set_state(Chat.waiting)
    await message.answer(T["ask_prompt"][lang], parse_mode="HTML", reply_markup=chat_keyboard(lang))
    # UX #5: FAQ inline-кнопки — частые вопросы одним нажатием
    await message.answer("❓ Популярные вопросы:", reply_markup=faq_keyboard(lang))


@router.callback_query(F.data.startswith("faq:"))
async def handle_faq_button(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    """
    UX #5: обрабатывает нажатие FAQ inline-кнопки.
    callback.data = "faq:{lang}:{index}"
    Имитирует ввод вопроса пользователем — запускает полный поиск через RAG.
    """
    await callback.answer()  # убираем «загрузку» на кнопке

    try:
        _, lang, idx_str = callback.data.split(":", 2)
        idx = int(idx_str)
    except (ValueError, IndexError):
        return

    items = FAQ_ITEMS.get(lang, FAQ_ITEMS["ru"])
    if idx >= len(items):
        return

    _label, question = items[idx]

    # UX #10: загружаем историю и сохраняем язык не затирая её
    data = await state.get_data()
    history: list = data.get("history", [])
    await state.set_data({"lang": lang, "history": history})
    await state.set_state(Chat.waiting)

    # Показываем индикатор печати
    try:
        await bot.send_chat_action(chat_id=callback.message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    # Отправляем placeholder-сообщение
    placeholder = await callback.message.answer(T["typing"][lang])

    answer = ""
    try:
        answer, used_rag, candidate_urls = await process_question(
            question, lang, placeholder, history=history
        )
        rag_status = "RAG=YES" if used_rag else "RAG=NO"
        logger.info(f"[FAQ] user={callback.from_user.id} lang={lang} q={question!r} {rag_status}")

        if not used_rag and candidate_urls:
            suffix = _format_candidate_links(candidate_urls, lang)
            if suffix:
                # answer от GPT — сырой текст, suffix — уже валидный HTML.
                # Эскейпим только answer, чтобы '<', '>', '&' в ответе модели
                # не сломали parse_mode="HTML".
                full_text = html.escape(answer) + "\n\n" + suffix
                try:
                    await placeholder.edit_text(
                        full_text, parse_mode="HTML", disable_web_page_preview=True
                    )
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"[FAQ] Ошибка обработки: {e}", exc_info=True)
        try:
            await placeholder.edit_text(T["error"][lang])
        except Exception:
            await callback.message.answer(T["error"][lang])

    # UX #10: обновляем историю
    if answer:
        history = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
        max_msgs = _MAX_HISTORY_PAIRS * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        await state.set_data({"lang": lang, "history": history})

    await callback.message.answer(T["ask_more"][lang], reply_markup=chat_keyboard(lang))


@router.message(Chat.waiting)
async def handle_message(message: Message, state: FSMContext, bot: Bot) -> None:
    text = (message.text or "").strip()
    data = await state.get_data()
    lang = data.get("lang", "ru")
    if lang not in ("kk", "ru", "en"):
        lang = "ru"

    # Назад → выбор языка
    if text in BACK_LABELS.values():
        await state.clear()
        await message.answer(T["greet_short"], parse_mode="HTML", reply_markup=lang_keyboard())
        return

    # UX #6: сразу показываем индикатор «печатает...» — ещё до rate limit проверки.
    # Пользователь видит три точки в ~50мс после отправки, даже пока бот думает.
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass  # не блокируем если Telegram временно недоступен

    # Проверка длины вопроса — защита от намеренно длинных запросов
    if len(text) > _MAX_QUESTION_LENGTH:
        logger.warning(f"[LengthLimit] user_id={message.from_user.id} длина={len(text)} симв.")
        await message.answer(_TOO_LONG_MSG[lang])
        return

    # Rate limit — защита от спама
    if _is_rate_limited(message.from_user.id):
        logger.warning(f"[RateLimit] user_id={message.from_user.id} превысил лимит")
        await message.answer(T["rate_limit"][lang])
        return

    # UX #10: загружаем историю диалога из FSM-стейта
    history: list = data.get("history", [])

    # Отправляем placeholder — в него будем стримить ответ GPT
    placeholder = await message.answer(T["typing"][lang])

    answer = ""
    try:
        answer, used_rag, candidate_urls = await process_question(
            text, lang, placeholder, history=history
        )
        rag_status = "RAG=YES" if used_rag else "RAG=NO"
        # Логируем сам вопрос (обрезанный) — без него при разборе жалоб
        # пользователей приходится сопоставлять время Telegram и сервера.
        # !r ставит кавычки и эскейпит спецсимволы.
        q_short = text[:120] + ("…" if len(text) > 120 else "")
        logger.info(
            f"[Answer] user={message.from_user.id} lang={lang} {rag_status} q={q_short!r}"
        )

        # Когда бот не нашёл уверенного ответа — ссылки на страницы-кандидаты
        # вставляются прямо в основной ответ (одно сообщение вместо двух).
        if not used_rag and candidate_urls:
            logger.info(f"[Candidates] Showing {len(candidate_urls)} links: {[c.get('url','') for c in candidate_urls]}")
            suffix = _format_candidate_links(candidate_urls, lang)
            if suffix:
                # answer от GPT — сырой, suffix — HTML с экранированными URL/title.
                # Эскейпим answer чтобы spec-символы не сломали parse_mode="HTML".
                escaped_answer = html.escape(answer)
                full_text = escaped_answer + "\n\n" + suffix
                try:
                    await placeholder.edit_text(
                        full_text, parse_mode="HTML", disable_web_page_preview=True
                    )
                except Exception:
                    # Fallback без HTML — суффикс с тегами тут отрендерится как
                    # текст, но пользователь хотя бы увидит ответ.
                    try:
                        await placeholder.edit_text(
                            answer + "\n\n" + suffix, disable_web_page_preview=True
                        )
                    except Exception:
                        pass

    except Exception as e:
        logger.error(f"[Bot] Ошибка обработки: {e}", exc_info=True)
        try:
            await placeholder.edit_text(T["error"][lang])
        except Exception:
            await message.answer(T["error"][lang])

    # UX #10: обновляем историю — добавляем пару (вопрос, ответ)
    if answer:
        history = history + [
            {"role": "user",      "content": text},
            {"role": "assistant", "content": answer},
        ]
        # Оставляем только последние _MAX_HISTORY_PAIRS пар
        max_msgs = _MAX_HISTORY_PAIRS * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        await state.set_data({"lang": lang, "history": history})

    await message.answer(T["ask_more"][lang], reply_markup=chat_keyboard(lang))


@router.message(StateFilter(None))
async def fallback(message: Message, state: FSMContext) -> None:
    await message.answer(T["greet_short"], parse_mode="HTML", reply_markup=lang_keyboard())


# ── Запуск ────────────────────────────────────────────────────────────────────
async def main() -> None:
    session = None
    if TELEGRAM_PROXY:
        session = AiohttpSession(proxy=TELEGRAM_PROXY)
        logger.info(f"Telegram proxy: {TELEGRAM_PROXY}")

    bot = Bot(token=BOT_TOKEN, session=session)

    # FSM storage: пробуем Redis, при ошибке откатываемся к MemoryStorage.
    # Почему Redis важен:
    #   MemoryStorage теряет всё состояние при рестарте бота. Пользователи
    #   которые были в Chat.waiting (выбрали язык, ждут ответа) после рестарта
    #   попадут в fallback — их сообщение не будет распознано как вопрос.
    #   История диалога (UX #10) тоже теряется.
    # С Redis: FSM-state и history переживают рестарт; пользователь не замечает.
    fsm_storage = None
    if _REDIS_STORAGE_AVAILABLE:
        try:
            # Используем DB=1 чтобы не пересекаться с кешем эмбеддингов (DB=0).
            fsm_storage = RedisStorage.from_url(
                f"redis://{_REDIS_HOST}:{_REDIS_PORT}/1"
            )
            # ping — убедиться что Redis реально доступен (импорт пакета сам по
            # себе ничего не гарантирует, Docker может быть выключен).
            await fsm_storage.redis.ping()
            logger.info(f"FSM storage: Redis ({_REDIS_HOST}:{_REDIS_PORT}/1)")
        except Exception as e:
            logger.warning(
                f"FSM storage: Redis недоступен ({e}) — fallback на MemoryStorage. "
                f"При рестарте бота состояние диалогов будет потеряно."
            )
            fsm_storage = None
    if fsm_storage is None:
        fsm_storage = MemoryStorage()
        if not _REDIS_STORAGE_AVAILABLE:
            logger.info("FSM storage: MemoryStorage (aiogram redis-extra не установлен)")

    dp = Dispatcher(storage=fsm_storage)
    dp.include_router(router)

    # ── Прогрев translation cache из Redis ────────────────────────────────────
    # При рестарте in-memory LRU (_TRANSLATION_CACHE) пустой, Redis — нет.
    # Без прогрева первые N уникальных вопросов делают async round-trip к Redis
    # вместо мгновенного ответа из памяти (~0.5 мс vs ~0 мс, + await overhead).
    #
    # Проблема: ключ в памяти — (text, lang), а в Redis — caiu:trans:{lang}:{md5}.
    # Мы не знаем оригинальный text из MD5. Поэтому используем вспомогательный
    # словарь _TRANSLATION_WARMUP_CACHE (redis_key → value), который проверяется
    # в translate_to_russian ПОСЛЕ вычисления redis_key.
    try:
        r_warm = await _get_async_redis()
        if r_warm:
            warm_count = 0
            async for rk in r_warm.scan_iter(f"{_REDIS_TRANS_PREFIX}*"):
                val = await r_warm.get(rk)
                if val and warm_count < _TRANSLATION_CACHE_MAX:
                    _TRANSLATION_WARMUP_CACHE[rk] = val
                    warm_count += 1
            if warm_count:
                logger.info(f"[TransCache] Warmed {warm_count} translations from Redis")
    except Exception as e:
        logger.warning(f"[TransCache] Warmup failed (non-critical): {e}")

    # Фоновая очистка in-memory словарей — раз в _MEMORY_CLEANUP_INTERVAL сек.
    # Без этого _user_timestamps и _greeted_users растут навсегда, по ~200 байт
    # на каждого когда-либо писавшего пользователя.
    cleanup_task = asyncio.create_task(_memory_cleanup_loop())

    async def on_shutdown():
        global _rag_client
        cleanup_task.cancel()
        if _rag_client and not _rag_client.is_closed:
            await _rag_client.aclose()
            logger.info("httpx RAG client closed")
        # Закрываем FSM Redis (если использовался) — иначе при следующем
        # рестарте может остаться зависшее TCP-соединение.
        try:
            close = getattr(fsm_storage, "close", None)
            if close:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass

    dp.shutdown.register(on_shutdown)
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
