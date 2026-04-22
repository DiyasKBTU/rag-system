# -*- coding: utf-8 -*-
"""
bot/run.py — Telegram-бот приёмной комиссии ЦАИУ.

Запуск:
    python bot/run.py

Что делает:
    1. Принимает вопрос от пользователя в Telegram
    2. Переводит на русский если вопрос на казахском или английском
    3. Ищет ответ в RAG-сервисе (Qdrant + embeddings)
    4. Формирует ответ через GPT-4.1-mini
    5. Отправляет ответ пользователю на его языке

Переменные окружения (.env):
    BOT_TOKEN       — токен Telegram-бота (от @BotFather)
    OPENAI_API_KEY  — ключ OpenAI
    RAG_SERVICE_URL — адрес RAG-сервиса (по умолчанию http://localhost:8001)
    RAG_API_KEY     — секретный ключ RAG-сервиса (API_SECRET_KEY из rag_service/.env)
"""

import asyncio
import logging
import os
import sys
import threading
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import httpx
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Загрузка конфига ──────────────────────────────────────────────────────────
load_dotenv()

BOT_TOKEN       = os.environ["BOT_TOKEN"]
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
RAG_API_KEY     = os.environ["RAG_API_KEY"]


def _setup_logging() -> None:
    """Логирование в stdout и в файл logs/bot_YYYY-MM-DD_HH-MM.log."""
    from datetime import datetime

    logs_dir = Path(__file__).resolve().parent.parent / "rag_service" / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"bot_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in [logging.FileHandler(log_file, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        root.addHandler(h)

    for noisy in ("httpx", "httpcore", "openai", "aiogram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(f"Bot logging started → {log_file}")


_setup_logging()
logger = logging.getLogger(__name__)

# ── OpenAI клиент ─────────────────────────────────────────────────────────────
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Казахские буквы которых нет в русском ────────────────────────────────────
# Если пользователь выбрал "kk" но написал по-русски — не переводим зря.
_KAZAKH_CHARS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# ── Кеш переводов {(вопрос, lang): перевод_на_русском} ───────────────────────
# OrderedDict + ограничение по размеру — простой LRU без зависимостей.
# Максимум 500 записей: хватит на все частые вопросы, не съест память.
_TRANSLATION_CACHE: OrderedDict = OrderedDict()
_TRANSLATION_CACHE_MAX = 500

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Защита от спама и случайного сжигания бюджета OpenAI.
_RATE_LIMIT_MESSAGES = 10   # максимум сообщений
_RATE_LIMIT_WINDOW   = 60   # за сколько секунд
_RATE_LIMIT_COOLDOWN = 30   # сколько секунд ждать после превышения

_user_timestamps: dict           = defaultdict(list)  # {user_id: [timestamp, ...]}
_rate_limit_lock: threading.Lock = threading.Lock()   # защита от race condition

# ── Отслеживание первого входа ────────────────────────────────────────────────
# Пользователи которые уже видели приветствие.
# Сбрасывается при перезапуске бота (это нормально, т.к. MemoryStorage тоже сбрасывается).
_greeted_users: set = set()

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
        "kk": "⏳ <i>Жауап дайындалып жатыр...</i>",
        "ru": "⏳ <i>Готовлю ответ...</i>",
        "en": "⏳ <i>Preparing an answer...</i>",
    },
    "ask_more": {
        "kk": "💬 Тағы сұрақ қойыңыз немесе артқа қайтыңыз:",
        "ru": "💬 Задайте ещё вопрос или вернитесь назад:",
        "en": "💬 Ask another question or go back:",
    },
    "error": {
        "kk": "⚠️ Қате орын алды. Кейінірек қайталап көріңіз.",
        "ru": "⚠️ Произошла ошибка. Попробуйте позже.",
        "en": "⚠️ An error occurred. Please try again later.",
    },
    "rate_limit": {
        "kk": f"⏱ Тым көп сұраныс. {_RATE_LIMIT_COOLDOWN} секунд күтіңіз.",
        "ru": f"⏱ Слишком много запросов. Подождите {_RATE_LIMIT_COOLDOWN} секунд.",
        "en": f"⏱ Too many requests. Please wait {_RATE_LIMIT_COOLDOWN} seconds.",
    },
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


# ── Rate limiting ─────────────────────────────────────────────────────────────
def _is_rate_limited(user_id: int) -> bool:
    """
    Проверяет не превысил ли пользователь лимит запросов.
    Возвращает True если нужно заблокировать сообщение.

    Алгоритм: хранит временные метки последних сообщений.
    Использует threading.Lock для защиты от race condition при concurrent запросах.
    """
    now = time.time()
    with _rate_limit_lock:
        _user_timestamps[user_id] = [
            t for t in _user_timestamps[user_id]
            if now - t < _RATE_LIMIT_WINDOW
        ]
        if len(_user_timestamps[user_id]) >= _RATE_LIMIT_MESSAGES:
            return True
        _user_timestamps[user_id].append(now)
        return False


# ── Кеш переводов (LRU) ───────────────────────────────────────────────────────
def _cache_get(key: tuple) -> str | None:
    """Возвращает закешированный перевод или None."""
    if key in _TRANSLATION_CACHE:
        _TRANSLATION_CACHE.move_to_end(key)
        return _TRANSLATION_CACHE[key]
    return None


def _cache_set(key: tuple, value: str) -> None:
    """Сохраняет перевод в кеш. Удаляет самый старый элемент при переполнении."""
    _TRANSLATION_CACHE[key] = value
    _TRANSLATION_CACHE.move_to_end(key)
    while len(_TRANSLATION_CACHE) > _TRANSLATION_CACHE_MAX:
        _TRANSLATION_CACHE.popitem(last=False)


# ── RAG поиск ─────────────────────────────────────────────────────────────────
async def get_rag_context(question: str) -> str:
    """Запрашивает контекст из RAG-сервиса. Возвращает пустую строку при ошибке."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{RAG_SERVICE_URL}/search",
                json={"question": question},
                headers={"X-API-Key": RAG_API_KEY},
            )
            if resp.status_code == 200:
                data    = resp.json()
                context = data.get("context", "")
                found   = data.get("total_found", 0)
                logger.info(f"[RAG] OK — найдено чанков: {found}, контекст: {len(context)} симв.")
                return context
            elif resp.status_code == 401:
                logger.error(
                    "[RAG] 401 Unauthorized — неверный API ключ. "
                    "Проверь RAG_API_KEY в .env и API_SECRET_KEY в rag_service/.env — они должны совпадать."
                )
            elif resp.status_code == 500:
                logger.error(f"[RAG] 500 Server Error: {resp.text[:200]}")
            else:
                logger.warning(f"[RAG] Неожиданный статус {resp.status_code}: {resp.text[:200]}")
    except httpx.TimeoutException:
        logger.warning("[RAG] Таймаут запроса (>15s) — RAG-сервис не успел ответить")
    except httpx.ConnectError:
        logger.error(f"[RAG] Не удалось подключиться к {RAG_SERVICE_URL} — сервис не запущен?")
    except Exception as e:
        logger.warning(f"[RAG] Ошибка запроса: {e}")
    return ""


# ── Перевод на русский ────────────────────────────────────────────────────────
async def translate_to_russian(text: str, source_lang: str) -> str:
    """
    Переводит текст на русский через GPT.
    Использует LRU-кеш: одинаковый вопрос переводится только один раз.
    Кеш ограничен 500 записями — не растёт бесконечно.
    """
    cache_key = (text, source_lang)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    lang_name = "казахского" if source_lang == "kk" else "английского"
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"Переведи с {lang_name} на русский. Верни ТОЛЬКО перевод."},
                {"role": "user",   "content": text},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        translated = resp.choices[0].message.content.strip()
        _cache_set(cache_key, translated)
        logger.info(f"[Translate] [{source_lang}→ru]: '{text}' → '{translated}'")
        return translated
    except Exception as e:
        logger.warning(f"[Translate] Ошибка перевода: {e}")
        return text  # fallback: ищем оригинал


# ── Генерация ответа через GPT ────────────────────────────────────────────────
async def generate_answer(question: str, lang: str, context: str) -> tuple[str, bool]:
    """
    Генерирует ответ Айданы через GPT.
    Возвращает (ответ, used_rag).
    """
    lang_name = _LANG_NAMES.get(lang, "Russian")
    u = UNI.get(lang, UNI["ru"])  # название университета на языке пользователя

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

    system_prompt = f"""Ты — Айдана, живой консультант приёмной комиссии {u["short"]} ({u["full"]}, {u["city"]}).

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

    resp = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.3,
        max_tokens=700,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.1,
    )
    answer = resp.choices[0].message.content.strip()

    is_no_info = any(signal in answer for signal in NO_INFO_SIGNALS)
    used_rag   = bool(context) and not is_no_info

    return answer, used_rag


# ── Основная логика обработки вопроса ─────────────────────────────────────────
async def process_question(question: str, lang: str) -> tuple[str, bool]:
    """
    Полный пайплайн:
    1. Перевод на русский (если kk с казахскими буквами, или en)
    2. RAG-поиск
    3. GPT-ответ на языке пользователя
    Возвращает (ответ, used_rag).
    """
    # kk — переводим ТОЛЬКО если есть казахские буквы (ә,ғ,қ,ң,ө,ұ,ү,һ,і).
    #   Если пользователь написал по-русски при kk-интерфейсе — не переводим.
    # en — всегда переводим.
    rag_question = question
    needs_translation = (
        lang == "en"
        or (lang == "kk" and any(ch in _KAZAKH_CHARS for ch in question))
    )
    if needs_translation:
        rag_question = await translate_to_russian(question, lang)

    context = await get_rag_context(rag_question)

    if context:
        logger.info(f"[RAG] Контекст для GPT ({len(context)} симв.): {context[:300]}...")
    else:
        logger.warning(
            f"[RAG] Контекст пустой — поиск не нашёл релевантных чанков "
            f"для: '{rag_question[:80]}'"
        )

    answer, used_rag = await generate_answer(question, lang, context)
    return answer, used_rag


# ── Telegram handlers ─────────────────────────────────────────────────────────
router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    if user_id not in _greeted_users:
        _greeted_users.add(user_id)
        await message.answer(T["greet"], parse_mode="HTML")
        await message.answer(T["choose_lang"], parse_mode="HTML", reply_markup=lang_keyboard())
    else:
        await message.answer(T["greet_short"], parse_mode="HTML", reply_markup=lang_keyboard())


@router.message(StateFilter(None), F.text.in_(LANGS))
async def select_language(message: Message, state: FSMContext) -> None:
    lang = LANGS[message.text]
    await state.set_data({"lang": lang})
    await state.set_state(Chat.waiting)
    await message.answer(T["ask_prompt"][lang], parse_mode="HTML", reply_markup=chat_keyboard(lang))


@router.message(Chat.waiting)
async def handle_message(message: Message, state: FSMContext) -> None:
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

    # Rate limit — защита от спама
    if _is_rate_limited(message.from_user.id):
        logger.warning(f"[RateLimit] user_id={message.from_user.id} превысил лимит")
        await message.answer(T["rate_limit"][lang])
        return

    await message.answer(T["typing"][lang], parse_mode="HTML")
    try:
        answer, used_rag = await process_question(text, lang)
        rag_status = "RAG=YES" if used_rag else "RAG=NO"
        logger.info(f"[Answer] user={message.from_user.id} lang={lang} {rag_status}")
        await message.answer(answer, parse_mode="HTML")
    except Exception as e:
        logger.error(f"[Bot] Ошибка обработки: {e}", exc_info=True)
        await message.answer(T["error"][lang])

    await message.answer(T["ask_more"][lang], reply_markup=chat_keyboard(lang))


@router.message(StateFilter(None))
async def fallback(message: Message, state: FSMContext) -> None:
    await message.answer(T["greet_short"], parse_mode="HTML", reply_markup=lang_keyboard())


# ── Запуск ────────────────────────────────────────────────────────────────────
async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp  = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
