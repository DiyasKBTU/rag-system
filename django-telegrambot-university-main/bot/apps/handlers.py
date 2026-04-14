import aiohttp
from decouple import config
from aiogram import Router, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

router = Router()
BACKEND_URL = config("BACKEND_URL")

# ── Языки ────────────────────────────────────────────────────────────────────
LANGS = {
    "Қазақша 🇰🇿": "kk",
    "Русский 🇷🇺": "ru",
    "English 🇬🇧": "en",
}

BACK_LABELS = {
    "kk": "⬅ Артқа",
    "ru": "⬅ Назад",
    "en": "⬅ Back",
}

# ── Тексты ───────────────────────────────────────────────────────────────────
T = {
    "greet": (
        "👋 Қабылдау комиссиясының Telegram ботына қош келдіңіз!\n"
        "Добро пожаловать в Telegram-бот приёмной комиссии!\n"
        "Welcome to the Admissions Committee Telegram bot!"
    ),
    "choose_lang": "🌐 Тілді таңдаңыз / Выберите язык / Choose a language:",
    "ask_prompt": {
        "kk": "✍️ Сұрағыңызды жазыңыз:",
        "ru": "✍️ Напишите ваш вопрос:",
        "en": "✍️ Write your question:",
    },
    "typing": {
        "kk": "⏳ Жауап дайындалып жатыр...",
        "ru": "⏳ Готовлю ответ...",
        "en": "⏳ Preparing an answer...",
    },
    "ask_more": {
        "kk": "✍️ Тағы сұрақ жазыңыз немесе артқа қайтыңыз:",
        "ru": "✍️ Задайте ещё вопрос или вернитесь назад:",
        "en": "✍️ Ask another question or go back:",
    },
    "error": {
        "kk": "⚠️ Қате орын алды. Кейінірек қайталап көріңіз.",
        "ru": "⚠️ Произошла ошибка. Попробуйте позже.",
        "en": "⚠️ An error occurred. Please try again later.",
    },
}

RAG_LABEL = {
    "ru": {True: "🟢 <i>[из базы знаний]</i>", False: "🔴 <i>[база знаний недоступна]</i>"},
    "kk": {True: "🟢 <i>[білім базасынан]</i>",  False: "🔴 <i>[база қол жетімді емес]</i>"},
    "en": {True: "🟢 <i>[from knowledge base]</i>", False: "🔴 <i>[knowledge base unavailable]</i>"},
}

# ── FSM ──────────────────────────────────────────────────────────────────────
class Chat(StatesGroup):
    waiting = State()

# ── Клавиатуры ───────────────────────────────────────────────────────────────
def lang_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=label)] for label in LANGS],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

def chat_keyboard(lang: str):
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_LABELS[lang])]],
        resize_keyboard=True,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def norm_lang(lang: str) -> str:
    return lang if lang in ("kk", "ru", "en") else "ru"

def is_back(text: str) -> bool:
    return text in BACK_LABELS.values()

async def ask_backend(question: str, lang: str) -> tuple[str, bool]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BACKEND_URL}/api/ask/",
            json={"question": question, "lang": lang},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return (data.get("answer") or "").strip(), bool(data.get("used_rag", False))

# ── Хендлеры ─────────────────────────────────────────────────────────────────
@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(T["greet"])
    await message.answer(T["choose_lang"], reply_markup=lang_keyboard())


@router.message(StateFilter(None), F.text.in_(LANGS))
async def select_language(message: Message, state: FSMContext):
    lang = LANGS[message.text]
    await state.set_data({"lang": lang})
    await state.set_state(Chat.waiting)
    await message.answer(T["ask_prompt"][lang], reply_markup=chat_keyboard(lang))


@router.message(Chat.waiting)
async def handle_message(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    data = await state.get_data()
    lang = norm_lang(data.get("lang", "ru"))

    # Назад → выбор языка
    if is_back(text):
        await state.clear()
        await message.answer(T["choose_lang"], reply_markup=lang_keyboard())
        return

    # Вопрос → GPT
    await message.answer(T["typing"][lang])
    try:
        answer, used_rag = await ask_backend(text, lang)
        rag_label = RAG_LABEL.get(lang, RAG_LABEL["ru"])[used_rag]
        await message.answer(f"{answer}\n\n{rag_label}", parse_mode="HTML")
    except Exception as e:
        print("ERROR:", e)
        await message.answer(T["error"][lang])

    await message.answer(T["ask_more"][lang], reply_markup=chat_keyboard(lang))


# Написал что-то вне состояния → возвращаем к выбору языка
@router.message(StateFilter(None))
async def fallback(message: Message, state: FSMContext):
    await message.answer(T["choose_lang"], reply_markup=lang_keyboard())
