# -*- coding: utf-8 -*-
"""
test_speed.py — Тест скорости RAG + OpenAI цепочки.

Запуск:
    python test_speed.py
    python test_speed.py "Как поступить в КАИУ?"

Что измеряется:
    1. Время поиска в RAG (FastAPI + Qdrant)
    2. Время ответа OpenAI
    3. Общее время (то что видит пользователь в Telegram)
"""

import sys
import os
import time
import requests
from pathlib import Path
from openai import OpenAI

# =============================================================================
# ⚙️  ЗАГРУЗКА КЛЮЧЕЙ ИЗ .env
# =============================================================================

# Ищем .env в папке rag_service/ рядом с этим файлом
_env_path = Path(__file__).parent / "rag_service" / ".env"

if _env_path.exists():
    # Читаем .env вручную (без доп. зависимостей)
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())
    print(f"✅ .env загружен: {_env_path}")
else:
    print(f"⚠️  .env не найден по пути: {_env_path}")
    print("   Убедись что файл rag_service/.env существует")

# Ключи берутся из .env автоматически
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAG_API_KEY    = os.environ.get("API_SECRET_KEY", "")

# 🌐 Адрес RAG сервиса (обычно не нужно менять)
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8001")

# 🤖 Модель OpenAI (gpt-4o-mini — быстрее и дешевле, gpt-4o — умнее)
OPENAI_MODEL = "gpt-4.1-mini"

# =============================================================================
# 📝 ПРОМПТ — инструкция для модели
# =============================================================================

SYSTEM_PROMPT = """Ты — дружелюбный помощник приёмной комиссии университета КАИУ (caiu.edu.kz).

Тебе будет дан контекст из базы знаний сайта университета и вопрос пользователя.

Правила ответа:
1. Отвечай своими словами, живо и по-человечески — не копируй текст из контекста дословно.
2. Используй только ту информацию, которая есть в контексте. Не придумывай и не домысливай.
3. Если в контексте есть ссылка на страницу сайта — упомяни её в конце ответа (например: "Подробнее читай здесь: [ссылка]").
4. Если контекст не содержит ответа на вопрос — честно скажи об этом и предложи позвонить в приёмную комиссию или зайти на сайт caiu.edu.kz.
5. Отвечай на том языке, на котором задан вопрос (русский или казахский).
6. Будь кратким: 2-4 предложения достаточно, если только вопрос не требует подробного объяснения.

Контекст из базы знаний:
{context}"""

# =============================================================================
# 🚀 КОД — дальше менять не нужно
# =============================================================================

client = OpenAI(api_key=OPENAI_API_KEY)


def search_rag(question: str) -> tuple[str, float, int]:
    """
    Ищет контекст в RAG сервисе.
    Возвращает: (контекст, время_мс, кол-во_найденных_чанков)
    """
    start = time.perf_counter()
    try:
        response = requests.post(
            url=f"{RAG_URL}/search",
            headers={"X-API-Key": RAG_API_KEY, "Content-Type": "application/json"},
            json={"question": question, "format_as_context": True},
            timeout=10.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            return data.get("context", ""), elapsed_ms, data.get("total_found", 0)
        elif response.status_code == 401:
            print(f"  ❌ RAG ошибка 401 — неверный RAG_API_KEY")
            return "", elapsed_ms, 0
        else:
            print(f"  ❌ RAG ошибка {response.status_code}: {response.text[:100]}")
            return "", elapsed_ms, 0

    except requests.exceptions.ConnectionError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ❌ RAG сервис недоступен — запусти: python run_api.py")
        return "", elapsed_ms, 0
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ❌ RAG таймаут (>10 сек)")
        return "", elapsed_ms, 0


def ask_openai(question: str, context: str) -> tuple[str, float]:
    """
    Отправляет вопрос + контекст в OpenAI.
    Возвращает: (ответ, время_мс)
    """
    system = SYSTEM_PROMPT.format(
        context=context if context else "Контекст не найден."
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.5,
        max_tokens=500,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = response.choices[0].message.content
    return answer, elapsed_ms


# Фразы, которые модель пишет когда НЕ нашла ответ в контексте
_FALLBACK_PHRASES = [
    "информация отсутствует",
    "не содержит",
    "не найдена",
    "нет информации",
    "обратитесь в приёмную",
    "обратитесь в приемную",
    "позвоните в",
    "зайдите на сайт",
    "нет данных",
    "не могу ответить",
    "информацией не располагаю",
]


def detect_rag_usage(answer: str, context: str) -> tuple[str, str]:
    """
    Определяет, использовал ли GPT контекст из RAG при ответе.

    Три возможных статуса:
      🟢 RAG  — контекст найден, модель явно опирается на него
      🟡 FALLBACK — контекст найден, но модель всё равно не дала конкретный ответ
      🔴 БЕЗ RAG — RAG вернул пустой контекст, модель отвечала вслепую

    Логика:
      1. Если context пустой → 🔴 (RAG не участвовал вообще)
      2. Если в ответе есть фраза-заглушка → 🟡 (RAG нашёл, но не помог)
      3. Иначе считаем пересечение значимых слов контекста и ответа:
         - ≥20% слов ответа встречаются в контексте → 🟢 уверенно с RAG
         - 8–19% → 🟢 вероятно с RAG
         - <8%   → 🟡 модель могла ответить из общих знаний
    """
    if not context:
        return "🔴", "БЕЗ RAG — контекст не был найден в базе"

    answer_lower = answer.lower()

    # Проверка на фразу-заглушку ("информации нет")
    if any(phrase in answer_lower for phrase in _FALLBACK_PHRASES):
        return "🟡", "FALLBACK — RAG нашёл контекст, но модель не использовала его"

    # Подсчёт пересечения значимых слов (длиннее 4 букв, без стоп-слов)
    stop = {"этого", "также", "более", "после", "можно", "нужно", "будет",
            "чтобы", "через", "когда", "здесь", "такие", "этому", "всего"}
    ctx_words  = {w for w in context.lower().split()     if len(w) > 4 and w not in stop}
    ans_words  = {w for w in answer_lower.split()        if len(w) > 4 and w not in stop}

    if not ans_words:
        return "🟡", "Ответ слишком короткий для анализа"

    overlap = ctx_words & ans_words
    ratio   = len(overlap) / len(ans_words)

    if ratio >= 0.20:
        return "🟢", f"С RAG — {ratio:.0%} слов ответа из контекста ({len(overlap)} совпадений)"
    elif ratio >= 0.08:
        return "🟢", f"Вероятно с RAG — {ratio:.0%} слов ответа из контекста ({len(overlap)} совпадений)"
    else:
        return "🟡", f"Слабое пересечение {ratio:.0%} — возможно модель ответила из общих знаний"


def run_test(question: str):
    """Запускает полный тест для одного вопроса."""
    print("\n" + "=" * 65)
    print(f"❓ Вопрос: {question}")
    print("=" * 65)

    # Шаг 1 — RAG поиск
    print("\n🔍 Шаг 1: RAG поиск...")
    total_start = time.perf_counter()
    context, rag_ms, chunks_found = search_rag(question)

    if context:
        print(f"  ✅ Найдено {chunks_found} чанков за {rag_ms:.0f} мс")
        print(f"  📄 Контекст (первые 200 символов): {context[:200]}...")
    else:
        print(f"  ⚠️  Контекст не найден ({rag_ms:.0f} мс) — ответ будет без RAG")

    # Шаг 2 — OpenAI
    print(f"\n🤖 Шаг 2: Запрос к OpenAI ({OPENAI_MODEL})...")
    try:
        answer, openai_ms = ask_openai(question, context)
    except Exception as e:
        print(f"  ❌ Ошибка OpenAI: {e}")
        print("  Проверь: правильный ли OPENAI_API_KEY и есть ли баланс на счёте")
        return

    total_ms = (time.perf_counter() - total_start) * 1000

    # Результаты
    print(f"\n💬 Ответ бота:")
    print("-" * 65)
    print(answer)
    print("-" * 65)

    # RAG-индикатор
    icon, label = detect_rag_usage(answer, context)
    print(f"\n🧭 Источник ответа: {icon} {label}")

    print(f"\n⏱️  Время:")
    print(f"   RAG поиск:    {rag_ms:>6.0f} мс")
    print(f"   OpenAI:       {openai_ms:>6.0f} мс")
    print(f"   Итого:        {total_ms:>6.0f} мс  ← это видит пользователь")

    # Оценка скорости
    if total_ms < 3000:
        print(f"   Оценка: ✅ Отлично (< 3 сек)")
    elif total_ms < 5000:
        print(f"   Оценка: ⚠️  Нормально (< 5 сек), но можно ускорить")
    else:
        print(f"   Оценка: ❌ Медленно (> 5 сек) — нужна оптимизация")


# =============================================================================
# Тестовые вопросы
# =============================================================================

DEFAULT_QUESTIONS = [
    "Как поступить в КАИУ?",
    "Какие специальности есть в университете?",
    "Сколько стоит обучение?",
]

if __name__ == "__main__":
    print("=" * 65)
    print("  RAG + OpenAI — Тест скорости")
    print("=" * 65)

    # Проверка настроек
    if not OPENAI_API_KEY:
        print("\n❌ OPENAI_API_KEY не найден в rag_service/.env")
        print("   Добавь строку: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not RAG_API_KEY:
        print("\n❌ API_SECRET_KEY не найден в rag_service/.env")
        print("   Добавь строку: API_SECRET_KEY=твой_ключ")
        sys.exit(1)

    print(f"🔑 OpenAI ключ: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
    print(f"🔑 RAG ключ:    {RAG_API_KEY[:6]}...{RAG_API_KEY[-4:]}")

    # Если вопрос передан аргументом — тестируем один вопрос
    if len(sys.argv) > 1:
        run_test(" ".join(sys.argv[1:]))
    else:
        # Иначе прогоняем все тестовые вопросы
        print(f"\nПрогоняем {len(DEFAULT_QUESTIONS)} вопроса(ов)...")
        for q in DEFAULT_QUESTIONS:
            run_test(q)

    print("\n" + "=" * 65)
    print("Тест завершён.")
    print("=" * 65)
