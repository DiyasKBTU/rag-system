# -*- coding: utf-8 -*-
"""
utils.py — Общие утилиты, используемые в нескольких модулях.

Централизует константы и функции, которые раньше были продублированы
в catalog_builder.py, question_generator.py и bot/run.py.
"""

# Казахские буквы которых нет в русском алфавите.
# Используется для определения языка текста и принятия решения о переводе.
KAZAKH_CHARS: frozenset = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def has_kazakh_chars(text: str, min_count: int = 1) -> bool:
    """
    Возвращает True если в тексте есть хотя бы min_count казахских букв.

    Используется в боте чтобы понять: пользователь написал по-казахски
    или по-русски при казахском интерфейсе (тогда переводить не надо).

    Args:
        text:      Проверяемый текст.
        min_count: Минимальное количество казахских букв (по умолчанию 1).

    Example:
        has_kazakh_chars("Сәлем, қалайсыз?")   → True
        has_kazakh_chars("Привет как дела?")    → False
    """
    count = sum(1 for ch in text if ch in KAZAKH_CHARS)
    return count >= min_count


def detect_lang(text: str, kk_threshold: int = 3) -> str:
    """
    Определяет язык текста: 'kk' (казахский) или 'ru' (русский/другой).

    Логика: если в тексте больше kk_threshold казахских символов — казахский.
    Порог 3 подобран эмпирически: нейтрализует случайные казахские буквы
    в русском тексте (е.g. аббревиатуры, имена).

    Args:
        text:         Анализируемый текст.
        kk_threshold: Минимум казахских букв для признания текста казахским.

    Returns:
        'kk' если казахский, 'ru' в остальных случаях.

    Example:
        detect_lang("Бұл мамандық өте маңызды")  → 'kk'
        detect_lang("Это важная специальность")   → 'ru'
    """
    count = sum(1 for ch in text if ch in KAZAKH_CHARS)
    return "kk" if count > kk_threshold else "ru"
