# -*- coding: utf-8 -*-
"""
logging_setup.py — Настройка логирования для RAG-сервиса.

Использование:
  from app.logging_setup import setup_logging
  setup_logging(log_name="indexing")   # → logs/indexing.log (с ежедневной ротацией)
  setup_logging(log_name="api")        # → logs/api.log

Логи пишутся одновременно в файл и в stdout.
Файлы хранятся в rag_service/logs/ (создаётся автоматически).

Ротация:
  TimedRotatingFileHandler с when="midnight", backupCount=30.
  Хранит логи 30 дней; старые файлы удаляются автоматически.
  Имя ротированного файла: api.log.2026-04-15 (суффикс добавляет handler).
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Папка для логов — rag_service/logs/
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Сколько дней хранить ротированные файлы
_LOG_BACKUP_COUNT = 30


def setup_logging(log_name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    """
    Настроить логирование: stdout + файл с ежедневной ротацией.

    Args:
        log_name: Имя лог-файла без расширения.
                  Пример: "indexing" → logs/indexing.log
                  Ротированные копии: logs/indexing.log.2026-04-15
        level:    Уровень логирования (по умолчанию INFO).

    Returns:
        Root logger, настроенный на оба хендлера.

    Почему TimedRotatingFileHandler, а не FileHandler:
        FileHandler создаёт новый файл при каждом рестарте (timestamp в имени).
        За месяц работы накапливаются десятки файлов без автоматической очистки.
        TimedRotatingFileHandler ротирует по midnight, держит один активный файл
        и автоматически удаляет старые (backupCount=30).
    """
    LOGS_DIR.mkdir(exist_ok=True)

    log_file = LOGS_DIR / f"{log_name}.log"

    # Формат: время [уровень] модуль: сообщение
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Очищаем существующие хендлеры чтобы не дублировать вывод
    if root.handlers:
        root.handlers.clear()

    # ── Хендлер 1: файл с ротацией ────────────────────────────
    # when="midnight" — новый файл каждую ночь в 00:00.
    # backupCount=30  — хранит 30 дней, старые удаляет автоматически.
    # encoding="utf-8" — важно для кириллицы в сообщениях.
    # delay=True      — не открывает файл пока нет первой записи
    #                   (безопасно при быстрых тестовых запусках).
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # ── Хендлер 2: stdout ─────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    # Заглушаем слишком verbose библиотеки
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root.info(
        f"Logging started → {log_file} "
        f"(daily rotation, {_LOG_BACKUP_COUNT} days retention)"
    )
    return root
