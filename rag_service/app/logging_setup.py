# -*- coding: utf-8 -*-
"""
logging_setup.py — Настройка логирования для RAG-сервиса.

Использование:
  from app.logging_setup import setup_logging
  setup_logging(log_name="indexing")   # → logs/indexing_2026-04-15_14-30.log
  setup_logging(log_name="api")        # → logs/api_2026-04-15_14-30.log

Логи пишутся одновременно в файл и в stdout.
Файлы хранятся в rag_service/logs/ (создаётся автоматически).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Папка для логов — rag_service/logs/
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(log_name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    """
    Настроить логирование: stdout + файл.

    Args:
        log_name: Имя лог-файла без расширения. К нему добавляется дата и время.
                  Пример: "indexing" → logs/indexing_2026-04-15_14-30.log
        level:    Уровень логирования (по умолчанию INFO).

    Returns:
        Root logger, настроенный на оба хендлера.
    """
    LOGS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = LOGS_DIR / f"{log_name}_{timestamp}.log"

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

    # ── Хендлер 1: файл ───────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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

    root.info(f"Logging started → {log_file}")
    return root
