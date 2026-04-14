# -*- coding: utf-8 -*-
"""
services/rag_client.py — Django-обёртка над RAG FastAPI сервисом.

Приоритет в views.py:
    1. get_rag_context(question) — семантический поиск в Qdrant (rag_service)
    2. Если пусто или сервис недоступен → build_knowledge_base() (Google Docs + сайт)

Настройки читаются из Django settings:
    RAG_URL      — адрес FastAPI сервера (default: http://localhost:8001)
    RAG_API_KEY  — совпадает с API_SECRET_KEY в .env rag_service
    RAG_TIMEOUT  — таймаут в секундах (default: 5.0)
"""

import logging
import requests
from typing import Optional
from django.conf import settings

logger = logging.getLogger(__name__)


def get_rag_context(question: str) -> str:
    """
    Запрашивает семантически релевантные чанки из Qdrant через FastAPI.

    Возвращает:
        str — отформатированный контекст для ChatGPT, или "" если:
              — RAG-сервис недоступен
              — по вопросу ничего не найдено
              — API-ключ не задан
    """
    if not question or not question.strip():
        return ""

    rag_url: str = getattr(settings, "RAG_URL", "http://localhost:8001")
    api_key: str = getattr(settings, "RAG_API_KEY", "")
    timeout: float = getattr(settings, "RAG_TIMEOUT", 5.0)

    if not api_key:
        logger.warning("[RAG] RAG_API_KEY не задан в settings — пропускаем Qdrant-поиск")
        return ""

    payload = {
        "question": question.strip(),
        "format_as_context": True,
    }

    try:
        resp = requests.post(
            url=f"{rag_url}/search",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            context: str = data.get("context", "")
            total: int = data.get("total_found", 0)
            ms: int = data.get("search_time_ms", 0)

            if context:
                logger.info(f"[RAG] ✅ Найдено {total} чанков за {ms}мс")
                logger.debug(f"[RAG] Первые 200 симв: {context[:200]}")
            else:
                logger.info("[RAG] ⚠️ Ничего не найдено по вопросу")

            return context

        elif resp.status_code == 401:
            logger.error("[RAG] ❌ Неверный API ключ (401) — проверь RAG_API_KEY в .env")
            return ""

        else:
            logger.error(f"[RAG] ❌ Сервер вернул {resp.status_code}: {resp.text[:200]}")
            return ""

    except requests.exceptions.ConnectionError:
        logger.warning(
            f"[RAG] ⚠️ Сервис недоступен (connection refused) на {rag_url} — "
            "переходим на Google Docs / сайт"
        )
        return ""

    except requests.exceptions.Timeout:
        logger.warning(
            f"[RAG] ⚠️ Таймаут ({timeout}с) — переходим на Google Docs / сайт"
        )
        return ""

    except Exception as e:
        logger.error(f"[RAG] ❌ Неожиданная ошибка: {e}", exc_info=True)
        return ""


def check_rag_health() -> bool:
    """Быстрая проверка доступности RAG-сервиса (не требует API-ключа)."""
    rag_url: str = getattr(settings, "RAG_URL", "http://localhost:8001")
    try:
        resp = requests.get(f"{rag_url}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False
