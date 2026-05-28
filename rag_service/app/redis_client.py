# -*- coding: utf-8 -*-
"""
redis_client.py — Утилита подключения к Redis.

Принцип: graceful degradation.
  Если Redis недоступен (не запущен, неверный хост) —
  все кеши просто не работают, бот/RAG продолжают работать как раньше.
  Ошибка логируется один раз, повторные попытки не делаются до рестарта.

Использование:
  from app.redis_client import get_redis, is_redis_available

  r = get_redis()
  if r:
      r.set("key", "value", ex=86400)
      val = r.get("key")

  # Если нужен только статус без объекта клиента:
  if is_redis_available():
      ...

Почему sync redis, а не aioredis:
  RAG-сервис использует sync FastAPI endpoints (не async def).
  Sync redis работает нативно в FastAPI threadpool без asyncio.run().
  Для бота (async) используем отдельный async redis клиент (bot/run.py).
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Настройки Redis (совпадают с docker-compose.yml)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0

# TTL по умолчанию для кешей — 30 дней
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 2_592_000

# После неудачного подключения повторяем попытку не чаще раза в 5 минут.
# Это позволяет подхватить Redis если он поднялся после старта сервиса.
_RETRY_INTERVAL = 300  # секунд

_redis_client = None
_redis_available: Optional[bool] = None  # None = ещё не проверяли
_last_connect_attempt: float = 0.0       # monotonic timestamp последней попытки


def get_redis():
    """
    Возвращает синхронный Redis клиент или None если Redis недоступен.

    При первом вызове пытается подключиться. После неудачи повторяет попытку
    не чаще раза в _RETRY_INTERVAL секунд — это позволяет подхватить Redis,
    если он запустился позже сервиса (например, Docker поднимается медленнее).

    Если Redis упал после успешного подключения — отдельные операции
    будут бросать исключения, которые должны перехватываться в месте использования.
    """
    global _redis_client, _redis_available, _last_connect_attempt

    # Быстрый путь — уже подключены
    if _redis_client is not None:
        return _redis_client

    # Недоступен, но ещё не время повторять
    if _redis_available is False:
        if time.monotonic() - _last_connect_attempt < _RETRY_INTERVAL:
            return None
        # Интервал истёк — сбрасываем состояние и пробуем снова
        logger.info("[Redis] Повторная попытка подключения...")

    # Первый вызов или повторная попытка после истечения интервала
    _last_connect_attempt = time.monotonic()
    try:
        import redis
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False,  # работаем с bytes напрямую (для векторов)
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()  # проверяем что Redis отвечает
        _redis_client = client
        _redis_available = True
        logger.info(f"[Redis] Подключён: {REDIS_HOST}:{REDIS_PORT}/db{REDIS_DB}")
        return _redis_client

    except Exception as e:
        _redis_available = False
        logger.warning(
            f"[Redis] Недоступен ({e}). Кеши будут работать только in-memory. "
            f"Следующая попытка через {_RETRY_INTERVAL // 60} мин. "
            f"Убедись что Docker запущен: docker-compose up -d"
        )
        return None


def is_redis_available() -> bool:
    """
    True если Redis успешно подключён, False — если недоступен или ещё не проверяли.

    Семантика:
      - До первого вызова get_redis()      → False (статус неизвестен)
      - После успешного подключения        → True
      - После неудачной попытки подключения → False (повторных попыток не делается)

    Зачем функция, а не переменная:
      _redis_available обновляется внутри get_redis() — переменная на момент
      импорта всегда False. Функция читает АКТУАЛЬНОЕ значение в момент вызова.
    """
    return _redis_available is True
