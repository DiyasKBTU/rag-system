#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
restore.py — Восстановить базу знаний из .snapshot файла.

Запуск из корня проекта:
    python tools/restore.py                         # выбрать последний бэкап
    python tools/restore.py backups/backup_2025-06-15_14-30.snapshot

Что делает:
    1. Загружает .snapshot файл в Qdrant через REST API
    2. Восстанавливает коллекцию под именем из настроек
    3. Если используется hot-swap (blue/green) — восстанавливает в активную коллекцию

Когда использовать:
    - Перенос на новый сервер / другой компьютер
    - Откат после неудачной индексации
    - Восстановление после docker-compose down -v

Требования:
    Qdrant должен быть запущен: docker-compose up -d
    RAG-сервис НЕ нужно останавливать (Qdrant принимает snapshot параллельно).
"""

import os
import sys
import requests
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.config import settings
from app.indexer.storage import get_active_collection, ensure_collection_exists


BACKUP_DIR  = Path(__file__).resolve().parent.parent / "backups"
QDRANT_BASE = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


def _find_snapshot_file(arg: str | None) -> Path:
    """
    Определяет файл для восстановления:
      - Если передан аргумент — использует его как путь к файлу.
      - Если не передан — ищет последний backup_*.snapshot в папке backups/.
    """
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            # Пробуем относительно корня проекта
            p = Path(__file__).resolve().parent.parent / arg
        if not p.exists():
            print(f"[✗] Файл не найден: {p}")
            sys.exit(1)
        return p

    # Автовыбор последнего бэкапа
    snapshots = sorted(BACKUP_DIR.glob("backup_*.snapshot"))
    if not snapshots:
        print(f"[✗] Файлы бэкапа не найдены в {BACKUP_DIR}")
        print("    Сначала создайте бэкап: python tools/backup.py")
        sys.exit(1)
    return snapshots[-1]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "=" * 55)
    print("ВОССТАНОВЛЕНИЕ БАЗЫ ЗНАНИЙ ИЗ БЭКАПА")
    print("=" * 55)

    # ── Шаг 0: найти файл бэкапа ─────────────────────────────────
    snapshot_path = _find_snapshot_file(arg)
    size_mb = snapshot_path.stat().st_size / (1024 * 1024)
    print(f"\nФайл: {snapshot_path.name}  ({size_mb:.1f} МБ)")

    # Определяем целевую коллекцию
    # get_active_collection() вернёт актуальное имя (с учётом blue/green)
    target_collection = get_active_collection()
    print(f"Целевая коллекция: {target_collection}")

    print("\nВНИМАНИЕ: существующие данные в коллекции будут перезаписаны.")
    answer = input("Продолжить? (да/нет): ").strip().lower()
    if answer not in ("да", "yes", "y", "д"):
        print("Отменено.")
        sys.exit(0)

    # ── Шаг 1: проверить Qdrant ────────────────────────────────────
    print("\n[1/4] Проверяю соединение с Qdrant...")
    try:
        resp = requests.get(f"{QDRANT_BASE}/healthz", timeout=5)
        resp.raise_for_status()
        print("      [✓] Qdrant доступен")
    except Exception:
        print("      [✗] Qdrant недоступен. Запустите: docker-compose up -d")
        sys.exit(1)

    # ── Шаг 2: убедиться что целевая коллекция существует ──────────
    print(f"\n[2/4] Проверяю коллекцию '{target_collection}'...")
    try:
        ensure_collection_exists(collection_name=target_collection)
        print(f"      [✓] Коллекция готова")
    except Exception as e:
        print(f"      [✗] Ошибка: {e}")
        sys.exit(1)

    # ── Шаг 3: загрузить snapshot в Qdrant ─────────────────────────
    print(f"\n[3/4] Загружаю snapshot ({size_mb:.1f} МБ)...")
    print("      Это может занять 1-5 минут...")
    try:
        upload_url = (
            f"{QDRANT_BASE}/collections/{target_collection}"
            f"/snapshots/upload?priority=snapshot"
        )
        with open(snapshot_path, "rb") as f:
            resp = requests.post(
                upload_url,
                files={"snapshot": (snapshot_path.name, f, "application/octet-stream")},
                timeout=600,  # 10 минут — большие базы могут восстанавливаться долго
            )

        if resp.status_code not in (200, 201):
            print(f"      [✗] Qdrant вернул {resp.status_code}: {resp.text[:300]}")
            sys.exit(1)

        print("      [✓] Snapshot загружен")
    except Exception as e:
        print(f"      [✗] Ошибка загрузки: {e}")
        sys.exit(1)

    # ── Шаг 4: проверить результат ─────────────────────────────────
    print(f"\n[4/4] Проверяю восстановленные данные...")
    try:
        resp = requests.get(
            f"{QDRANT_BASE}/collections/{target_collection}", timeout=10
        )
        resp.raise_for_status()
        info = resp.json()
        count = info["result"]["points_count"]
        print(f"      [✓] Записей в базе: {count}")
    except Exception as e:
        print(f"      [!] Не удалось проверить количество записей: {e}")
        count = "?"

    # ── Итог ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[✓] ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО")
    print("=" * 55)
    print(f"\nКоллекция: {target_collection}")
    print(f"Записей:   {count}")
    print("\nМожно запускать RAG-сервис и бота.")
    print("Переиндексация НЕ нужна.")


if __name__ == "__main__":
    main()
