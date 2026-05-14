#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backup.py — Создать резервную копию базы знаний (Qdrant).

Запуск из корня проекта:
    python tools/backup.py

Что делает:
    Просит Qdrant сохранить снимок базы в папку backups/.
    Файл называется по дате: backup_2025-06-15_14-30.snapshot

Когда запускать:
    - Перед переиндексацией (на случай если что-то пойдёт не так)
    - После успешной индексации (сохранить рабочую версию)
    - Раз в неделю / раз в месяц на своё усмотрение

Как восстановить из бэкапа:
    1. Остановить RAG-сервис
    2. Зайти в Qdrant Dashboard: http://localhost:6333/dashboard
    3. Вкладка Collections → ваша коллекция → Snapshots → Upload
    4. Загрузить .snapshot файл из папки backups/
    5. Запустить RAG-сервис обратно

Важно:
    Qdrant должен быть запущен (docker-compose up -d).
    Файл бэкапа создаётся ВНУТРИ Docker-контейнера, затем скачивается
    на ваш компьютер в папку backups/.
"""

import os
import sys
import shutil
import requests
from datetime import datetime
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(__file__), "..", "rag_service"))
sys.path.insert(0, ".")

from app.config import settings
from app.indexer.storage import get_active_collection


# Папка для хранения бэкапов — рядом с корнем проекта
BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"

QDRANT_BASE = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

# После первого hot-swap settings.QDRANT_COLLECTION_NAME — это алиас, а не
# реальная коллекция. get_active_collection() возвращает реальную (blue/green).
COLLECTION  = get_active_collection()


def main():
    print("\n" + "=" * 50)
    print("БЭКАП БАЗЫ ЗНАНИЙ")
    print("=" * 50)
    print(f"\nАктивная коллекция: {COLLECTION}")

    # ── Шаг 1: проверить что Qdrant запущен ──────────────
    print("\n[1/4] Проверяю соединение с Qdrant...")
    try:
        resp = requests.get(f"{QDRANT_BASE}/healthz", timeout=5)
        resp.raise_for_status()
        print("      [✓] Qdrant доступен")
    except Exception:
        print("      [✗] Qdrant недоступен. Запустите: docker-compose up -d")
        sys.exit(1)

    # ── Шаг 2: проверить что коллекция существует ────────
    print(f"\n[2/4] Проверяю коллекцию '{COLLECTION}'...")
    try:
        resp = requests.get(f"{QDRANT_BASE}/collections/{COLLECTION}", timeout=5)
        if resp.status_code == 404:
            print(f"      [✗] Коллекция '{COLLECTION}' не найдена.")
            print("          База пустая — нечего бэкапить.")
            sys.exit(1)
        resp.raise_for_status()
        info = resp.json()
        count = info["result"]["points_count"]
        print(f"      [✓] Коллекция найдена: {count} записей")
    except Exception as e:
        print(f"      [✗] Ошибка: {e}")
        sys.exit(1)

    # ── Шаг 3: создать снапшот внутри Qdrant ─────────────
    print(f"\n[3/4] Создаю снапшот (это может занять 10-30 секунд)...")
    try:
        resp = requests.post(
            f"{QDRANT_BASE}/collections/{COLLECTION}/snapshots",
            timeout=120,
        )
        resp.raise_for_status()
        snapshot_name = resp.json()["result"]["name"]
        print(f"      [✓] Снапшот создан: {snapshot_name}")
    except Exception as e:
        print(f"      [✗] Ошибка при создании снапшота: {e}")
        sys.exit(1)

    # ── Шаг 4: скачать снапшот на диск ───────────────────
    print(f"\n[4/4] Скачиваю файл бэкапа...")
    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = BACKUP_DIR / f"backup_{timestamp}.snapshot"

    try:
        download_url = f"{QDRANT_BASE}/collections/{COLLECTION}/snapshots/{snapshot_name}"
        resp = requests.get(download_url, timeout=120, stream=True)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"      [✓] Сохранён: {save_path}")
        print(f"          Размер: {size_mb:.1f} МБ")
    except Exception as e:
        print(f"      [✗] Ошибка при скачивании: {e}")
        sys.exit(1)

    all_backups = sorted(BACKUP_DIR.glob("backup_*.snapshot"))

    # ── Шаг 5: скопировать JSON-конфиги ──────────────────
    # manual_knowledge.json содержит все ручные описания страниц и факты —
    # их важно хранить вместе со снапшотом (они не попадают в Qdrant snapshot).
    print("\n[5/5] Копирую JSON-конфиги...")

    # Корень проекта — на два уровня выше tools/
    project_root = Path(__file__).resolve().parent.parent
    rag_service_dir = project_root / "rag_service"

    json_configs = [
        project_root / "manual_knowledge.json",
    ]

    backup_configs_dir = BACKUP_DIR / f"configs_{timestamp}"
    backup_configs_dir.mkdir(exist_ok=True)

    for cfg_path in json_configs:
        if cfg_path.exists():
            dest = backup_configs_dir / cfg_path.name
            shutil.copy2(cfg_path, dest)
            print(f"      [✓] {cfg_path.name} → {backup_configs_dir.name}/")
        else:
            print(f"      [–] {cfg_path.name} не найден (пропускаем)")

    print("\n" + "=" * 50)
    print("[✓] БЭКАП УСПЕШНО СОЗДАН")
    print("=" * 50)
    print(f"\nФайл: {save_path.name}")
    print(f"Конфиги: {backup_configs_dir.name}/")
    print(f"Папка: {BACKUP_DIR}")

    if len(all_backups) > 1:
        print(f"\nВсе бэкапы ({len(all_backups)} шт.):")
        for b in all_backups:
            size_mb = b.stat().st_size / (1024 * 1024)
            marker = " ← только что" if b == save_path else ""
            print(f"  {b.name}  ({size_mb:.1f} МБ){marker}")

    print("\nЧтобы восстановить базу из этого файла:")
    print("  Откройте http://localhost:6333/dashboard")
    print(f"  Collections → {COLLECTION} → Snapshots → Upload")
    print(f"  Загрузите файл: {save_path.name}")
    print(f"\nДля восстановления JSON-конфигов:")
    print(f"  Скопируйте файлы из папки {backup_configs_dir.name}/ обратно в проект")


if __name__ == "__main__":
    main()
