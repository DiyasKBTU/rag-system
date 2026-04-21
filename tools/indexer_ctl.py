#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
indexer_ctl.py — Управление индексацией через терминал.

Команды:
  python tools/indexer_ctl.py start                     — запустить переиндексацию сейчас
  python tools/indexer_ctl.py status                    — показать статус
  python tools/indexer_ctl.py schedule "15.06 03:00"    — запланировать на дату+время
  python tools/indexer_ctl.py schedule "03:00"          — запланировать на сегодня/завтра
  python tools/indexer_ctl.py unschedule                — отменить расписание

Форматы даты:
  "HH:MM"            — сегодня в это время (если прошло — завтра)
  "DD.MM HH:MM"      — конкретный день текущего года
  "DD.MM.YYYY HH:MM" — конкретный день с годом

Требования:
  RAG-сервис должен быть запущен: cd rag_service && python run_api.py
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ── Загрузка .env ─────────────────────────────────────────────────────────────
_env_file = Path(__file__).resolve().parent.parent / "rag_service" / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

RAG_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
API_KEY = os.environ.get("API_SECRET_KEY") or os.environ.get("RAG_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY}


def _check_service():
    """Проверяет что RAG-сервис доступен."""
    try:
        r = requests.get(f"{RAG_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception:
        print("❌ RAG-сервис недоступен.")
        print(f"   Убедись что запущен: cd rag_service && python run_api.py")
        sys.exit(1)


def cmd_start():
    """Немедленно запустить горячую переиндексацию в фоне."""
    _check_service()
    print("⏳ Запускаю горячую переиндексацию...")
    print("   Бот продолжает работать. Используй 'status' для мониторинга.\n")

    try:
        r = requests.post(
            f"{RAG_URL}/index",
            json={"background": True},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code == 409:
            print(f"⚠️  Индексация уже идёт. Проверь статус командой: python tools/indexer_ctl.py status")
            sys.exit(1)
        r.raise_for_status()
        data = r.json()
        print(f"✅ {data.get('message', 'Запущено')}")
    except requests.HTTPError as e:
        print(f"❌ Ошибка: {e.response.text if e.response else e}")
        sys.exit(1)


def cmd_status():
    """Показать текущий статус индексации."""
    _check_service()

    try:
        r = requests.get(f"{RAG_URL}/index/status", headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("СТАТУС ИНДЕКСАЦИИ")
    print("=" * 50)

    in_progress = data.get("in_progress", False)
    print(f"В процессе:         {'да ⏳' if in_progress else 'нет ✅'}")
    if in_progress and data.get("started_at"):
        print(f"Запущено:           {data['started_at']}")

    print(f"Активная коллекция: {data.get('active_collection', '?')}")
    print(f"Записей в Qdrant:   {data.get('total_chunks', '?')}")

    schedule = data.get("schedule")
    if schedule and not schedule.get("fired"):
        human = schedule.get("scheduled_datetime_human") or schedule.get("scheduled_datetime", "?")
        print(f"\n📅 Запланировано:    {human}")
        print(f"   Отменить:        python tools/indexer_ctl.py unschedule")
    elif schedule and schedule.get("fired"):
        print(f"\n📅 Расписание выполнено: {schedule.get('fired_at', '?')}")

    last = data.get("last_result")
    if last:
        print(f"\n📋 Последняя индексация:")
        print(f"   Статус:   {last.get('status', '?')}")
        if last.get("swapped_to"):
            print(f"   Коллекция: {last['swapped_to']} ({last.get('swap_type', '?')})")
        print(f"   Страниц:  {last.get('pages_indexed', '?')} (пропущено: {last.get('pages_skipped', '?')})")
        print(f"   Чанков:   {last.get('chunks', '?')}")
        print(f"   Вопросов: {last.get('questions', '?')}")
        if last.get("error"):
            print(f"   Ошибка:   {last['error']}")

    print("=" * 50)


def cmd_schedule(dt_str: str):
    """Запланировать переиндексацию на указанную дату/время."""
    _check_service()

    try:
        r = requests.post(
            f"{RAG_URL}/index/schedule",
            json={"datetime": dt_str},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code == 409:
            err = r.json().get("detail", "")
            print(f"⚠️  {err}")
            print(f"   Отменить текущее: python tools/indexer_ctl.py unschedule")
            sys.exit(1)
        if r.status_code == 400:
            print(f"❌ Неверный формат: {r.json().get('detail', '')}")
            print(f"\nПоддерживаемые форматы:")
            print(f"  \"HH:MM\"            — сегодня/завтра в это время")
            print(f"  \"DD.MM HH:MM\"      — конкретный день текущего года")
            print(f"  \"DD.MM.YYYY HH:MM\" — конкретный день с годом")
            sys.exit(1)
        r.raise_for_status()
        data = r.json()
        print(f"✅ {data.get('message', 'Запланировано')}")
    except requests.HTTPError as e:
        print(f"❌ Ошибка: {e.response.text if e.response else e}")
        sys.exit(1)


def cmd_unschedule():
    """Отменить запланированную переиндексацию."""
    _check_service()

    try:
        r = requests.delete(f"{RAG_URL}/index/schedule", headers=HEADERS, timeout=10)
        if r.status_code == 404:
            print("ℹ️  Нет активного расписания.")
            sys.exit(0)
        r.raise_for_status()
        data = r.json()
        print(f"✅ {data.get('message', 'Расписание отменено')}")
    except requests.HTTPError as e:
        print(f"❌ Ошибка: {e.response.text if e.response else e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Управление горячей переиндексацией RAG-сервиса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/indexer_ctl.py start
  python tools/indexer_ctl.py status
  python tools/indexer_ctl.py schedule "03:00"
  python tools/indexer_ctl.py schedule "15.06 03:00"
  python tools/indexer_ctl.py schedule "15.06.2025 03:00"
  python tools/indexer_ctl.py unschedule
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start",       help="Запустить переиндексацию немедленно")
    subparsers.add_parser("status",      help="Показать статус индексации")
    subparsers.add_parser("unschedule",  help="Отменить запланированную переиндексацию")

    schedule_p = subparsers.add_parser("schedule", help="Запланировать переиндексацию")
    schedule_p.add_argument(
        "datetime",
        help='Дата и время: "HH:MM", "DD.MM HH:MM" или "DD.MM.YYYY HH:MM"',
    )

    args = parser.parse_args()

    if args.command == "start":
        cmd_start()
    elif args.command == "status":
        cmd_status()
    elif args.command == "schedule":
        cmd_schedule(args.datetime)
    elif args.command == "unschedule":
        cmd_unschedule()


if __name__ == "__main__":
    main()
