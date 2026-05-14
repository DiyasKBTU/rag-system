#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
load_test.py — Нагрузочное тестирование RAG-сервиса.

Запуск из корня проекта:
    python tools/load_test.py                          # стандартный тест (10 пользователей)
    python tools/load_test.py --users 30               # 30 одновременных пользователей
    python tools/load_test.py --users 50 --rounds 3    # 50 пользователей, 3 раунда
    python tools/load_test.py --url http://server:8001 # тест удалённого сервера

Что делает:
    Симулирует N одновременных пользователей, каждый отправляет вопрос.
    Измеряет: время ответа, процент успешных запросов, кеш hit rate.

Требования:
    RAG-сервис должен быть запущен: cd rag_service && python run_api.py
    Зависимости: pip install httpx
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import httpx
from dotenv import load_dotenv

# Загружаем .env чтобы взять API ключ
load_dotenv(Path(__file__).resolve().parent.parent / "rag_service" / ".env")

RAG_URL     = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
RAG_API_KEY = os.getenv("API_SECRET_KEY", "")

# ── Тестовые вопросы (разные темы → разные чанки, меньше cache hit) ─────────
TEST_QUESTIONS = [
    "Сколько стоит обучение в ЦАИУ?",
    "Какие специальности есть в ЦАИУ?",
    "Какие документы нужны для поступления?",
    "Есть ли общежитие в ЦАИУ?",
    "Какой проходной балл ЕНТ нужен?",
    "Когда начинается приём документов?",
    "Какие факультеты есть в университете?",
    "Сколько студентов учится в ЦАИУ?",
    "Как получить скидку на обучение?",
    "Где находится ЦАИУ, адрес?",
    "Есть ли военная кафедра в ЦАИУ?",
    "Какие гранты доступны в ЦАИУ?",
    "Сколько корпусов у университета?",
    "Как поступить в магистратуру ЦАИУ?",
    "Какие творческие экзамены нужны?",
    "Расскажи об истории университета",
    "Какой номер телефона приёмной комиссии?",
    "Есть ли дистанционное обучение?",
    "Как подать документы онлайн?",
    "Сколько мест в общежитии?",
]


async def single_request(
    client: httpx.AsyncClient,
    question: str,
    user_id: int,
) -> dict:
    """Один запрос к /search. Возвращает метрики."""
    headers = {"X-API-Key": RAG_API_KEY}
    payload = {"question": question, "format_as_context": False}

    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{RAG_URL}/search",
            json=payload,
            headers=headers,
        )
        elapsed = time.perf_counter() - start

        if resp.status_code == 200:
            data = resp.json()
            return {
                "ok": True,
                "user_id": user_id,
                "question": question[:50],
                "elapsed_ms": int(elapsed * 1000),
                "chunks_found": data.get("total_found", 0),
                "search_time_ms": data.get("search_time_ms", 0),
                "status_code": 200,
            }
        else:
            return {
                "ok": False,
                "user_id": user_id,
                "question": question[:50],
                "elapsed_ms": int(elapsed * 1000),
                "status_code": resp.status_code,
                "error": resp.text[:200],
            }

    except httpx.TimeoutException:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "user_id": user_id,
            "question": question[:50],
            "elapsed_ms": int(elapsed * 1000),
            "status_code": 0,
            "error": "TIMEOUT",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "user_id": user_id,
            "question": question[:50],
            "elapsed_ms": int(elapsed * 1000),
            "status_code": 0,
            "error": str(e)[:200],
        }


async def run_round(n_users: int, round_num: int) -> list[dict]:
    """
    Один раунд: N пользователей отправляют вопросы одновременно.
    Возвращает список результатов.
    """
    questions = [
        TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        for i in range(n_users)
    ]

    print(f"\n  Раунд {round_num}: {n_users} одновременных запросов...")

    timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
    limits  = httpx.Limits(max_connections=n_users + 5, max_keepalive_connections=10)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            single_request(client, questions[i], user_id=i + 1)
            for i in range(n_users)
        ]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

    print(f"  Все запросы завершены за {total_elapsed:.1f} сек")
    return list(results)


def print_report(all_results: list[dict], n_users: int, n_rounds: int) -> None:
    """Выводит итоговый отчёт."""
    ok_results  = [r for r in all_results if r["ok"]]
    err_results = [r for r in all_results if not r["ok"]]

    total     = len(all_results)
    ok_count  = len(ok_results)
    err_count = len(err_results)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТА")
    print("=" * 60)
    print(f"Пользователей:   {n_users}")
    print(f"Раундов:         {n_rounds}")
    print(f"Всего запросов:  {total}")
    print(f"Успешных:        {ok_count}  ({100 * ok_count / total:.1f}%)")
    print(f"Ошибок:          {err_count}  ({100 * err_count / total:.1f}%)")

    if ok_results:
        times = [r["elapsed_ms"] for r in ok_results]
        print(f"\n── Время ответа (успешные) ──────────────────")
        print(f"  Минимум:   {min(times)} мс")
        print(f"  Медиана:   {int(median(times))} мс")
        print(f"  Среднее:   {int(mean(times))} мс")
        print(f"  Максимум:  {max(times)} мс")
        if len(times) > 1:
            print(f"  Разброс:   ±{int(stdev(times))} мс")

        # P95 — 95% запросов отвечают быстрее этого значения
        sorted_times = sorted(times)
        p95_idx = int(0.95 * len(sorted_times))
        print(f"  P95:       {sorted_times[p95_idx]} мс")

        chunks_found = [r["chunks_found"] for r in ok_results]
        no_results = sum(1 for c in chunks_found if c == 0)
        print(f"\n── Качество поиска ──────────────────────────")
        print(f"  Нашли чанки:   {ok_count - no_results} запросов")
        print(f"  Пустой ответ:  {no_results} запросов")

    if err_results:
        print(f"\n── Ошибки ───────────────────────────────────")
        error_types: dict = {}
        for r in err_results:
            key = f"HTTP {r['status_code']}: {r.get('error', '')[:50]}"
            error_types[key] = error_types.get(key, 0) + 1
        for err, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {count}× {err}")

    # Вывод вердикта
    print(f"\n── Вердикт ──────────────────────────────────")
    success_rate = ok_count / total if total else 0
    if ok_results:
        p95 = sorted([r["elapsed_ms"] for r in ok_results])[int(0.95 * ok_count)]
    else:
        p95 = 999999

    if success_rate >= 0.99 and p95 <= 5000:
        print("  ✅ ОТЛИЧНО — сервис справляется с нагрузкой")
    elif success_rate >= 0.95 and p95 <= 10000:
        print("  ⚠️  НОРМ — несколько медленных/упавших запросов")
    elif success_rate >= 0.80:
        print("  ⚠️  ПРОБЛЕМЫ — заметные ошибки под нагрузкой")
    else:
        print("  ❌ ПЕРЕГРУЗКА — сервис не справляется")

    print()


async def check_health(url: str) -> bool:
    """Проверить что RAG-сервис запущен."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def main_async(args: argparse.Namespace) -> None:
    global RAG_URL
    if args.url:
        RAG_URL = args.url

    print(f"\n{'=' * 60}")
    print("НАГРУЗОЧНЫЙ ТЕСТ RAG-СЕРВИСА")
    print(f"{'=' * 60}")
    print(f"URL:         {RAG_URL}")
    print(f"Пользователей: {args.users}")
    print(f"Раундов:     {args.rounds}")

    # Проверить что сервис запущен
    print("\nПроверяю доступность сервиса...")
    if not await check_health(RAG_URL):
        print(f"[✗] RAG-сервис недоступен: {RAG_URL}")
        print("    Запустите: cd rag_service && python run_api.py")
        sys.exit(1)
    print("[✓] Сервис доступен")

    if not RAG_API_KEY:
        print("[✗] Не найден API ключ. Убедитесь что rag_service/.env содержит API_SECRET_KEY")
        sys.exit(1)

    all_results: list[dict] = []

    for round_num in range(1, args.rounds + 1):
        results = await run_round(args.users, round_num)
        all_results.extend(results)

        # Пауза между раундами — дать Redis cache накопиться
        if round_num < args.rounds:
            print(f"  Пауза 2 сек перед следующим раундом...")
            await asyncio.sleep(2)

    print_report(all_results, args.users, args.rounds)

    # Сохранить детальный отчёт в файл
    if args.save:
        report_path = Path(__file__).resolve().parent.parent / "load_test_report.json"
        report_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Детальный отчёт сохранён: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Нагрузочный тест RAG-сервиса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/load_test.py                   # 10 пользователей, 1 раунд
  python tools/load_test.py --users 30        # 30 одновременных пользователей
  python tools/load_test.py --users 50 --rounds 3  # 3 раунда по 50 пользователей
  python tools/load_test.py --url http://192.168.1.10:8001  # удалённый сервер
        """,
    )
    parser.add_argument(
        "--users", type=int, default=10,
        help="Число одновременных пользователей (по умолчанию: 10)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Сколько раундов провести (по умолчанию: 1)",
    )
    parser.add_argument(
        "--url", type=str, default="",
        help="URL RAG-сервиса (по умолчанию: из .env или http://localhost:8001)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Сохранить детальный JSON-отчёт в load_test_report.json",
    )
    args = parser.parse_args()

    if args.users < 1 or args.users > 200:
        print("[✗] --users должен быть от 1 до 200")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
