#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/load_test_realistic.py — Реалистичный нагрузочный тест RAG-сервиса.

Имитирует реальных пользователей: случайные задержки, постепенное нарастание
нагрузки, несколько волн. Показывает время ответа, процент ошибок, пропускную
способность.

Запуск (из папки rag_service/):
    python ../tools/load_test_realistic.py

Параметры:
    --url       URL RAG-сервиса (по умолчанию http://localhost:8001)
    --key       API-ключ (берётся из .env если не указан)
    --users     Максимум одновременных пользователей (по умолчанию 25)
    --duration  Длительность теста в секундах (по умолчанию 120)
    --ramp      Время нарастания нагрузки в секундах (по умолчанию 30)

Примеры:
    python ../tools/load_test_realistic.py
    python ../tools/load_test_realistic.py --users 10 --duration 60
    python ../tools/load_test_realistic.py --users 30 --duration 180 --ramp 60
"""

import asyncio
import time
import random
import argparse
import os
import sys
import statistics
from collections import defaultdict
from datetime import datetime

import httpx
from dotenv import load_dotenv

# Загружаем .env если запускаем из rag_service/
load_dotenv()

# ── Реалистичные вопросы абитуриентов ─────────────────────────────────────────
QUESTIONS = [
    # Стоимость
    "сколько стоит обучение в ЦАИУ",
    "какая стоимость обучения на IT специальности",
    "сколько стоит контракт на юридическом факультете",
    "можно ли платить за учёбу в рассрочку",

    # Специальности
    "какие специальности есть в ЦАИУ",
    "есть ли специальность программирование",
    "какие факультеты в университете",
    "на кого можно поступить в ЦАИУ",
    "есть ли медицинские специальности",

    # Поступление
    "какие документы нужны для поступления",
    "когда начинается приём документов",
    "какой проходной балл ЕНТ для поступления",
    "можно ли поступить без ЕНТ",
    "как поступить иностранному гражданину",

    # Гранты и скидки
    "есть ли гранты в ЦАИУ",
    "какие льготы для многодетных семей",
    "есть ли скидки на обучение",
    "как получить грант на обучение",

    # Общежитие
    "есть ли общежитие в ЦАИУ",
    "сколько стоит общежитие",
    "как заселиться в общежитие",

    # Контакты и адрес
    "где находится университет",
    "как доехать до ЦАИУ",
    "номер телефона приёмной комиссии",
    "когда работает приёмная комиссия",

    # Разное
    "есть ли военная кафедра",
    "какие кружки и секции есть в университете",
    "есть ли дистанционное обучение",
    "как перевестись из другого вуза",
]


class Stats:
    """Сбор и отображение статистики."""

    def __init__(self, ramp_duration: int = 30):
        self.response_times: list = []       # (timestamp, seconds) успешных
        self.errors: list = []
        self.status_codes: dict = defaultdict(int)
        self.start_time: float = time.monotonic()
        self.total_requests: int = 0
        self.ramp_duration = ramp_duration   # сек нарастания — для разделения фаз
        self._lock = asyncio.Lock()

    async def record(self, response_time: float, status_code: int, error: str = None):
        async with self._lock:
            self.total_requests += 1
            self.status_codes[status_code] += 1
            ts = time.monotonic() - self.start_time
            if error or status_code >= 400:
                self.errors.append({
                    "ts":     ts,
                    "time":   response_time,
                    "status": status_code,
                    "error":  error or f"HTTP {status_code}",
                })
            else:
                self.response_times.append((ts, response_time))

    def percentile(self, p: float, times: list = None) -> float:
        src = [t for _, t in (times or self.response_times)]
        if not src:
            return 0.0
        s = sorted(src)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def print_live(self, active_users: int):
        """Живая строка прогресса с min/avg/max времени ответа."""
        elapsed = time.monotonic() - self.start_time
        times = [t for _, t in self.response_times]
        ok  = len(times)
        err = len(self.errors)
        avg = statistics.mean(times) if times else 0
        mn  = min(times) if times else 0
        mx  = max(times) if times else 0
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        print(
            f"\r  👥 {active_users:2d} users | "
            f"✅ {ok:4d} ok | "
            f"❌ {err:3d} err | "
            f"⏱  {mn:.1f}s / {avg:.1f}s / {mx:.1f}s (min/avg/max) | "
            f"📈 {rps:.1f} req/s",
            end="", flush=True,
        )

    def _histogram(self, times: list, width: int = 40) -> str:
        """ASCII-гистограмма распределения времени ответа."""
        if not times:
            return "  (нет данных)"
        mn, mx = min(times), max(times)
        if mx == mn:
            return f"  все запросы: {mn:.2f} сек"
        buckets = 8
        step = (mx - mn) / buckets
        counts = [0] * buckets
        for t in times:
            idx = min(int((t - mn) / step), buckets - 1)
            counts[idx] += 1
        max_count = max(counts) or 1
        lines = []
        for i, cnt in enumerate(counts):
            lo = mn + i * step
            hi = lo + step
            bar = "█" * int(cnt / max_count * width)
            lines.append(f"  {lo:5.1f}–{hi:4.1f}s │{bar:<{width}}│ {cnt}")
        return "\n".join(lines)

    def print_final(self):
        """Итоговый отчёт с фазами и гистограммой."""
        elapsed = time.monotonic() - self.start_time
        all_times = [t for _, t in self.response_times]
        ok    = len(all_times)
        err   = len(self.errors)
        total = self.total_requests

        # Разбиваем на фазу нарастания и фазу пика
        ramp_times = [t for ts, t in self.response_times if ts <= self.ramp_duration]
        peak_times = [t for ts, t in self.response_times if ts  > self.ramp_duration]

        print("\n\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТА")
        print("=" * 60)

        print(f"\n📊 Общее:")
        print(f"   Длительность:     {elapsed:.1f} сек")
        print(f"   Всего запросов:   {total}")
        if total:
            print(f"   Успешных:         {ok} ({ok/total*100:.1f}%)")
            print(f"   Ошибок:           {err} ({err/total*100:.1f}%)")
        print(f"   Пропускная спос.: {total/elapsed:.2f} req/s")

        if all_times:
            print(f"\n⏱  Время ответа — всего теста:")
            print(f"   Минимум:   {min(all_times):.2f} сек")
            print(f"   Среднее:   {statistics.mean(all_times):.2f} сек")
            print(f"   Медиана:   {self.percentile(50):.2f} сек")
            print(f"   P90:       {self.percentile(90):.2f} сек  ← 90% запросов быстрее")
            print(f"   P95:       {self.percentile(95):.2f} сек")
            print(f"   P99:       {self.percentile(99):.2f} сек")
            print(f"   Максимум:  {max(all_times):.2f} сек")

        # Сравнение фаз
        if ramp_times and peak_times:
            ramp_avg = statistics.mean(ramp_times)
            peak_avg = statistics.mean(peak_times)
            delta    = peak_avg - ramp_avg
            sign     = "+" if delta >= 0 else ""
            print(f"\n📈 Сравнение фаз (нарастание vs пик):")
            print(f"   Нарастание ({self.ramp_duration}с):  avg {ramp_avg:.2f}s, P95 {self.percentile(95, [(0,t) for t in ramp_times]):.2f}s  ({len(ramp_times)} запросов)")
            print(f"   Пик нагрузки:        avg {peak_avg:.2f}s, P95 {self.percentile(95, [(0,t) for t in peak_times]):.2f}s  ({len(peak_times)} запросов)")
            print(f"   Деградация:          {sign}{delta:.2f}s ({sign}{delta/ramp_avg*100:.1f}%)" if ramp_avg else "")

        # Гистограмма
        if all_times:
            print(f"\n📊 Распределение времени ответа:")
            print(self._histogram(all_times))

        if self.status_codes:
            print(f"\n📋 Коды ответов:")
            for code, count in sorted(self.status_codes.items()):
                label = {
                    200: "✅ OK",
                    401: "🔑 Unauthorized",
                    429: "⚠️  Rate limit",
                    500: "💥 Server error",
                    503: "🚫 Semaphore full (слишком много одновременных)",
                }.get(code, f"   HTTP {code}")
                print(f"   {label}: {count}")

        if self.errors:
            print(f"\n❌ Последние ошибки:")
            for e in self.errors[-5:]:
                print(f"   [{e['status']}] {e['error'][:80]}")

        # Вердикт
        print(f"\n{'=' * 60}")
        error_rate = err / total if total else 0
        p95 = self.percentile(95)
        ramp_avg = statistics.mean(ramp_times) if ramp_times else 0
        peak_avg = statistics.mean(peak_times) if peak_times else 0
        degradation = (peak_avg - ramp_avg) / ramp_avg if ramp_avg else 0

        if error_rate < 0.01 and p95 < 10 and degradation < 0.3:
            verdict = "✅ ОТЛИЧНО — система справляется, деградация минимальная"
        elif error_rate < 0.05 and p95 < 20:
            verdict = "⚠️  НОРМАЛЬНО — небольшие задержки и ошибки при пике"
        elif error_rate < 0.15:
            verdict = "⚠️  ПРЕДЕЛ — заметная деградация, появляются 503"
        else:
            verdict = "❌ ПЕРЕГРУЗКА — система не справляется с нагрузкой"

        print(f"Вердикт: {verdict}")
        print("=" * 60)


async def single_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    question: str,
    stats: Stats,
):
    """Один запрос к RAG /search."""
    start = time.monotonic()
    status = 0
    error = None
    try:
        resp = await client.post(
            f"{url}/search",
            json={"question": question},
            headers={"X-API-Key": api_key},
        )
        status = resp.status_code
        if status != 200:
            error = resp.text[:100]
    except httpx.TimeoutException:
        status = 0
        error = "Timeout"
    except Exception as e:
        status = 0
        error = str(e)[:80]

    elapsed = time.monotonic() - start
    await stats.record(elapsed, status, error)


async def virtual_user(
    user_id: int,
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    stats: Stats,
    stop_event: asyncio.Event,
):
    """
    Виртуальный пользователь: задаёт вопросы с паузами как живой человек.
    Пауза между вопросами: 3–12 секунд (пользователь читает ответ, думает).
    """
    # Небольшой разброс старта чтобы не все стартовали одновременно
    await asyncio.sleep(random.uniform(0, 2))

    while not stop_event.is_set():
        question = random.choice(QUESTIONS)
        await single_request(client, url, api_key, question, stats)

        # Пауза как у реального пользователя (3–12 сек)
        wait = random.uniform(3, 12)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass


async def run_test(url: str, api_key: str, max_users: int, duration: int, ramp: int):
    """
    Основной цикл теста:
      0..ramp сек   — постепенно добавляем пользователей
      ramp..duration — держим максимальную нагрузку
    """
    stats = Stats(ramp_duration=ramp)
    stop_event = asyncio.Event()
    tasks = []

    timeout = httpx.Timeout(connect=5.0, read=35.0, write=5.0, pool=5.0)
    limits  = httpx.Limits(max_connections=max_users + 5, max_keepalive_connections=max_users)

    print(f"\n  Параметры теста:")
    print(f"  URL:              {url}")
    print(f"  Макс. пользоват.: {max_users}")
    print(f"  Длительность:     {duration} сек")
    print(f"  Нарастание:       {ramp} сек")
    print(f"  Вопросов в пуле:  {len(QUESTIONS)}")
    print(f"\n  Начинаем...\n")

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Проверка доступности
        try:
            r = await client.get(f"{url}/health", headers={"X-API-Key": api_key})
            print(f"  Health check: HTTP {r.status_code}")
            if r.status_code not in (200, 404):
                print(f"  [!] Сервис может быть недоступен")
        except Exception as e:
            print(f"  [!] Не могу достучаться до {url}: {e}")
            return

        test_start = time.monotonic()

        # Запускаем виртуальных пользователей постепенно
        user_interval = ramp / max_users if max_users > 0 else 1

        for i in range(max_users):
            if stop_event.is_set():
                break
            task = asyncio.create_task(
                virtual_user(i, client, url, api_key, stats, stop_event)
            )
            tasks.append(task)
            active = i + 1

            # Показываем прогресс
            stats.print_live(active)

            # Ждём перед добавлением следующего пользователя
            remaining = duration - (time.monotonic() - test_start)
            if remaining <= 0:
                break
            await asyncio.sleep(min(user_interval, remaining))

        # Держим максимальную нагрузку до конца теста
        while True:
            elapsed = time.monotonic() - test_start
            if elapsed >= duration:
                break
            stats.print_live(len([t for t in tasks if not t.done()]))
            await asyncio.sleep(2)

        # Останавливаем всех пользователей
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    stats.print_final()


def main():
    parser = argparse.ArgumentParser(description="Реалистичный нагрузочный тест RAG")
    parser.add_argument("--url",      default=os.getenv("RAG_SERVICE_URL", "http://localhost:8001"))
    parser.add_argument("--key",      default=os.getenv("RAG_API_KEY", os.getenv("API_SECRET_KEY", "")))
    parser.add_argument("--users",    type=int, default=25,  help="Макс. одновременных пользователей")
    parser.add_argument("--duration", type=int, default=120, help="Длительность теста (сек)")
    parser.add_argument("--ramp",     type=int, default=30,  help="Время нарастания нагрузки (сек)")
    args = parser.parse_args()

    if not args.key:
        print("[!] API ключ не найден. Укажи --key или проверь RAG_API_KEY в .env")
        sys.exit(1)

    print("=" * 60)
    print("РЕАЛИСТИЧНЫЙ НАГРУЗОЧНЫЙ ТЕСТ")
    print(f"Запущен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    asyncio.run(run_test(args.url, args.key, args.users, args.duration, args.ramp))


if __name__ == "__main__":
    main()
