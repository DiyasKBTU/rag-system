#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/load_test_realistic.py — Реалистичный нагрузочный тест RAG-сервиса.

Ключевые особенности:
  - Пользователи стартуют почти одновременно (как в реальном сценарии «все сразу»)
  - Учитывает время ответа: пока сервер думает 7 сек, запросы накапливаются
  - Показывает сколько запросов реально в полёте одновременно (in-flight)
  - Три режима нагрузки: normal (постепенно), burst (всё сразу), wave (волны)

Запуск из папки rag_service/:
    python ../tools/load_test_realistic.py               # обычный тест
    python ../tools/load_test_realistic.py --mode burst  # все разом
    python ../tools/load_test_realistic.py --mode wave   # волны нагрузки

Параметры:
    --url       URL RAG-сервиса (по умолчанию http://localhost:8001)
    --key       API-ключ (берётся из .env если не указан)
    --users     Кол-во одновременных пользователей (по умолчанию 20)
    --duration  Длительность теста в секундах (по умолчанию 120)
    --mode      normal | burst | wave (по умолчанию normal)

Примеры:
    python ../tools/load_test_realistic.py --users 10 --duration 60
    python ../tools/load_test_realistic.py --users 30 --mode burst --duration 90
    python ../tools/load_test_realistic.py --users 25 --mode wave --duration 180
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

load_dotenv()

# ── Реалистичные вопросы абитуриентов ─────────────────────────────────────────
QUESTIONS = [
    "сколько стоит обучение в ЦАИУ",
    "какая стоимость обучения на IT специальности",
    "сколько стоит контракт на юридическом факультете",
    "можно ли платить за учёбу в рассрочку",
    "какие специальности есть в ЦАИУ",
    "есть ли специальность программирование",
    "какие факультеты в университете",
    "на кого можно поступить в ЦАИУ",
    "есть ли медицинские специальности",
    "какие документы нужны для поступления",
    "когда начинается приём документов",
    "какой проходной балл ЕНТ для поступления",
    "можно ли поступить без ЕНТ",
    "как поступить иностранному гражданину",
    "есть ли гранты в ЦАИУ",
    "какие льготы для многодетных семей",
    "есть ли скидки на обучение",
    "как получить грант на обучение",
    "есть ли общежитие в ЦАИУ",
    "сколько стоит общежитие",
    "как заселиться в общежитие",
    "где находится университет",
    "как доехать до ЦАИУ",
    "номер телефона приёмной комиссии",
    "когда работает приёмная комиссия",
    "есть ли военная кафедра",
    "какие кружки и секции есть в университете",
    "есть ли дистанционное обучение",
    "как перевестись из другого вуза",
]


class Stats:
    """Сбор и отображение статистики с учётом in-flight запросов."""

    def __init__(self):
        self.response_times: list = []       # (timestamp, seconds) — только успешные
        self.all_response_times: list = []   # (timestamp, seconds) — все запросы
        self.errors: list = []
        self.status_codes: dict = defaultdict(int)
        self.start_time: float = time.monotonic()
        self.total_requests: int = 0
        self._inflight: int = 0              # сейчас в полёте
        self.peak_inflight: int = 0          # максимум одновременных
        self.inflight_history: list = []     # [(ts, count)] для графика
        self._lock = asyncio.Lock()

    async def enter_request(self):
        async with self._lock:
            self._inflight += 1
            if self._inflight > self.peak_inflight:
                self.peak_inflight = self._inflight
            ts = time.monotonic() - self.start_time
            self.inflight_history.append((ts, self._inflight))

    async def exit_request(self, response_time: float, status_code: int, error: str = None):
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self.total_requests += 1
            self.status_codes[status_code] += 1
            ts = time.monotonic() - self.start_time
            self.inflight_history.append((ts, self._inflight))
            self.all_response_times.append((ts, response_time))   # все
            if error or status_code >= 400:
                self.errors.append({
                    "ts":     ts,
                    "time":   response_time,
                    "status": status_code,
                    "error":  error or f"HTTP {status_code}",
                })
            else:
                self.response_times.append((ts, response_time))   # только успешные

    def inflight_now(self) -> int:
        return self._inflight

    def percentile(self, p: float, data: list = None) -> float:
        src = data if data is not None else [t for _, t in self.response_times]
        if not src:
            return 0.0
        s = sorted(src)
        idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
        return s[idx]

    def print_live(self, active_users: int):
        elapsed = time.monotonic() - self.start_time
        times = [t for _, t in self.response_times]
        ok  = len(times)
        err = len(self.errors)
        avg = statistics.mean(times) if times else 0.0
        mn  = min(times) if times else 0.0
        mx  = max(times) if times else 0.0
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        print(
            f"\r  👥 {active_users:2d} users | "
            f"🔄 {self._inflight:2d} in-flight | "
            f"✅ {ok:4d} ok | ❌ {err:3d} err | "
            f"⏱ {mn:.1f}/{avg:.1f}/{mx:.1f}s (min/avg/max) | "
            f"📈 {rps:.1f} req/s",
            end="", flush=True,
        )

    @staticmethod
    def _fmt_time(sec: float) -> str:
        """Умное форматирование: ms если < 1s, иначе секунды."""
        if sec < 1.0:
            return f"{sec*1000:.0f}ms"
        return f"{sec:.2f}s "

    def _histogram(self, times: list, width: int = 36, label: str = "") -> str:
        """Гистограмма. times — список секунд (float)."""
        if not times:
            return "  (нет данных)"
        mn, mx = min(times), max(times)
        rng = mx - mn
        if rng < 0.001:
            return f"  все запросы ≈ {self._fmt_time(mn)}"
        buckets = 10
        step = rng / buckets
        counts = [0] * buckets
        for t in times:
            idx = min(int((t - mn) / step), buckets - 1)
            counts[idx] += 1
        max_count = max(counts) or 1
        lines = []
        if label:
            lines.append(f"  {label}")
        for i, cnt in enumerate(counts):
            lo = mn + i * step
            hi = lo + step
            bar = "█" * int(cnt / max_count * width)
            pct = cnt / len(times) * 100
            lo_s = self._fmt_time(lo)
            hi_s = self._fmt_time(hi)
            lines.append(f"  {lo_s:>6}–{hi_s:<7} │{bar:<{width}}│ {cnt:3d} ({pct:.0f}%)")
        return "\n".join(lines)

    def _inflight_chart(self, width: int = 50) -> str:
        """ASCII-график изменения in-flight запросов во времени."""
        if len(self.inflight_history) < 2:
            return "  (недостаточно данных)"
        # Разбиваем время на width бакетов
        total_time = self.inflight_history[-1][0]
        if total_time < 1:
            return "  (тест слишком короткий)"
        step = total_time / width
        buckets = []
        for i in range(width):
            lo = i * step
            hi = (i + 1) * step
            vals = [v for ts, v in self.inflight_history if lo <= ts < hi]
            buckets.append(max(vals) if vals else 0)

        max_val = max(buckets) or 1
        # Нормализуем до 6 строк высоты
        height = 6
        lines = []
        for row in range(height, 0, -1):
            threshold = max_val * row / height
            line = ""
            for b in buckets:
                line += "█" if b >= threshold else " "
            label = f"{int(max_val * row / height):2d}"
            lines.append(f"  {label} │{line}│")
        lines.append(f"   0 └{'─' * width}┘")
        lines.append(f"     0{' ' * (width // 2 - 2)}{total_time/2:.0f}s{' ' * (width // 2 - 3)}{total_time:.0f}s")
        return "\n".join(lines)

    def print_final(self, mode: str, max_users: int):
        elapsed = time.monotonic() - self.start_time
        all_times = [t for _, t in self.response_times]
        ok    = len(all_times)
        err   = len(self.errors)
        total = self.total_requests

        print("\n\n" + "=" * 62)
        print("РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТА")
        print("=" * 62)

        print(f"\n📊 Общее:")
        print(f"   Режим:                  {mode} | {max_users} пользователей")
        print(f"   Длительность:           {elapsed:.1f} сек")
        print(f"   Всего запросов:         {total}")
        if total:
            print(f"   Успешных:               {ok}  ({ok/total*100:.1f}%)")
            print(f"   Ошибок:                 {err}  ({err/total*100:.1f}%)")
        print(f"   Пропускная способность: {total/elapsed:.2f} req/s")
        print(f"   Пик одновременных (in-flight): {self.peak_inflight}")

        if all_times:
            avg = statistics.mean(all_times)
            print(f"\n⏱  Время ответа:")
            print(f"   Минимум:   {min(all_times):.2f}s")
            print(f"   Среднее:   {avg:.2f}s")
            print(f"   Медиана:   {self.percentile(50):.2f}s")
            print(f"   P90:       {self.percentile(90):.2f}s  ← 90% запросов быстрее")
            print(f"   P95:       {self.percentile(95):.2f}s")
            print(f"   P99:       {self.percentile(99):.2f}s")
            print(f"   Максимум:  {max(all_times):.2f}s")

            # Сравнение первой и второй половины теста (деградация под нагрузкой)
            # Используем ВСЕ запросы (включая ошибки) для честной картины
            half = elapsed / 2
            first_half  = [t for ts, t in self.all_response_times if ts < half]
            second_half = [t for ts, t in self.all_response_times if ts >= half]
            if first_half and second_half:
                avg1 = statistics.mean(first_half)
                avg2 = statistics.mean(second_half)
                delta = avg2 - avg1
                sign  = "+" if delta >= 0 else ""
                print(f"\n📈 Деградация (первая половина vs вторая половина теста):")
                print(f"   1-я половина: avg {avg1:.2f}s, P95 {self.percentile(95, first_half):.2f}s  ({len(first_half)} запросов)")
                print(f"   2-я половина: avg {avg2:.2f}s, P95 {self.percentile(95, second_half):.2f}s  ({len(second_half)} запросов)")
                print(f"   Изменение:    {sign}{delta:.2f}s  ({sign}{delta/avg1*100:.1f}%)")

        # График in-flight
        print(f"\n📉 In-flight запросов во времени (пик: {self.peak_inflight}):")
        print(self._inflight_chart())

        # Гистограмма — все запросы (включая ошибки)
        all_req_times = [t for _, t in self.all_response_times]
        if all_req_times:
            print(f"\n📊 Распределение времени ответа (все {len(all_req_times)} запросов):")
            print(self._histogram(all_req_times))
            if all_times and len(all_times) < len(all_req_times):
                print(f"\n   Только успешные ({len(all_times)} из {len(all_req_times)}):")
                print(self._histogram(all_times))

        if self.status_codes:
            print(f"\n📋 Коды ответов:")
            for code, count in sorted(self.status_codes.items()):
                label = {
                    200: "✅ OK",
                    401: "🔑 Unauthorized",
                    429: "⚠️  Rate limit",
                    500: "💥 Server error",
                    503: "🚫 Перегрузка (семафор переполнен)",
                    0:   "🔌 Timeout / нет связи",
                }.get(code, f"   HTTP {code}")
                print(f"   {label}: {count}")

        if self.errors:
            print(f"\n❌ Последние ошибки:")
            for e in self.errors[-5:]:
                print(f"   t={e['ts']:.1f}s [{e['status']}] {e['error'][:80]}")

        # Вердикт
        print(f"\n{'=' * 62}")
        error_rate   = err / total if total else 0
        p95          = self.percentile(95)
        _fh = [t for ts, t in self.all_response_times if ts < elapsed / 2]
        _sh = [t for ts, t in self.all_response_times if ts >= elapsed / 2]
        avg1 = statistics.mean(_fh) if _fh else 0
        avg2 = statistics.mean(_sh) if _sh else 0
        degradation  = (avg2 - avg1) / avg1 if avg1 else 0

        if error_rate < 0.01 and p95 < 12 and degradation < 0.30:
            verdict = "✅ ОТЛИЧНО — система справляется, деградация минимальная"
        elif error_rate < 0.05 and p95 < 20:
            verdict = "⚠️  НОРМАЛЬНО — небольшие задержки при пике нагрузки"
        elif error_rate < 0.15:
            verdict = "⚠️  ПРЕДЕЛ — заметная деградация, появляются ошибки"
        else:
            verdict = "❌ ПЕРЕГРУЗКА — система не справляется с нагрузкой"

        print(f"Вердикт: {verdict}")
        print("=" * 62)


async def do_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    question: str,
    stats: Stats,
):
    """Один запрос к /search с учётом in-flight счётчика."""
    await stats.enter_request()
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
        error = "Timeout (>35s)"
    except Exception as e:
        status = 0
        error = str(e)[:80]
    finally:
        elapsed = time.monotonic() - start
        await stats.exit_request(elapsed, status, error)


async def virtual_user(
    user_id: int,
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    stats: Stats,
    stop_event: asyncio.Event,
    start_jitter: float = 2.0,
):
    """
    Виртуальный пользователь: задаёт вопросы с паузами.
    start_jitter: максимальный разброс старта (сек).
    Пауза между вопросами: 4–10 сек (пользователь читает ответ, думает).
    """
    if start_jitter > 0:
        await asyncio.sleep(random.uniform(0, start_jitter))

    while not stop_event.is_set():
        question = random.choice(QUESTIONS)
        await do_request(client, url, api_key, question, stats)

        # Пауза как у реального пользователя
        wait = random.uniform(4, 10)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass


async def run_normal(
    client: httpx.AsyncClient,
    url: str, api_key: str,
    max_users: int, duration: int,
    stats: Stats, stop_event: asyncio.Event,
):
    """
    Normal: постепенно набираем пользователей за первые 20% времени,
    потом держим до конца. Стартовый разброс ≤2 сек.
    """
    tasks = []
    ramp_time = duration * 0.2
    interval  = ramp_time / max_users if max_users else 1
    test_start = time.monotonic()

    for i in range(max_users):
        if stop_event.is_set():
            break
        t = asyncio.create_task(
            virtual_user(i, client, url, api_key, stats, stop_event, start_jitter=1.0)
        )
        tasks.append(t)
        stats.print_live(i + 1)

        remaining = duration - (time.monotonic() - test_start)
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))

    # Держим нагрузку
    while True:
        elapsed = time.monotonic() - test_start
        if elapsed >= duration:
            break
        stats.print_live(len([t for t in tasks if not t.done()]))
        await asyncio.sleep(1.5)

    stop_event.set()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_burst(
    client: httpx.AsyncClient,
    url: str, api_key: str,
    max_users: int, duration: int,
    stats: Stats, stop_event: asyncio.Event,
):
    """
    Burst: все пользователи стартуют почти одновременно (разброс ≤1 сек).
    Имитирует «всё сразу»: открытие дня, объявление и т.п.
    """
    print(f"\n  ⚡ BURST: {max_users} пользователей стартуют одновременно!\n")
    test_start = time.monotonic()

    tasks = [
        asyncio.create_task(
            virtual_user(i, client, url, api_key, stats, stop_event, start_jitter=0.5)
        )
        for i in range(max_users)
    ]

    while True:
        elapsed = time.monotonic() - test_start
        if elapsed >= duration:
            break
        stats.print_live(len([t for t in tasks if not t.done()]))
        await asyncio.sleep(1.5)

    stop_event.set()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_wave(
    client: httpx.AsyncClient,
    url: str, api_key: str,
    max_users: int, duration: int,
    stats: Stats, stop_event: asyncio.Event,
):
    """
    Wave: волны нагрузки.
    Тихо (20% пользователей) → пик (100%) → тихо → пик → ...
    Период волны: ~30 сек.
    """
    print(f"\n  🌊 WAVE: волны нагрузки, пик {max_users} / тихо {max(1, max_users // 5)} пользователей\n")
    wave_period = 30
    test_start  = time.monotonic()
    active_tasks: list = []
    wave_num = 0

    while True:
        elapsed = time.monotonic() - test_start
        if elapsed >= duration or stop_event.is_set():
            break

        phase_in_wave = elapsed % wave_period
        # 0–10 сек нарастание, 10–20 пик, 20–30 спад
        if phase_in_wave < wave_period * 0.33:
            target = max(1, int(max_users * phase_in_wave / (wave_period * 0.33)))
        elif phase_in_wave < wave_period * 0.67:
            target = max_users
        else:
            target = max(1, max_users // 5)

        current = len([t for t in active_tasks if not t.done()])

        # Добавляем пользователей если надо
        while current < target and not stop_event.is_set():
            t = asyncio.create_task(
                virtual_user(len(active_tasks), client, url, api_key, stats, stop_event, start_jitter=0.5)
            )
            active_tasks.append(t)
            current += 1

        stats.print_live(current)
        await asyncio.sleep(1.5)

    stop_event.set()
    await asyncio.gather(*active_tasks, return_exceptions=True)


async def run_test(url: str, api_key: str, max_users: int, duration: int, mode: str):
    stats      = Stats()
    stop_event = asyncio.Event()

    timeout = httpx.Timeout(connect=5.0, read=40.0, write=5.0, pool=5.0)
    limits  = httpx.Limits(max_connections=max_users + 10, max_keepalive_connections=max_users)

    print(f"\n  Параметры теста:")
    print(f"  URL:          {url}")
    print(f"  Пользоват.:   {max_users}")
    print(f"  Длительность: {duration} сек")
    print(f"  Режим:        {mode}")
    print(f"  Вопросов:     {len(QUESTIONS)}\n")

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Проверка доступности
        try:
            r = await client.get(f"{url}/health", headers={"X-API-Key": api_key})
            status_icon = "✅" if r.status_code in (200,) else "⚠️ "
            print(f"  Health check: {status_icon} HTTP {r.status_code}\n")
        except Exception as e:
            print(f"  [!] Не могу достучаться до {url}: {e}")
            return

        if mode == "burst":
            await run_burst(client, url, api_key, max_users, duration, stats, stop_event)
        elif mode == "wave":
            await run_wave(client, url, api_key, max_users, duration, stats, stop_event)
        else:
            await run_normal(client, url, api_key, max_users, duration, stats, stop_event)

    stats.print_final(mode, max_users)


def main():
    parser = argparse.ArgumentParser(description="Реалистичный нагрузочный тест RAG")
    parser.add_argument("--url",      default=os.getenv("RAG_SERVICE_URL", "http://localhost:8001"))
    parser.add_argument("--key",      default=os.getenv("RAG_API_KEY", os.getenv("API_SECRET_KEY", "")))
    parser.add_argument("--users",    type=int, default=20,     help="Кол-во пользователей")
    parser.add_argument("--duration", type=int, default=120,    help="Длительность теста (сек)")
    parser.add_argument("--mode",     default="normal",
                        choices=["normal", "burst", "wave"],
                        help="normal=постепенно, burst=все сразу, wave=волны")
    args = parser.parse_args()

    if not args.key:
        print("[!] API ключ не найден. Укажи --key или проверь RAG_API_KEY в .env")
        sys.exit(1)

    print("=" * 62)
    print("РЕАЛИСТИЧНЫЙ НАГРУЗОЧНЫЙ ТЕСТ")
    print(f"Запущен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    asyncio.run(run_test(args.url, args.key, args.users, args.duration, args.mode))


if __name__ == "__main__":
    main()
