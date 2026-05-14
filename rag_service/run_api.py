# -*- coding: utf-8 -*-
"""
run_api.py - Start the FastAPI server.

Usage:
  python run_api.py

The server will start on http://localhost:8001
API docs: http://localhost:8001/docs  (interactive Swagger UI)
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting CAIU RAG Service")
    print("=" * 60)
    print("API:      http://localhost:8001")
    print("Docs:     http://localhost:8001/docs")
    print("Health:   http://localhost:8001/health")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,       # Set to True for development (auto-restart on file changes)
        log_level="info",
        # workers=2 даёт два независимых процесса Python с отдельными event loop'ами
        # и threadpool'ами. Это удваивает пропускную способность для I/O-bound задач
        # (OpenAI, Qdrant, Redis).
        #
        # ВАЖНО: каждый процесс имеет СВОЙ asyncio.Semaphore (_search_semaphore)
        # и СВОЙ threadpool. Семантика per-process:
        #   - 2 workers × 20 search slots = 40 параллельных поисков на машине
        #   - 2 workers × ~40 threadpool threads = до 80 одновременных sync операций
        # Файловые блокировки (hot_swap.indexing.lock, scheduler.lock) корректно
        # синхронизируют переиндексацию между процессами.
        #
        # Для разработки оставьте workers=1 (или уберите параметр).
        workers=2,
    )
