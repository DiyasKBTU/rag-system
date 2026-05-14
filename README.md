# ЦАИУ — Telegram-бот приёмной комиссии

Бот отвечает на вопросы абитуриентов на **русском, казахском и английском** языках.
Использует умный поиск по сайту университета и ChatGPT для формулировки ответов.

---

## Содержание

**Часть 1 — Инструкции по эксплуатации**
1. [Первый запуск](#первый-запуск)
2. [Ежедневная работа](#ежедневная-работа)
3. [Автозапуск через systemd](#автозапуск-через-systemd-продакшн-на-linux-сервере)
4. [Обновление базы знаний](#обновление-базы-знаний)
5. [Добавление Google Docs](#добавление-google-docs)
6. [Ручное редактирование базы](#ручное-редактирование-базы)
7. [Диагностика и отладка](#диагностика-и-отладка)
8. [Частые вопросы](#частые-вопросы)

**Часть 2 — Техническое устройство**
8. [Архитектура системы](#архитектура-системы)
9. [Пайплайн индексации](#пайплайн-индексации)
10. [Пайплайн поиска](#пайплайн-поиска)
11. [Blue/Green горячая замена](#bluegreen-горячая-замена)
12. [Кеширование](#кеширование)
13. [Streaming ответа](#streaming-ответа)
14. [Структура данных в Qdrant](#структура-данных-в-qdrant)
15. [Параметры системы](#параметры-системы)

---

# Часть 1 — Инструкции по эксплуатации

## Первый запуск

### Требования

- Python 3.10+
- Docker Desktop (установлен и запущен)
- Аккаунт OpenAI с пополненным балансом
- Telegram Bot Token (от @BotFather)

### Шаг 1 — Установить зависимости Python

```bash
pip install -r requirements.txt
```

### Шаг 2 — Создать файлы конфигурации

Нужно создать **два** `.env` файла.

**`rag_service/.env`** — для сервиса поиска:

```ini
OPENAI_API_KEY=sk-proj-...         # Ключ OpenAI (platform.openai.com/api-keys)
API_SECRET_KEY=придумайте_ключ_16+ # Пароль для защиты API (минимум 16 символов)

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=caiu_knowledge_base

SITE_BASE_URL=https://caiu.edu.kz
SITEMAP_URL=https://caiu.edu.kz/sitemap.xml

# Необязательно — для Google Docs:
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
GOOGLE_DOC_ID=             # ID одного документа (можно оставить пустым)
```

**`.env`** в корне проекта — для бота:

```ini
BOT_TOKEN=1234567890:AAF...        # Токен бота от @BotFather
OPENAI_API_KEY=sk-proj-...         # Тот же ключ OpenAI
RAG_SERVICE_URL=http://localhost:8001
RAG_API_KEY=придумайте_ключ_16+   # Тот же пароль что в API_SECRET_KEY выше
```

> **Важно:** `API_SECRET_KEY` в `rag_service/.env` и `RAG_API_KEY` в `.env` должны совпадать.

### Шаг 3 — Запустить базу данных

```bash
cd rag_service
docker-compose up -d
```

Проверить:
```bash
docker ps   # должен быть контейнер qdrant/qdrant со статусом Up
```

### Шаг 4 — Первая индексация сайта (~4–6 минут)

```bash
cd rag_service
python -m app.parser
```

Стоимость: **~$0.18–0.20** (OpenAI API). Что происходит во время индексации — см. [Пайплайн индексации](#пайплайн-индексации).

Пример вывода:
```
[1/204] https://caiu.edu.kz/history-of-the-university-ru/
  -> Chunks: 4  |  Questions: 16  |  Saved: 20
...
[Pipeline] Done! pages=198, chunks=412, questions=1648, total_qdrant=2060
```

### Шаг 5 — Проверить что всё работает

```bash
# Терминал 1:
cd rag_service && python run_api.py
# → Uvicorn running on http://0.0.0.0:8001

# Другой терминал:
curl http://localhost:8001/health
# → {"status": "ok", ...}
```

---

## Ежедневная работа

### Запуск (ручной — для разработки)

Открыть два терминала:

```bash
# Терминал 1 — сервис поиска:
cd rag_service && python run_api.py

# Терминал 2 — бот:
python bot/run.py
```

Оба процесса должны работать одновременно.

### Остановка (ручной запуск)

В каждом терминале нажать `Ctrl+C`.

---

## Автозапуск через systemd (продакшн на Linux-сервере)

Для продакшн-деплоя используйте systemd — бот и RAG-сервис будут автоматически
стартовать при загрузке сервера и перезапускаться при падении.

### Установка

```bash
cd /home/YOUR_USER/caiu-bot/deploy

# Сделать скрипт исполняемым
chmod +x install.sh

# Установить (замените YOUR_USER и путь на реальные)
sudo ./install.sh ubuntu /home/ubuntu/caiu-bot
```

Скрипт создаст два юнита: `caiu-rag` (RAG-сервис) и `caiu-bot` (Telegram-бот),
включит автозапуск при загрузке.

### Первый запуск после установки

```bash
# Запустить оба сервиса
sudo systemctl start caiu-rag
sudo systemctl start caiu-bot

# Проверить статус
sudo systemctl status caiu-rag caiu-bot
```

### Управление

```bash
# Перезапустить бота (например, после обновления кода)
sudo systemctl restart caiu-bot

# Перезапустить RAG-сервис
sudo systemctl restart caiu-rag

# Остановить всё
sudo systemctl stop caiu-bot caiu-rag

# Отключить автозапуск
sudo systemctl disable caiu-bot caiu-rag
```

### Просмотр логов

```bash
# Логи бота в реальном времени
journalctl -u caiu-bot -f

# Последние 100 строк логов RAG
journalctl -u caiu-rag -n 100

# Логи за сегодня
journalctl -u caiu-bot --since today

# Логи с временны́ми метками
journalctl -u caiu-bot -o short-iso
```

Помимо journald логи также пишутся в файлы `rag_service/logs/` (RotatingFileHandler,
5 файлов по 5 МБ).

### Что делать если бот не запускается

1. Проверить статус: `sudo systemctl status caiu-bot`
2. Посмотреть ошибку: `journalctl -u caiu-bot -n 50`
3. Убедиться что Docker запущен: `docker ps` — должен быть контейнер `qdrant`
4. Проверить что RAG работает: `curl http://localhost:8001/health`
5. Проверить `.env` файлы — все ключи должны быть заполнены

### Индикаторы в ответах бота

| Индикатор | Значение |
|---|---|
| 🟢 | Ответ основан на данных из базы знаний |
| 🔴 | Сервис поиска недоступен, GPT отвечает без базы |

Если видите 🔴 — сервис поиска не запущен или упал.

---

## Обновление базы знаний

### Когда нужно обновлять

- На сайте caiu.edu.kz изменились страницы
- Изменились правила приёма, стоимость, список специальностей
- Добавили новый Google Doc в конфигурацию

### Горячая переиндексация (рекомендуется)

Бот **не останавливается** во время обновления — абитуриенты продолжают получать ответы.

```bash
python tools/indexer_ctl.py start
```

Следить за процессом:
```bash
python tools/indexer_ctl.py status
```

Пример вывода после завершения:
```
В процессе:         нет ✅
Активная коллекция: caiu_knowledge_base_green
Записей в Qdrant:   2087

📋 Последняя индексация:
   Страниц:  8 обновлено (193 пропущено без изменений)
   Чанков:   428
   Вопросов: 1712
```

Если сайт не менялся — переиндексация завершится за ~30 секунд, не расходуя токены OpenAI. Перерабатываются только страницы с изменившимся содержимым.

### Запланировать переиндексацию

```bash
# Сегодня/завтра в 03:00
python tools/indexer_ctl.py schedule "03:00"

# На конкретную дату
python tools/indexer_ctl.py schedule "15.06 03:00"
python tools/indexer_ctl.py schedule "15.06.2025 03:00"

# Посмотреть расписание
python tools/indexer_ctl.py status

# Отменить
python tools/indexer_ctl.py unschedule
```

> Для срабатывания расписания `rag_service` должен быть запущен — планировщик встроен в сервис.

---

## Добавление Google Docs

Если информация есть в Google Docs, но не на сайте (правила приёма, FAQ, прейскурант) — её можно добавить в базу знаний.

### Шаг 1 — Подготовить сервисный аккаунт (один раз)

Если `service_account.json` уже есть в `rag_service/` — пропустите этот шаг.

1. Откройте [Google Cloud Console](https://console.cloud.google.com/)
2. Включите **Google Drive API** и **Google Docs API** (APIs & Services → Library)
3. Создайте сервисный аккаунт (APIs & Services → Credentials → Create Credentials → Service Account)
4. Скачайте JSON-ключ: кликните на аккаунт → Keys → Add Key → JSON
5. Переименуйте скачанный файл в `service_account.json`, положите в `rag_service/`
6. Запомните email аккаунта вида: `my-bot@my-project.iam.gserviceaccount.com`

### Шаг 2 — Дать доступ к документу

Откройте нужный Google Doc → **Поделиться** → добавьте email сервисного аккаунта с правом **"Читатель"**.

### Шаг 3 — Получить ID документа

```
https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/edit
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   Это и есть ID документа
```

### Шаг 4 — Добавить в конфигурацию

`rag_service/.env` для одного документа:
```ini
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
GOOGLE_DOC_ID=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
```

Для нескольких — в `rag_service/app/config.py`:
```python
GOOGLE_DOC_IDS = [
    {"id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms", "title": "Правила приёма 2025"},
    {"id": "2CyiNWt1YSB6...", "title": "Стоимость обучения 2025"},
]
```

### Шаг 5 — Переиндексировать

```bash
python tools/indexer_ctl.py start
```

### Как правильно оформить Google Doc

Система разбивает документ на фрагменты по заголовкам. Каждый раздел становится отдельным фрагментом с понятным контекстом.

Используйте **стили заголовков** (выпадающий список в Google Docs), а не просто жирный текст:

```
[Заголовок 1] Правила приёма в ЦАИУ 2025
[Заголовок 2] Необходимые документы
[Обычный текст] Аттестат, удостоверение личности, фото 3×4...
[Заголовок 2] Стоимость обучения
[Обычный текст] Технические специальности: 450 000 тг...
[Заголовок 2] Гранты и скидки
[Обычный текст] Университет предоставляет скидки отличникам...
```

Жирный текст (`Ctrl+B`) система не распознаёт как заголовок.

Рекомендации:
- Каждый раздел должен отвечать на один конкретный вопрос
- Пишите конкретные цифры, даты, суммы — они важнее общих описаний
- Оптимальный объём раздела: 100–300 слов
- Не дублируйте информацию которая уже есть на сайте caiu.edu.kz

---

## Ручное редактирование базы

Автоматический парсинг иногда даёт неточные результаты: таблицы не читаются, важная информация теряется. Для таких случаев есть инструменты ручной правки.

### Просмотр содержимого базы

```bash
# Все проиндексированные страницы
python tools/inspect_db.py

# Фрагменты конкретной страницы
python tools/inspect_db.py https://caiu.edu.kz/obshhezhitie/

# Только текстовые фрагменты, без вопросов
python tools/inspect_db.py https://caiu.edu.kz/obshhezhitie/ --no-questions
```

### Переиндексация одной страницы

Если изменилась одна конкретная страница и не хочется ждать полную переиндексацию:

```bash
# С генерацией вопросов (стандартно):
python tools/index_page.py https://caiu.edu.kz/about-ru/

# Быстро, без GPT:
python tools/index_page.py https://caiu.edu.kz/contacts/ --no-questions
```

### Браузерный редактор (рекомендуется)

```bash
cd rag_service
python ../tools/chunk_editor.py
# → открыть: http://localhost:8080
```

Что можно делать:
- Просматривать страницы с поиском по URL и названию
- Редактировать текст, заголовок раздела и теги каждого фрагмента
- Удалять мусорные фрагменты, добавлять новые вручную
- Просматривать и редактировать вопросы к каждому фрагменту
- Генерировать вопросы через GPT кнопкой 🤖
- Просматривать историю изменений и откатываться к предыдущей версии
- **Кнопка "➕ URL" в шапке** — проиндексировать одну страницу прямо из браузера

Для защиты паролем — добавьте в `rag_service/.env`:
```ini
CHUNK_EDITOR_PASSWORD=ваш_пароль
```

### Страницы где контент — изображение, схема или форма

На некоторых страницах сайта информация представлена только в виде картинки (например, организационная структура университета). Парсер не может прочитать картинку — бот о ней не знает.

### Добавление фактов которых нет на сайте

Оба случая (страницы-картинки и дополнительные факты) управляются через единый файл **`manual_knowledge.json`** в корне проекта.

Формат записи:

```json
{
  "entries": [
    {
      "id": "my-topic",
      "title": "Краткое название",
      "description": "Текст который увидит бот при поиске",
      "search_queries": [
        "как задаст вопрос пользователь",
        "другой вариант вопроса"
      ],
      "link": "https://caiu.edu.kz/страница/"
    }
  ]
}
```

Поле `link` — **опционально**:
- Есть → запись о странице-картинке (бот даст ссылку пользователю)
- Нет → просто факт (бот отвечает текстом без ссылки)

После редактирования файла — запустить (занимает несколько секунд):

```bash
python tools/rebuild_manual_knowledge.py
```

Бот сразу найдёт новые записи по поисковым запросам.

### Массовое редактирование через JSON

```bash
# Экспорт всей базы в файл
python tools/export_chunks.py       # → chunks_export.json

# Отредактируйте файл в любом редакторе, затем загрузите обратно:
python tools/index_from_chunks.py
```

---

## Диагностика и отладка

### Юнит-тесты

```bash
cd rag_service
python -m pytest ../tests/ -v
```

Тесты не требуют Qdrant, Redis или OpenAI — все внешние зависимости замоканы.
Покрытие: `chunker.py`, `extractor.py`, `search.py` (чистые функции), `storage.py` (state-файл).

### Предпросмотр — что будет проиндексировано

```bash
python tools/preview_urls.py
python tools/preview_urls.py https://caiu.edu.kz/history-of-the-university-ru/
```

Статусы URL: `[НОВЫЙ]`, `[В БАЗЕ]` (без изменений — пропустится), `[ИСКЛЮЧЁН]`.

### Тест поиска по вопросу

```bash
python tools/debug_search.py "сколько корпусов у университета"
```

Показывает найденные фрагменты, их рейтинг релевантности и текст.

### Проверка сервисов

```bash
# RAG-сервис
curl http://localhost:8001/health

# Статистика базы
curl http://localhost:8001/stats

# Веб-интерфейс Qdrant
# http://localhost:6333/dashboard
```

### Тест качества поиска

```bash
python tools/test_search_quality.py
```

Прогоняет ~50 типичных вопросов и показывает: `✅ ХОРОШО` / `⚠️ СЛАБО` / `❌ НЕ НАЙДЕНО`.

### Очистить базу и начать с нуля

```bash
python tools/clear_index.py
cd rag_service && python -m app.parser
```

---

## Частые вопросы

**Бот отвечает с индикатором 🔴**
Сервис поиска не запущен. Запустите: `cd rag_service && python run_api.py`

**Ошибка "Connection refused" при запуске сервиса**
База данных не запущена. Выполните: `cd rag_service && docker-compose up -d`

**Бот не знает о конкретной теме / даёт устаревшую информацию**
Запустите переиндексацию: `python tools/indexer_ctl.py start`

**Поиск находит нерелевантные фрагменты**
Проверьте через `debug_search.py`. Если нужная информация есть в базе (`inspect_db.py`), но поиск её не находит — попробуйте переиндексацию с очисткой.

**Нужно добавить информацию которой нет на сайте**
Создайте Google Doc → расшарьте на email сервисного аккаунта → добавьте ID в `config.py` → переиндексируйте. Или добавьте через JSON-файл напрямую.

**Страница есть на сайте но бот о ней не знает (контент — картинка)**
Откройте `manual_knowledge.json` в корне проекта, добавьте запись с полями `id`, `title`, `description`, `search_queries` и `link` (URL страницы), затем запустите `python tools/rebuild_manual_knowledge.py`. Полная переиндексация не нужна.

**Изменилась одна страница, не хочу ждать полную переиндексацию**
`python tools/index_page.py https://caiu.edu.kz/page/` или кнопка **"➕ URL"** в браузерном редакторе.

**Отредактировал чанки в chunk_editor — они пропадают при переиндексации**
Если все чанки страницы созданы в chunk_editor (флаг `manually_edited`), переиндексация их **не трогает** — защита встроена. Исключение: если вы запустили `clear_index.py` (полная очистка), после неё нужно заново добавить ручные чанки или восстановить из бэкапа.

**Сколько стоит работа бота?**
- Один вопрос пользователя: ~$0.001–0.003
- 10 000 вопросов в месяц: ~$10–30
- Полная переиндексация сайта: ~$0.18–0.20 (если сайт не менялся — ~$0, ~30 сек)

**Можно ли запустить без интернета?**
Нет. Система требует интернет для обращений к OpenAI API (поиск и генерация ответов) и скачивания страниц сайта (только при индексации). База Qdrant работает локально.

---

---

# Часть 2 — Техническое устройство

## Архитектура системы

```
Пользователь (Telegram)
        │
        ▼
  bot/run.py                         ← aiogram 3.x
  ├── Определение языка (ru/kk/en)
  ├── Перевод на русский (если нужно) → OpenAI GPT-4.1-mini
  │   └── кеш: in-memory + Redis TTL 30 дней
  ├── Запрос контекста (httpx POST /search)
  │   └── retry: 3 попытки, backoff 1/2/4 сек
  ├── Формирование системного промпта (персонаж Айданы)
  └── Streaming-ответ → OpenAI GPT-4.1-mini → Telegram (edit placeholder)

        │ POST /search  (X-API-Key: secret)
        ▼
  rag_service/run_api.py             ← FastAPI, порт 8001
  └── app/retrieval/search.py
      ├── Расширение запроса (SYNONYM_MAP + rapidfuzz)
      ├── Embedding (OpenAI text-embedding-3-small, 1536 dim)
      │   └── кеш: in-memory (512 слотов) + Redis TTL 30 дней
      ├── Qdrant cosine search через alias
      ├── Фильтр уверенности (score ≥ 0.28)
      ├── Фильтр хвоста (score ≥ best - 0.25)
      ├── Дедупликация (max 2 чанка с одной страницы)
      └── Топ 5 фрагментов → JSON-ответ

        │
        ▼
  Qdrant (Docker, порт 6333)
  └── alias: caiu_knowledge_base → _blue или _green
```

**Стек:**

| Компонент | Технология |
|---|---|
| Telegram-бот | aiogram 3.27 (Python) |
| RAG-сервис | FastAPI (Python, порт 8001) |
| Векторная БД | Qdrant (Docker, порт 6333) |
| Кеш | Redis (Docker, порт 6379) |
| Эмбеддинги | OpenAI text-embedding-3-small (1536 dim) |
| Генерация ответов | OpenAI gpt-4.1-mini |

---

## Пайплайн индексации

```
crawler.get_urls_for_indexing()
    │  sitemap.xml → ~200 URL с фильтрацией
    │  Исключаются: /kz/ страницы, новости, pagination, lang= параметры
    │
    ▼
extractor.get_page_content(url)
    │  requests + BeautifulSoup
    │  Удаляется: навигация, футер, скрипты, форм-блоки
    │  h2/h3/h4 теги → маркеры "## Заголовок" в тексте
    │  Результат: PageContent(text, title, url, content_hash)
    │  content_hash = md5(clean_text) — для определения изменений
    │
    ▼
page_needs_update(url, hash, collection_name)
    │  Ищет в Qdrant чанки с этим page_url
    │  Сравнивает content_hash → если совпадает, страница пропускается
    │  Если не совпадает или страницы нет — продолжаем
    │
    ▼
chunker.split_into_chunks(text, page_url, page_title)
    │  1. Текст разбивается по маркерам "## Заголовок" → секции
    │  2. Каждая секция режется по размеру слов (~200 слов)
    │  3. Между соседними чанками перекрытие 30 слов (overlap)
    │  4. Каждый чанк получает префикс "[Страница > Раздел]"
    │  Результат: List[TextChunk]
    │
    ▼
question_generator.generate_question_chunks(chunks)
    │  Для каждого TextChunk — параллельные запросы к GPT (5 одновременно)
    │  Промпт: "Сгенерируй 4 вопроса на русском, на которые отвечает этот текст"
    │  Каждый вопрос сохраняется как отдельный TextChunk:
    │    embed_text = "вопрос" (векторизуется)
    │    text = оригинальный текст чанка (возвращается при поиске)
    │  Результат: List[TextChunk] (вопрос-чанки)
    │
    ▼
storage.save_chunks(chunks + question_chunks, url, hash)
    │  embeddings.py: батч-запросы к OpenAI (20 текстов за раз)
    │  Для каждого чанка: PointStruct(id=uuid, vector=embed, payload={...})
    │  Удаляет старые чанки страницы (delete_chunks_by_url)
    │  Upsert в Qdrant
    │
    ▼
catalog_builder.build_and_save_catalog_chunks()
    │  Скачивает страницы факультетов/специальностей отдельно
    │  Собирает синтетические "сводные" чанки по каждому факультету
    │  Сохраняет в ту же коллекцию
    │
    ▼
Qdrant: ~2000 записей
  ~400 текстовых чанков + ~1600 вопрос-чанков + каталог
```

**Почему question-chunk индексация:** поисковый запрос пользователя — это вопрос, а в базе хранятся ответы. Эмбеддинги вопроса и ответа не идентичны: "Какие документы нужны?" и "Для поступления нужны: аттестат, удостоверение..." — семантически разные векторы. Сохранение вопросов как отдельных векторов (указывающих на тот же текст) решает эту проблему без BM25 гибридного поиска.

---

## Пайплайн поиска

```
POST /search {"query": "какие документы нужны для поступления", "lang": "ru"}
    │
    ▼
_expand_query(query)
    │  1. SYNONYM_MAP: "поступление" → добавляет "приём", "абитуриент", "зачисление"
    │  2. rapidfuzz: fuzzy matching по ключам словаря (порог схожести: 85%)
    │  Ключи сортируются по длине (длинные составные → приоритет)
    │  Результат: несколько вариантов запроса
    │
    ▼
_get_embeddings_batch_cached(queries)
    │  Для каждого варианта запроса:
    │    1. in-memory cache (_EMBEDDING_CACHE, 512 слотов, LRU)
    │    2. Redis: GET "caiu:emb:{md5(text)}" → bytes → float32[]
    │    3. OpenAI API batch (все cache-miss'ы за один запрос)
    │    4. Записать результат в Redis (TTL 30 дней) и in-memory
    │  Валидация: вектор должен быть 1536-мерным и ненулевым
    │
    ▼
qdrant.search() × N вариантов запроса
    │  collection: alias "caiu_knowledge_base" (→ _blue или _green)
    │  metric: cosine similarity
    │  score_threshold: 0.28
    │  limit: 8 (TOP_K_RESULTS)
    │  Результаты объединяются, дубликаты по id убираются
    │
    ▼
confidence check
    │  best_score = max(score для всех результатов)
    │  Если best_score < MIN_CONFIDENT_SCORE (0.28) → возврат []
    │  Бот ответит: "К сожалению, у меня нет точной информации по этому вопросу"
    │
    ▼
_apply_score_gap_filter(results, best_score)
    │  Убирает результаты с score < best_score - SCORE_GAP_THRESHOLD (0.25)
    │  Устраняет нерелевантный "хвост" который мог попасть выше порога 0.28
    │
    ▼
_deduplicate_results(results)
    │  Max 2 чанка с одного page_url (параметр MAX_CHUNKS_PER_PAGE)
    │  Предотвращает доминирование одной страницы в контексте
    │
    ▼
[:MAX_CONTEXT_CHUNKS]  (топ 5)
    │
    ▼
JSON-ответ: [{text, page_url, page_title, section_title, score}, ...]
```

---

## Blue/Green горячая замена

Проблема: полная индексация занимает 4–6 минут. Останавливать бота нельзя.

Решение: два реальных набора данных в Qdrant + один алиас. Бот всегда работает через алиас, не зная о реальных коллекциях.

```
Qdrant коллекции:
  caiu_knowledge_base_blue   ← реальная коллекция (данные)
  caiu_knowledge_base_green  ← реальная коллекция (данные)

Qdrant алиас:
  caiu_knowledge_base  →  _blue  (бот работает через этот алиас)
```

**Процесс переиндексации:**

```
1. get_shadow_collection_name()
   Активна blue? → shadow = green. Активна green? → shadow = blue.

2. Очистка shadow
   client.delete_collection(shadow)  # если была с прошлого раза
   ensure_collection_exists(shadow)  # создаём чистую

3. Snapshot: копирование active → shadow (НОВОЕ, incremental reindex)
   client.scroll(active, limit=256, with_vectors=True) → батчи
   client.upsert(shadow, points=[...])
   После: shadow содержит те же ~2000 точек что и active
   page_needs_update() в pipeline находит content_hash-и → пропускает неизменённые страницы

4. run_indexing(collection_name=shadow)
   Переиндексирует только изменившиеся страницы
   Бот всё это время работает через алиас на active

5. _atomic_swap(shadow)
   Один Qdrant-запрос: удалить старый алиас + создать новый
   Атомарность: нет момента когда алиас не существует
   client.update_collection_aliases([
       DeleteAliasOperation(alias_name="caiu_knowledge_base"),
       CreateAliasOperation(collection_name=shadow, alias_name="caiu_knowledge_base"),
   ])

6. client.delete_collection(old_active)
   Удаление старой коллекции для освобождения памяти
```

**Состояние хранится** в `hot_swap_state.json` — атомарная запись через `os.replace()`. При рестарте сервис читает этот файл чтобы знать какая коллекция активна.

**Первая миграция** (один раз): если существует реальная коллекция `caiu_knowledge_base` (без blue/green) — она удаляется и создаётся алиас. Небольшое окно (~10 мс) когда коллекция недоступна — за это время один запрос может получить ошибку.

---

## Кеширование

Система использует двухуровневый кеш: быстрый in-memory (живёт пока процесс запущен) и персистентный Redis (выживает при рестарте, TTL 30 дней).

### Кеш эмбеддингов (`search.py`)

```
Запрос "какие документы нужны"
    │
    ▼ Level 1: in-memory dict (_EMBEDDING_CACHE)
    │  Хранит: {text → vector}
    │  Размер: 512 слотов (LRU вытеснение)
    │  Thread-safe: threading.Lock (_CACHE_LOCK)
    │  Lock не держится во время сетевого вызова (только чтение/запись кеша)
    │
    ▼ Level 2: Redis (sync клиент, redis_client.py)
    │  Ключ: "caiu:emb:{md5(text)}"
    │  Значение: bytes через struct.pack("1536f", *vector)  ← компактный float32
    │  TTL: 2 592 000 сек (30 дней)
    │  Graceful fail: Redis недоступен → логируется один раз, работаем без кеша
    │
    ▼ OpenAI API (только при cache miss)
    │  Batch-запрос: все miss'ы за один API вызов
    │  Валидация: len(vector)==1536 and not all_zeros
    │
    ▼ Запись в оба кеша
```

### Кеш переводов (`bot/run.py`)

```
Казахский/английский текст
    │
    ▼ Level 1: in-memory OrderedDict (_TRANSLATION_CACHE)
    │  Ключ: (lang, text)
    │  Размер: 1000 слотов (LRU)
    │
    ▼ Level 2: Redis (async клиент, redis.asyncio)
    │  Ключ: "caiu:trans:{lang}:{md5(text)}"
    │  TTL: 30 дней
    │  Graceful fail: Redis недоступен → только in-memory
    │
    ▼ OpenAI GPT (только при cache miss)
    │
    ▼ Запись в оба кеша
```

**Почему sync Redis в RAG-сервисе:** FastAPI endpoint `/search` — это обычная (не async) функция, работающая в threadpool. Sync Redis работает нативно. Async Redis потребовал бы `asyncio.run()` что опасно в threadpool.

**Почему async Redis в боте:** весь бот написан на asyncio (aiogram). Sync Redis блокировал бы event loop.

---

## Streaming ответа

До оптимизации: бот отправлял сообщение после того как GPT закончил генерацию — задержка 2–5 секунд.

После: бот отправляет placeholder `⏳`, затем редактирует его по мере генерации.

```python
# Алгоритм stream_answer_to_message():

1. await message.answer("⏳")  → placeholder

2. client.chat.completions.create(stream=True, ...)

3. Цикл по токенам:
   collected += token
   
   # Первый edit — после _STREAM_FIRST_EDIT_CHARS символов (80)
   # Последующие — каждые _STREAM_EDIT_INTERVAL секунд (1.1)
   
   if time.now() - last_edit > 1.1s:
       await placeholder.edit_text(collected)

4. Финальный edit с parse_mode="HTML"
   Fallback: если Telegram вернул MarkupParseError → edit без parse_mode

5. Возврат: (full_answer, used_rag)
```

Ограничение Telegram: не более 20 правок сообщения в секунду (на весь бот), не более 1 в секунду на одно сообщение. Интервал 1.1 сек выбран с небольшим запасом.

---

## Структура данных в Qdrant

Каждая запись (Point) в векторной БД:

```python
{
    "id":           "uuid4-string",
    "vector":       [0.023, -0.14, ..., 0.087],  # 1536 float32

    "payload": {
        "text":          "оригинальный текст чанка",   # возвращается в GPT-контекст
        "page_url":      "https://caiu.edu.kz/...",
        "page_title":    "Название страницы",
        "section_title": "Название раздела h2/h3",
        "chunk_index":   0,              # порядковый номер чанка на странице
        "content_hash":  "md5-hex",      # для проверки изменений при переиндексации
        "embed_text":    "",             # пустая строка = обычный чанк
                                         # непустая = вопрос-чанк (vector вычислен из этого текста)
        "tags":          ["admission"],  # для ручных чанков из JSON
        "manually_edited": false,        # true = редактировался в chunk_editor
    }
}
```

**Оригинальный чанк:** `embed_text=""` — вектор вычислен из `text`. Возвращается при поиске.

**Вопрос-чанк:** `embed_text="Какие документы нужны для поступления?"` — вектор вычислен из вопроса, а `text` содержит оригинальный текст. При поиске вопросом пользователя этот вектор "ближе" → чанк находится, возвращается оригинальный `text`.

**Ожидаемое количество записей при 200 страницах:**
~400 текстовых чанков + ~1600 вопрос-чанков + ~100 каталог-чанков = **~2100 записей**

---

## Параметры системы

### Чанкинг (`app/config.py`)

| Параметр | Значение | Описание |
|---|---|---|
| `CHUNK_SIZE` | 200 слов | Максимальный размер чанка |
| `CHUNK_OVERLAP` | 30 слов | Перекрытие между соседними чанками |
| `REQUEST_DELAY` | 1.5 сек | Задержка между скачиванием страниц |

### Поиск (`app/config.py`)

| Параметр | Значение | Описание |
|---|---|---|
| `TOP_K_RESULTS` | 8 | Сколько брать из Qdrant до фильтрации |
| `MAX_CONTEXT_CHUNKS` | 5 | Сколько отдавать в GPT |
| `MIN_CONFIDENT_SCORE` | 0.28 | Ниже — "нет информации" |
| `SCORE_GAP_THRESHOLD` | 0.25 | Отсекаем хвост (score < best - 0.25) |
| `MAX_CHUNKS_PER_PAGE` | 2 | Максимум чанков с одной страницы |

### Модели OpenAI

| Назначение | Модель |
|---|---|
| Эмбеддинги | text-embedding-3-small (1536 dim) |
| Генерация ответов | gpt-4.1-mini |
| Генерация вопросов при индексации | gpt-4.1-mini |
| Перевод запросов | gpt-4.1-mini |

### Порты

| Сервис | Порт |
|---|---|
| RAG-сервис (FastAPI) | 8001 |
| Qdrant | 6333 |
| Redis | 6379 |
| Браузерный редактор чанков | 8080 |

### Кеш

| Кеш | Тип | Размер / TTL |
|---|---|---|
| Эмбеддинги in-memory | LRU dict | 512 слотов |
| Эмбеддинги Redis | bytes (struct float32) | TTL 30 дней |
| Переводы in-memory | OrderedDict | 1000 слотов |
| Переводы Redis | UTF-8 строки | TTL 30 дней |

### Timeout и retry (`bot/run.py`)

| Параметр | Значение |
|---|---|
| connect timeout | 5 сек |
| read timeout | 25 сек |
| Количество попыток | 3 |
| Backoff | 1с → 2с → 4с |
