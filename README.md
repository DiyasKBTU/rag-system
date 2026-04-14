# University Telegram Bot with RAG

Telegram-бот для абитуриентов университета с RAG-поиском (Retrieval-Augmented Generation). Бот отвечает на вопросы по информации с сайта университета, используя семантический поиск через векторную базу данных.

---

## Архитектура

```
Telegram User
    │
    ▼
Telegram Bot (aiogram)          ← bot/run.py
    │
    ▼ HTTP POST /api/ask/
Django Backend (DRF)            ← порт 8000
    │
    ├── POST /search ──►  RAG Service (FastAPI)   ← порт 8001
    │                          │
    │                     Qdrant (Docker)          ← порт 6333
    │
    └── fallback (если RAG пуст или недоступен)
              ├── Google Docs API
              └── парсинг сайта университета
```

**Django Backend** — принимает вопрос от бота, запрашивает контекст у RAG-сервиса, формирует промпт и отправляет в OpenAI GPT-4.1-mini.

**RAG Service** — индексирует страницы сайта: скачивает sitemap, парсит HTML, режет на чанки, создаёт векторы через OpenAI text-embedding-3-small, сохраняет в Qdrant. При поиске — расширяет запрос синонимами, делает multi-query поиск, фильтрует нерелевантные чанки.

---

## Структура репозитория

```
.
├── requirements.txt                        # ← единый файл зависимостей (установить один раз)
│
├── django-telegrambot-university-main/
│   ├── backend/                            # Django REST API
│   │   ├── config/settings.py
│   │   ├── apps/applicants/
│   │   │   ├── views.py                    # POST /api/ask/, GET /api/menu/
│   │   │   └── services/
│   │   │       ├── rag_client.py           # HTTP-клиент к RAG-сервису
│   │   │       └── google_docs.py          # Fallback: Google Drive API
│   │   └── manage.py
│   ├── bot/
│   │   ├── run.py                          # ← точка входа бота
│   │   └── apps/handlers.py
│   └── .env                               # заполнить своими данными
│
├── rag_service/
│   ├── app/
│   │   ├── main.py                         # FastAPI приложение
│   │   ├── config.py
│   │   ├── indexer/                        # embeddings + Qdrant storage
│   │   ├── parser/                         # crawler + extractor + chunker
│   │   └── retrieval/search.py             # семантический поиск
│   ├── run_api.py                          # ← точка входа FastAPI
│   ├── docker-compose.yml                  # Qdrant + Redis
│   └── .env                               # заполнить своими данными
│
└── test_speed.py
```

---

## Предварительные требования

- Python 3.11+
- Docker и Docker Compose
- Аккаунт OpenAI с доступом к API
- Telegram Bot Token (получить через @BotFather)
- Google Service Account с доступом к Google Drive (для fallback через Google Docs)

---

## Быстрый старт — полный запуск проекта

### Шаг 1. Клонировать репо, создать venv и настроить VS Code

```bash
# Из корня проекта (там где лежит этот README)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Установить все зависимости одной командой
pip install -r requirements.txt
```

> Виртуальное окружение создаётся один раз. При каждом новом сеансе достаточно только активировать его командой `venv\Scripts\activate` (Windows) или `source venv/bin/activate` (Mac/Linux).

**Настройка интерпретатора в VS Code (один раз после клонирования):**

Нажми `Ctrl+Shift+P` → `Python: Select Interpreter` → выбери интерпретатор из папки `venv` в корне проекта:
- Windows: `./venv/Scripts/python.exe`
- Mac/Linux: `./venv/bin/python`

Это нужно сделать вручную, потому что файл `.vscode/settings.json` намеренно исключён из git (он у каждого разработчика свой).

---

### Шаг 2. Заполнить файлы .env

**`rag_service/.env`:**

```env
OPENAI_API_KEY=sk-...                        # ключ OpenAI API
API_SECRET_KEY=придумай_любую_строку         # секрет для защиты API
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=caiu_knowledge_base
API_PORT=8001
SITE_BASE_URL=https://caiu.edu.kz
```

**`django-telegrambot-university-main/.env`:**

```env
SECRET_KEY=замени-на-случайную-строку-50-символов
DEBUG=True

# PostgreSQL
DB_NAME=university_bot
DB_USER=postgres
DB_PASSWORD=твой_пароль

# OpenAI
OPENAI_API_KEY=sk-...

# Telegram
BOT_TOKEN=токен_от_BotFather

# URL сервисов
BACKEND_URL=http://127.0.0.1:8000
RAG_URL=http://localhost:8001
RAG_API_KEY=тот_же_ключ_что_в_API_SECRET_KEY_rag_service   # ← должны совпадать!
RAG_TIMEOUT=5.0

# Google Docs (fallback, необязательно)
GOOGLE_DOC_ID=id_документа_из_url
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
```

---

### Шаг 3. Запустить Qdrant через Docker

```bash
cd rag_service
docker-compose up -d
cd ..
```

Qdrant поднимается на порту `6333`, Redis — на `6379`. Данные сохраняются в Docker volume и не теряются при перезапуске.

Проверить что Qdrant работает:

```bash
curl http://localhost:6333/healthz
```

---

### Шаг 4. Применить миграции Django

База данных — SQLite, файл `db.sqlite3` создаётся автоматически, ничего устанавливать не нужно.

```bash
cd django-telegrambot-university-main/backend
python manage.py migrate
python manage.py loaddata apps/applicants/fixtures/menu.json
python manage.py createsuperuser
cd ../..
```

`loaddata` загружает начальную структуру меню (Поступление, Специальности, Стоимость, Общежитие, Контакты, Задать вопрос). Редактировать пункты и добавлять контент можно через Django Admin на `http://localhost:8000/admin/`.

---

### Шаг 5. Запустить RAG-сервис (FastAPI) — Терминал 1

```bash
# Убедись что venv активирован
cd rag_service
python run_api.py
```

Сервер стартует на `http://localhost:8001`.
Swagger UI с документацией: `http://localhost:8001/docs`.

---

### Шаг 6. Запустить первичную индексацию сайта

> Делается один раз. Повторно запускать только при обновлении сайта.

```bash
# В том же терминале (или отдельном с активированным venv)
cd rag_service
python -m app.parser
```

Скрипт скачает sitemap.xml, спарсит страницы, нарежет на чанки ~400 слов, создаст векторы и сохранит в Qdrant. Занимает 10–30 минут. Прогресс выводится в консоль.

Проверить результат:

```bash
curl -H "X-API-Key: твой_API_SECRET_KEY" http://localhost:8001/stats
```

---

### Шаг 7. Запустить Django Backend — Терминал 2

```bash
# Новый терминал, активировать venv
cd django-telegrambot-university-main/backend
python manage.py runserver
```

Django Admin: `http://localhost:8000/admin/` — здесь создаётся структура меню бота (модель Menu).

---

### Шаг 8. Запустить Telegram-бот — Терминал 3

```bash
# Новый терминал, активировать venv
cd django-telegrambot-university-main/bot
python run.py
```

---

## Итого: три параллельных терминала

| Терминал | Команда | Что делает |
|---|---|---|
| **1** | `cd rag_service && python run_api.py` | FastAPI RAG-сервис на :8001 |
| **2** | `cd django-telegrambot-university-main/backend && python manage.py runserver` | Django Backend на :8000 |
| **3** | `cd django-telegrambot-university-main/bot && python run.py` | Telegram-бот |

Docker (Qdrant + Redis) работает в фоне и запускается один раз командой `docker-compose up -d`.

---

## Настройка Google Service Account (для fallback)

Если RAG-сервис недоступен или не находит релевантных чанков, Django автоматически переключается на Google Docs + парсинг сайта.

1. Создать Service Account в Google Cloud Console с ролью "Viewer".
2. Скачать JSON-ключ, переименовать в `service_account.json`, положить в `backend/config/`.
3. Расшарить нужный Google Doc на email сервисного аккаунта (только просмотр).
4. Указать ID документа в `.env` (из URL: `docs.google.com/document/d/<ID>/edit`).

---

## API RAG-сервиса

Все эндпоинты кроме `/health` требуют заголовок `X-API-Key`.

```
GET  /health       Проверка доступности (без авторизации)
GET  /stats        Количество чанков в Qdrant
POST /search       Семантический поиск по чанкам
POST /index        Запустить переиндексацию сайта
```

Пример запроса к `/search`:

```bash
curl -X POST http://localhost:8001/search \
  -H "X-API-Key: твой_ключ" \
  -H "Content-Type: application/json" \
  -d '{"question": "Как поступить в университет?", "top_k": 5}'
```

---

## API Django Backend

```
GET  /api/menu/?lang=ru    Дерево меню для бота (ru/kk/en)
POST /api/ask/             Задать вопрос, получить ответ от GPT
```

Пример ответа `/api/ask/`:

```json
{
  "answer": "Приём документов начинается с 20 июня...",
  "used_rag": true,
  "knowledge_source": "rag"
}
```

`used_rag: true` означает что ответ построен на основе Qdrant.

---

## Переиндексация

Запустить вручную через API (в фоне):

```bash
curl -X POST http://localhost:8001/index \
  -H "X-API-Key: твой_ключ" \
  -H "Content-Type: application/json" \
  -d '{"background": true}'
```

Повторная индексация пропускает страницы с неизменённым MD5-хешем. Изменённые страницы переиндексируются: старые чанки удаляются, создаются новые.

---

## Как работает поиск

1. **Нормализация** — убираются лишние пробелы.
2. **Query Expansion** — к вопросу добавляются синонимы из словаря (например, "специальности" → "образовательные программы").
3. **Batch embeddings** — все варианты запроса векторизуются одним запросом к OpenAI.
4. **Multi-query search** — каждый вариант ищется в Qdrant, результаты объединяются и дедублируются.
5. **Score gap filter** — отбрасываются чанки со score значительно ниже лучшего (разрыв > 0.18).
6. **Лимит контекста** — итоговый контекст обрезается до 6000 символов (~1500 токенов).

Если лучший результат имеет score < 0.35 — возвращается пустой список, Django переключается на fallback.

---

## Отладочные скрипты

В папке `rag_service/` есть скрипты для диагностики:

```
debug_search.py      Протестировать поиск по конкретному вопросу
debug_embeddings.py  Проверить работу OpenAI embeddings
debug_extractor.py   Посмотреть что парсится с конкретной страницы
debug_page.py        Отладка HTML-извлечения
debug_urls.py        Показать список URL из sitemap после фильтрации
clear_index.py       Полностью очистить коллекцию в Qdrant
```

---

## Переменные окружения

### `rag_service/.env`

| Переменная | Обязательная | Описание |
|---|---|---|
| `OPENAI_API_KEY` | да | Ключ OpenAI API |
| `API_SECRET_KEY` | да | Секретный ключ для защиты API |
| `QDRANT_HOST` | нет | Хост Qdrant (default: localhost) |
| `QDRANT_PORT` | нет | Порт Qdrant (default: 6333) |
| `QDRANT_COLLECTION_NAME` | нет | Название коллекции (default: caiu_knowledge_base) |
| `SITE_BASE_URL` | нет | URL сайта для парсинга |

### `django-telegrambot-university-main/.env`

| Переменная | Обязательная | Описание |
|---|---|---|
| `SECRET_KEY` | да | Django secret key |
| `OPENAI_API_KEY` | да | Ключ OpenAI API |
| `USE_POSTGRES` | нет | `True` чтобы переключиться на PostgreSQL (default: SQLite) |
| `DB_NAME` | если USE_POSTGRES | Имя базы PostgreSQL |
| `DB_USER` | если USE_POSTGRES | Пользователь PostgreSQL |
| `DB_PASSWORD` | если USE_POSTGRES | Пароль PostgreSQL |
| `BOT_TOKEN` | да | Telegram Bot Token |
| `BACKEND_URL` | да | URL Django (для бота) |
| `RAG_URL` | нет | URL RAG-сервиса (default: http://localhost:8001) |
| `RAG_API_KEY` | да | Должен совпадать с `API_SECRET_KEY` RAG-сервиса |
| `GOOGLE_DOC_ID` | нет | ID Google Doc для fallback |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | нет | Имя JSON-ключа сервисного аккаунта |

---

## Частые проблемы

**RAG-сервис не подключается к Qdrant**

```bash
docker ps
docker logs caiu_qdrant
```

**Бот не отвечает**

Проверьте что Django запущен. В логах `manage.py runserver` должны быть видны входящие запросы от бота.

**RAG возвращает пустой контекст**

```bash
curl -H "X-API-Key: ключ" http://localhost:8001/stats
```

Если `total_chunks: 0` — запустите `python -m app.parser` из папки `rag_service/`.

**Ошибка 401 от RAG-сервиса**

`RAG_API_KEY` в `.env` Django не совпадает с `API_SECRET_KEY` в `.env` rag_service. Значения должны быть идентичными.

**Google Docs: ошибка доступа**

Service Account не добавлен как читатель документа. Расшарьте Google Doc на email аккаунта вида `...@....iam.gserviceaccount.com`.
