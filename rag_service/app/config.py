"""
config.py — Все настройки RAG-сервиса в одном месте.

Как это работает:
- Настройки читаются из файла .env (секретные ключи и важные переменные)
- .env должен находиться в папке rag_service/ (rag_service/.env)
- Остальные параметры прописаны здесь напрямую
- Менять параметры можно здесь без правки других файлов

Где найти значения для .env:
- OPENAI_API_KEY — https://platform.openai.com/api-keys (ключ начинается с sk-)
- API_SECRET_KEY — любая секретная строка (придумайте сами)
"""

from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path
from dotenv import load_dotenv


def _year_exclude_patterns() -> List[str]:
    """
    Возвращает паттерны вида '/YYYY/' для исключения новостных URL по годам.

    Покрывает диапазон от 2019 до (текущий год + 1) — следующий год добавляется
    автоматически, чтобы в декабре не пришлось править config.py перед Новым годом.
    Раньше годы были хардкодом — приходилось добавлять `/2027/`, `/2028/` руками,
    про что легко забыть.
    """
    next_year = datetime.now().year + 1
    return [f"/{year}/" for year in range(2019, next_year + 1)]

# Путь к папке rag_service (где находится .env)
# __file__ = rag_service/app/config.py
# .parent  = rag_service/app
# .parent.parent = rag_service  ← здесь лежит .env
RAG_SERVICE_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = RAG_SERVICE_ROOT / ".env"

# Явно загружаем .env ДО создания Settings
# override=True — перезаписывает даже если переменная уже есть в окружении системы
load_dotenv(ENV_FILE, override=True)


class Settings(BaseSettings):
    """
    Класс настроек. Pydantic автоматически читает значения
    из файла .env и проверяет что они заполнены.
    """

    # ─── OPENAI ───────────────────────────────────────────────
    # Ключ OpenAI API — читается из .env
    OPENAI_API_KEY: str

    # Модель для создания векторов (embeddings)
    # text-embedding-3-small — дешёвая и качественная
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Размерность вектора для text-embedding-3-small
    # Это фиксированное значение модели, не менять
    EMBEDDING_DIMENSIONS: int = 1536

    # ─── QDRANT ───────────────────────────────────────────────
    # Адрес и порт Qdrant (Docker запускает на localhost)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Название "коллекции" в Qdrant
    # Коллекция = как таблица в обычной базе данных
    QDRANT_COLLECTION_NAME: str = "caiu_knowledge_base"

    # ─── FASTAPI ──────────────────────────────────────────────
    API_PORT: int = 8001
    API_SECRET_KEY: str

    # ─── САЙТ ─────────────────────────────────────────────────
    SITE_BASE_URL: str = "https://caiu.edu.kz"

    # URL до sitemap.xml — сайт сам его генерирует
    SITEMAP_URL: str = "https://caiu.edu.kz/sitemap.xml"

    # Задержка между запросами к сайту (секунды)
    # Нужно чтобы не перегружать сервер университета
    REQUEST_DELAY: float = 1.5

    # Таймаут одного запроса к сайту (секунды)
    REQUEST_TIMEOUT: float = 15.0

    # ─── ПАРСИНГ — ФИЛЬТРАЦИЯ URL ─────────────────────────────
    # Паттерны URL которые НЕ нужно индексировать
    # Если URL содержит любой из этих фрагментов — пропускаем.
    # Используем Field(default_factory=...) чтобы годы вычислялись динамически
    # при старте сервиса (см. _year_exclude_patterns выше).
    EXCLUDE_URL_PATTERNS: List[str] = Field(default_factory=lambda: [
        # ─── Языки и версии ───────────────────────────
        "/en/",           # Английская версия — не индексируем
        "/kz/",           # Казахская версия (старый префикс) — не используется на сайте
        # "/kk/" убран — казахские страницы индексируем (23 страницы с русскими парами)
        # Ненужные kk-страницы отфильтруются паттернами ниже (/news, /2024/ и т.д.)

        # ─── WordPress системные страницы ─────────────
        "wp-sitemap",     # Файлы индекса сайтмапа WordPress
        "/category/",     # Страницы категорий (списки постов, не контент)
        "/author/",       # Страницы авторов WordPress

        # ─── Системные и служебные ────────────────────
        "/platonus",      # Внешняя система обучения
        "/survey",        # Анкеты
        "/anketirovanie", # Анкетирование
        "/attachment",    # Файлы/вложения
        "/wp-admin",      # Админка WordPress
        "/wp-content",    # Системные файлы WordPress
        "/feed",          # RSS лента
        "/page/",         # Пагинация (стр. 2, 3...)
        "#",              # Якоря (#section)

        # ─── Новости и блог (НЕ нужны абитуриентам) ────
        "/news",          # Новости и новостные архивы
        "/blog",          # Блог
        "/media",         # Медиа/пресс-релизы
        "/press",         # Пресс-центр
        "/announcements",  # Объявления
        "/events",        # События/мероприятия
        "/calendar",      # Календарь событий

        # Годовые архивы: 2019…следующий_год — генерятся автоматически
        # _year_exclude_patterns() = ["/2019/", "/2020/", ..., "/<next>/"]
        *_year_exclude_patterns(),

        # ─── Внутренние/сотрудники (НЕ для абитуриентов) ──
        "/staff",         # Сотрудники
        "/teachers",      # Преподаватели (личные страницы)
        "/professor",     # Профессора
        "/employees",     # Сотрудники внутренние
        "/internal",      # Внутренние страницы
        "/hr",            # HR страницы

        # ─── Не релевантные разделы ───────────────────
        "/alumni",        # Выпускники
        "/research",      # Научные работы (подробно)
        "/shop",          # Магазин
        "/store",         # Магазин
        "/career",        # Вакансии (не для абитуриентов)
        "/jobs",          # Работа
        "/subscribe",     # Подписка
        "/checkout",      # Оплата
    ])

    # ─── GOOGLE DOCS ──────────────────────────────────────────────────────
    # Один документ — читается из .env (совместимо с Django настройками).
    # ID берётся из URL: docs.google.com/document/d/ВОТ_ЭТА_ЧАСТЬ/edit
    GOOGLE_DOC_ID: str = ""

    # Путь к service_account.json для приватных документов.
    # Оставьте пустым если документ публичный.
    GOOGLE_SERVICE_ACCOUNT_FILE: str = ""

    # Несколько документов — добавляются в коде (дополнительно к GOOGLE_DOC_ID).
    # Пример:
    # GOOGLE_DOC_IDS = [
    #   {"id": "1abc...xyz", "title": "Список специальностей"},
    # ]
    GOOGLE_DOC_IDS: List[dict] = []

    # ─── СПИСОК СТРАНИЦ ДЛЯ ИНДЕКСАЦИИ ───────────────────────
    # Только страницы полезные для абитуриентов.
    # 82 русских + 67 казахских = 149 страниц.
    # Отобраны вручную из 362 доступных (остальные — внутренние документы,
    # нормативы, новости, контент для сотрудников и студентов).
    MANUAL_TEST_URLS: List[str] = [

        # ══════════════════════════════════════════════════════
        # РУССКИЕ СТРАНИЦЫ (82)
        # ══════════════════════════════════════════════════════

        # ── Главная и общая информация ────────────────────────
        "https://caiu.edu.kz/",
        "https://caiu.edu.kz/postupayushhim/",
        "https://caiu.edu.kz/history-of-the-university-ru/",
        "https://caiu.edu.kz/mission-vision-ru/",
        "https://caiu.edu.kz/licenses-ru/",
        "https://caiu.edu.kz/naar-ru/",
        "https://caiu.edu.kz/struktura-universiteta/",
        "https://caiu.edu.kz/administration-ru/",
        "https://caiu.edu.kz/putevoditel-ru/",

        # ── Руководство ───────────────────────────────────────
        "https://caiu.edu.kz/rektor-czaiu/",
        "https://caiu.edu.kz/pervyj-prorektor-czaiu/",
        "https://caiu.edu.kz/prorektor-po-ur-czaiu/",
        "https://caiu.edu.kz/prorektor-po-umr-czaiu/",
        "https://caiu.edu.kz/prorektor-po-nir-czaiu/",
        "https://caiu.edu.kz/prorektor-po-vr-czaiu/",

        # ── Факультеты и кафедры ──────────────────────────────
        "https://caiu.edu.kz/faculties-ru/",
        "https://caiu.edu.kz/kaferdra-ru/",
        "https://caiu.edu.kz/estestvenno-nauchnyi/",
        "https://caiu.edu.kz/pedagogiki-i-biznesa/",
        "https://caiu.edu.kz/ru-business-and-law/",
        "https://caiu.edu.kz/tvorcheskiy/",
        "https://caiu.edu.kz/technologies-and-informatization-ru/",
        "https://caiu.edu.kz/chemistry-biology-and-ecologyru/",
        "https://caiu.edu.kz/arts-ru/",
        "https://caiu.edu.kz/management-and-finance-ru/",
        "https://caiu.edu.kz/sport-ru/",
        "https://caiu.edu.kz/prava-ru/",
        "https://caiu.edu.kz/business-and-tourism-ru/",
        "https://caiu.edu.kz/engineering-and-information-technology-rus/",
        "https://caiu.edu.kz/nvp-fk-ru/",
        "https://caiu.edu.kz/pedagogy-ru/",
        "https://caiu.edu.kz/languages-and-literature-ru/",

        # ── Образовательные программы ─────────────────────────
        "https://caiu.edu.kz/op/",
        "https://caiu.edu.kz/ru-bachelor/",
        "https://caiu.edu.kz/obrazovatelnye-programmy/",
        "https://caiu.edu.kz/groups-of-educational-programs-ru/",
        "https://caiu.edu.kz/bachelors-profile-subjects-ru/",
        "https://caiu.edu.kz/bachelors-degree-process-of-admission-ru/",
        "https://caiu.edu.kz/dualnoe-obuchenie/",
        "https://caiu.edu.kz/priemnaya-komissiya-dlya-magistratury/",

        # ── Специальности (коды) ──────────────────────────────
        "https://caiu.edu.kz/6b01409-ru/",
        "https://caiu.edu.kz/6b01501-ru/",
        "https://caiu.edu.kz/6b01509-ru/",
        "https://caiu.edu.kz/6b02101-ru/",
        "https://caiu.edu.kz/6b06103-ru/",
        "https://caiu.edu.kz/6b11104-ru/",

        # ── Специальности (bachelor-) ─────────────────────────
        "https://caiu.edu.kz/bachelor-law-ru/",
        "https://caiu.edu.kz/bachelor-customs-ru/",
        "https://caiu.edu.kz/https-caiu-edu-kz-bachelor-kaz-lang-ru/",
        "https://caiu.edu.kz/bachelor-foreign-language-ru/",
        "https://caiu.edu.kz/bachelor-nvp-ru/",
        "https://caiu.edu.kz/bachelor-sport-ru/",
        "https://caiu.edu.kz/bachelor-perevod-delo-ru/",
        "https://caiu.edu.kz/bachelor-gmu-ru/",
        "https://caiu.edu.kz/bachelor-uchet-audit-ru/",
        "https://caiu.edu.kz/bachelor-finance-ru/",
        "https://caiu.edu.kz/bachelor-turism-ru/",
        "https://caiu.edu.kz/bachelor-report-financial-analytics-ru/",

        # ── Поступление ───────────────────────────────────────
        "https://caiu.edu.kz/adminssions-ru/",
        "https://caiu.edu.kz/priem-online/",
        "https://caiu.edu.kz/priyomnaya-komissiya1/",
        "https://caiu.edu.kz/priem-msi/",
        "https://caiu.edu.kz/priem-mgti/",
        "https://caiu.edu.kz/priem-v-korpus-akademika-mardana-saparbaeva/",
        "https://caiu.edu.kz/list-of-documents-ru/",
        "https://caiu.edu.kz/doc-rus/",
        "https://caiu.edu.kz/ehreshold-scores-ru/",
        "https://caiu.edu.kz/creative-exams-ru/",
        "https://caiu.edu.kz/special-examination-ru/",
        "https://caiu.edu.kz/ent/",
        "https://caiu.edu.kz/testovyj-czentr/",
        "https://caiu.edu.kz/question-ru/",

        # ── Финансы ───────────────────────────────────────────
        "https://caiu.edu.kz/discounts-grants-ru/",
        "https://caiu.edu.kz/grants-and-discounts-ru/",

        # ── Военная кафедра ───────────────────────────────────
        "https://caiu.edu.kz/military-department-ru/",

        # ── Общежитие ─────────────────────────────────────────
        "https://caiu.edu.kz/obshhezhitie/",
        "https://caiu.edu.kz/stud-dom-ru/",
        "https://caiu.edu.kz/svobodnye-mesta-v-obshhezhitiyah-ocherednost-raspredeleniya/",

        # ── Контакты ──────────────────────────────────────────
        "https://caiu.edu.kz/contacts-ru/",
        "https://caiu.edu.kz/contacts-ru-2/",
        "https://caiu.edu.kz/kontakty/",
        "https://caiu.edu.kz/call-center/",

        # ══════════════════════════════════════════════════════
        # КАЗАХСКИЕ СТРАНИЦЫ (67)
        # ══════════════════════════════════════════════════════

        # ── Главная и общая информация ────────────────────────
        "https://caiu.edu.kz/kk/",
        "https://caiu.edu.kz/kk/history-of-the-university-kz/",
        "https://caiu.edu.kz/kk/mission-vision-kz/",
        "https://caiu.edu.kz/kk/licenses-kz/",
        "https://caiu.edu.kz/kk/akkreditaciya/",
        "https://caiu.edu.kz/kk/university-structure-kz/",
        "https://caiu.edu.kz/kk/kaz-guide/",

        # ── Руководство ───────────────────────────────────────
        "https://caiu.edu.kz/kk/rektor-kz/",
        "https://caiu.edu.kz/kk/1prorector-kk/",
        "https://caiu.edu.kz/kk/vice-rector-for-academic-workbot-kz/",
        "https://caiu.edu.kz/kk/vice-rector-for-emw-caiu-kk/",
        "https://caiu.edu.kz/kk/vice-rector-for-rcr-caiu-kz/",
        "https://caiu.edu.kz/kk/vice-rector-for-sdep-caiu-kz/",

        # ── Факультеты и кафедры ──────────────────────────────
        "https://caiu.edu.kz/kk/kk-faculty/",
        "https://caiu.edu.kz/kk/kafedralar/",
        "https://caiu.edu.kz/kk/busines-and-finance/",
        "https://caiu.edu.kz/kk/natural-technical-kz/",
        "https://caiu.edu.kz/kk/kk-pedagogy-tilder/",
        "https://caiu.edu.kz/kk/pedagogy-kz/",
        "https://caiu.edu.kz/kk/filologiya/",
        "https://caiu.edu.kz/kk/kaz-tarikh-zh-gylymi-pander/",
        "https://caiu.edu.kz/kk/quqyq/",
        "https://caiu.edu.kz/kk/business-and-tourism-kaz/",
        "https://caiu.edu.kz/kk/basqarw-zhaene-qarzhy/",
        "https://caiu.edu.kz/kk/mathematics-physics-and-cs/",
        "https://caiu.edu.kz/kk/at-zhaene-dizajn/",
        "https://caiu.edu.kz/kk/oner/",
        "https://caiu.edu.kz/kk/nvp-sport/",
        "https://caiu.edu.kz/kk/sport/",
        "https://caiu.edu.kz/kk/quqyq-kafedrasy/",
        "https://caiu.edu.kz/kk/ekonomika-zhaene-basqarw-kafedrasy/",
        "https://caiu.edu.kz/kk/creative-kk/",

        # ── Образовательные программы ─────────────────────────
        "https://caiu.edu.kz/kk/bb-kaz/",
        "https://caiu.edu.kz/kk/bakalavriat-2/",
        "https://caiu.edu.kz/kk/bilim-beru-baghdarlamalary/",
        "https://caiu.edu.kz/kk/bejindi-paender/",
        "https://caiu.edu.kz/kk/bb-kuzhat-kz/",
        "https://caiu.edu.kz/kk/opq-zhetistikteri/",
        "https://caiu.edu.kz/kk/zhoghary-oqw-ornynan-kejingi-bilim-berw-2/",
        # kaz-higher-education-postgraduate — нормативные документы магистратуры,
        # шумит в поиске по вопросам о специальностях — убрана
        # "https://caiu.edu.kz/kk/kaz-higher-education-postgraduate-education-3/",

        # ── Специальности (коды) ──────────────────────────────
        "https://caiu.edu.kz/kk/6b01409-kz/",
        "https://caiu.edu.kz/kk/6b01501-kz/",
        "https://caiu.edu.kz/kk/6b01509-kz/",
        "https://caiu.edu.kz/kk/6b02101-kz/",
        "https://caiu.edu.kz/kk/6b06103-kz/",
        "https://caiu.edu.kz/kk/6b11104-kz/",

        # ── Специальности (bachelor-) ─────────────────────────
        "https://caiu.edu.kz/kk/bachelor-kaz-lang-kz/",
        "https://caiu.edu.kz/kk/bachelor-foreign-language-kz/",
        "https://caiu.edu.kz/kk/bachelor-gmu-kz/",
        "https://caiu.edu.kz/kk/bachelor-uchet-audit-kz/",
        "https://caiu.edu.kz/kk/bachelor-finance-kz/",
        "https://caiu.edu.kz/kk/bachelor-turism-kz/",
        "https://caiu.edu.kz/kk/bachelor-report-financial-analytics-kz/",

        # ── Поступление ───────────────────────────────────────
        "https://caiu.edu.kz/kk/priem-oaiu/",
        "https://caiu.edu.kz/kk/bakalavriatqa-qabyldau-komissiyasy/",
        "https://caiu.edu.kz/kk/akademik-mardan-saparbaev-ghimaratyna-qabyldau/",
        "https://caiu.edu.kz/kk/quzhattar-tizimi/",
        "https://caiu.edu.kz/kk/shekti-balldar/",
        "https://caiu.edu.kz/kk/shygharmashylyq-emtihandar/",
        "https://caiu.edu.kz/kk/arnajy-emtihan/",
        "https://caiu.edu.kz/kk/test-ortalyghy/",

        # ── Финансы ───────────────────────────────────────────
        "https://caiu.edu.kz/kk/granttar-zhaene-zhengildikter/",
        "https://caiu.edu.kz/kk/granntar-kaz/",

        # ── Военная кафедра ───────────────────────────────────
        "https://caiu.edu.kz/kk/aeskeri-kafedra/",

        # ── Общежитие ─────────────────────────────────────────
        "https://caiu.edu.kz/kk/zhataqkhana/",

        # ── Контакты ──────────────────────────────────────────
        "https://caiu.edu.kz/kk/bajlanystar/",
        "https://caiu.edu.kz/kk/bajlanystar-2/",
        # "https://caiu.edu.kz/kaferdra-ru/",
        
    ]

    # ─── РАЗБИЕНИЕ ТЕКСТА НА ФРАГМЕНТЫ ───────────────────────
    # Размер одного фрагмента в словах.
    # 200 слов — оптимально: каждый чанк покрывает одну тему.
    # При 400 словах один чанк мог содержать историю, корпуса и
    # факультеты одновременно — поиск "сколько корпусов" его не находил.
    CHUNK_SIZE: int = 200

    # Перекрытие между соседними фрагментами в словах.
    # Нужно чтобы смысл не терялся на границах чанков.
    CHUNK_OVERLAP: int = 30

    # Минимальная длина фрагмента (символы)
    # Слишком короткие фрагменты — мусор, пропускаем
    MIN_CHUNK_LENGTH: int = 100

    # ─── ПОИСК ────────────────────────────────────────────────
    # Сколько фрагментов запрашивать из Qdrant (до фильтрации)
    # Берём с запасом — дальше отфильтруем по score gap
    TOP_K_RESULTS: int = 8

    # Сколько фрагментов максимально отдавать в контекст ChatGPT
    # Реальный лимит после score gap фильтрации
    MAX_CONTEXT_CHUNKS: int = 5

    # Минимальный порог косинусной близости (0.0 — 1.0)
    # Фрагменты ниже этого порога не возвращаются из Qdrant
    # 0.28 — снижено для лучшего recall по историческим/описательным темам
    # ("история университета", "о вузе" дают более размытые эмбеддинги)
    SIMILARITY_THRESHOLD: float = 0.28

    # Порог "уверенности" — если лучший найденный чанк ниже этого
    # значения, возвращаем пустой список (лучше без контекста,
    # чем с нерелевантным).
    # 0.28 — снижено с 0.32 чтобы не блокировать запросы типа
    # "расскажи об истории университета" (семантика ~0.29-0.31)
    MIN_CONFIDENT_SCORE: float = 0.28

    # Максимальный разрыв в score от лучшего результата.
    # Чанки, у которых score < (best_score - SCORE_GAP_THRESHOLD),
    # отбрасываются как нерелевантные.
    # Пример: best=0.75, gap=0.25 → отбрасываем всё ниже 0.50
    # 0.25 вместо 0.18 — чтобы не резать релевантные чанки когда
    # один результат случайно даёт аномально высокий score
    SCORE_GAP_THRESHOLD: float = 0.25

    # Максимальное количество символов в итоговом контексте для ChatGPT.
    # Предотвращает слишком большой контекст (дорого + путает модель).
    # ~6000 симв = ~1500 токенов — достаточно для списков специальностей/документов
    MAX_CONTEXT_CHARS: int = 6000

    # Включить расширение запроса (Query Expansion).
    # При True: автоматически добавляем синонимы/вариации для
    # университетских терминов (специальности, общежитие и т.д.)
    QUERY_EXPANSION_ENABLED: bool = True

    # ─── QUERY EXPANSION ──────────────────────────────────────────
    # Синонимы и вариации для университетских терминов.
    # Позволяет искать по нескольким вариантам одного запроса.
    # (словарь SYNONYM_MAP находится в search.py)

    class Config:
        # Откуда читать переменные окружения
        # Абсолютный путь к .env файлу (находится в rag_service/.env)
        env_file = str(ENV_FILE)
        # Разрешить дополнительные поля
        extra = "ignore"


# Создаём единственный экземпляр настроек
# Все модули импортируют именно его:
# from app.config import settings
settings = Settings()