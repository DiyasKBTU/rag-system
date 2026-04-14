"""
config.py — Все настройки RAG-сервиса в одном месте.

Как это работает:
- Настройки читаются из файла .env (секретные ключи)
- Остальные параметры прописаны здесь напрямую
- Менять параметры можно здесь без правки других файлов
"""

from pydantic_settings import BaseSettings
from typing import List


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
    # Если URL содержит любой из этих фрагментов — пропускаем
    EXCLUDE_URL_PATTERNS: List[str] = [
        # ─── Языки и версии ───────────────────────────
        "/en/",           # Английская версия
        "/kz/",           # Казахская версия — не индексируем, переводим на лету
        "/kk/",           # Казахская версия (альтернативный префикс)

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
        "/calendar",
        "/2026/",
        "/2025/",# Календарь событий
        "/2024/",         # Новости по годам (любой год)
        "/2023/",
        "/2022/",
        "/2021/",
        "/2020/",
        "/2019/",

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
    ]

    # ─── ТЕСТОВЫЙ РЕЖИМ — РУЧНЫЕ URL ──────────────────────────
    # Для тестирования: добавь сюда несколько URL
    # Если список НЕ пустой — парсятся ТОЛЬКО эти URL
    # Когда тестирование завершено — очисти список []
    MANUAL_TEST_URLS: List[str] = [
        # "https://caiu.edu.kz/ru/",
        # "https://caiu.edu.kz/ru/contacts/",
        # "https://caiu.edu.kz/ru/faculties/",
        # "https://caiu.edu.kz/contacts-ru-2/",
        # "https://caiu.edu.kz/military-department-ru/",
        # "https://caiu.edu.kz/grants-and-discounts-ru/",
        # "https://caiu.edu.kz/obshhezhitie/",
        # "https://caiu.edu.kz/special-examination-ru/",
        # "https://caiu.edu.kz/creative-exams-ru/",
        # # Google Drive links removed — they cannot be parsed (access denied)
        # "https://caiu.edu.kz/ehreshold-scores-ru/",
        # "https://caiu.edu.kz/list-of-documents-ru/",
        # "https://caiu.edu.kz/bachelors-profile-subjects-ru/",
        # "https://caiu.edu.kz/groups-of-educational-programs-ru/",
        # "https://caiu.edu.kz/bachelors-degree-process-of-admission-ru/",
        # "https://caiu.edu.kz/adminssions-ru/",
        # "https://caiu.edu.kz/op/",
        # # Google Drive links removed — they cannot be parsed (access denied)
        # "https://caiu.edu.kz/mission-vision-ru/",
        # "https://caiu.edu.kz/licenses-ru/",
        # "https://caiu.edu.kz/history-of-the-university-ru/",
        # "https://caiu.edu.kz/pervyj-prorektor-czaiu/",
        # "https://caiu.edu.kz/rektor-czaiu/",
        # "https://caiu.edu.kz/prorektor-po-ur-czaiu/",
        # "https://caiu.edu.kz/prorektor-po-umr-czaiu/",
        # "https://caiu.edu.kz/prorektor-po-nir-czaiu/",
        # "https://caiu.edu.kz/prorektor-po-vr-czaiu/",
        # "https://caiu.edu.kz/otdely/",
        # "https://caiu.edu.kz/faculties-ru/",
        # "https://caiu.edu.kz/kaferdra-ru/",
        
    ]

    # ─── РАЗБИЕНИЕ ТЕКСТА НА ФРАГМЕНТЫ ───────────────────────
    # Размер одного фрагмента в словах
    # 400-500 слов — оптимально для RAG
    CHUNK_SIZE: int = 400

    # Перекрытие между соседними фрагментами в словах
    # Нужно чтобы смысл не терялся на границах
    CHUNK_OVERLAP: int = 50

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
    # 0.30 — снижено с 0.35 для лучшего recall по темам
    # "история", "стоимость", "о вузе" (дают чуть более размытые эмбеддинги)
    SIMILARITY_THRESHOLD: float = 0.30

    # Порог "уверенности" — если лучший найденный чанк ниже этого
    # значения, возвращаем пустой список (лучше без контекста,
    # чем с нерелевантным).
    # 0.32 — достаточно, чтобы отсечь совсем нерелевантные результаты
    MIN_CONFIDENT_SCORE: float = 0.32

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

    class Config:
        # Откуда читать переменные окружения
        env_file = ".env"
        # Разрешить дополнительные поля
        extra = "ignore"


# Создаём единственный экземпляр настроек
# Все модули импортируют именно его:
# from app.config import settings
settings = Settings()