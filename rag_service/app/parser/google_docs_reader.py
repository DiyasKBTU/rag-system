# -*- coding: utf-8 -*-
"""
google_docs_reader.py — Читает содержимое Google Docs для индексации в Qdrant.

Поддерживает два режима:
  1. Публичные документы ("Все у кого есть ссылка могут читать")
     → скачивает через export URL без авторизации
     → https://docs.google.com/document/d/{ID}/export?format=txt
     → применяет эвристику для определения заголовков

  2. Приватные документы
     → читает через Google Docs API с сервисным аккаунтом
     → получает структуру документа (стили HEADING_1, HEADING_2, ...) напрямую
     → точные заголовки без эвристики

Заголовки в обоих режимах конвертируются в ## маркеры (формат extractor.py),
что позволяет chunker.py разбивать документ по разделам, а не только по размеру.

Как добавить документ:
  1. Откройте Google Doc
  2. Скопируйте ID из URL: docs.google.com/document/d/THIS_PART/edit
  3. Добавьте в config.py → GOOGLE_DOC_IDS

Как получить ID:
  URL:  https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/edit
  ID:   1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
"""

import re
import logging
import httpx
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Таймаут на скачивание Google Doc (может быть медленнее чем обычный сайт)
GOOGLE_DOCS_TIMEOUT = 20.0

# Паттерн для определения маркеров списка (НЕ заголовков)
_BULLET_RE = re.compile(r"^[-*•–—·]\s")


@dataclass
class GoogleDocContent:
    """Содержимое одного Google документа."""
    doc_id: str
    title: str       # Название документа (из конфига или из содержимого)
    text: str        # Полный текст документа (с ## маркерами заголовков)
    source_url: str  # Публичный URL для атрибуции в ответах


# ── Эвристическое определение заголовков (для публичных документов) ───────────

def _is_heading_candidate(stripped: str, lines: list, i: int) -> bool:
    """
    Проверяет: является ли строка заголовком раздела.

    Правила:
      1. ВСЕ ЗАГЛАВНЫЕ буквы с ≥ 2 словами — типичный стиль официальных
         русских документов (УСЛОВИЯ ПРИЁМА, ОБЩИЕ ПОЛОЖЕНИЯ и т.д.)
      2. Короткая изолированная строка (окружена пустыми строками, ≤ 60 символов)
         — стандартный формат заголовка в Google Docs при экспорте
      3. Строка, оканчивающаяся на ":" и предшествующая пустой строке
         — подзаголовок типа "Необходимые документы:"
    """
    n = len(lines)

    # Слишком короткая или длинная — точно не заголовок
    if len(stripped) < 3 or len(stripped) > 100:
        return False

    # Маркер списка — пропускаем (дефис, звёздочка, маркированный список)
    if _BULLET_RE.match(stripped):
        return False

    words = stripped.split()
    prev_blank = (i == 0) or not lines[i - 1].strip()
    next_blank = (i >= n - 1) or not lines[i + 1].strip()

    # Правило 1: ВСЕ ЗАГЛАВНЫЕ с ≥ 2 значимыми словами
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if (len(alpha_words) >= 2
            and stripped.upper() == stripped
            and not stripped.endswith(".")):
        return True

    # Правило 2: Короткая строка, изолированная пустыми строками
    if prev_blank and next_blank and len(stripped) <= 60:
        # Не заканчивается на . ! , — это предложения, а не заголовки
        if stripped[-1] not in ".!,":
            return True

    # Правило 3: Подзаголовок с двоеточием, предшествует пустой строке
    if stripped.endswith(":") and len(stripped) <= 60 and prev_blank:
        return True

    return False


def _inject_heading_markers_from_text(text: str) -> str:
    """
    Обнаруживает заголовки в тексте эвристически и вставляет ## маркеры.

    Применяется для публичных Google Docs (экспорт в .txt теряет стили).

    Примеры того, что станет заголовком:
      "УСЛОВИЯ ПРИЁМА"          → "## УСЛОВИЯ ПРИЁМА"    (ALL CAPS)
      "Стоимость обучения"      → "## Стоимость обучения" (изолированная короткая)
      "Необходимые документы:"  → "## Необходимые документы:"  (с двоеточием)

    Примеры того, что НЕ станет заголовком:
      "- пункт списка"          → без изменений (маркер списка)
      "Студент обязан сдать..."  → без изменений (предложение с точкой)
      "Длинная строка текста которая не является заголовком" → без изменений (> 60 символов)
    """
    lines = text.splitlines()
    result = []

    for i, raw_line in enumerate(lines):
        stripped = raw_line.strip()

        if not stripped:
            result.append(raw_line)
            continue

        # Уже содержит ## (на всякий случай)
        if stripped.startswith("## "):
            result.append(raw_line)
            continue

        if _is_heading_candidate(stripped, lines, i):
            result.append(f"## {stripped}")
        else:
            result.append(raw_line)

    return "\n".join(result)


# ── Чтение структуры через Google Docs API (для приватных документов) ─────────

def _read_doc_with_structure(doc_id: str, creds) -> Optional[str]:
    """
    Читает Google Doc через Docs API (v1) и возвращает текст с ## маркерами.

    Использует стили абзацев (HEADING_1, HEADING_2, TITLE и т.д.) напрямую
    — никакой эвристики, точные заголовки как в оригинальном документе.

    В отличие от экспорта в .txt, Docs API возвращает структуру:
      - paragraph.paragraphStyle.namedStyleType = "HEADING_2"
      - paragraph.elements[].textRun.content = "Условия поступления"
    → "## Условия поступления"

    Args:
        doc_id: ID документа
        creds:  Credentials (google.oauth2.service_account.Credentials)

    Returns:
        Текст с ## маркерами, или None если API недоступен (будет fallback на txt export)
    """
    try:
        from googleapiclient.discovery import build

        HEADING_STYLES = {
            "TITLE", "SUBTITLE",
            "HEADING_1", "HEADING_2", "HEADING_3", "HEADING_4", "HEADING_5", "HEADING_6",
        }

        docs_service = build("docs", "v1", credentials=creds)
        doc = docs_service.documents().get(documentId=doc_id).execute()

        lines = []
        for element in doc.get("body", {}).get("content", []):
            if "paragraph" not in element:
                continue

            para = element["paragraph"]
            style_type = (
                para.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
            )

            # Собираем текст из всех textRun элементов абзаца
            para_text = ""
            for elem in para.get("elements", []):
                if "textRun" in elem:
                    para_text += elem["textRun"].get("content", "")

            # Google Docs добавляет \n в конце каждого абзаца — убираем
            para_text = para_text.rstrip("\n")

            if not para_text.strip():
                lines.append("")
                continue

            if style_type in HEADING_STYLES:
                lines.append(f"## {para_text.strip()}")
            else:
                lines.append(para_text)

        full_text = "\n".join(lines)
        logger.info(
            f"[GoogleDocs] Docs API: получена структура для {doc_id} "
            f"({full_text.count('## ')} заголовков)"
        )
        return full_text

    except Exception as e:
        logger.warning(
            f"[GoogleDocs] Docs API недоступен для {doc_id}: {e}. "
            f"Переключаемся на txt-экспорт + эвристику."
        )
        return None


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _extract_title_from_text(text: str, doc_id: str) -> str:
    """
    Пробует извлечь заголовок из первой непустой строки текста.
    Если не получается — возвращает заглушку.
    """
    for line in text.splitlines():
        # Убираем ## маркер если строка уже обработана
        line = line.strip().lstrip("# ").strip()
        if len(line) > 3:
            return line[:100]  # Первые 100 символов первой строки
    return f"Google Doc {doc_id[:8]}..."


# ── Публичные функции чтения ──────────────────────────────────────────────────

def read_public_doc(doc_id: str, title: str = "") -> Optional[GoogleDocContent]:
    """
    Читает публичный Google Doc через export URL.

    Работает если документ открыт для всех у кого есть ссылка.
    Не требует авторизации и API ключей.

    После получения текста применяет эвристическое определение заголовков
    (_inject_heading_markers_from_text), чтобы chunker.py мог разбить документ
    по разделам, а не только по количеству слов.

    Args:
        doc_id: ID документа из URL
        title:  Название для атрибуции (если пустое — берём из текста)

    Returns:
        GoogleDocContent или None если не удалось прочитать
    """
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    source_url = f"https://docs.google.com/document/d/{doc_id}/view"

    try:
        with httpx.Client(timeout=GOOGLE_DOCS_TIMEOUT, follow_redirects=True) as client:
            response = client.get(export_url)

        if response.status_code == 200:
            raw_text = response.text.strip()
            if not raw_text:
                logger.warning(f"[GoogleDocs] Пустой документ: {doc_id}")
                return None

            # Вставляем ## маркеры заголовков эвристически
            text = _inject_heading_markers_from_text(raw_text)
            heading_count = text.count("\n## ")

            doc_title = title or _extract_title_from_text(raw_text, doc_id)
            logger.info(
                f"[GoogleDocs] Прочитан публичный документ: '{doc_title}' "
                f"({len(raw_text)} символов, {heading_count} заголовков обнаружено)"
            )
            return GoogleDocContent(
                doc_id=doc_id,
                title=doc_title,
                text=text,
                source_url=source_url,
            )

        elif response.status_code in (401, 403):
            logger.warning(
                f"[GoogleDocs] Документ {doc_id} закрыт — попробуйте private режим "
                f"(service_account.json) или откройте доступ по ссылке"
            )
            return None

        elif response.status_code == 404:
            logger.warning(f"[GoogleDocs] Документ не найден: {doc_id}")
            return None

        else:
            logger.warning(f"[GoogleDocs] HTTP {response.status_code} для {doc_id}")
            return None

    except httpx.TimeoutException:
        logger.warning(f"[GoogleDocs] Таймаут при чтении: {doc_id}")
        return None
    except Exception as e:
        logger.error(f"[GoogleDocs] Ошибка при чтении {doc_id}: {e}")
        return None


def read_private_doc(
    doc_id: str,
    service_account_file: str,
    title: str = "",
) -> Optional[GoogleDocContent]:
    """
    Читает приватный Google Doc через API с сервисным аккаунтом.

    Сначала пробует Google Docs API (documents().get()) для точного
    извлечения структуры заголовков. Если API недоступен — fallback
    на Drive API text export + эвристика.

    Требует:
      1. Google Cloud проект с включёнными Google Drive API и Google Docs API
      2. Сервисный аккаунт с JSON ключом
      3. Документ расшарен на email сервисного аккаунта

    Args:
        doc_id:               ID документа
        service_account_file: Путь к service_account.json
        title:                Название для атрибуции

    Returns:
        GoogleDocContent или None если не удалось прочитать
    """
    source_url = f"https://docs.google.com/document/d/{doc_id}/view"

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=[
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/documents.readonly",
            ],
        )

        # ── Вариант 1: Google Docs API — точная структура заголовков ──
        structured_text = _read_doc_with_structure(doc_id, creds)

        if structured_text and structured_text.strip():
            raw_text = re.sub(r"^## ", "", structured_text, flags=re.MULTILINE)
            doc_title = title or _extract_title_from_text(raw_text, doc_id)
            heading_count = structured_text.count("\n## ")
            logger.info(
                f"[GoogleDocs] Прочитан приватный документ (Docs API): '{doc_title}' "
                f"({len(structured_text)} символов, {heading_count} заголовков)"
            )
            return GoogleDocContent(
                doc_id=doc_id,
                title=doc_title,
                text=structured_text,
                source_url=source_url,
            )

        # ── Вариант 2: Drive API text export + эвристика ──────────────
        logger.info(f"[GoogleDocs] Fallback: Drive API text export для {doc_id}")
        drive = build("drive", "v3", credentials=creds)
        data = (
            drive.files()
            .export_media(fileId=doc_id, mimeType="text/plain")
            .execute()
        )
        raw_text = data.decode("utf-8", errors="ignore").strip()

        if not raw_text:
            logger.warning(f"[GoogleDocs] Пустой документ: {doc_id}")
            return None

        text = _inject_heading_markers_from_text(raw_text)
        heading_count = text.count("\n## ")
        doc_title = title or _extract_title_from_text(raw_text, doc_id)
        logger.info(
            f"[GoogleDocs] Прочитан приватный документ (Drive API): '{doc_title}' "
            f"({len(raw_text)} символов, {heading_count} заголовков через эвристику)"
        )
        return GoogleDocContent(
            doc_id=doc_id,
            title=doc_title,
            text=text,
            source_url=source_url,
        )

    except FileNotFoundError:
        logger.error(f"[GoogleDocs] service_account.json не найден: {service_account_file}")
        return None
    except ImportError:
        logger.error("[GoogleDocs] Установите: pip install google-api-python-client google-auth")
        return None
    except Exception as e:
        logger.error(f"[GoogleDocs] Ошибка при чтении {doc_id}: {e}")
        return None


def read_google_doc(
    doc_id: str,
    title: str = "",
    service_account_file: str = "",
) -> Optional[GoogleDocContent]:
    """
    Главная функция — читает Google Doc публично или через API.

    Сначала пробует публичный режим. Если не получилось и указан
    service_account_file — пробует через API (с точной структурой заголовков).

    В обоих случаях текст в GoogleDocContent.text содержит ## маркеры заголовков
    — в том же формате, что extractor.py генерирует для HTML страниц сайта.
    Это позволяет chunker.py разбивать Google Doc по разделам (section-aware chunking).

    Args:
        doc_id:               ID документа из URL
        title:                Название (опционально)
        service_account_file: Путь к service_account.json (для приватных)

    Returns:
        GoogleDocContent или None
    """
    # Сначала пробуем публичный доступ
    result = read_public_doc(doc_id, title)
    if result:
        return result

    # Если не получилось и есть сервисный аккаунт — пробуем через API
    if service_account_file:
        logger.info(f"[GoogleDocs] Пробуем через service account: {doc_id}")
        return read_private_doc(doc_id, service_account_file, title)

    return None
