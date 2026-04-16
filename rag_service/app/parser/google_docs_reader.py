# -*- coding: utf-8 -*-
"""
google_docs_reader.py — Читает содержимое Google Docs для индексации в Qdrant.

Поддерживает два режима:
  1. Публичные документы ("Все у кого есть ссылка могут читать")
     → скачивает через export URL без авторизации
     → https://docs.google.com/document/d/{ID}/export?format=txt

  2. Приватные документы
     → читает через Google Docs API с сервисным аккаунтом
     → требует service_account.json и разрешения в Google Cloud

Как добавить документ:
  1. Откройте Google Doc
  2. Скопируйте ID из URL: docs.google.com/document/d/THIS_PART/edit
  3. Добавьте в config.py → GOOGLE_DOC_IDS

Как получить ID:
  URL:  https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/edit
  ID:   1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms
"""

import logging
import httpx
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Таймаут на скачивание Google Doc (может быть медленнее чем обычный сайт)
GOOGLE_DOCS_TIMEOUT = 20.0


@dataclass
class GoogleDocContent:
    """Содержимое одного Google документа."""
    doc_id: str
    title: str       # Название документа (из конфига или из содержимого)
    text: str        # Полный текст документа
    source_url: str  # Публичный URL для атрибуции в ответах


def _extract_title_from_text(text: str, doc_id: str) -> str:
    """
    Пробует извлечь заголовок из первой непустой строки текста.
    Если не получается — возвращает заглушку.
    """
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 3:
            return line[:100]  # Первые 100 символов первой строки
    return f"Google Doc {doc_id[:8]}..."


def read_public_doc(doc_id: str, title: str = "") -> Optional[GoogleDocContent]:
    """
    Читает публичный Google Doc через export URL.

    Работает если документ открыт для всех у кого есть ссылка.
    Не требует авторизации и API ключей.

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
            text = response.text.strip()
            if not text:
                logger.warning(f"[GoogleDocs] Пустой документ: {doc_id}")
                return None

            doc_title = title or _extract_title_from_text(text, doc_id)
            logger.info(f"[GoogleDocs] Прочитан публичный документ: '{doc_title}' ({len(text)} символов)")
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

    Требует:
      1. Google Cloud проект с включённым Google Drive API
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
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive = build("drive", "v3", credentials=creds)
        data = (
            drive.files()
            .export_media(fileId=doc_id, mimeType="text/plain")
            .execute()
        )
        text = data.decode("utf-8", errors="ignore").strip()

        if not text:
            logger.warning(f"[GoogleDocs] Пустой документ: {doc_id}")
            return None

        doc_title = title or _extract_title_from_text(text, doc_id)
        logger.info(f"[GoogleDocs] Прочитан приватный документ: '{doc_title}' ({len(text)} символов)")
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
    service_account_file — пробует через API.

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
