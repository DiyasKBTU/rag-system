import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def get_google_doc_text() -> str:
    """
    Возвращает текст Google Doc или "" если:
    - service_account.json не настроен
    - GOOGLE_DOC_ID не указан
    - любая ошибка при обращении к Google API
    """
    sa_file = getattr(settings, "GOOGLE_SERVICE_ACCOUNT_FILE", "")
    doc_id = getattr(settings, "GOOGLE_DOC_ID", "")

    if not sa_file or not doc_id:
        logger.info("[GoogleDocs] Не настроен — пропускаем (GOOGLE_DOC_ID или service_account.json отсутствует)")
        return ""

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            sa_file,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        drive = build("drive", "v3", credentials=creds)
        data = (
            drive.files()
            .export_media(fileId=doc_id, mimeType="text/plain")
            .execute()
        )
        return data.decode("utf-8", errors="ignore")

    except FileNotFoundError:
        logger.warning(f"[GoogleDocs] Файл не найден: {sa_file}")
        return ""
    except Exception as e:
        logger.warning(f"[GoogleDocs] Ошибка: {e}")
        return ""
