# -*- coding: utf-8 -*-
"""
tools/chunk_editor.py — Browser-based Qdrant chunk editor.

Standalone FastAPI server (port 8080) for viewing and editing
chunks stored in Qdrant directly from a web browser.

Features:
  - Left sidebar: all indexed pages with chunk counts + URL/title search
    with highlighted matches and virtual scroll (loads in batches of 50)
  - Right panel: chunk cards with inline editing
  - Per-chunk: edit text, section title, tags, questions
  - Per-chunk save (re-embeds via OpenAI) and delete
  - Per-chunk history with restore (last 20 versions) + word-level diff
  - Add new chunk to any page
  - Bulk select: delete or re-tag multiple chunks at once
  - Auto-generate questions via GPT (robot button on each card)
  - Dirty-state tracking: orange border + browser unload warning
  - Optional password auth: set CHUNK_EDITOR_PASSWORD in env

Usage:
  cd rag_service
  python ../tools/chunk_editor.py

Then open: http://localhost:8080

Requirements: fastapi, uvicorn (already in requirements.txt)
"""

import os
import sys
import json
import uuid
import time
import secrets
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# -- Path setup: import from rag_service --------------------------------------
_RAG_SERVICE = Path(__file__).resolve().parent.parent / "rag_service"
sys.path.insert(0, str(_RAG_SERVICE))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from app.config import settings
from app.indexer.embeddings import get_embeddings_batch
from app.indexer.storage import get_client, get_active_collection
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Auth config ---------------------------------------------------------------
# Set CHUNK_EDITOR_PASSWORD in your .env to enable password protection.
# If empty, the editor is accessible without authentication.
CHUNK_EDITOR_PASSWORD = os.getenv("CHUNK_EDITOR_PASSWORD", "")
_sessions: set = set()          # valid session tokens (in-memory)

# -- Rate limiting -------------------------------------------------------------
_rate_data: Dict[str, List[float]] = {}   # "ip:action" -> [timestamps]

# -- History file --------------------------------------------------------------
HISTORY_FILE = Path(__file__).resolve().parent.parent / "chunk_editor_history.json"
MAX_HISTORY_PER_CHUNK = 20
HISTORY_MAX_AGE_DAYS = 90   # entries older than this are pruned on load

AVAILABLE_TAGS = [
    "admission", "dormitory", "fees", "grants", "specialties",
    "faculty", "department", "contacts", "military", "exams",
    "history", "management", "licenses", "general",
]

app = FastAPI(title="Chunk Editor", docs_url=None, redoc_url=None)


# -- Auth helpers --------------------------------------------------------------

def require_auth(request: Request) -> None:
    """Raise 401 if password is configured and session cookie is missing/invalid."""
    if not CHUNK_EDITOR_PASSWORD:
        return  # auth disabled
    token = request.cookies.get("ce_session")
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="Требуется авторизация")


def _check_rate(ip: str, action: str = "write", limit: int = 60, window: int = 60) -> None:
    """Simple in-memory rate limiter -- raises 429 if limit exceeded."""
    now = time.monotonic()
    key = f"{ip}:{action}"
    times = [t for t in _rate_data.get(key, []) if now - t < window]
    if len(times) >= limit:
        raise HTTPException(status_code=429, detail="Слишком много запросов. Подождите немного.")
    times.append(now)
    _rate_data[key] = times


# -- History helpers -----------------------------------------------------------

def _cleanup_history(history: Dict) -> Dict:
    """Remove history entries older than HISTORY_MAX_AGE_DAYS."""
    cutoff = datetime.now() - timedelta(days=HISTORY_MAX_AGE_DAYS)
    cleaned = {}
    for key, versions in history.items():
        fresh = []
        for v in versions:
            try:
                ts = datetime.strptime(v["timestamp"], "%Y-%m-%d %H:%M:%S")
                if ts >= cutoff:
                    fresh.append(v)
            except (KeyError, ValueError):
                fresh.append(v)   # keep entries with malformed timestamps
        if fresh:
            cleaned[key] = fresh
    return cleaned


def load_history() -> Dict:
    if HISTORY_FILE.exists():
        try:
            raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return _cleanup_history(raw)
        except Exception:
            pass
    return {}


def save_history_file(history: Dict) -> None:
    """Atomic write: write to .tmp then rename to avoid corruption on crash."""
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(HISTORY_FILE)


def add_to_history(page_url: str, chunk_index: int, snapshot: dict) -> None:
    history = load_history()
    key = f"{page_url}::{chunk_index}"
    if key not in history:
        history[key] = []
    history[key].insert(0, snapshot)
    history[key] = history[key][:MAX_HISTORY_PER_CHUNK]
    save_history_file(history)


# -- Qdrant helpers ------------------------------------------------------------

def scroll_all_for_url(page_url: str, collection: str) -> List[Any]:
    """Paginate through all Qdrant points for a given page URL."""
    client = get_client()
    all_points = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="page_url", match=MatchValue(value=page_url))
            ]),
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def scroll_all_points(collection: str) -> List[Any]:
    """Paginate through ALL Qdrant points in the collection."""
    client = get_client()
    all_points = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def delete_chunk_points(page_url: str, chunk_index: int, collection: str) -> None:
    """Delete all Qdrant points (text + question) for a given (url, chunk_index)."""
    client = get_client()
    all_points = scroll_all_for_url(page_url, collection)
    ids_to_delete = [
        str(p.id) for p in all_points
        if p.payload.get("chunk_index") == chunk_index
    ]
    if ids_to_delete:
        client.delete(collection_name=collection, points_selector=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} points for ({page_url}, idx={chunk_index})")


def build_points(
    page_url: str,
    page_title: str,
    chunk_index: int,
    section: str,
    text: str,
    tags: List[str],
    questions: List[str],
) -> List[PointStruct]:
    """Build Qdrant PointStructs for one text chunk + its question chunks."""
    if not text.strip():
        return []

    clean_questions = [q.strip() for q in questions if q.strip()]
    texts_to_embed = [text] + clean_questions
    vectors = get_embeddings_batch(texts_to_embed)

    if len(vectors) != len(texts_to_embed):
        logger.error(f"Embedding mismatch: {len(texts_to_embed)} texts -> {len(vectors)} vectors")
        return []

    base_payload = {
        "page_url": page_url,
        "page_title": page_title,
        "section_title": section,
        "chunk_index": chunk_index,
        "content_hash": "",
        "tags": tags,
        "manually_edited": True,
    }

    points = [PointStruct(
        id=str(uuid.uuid4()),
        vector=vectors[0],
        payload={**base_payload, "text": text, "embed_text": ""},
    )]
    for question, vector in zip(clean_questions, vectors[1:]):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={**base_payload, "text": text, "embed_text": question},
        ))
    return points


# -- API request models --------------------------------------------------------

class AuthLoginRequest(BaseModel):
    password: str


class SaveRequest(BaseModel):
    chunk_id: str
    page_url: str
    page_title: str
    chunk_index: int
    section: str
    text: str
    tags: List[str]
    questions: List[str]


class DeleteRequest(BaseModel):
    page_url: str
    chunk_index: int


class NewChunkRequest(BaseModel):
    page_url: str
    page_title: str
    section: str
    text: str
    tags: List[str]
    questions: List[str]


class RestoreRequest(BaseModel):
    url: str
    index: int
    version_index: int


class GenerateQuestionsRequest(BaseModel):
    text: str
    page_url: str = ""
    page_title: str = ""
    section: str = ""


class BulkDeleteRequest(BaseModel):
    items: List[Dict]   # [{page_url, chunk_index}, ...]


class BulkTagRequest(BaseModel):
    items: List[Dict]   # [{page_url, page_title, chunk_index, section, text, questions, tags}, ...]
    tags: List[str]
    mode: str = "set"   # "set" = replace all tags, "add" = add to existing


class IndexPageRequest(BaseModel):
    url: str
    with_questions: bool = True


# -- Auth routes ---------------------------------------------------------------

@app.post("/auth/login")
async def auth_login(req: AuthLoginRequest, response: Response):
    if not CHUNK_EDITOR_PASSWORD:
        return {"ok": True, "message": "Аутентификация отключена"}
    if req.password != CHUNK_EDITOR_PASSWORD:
        raise HTTPException(status_code=403, detail="Неверный пароль")
    token = secrets.token_hex(32)
    _sessions.add(token)
    response.set_cookie(
        "ce_session", token,
        httponly=True, max_age=86400 * 7,
        samesite="strict",
    )
    return {"ok": True}


@app.post("/auth/logout")
async def auth_logout(response: Response):
    response.delete_cookie("ce_session")
    return {"ok": True}


@app.get("/auth/status")
def auth_status(request: Request):
    if not CHUNK_EDITOR_PASSWORD:
        return {"required": False, "authenticated": True}
    token = request.cookies.get("ce_session")
    return {"required": True, "authenticated": bool(token and token in _sessions)}


# -- API routes ----------------------------------------------------------------

@app.get("/api/collection")
def api_collection(request: Request):
    require_auth(request)
    return {"collection": get_active_collection()}


@app.get("/api/pages")
def api_pages(request: Request):
    """Return all pages with text/question chunk counts."""
    require_auth(request)
    collection = get_active_collection()
    all_points = scroll_all_points(collection)

    pages: Dict[str, Dict] = {}
    for point in all_points:
        p = point.payload
        url = p.get("page_url", "")
        if not url:
            continue
        is_question = bool(p.get("embed_text", ""))
        if url not in pages:
            pages[url] = {
                "url": url,
                "title": p.get("page_title", ""),
                "text_count": 0,
                "question_count": 0,
            }
        if is_question:
            pages[url]["question_count"] += 1
        else:
            pages[url]["text_count"] += 1

    return sorted(pages.values(), key=lambda x: x["url"])


@app.get("/api/chunks")
def api_chunks(url: str, request: Request):
    require_auth(request)
    collection = get_active_collection()
    all_points = scroll_all_for_url(url, collection)

    text_chunks: Dict[int, Dict] = {}
    question_map: Dict[int, List[str]] = {}

    for point in all_points:
        p = point.payload
        idx = p.get("chunk_index", 0)
        embed_text = p.get("embed_text", "")

        if embed_text:
            if idx not in question_map:
                question_map[idx] = []
            question_map[idx].append(embed_text)
        else:
            if idx not in text_chunks:
                text_chunks[idx] = {
                    "id": str(point.id),
                    "chunk_index": idx,
                    "section": p.get("section_title", ""),
                    "text": p.get("text", ""),
                    "tags": p.get("tags", []),
                    "page_url": p.get("page_url", ""),
                    "page_title": p.get("page_title", ""),
                }

    result = []
    for idx in sorted(text_chunks.keys()):
        chunk = text_chunks[idx]
        chunk["questions"] = question_map.get(idx, [])
        result.append(chunk)

    return result


@app.post("/api/save")
def api_save(req: SaveRequest, request: Request):
    """Save a chunk: build new embeddings FIRST (rollback protection),
    then record history, delete old points, upsert new ones."""
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    collection = get_active_collection()
    client = get_client()

    # 1. Build new points BEFORE deleting old ones.
    #    If OpenAI fails here, old data is preserved unchanged.
    points = build_points(
        req.page_url, req.page_title, req.chunk_index,
        req.section, req.text, req.tags, req.questions,
    )
    if not points:
        raise HTTPException(status_code=500, detail="Не удалось создать эмбеддинги")

    # 2. Record history snapshot
    add_to_history(req.page_url, req.chunk_index, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": req.text,
        "section": req.section,
        "tags": req.tags,
        "questions": req.questions,
    })

    # 3. Delete old points -- only after embeddings succeeded
    delete_chunk_points(req.page_url, req.chunk_index, collection)

    # 4. Upsert new points
    client.upsert(collection_name=collection, points=points)
    logger.info(f"Saved chunk ({req.page_url}, idx={req.chunk_index}): "
                f"1 text + {len(points) - 1} questions")

    return {"ok": True, "new_id": str(points[0].id), "question_count": len(points) - 1}


@app.post("/api/delete")
def api_delete(req: DeleteRequest, request: Request):
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)
    collection = get_active_collection()
    delete_chunk_points(req.page_url, req.chunk_index, collection)
    return {"ok": True}


@app.post("/api/new")
def api_new(req: NewChunkRequest, request: Request):
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    collection = get_active_collection()
    client = get_client()

    existing = scroll_all_for_url(req.page_url, collection)
    max_index = max((p.payload.get("chunk_index", 0) for p in existing), default=-1)
    new_index = max_index + 1

    points = build_points(
        req.page_url, req.page_title, new_index,
        req.section, req.text, req.tags, req.questions,
    )
    if not points:
        raise HTTPException(status_code=500, detail="Не удалось создать эмбеддинги")

    client.upsert(collection_name=collection, points=points)
    logger.info(f"New chunk added ({req.page_url}, idx={new_index})")
    return {"ok": True, "id": str(points[0].id), "chunk_index": new_index}


@app.get("/api/history")
def api_history(url: str, index: int, request: Request):
    require_auth(request)
    history = load_history()
    key = f"{url}::{index}"
    return history.get(key, [])


@app.post("/api/restore")
def api_restore(req: RestoreRequest, request: Request):
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)

    history = load_history()
    key = f"{req.url}::{req.index}"
    versions = history.get(key, [])

    if req.version_index >= len(versions):
        raise HTTPException(status_code=404, detail="Версия не найдена")

    version = versions[req.version_index]
    collection = get_active_collection()
    client = get_client()

    existing = scroll_all_for_url(req.url, collection)
    page_title = next(
        (p.payload.get("page_title", "") for p in existing if p.payload.get("page_title")),
        ""
    )

    # Build before delete (rollback protection)
    points = build_points(
        req.url, page_title, req.index,
        version.get("section", ""),
        version["text"],
        version.get("tags", []),
        version.get("questions", []),
    )
    if not points:
        raise HTTPException(status_code=500, detail="Не удалось создать эмбеддинги при восстановлении")

    delete_chunk_points(req.url, req.index, collection)
    client.upsert(collection_name=collection, points=points)
    logger.info(f"Restored chunk ({req.url}, idx={req.index}) to version {req.version_index}")

    # Return restored chunk data so frontend can rebuild the card immediately
    return {
        "ok": True,
        "chunk": {
            "id": str(points[0].id),
            "chunk_index": req.index,
            "section": version.get("section", ""),
            "text": version["text"],
            "tags": version.get("tags", []),
            "questions": version.get("questions", []),
            "page_url": req.url,
            "page_title": page_title,
        }
    }


@app.post("/api/generate-questions")
def api_generate_questions(req: GenerateQuestionsRequest, request: Request):
    """Generate questions for a chunk via GPT (uses question_generator)."""
    require_auth(request)
    _check_rate(request.client.host, "generate", limit=20, window=60)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    try:
        from app.parser.chunker import TextChunk as ParserTextChunk
        from app.indexer.question_generator import generate_question_chunks

        chunk = ParserTextChunk(
            text=req.text.strip(),
            index=0,
            page_url=req.page_url or "",
            page_title=req.page_title or "",
            section_title=req.section or "",
        )
        question_chunks = generate_question_chunks([chunk], questions_per_chunk=4)
        questions = [qc.embed_text for qc in question_chunks if qc.embed_text]
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")


@app.post("/api/bulk-delete")
def api_bulk_delete(req: BulkDeleteRequest, request: Request):
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)
    collection = get_active_collection()
    for item in req.items:
        try:
            delete_chunk_points(item["page_url"], item["chunk_index"], collection)
        except Exception as e:
            logger.warning(f"Bulk delete failed for {item}: {e}")
    return {"ok": True, "deleted": len(req.items)}


@app.post("/api/bulk-tag")
def api_bulk_tag(req: BulkTagRequest, request: Request):
    require_auth(request)
    _check_rate(request.client.host, "save", limit=60, window=60)
    collection = get_active_collection()
    client = get_client()
    ok, fail = 0, 0

    for item in req.items:
        try:
            existing_tags = item.get("tags", [])
            new_tags = list(set(existing_tags) | set(req.tags)) if req.mode == "add" else req.tags
            points = build_points(
                item["page_url"], item.get("page_title", ""),
                item["chunk_index"], item.get("section", ""),
                item["text"], new_tags, item.get("questions", []),
            )
            if points:
                delete_chunk_points(item["page_url"], item["chunk_index"], collection)
                client.upsert(collection_name=collection, points=points)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            logger.warning(f"Bulk tag failed for chunk {item.get('chunk_index')}: {e}")
            fail += 1

    return {"ok": True, "updated": ok, "failed": fail}


@app.post("/api/index_page")
def api_index_page(req: IndexPageRequest, request: Request):
    """
    Fetch, chunk, and index a single page URL into Qdrant.
    Deletes existing chunks for this URL first, then saves fresh ones.
    """
    require_auth(request)
    _check_rate(request.client.host, "index_page", limit=10, window=60)

    url = req.url.strip()
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Укажите корректный URL (начинается с http)")

    try:
        from app.parser.extractor import get_page_content
        from app.parser.chunker import split_into_chunks
        from app.indexer.storage import delete_chunks_by_url, save_chunks
        from app.indexer.question_generator import generate_question_chunks
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка импорта: {e}")

    # 1. Fetch page
    try:
        content = get_page_content(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Не удалось скачать страницу: {e}")

    if not content or not content.text.strip():
        raise HTTPException(status_code=422, detail="Страница пустая или недоступна")

    # 2. Chunk
    chunks = split_into_chunks(
        text=content.text,
        page_url=content.url,
        page_title=content.title,
        external_links=getattr(content, "external_links", []),
    )
    if not chunks:
        raise HTTPException(status_code=422, detail="Не удалось разбить страницу на чанки (пустой текст?)")

    # 3. Questions
    question_chunks = []
    if req.with_questions:
        try:
            question_chunks = generate_question_chunks(chunks)
        except Exception as e:
            logger.warning(f"Question generation failed for {url}: {e}")

    all_chunks = chunks + question_chunks

    # 4. Save (save_chunks deletes old points for this URL internally)
    collection = get_active_collection()
    try:
        saved = save_chunks(all_chunks, content.url, content.content_hash,
                            collection_name=collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения в Qdrant: {e}")

    logger.info(
        f"[IndexPage] {url}: {len(chunks)} chunks + {len(question_chunks)} questions → {saved} saved"
    )
    return {
        "ok": True,
        "url": content.url,
        "title": content.title,
        "chunks": len(chunks),
        "questions": len(question_chunks),
        "saved": saved,
    }


# -- Frontend ------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chunk Editor \u2014 \u0426\u0410\u0418\u0423</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f0f2f5;
  color: #2c3e50;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
.header {
  background: #1a252f;
  color: white;
  padding: 0 20px;
  height: 50px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  z-index: 10;
}
.header h1 { font-size: 16px; font-weight: 600; letter-spacing: 0.3px; }
.header .sep { color: #455a64; }
.header .coll { font-size: 12px; color: #78909c; font-family: monospace; }
.header-right { margin-left: auto; display: flex; gap: 8px; align-items: center; }

/* Layout */
.layout { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar {
  width: 310px;
  background: white;
  border-right: 1px solid #dde0e4;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
.sidebar-top {
  padding: 12px;
  border-bottom: 1px solid #eee;
  background: #fafafa;
}
.search-box {
  width: 100%;
  padding: 7px 10px 7px 32px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
  background: white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23999' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E") 10px center no-repeat;
  outline: none;
  transition: border-color 0.15s;
}
.search-box:focus { border-color: #3498db; }
.sidebar-stats { font-size: 11px; color: #999; margin-top: 7px; display: flex; gap: 10px; }
.stat-badge { background: #f0f2f5; border-radius: 4px; padding: 2px 7px; }
.page-list { flex: 1; overflow-y: auto; }
.page-item {
  padding: 9px 12px;
  cursor: pointer;
  border-bottom: 1px solid #f5f5f5;
  transition: background 0.1s;
  position: relative;
}
.page-item:hover { background: #f8f9fb; }
.page-item.active { background: #ebf4ff; border-left: 3px solid #3498db; padding-left: 9px; }
.page-item-title {
  font-size: 12px; font-weight: 500; color: #2c3e50;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 36px;
}
.page-item-url {
  font-size: 11px; color: #aaa; margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 36px;
}
.page-badge {
  position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
  background: #e8f4ff; color: #3498db; font-size: 11px; font-weight: 700;
  border-radius: 10px; padding: 2px 7px; min-width: 24px; text-align: center;
}
mark.hl { background: #fff176; color: inherit; border-radius: 2px; padding: 0 1px; }

/* Main */
.main { flex: 1; overflow-y: auto; padding: 20px 24px 80px; }
.main-empty {
  height: 100%; display: flex; align-items: center; justify-content: center;
  color: #bbb; font-size: 15px; flex-direction: column; gap: 12px;
}
.main-empty-icon { font-size: 48px; }

/* Page heading */
.page-heading { margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #e0e3e8; }
.page-heading h2 { font-size: 15px; color: #1a252f; margin-bottom: 4px; }
.page-heading a { font-size: 11px; color: #3498db; text-decoration: none; word-break: break-all; }
.page-heading a:hover { text-decoration: underline; }
.page-heading-meta { display: flex; gap: 12px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
.meta-tag { font-size: 12px; color: #666; background: #f0f2f5; border-radius: 4px; padding: 3px 8px; }

/* Buttons */
.btn {
  padding: 6px 14px; border: none; border-radius: 5px; cursor: pointer;
  font-size: 13px; font-weight: 500; transition: all 0.15s;
  display: inline-flex; align-items: center; gap: 5px; white-space: nowrap;
}
.btn:disabled { opacity: 0.55; cursor: not-allowed; pointer-events: none; }
.btn-primary { background: #3498db; color: white; }
.btn-primary:hover { background: #2980b9; }
.btn-success { background: #27ae60; color: white; }
.btn-success:hover { background: #219a52; }
.btn-danger { background: #e74c3c; color: white; }
.btn-danger:hover { background: #c0392b; }
.btn-ghost { background: #ecf0f1; color: #555; }
.btn-ghost:hover { background: #dde; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-xs { padding: 2px 7px; font-size: 11px; }

/* Chunk card */
.chunk-card {
  background: white; border: 1px solid #dde0e4; border-radius: 8px;
  margin-bottom: 14px; transition: box-shadow 0.15s, border-color 0.2s;
}
.chunk-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.chunk-card.saving { opacity: 0.8; pointer-events: none; }
.chunk-card.dirty { border-color: #e67e22; border-left: 3px solid #e67e22; }
.chunk-card.selected { border-color: #3498db; background: #f8fbff; }

.card-head {
  padding: 9px 14px; background: #f8f9fb; border-bottom: 1px solid #eee;
  display: flex; align-items: center; gap: 8px; border-radius: 8px 8px 0 0;
}
.card-checkbox { display: none; width: 16px; height: 16px; cursor: pointer; flex-shrink: 0; }
.bulk-active .card-checkbox { display: block; }
.card-idx {
  font-size: 11px; color: #999; font-family: monospace;
  background: #eee; padding: 1px 6px; border-radius: 3px; flex-shrink: 0;
}
.section-input {
  font-size: 13px; font-weight: 500; color: #2c3e50;
  border: 1px solid transparent; padding: 3px 8px; border-radius: 4px;
  flex: 1; background: transparent; transition: all 0.15s;
}
.section-input::placeholder { color: #bbb; font-weight: 400; }
.section-input:hover { border-color: #ddd; background: white; }
.section-input:focus { border-color: #3498db; outline: none; background: white; }

.card-body { padding: 14px; }
.text-area {
  width: 100%; min-height: 110px; max-height: 400px;
  padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 13px; font-family: inherit; resize: vertical;
  line-height: 1.55; color: #2c3e50; transition: border-color 0.15s;
}
.text-area:focus { outline: none; border-color: #3498db; }

.section-divider {
  margin: 12px 0 8px; font-size: 11px; font-weight: 600;
  color: #999; text-transform: uppercase; letter-spacing: 0.5px;
}
.tags-wrap { display: flex; flex-wrap: wrap; gap: 5px; }
.tag-pill {
  padding: 3px 10px; border-radius: 12px; font-size: 12px; cursor: pointer;
  border: 1px solid #ddd; background: #fafafa; color: #777;
  transition: all 0.15s; user-select: none;
}
.tag-pill:hover { border-color: #3498db; color: #3498db; }
.tag-pill.on { background: #3498db; color: white; border-color: #3498db; }

.questions-header {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 7px;
}
.q-count { font-size: 11px; color: #999; margin-left: 5px; }
.question-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.question-in {
  flex: 1; padding: 6px 10px; border: 1px solid #ddd; border-radius: 5px;
  font-size: 13px; font-family: inherit; color: #2c3e50; transition: border-color 0.15s;
}
.question-in:focus { outline: none; border-color: #3498db; }
.question-in::placeholder { color: #bbb; }
.q-del {
  background: none; border: none; color: #ccc; cursor: pointer;
  font-size: 18px; line-height: 1; padding: 0 4px; transition: color 0.1s; flex-shrink: 0;
}
.q-del:hover { color: #e74c3c; }

/* Generated questions panel */
.gen-panel {
  background: #f0f8e8; border: 1px solid #b7d99b; border-radius: 6px;
  padding: 10px 12px; margin-top: 10px;
}
.gen-panel-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px; font-size: 12px; font-weight: 600; color: #3d6b1e;
}
.gen-q-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; padding: 4px 0; border-bottom: 1px solid #cde8b0;
}
.gen-q-row:last-child { border-bottom: none; }
.gen-q-text { font-size: 12px; color: #2c3e50; flex: 1; }

.card-foot {
  padding: 9px 14px; border-top: 1px solid #f0f0f0;
  display: flex; gap: 7px; align-items: center;
  border-radius: 0 0 8px 8px; flex-wrap: wrap;
}
.save-msg { font-size: 12px; margin-left: auto; transition: opacity 0.3s; }
.save-msg.ok { color: #27ae60; }
.save-msg.err { color: #e74c3c; }

.spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,0.4); border-top-color: white;
  border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle;
}
.spinner.dark { border-color: rgba(0,0,0,0.15); border-top-color: #3498db; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Bulk bar */
.bulk-bar {
  position: fixed; bottom: 0; left: 310px; right: 0;
  background: #1a252f; color: white; padding: 10px 24px;
  display: flex; align-items: center; gap: 12px; z-index: 50;
  box-shadow: 0 -3px 12px rgba(0,0,0,0.2); border-top: 2px solid #3498db; font-size: 13px;
}
.bulk-bar .bulk-count { font-weight: 600; color: #3498db; }

/* Modals */
.overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center; z-index: 1000;
}
.modal {
  background: white; border-radius: 10px; width: 640px; max-height: 82vh;
  display: flex; flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,0.25); overflow: hidden;
}
.modal-head {
  padding: 14px 18px; border-bottom: 1px solid #eee;
  display: flex; align-items: center; justify-content: space-between;
  background: #fafafa; flex-shrink: 0;
}
.modal-head h3 { font-size: 14px; color: #2c3e50; }
.modal-close {
  background: none; border: none; font-size: 20px; cursor: pointer;
  color: #aaa; line-height: 1; transition: color 0.1s;
}
.modal-close:hover { color: #333; }
.modal-body { flex: 1; overflow-y: auto; padding: 14px 18px; }

.hist-item {
  border: 1px solid #eee; border-radius: 6px; padding: 12px;
  margin-bottom: 10px; background: #fafafa;
}
.hist-ts { font-size: 11px; color: #999; margin-bottom: 7px; font-family: monospace; }
.hist-text {
  font-size: 12px; color: #555; white-space: pre-wrap; max-height: 80px; overflow: hidden;
  line-height: 1.4; background: white; border: 1px solid #eee; border-radius: 4px; padding: 6px 8px;
}
.hist-qs { font-size: 11px; color: #888; margin-top: 6px; font-style: italic; }
.hist-foot { margin-top: 8px; display: flex; gap: 8px; align-items: center; }

/* Diff */
.diff-view {
  font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
  background: white; border: 1px solid #eee; border-radius: 4px;
  padding: 8px 10px; max-height: 280px; overflow-y: auto;
}
ins.diff-add { background: #d4edda; color: #155724; text-decoration: none; border-radius: 2px; padding: 0 1px; }
del.diff-del { background: #f8d7da; color: #721c24; border-radius: 2px; padding: 0 1px; }

/* Login overlay */
.login-overlay {
  position: fixed; inset: 0; background: rgba(20,30,40,0.92);
  display: flex; align-items: center; justify-content: center; z-index: 9000;
}
.login-box {
  background: white; border-radius: 12px; padding: 36px 40px; width: 340px;
  box-shadow: 0 24px 64px rgba(0,0,0,0.4); text-align: center;
}
.login-box h2 { font-size: 18px; margin-bottom: 8px; color: #1a252f; }
.login-box p { font-size: 13px; color: #888; margin-bottom: 20px; }
.login-input {
  width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 14px; margin-bottom: 12px; outline: none; transition: border-color 0.15s;
}
.login-input:focus { border-color: #3498db; }
.login-error { color: #e74c3c; font-size: 12px; margin-top: 8px; min-height: 18px; }

/* Toast */
.toast {
  position: fixed; bottom: 24px; right: 24px; background: #2c3e50; color: white;
  padding: 11px 18px; border-radius: 7px; font-size: 13px; z-index: 2000;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2); animation: toastIn 0.25s ease; max-width: 360px;
}
.toast.success { background: #27ae60; }
.toast.error { background: #e74c3c; }
@keyframes toastIn { from { transform: translateY(16px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
</style>
</head>
<body>

<!-- Login overlay -->
<div class="login-overlay" id="login-overlay" style="display:none">
  <div class="login-box">
    <h2>\U0001f512 Chunk Editor</h2>
    <p>\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u043f\u0430\u0440\u043e\u043b\u044c \u0434\u043b\u044f \u0434\u043e\u0441\u0442\u0443\u043f\u0430</p>
    <input class="login-input" id="login-pwd" type="password"
           placeholder="\u041f\u0430\u0440\u043e\u043b\u044c..."
           onkeydown="if(event.key==='Enter') doLogin()">
    <button class="btn btn-primary" style="width:100%" onclick="doLogin()">\u0412\u043e\u0439\u0442\u0438</button>
    <div class="login-error" id="login-err"></div>
  </div>
</div>

<!-- Header -->
<div class="header">
  <h1>\U0001f527 Chunk Editor</h1>
  <span class="sep">|</span>
  <span class="coll" id="coll-name">\u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0430...</span>
  <div class="header-right">
    <button class="btn btn-ghost btn-sm" onclick="reloadAll()">\u21bb \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c</button>
    <button class="btn btn-ghost btn-sm" onclick="showIndexModal()" title="\u0418\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043e\u0434\u043d\u0443 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443 \u043f\u043e URL">\u2795 URL</button>
    <button class="btn btn-ghost btn-sm" id="logout-btn" style="display:none" onclick="doLogout()">\u0412\u044b\u0439\u0442\u0438</button>
  </div>
</div>

<!-- Index Page Modal -->
<div class="modal-overlay" id="index-modal" style="display:none" onclick="if(event.target===this)closeIndexModal()">
  <div class="modal-box" style="width:500px">
    <div class="modal-head">
      <span>\u2795 \u0418\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443 \u043f\u043e URL</span>
      <button class="modal-close" onclick="closeIndexModal()">\u2715</button>
    </div>
    <div class="modal-body" style="padding:20px">
      <div style="margin-bottom:12px">
        <label style="font-size:13px;font-weight:500;display:block;margin-bottom:6px">URL \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b</label>
        <input type="url" id="index-url-input"
          placeholder="https://caiu.edu.kz/..."
          style="width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:6px;font-size:13px;font-family:inherit;outline:none"
          onkeydown="if(event.key==='Enter')doIndexPage()"
          onfocus="this.style.borderColor='#3498db'" onblur="this.style.borderColor='#ddd'">
      </div>
      <div style="margin-bottom:16px;display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="index-with-questions" checked style="width:15px;height:15px;cursor:pointer">
        <label for="index-with-questions" style="font-size:13px;cursor:pointer">\u0413\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0432\u043e\u043f\u0440\u043e\u0441\u044b (GPT, ~$0.01)</label>
      </div>
      <div id="index-result" style="display:none;padding:12px;border-radius:6px;font-size:13px;margin-bottom:14px"></div>
      <div style="display:flex;gap:8px">
        <button class="btn btn-primary btn-sm" id="index-run-btn" onclick="doIndexPage()">\u2795 \u0418\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c</button>
        <button class="btn btn-ghost btn-sm" onclick="closeIndexModal()">\u041e\u0442\u043c\u0435\u043d\u0430</button>
      </div>
    </div>
  </div>
</div>

<!-- Layout -->
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-top">
      <input class="search-box" id="search" type="text"
             placeholder="\u041f\u043e\u0438\u0441\u043a \u043f\u043e URL / \u043d\u0430\u0437\u0432\u0430\u043d\u0438\u044e..."
             oninput="filterPages()">
      <div class="sidebar-stats">
        <span class="stat-badge" id="stat-pages">...</span>
        <span class="stat-badge" id="stat-chunks">...</span>
        <span class="stat-badge" id="stat-qs">...</span>
      </div>
    </div>
    <div class="page-list" id="page-list">
      <div style="padding:24px;text-align:center;color:#bbb">
        <div class="spinner dark" style="width:20px;height:20px;margin:auto"></div>
      </div>
    </div>
  </div>

  <div class="main" id="main" oninput="onMainInput(event)" onchange="onMainChange(event)">
    <div class="main-empty">
      <div class="main-empty-icon">\U0001f4cb</div>
      <span>\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443 \u0438\u0437 \u0441\u043f\u0438\u0441\u043a\u0430 \u0441\u043b\u0435\u0432\u0430</span>
    </div>
  </div>
</div>

<!-- Bulk action bar -->
<div class="bulk-bar" id="bulk-bar" style="display:none">
  <span class="bulk-count" id="bulk-count-label">0 \u0432\u044b\u0431\u0440\u0430\u043d\u043e</span>
  <button class="btn btn-danger btn-sm" onclick="bulkDelete()">\U0001f5d1 \u0423\u0434\u0430\u043b\u0438\u0442\u044c</button>
  <button class="btn btn-primary btn-sm" onclick="showBulkTagModal()">\U0001f3f7 \u0422\u0435\u0433\u0438</button>
  <button class="btn btn-ghost btn-sm" onclick="clearBulkSelection()">\u0421\u043d\u044f\u0442\u044c \u0432\u044b\u0431\u043e\u0440</button>
  <button class="btn btn-ghost btn-sm" style="margin-left:auto" onclick="toggleBulkMode()">\u2715 \u041e\u0442\u043c\u0435\u043d\u0430</button>
</div>

<!-- History modal -->
<div class="overlay" id="hist-modal" style="display:none" onclick="closeHistModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-head">
      <h3 id="hist-title">\u0418\u0441\u0442\u043e\u0440\u0438\u044f \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0439</h3>
      <button class="modal-close" onclick="closeHistModal()">\u2715</button>
    </div>
    <div class="modal-body" id="hist-body">
      <div style="text-align:center;padding:30px">
        <div class="spinner dark" style="width:20px;height:20px;margin:auto"></div>
      </div>
    </div>
  </div>
</div>

<!-- Bulk tag modal -->
<div class="overlay" id="bulk-tag-modal" style="display:none" onclick="closeBulkTagModal(event)">
  <div class="modal" style="width:480px;max-height:60vh" onclick="event.stopPropagation()">
    <div class="modal-head">
      <h3>\u0422\u0435\u0433\u0438 \u0434\u043b\u044f <span id="bulk-tag-count">?</span> \u0447\u0430\u043d\u043a\u043e\u0432</h3>
      <button class="modal-close" onclick="closeBulkTagModal()">\u2715</button>
    </div>
    <div class="modal-body">
      <div style="font-size:12px;color:#888;margin-bottom:12px">
        \u0420\u0435\u0436\u0438\u043c:
        <label style="margin-left:8px"><input type="radio" name="bulk-mode" value="set" checked> \u0417\u0430\u043c\u0435\u043d\u0438\u0442\u044c</label>
        <label style="margin-left:8px"><input type="radio" name="bulk-mode" value="add"> \u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c</label>
      </div>
      <div class="tags-wrap" id="bulk-tag-pills"></div>
      <div style="margin-top:16px;display:flex;gap:8px">
        <button class="btn btn-primary btn-sm" onclick="applyBulkTags()">\u041f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c</button>
        <button class="btn btn-ghost btn-sm" onclick="closeBulkTagModal()">\u041e\u0442\u043c\u0435\u043d\u0430</button>
      </div>
    </div>
  </div>
</div>

<script>
'use strict';

// State
let allPages = [];
let _sidebarPages = [];
let _sidebarOffset = 0;
let _sidebarObserver = null;
const SIDEBAR_BATCH = 50;

let curUrl = null;
let curChunks = [];
let histUrl = null;
let histIdx = null;
let histCi = null;
let bulkMode = false;
let selectedCards = new Set();
let dirtyCards = new Set();

const TAGS = [
  "admission","dormitory","fees","grants","specialties",
  "faculty","department","contacts","military","exams",
  "history","management","licenses","general"
];

// Utilities
function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function highlightText(text, query) {
  if (!query || !text) return esc(text || '');
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const parts = [];
  let last = 0, idx;
  while ((idx = t.indexOf(q, last)) !== -1) {
    parts.push(esc(text.slice(last, idx)));
    parts.push('<mark class="hl">' + esc(text.slice(idx, idx + query.length)) + '</mark>');
    last = idx + query.length;
  }
  parts.push(esc(text.slice(last)));
  return parts.join('');
}

function toast(msg, type) {
  type = type || '';
  var el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(function() { el.style.opacity = '0'; }, 2600);
  setTimeout(function() { el.remove(); }, 3000);
}

async function api(method, path, body) {
  var opts = { method: method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  var r = await fetch(path, opts);
  if (r.status === 401) { showLoginOverlay(); throw new Error('\\u0422\\u0440\\u0435\\u0431\\u0443\\u0435\\u0442\\u0441\\u044f \\u0430\\u0432\\u0442\\u043e\\u0440\\u0438\\u0437\\u0430\\u0446\\u0438\\u044f'); }
  if (!r.ok) {
    var err = await r.json().catch(function() { return {detail: r.statusText}; });
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// Diff
function _lcs(a, b) {
  var m = a.length, n = b.length;
  if (m * n > 120000) {
    return a.map(function(v){ return {type:'del',val:v}; }).concat(b.map(function(v){ return {type:'add',val:v}; }));
  }
  var dp = [];
  for (var i = 0; i <= m; i++) { dp[i] = new Int32Array(n + 1); }
  for (var i = 1; i <= m; i++)
    for (var j = 1; j <= n; j++)
      dp[i][j] = a[i-1] === b[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
  var res = [], i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i-1] === b[j-1]) { res.unshift({type:'same',val:a[i-1]}); i--; j--; }
    else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) { res.unshift({type:'add',val:b[j-1]}); j--; }
    else { res.unshift({type:'del',val:a[i-1]}); i--; }
  }
  return res;
}

function renderDiff(oldText, newText) {
  if (!oldText && !newText) return '<em style="color:#bbb">\\u041f\\u0443\\u0441\\u0442\\u043e</em>';
  if (oldText === newText) return '<span>' + esc(oldText) + '</span>';
  var oldTok = oldText.split(/(\\s+)/);
  var newTok = newText.split(/(\\s+)/);
  var ops = _lcs(oldTok, newTok);
  return ops.map(function(op) {
    if (op.type === 'same') return esc(op.val);
    if (op.type === 'add')  return '<ins class="diff-add">' + esc(op.val) + '</ins>';
    if (op.type === 'del')  return '<del class="diff-del">' + esc(op.val) + '</del>';
    return '';
  }).join('');
}

// Auth
function showLoginOverlay() {
  document.getElementById('login-overlay').style.display = 'flex';
  document.getElementById('logout-btn').style.display = 'none';
  setTimeout(function() { document.getElementById('login-pwd').focus(); }, 50);
}

async function checkAuth() {
  try {
    var s = await fetch('/auth/status').then(function(r) { return r.json(); });
    if (s.required && !s.authenticated) { showLoginOverlay(); return false; }
    if (s.required) document.getElementById('logout-btn').style.display = 'inline-flex';
    return true;
  } catch(e) { return true; }
}

async function doLogin() {
  var pwd = document.getElementById('login-pwd').value;
  var errEl = document.getElementById('login-err');
  errEl.textContent = '';
  try {
    await api('POST', '/auth/login', {password: pwd});
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('login-pwd').value = '';
    document.getElementById('logout-btn').style.display = 'inline-flex';
    await reloadAll();
  } catch(e) {
    errEl.textContent = '\\u041d\\u0435\\u0432\\u0435\\u0440\\u043d\\u044b\\u0439 \\u043f\\u0430\\u0440\\u043e\\u043b\\u044c';
    document.getElementById('login-pwd').select();
  }
}

async function doLogout() {
  await fetch('/auth/logout', {method:'POST'}).catch(function(){});
  showLoginOverlay();
}

// Init
async function init() {
  var ok = await checkAuth();
  if (!ok) return;
  try {
    var c = await api('GET', '/api/collection');
    document.getElementById('coll-name').textContent = c.collection;
  } catch(e) {}
  await reloadAll();
}

async function reloadAll() {
  document.getElementById('page-list').innerHTML =
    '<div style="padding:24px;text-align:center;color:#bbb"><div class="spinner dark" style="width:20px;height:20px;margin:auto"></div></div>';
  try {
    allPages = await api('GET', '/api/pages');
    var totalText = allPages.reduce(function(s,p){ return s+p.text_count; }, 0);
    var totalQ    = allPages.reduce(function(s,p){ return s+p.question_count; }, 0);
    document.getElementById('stat-pages').textContent  = allPages.length + ' \\u0441\\u0442\\u0440.';
    document.getElementById('stat-chunks').textContent = totalText + ' \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432';
    document.getElementById('stat-qs').textContent     = totalQ + ' \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441\\u043e\\u0432';
    filterPages();
  } catch(e) {
    document.getElementById('page-list').innerHTML =
      '<div style="padding:20px;color:#e74c3c">\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + esc(e.message) + '</div>';
  }
}

// Sidebar
function filterPages() {
  var q = document.getElementById('search').value.toLowerCase().trim();
  _sidebarPages = q
    ? allPages.filter(function(p){ return p.url.toLowerCase().indexOf(q)>=0 || p.title.toLowerCase().indexOf(q)>=0; })
    : allPages;
  renderPageList(_sidebarPages, q);
}

function renderPageList(pages, query) {
  query = query || '';
  var el = document.getElementById('page-list');
  _sidebarOffset = 0;
  if (_sidebarObserver) { _sidebarObserver.disconnect(); _sidebarObserver = null; }
  if (!pages.length) {
    el.innerHTML = '<div style="padding:20px;text-align:center;color:#bbb">\\u041d\\u0438\\u0447\\u0435\\u0433\\u043e \\u043d\\u0435 \\u043d\\u0430\\u0439\\u0434\\u0435\\u043d\\u043e</div>';
    return;
  }
  var slice = pages.slice(0, SIDEBAR_BATCH);
  _sidebarOffset = slice.length;
  el.innerHTML = slice.map(function(p){ return renderPageItem(p, query); }).join('') +
    (pages.length > SIDEBAR_BATCH ? '<div id="sidebar-sentinel" style="height:4px;margin:2px 0"></div>' : '');
  if (pages.length > SIDEBAR_BATCH) _setupSidebarObserver(query);
}

function _setupSidebarObserver(query) {
  var sentinel = document.getElementById('sidebar-sentinel');
  if (!sentinel) return;
  _sidebarObserver = new IntersectionObserver(function(entries) {
    if (!entries[0].isIntersecting) return;
    var next = _sidebarPages.slice(_sidebarOffset, _sidebarOffset + SIDEBAR_BATCH);
    if (!next.length) { _sidebarObserver.disconnect(); return; }
    sentinel.insertAdjacentHTML('beforebegin', next.map(function(p){ return renderPageItem(p, query); }).join(''));
    _sidebarOffset += next.length;
    if (_sidebarOffset >= _sidebarPages.length) _sidebarObserver.disconnect();
  }, {root: document.getElementById('page-list'), threshold: 0.1});
  _sidebarObserver.observe(sentinel);
}

function renderPageItem(p, query) {
  var q = query || document.getElementById('search').value.toLowerCase().trim();
  var titleHtml = q ? highlightText(p.title || p.url, q) : esc(p.title || p.url);
  var urlHtml   = q ? highlightText(p.url, q) : esc(p.url);
  var active    = p.url === curUrl ? 'active' : '';
  return '<div class="page-item ' + active + '" data-url="' + esc(p.url) + '" onclick="loadPage(this.dataset.url)">' +
    '<div class="page-badge">' + p.text_count + '</div>' +
    '<div class="page-item-title">' + titleHtml + '</div>' +
    '<div class="page-item-url">' + urlHtml + '</div></div>';
}

// Load page
async function loadPage(url) {
  if (dirtyCards.size > 0 && curUrl && curUrl !== url) {
    if (!confirm(dirtyCards.size + ' \\u043d\\u0435\\u0441\\u043e\\u0445\\u0440\\u0430\\u043d\\u0451\\u043d\\u043d\\u044b\\u0445 \\u0438\\u0437\\u043c\\u0435\\u043d\\u0435\\u043d\\u0438\\u0439. \\u0412\\u0441\\u0451 \\u0440\\u0430\\u0432\\u043d\\u043e \\u043f\\u0435\\u0440\\u0435\\u0439\\u0442\\u0438?')) return;
  }
  curUrl = url;
  dirtyCards.clear();
  selectedCards.clear();
  exitBulkMode();
  filterPages();
  var main = document.getElementById('main');
  main.innerHTML = '<div class="main-empty"><div class="spinner dark" style="width:24px;height:24px"></div><span>\\u0417\\u0430\\u0433\\u0440\\u0443\\u0437\\u043a\\u0430 \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432...</span></div>';
  try {
    curChunks = await api('GET', '/api/chunks?url=' + encodeURIComponent(url));
    var page = allPages.find(function(p){ return p.url === url; }) || {};
    renderPageContent(url, page.title || url, curChunks);
  } catch(e) {
    main.innerHTML = '<div class="main-empty" style="color:#e74c3c">\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + esc(e.message) + '</div>';
  }
}

function renderPageContent(url, title, chunks) {
  var main = document.getElementById('main');
  var qTotal = chunks.reduce(function(s,c){ return s+(c.questions||[]).length; }, 0);
  var isActive = bulkMode ? 'bulk-active' : '';
  var heading = '<div class="page-heading">' +
    '<h2>' + esc(title) + '</h2>' +
    '<a href="' + esc(url) + '" target="_blank">' + esc(url) + '</a>' +
    '<div class="page-heading-meta">' +
    '<span class="meta-tag">' + chunks.length + ' \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432</span>' +
    '<span class="meta-tag">' + qTotal + ' \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441\\u043e\\u0432</span>' +
    '<button class="btn btn-success btn-sm" style="margin-left:auto" onclick="addNewChunk()">+ \\u0414\\u043e\\u0431\\u0430\\u0432\\u0438\\u0442\\u044c \\u0447\\u0430\\u043d\\u043a</button>' +
    '<button class="btn btn-ghost btn-sm" id="bulk-toggle-btn" onclick="toggleBulkMode()">' + (bulkMode ? '\\u2715 \\u041e\\u0442\\u043c\\u0435\\u043d\\u0430 \\u0432\\u044b\\u0431\\u043e\\u0440\\u0430' : '\\u2611 \\u0412\\u044b\\u0431\\u0440\\u0430\\u0442\\u044c') + '</button>' +
    '</div></div>';
  if (!chunks.length) {
    main.innerHTML = '<div id="cards-container" class="' + isActive + '">' + heading + '<div style="color:#bbb;padding:10px 0">\\u041d\\u0435\\u0442 \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432</div></div>';
    return;
  }
  var cards = chunks.map(function(c, i){ return buildCard(c, i); }).join('');
  main.innerHTML = '<div id="cards-container" class="' + isActive + '">' + heading + cards + '</div>';
}

// Dirty state
function markDirty(ci) {
  dirtyCards.add(ci);
  var card = document.getElementById('card-' + ci);
  if (card) card.classList.add('dirty');
}
function clearDirty(ci) {
  dirtyCards.delete(ci);
  var card = document.getElementById('card-' + ci);
  if (card) card.classList.remove('dirty');
}
function onMainInput(e) {
  var card = e.target.closest('.chunk-card');
  if (card) markDirty(parseInt(card.dataset.ci));
}
function onMainChange(e) {
  var card = e.target.closest('.chunk-card');
  if (card) markDirty(parseInt(card.dataset.ci));
}
window.addEventListener('beforeunload', function(e) {
  if (dirtyCards.size > 0) { e.preventDefault(); e.returnValue = ''; }
});

// Card builder
function buildCard(chunk, ci) {
  var tagsHtml = TAGS.map(function(t) {
    var on = (chunk.tags||[]).indexOf(t) >= 0 ? 'on' : '';
    return '<span class="tag-pill ' + on + '" onclick="this.classList.toggle(\\'on\\'); markDirty(' + ci + ')">' + esc(t) + '</span>';
  }).join('');
  var qHtml = (chunk.questions||[]).map(function(q, qi){ return qRow(ci, qi, q); }).join('');
  var isSelected = selectedCards.has(ci) ? 'selected' : '';
  var isDirty    = dirtyCards.has(ci) ? 'dirty' : '';
  return '<div class="chunk-card ' + isSelected + ' ' + isDirty + '" id="card-' + ci + '"' +
    ' data-ci="' + ci + '" data-id="' + esc(chunk.id) + '" data-cidx="' + chunk.chunk_index + '"' +
    ' data-url="' + esc(chunk.page_url) + '" data-title="' + esc(chunk.page_title) + '">' +
    '<div class="card-head">' +
    '<input type="checkbox" class="card-checkbox" ' + (selectedCards.has(ci)?'checked':'') + ' onchange="toggleSelectCard(' + ci + ', this.checked)">' +
    '<span class="card-idx">#' + chunk.chunk_index + '</span>' +
    '<input class="section-input" id="sec-' + ci + '" type="text" value="' + esc(chunk.section||'') + '" placeholder="\\u0420\\u0430\\u0437\\u0434\\u0435\\u043b (h2/h3)...">' +
    '</div>' +
    '<div class="card-body">' +
    '<textarea class="text-area" id="txt-' + ci + '">' + esc(chunk.text) + '</textarea>' +
    '<div class="section-divider">\\u0422\\u0435\\u0433\\u0438</div>' +
    '<div class="tags-wrap" id="tags-' + ci + '">' + tagsHtml + '</div>' +
    '<div class="section-divider" style="margin-top:14px">' +
    '<div class="questions-header">' +
    '<span>\\u0412\\u043e\\u043f\\u0440\\u043e\\u0441\\u044b <span class="q-count" id="qcnt-' + ci + '">(' + (chunk.questions||[]).length + ')</span></span>' +
    '<button class="btn btn-ghost btn-xs" onclick="addQ(' + ci + ')">+ \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441</button>' +
    '</div></div>' +
    '<div id="qs-' + ci + '">' + qHtml + '</div>' +
    '</div>' +
    '<div class="card-foot">' +
    '<button class="btn btn-primary btn-sm" id="savebtn-' + ci + '" onclick="saveChunk(' + ci + ')">\U0001f4be \\u0421\\u043e\\u0445\\u0440\\u0430\\u043d\\u0438\\u0442\\u044c</button>' +
    '<button class="btn btn-ghost btn-sm" onclick="showHist(' + ci + ')">\U0001f4dc \\u0418\\u0441\\u0442\\u043e\\u0440\\u0438\\u044f</button>' +
    '<button class="btn btn-ghost btn-sm" id="genbtn-' + ci + '" onclick="generateQuestions(' + ci + ')">\U0001f916 \\u0412\\u043e\\u043f\\u0440\\u043e\\u0441\\u044b</button>' +
    '<button class="btn btn-danger btn-sm" onclick="delChunk(' + ci + ')">\U0001f5d1</button>' +
    '<span class="save-msg" id="msg-' + ci + '"></span>' +
    '</div></div>';
}

function qRow(ci, qi, val) {
  return '<div class="question-row" id="qrow-' + ci + '-' + qi + '">' +
    '<input class="question-in" type="text" value="' + esc(val) + '" placeholder="\\u0412\\u043e\\u043f\\u0440\\u043e\\u0441, \\u043d\\u0430 \\u043a\\u043e\\u0442\\u043e\\u0440\\u044b\\u0439 \\u043e\\u0442\\u0432\\u0435\\u0447\\u0430\\u0435\\u0442 \\u044d\\u0442\\u043e\\u0442 \\u0447\\u0430\\u043d\\u043a...">' +
    '<button class="q-del" onclick="delQ(this)">\u00d7</button></div>';
}

function rebuildCard(ci, chunkData) {
  var main = document.getElementById('main');
  var scrollY = main.scrollTop;
  var oldCard = document.getElementById('card-' + ci);
  if (!oldCard) return;
  oldCard.outerHTML = buildCard(chunkData, ci);
  main.scrollTop = scrollY;
}

// Tags
function getActiveTags(ci) {
  return Array.from(document.getElementById('tags-' + ci).querySelectorAll('.tag-pill.on'))
    .map(function(el){ return el.textContent.trim(); });
}

// Questions
function getQs(ci) {
  return Array.from(document.getElementById('qs-' + ci).querySelectorAll('.question-in'))
    .map(function(el){ return el.value.trim(); }).filter(Boolean);
}
function addQ(ci) {
  var container = document.getElementById('qs-' + ci);
  var qi = container.querySelectorAll('.question-row').length;
  container.insertAdjacentHTML('beforeend', qRow(ci, qi, ''));
  container.lastElementChild.querySelector('input').focus();
  updateQCount(ci); markDirty(ci);
}
function delQ(btn) {
  var row = btn.closest('.question-row');
  var ci = parseInt(row.id.split('-')[1]);
  row.remove(); updateQCount(ci); markDirty(ci);
}
function updateQCount(ci) {
  var cnt = document.getElementById('qs-' + ci).querySelectorAll('.question-row').length;
  var el = document.getElementById('qcnt-' + ci);
  if (el) el.textContent = '(' + cnt + ')';
}

// Save
async function saveChunk(ci) {
  var card = document.getElementById('card-' + ci);
  var btn  = document.getElementById('savebtn-' + ci);
  var payload = {
    chunk_id:    card.dataset.id,
    page_url:    card.dataset.url,
    page_title:  card.dataset.title,
    chunk_index: parseInt(card.dataset.cidx),
    section:     document.getElementById('sec-' + ci).value.trim(),
    text:        document.getElementById('txt-' + ci).value,
    tags:        getActiveTags(ci),
    questions:   getQs(ci),
  };
  if (!payload.text.trim()) { toast('\\u0422\\u0435\\u043a\\u0441\\u0442 \\u043d\\u0435 \\u043c\\u043e\\u0436\\u0435\\u0442 \\u0431\\u044b\\u0442\\u044c \\u043f\\u0443\\u0441\\u0442\\u044b\\u043c', 'error'); return; }
  card.classList.add('saving');
  btn.innerHTML = '<span class="spinner"></span> \\u0421\\u043e\\u0445\\u0440\\u0430\\u043d\\u0435\\u043d\\u0438\\u0435...';
  try {
    var res = await api('POST', '/api/save', payload);
    curChunks[ci] = Object.assign({}, curChunks[ci], {
      id: res.new_id, text: payload.text, section: payload.section,
      tags: payload.tags, questions: payload.questions,
    });
    rebuildCard(ci, curChunks[ci]);
    clearDirty(ci);
    toast('\\u0421\\u043e\\u0445\\u0440\\u0430\\u043d\\u0435\\u043d\\u043e (+' + res.question_count + ' \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441\\u043e\\u0432)', 'success');
    reloadAll();
  } catch(e) {
    card.classList.remove('saving');
    btn.innerHTML = '\U0001f4be \\u0421\\u043e\\u0445\\u0440\\u0430\\u043d\\u0438\\u0442\\u044c';
    var m = document.getElementById('msg-' + ci);
    if (m) { m.textContent = '\\u2717 ' + e.message; m.className = 'save-msg err'; }
    toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + e.message, 'error');
  }
}

// Delete
async function delChunk(ci) {
  if (!confirm('\\u0423\\u0434\\u0430\\u043b\\u0438\\u0442\\u044c \\u044d\\u0442\\u043e\\u0442 \\u0447\\u0430\\u043d\\u043a? \\u0418\\u0441\\u0442\\u043e\\u0440\\u0438\\u044f \\u0441\\u043e\\u0445\\u0440\\u0430\\u043d\\u0438\\u0442\\u0441\\u044f.')) return;
  var card = document.getElementById('card-' + ci);
  try {
    await api('POST', '/api/delete', {page_url: card.dataset.url, chunk_index: parseInt(card.dataset.cidx)});
    card.remove(); curChunks[ci] = null; clearDirty(ci); selectedCards.delete(ci); updateBulkBar();
    toast('\\u0427\\u0430\\u043d\\u043a \\u0443\\u0434\\u0430\\u043b\\u0451\\u043d', 'success');
    reloadAll();
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430 \\u0443\\u0434\\u0430\\u043b\\u0435\\u043d\\u0438\\u044f: ' + e.message, 'error'); }
}

// New chunk
async function addNewChunk() {
  var text = prompt('\\u0422\\u0435\\u043a\\u0441\\u0442 \\u043d\\u043e\\u0432\\u043e\\u0433\\u043e \\u0447\\u0430\\u043d\\u043a\\u0430:');
  if (!text || !text.trim()) return;
  var section = prompt('\\u0420\\u0430\\u0437\\u0434\\u0435\\u043b (h2/h3, \\u043c\\u043e\\u0436\\u043d\\u043e \\u043e\\u0441\\u0442\\u0430\\u0432\\u0438\\u0442\\u044c \\u043f\\u0443\\u0441\\u0442\\u044b\\u043c):') || '';
  var page = allPages.find(function(p){ return p.url === curUrl; }) || {};
  try {
    var res = await api('POST', '/api/new', {page_url: curUrl, page_title: page.title||'', section: section.trim(), text: text.trim(), tags: [], questions: []});
    toast('\\u0427\\u0430\\u043d\\u043a \\u0434\\u043e\\u0431\\u0430\\u0432\\u043b\\u0435\\u043d (#' + res.chunk_index + ')', 'success');
    await loadPage(curUrl);
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + e.message, 'error'); }
}

// Bulk ops
function toggleBulkMode() {
  bulkMode = !bulkMode;
  if (!bulkMode) { selectedCards.clear(); exitBulkMode(); } else { enterBulkMode(); }
}
function enterBulkMode() {
  var c = document.getElementById('cards-container');
  if (c) c.classList.add('bulk-active');
  var btn = document.getElementById('bulk-toggle-btn');
  if (btn) btn.textContent = '\\u2715 \\u041e\\u0442\\u043c\\u0435\\u043d\\u0430 \\u0432\\u044b\\u0431\\u043e\\u0440\\u0430';
  updateBulkBar();
}
function exitBulkMode() {
  bulkMode = false; selectedCards.clear();
  var c = document.getElementById('cards-container');
  if (c) c.classList.remove('bulk-active');
  var btn = document.getElementById('bulk-toggle-btn');
  if (btn) btn.textContent = '\\u2611 \\u0412\\u044b\\u0431\\u0440\\u0430\\u0442\\u044c';
  document.getElementById('bulk-bar').style.display = 'none';
}
function toggleSelectCard(ci, checked) {
  if (checked) { selectedCards.add(ci); document.getElementById('card-' + ci)?.classList.add('selected'); }
  else { selectedCards.delete(ci); document.getElementById('card-' + ci)?.classList.remove('selected'); }
  updateBulkBar();
}
function updateBulkBar() {
  var bar = document.getElementById('bulk-bar');
  var label = document.getElementById('bulk-count-label');
  bar.style.display = (bulkMode && selectedCards.size > 0) ? 'flex' : (bulkMode ? 'flex' : 'none');
  if (label) label.textContent = selectedCards.size + ' \\u0432\\u044b\\u0431\\u0440\\u0430\\u043d\\u043e';
}
function clearBulkSelection() {
  selectedCards.forEach(function(ci) {
    var card = document.getElementById('card-' + ci);
    if (card) { card.classList.remove('selected'); var cb = card.querySelector('.card-checkbox'); if (cb) cb.checked = false; }
  });
  selectedCards.clear(); updateBulkBar();
}
async function bulkDelete() {
  var n = selectedCards.size;
  if (!n) return;
  if (!confirm('\\u0423\\u0434\\u0430\\u043b\\u0438\\u0442\\u044c ' + n + ' \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432? \\u042d\\u0442\\u043e \\u043d\\u0435\\u043b\\u044c\\u0437\\u044f \\u043e\\u0442\\u043c\\u0435\\u043d\\u0438\\u0442\\u044c.')) return;
  var items = [];
  selectedCards.forEach(function(ci) {
    var card = document.getElementById('card-' + ci);
    if (card) items.push({page_url: card.dataset.url, chunk_index: parseInt(card.dataset.cidx)});
  });
  try {
    var res = await api('POST', '/api/bulk-delete', {items: items});
    selectedCards.forEach(function(ci) { document.getElementById('card-' + ci)?.remove(); clearDirty(ci); });
    selectedCards.clear(); exitBulkMode();
    toast('\\u0423\\u0434\\u0430\\u043b\\u0435\\u043d\\u043e ' + res.deleted + ' \\u0447\\u0430\\u043d\\u043a\\u043e\\u0432', 'success');
    reloadAll();
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + e.message, 'error'); }
}
function showBulkTagModal() {
  if (!selectedCards.size) return;
  document.getElementById('bulk-tag-count').textContent = selectedCards.size;
  var pills = TAGS.map(function(t){ return '<span class="tag-pill" onclick="this.classList.toggle(\\'on\\')">' + esc(t) + '</span>'; }).join('');
  document.getElementById('bulk-tag-pills').innerHTML = pills;
  document.getElementById('bulk-tag-modal').style.display = 'flex';
}
function closeBulkTagModal(event) {
  if (event && event.target !== document.getElementById('bulk-tag-modal')) return;
  document.getElementById('bulk-tag-modal').style.display = 'none';
}
async function applyBulkTags() {
  var selectedTags = Array.from(document.querySelectorAll('#bulk-tag-pills .tag-pill.on')).map(function(el){ return el.textContent.trim(); });
  var mode = (document.querySelector('input[name="bulk-mode"]:checked') || {}).value || 'set';
  var items = [];
  selectedCards.forEach(function(ci) {
    var card = document.getElementById('card-' + ci);
    if (!card) return;
    items.push({page_url: card.dataset.url, page_title: card.dataset.title, chunk_index: parseInt(card.dataset.cidx),
      section: (document.getElementById('sec-' + ci)||{}).value||'',
      text: (document.getElementById('txt-' + ci)||{}).value||'',
      questions: getQs(ci), tags: getActiveTags(ci)});
  });
  document.getElementById('bulk-tag-modal').style.display = 'none';
  try {
    var res = await api('POST', '/api/bulk-tag', {items: items, tags: selectedTags, mode: mode});
    toast('\\u0422\\u0435\\u0433\\u0438 \\u043f\\u0440\\u0438\\u043c\\u0435\\u043d\\u0435\\u043d\\u044b \\u043a ' + res.updated + ' \\u0447\\u0430\\u043d\\u043a\\u0430\\u043c', 'success');
    await loadPage(curUrl); exitBulkMode(); reloadAll();
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + e.message, 'error'); }
}

// History
async function showHist(ci) {
  var card = document.getElementById('card-' + ci);
  histUrl = card.dataset.url; histIdx = parseInt(card.dataset.cidx); histCi = ci;
  document.getElementById('hist-title').textContent = '\\u0418\\u0441\\u0442\\u043e\\u0440\\u0438\\u044f \\u0447\\u0430\\u043d\\u043a\\u0430 #' + histIdx;
  document.getElementById('hist-body').innerHTML = '<div style="text-align:center;padding:30px"><div class="spinner dark" style="width:20px;height:20px;margin:auto"></div></div>';
  document.getElementById('hist-modal').style.display = 'flex';
  try {
    var history = await api('GET', '/api/history?url=' + encodeURIComponent(histUrl) + '&index=' + histIdx);
    renderHist(history);
  } catch(e) {
    document.getElementById('hist-body').innerHTML = '<div style="color:#e74c3c;padding:20px">\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430: ' + esc(e.message) + '</div>';
  }
}

function renderHist(history) {
  var body = document.getElementById('hist-body');
  if (!history.length) { body.innerHTML = '<div style="color:#bbb;text-align:center;padding:24px">\\u0418\\u0441\\u0442\\u043e\\u0440\\u0438\\u044f \\u043f\\u0443\\u0441\\u0442\\u0430</div>'; return; }
  body.innerHTML = history.map(function(v, i) {
    var preview = v.text.length > 300 ? v.text.slice(0,300)+'\\u2026' : v.text;
    var qsHtml = (v.questions && v.questions.length)
      ? '<div class="hist-qs">\\u0412\\u043e\\u043f\\u0440\\u043e\\u0441\\u044b: ' + v.questions.slice(0,3).map(esc).join(' \u00b7 ') + (v.questions.length>3?' +'+v.questions.length:'') + '</div>'
      : '';
    var tagsHtml = (v.tags && v.tags.length)
      ? '<span style="font-size:11px;color:#aaa">' + v.tags.map(esc).join(', ') + '</span>'
      : '';
    return '<div class="hist-item">' +
      '<div class="hist-ts">' + esc(v.timestamp) + '</div>' +
      '<div class="hist-text" id="hist-raw-' + i + '" data-full="' + esc(v.text) + '">' + esc(preview) + '</div>' +
      qsHtml +
      '<div class="hist-foot">' +
      '<button class="btn btn-ghost btn-sm" onclick="restoreHist(' + i + ')">\\u21a9 \\u0412\\u043e\\u0441\\u0441\\u0442\\u0430\\u043d\\u043e\\u0432\\u0438\\u0442\\u044c</button>' +
      '<button class="btn btn-ghost btn-sm" onclick="showDiff(' + i + ')">\\u21fa \\u0421\\u0440\\u0430\\u0432\\u043d\\u0438\\u0442\\u044c</button>' +
      tagsHtml +
      '</div>' +
      '<div id="diff-' + i + '" style="display:none;margin-top:8px">' +
      '<div style="font-size:11px;color:#888;margin-bottom:4px">\\u0421\\u0440\\u0430\\u0432\\u043d\\u0435\\u043d\\u0438\\u0435 \\u0441 \\u0442\\u0435\\u043a\\u0443\\u0449\\u0438\\u043c <button class="btn btn-ghost btn-xs" onclick="document.getElementById(\\'diff-' + i + '\\').style.display=\\'none\\'" style="margin-left:4px">\\u2715</button></div>' +
      '<div class="diff-view" id="diff-content-' + i + '"></div></div>' +
      '</div>';
  }).join('');
}

function showDiff(vi) {
  var currentText = (histCi !== null && document.getElementById('txt-' + histCi))
    ? document.getElementById('txt-' + histCi).value
    : (curChunks[histCi] || {}).text || '';
  var rawEl = document.getElementById('hist-raw-' + vi);
  var histText = rawEl ? (rawEl.dataset.full || rawEl.textContent) : '';
  var diffEl = document.getElementById('diff-' + vi);
  var diffContent = document.getElementById('diff-content-' + vi);
  if (!diffEl || !diffContent) return;
  if (diffEl.style.display !== 'none') { diffEl.style.display = 'none'; return; }
  diffContent.innerHTML = renderDiff(histText, currentText);
  diffEl.style.display = 'block';
}

async function restoreHist(vi) {
  if (!confirm('\\u0412\\u043e\\u0441\\u0441\\u0442\\u0430\\u043d\\u043e\\u0432\\u0438\\u0442\\u044c \\u044d\\u0442\\u0443 \\u0432\\u0435\\u0440\\u0441\\u0438\\u044e? \\u0422\\u0435\\u043a\\u0443\\u0449\\u0438\\u0439 \\u0447\\u0430\\u043d\\u043a \\u0431\\u0443\\u0434\\u0435\\u0442 \\u0437\\u0430\\u043c\\u0435\\u043d\\u0451\\u043d.')) return;
  try {
    var res = await api('POST', '/api/restore', {url: histUrl, index: histIdx, version_index: vi});
    closeHistModal();
    toast('\\u0412\\u0435\\u0440\\u0441\\u0438\\u044f \\u0432\\u043e\\u0441\\u0441\\u0442\\u0430\\u043d\\u043e\\u0432\\u043b\\u0435\\u043d\\u0430', 'success');
    if (histCi !== null && res.chunk) {
      curChunks[histCi] = res.chunk;
      rebuildCard(histCi, res.chunk);
      clearDirty(histCi);
    }
    reloadAll();
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430 \\u0432\\u043e\\u0441\\u0441\\u0442\\u0430\\u043d\\u043e\\u0432\\u043b\\u0435\\u043d\\u0438\\u044f: ' + e.message, 'error'); }
}

function closeHistModal(event) {
  if (event && event.target !== document.getElementById('hist-modal')) return;
  document.getElementById('hist-modal').style.display = 'none';
}

// Generate questions
async function generateQuestions(ci) {
  var btn = document.getElementById('genbtn-' + ci);
  if (!btn) return;
  var text = (document.getElementById('txt-' + ci) || {}).value;
  if (!text || !text.trim()) { toast('\\u0421\\u043d\\u0430\\u0447\\u0430\\u043b\\u0430 \\u0432\\u0432\\u0435\\u0434\\u0438\\u0442\\u0435 \\u0442\\u0435\\u043a\\u0441\\u0442 \\u0447\\u0430\\u043d\\u043a\\u0430', 'error'); return; }
  var card = document.getElementById('card-' + ci);
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner dark"></span>';
  try {
    var res = await api('POST', '/api/generate-questions', {
      text: text.trim(),
      page_url:   card.dataset.url,
      page_title: card.dataset.title,
      section:    (document.getElementById('sec-' + ci)||{}).value||'',
    });
    if (!res.questions || !res.questions.length) { toast('GPT \\u043d\\u0435 \\u0432\\u0435\\u0440\\u043d\\u0443\\u043b \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441\\u043e\\u0432', 'error'); return; }
    showGenSuggestions(ci, res.questions);
    toast('\\u0421\\u0433\\u0435\\u043d\\u0435\\u0440\\u0438\\u0440\\u043e\\u0432\\u0430\\u043d\\u043e ' + res.questions.length + ' \\u0432\\u043e\\u043f\\u0440\\u043e\\u0441\\u043e\\u0432', 'success');
  } catch(e) { toast('\\u041e\\u0448\\u0438\\u0431\\u043a\\u0430 \\u0433\\u0435\\u043d\\u0435\\u0440\\u0430\\u0446\\u0438\\u0438: ' + e.message, 'error'); }
  finally { btn.disabled = false; btn.innerHTML = '\U0001f916 \\u0412\\u043e\\u043f\\u0440\\u043e\\u0441\\u044b'; }
}

function showGenSuggestions(ci, questions) {
  var old = document.getElementById('gen-panel-' + ci);
  if (old) old.remove();
  var qs = document.getElementById('qs-' + ci);
  var panel = document.createElement('div');
  panel.id = 'gen-panel-' + ci;
  panel.className = 'gen-panel';
  panel.innerHTML = '<div class="gen-panel-head"><span>\\u041f\\u0440\\u0435\\u0434\\u043b\\u043e\\u0436\\u0435\\u043d\\u0438\\u044f GPT (' + questions.length + ')</span>' +
    '<div style="display:flex;gap:5px">' +
    '<button class="btn btn-ghost btn-xs" onclick="addAllGenQ(' + ci + ')">\\u0414\\u043e\\u0431\\u0430\\u0432\\u0438\\u0442\\u044c \\u0432\\u0441\\u0435</button>' +
    '<button class="btn btn-ghost btn-xs" onclick="replaceAllGenQ(' + ci + ')">\\u0417\\u0430\\u043c\\u0435\\u043d\\u0438\\u0442\\u044c</button>' +
    '<button class="btn btn-ghost btn-xs" onclick="closeGenPanel(' + ci + ')">\\u2715</button></div></div>' +
    questions.map(function(q) {
      return '<div class="gen-q-row" data-q="' + esc(q) + '">' +
        '<span class="gen-q-text">' + esc(q) + '</span>' +
        '<button class="btn btn-ghost btn-xs" onclick="addOneGenQ(' + ci + ', this)">\u2795</button></div>';
    }).join('');
  qs.after(panel);
}

function addAllGenQ(ci) {
  document.querySelectorAll('#gen-panel-' + ci + ' .gen-q-row').forEach(function(row){ _addQWithValue(ci, row.dataset.q); });
  closeGenPanel(ci); markDirty(ci);
}
function replaceAllGenQ(ci) {
  document.getElementById('qs-' + ci).innerHTML = '';
  addAllGenQ(ci);
}
function addOneGenQ(ci, btn) {
  var row = btn.closest('.gen-q-row');
  _addQWithValue(ci, row.dataset.q); row.remove(); markDirty(ci);
  var panel = document.getElementById('gen-panel-' + ci);
  if (panel && !panel.querySelectorAll('.gen-q-row').length) closeGenPanel(ci);
}
function _addQWithValue(ci, value) {
  var container = document.getElementById('qs-' + ci);
  var qi = container.querySelectorAll('.question-row').length;
  container.insertAdjacentHTML('beforeend', qRow(ci, qi, value));
  updateQCount(ci);
}
function closeGenPanel(ci) {
  var p = document.getElementById('gen-panel-' + ci);
  if (p) p.remove();
}

// ── Index Page modal ──────────────────────────────────────────────────────────

function showIndexModal() {
  var m = document.getElementById('index-modal');
  m.style.display = 'flex';
  setTimeout(function() { document.getElementById('index-url-input').focus(); }, 80);
  var res = document.getElementById('index-result');
  res.style.display = 'none';
  res.textContent = '';
  document.getElementById('index-run-btn').disabled = false;
}

function closeIndexModal() {
  document.getElementById('index-modal').style.display = 'none';
  document.getElementById('index-url-input').value = '';
}

function doIndexPage() {
  var url = document.getElementById('index-url-input').value.trim();
  if (!url || !url.startsWith('http')) {
    alert('Укажите корректный URL (например https://caiu.edu.kz/page/)');
    return;
  }
  var withQ = document.getElementById('index-with-questions').checked;
  var btn = document.getElementById('index-run-btn');
  var res = document.getElementById('index-result');

  btn.disabled = true;
  btn.textContent = '⏳ Индексирование...';
  res.style.display = 'block';
  res.style.background = '#f0f7ff';
  res.style.color = '#2c3e50';
  res.textContent = 'Скачиваю и обрабатываю страницу' + (withQ ? ', генерирую вопросы...' : '...');

  fetch('/api/index_page', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: url, with_questions: withQ})
  })
  .then(function(r) { return r.json().then(function(d) { return {ok: r.ok, status: r.status, data: d}; }); })
  .then(function(r) {
    btn.disabled = false;
    btn.textContent = '➕ Индексировать';
    if (r.ok) {
      res.style.background = '#e8f8f0';
      res.style.color = '#1a6b3c';
      res.innerHTML =
        '✅ <b>Готово!</b><br>' +
        '📄 ' + esc(r.data.title || r.data.url) + '<br>' +
        '🧩 Чанков: ' + r.data.chunks + ' текст + ' + r.data.questions + ' вопросов = ' + r.data.saved + ' векторов';
      // Reload pages list so the new page appears in sidebar
      setTimeout(loadPages, 800);
    } else {
      res.style.background = '#fff0f0';
      res.style.color = '#c0392b';
      res.innerHTML = '❌ <b>Ошибка:</b> ' + esc(r.data.detail || JSON.stringify(r.data));
    }
  })
  .catch(function(e) {
    btn.disabled = false;
    btn.textContent = '➕ Индексировать';
    res.style.background = '#fff0f0';
    res.style.color = '#c0392b';
    res.innerHTML = '❌ <b>Сеть:</b> ' + esc(String(e));
  });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') { closeHistModal(); closeBulkTagModal(); closeIndexModal(); }
});

init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=HTML_PAGE)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk Editor -- browser-based Qdrant editor")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Chunk Editor starting on http://localhost:{args.port}")
    logger.info(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    if CHUNK_EDITOR_PASSWORD:
        logger.info("Auth: ENABLED (password set via CHUNK_EDITOR_PASSWORD)")
    else:
        logger.info("Auth: disabled (set CHUNK_EDITOR_PASSWORD to enable)")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
