# -*- coding: utf-8 -*-
"""
tools/chunk_editor.py — Browser-based Qdrant chunk editor.

Standalone FastAPI server (port 8080) for viewing and editing
chunks stored in Qdrant directly from a web browser.

Features:
  - Left sidebar: all indexed pages with chunk counts + URL/title search
  - Right panel: chunk cards with inline editing
  - Per-chunk: edit text, section title, tags, questions
  - Per-chunk save (re-embeds via OpenAI) and delete
  - Per-chunk history with restore (last 20 versions)
  - Add new chunk to any page

Usage:
  cd rag_service
  python ../tools/chunk_editor.py

Then open: http://localhost:8080

Requirements: fastapi, uvicorn (already in requirements.txt)
"""

import sys
import json
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# ── Path setup: import from rag_service ──────────────────────────────────────
_RAG_SERVICE = Path(__file__).resolve().parent.parent / "rag_service"
sys.path.insert(0, str(_RAG_SERVICE))

from fastapi import FastAPI, HTTPException
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

# History file: stored next to chunks_export.json
HISTORY_FILE = Path(__file__).resolve().parent.parent / "chunk_editor_history.json"
MAX_HISTORY_PER_CHUNK = 20

AVAILABLE_TAGS = [
    "admission", "dormitory", "fees", "grants", "specialties",
    "faculty", "department", "contacts", "military", "exams",
    "history", "management", "licenses", "general",
]

app = FastAPI(title="Chunk Editor", docs_url=None, redoc_url=None)


# ── History helpers ───────────────────────────────────────────────────────────

def load_history() -> Dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_history_file(history: Dict) -> None:
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_to_history(page_url: str, chunk_index: int, snapshot: dict) -> None:
    history = load_history()
    key = f"{page_url}::{chunk_index}"
    if key not in history:
        history[key] = []
    history[key].insert(0, snapshot)
    history[key] = history[key][:MAX_HISTORY_PER_CHUNK]
    save_history_file(history)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

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
    """
    Build Qdrant PointStructs for one text chunk + its question chunks.

    Text to embed:
      - Text chunk: embed the chunk text itself (embed_text="")
      - Question chunk: embed the question string (embed_text=question)
    """
    if not text.strip():
        return []

    clean_questions = [q.strip() for q in questions if q.strip()]
    texts_to_embed = [text] + clean_questions
    vectors = get_embeddings_batch(texts_to_embed)

    if len(vectors) != len(texts_to_embed):
        logger.error(f"Embedding mismatch: {len(texts_to_embed)} texts → {len(vectors)} vectors")
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

    points = []

    # Text chunk
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=vectors[0],
        payload={**base_payload, "text": text, "embed_text": ""},
    ))

    # Question chunks
    for question, vector in zip(clean_questions, vectors[1:]):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={**base_payload, "text": text, "embed_text": question},
        ))

    return points


# ── API request models ────────────────────────────────────────────────────────

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


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/collection")
def api_collection():
    collection = get_active_collection()
    return {"collection": collection}


@app.get("/api/pages")
def api_pages():
    """Return all pages with text/question chunk counts."""
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

    result = sorted(pages.values(), key=lambda x: x["url"])
    return result


@app.get("/api/chunks")
def api_chunks(url: str):
    """
    Return text chunks for a page URL, with questions attached.

    Handles both old (no embed_text in payload) and new data:
    - New: embed_text="" → text chunk; embed_text=question → question chunk
    - Old: no embed_text field → treat as text chunk, deduplicate by chunk_index
    """
    collection = get_active_collection()
    all_points = scroll_all_for_url(url, collection)

    text_chunks: Dict[int, Dict] = {}   # chunk_index → chunk data
    question_map: Dict[int, List[str]] = {}  # chunk_index → [question, ...]

    for point in all_points:
        p = point.payload
        idx = p.get("chunk_index", 0)
        embed_text = p.get("embed_text", "")

        if embed_text:
            # This is a question chunk
            if idx not in question_map:
                question_map[idx] = []
            question_map[idx].append(embed_text)
        else:
            # Text chunk (or old data without embed_text — deduplicate by chunk_index)
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
def api_save(req: SaveRequest):
    """Save a chunk: record history, delete old points, create new ones with fresh embeddings."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    collection = get_active_collection()
    client = get_client()

    # 1. Save to history
    add_to_history(req.page_url, req.chunk_index, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": req.text,
        "section": req.section,
        "tags": req.tags,
        "questions": req.questions,
    })

    # 2. Delete old points for this (url, chunk_index)
    delete_chunk_points(req.page_url, req.chunk_index, collection)

    # 3. Create new points (re-embed)
    points = build_points(
        req.page_url, req.page_title, req.chunk_index,
        req.section, req.text, req.tags, req.questions,
    )

    if not points:
        raise HTTPException(status_code=500, detail="Failed to create embeddings")

    client.upsert(collection_name=collection, points=points)
    logger.info(f"Saved chunk ({req.page_url}, idx={req.chunk_index}): "
                f"1 text + {len(points) - 1} questions")

    new_id = str(points[0].id)
    return {"ok": True, "new_id": new_id, "question_count": len(points) - 1}


@app.post("/api/delete")
def api_delete(req: DeleteRequest):
    """Delete a chunk and all its question vectors."""
    collection = get_active_collection()
    delete_chunk_points(req.page_url, req.chunk_index, collection)
    return {"ok": True}


@app.post("/api/new")
def api_new(req: NewChunkRequest):
    """Add a new chunk to a page (appended at the end with next chunk_index)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    collection = get_active_collection()
    client = get_client()

    # Find next available chunk_index for this page
    existing = scroll_all_for_url(req.page_url, collection)
    max_index = max((p.payload.get("chunk_index", 0) for p in existing), default=-1)
    new_index = max_index + 1

    points = build_points(
        req.page_url, req.page_title, new_index,
        req.section, req.text, req.tags, req.questions,
    )

    if not points:
        raise HTTPException(status_code=500, detail="Failed to create embeddings")

    client.upsert(collection_name=collection, points=points)
    logger.info(f"New chunk added ({req.page_url}, idx={new_index})")

    return {"ok": True, "id": str(points[0].id), "chunk_index": new_index}


@app.get("/api/history")
def api_history(url: str, index: int):
    """Return history snapshots for a chunk."""
    history = load_history()
    key = f"{url}::{index}"
    return history.get(key, [])


@app.post("/api/restore")
def api_restore(req: RestoreRequest):
    """Restore a chunk to a historical version."""
    history = load_history()
    key = f"{req.url}::{req.index}"
    versions = history.get(key, [])

    if req.version_index >= len(versions):
        raise HTTPException(status_code=404, detail="Version not found")

    version = versions[req.version_index]
    collection = get_active_collection()
    client = get_client()

    # Get page_title from any existing point for this URL
    existing = scroll_all_for_url(req.url, collection)
    page_title = ""
    for p in existing:
        t = p.payload.get("page_title", "")
        if t:
            page_title = t
            break

    # Delete current and recreate from historical snapshot
    delete_chunk_points(req.url, req.index, collection)
    points = build_points(
        req.url, page_title, req.index,
        version.get("section", ""),
        version["text"],
        version.get("tags", []),
        version.get("questions", []),
    )

    if points:
        client.upsert(collection_name=collection, points=points)

    logger.info(f"Restored chunk ({req.url}, idx={req.index}) to version {req.version_index}")
    return {"ok": True}


# ── Frontend ──────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chunk Editor — ЦАИУ</title>
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

/* ── Header ── */
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

/* ── Layout ── */
.layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ── */
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
.sidebar-stats {
  font-size: 11px;
  color: #999;
  margin-top: 7px;
  display: flex;
  gap: 10px;
}
.stat-badge {
  background: #f0f2f5;
  border-radius: 4px;
  padding: 2px 7px;
}
.page-list {
  flex: 1;
  overflow-y: auto;
}
.page-item {
  padding: 9px 12px;
  cursor: pointer;
  border-bottom: 1px solid #f5f5f5;
  transition: background 0.1s;
  position: relative;
}
.page-item:hover { background: #f8f9fb; }
.page-item.active {
  background: #ebf4ff;
  border-left: 3px solid #3498db;
  padding-left: 9px;
}
.page-item-title {
  font-size: 12px;
  font-weight: 500;
  color: #2c3e50;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 36px;
}
.page-item-url {
  font-size: 11px;
  color: #aaa;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 36px;
}
.page-badge {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  background: #e8f4ff;
  color: #3498db;
  font-size: 11px;
  font-weight: 700;
  border-radius: 10px;
  padding: 2px 7px;
  min-width: 24px;
  text-align: center;
}

/* ── Main ── */
.main {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
}
.main-empty {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #bbb;
  font-size: 15px;
  flex-direction: column;
  gap: 12px;
}
.main-empty-icon { font-size: 48px; }

/* ── Page header in main ── */
.page-heading {
  margin-bottom: 18px;
  padding-bottom: 14px;
  border-bottom: 1px solid #e0e3e8;
}
.page-heading h2 {
  font-size: 15px;
  color: #1a252f;
  margin-bottom: 4px;
}
.page-heading a {
  font-size: 11px;
  color: #3498db;
  text-decoration: none;
  word-break: break-all;
}
.page-heading a:hover { text-decoration: underline; }
.page-heading-meta {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-top: 10px;
}
.meta-tag {
  font-size: 12px;
  color: #666;
  background: #f0f2f5;
  border-radius: 4px;
  padding: 3px 8px;
}

/* ── Buttons ── */
.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.15s;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
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

/* ── Chunk card ── */
.chunk-card {
  background: white;
  border: 1px solid #dde0e4;
  border-radius: 8px;
  margin-bottom: 14px;
  transition: box-shadow 0.15s;
}
.chunk-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.chunk-card.saving { opacity: 0.8; pointer-events: none; }

.card-head {
  padding: 9px 14px;
  background: #f8f9fb;
  border-bottom: 1px solid #eee;
  display: flex;
  align-items: center;
  gap: 8px;
  border-radius: 8px 8px 0 0;
}
.card-idx {
  font-size: 11px;
  color: #999;
  font-family: monospace;
  background: #eee;
  padding: 1px 6px;
  border-radius: 3px;
  flex-shrink: 0;
}
.section-input {
  font-size: 13px;
  font-weight: 500;
  color: #2c3e50;
  border: 1px solid transparent;
  padding: 3px 8px;
  border-radius: 4px;
  flex: 1;
  background: transparent;
  transition: all 0.15s;
}
.section-input::placeholder { color: #bbb; font-weight: 400; }
.section-input:hover { border-color: #ddd; background: white; }
.section-input:focus { border-color: #3498db; outline: none; background: white; }

.card-body { padding: 14px; }

.text-area {
  width: 100%;
  min-height: 110px;
  max-height: 400px;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
  resize: vertical;
  line-height: 1.55;
  color: #2c3e50;
  transition: border-color 0.15s;
}
.text-area:focus { outline: none; border-color: #3498db; }

/* Tags */
.section-divider {
  margin: 12px 0 8px;
  font-size: 11px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.tags-wrap { display: flex; flex-wrap: wrap; gap: 5px; }
.tag-pill {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 12px;
  cursor: pointer;
  border: 1px solid #ddd;
  background: #fafafa;
  color: #777;
  transition: all 0.15s;
  user-select: none;
}
.tag-pill:hover { border-color: #3498db; color: #3498db; }
.tag-pill.on { background: #3498db; color: white; border-color: #3498db; }

/* Questions */
.questions-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 7px;
}
.q-count {
  font-size: 11px;
  color: #999;
  margin-left: 5px;
}
.question-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.question-in {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 13px;
  font-family: inherit;
  color: #2c3e50;
  transition: border-color 0.15s;
}
.question-in:focus { outline: none; border-color: #3498db; }
.question-in::placeholder { color: #bbb; }
.q-del {
  background: none;
  border: none;
  color: #ccc;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
  padding: 0 4px;
  transition: color 0.1s;
  flex-shrink: 0;
}
.q-del:hover { color: #e74c3c; }

/* Card footer */
.card-foot {
  padding: 9px 14px;
  border-top: 1px solid #f0f0f0;
  display: flex;
  gap: 7px;
  align-items: center;
  border-radius: 0 0 8px 8px;
}
.save-msg {
  font-size: 12px;
  margin-left: auto;
  transition: opacity 0.3s;
}
.save-msg.ok { color: #27ae60; }
.save-msg.err { color: #e74c3c; }

/* Loading */
.spinner {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
  vertical-align: middle;
}
.spinner.dark {
  border-color: rgba(0,0,0,0.15);
  border-top-color: #3498db;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── History Modal ── */
.overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center;
  z-index: 1000;
}
.modal {
  background: white;
  border-radius: 10px;
  width: 580px;
  max-height: 78vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  overflow: hidden;
}
.modal-head {
  padding: 14px 18px;
  border-bottom: 1px solid #eee;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fafafa;
}
.modal-head h3 { font-size: 14px; color: #2c3e50; }
.modal-close {
  background: none; border: none;
  font-size: 20px; cursor: pointer;
  color: #aaa; line-height: 1;
  transition: color 0.1s;
}
.modal-close:hover { color: #333; }
.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px 18px;
}
.hist-item {
  border: 1px solid #eee;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 10px;
  background: #fafafa;
}
.hist-ts {
  font-size: 11px;
  color: #999;
  margin-bottom: 7px;
  font-family: monospace;
}
.hist-text {
  font-size: 12px;
  color: #555;
  white-space: pre-wrap;
  max-height: 80px;
  overflow: hidden;
  line-height: 1.4;
  background: white;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 6px 8px;
}
.hist-qs {
  font-size: 11px;
  color: #888;
  margin-top: 6px;
  font-style: italic;
}
.hist-foot { margin-top: 8px; display: flex; gap: 8px; align-items: center; }

/* ── Toast ── */
.toast {
  position: fixed;
  bottom: 24px; right: 24px;
  background: #2c3e50;
  color: white;
  padding: 11px 18px;
  border-radius: 7px;
  font-size: 13px;
  z-index: 2000;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  animation: toastIn 0.25s ease;
  max-width: 360px;
}
.toast.success { background: #27ae60; }
.toast.error { background: #e74c3c; }
@keyframes toastIn {
  from { transform: translateY(16px); opacity: 0; }
  to   { transform: translateY(0);    opacity: 1; }
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>🔧 Chunk Editor</h1>
  <span class="sep">|</span>
  <span class="coll" id="coll-name">загрузка...</span>
  <div class="header-right">
    <button class="btn btn-ghost btn-sm" onclick="reloadAll()">↻ Обновить</button>
  </div>
</div>

<!-- Layout -->
<div class="layout">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-top">
      <input class="search-box" id="search" type="text"
             placeholder="Поиск по URL / названию..."
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

  <!-- Main content -->
  <div class="main" id="main">
    <div class="main-empty">
      <div class="main-empty-icon">📋</div>
      <span>Выберите страницу из списка слева</span>
    </div>
  </div>

</div>

<!-- History Modal -->
<div class="overlay" id="hist-modal" style="display:none" onclick="closeHistModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-head">
      <h3 id="hist-title">История изменений</h3>
      <button class="modal-close" onclick="closeHistModal()">✕</button>
    </div>
    <div class="modal-body" id="hist-body">
      <div style="text-align:center;padding:30px">
        <div class="spinner dark" style="width:20px;height:20px;margin:auto"></div>
      </div>
    </div>
  </div>
</div>

<script>
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let allPages = [];
let curUrl = null;
let curChunks = [];
let histUrl = null;
let histIdx = null;

const TAGS = [
  "admission","dormitory","fees","grants","specialties",
  "faculty","department","contacts","military","exams",
  "history","management","licenses","general"
];

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function toast(msg, type='') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.style.opacity = '0', 2600);
  setTimeout(() => el.remove(), 3000);
}

async function api(method, path, body=null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({detail: r.statusText}));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const c = await api('GET', '/api/collection');
    document.getElementById('coll-name').textContent = c.collection;
  } catch(e) {}
  await reloadAll();
}

async function reloadAll() {
  document.getElementById('page-list').innerHTML =
    '<div style="padding:24px;text-align:center;color:#bbb"><div class="spinner dark" style="width:20px;height:20px;margin:auto"></div></div>';

  try {
    allPages = await api('GET', '/api/pages');
    filterPages();
    const totalText = allPages.reduce((s,p) => s+p.text_count, 0);
    const totalQ = allPages.reduce((s,p) => s+p.question_count, 0);
    document.getElementById('stat-pages').textContent = `${allPages.length} стр.`;
    document.getElementById('stat-chunks').textContent = `${totalText} чанков`;
    document.getElementById('stat-qs').textContent = `${totalQ} вопросов`;
  } catch(e) {
    document.getElementById('page-list').innerHTML =
      `<div style="padding:20px;color:#e74c3c">Ошибка: ${esc(e.message)}</div>`;
  }
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function filterPages() {
  const q = document.getElementById('search').value.toLowerCase().trim();
  const filtered = q
    ? allPages.filter(p => p.url.toLowerCase().includes(q) || p.title.toLowerCase().includes(q))
    : allPages;
  renderPageList(filtered);
}

function renderPageList(pages) {
  const el = document.getElementById('page-list');
  if (!pages.length) {
    el.innerHTML = '<div style="padding:20px;text-align:center;color:#bbb">Ничего не найдено</div>';
    return;
  }
  el.innerHTML = pages.map(p => `
    <div class="page-item ${p.url === curUrl ? 'active' : ''}"
         onclick="loadPage(${JSON.stringify(p.url)})">
      <div class="page-badge">${p.text_count}</div>
      <div class="page-item-title">${esc(p.title || p.url)}</div>
      <div class="page-item-url">${esc(p.url)}</div>
    </div>
  `).join('');
}

// ── Load page ─────────────────────────────────────────────────────────────────
async function loadPage(url) {
  curUrl = url;
  filterPages();  // update active state

  const main = document.getElementById('main');
  main.innerHTML = `<div class="main-empty">
    <div class="spinner dark" style="width:24px;height:24px"></div>
    <span>Загрузка чанков...</span>
  </div>`;

  try {
    curChunks = await api('GET', '/api/chunks?url=' + encodeURIComponent(url));
    const page = allPages.find(p => p.url === url) || {};
    renderPageContent(url, page.title || url, curChunks);
  } catch(e) {
    main.innerHTML = `<div class="main-empty" style="color:#e74c3c">Ошибка: ${esc(e.message)}</div>`;
  }
}

function renderPageContent(url, title, chunks) {
  const main = document.getElementById('main');
  const qTotal = chunks.reduce((s, c) => s + (c.questions||[]).length, 0);

  const heading = `
    <div class="page-heading">
      <h2>${esc(title)}</h2>
      <a href="${esc(url)}" target="_blank">${esc(url)}</a>
      <div class="page-heading-meta">
        <span class="meta-tag">${chunks.length} чанков</span>
        <span class="meta-tag">${qTotal} вопросов</span>
        <button class="btn btn-success btn-sm" style="margin-left:auto" onclick="addNewChunk()">+ Добавить чанк</button>
      </div>
    </div>
  `;

  if (!chunks.length) {
    main.innerHTML = heading + '<div style="color:#bbb;padding:10px 0">Нет чанков</div>';
    return;
  }

  main.innerHTML = heading + chunks.map((c, i) => buildCard(c, i)).join('');
}

// ── Card builder ──────────────────────────────────────────────────────────────
function buildCard(chunk, ci) {
  const tagsHtml = TAGS.map(t => {
    const on = (chunk.tags||[]).includes(t) ? 'on' : '';
    return `<span class="tag-pill ${on}" onclick="this.classList.toggle('on')">${esc(t)}</span>`;
  }).join('');

  const qHtml = (chunk.questions||[]).map((q, qi) => qRow(ci, qi, q)).join('');

  return `
<div class="chunk-card" id="card-${ci}"
     data-id="${esc(chunk.id)}"
     data-cidx="${chunk.chunk_index}"
     data-url="${esc(chunk.page_url)}"
     data-title="${esc(chunk.page_title)}">

  <div class="card-head">
    <span class="card-idx">#${chunk.chunk_index}</span>
    <input class="section-input" id="sec-${ci}" type="text"
           value="${esc(chunk.section || '')}"
           placeholder="Раздел (h2/h3)...">
  </div>

  <div class="card-body">
    <textarea class="text-area" id="txt-${ci}">${esc(chunk.text)}</textarea>

    <div class="section-divider">Теги</div>
    <div class="tags-wrap" id="tags-${ci}">${tagsHtml}</div>

    <div class="section-divider" style="margin-top:14px">
      <div class="questions-header">
        <span>Вопросы <span class="q-count" id="qcnt-${ci}">(${(chunk.questions||[]).length})</span></span>
        <button class="btn btn-ghost btn-xs" onclick="addQ(${ci})">+ вопрос</button>
      </div>
    </div>
    <div id="qs-${ci}">${qHtml}</div>
  </div>

  <div class="card-foot">
    <button class="btn btn-primary btn-sm" id="savebtn-${ci}" onclick="saveChunk(${ci})">💾 Сохранить</button>
    <button class="btn btn-ghost btn-sm" onclick="showHist(${ci})">📜 История</button>
    <button class="btn btn-danger btn-sm" onclick="delChunk(${ci})">🗑</button>
    <span class="save-msg" id="msg-${ci}"></span>
  </div>
</div>`;
}

function qRow(ci, qi, val) {
  return `<div class="question-row" id="qrow-${ci}-${qi}">
    <input class="question-in" type="text" value="${esc(val)}"
           placeholder="Введите вопрос на который отвечает этот чанк...">
    <button class="q-del" onclick="delQ(this)">×</button>
  </div>`;
}

// ── Tags helpers ──────────────────────────────────────────────────────────────
function getActiveTags(ci) {
  return Array.from(document.getElementById(`tags-${ci}`)
    .querySelectorAll('.tag-pill.on'))
    .map(el => el.textContent);
}

// ── Questions helpers ─────────────────────────────────────────────────────────
function getQs(ci) {
  return Array.from(document.getElementById(`qs-${ci}`)
    .querySelectorAll('.question-in'))
    .map(el => el.value.trim())
    .filter(Boolean);
}

function addQ(ci) {
  const container = document.getElementById(`qs-${ci}`);
  const qi = container.querySelectorAll('.question-row').length;
  container.insertAdjacentHTML('beforeend', qRow(ci, qi, ''));
  container.lastElementChild.querySelector('input').focus();
  updateQCount(ci);
}

function delQ(btn) {
  const row = btn.closest('.question-row');
  const ci = row.id.split('-')[1];
  row.remove();
  updateQCount(ci);
}

function updateQCount(ci) {
  const cnt = document.getElementById(`qs-${ci}`)
    .querySelectorAll('.question-row').length;
  const el = document.getElementById(`qcnt-${ci}`);
  if (el) el.textContent = `(${cnt})`;
}

// ── Save ──────────────────────────────────────────────────────────────────────
async function saveChunk(ci) {
  const card = document.getElementById(`card-${ci}`);
  const btn  = document.getElementById(`savebtn-${ci}`);
  const msg  = document.getElementById(`msg-${ci}`);

  const payload = {
    chunk_id:    card.dataset.id,
    page_url:    card.dataset.url,
    page_title:  card.dataset.title,
    chunk_index: parseInt(card.dataset.cidx),
    section:     document.getElementById(`sec-${ci}`).value.trim(),
    text:        document.getElementById(`txt-${ci}`).value,
    tags:        getActiveTags(ci),
    questions:   getQs(ci),
  };

  if (!payload.text.trim()) { toast('Текст не может быть пустым', 'error'); return; }

  card.classList.add('saving');
  btn.innerHTML = '<span class="spinner"></span> Сохранение...';
  msg.textContent = '';

  try {
    const res = await api('POST', '/api/save', payload);
    card.dataset.id = res.new_id;
    msg.textContent = `✓ Сохранено (+${res.question_count} вопросов)`;
    msg.className = 'save-msg ok';
    toast('Чанк сохранён и переиндексирован', 'success');
    await reloadAll();
  } catch(e) {
    msg.textContent = '✗ ' + e.message;
    msg.className = 'save-msg err';
    toast('Ошибка: ' + e.message, 'error');
  } finally {
    card.classList.remove('saving');
    btn.innerHTML = '💾 Сохранить';
  }
}

// ── Delete ────────────────────────────────────────────────────────────────────
async function delChunk(ci) {
  if (!confirm('Удалить этот чанк? Это нельзя отменить (история сохранится).')) return;
  const card = document.getElementById(`card-${ci}`);
  try {
    await api('POST', '/api/delete', {
      page_url: card.dataset.url,
      chunk_index: parseInt(card.dataset.cidx),
    });
    card.remove();
    toast('Чанк удалён', 'success');
    await reloadAll();
  } catch(e) {
    toast('Ошибка удаления: ' + e.message, 'error');
  }
}

// ── New chunk ─────────────────────────────────────────────────────────────────
async function addNewChunk() {
  const text = prompt('Текст нового чанка:');
  if (!text || !text.trim()) return;
  const section = prompt('Раздел (h2/h3, можно оставить пустым):') || '';
  const page = allPages.find(p => p.url === curUrl) || {};
  try {
    const res = await api('POST', '/api/new', {
      page_url: curUrl,
      page_title: page.title || '',
      section: section.trim(),
      text: text.trim(),
      tags: [],
      questions: [],
    });
    toast(`Чанк добавлен (#${res.chunk_index})`, 'success');
    await loadPage(curUrl);
    await reloadAll();
  } catch(e) {
    toast('Ошибка: ' + e.message, 'error');
  }
}

// ── History ───────────────────────────────────────────────────────────────────
async function showHist(ci) {
  const card = document.getElementById(`card-${ci}`);
  histUrl = card.dataset.url;
  histIdx = parseInt(card.dataset.cidx);

  document.getElementById('hist-title').textContent =
    `История чанка #${histIdx}`;
  document.getElementById('hist-body').innerHTML =
    '<div style="text-align:center;padding:30px"><div class="spinner dark" style="width:20px;height:20px;margin:auto"></div></div>';
  document.getElementById('hist-modal').style.display = 'flex';

  try {
    const history = await api('GET',
      `/api/history?url=${encodeURIComponent(histUrl)}&index=${histIdx}`);
    renderHist(history);
  } catch(e) {
    document.getElementById('hist-body').innerHTML =
      `<div style="color:#e74c3c;padding:20px">Ошибка: ${esc(e.message)}</div>`;
  }
}

function renderHist(history) {
  const body = document.getElementById('hist-body');
  if (!history.length) {
    body.innerHTML = '<div style="color:#bbb;text-align:center;padding:24px">История пуста</div>';
    return;
  }
  body.innerHTML = history.map((v, i) => `
    <div class="hist-item">
      <div class="hist-ts">${esc(v.timestamp)}</div>
      <div class="hist-text">${esc(v.text.length > 250 ? v.text.slice(0,250)+'…' : v.text)}</div>
      ${v.questions && v.questions.length
        ? `<div class="hist-qs">Вопросы: ${v.questions.slice(0,3).map(q=>esc(q)).join(' · ')}${v.questions.length>3?` +${v.questions.length-3}`:''}</div>`
        : ''}
      <div class="hist-foot">
        <button class="btn btn-ghost btn-sm" onclick="restoreHist(${i})">↩ Восстановить</button>
        ${v.tags && v.tags.length ? `<span style="font-size:11px;color:#aaa">${v.tags.map(t=>esc(t)).join(', ')}</span>` : ''}
      </div>
    </div>
  `).join('');
}

async function restoreHist(vi) {
  if (!confirm('Восстановить эту версию? Текущий чанк будет заменён.')) return;
  try {
    await api('POST', '/api/restore', { url: histUrl, index: histIdx, version_index: vi });
    closeHistModal();
    toast('Версия восстановлена', 'success');
    await loadPage(curUrl);
    await reloadAll();
  } catch(e) {
    toast('Ошибка восстановления: ' + e.message, 'error');
  }
}

function closeHistModal(event) {
  if (event && event.target !== document.getElementById('hist-modal')) return;
  document.getElementById('hist-modal').style.display = 'none';
}

// Keyboard shortcut: Escape closes modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeHistModal();
});

// ── Start ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=HTML_PAGE)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk Editor — browser-based Qdrant editor")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Chunk Editor starting on http://localhost:{args.port}")
    logger.info(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
