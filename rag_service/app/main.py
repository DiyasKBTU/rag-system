# -*- coding: utf-8 -*-
"""
main.py - FastAPI server for RAG service.

Endpoints:
  POST /search  - Find relevant chunks for a question (called by Django bot)
  POST /index   - Trigger manual re-indexing of all pages
  GET  /health  - Health check (is the server alive?)
  GET  /stats   - How many chunks are stored in Qdrant?

How Django bot uses this:
  1. User sends question to Telegram bot
  2. Django bot calls POST /search with the question
  3. RAG service returns relevant text chunks
  4. Django bot sends question + chunks to ChatGPT
  5. ChatGPT returns answer -> bot sends to user

Security:
  All endpoints (except /health) require API key in header:
  X-API-Key: your_secret_key_from_env
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.retrieval.search import search, search_and_format_context
from app.indexer.storage import get_collection_stats, ensure_collection_exists


# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called once when server starts.
    We connect to Qdrant and make sure collection exists.
    """
    logger.info("Starting RAG service...")
    try:
        ensure_collection_exists()
        stats = get_collection_stats()
        logger.info(f"Qdrant ready. Chunks in collection: {stats['total_chunks']}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.warning("Make sure Docker is running: docker-compose up -d")

    yield  # Server is running

    logger.info("RAG service shutting down.")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="CAIU RAG Service",
    description="Semantic search API for CAIU Telegram bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (needed if Django is on different host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security: API Key check ────────────────────────────────────
def verify_api_key(x_api_key: str = Header(..., description="Secret API key")):
    """
    Check that the request has a valid API key.
    Django bot sends this key in every request header.

    Usage: add  X-API-Key: your_secret_key  to request headers.
    """
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Request / Response models ─────────────────────────────────
class SearchRequest(BaseModel):
    """What Django bot sends to /search"""
    question: str                      # User's question
    top_k: Optional[int] = None        # How many chunks to return (default from config)
    min_score: Optional[float] = None  # Minimum relevance score (default from config)
    format_as_context: bool = True     # True = return formatted string; False = return raw list

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Как поступить в КАИУ?",
                "top_k": 5,
                "min_score": 0.3,
                "format_as_context": True,
            }
        }


class SearchResultItem(BaseModel):
    """One chunk returned by /search"""
    text: str          # Chunk text
    page_url: str      # Source URL
    page_title: str    # Source page title
    chunk_index: int   # Position on page (0 = first chunk)
    score: float       # Relevance score (0.0 to 1.0)


class SearchResponse(BaseModel):
    """What /search returns to Django bot"""
    question: str
    context: str                        # Formatted context string for ChatGPT
    results: List[SearchResultItem]     # Raw chunks (for debugging)
    total_found: int                    # How many chunks found
    search_time_ms: int                 # How long search took


class IndexRequest(BaseModel):
    """Request body for /index endpoint"""
    # Optional: run in background (default True)
    # If False, wait for indexing to complete (can take minutes!)
    background: bool = True


class IndexResponse(BaseModel):
    """Response from /index"""
    message: str
    status: str  # "started" or "completed"


class StatsResponse(BaseModel):
    """Response from /stats"""
    total_chunks: int
    collection: str
    status: str


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check - no auth required.
    Returns 200 if server is alive.
    Used by monitoring tools, load balancers, etc.
    """
    return {"status": "ok", "service": "CAIU RAG Service"}


@app.get("/stats", response_model=StatsResponse)
def get_stats(api_key: str = Depends(verify_api_key)):
    """
    Get collection statistics.
    How many chunks are in Qdrant?
    """
    stats = get_collection_stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        collection=stats["collection"],
        status=str(stats["status"]),
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Main endpoint - semantic search over indexed chunks.

    Django bot calls this for every user question.
    Returns relevant text chunks that get passed to ChatGPT.

    Example request:
        POST /search
        Headers: X-API-Key: your_secret_key
        Body: {"question": "Как поступить в КАИУ?"}

    Example response:
        {
            "question": "Как поступить в КАИУ?",
            "context": "[Source: Поступление]\\nДокументы принимаются...",
            "results": [...],
            "total_found": 3,
            "search_time_ms": 145
        }
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question.strip()
    logger.info(f"Search request: '{question[:80]}...' " if len(question) > 80 else f"Search request: '{question}'")

    start_time = time.time()

    try:
        # Search for relevant chunks
        results = search(
            question=question,
            top_k=request.top_k,
            min_score=request.min_score,
        )

        # Format as context string for ChatGPT
        # Uses search_and_format_context to respect MAX_CONTEXT_CHARS limit
        if results:
            context = search_and_format_context(question)
            # If search_and_format_context returned empty (shouldn't happen
            # if results is non-empty), fall back to manual formatting
            if not context:
                context_parts = []
                for result in results:
                    source = result.page_title or result.page_url
                    part = f"[Источник: {source}]\n{result.text}"
                    context_parts.append(part)
                context = "\n\n".join(context_parts)
        else:
            context = ""

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Found {len(results)} chunks in {elapsed_ms}ms")

        return SearchResponse(
            question=question,
            context=context,
            results=[
                SearchResultItem(
                    text=r.text,
                    page_url=r.page_url,
                    page_title=r.page_title,
                    chunk_index=r.chunk_index,
                    score=r.score,
                )
                for r in results
            ],
            total_found=len(results),
            search_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def _run_indexing():
    """
    Background indexing task.
    Imports here to avoid circular imports and slow startup.
    """
    import time as time_module
    from app.parser.crawler import get_urls_for_indexing
    from app.parser.extractor import get_page_content
    from app.parser.chunker import split_into_chunks
    from app.indexer.storage import save_chunks, page_needs_update

    logger.info("Background indexing started")

    urls = get_urls_for_indexing()
    logger.info(f"Indexing {len(urls)} pages...")

    total_pages = 0
    total_chunks = 0

    for i, url_item in enumerate(urls, 1):
        url = url_item.url
        logger.info(f"[{i}/{len(urls)}] Processing: {url}")

        content = get_page_content(url)
        if not content:
            logger.warning(f"  Skipped (no content): {url}")
            time_module.sleep(settings.REQUEST_DELAY)
            continue

        if not page_needs_update(url, content.content_hash):
            logger.info(f"  Skipped (not changed): {url}")
            continue

        chunks = split_into_chunks(
            text=content.text,
            page_url=content.url,
            page_title=content.title,
        )

        if not chunks:
            logger.warning(f"  Skipped (no chunks): {url}")
            time_module.sleep(settings.REQUEST_DELAY)
            continue

        saved = save_chunks(chunks, content.url, content.content_hash)
        total_pages += 1
        total_chunks += saved
        logger.info(f"  Saved {saved} chunks for: {content.title or url}")

        if i < len(urls):
            time_module.sleep(settings.REQUEST_DELAY)

    stats = get_collection_stats()
    logger.info(
        f"Indexing complete! "
        f"Pages: {total_pages}, "
        f"New chunks: {total_chunks}, "
        f"Total in Qdrant: {stats['total_chunks']}"
    )


@app.post("/index", response_model=IndexResponse)
def index_endpoint(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Trigger re-indexing of all pages.

    If background=True (default): starts indexing in background, returns immediately.
    If background=False: waits for indexing to complete (WARNING: can take 10-30 minutes!).

    Call this after updating content on caiu.edu.kz.

    Example request:
        POST /index
        Headers: X-API-Key: your_secret_key
        Body: {"background": true}
    """
    logger.info(f"Index request received. background={request.background}")

    if request.background:
        background_tasks.add_task(_run_indexing)
        return IndexResponse(
            message="Indexing started in background. Check /stats to monitor progress.",
            status="started",
        )
    else:
        # Synchronous - wait for completion (blocks the request!)
        _run_indexing()
        stats = get_collection_stats()
        return IndexResponse(
            message=f"Indexing complete. Total chunks: {stats['total_chunks']}",
            status="completed",
        )
