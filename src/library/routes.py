"""
NeuroSynth - Library Browser API Routes
=======================================

FastAPI routes for the reference library browser.
Provides endpoints for scanning, browsing, filtering, and selecting
documents/chapters for ingestion.

Mount at: /api/v1/library
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .scanner import LibraryScanner, LibraryCatalog, ReferenceDocument
from .models import (
    ScanRequest,
    ScanStatusResponse,
    ScanStatus,
    ScanProgressUpdate,
    ScanCompleteUpdate,
    DocumentSummary,
    DocumentDetail,
    DocumentListResponse,
    ChapterDetail,
    ChapterSearchResult,
    ChapterSearchResponse,
    LibraryStatistics,
    FilterOptions,
    IngestSelectionRequest,
    IngestSelectionResponse,
)

# Fuzzy search - graceful fallback if not installed
try:
    from thefuzz import fuzz, process
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/library", tags=["Library"])


# =============================================================================
# In-Memory State (Replace with DB/Redis in production)
# =============================================================================

class LibraryState:
    """Singleton state manager for library catalog."""

    def __init__(self):
        self._catalog: Optional[LibraryCatalog] = None
        self._scan_status: ScanStatus = ScanStatus.IDLE
        self._scan_progress: dict = {"current": 0, "total": 0, "current_file": ""}
        self._scan_error: Optional[str] = None
        self._active_websockets: List[WebSocket] = []
        self._catalog_path: str = "./data/library_catalog.json"

    @property
    def catalog(self) -> Optional[LibraryCatalog]:
        return self._catalog

    @catalog.setter
    def catalog(self, value: LibraryCatalog):
        self._catalog = value

    def get_scan_status(self) -> ScanStatusResponse:
        percent = 0.0
        if self._scan_progress["total"] > 0:
            percent = (self._scan_progress["current"] / self._scan_progress["total"]) * 100
        return ScanStatusResponse(
            status=self._scan_status,
            current=self._scan_progress["current"],
            total=self._scan_progress["total"],
            current_file=self._scan_progress["current_file"],
            percent_complete=percent,
            error=self._scan_error,
        )


# Global state instance
_state = LibraryState()


# =============================================================================
# Helper Functions
# =============================================================================

def _doc_to_summary(doc: ReferenceDocument) -> DocumentSummary:
    """Convert internal document to API summary model."""
    return DocumentSummary(
        id=doc.id,
        title=doc.title,
        file_name=doc.file_name,
        file_size_mb=doc.file_size_mb,
        page_count=doc.page_count,
        chapter_count=doc.chapter_count,
        document_type=doc.document_type.value,
        primary_specialty=doc.primary_specialty.value,
        specialties=doc.specialties,
        authority_source=doc.authority_source,
        authority_score=doc.authority_score,
        has_images=doc.has_images,
        image_count_estimate=doc.image_count_estimate,
        is_ingested=doc.is_ingested,
    )


def _doc_to_detail(doc: ReferenceDocument) -> DocumentDetail:
    """Convert internal document to API detail model."""
    chapters = [
        ChapterDetail(
            id=c.id,
            title=c.title,
            level=c.level,
            page_start=c.page_start,
            page_end=c.page_end,
            page_count=c.page_count,
            has_images=c.has_images,
            specialties=c.specialties,
            word_count_estimate=c.word_count_estimate,
            image_count_estimate=c.image_count_estimate,
            keywords=c.keywords,
            preview=c.preview,
        )
        for c in doc.chapters
    ]

    return DocumentDetail(
        id=doc.id,
        title=doc.title,
        file_name=doc.file_name,
        file_path=doc.file_path,
        file_size_mb=doc.file_size_mb,
        content_hash=doc.content_hash,
        page_count=doc.page_count,
        chapter_count=doc.chapter_count,
        document_type=doc.document_type.value,
        primary_specialty=doc.primary_specialty.value,
        specialties=doc.specialties,
        authority_source=doc.authority_source,
        authority_score=doc.authority_score,
        has_images=doc.has_images,
        image_count_estimate=doc.image_count_estimate,
        is_ingested=doc.is_ingested,
        authors=doc.authors,
        year=doc.year,
        publisher=doc.publisher,
        has_toc=doc.has_toc,
        word_count_estimate=doc.word_count_estimate,
        chapters=chapters,
        all_keywords=doc.all_keywords,
        scan_date=doc.scan_date,
        ingested_date=doc.ingested_date,
        ingested_document_id=doc.ingested_document_id,
    )


async def _broadcast_to_websockets(message: dict):
    """Send message to all connected WebSocket clients."""
    disconnected = []
    for ws in _state._active_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected sockets
    for ws in disconnected:
        _state._active_websockets.remove(ws)


# =============================================================================
# Scan Endpoints
# =============================================================================

@router.post("/scan", response_model=ScanStatusResponse)
async def start_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start scanning a PDF library directory.

    Returns immediately with scan status. Use /scan/status or WebSocket
    to monitor progress.
    """
    library_path = Path(request.path)

    if not library_path.exists():
        raise HTTPException(404, f"Path not found: {request.path}")

    if not library_path.is_dir():
        raise HTTPException(400, f"Path is not a directory: {request.path}")

    if _state._scan_status == ScanStatus.SCANNING:
        raise HTTPException(409, "A scan is already in progress")

    # Reset state
    _state._scan_status = ScanStatus.SCANNING
    _state._scan_progress = {"current": 0, "total": 0, "current_file": ""}
    _state._scan_error = None

    # Start background scan
    background_tasks.add_task(
        _run_scan_task,
        str(library_path),
        request.recursive,
    )

    return _state.get_scan_status()


@router.get("/scan", response_model=ScanStatusResponse)
async def trigger_scan_get(
    path: str = Query(..., description="Path to PDF library"),
    recursive: bool = Query(True),
    background_tasks: BackgroundTasks = None,
):
    """GET version of scan endpoint for easier testing."""
    request = ScanRequest(path=path, recursive=recursive)
    return await start_scan(request, background_tasks)


@router.get("/scan/status", response_model=ScanStatusResponse)
async def get_scan_status():
    """Get current scan status."""
    return _state.get_scan_status()


@router.websocket("/scan/ws")
async def scan_websocket(websocket: WebSocket):
    """
    WebSocket for real-time scan progress updates.

    Send: {"path": "/path/to/pdfs", "recursive": true}
    Receive: ScanProgressUpdate messages
    """
    await websocket.accept()
    _state._active_websockets.append(websocket)

    try:
        while True:
            # Wait for scan request or heartbeat
            data = await websocket.receive_json()

            if "path" in data:
                # Start scan
                path = data["path"]
                recursive = data.get("recursive", True)

                if not Path(path).exists():
                    await websocket.send_json({"type": "error", "error": f"Path not found: {path}"})
                    continue

                if _state._scan_status == ScanStatus.SCANNING:
                    await websocket.send_json({"type": "error", "error": "Scan already in progress"})
                    continue

                # Run scan with WebSocket progress
                _state._scan_status = ScanStatus.SCANNING
                asyncio.create_task(_run_scan_task(path, recursive))

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _state._active_websockets:
            _state._active_websockets.remove(websocket)


async def _run_scan_task(library_path: str, recursive: bool):
    """Background task to run the library scan."""
    try:
        scanner = LibraryScanner(library_path)

        def progress_callback(current: int, total: int, filename: str):
            _state._scan_progress = {
                "current": current,
                "total": total,
                "current_file": filename,
            }
            # Broadcast to WebSockets
            asyncio.create_task(_broadcast_to_websockets({
                "type": "scan_progress",
                "status": "scanning",
                "current": current,
                "total": total,
                "current_file": filename,
                "percent_complete": (current / total * 100) if total > 0 else 0,
            }))

        catalog = await scanner.scan_library(
            recursive=recursive,
            progress_callback=progress_callback,
        )

        # Save catalog
        _state._catalog = catalog
        _state._scan_status = ScanStatus.COMPLETED

        # Persist to disk
        Path("./data").mkdir(exist_ok=True)
        catalog.to_json(_state._catalog_path)

        # Broadcast completion
        await _broadcast_to_websockets({
            "type": "scan_complete",
            "total_documents": catalog.total_documents,
            "total_pages": catalog.total_pages,
            "total_chapters": catalog.total_chapters,
            "scan_date": catalog.scan_date,
        })

        logger.info(f"Library scan complete: {catalog.total_documents} documents")

    except Exception as e:
        _state._scan_status = ScanStatus.FAILED
        _state._scan_error = str(e)
        logger.error(f"Library scan failed: {e}")

        await _broadcast_to_websockets({
            "type": "scan_error",
            "error": str(e),
        })


# =============================================================================
# Document Endpoints
# =============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    query: Optional[str] = Query(None, description="Search in titles, chapters"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    authority_source: Optional[str] = Query(None, description="Filter by authority"),
    min_authority: float = Query(0.0, ge=0.0, le=1.0),
    has_images: Optional[bool] = Query(None),
    is_ingested: Optional[bool] = Query(None),
    min_pages: int = Query(0, ge=0),
    max_pages: int = Query(999999, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List and search documents in the library catalog.

    Supports filtering by specialty, document type, authority source,
    and text search in titles/chapter names.
    """
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first or load from file.")

    results = _state._catalog.search(
        query=query,
        specialty=specialty,
        document_type=document_type,
        authority_source=authority_source,
        min_authority_score=min_authority,
        has_images=has_images,
        is_ingested=is_ingested,
        min_pages=min_pages,
        max_pages=max_pages,
    )

    total = len(results)
    paginated = results[offset:offset + limit]

    return DocumentListResponse(
        total=total,
        offset=offset,
        limit=limit,
        documents=[_doc_to_summary(d) for d in paginated],
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
    """Get detailed document metadata including all chapters."""
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first.")

    doc = _state._catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"Document not found: {doc_id}")

    return _doc_to_detail(doc)


# =============================================================================
# Chapter Search Endpoints
# =============================================================================

@router.get("/chapters/search", response_model=ChapterSearchResponse)
async def search_chapters(
    query: str = Query(..., min_length=2, description="Search in chapter titles"),
    specialty: Optional[str] = Query(None),
    min_score: float = Query(60.0, ge=0.0, le=100.0, description="Minimum fuzzy match score"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Fuzzy search within chapter titles and previews.

    Useful for finding content across all documents, e.g.,
    "Find me everything about aneurysms" or "pterional approach".

    Uses fuzzy matching to find partial matches and typo-tolerant search.
    """
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first.")

    # Build list of all chapters with document info
    all_chapters = []
    for doc in _state._catalog.documents:
        # Apply specialty filter at document level
        if specialty and specialty.lower() not in [s.lower() for s in doc.specialties]:
            continue

        for chapter in doc.chapters:
            all_chapters.append({
                "document": doc,
                "chapter": chapter,
                "searchable": f"{chapter.title} {chapter.preview}",
            })

    results = []

    if HAS_FUZZY:
        # Fuzzy search using thefuzz
        choices = [c["searchable"] for c in all_chapters]
        matches = process.extract(query, choices, limit=limit * 2, scorer=fuzz.token_set_ratio)

        for matched_text, score in matches:
            if score >= min_score:
                # Find original chapter
                for item in all_chapters:
                    if item["searchable"] == matched_text:
                        results.append({
                            "document": item["document"],
                            "chapter": item["chapter"],
                            "score": score,
                        })
                        break
    else:
        # Simple substring search fallback
        query_lower = query.lower()
        for item in all_chapters:
            if query_lower in item["searchable"].lower():
                results.append({
                    "document": item["document"],
                    "chapter": item["chapter"],
                    "score": 100.0,  # Exact match
                })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    total = len(results)
    paginated = results[offset:offset + limit]

    return ChapterSearchResponse(
        total=total,
        offset=offset,
        limit=limit,
        query=query,
        results=[
            ChapterSearchResult(
                document_id=r["document"].id,
                document_title=r["document"].title,
                file_name=r["document"].file_name,
                authority_source=r["document"].authority_source,
                authority_score=r["document"].authority_score,
                match_score=r["score"],
                chapter=ChapterDetail(
                    id=r["chapter"].id,
                    title=r["chapter"].title,
                    level=r["chapter"].level,
                    page_start=r["chapter"].page_start,
                    page_end=r["chapter"].page_end,
                    page_count=r["chapter"].page_count,
                    has_images=r["chapter"].has_images,
                    specialties=r["chapter"].specialties,
                    word_count_estimate=r["chapter"].word_count_estimate,
                    image_count_estimate=r["chapter"].image_count_estimate,
                    keywords=r["chapter"].keywords,
                    preview=r["chapter"].preview,
                ),
            )
            for r in paginated
        ],
    )


# =============================================================================
# Statistics & Filters Endpoints
# =============================================================================

@router.get("/statistics", response_model=LibraryStatistics)
async def get_statistics():
    """Get catalog statistics for dashboard display."""
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first.")

    stats = _state._catalog.get_statistics()
    return LibraryStatistics(**stats)


@router.get("/filters", response_model=FilterOptions)
async def get_filter_options():
    """Get available filter options for UI dropdowns."""
    if _state._catalog is None:
        # Return defaults
        from .scanner import Specialty, DocumentType
        return FilterOptions(
            specialties=[s.value for s in Specialty],
            document_types=[d.value for d in DocumentType],
            authority_sources=list(_state._catalog._by_authority.keys()) if _state._catalog else [],
        )

    options = _state._catalog.get_filter_options()
    return FilterOptions(**options)


# =============================================================================
# Catalog Persistence Endpoints
# =============================================================================

@router.post("/catalog/save")
async def save_catalog(
    path: str = Query("./data/library_catalog.json", description="Output JSON path"),
):
    """Save current catalog to JSON file."""
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first.")

    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _state._catalog.to_json(str(output_path))
        return {"status": "saved", "path": str(output_path)}
    except Exception as e:
        raise HTTPException(500, f"Failed to save catalog: {e}")


@router.post("/catalog/load")
async def load_catalog(
    path: str = Query("./data/library_catalog.json", description="JSON catalog path"),
):
    """Load catalog from JSON file."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise HTTPException(404, f"Catalog file not found: {path}")

    try:
        _state._catalog = LibraryCatalog.from_json(str(catalog_path))
        return {
            "status": "loaded",
            "statistics": _state._catalog.get_statistics(),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load catalog: {e}")


# =============================================================================
# Ingestion Bridge Endpoint
# =============================================================================

@router.post("/ingest-selected", response_model=IngestSelectionResponse)
async def ingest_selected(request: IngestSelectionRequest):
    """
    Queue selected documents/chapters for ingestion.

    Supports both full document ingestion and chapter-level (page range) ingestion.
    Returns batch_id to track progress via the standard ingest endpoints.
    """
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded. Run /scan first.")

    # Import ingest bridge (lazy to avoid circular imports)
    from .ingest_bridge import create_ingest_jobs

    try:
        batch_id, document_jobs, chapter_jobs = await create_ingest_jobs(
            catalog=_state._catalog,
            document_ids=request.document_ids,
            chapter_selections=request.chapter_selections,
            config=request.config,
        )

        return IngestSelectionResponse(
            batch_id=batch_id,
            jobs_queued=len(document_jobs) + len(chapter_jobs),
            document_jobs=document_jobs,
            chapter_jobs=chapter_jobs,
        )

    except Exception as e:
        logger.error(f"Failed to queue ingestion: {e}")
        raise HTTPException(500, f"Failed to queue ingestion: {e}")


# =============================================================================
# Ingestion Status Sync
# =============================================================================

@router.post("/sync-ingested")
async def sync_ingested_status():
    """
    Sync catalog with database to update ingested status.

    Call this after ingestion completes to mark documents as ingested.
    """
    if _state._catalog is None:
        raise HTTPException(400, "No catalog loaded.")

    # Import database check (lazy)
    from .ingest_bridge import sync_ingested_documents

    try:
        updated_count = await sync_ingested_documents(_state._catalog)
        # Save updated catalog
        _state._catalog.to_json(_state._catalog_path)

        return {
            "status": "synced",
            "updated_documents": updated_count,
        }
    except Exception as e:
        logger.error(f"Failed to sync ingested status: {e}")
        raise HTTPException(500, f"Failed to sync: {e}")
