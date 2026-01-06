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

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from .scanner import LibraryScanner, LibraryCatalog, ReferenceDocument


# =============================================================================
# Database Dependency
# =============================================================================

async def get_db_pool():
    """Get database pool from ServiceContainer."""
    try:
        from src.api.dependencies import get_container
        container = await get_container()
        if not container or not container.database:
            return None
        return container.database.pool
    except Exception:
        return None
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
        self._auto_loaded: bool = False

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

    def ensure_catalog_loaded(self) -> bool:
        """Auto-load catalog from disk if not already loaded. Returns True if catalog exists."""
        if self._catalog is not None:
            return True
        if self._auto_loaded:
            return False  # Already tried, file doesn't exist

        self._auto_loaded = True
        catalog_file = Path(self._catalog_path)
        if catalog_file.exists():
            try:
                self._catalog = LibraryCatalog.from_json(str(catalog_file))
                logger.info(f"Auto-loaded library catalog: {self._catalog.total_documents} documents")
                return True
            except Exception as e:
                logger.warning(f"Failed to auto-load catalog: {e}")
        return False


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
        subspecialties=getattr(doc, 'subspecialties', []),
        evidence_level=getattr(doc, 'evidence_level', None),
        authority_source=doc.authority_source,
        authority_score=doc.authority_score,
        has_images=doc.has_images,
        image_count_estimate=doc.image_count_estimate,
        is_ingested=doc.is_ingested,
        is_new=getattr(doc, 'is_new', False),
        first_seen_date=getattr(doc, 'first_seen_date', None),
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
        subspecialties=getattr(doc, 'subspecialties', []),
        evidence_level=getattr(doc, 'evidence_level', None),
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

        # Pass previous catalog for incremental scanning
        previous_catalog = _state._catalog if _state._catalog and _state._catalog.documents else None

        catalog = await scanner.scan_library(
            recursive=recursive,
            progress_callback=progress_callback,
            previous_catalog=previous_catalog,  # Enable incremental scanning
            incremental=True,  # Skip unchanged files
        )

        # Merge with previous catalog to detect new documents
        new_count = 0
        if previous_catalog:
            new_count = catalog.merge_with_previous(previous_catalog)
            logger.info(f"Detected {new_count} new documents since last scan")

        # Extract scan statistics
        scan_stats = getattr(catalog, 'scan_stats', {})
        scanned_count = scan_stats.get('scanned_count', catalog.total_documents)
        cached_count = scan_stats.get('cached_count', 0)

        # Save catalog
        _state._catalog = catalog
        _state._scan_status = ScanStatus.COMPLETED

        # Persist to disk
        Path("./data").mkdir(exist_ok=True)
        catalog.to_json(_state._catalog_path)

        # Broadcast completion with scan stats
        await _broadcast_to_websockets({
            "type": "scan_complete",
            "total_documents": catalog.total_documents,
            "total_pages": catalog.total_pages,
            "total_chapters": catalog.total_chapters,
            "new_documents": new_count,
            "scanned_count": scanned_count,
            "cached_count": cached_count,
            "scan_date": catalog.scan_date,
        })

        if cached_count > 0:
            logger.info(f"Library scan complete: {catalog.total_documents} documents ({scanned_count} scanned, {cached_count} cached), {new_count} new")
        else:
            logger.info(f"Library scan complete: {catalog.total_documents} documents, {new_count} new")

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
    subspecialty: Optional[str] = Query(None, description="Filter by subspecialty"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    authority_source: Optional[str] = Query(None, description="Filter by authority"),
    evidence_level: Optional[str] = Query(None, description="Filter by evidence level (Ia, Ib, IIa, IIb, III, IV)"),
    min_authority: float = Query(0.0, ge=0.0, le=1.0),
    has_images: Optional[bool] = Query(None),
    is_ingested: Optional[bool] = Query(None),
    is_new: Optional[bool] = Query(None, description="Filter by new documents only"),
    min_pages: int = Query(0, ge=0),
    max_pages: int = Query(999999, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List and search documents in the library catalog.

    Supports filtering by specialty, subspecialty, document type, authority source,
    evidence level, and text search in titles/chapter names.
    """
    # Auto-load catalog if exists, otherwise return empty list
    if not _state.ensure_catalog_loaded():
        return DocumentListResponse(total=0, offset=offset, limit=limit, documents=[])

    results = _state._catalog.search(
        query=query,
        specialty=specialty,
        subspecialty=subspecialty,
        document_type=document_type,
        authority_source=authority_source,
        evidence_level=evidence_level,
        min_authority_score=min_authority,
        has_images=has_images,
        is_ingested=is_ingested,
        is_new=is_new,
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
    # Auto-load or return empty stats
    if not _state.ensure_catalog_loaded():
        return LibraryStatistics(
            total_documents=0,
            total_pages=0,
            total_chapters=0,
            ingested_count=0,
            not_ingested_count=0,
            new_count=0,
            scanned_count=0,
            cached_count=0,
            by_specialty={},
            by_type={},
            by_authority={},
            scan_date=None,
        )

    stats = _state._catalog.get_statistics()
    return LibraryStatistics(**stats)


@router.get("/filters", response_model=FilterOptions)
async def get_filter_options():
    """Get available filter options for UI dropdowns."""
    # Auto-load or return defaults
    if not _state.ensure_catalog_loaded():
        from .scanner import Specialty, DocumentType
        return FilterOptions(
            specialties=[s.value for s in Specialty],
            document_types=[d.value for d in DocumentType],
            authority_sources=[],
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


# =============================================================================
# Book Hierarchy & Tree API
# =============================================================================

@router.get("/tree")
async def get_library_tree(
    specialty: Optional[str] = None,
    include_standalone: bool = True,
):
    """
    Get documents organized in book→chapter hierarchy.

    Returns:
        Tree structure with books containing chapters, plus standalone documents.
        Books are flattened: {id, title, chapters: [...]} instead of {book: {...}, chapters: [...]}
    """
    if not _state.ensure_catalog_loaded():
        raise HTTPException(400, "No catalog available. Run scan first.")

    hierarchy = _state._catalog.get_book_hierarchy()

    # Flatten book structure: move book properties to root level
    flattened_books = []
    for book_data in hierarchy.get('books', []):
        book = book_data.get('book', {})
        chapters = book_data.get('chapters', [])
        # Merge book properties with chapters at root level
        flattened_book = {**book, 'chapters': chapters}
        flattened_books.append(flattened_book)

    hierarchy['books'] = flattened_books

    # Filter by specialty if specified
    if specialty:
        specialty_lower = specialty.lower()
        filtered_books = []
        for book in hierarchy['books']:
            if specialty_lower in [s.lower() for s in book.get('specialties', [])]:
                filtered_books.append(book)
        hierarchy['books'] = filtered_books
        hierarchy['total_books'] = len(filtered_books)

        if include_standalone:
            filtered_standalone = [
                d for d in hierarchy['standalone']
                if specialty_lower in [s.lower() for s in d.get('specialties', [])]
            ]
            hierarchy['standalone'] = filtered_standalone
            hierarchy['total_standalone'] = len(filtered_standalone)

    if not include_standalone:
        hierarchy['standalone'] = []
        hierarchy['total_standalone'] = 0

    return hierarchy


@router.get("/hierarchy/{doc_id}")
async def get_document_hierarchy(doc_id: str):
    """
    Get the full hierarchy for a specific document.

    For a chapter: returns parent book and siblings.
    For a book: returns all children.
    """
    if not _state.ensure_catalog_loaded():
        raise HTTPException(400, "No catalog available. Run scan first.")

    doc = _state._catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"Document {doc_id} not found")

    result = {
        "document": doc.to_dict(),
        "parent": None,
        "siblings": [],
        "children": [],
    }

    # If document has a parent, find parent and siblings
    if doc.parent_id:
        parent = _state._catalog.get_document(doc.parent_id)
        if parent:
            result["parent"] = parent.to_dict()

            # Find siblings (same parent, excluding self)
            for d in _state._catalog.documents:
                if d.parent_id == doc.parent_id and d.id != doc_id:
                    result["siblings"].append({
                        "id": d.id,
                        "title": d.title,
                        "sort_order": d.sort_order,
                    })

            result["siblings"].sort(key=lambda s: s["sort_order"])

    # If document is a book, find children
    if doc.section_type == "book":
        for d in _state._catalog.documents:
            if d.parent_id == doc_id:
                result["children"].append({
                    "id": d.id,
                    "title": d.title,
                    "sort_order": d.sort_order,
                    "page_count": d.page_count,
                })
        result["children"].sort(key=lambda c: c["sort_order"])

    return result


# =============================================================================
# Specialty Hierarchy API
# =============================================================================

@router.get("/specialties/tree")
async def get_specialty_tree():
    """
    Get hierarchical specialty taxonomy.

    Returns specialty→subspecialty→sub-subspecialty tree structure.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise Exception("Database pool not available")
        async with pool.acquire() as conn:
            # Fetch all specialties
            rows = await conn.fetch("""
                SELECT id, name, parent_id, level, keywords, icon, description, sort_order
                FROM specialties
                WHERE is_active = true
                ORDER BY level, sort_order, name
            """)

            # Build tree structure
            specialties_by_id = {}
            root_specialties = []

            for row in rows:
                spec = {
                    "id": row["id"],
                    "name": row["name"],
                    "parent_id": row["parent_id"],
                    "level": row["level"],
                    "keywords": row["keywords"] or [],
                    "icon": row["icon"],
                    "description": row["description"],
                    "children": [],
                }
                specialties_by_id[row["id"]] = spec

                if row["parent_id"] is None:
                    root_specialties.append(spec)

            # Link children to parents
            for spec in specialties_by_id.values():
                if spec["parent_id"] and spec["parent_id"] in specialties_by_id:
                    specialties_by_id[spec["parent_id"]]["children"].append(spec)

            return {
                "specialties": root_specialties,
                "total": len(rows),
            }

    except Exception as e:
        logger.warning(f"Failed to load specialty tree from DB: {e}")
        # Fallback to hardcoded enum values
        from .scanner import Specialty
        return {
            "specialties": [
                {"id": i, "name": s.value, "parent_id": None, "level": 1, "children": []}
                for i, s in enumerate(Specialty)
            ],
            "total": len(Specialty),
            "source": "enum_fallback",
        }


@router.get("/specialties/{specialty_id}/documents")
async def get_documents_by_specialty(
    specialty_id: int,
    include_children: bool = True,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get documents belonging to a specialty.

    Args:
        specialty_id: ID of the specialty
        include_children: Include documents from subspecialties (default: True)
        limit: Max documents to return
        offset: Pagination offset
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise Exception("Database pool not available")
        async with pool.acquire() as conn:
            if include_children:
                # Get specialty and all descendants
                spec_ids = await conn.fetch("""
                    SELECT specialty_id FROM get_specialty_with_descendants($1)
                """, specialty_id)
                spec_id_list = [r["specialty_id"] for r in spec_ids]
            else:
                spec_id_list = [specialty_id]

            # Fetch documents
            rows = await conn.fetch("""
                SELECT DISTINCT d.id, d.title, d.file_path,
                       ds.relevance_score, ds.is_primary
                FROM documents d
                JOIN document_specialties ds ON d.id = ds.document_id
                WHERE ds.specialty_id = ANY($1)
                ORDER BY ds.is_primary DESC, ds.relevance_score DESC
                LIMIT $2 OFFSET $3
            """, spec_id_list, limit, offset)

            # Get total count
            count_row = await conn.fetchrow("""
                SELECT COUNT(DISTINCT d.id) as total
                FROM documents d
                JOIN document_specialties ds ON d.id = ds.document_id
                WHERE ds.specialty_id = ANY($1)
            """, spec_id_list)

            return {
                "documents": [dict(row) for row in rows],
                "total": count_row["total"],
                "specialty_ids": spec_id_list,
            }

    except Exception as e:
        logger.error(f"Failed to get documents by specialty: {e}")
        raise HTTPException(500, f"Database error: {e}")


# =============================================================================
# Entity Search API
# =============================================================================

@router.get("/entities/search")
async def search_entities(
    q: str = Query(..., min_length=2, description="Search query"),
    entity_type: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """
    Search entities with autocomplete support.

    Uses trigram similarity for fuzzy matching.

    Args:
        q: Search query (minimum 2 characters)
        entity_type: Filter by entity type (anatomy, procedure, pathology)
        limit: Max results to return
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise Exception("Database pool not available")
        async with pool.acquire() as conn:
            # Enable trigram extension if not already
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

            # Search entities with similarity scoring
            # Note: entities table uses 'category' not 'entity_type', and 'name' column
            query = """
                SELECT
                    id,
                    name,
                    category,
                    cui,
                    semantic_type,
                    similarity(name, $1) as score
                FROM entities
                WHERE name IS NOT NULL AND (
                    name ILIKE $2
                    OR name % $1
                )
            """
            params = [q, f"%{q}%"]

            if entity_type:
                query += " AND category = $3"
                params.append(entity_type)

            query += " ORDER BY score DESC, name LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(query, *params)

            return {
                "entities": [
                    {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "type": row["category"] or row["semantic_type"] or "unknown",
                        "cui": row["cui"],
                        "score": float(row["score"]) if row["score"] else 0.0,
                    }
                    for row in rows
                ],
                "query": q,
                "total": len(rows),
            }

    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        # Fallback to in-memory search if DB fails
        return {
            "entities": [],
            "query": q,
            "total": 0,
            "error": str(e),
        }


@router.get("/entities/{entity_id}/documents")
async def get_documents_by_entity(
    entity_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get documents that mention a specific entity.

    Uses the chunks.entities JSONB field to find documents.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise Exception("Database pool not available")
        async with pool.acquire() as conn:
            # Find chunks containing this entity and get their documents
            rows = await conn.fetch("""
                SELECT DISTINCT
                    d.id,
                    d.title,
                    d.file_path,
                    d.authority_score,
                    COUNT(c.id) as mention_count
                FROM documents d
                JOIN chunks c ON c.document_id = d.id
                WHERE c.entities @> $1::jsonb
                   OR $2 = ANY(c.cuis)
                GROUP BY d.id
                ORDER BY mention_count DESC, d.authority_score DESC
                LIMIT $3 OFFSET $4
            """, f'[{{"id": "{entity_id}"}}]', entity_id, limit, offset)

            return {
                "documents": [dict(row) for row in rows],
                "entity_id": entity_id,
                "total": len(rows),
            }

    except Exception as e:
        logger.error(f"Failed to get documents by entity: {e}")
        raise HTTPException(500, f"Database error: {e}")


# =============================================================================
# Reading Lists API
# =============================================================================

from .reading_lists import (
    ReadingListManager,
    ReadingListCreate,
    ReadingListUpdate,
    ReadingListItemAdd,
    ReadingListItemUpdate,
)


@router.get("/lists")
async def get_reading_lists(
    procedure_slug: Optional[str] = Query(None, description="Filter by linked procedure"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
):
    """
    Get all reading lists, optionally filtered by procedure or specialty.

    Returns lists with item counts, sorted by most recently updated.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        lists = await manager.get_lists(
            procedure_slug=procedure_slug,
            specialty=specialty,
        )
        return {"lists": lists, "total": len(lists)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reading lists: {e}")
        raise HTTPException(500, f"Failed to get reading lists: {e}")


@router.post("/lists")
async def create_reading_list(data: ReadingListCreate):
    """
    Create a new reading list.

    Can optionally be linked to a procedure for case preparation context.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        new_list = await manager.create_list(data)
        return new_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create reading list: {e}")
        raise HTTPException(500, f"Failed to create reading list: {e}")


@router.get("/lists/{list_id}")
async def get_reading_list(list_id: str):
    """
    Get a reading list with all its items.

    Items are returned in order with priority and notes.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        result = await manager.get_list_with_items(list_id)

        if not result:
            raise HTTPException(404, f"Reading list not found: {list_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reading list {list_id}: {e}")
        raise HTTPException(500, f"Failed to get reading list: {e}")


@router.patch("/lists/{list_id}")
async def update_reading_list(list_id: str, data: ReadingListUpdate):
    """
    Update a reading list's metadata (name, description, etc.).
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        result = await manager.update_list(list_id, data)

        if not result:
            raise HTTPException(404, f"Reading list not found: {list_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update reading list {list_id}: {e}")
        raise HTTPException(500, f"Failed to update reading list: {e}")


@router.delete("/lists/{list_id}")
async def delete_reading_list(list_id: str):
    """
    Delete a reading list and all its items.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        deleted = await manager.delete_list(list_id)

        if not deleted:
            raise HTTPException(404, f"Reading list not found: {list_id}")

        return {"status": "deleted", "id": list_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete reading list {list_id}: {e}")
        raise HTTPException(500, f"Failed to delete reading list: {e}")


@router.post("/lists/{list_id}/items")
async def add_item_to_list(list_id: str, data: ReadingListItemAdd):
    """
    Add a document to a reading list.

    Priority levels:
    - 1 = Essential (must read)
    - 2 = Recommended (should read)
    - 3 = Optional (nice to have)
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        added = await manager.add_item(list_id, data)

        if not added:
            raise HTTPException(409, "Document already in list")

        return {"status": "added", "document_id": data.document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add item to list {list_id}: {e}")
        raise HTTPException(500, f"Failed to add item: {e}")


@router.post("/lists/{list_id}/items/batch")
async def add_items_batch(
    list_id: str,
    document_ids: List[str],
    priority: int = Query(2, ge=1, le=3),
):
    """
    Add multiple documents to a reading list at once.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        added_count = await manager.add_items_batch(list_id, document_ids, priority)

        return {
            "status": "added",
            "added_count": added_count,
            "requested_count": len(document_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add items batch to list {list_id}: {e}")
        raise HTTPException(500, f"Failed to add items: {e}")


@router.patch("/lists/{list_id}/items/{document_id:path}")
async def update_list_item(
    list_id: str,
    document_id: str,
    data: ReadingListItemUpdate,
):
    """
    Update an item's priority, notes, or position.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        updated = await manager.update_item(list_id, document_id, data)

        if not updated:
            raise HTTPException(404, "Item not found in list")

        return {"status": "updated", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update item in list {list_id}: {e}")
        raise HTTPException(500, f"Failed to update item: {e}")


@router.delete("/lists/{list_id}/items/{document_id:path}")
async def remove_item_from_list(list_id: str, document_id: str):
    """
    Remove a document from a reading list.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        removed = await manager.remove_item(list_id, document_id)

        if not removed:
            raise HTTPException(404, "Item not found in list")

        return {"status": "removed", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove item from list {list_id}: {e}")
        raise HTTPException(500, f"Failed to remove item: {e}")


@router.put("/lists/{list_id}/reorder")
async def reorder_list_items(list_id: str, document_ids: List[str]):
    """
    Reorder items in a reading list.

    Pass the document IDs in the desired order.
    """
    try:
        pool = await get_db_pool()
        if not pool:
            raise HTTPException(503, "Database not available")

        manager = ReadingListManager(pool)
        await manager.reorder_items(list_id, document_ids)

        return {"status": "reordered", "list_id": list_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reorder items in list {list_id}: {e}")
        raise HTTPException(500, f"Failed to reorder items: {e}")
