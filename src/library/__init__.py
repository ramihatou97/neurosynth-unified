"""
NeuroSynth - Library Scanner Module
====================================

PDF library scanning and metadata extraction for selective ingestion.

Features:
- Scan PDF directories for metadata extraction
- Visual chapter detection (fallback for PDFs without TOC)
- Authority source scoring
- Specialty classification
- Fuzzy chapter title search
- Chapter-level ingestion via page ranges
"""

from .scanner import (
    LibraryScanner,
    LibraryCatalog,
    ReferenceDocument,
    ChapterInfo,
    DocumentType,
    Specialty,
    AUTHORITY_PATTERNS,
    SPECIALTY_KEYWORDS,
)

from .models import (
    DocumentSummary,
    DocumentDetail,
    ChapterSummary,
    ChapterDetail,
    ChapterSearchResult,
    ChapterSearchResponse,
    DocumentListResponse,
    LibraryStatistics,
    FilterOptions,
    ScanRequest,
    DocumentSearchRequest,
    ChapterSearchRequest,
    IngestSelectionRequest,
    IngestSelectionResponse,
    ScanStatusResponse,
    ScanProgressUpdate,
)

from .routes import router as library_router

from .ingest_bridge import (
    create_ingest_jobs,
    sync_ingested_documents,
    extract_chapter_pages,
    get_ingestion_progress,
)

__all__ = [
    # Scanner
    "LibraryScanner",
    "LibraryCatalog",
    "ReferenceDocument",
    "ChapterInfo",
    "DocumentType",
    "Specialty",
    "AUTHORITY_PATTERNS",
    "SPECIALTY_KEYWORDS",
    # Models
    "DocumentSummary",
    "DocumentDetail",
    "ChapterSummary",
    "ChapterDetail",
    "ChapterSearchResult",
    "ChapterSearchResponse",
    "DocumentListResponse",
    "LibraryStatistics",
    "FilterOptions",
    "ScanRequest",
    "DocumentSearchRequest",
    "ChapterSearchRequest",
    "IngestSelectionRequest",
    "IngestSelectionResponse",
    "ScanStatusResponse",
    "ScanProgressUpdate",
    # Router
    "library_router",
    # Bridge
    "create_ingest_jobs",
    "sync_ingested_documents",
    "extract_chapter_pages",
    "get_ingestion_progress",
]
