"""
NeuroSynth - Library Browser API Models
=======================================

Pydantic models for the library browser API.
These models define the contract between backend and frontend.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Enums (mirror scanner.py)
# =============================================================================

class DocumentType(str, Enum):
    TEXTBOOK = "textbook"
    ATLAS = "atlas"
    HANDBOOK = "handbook"
    JOURNAL_ARTICLE = "journal_article"
    REVIEW = "review"
    CASE_SERIES = "case_series"
    GUIDELINES = "guidelines"
    EXAM_QUESTIONS = "exam_questions"
    COURSE_MATERIAL = "course_material"
    LECTURE_NOTES = "lecture_notes"
    OPERATIVE_VIDEO_COMPANION = "operative_video_companion"
    CHAPTER = "chapter"
    UNKNOWN = "unknown"


class Specialty(str, Enum):
    VASCULAR = "vascular"
    TUMOR = "tumor"
    SKULL_BASE = "skull_base"
    SPINE = "spine"
    FUNCTIONAL = "functional"
    PEDIATRIC = "pediatric"
    TRAUMA = "trauma"
    PERIPHERAL_NERVE = "peripheral_nerve"
    NEURORADIOLOGY = "neuroradiology"
    NEUROANATOMY = "neuroanatomy"
    GENERAL = "general"


class ScanStatus(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestStatus(str, Enum):
    NOT_INGESTED = "not_ingested"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Response Models
# =============================================================================

class ChapterSummary(BaseModel):
    """Chapter metadata for list views."""
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    page_count: int
    has_images: bool
    specialties: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chap-123",
                "title": "Retrosigmoid Approach",
                "level": 1,
                "page_start": 245,
                "page_end": 278,
                "page_count": 34,
                "has_images": True,
                "specialties": ["skull_base"]
            }
        }


class ChapterDetail(ChapterSummary):
    """Full chapter details including preview."""
    word_count_estimate: int = 0
    image_count_estimate: int = 0
    keywords: List[str] = Field(default_factory=list)
    preview: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chap-123",
                "title": "Retrosigmoid Approach",
                "level": 1,
                "page_start": 245,
                "page_end": 278,
                "page_count": 34,
                "has_images": True,
                "specialties": ["skull_base"],
                "word_count_estimate": 10200,
                "image_count_estimate": 15,
                "keywords": ["retrosigmoid", "cerebellopontine", "sigmoid sinus"],
                "preview": "The retrosigmoid approach provides excellent exposure..."
            }
        }


class DocumentSummary(BaseModel):
    """Document metadata for list views."""
    id: str
    title: str
    file_name: str
    file_size_mb: float
    page_count: int
    chapter_count: int
    document_type: str
    primary_specialty: str
    specialties: List[str]
    subspecialties: List[str] = Field(default_factory=list)  # Nested under primary_specialty
    evidence_level: Optional[str] = None  # Ia, Ib, IIa, IIb, III, IV (for journal articles)
    authority_source: str
    authority_score: float
    has_images: bool
    image_count_estimate: int
    is_ingested: bool = False
    is_new: bool = False  # True if discovered in most recent scan
    first_seen_date: Optional[str] = None  # Date document was first discovered

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc-456",
                "title": "Rhoton's Cranial Anatomy and Surgical Approaches",
                "file_name": "rhoton_cranial_vol1.pdf",
                "file_size_mb": 125.4,
                "page_count": 847,
                "chapter_count": 24,
                "document_type": "atlas",
                "primary_specialty": "skull_base",
                "specialties": ["skull_base", "neuroanatomy"],
                "subspecialties": ["microsurgical_anatomy", "surgical_approaches"],
                "evidence_level": None,
                "authority_source": "RHOTON",
                "authority_score": 1.0,
                "has_images": True,
                "image_count_estimate": 892,
                "is_ingested": False
            }
        }


class DocumentDetail(DocumentSummary):
    """Full document details including chapters."""
    file_path: str
    content_hash: str
    authors: Optional[str] = None
    year: Optional[int] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    isbn: Optional[str] = None
    series: Optional[str] = None
    volume: Optional[int] = None
    has_toc: bool = False
    has_index: bool = False
    has_tables: bool = False
    word_count_estimate: int = 0
    chapters: List[ChapterDetail] = Field(default_factory=list)
    all_keywords: List[str] = Field(default_factory=list)
    scan_date: str = ""
    ingested_date: Optional[str] = None
    ingested_document_id: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    total: int
    offset: int
    limit: int
    documents: List[DocumentSummary]


class ChapterSearchResult(BaseModel):
    """Chapter search result with parent document info."""
    document_id: str
    document_title: str
    file_name: str
    authority_source: str
    authority_score: float
    chapter: ChapterDetail
    match_score: float = 0.0  # Fuzzy match score


class ChapterSearchResponse(BaseModel):
    """Paginated chapter search results."""
    total: int
    offset: int
    limit: int
    query: str
    results: List[ChapterSearchResult]


class LibraryStatistics(BaseModel):
    """Library catalog statistics."""
    total_documents: int
    total_pages: int
    total_chapters: int
    ingested_count: int = 0
    not_ingested_count: int = 0
    new_count: int = 0  # Documents discovered in most recent scan
    scanned_count: int = 0  # Files actually scanned (not cached)
    cached_count: int = 0  # Files reused from previous scan
    by_specialty: Dict[str, int]
    by_type: Dict[str, int]
    by_authority: Dict[str, int]
    scan_date: str


class FilterOptions(BaseModel):
    """Available filter options for the UI."""
    specialties: List[str]
    subspecialties: Dict[str, List[str]] = Field(default_factory=dict)  # specialty -> subspecialties
    document_types: List[str]
    authority_sources: List[str]
    evidence_levels: List[str] = Field(default_factory=list)  # Ia, Ib, IIa, IIb, III, IV


# =============================================================================
# Request Models
# =============================================================================

class ScanRequest(BaseModel):
    """Request to scan a library."""
    path: str = Field(..., description="Path to PDF library directory")
    recursive: bool = Field(True, description="Scan subdirectories")


class DocumentSearchRequest(BaseModel):
    """Document search/filter request."""
    query: Optional[str] = Field(None, description="Full-text search")
    specialty: Optional[str] = Field(None, description="Filter by specialty")
    subspecialty: Optional[str] = Field(None, description="Filter by subspecialty")
    document_type: Optional[str] = Field(None, description="Filter by document type")
    authority_source: Optional[str] = Field(None, description="Filter by authority")
    evidence_level: Optional[str] = Field(None, description="Filter by evidence level (Ia, Ib, IIa, IIb, III, IV)")
    min_authority_score: float = Field(0.0, ge=0.0, le=1.0)
    has_images: Optional[bool] = None
    is_ingested: Optional[bool] = None
    min_pages: int = Field(0, ge=0)
    max_pages: int = Field(999999, ge=0)
    offset: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


class ChapterSearchRequest(BaseModel):
    """Chapter search request."""
    query: str = Field(..., min_length=2, description="Search query")
    specialty: Optional[str] = None
    min_score: float = Field(60.0, ge=0.0, le=100.0, description="Minimum fuzzy match score")
    offset: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


class ChapterSelection(BaseModel):
    """Selection of specific chapters from a document."""
    document_id: str
    chapter_ids: List[str]


class IngestSelectionRequest(BaseModel):
    """Request to ingest selected documents/chapters."""
    document_ids: List[str] = Field(default_factory=list, description="Full documents to ingest")
    chapter_selections: List[ChapterSelection] = Field(
        default_factory=list,
        description="Specific chapters to ingest"
    )
    priority: int = Field(1, ge=1, le=5, description="Queue priority (1=highest)")
    config: Optional[Dict[str, Any]] = Field(None, description="Pipeline config overrides")


class IngestSelectionResponse(BaseModel):
    """Response after queueing ingestion."""
    batch_id: str
    jobs_queued: int
    document_jobs: List[Dict[str, Any]]
    chapter_jobs: List[Dict[str, Any]]


# =============================================================================
# Status/Progress Models
# =============================================================================

class ScanStatusResponse(BaseModel):
    """Current scan status."""
    status: ScanStatus
    current: int = 0
    total: int = 0
    current_file: str = ""
    percent_complete: float = 0.0
    error: Optional[str] = None


class ScanProgressUpdate(BaseModel):
    """Real-time scan progress update (WebSocket)."""
    type: str = "scan_progress"
    status: ScanStatus
    current: int
    total: int
    current_file: str
    percent_complete: float


class ScanCompleteUpdate(BaseModel):
    """Scan completion message (WebSocket)."""
    type: str = "scan_complete"
    total_documents: int
    total_pages: int
    total_chapters: int
    new_documents: int = 0  # Newly discovered documents
    scanned_count: int = 0  # Files actually scanned
    cached_count: int = 0  # Files reused from cache
    scan_date: str


class ScanErrorUpdate(BaseModel):
    """Scan error message (WebSocket)."""
    type: str = "scan_error"
    error: str
    failed_files: List[str] = Field(default_factory=list)
