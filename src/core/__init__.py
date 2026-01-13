"""
NeuroSynth v2.0 - Core Module
"""

from src.core.logging_config import (
    configure_logging,
    get_logger,
    RequestLoggingMiddleware,
    get_request_id,
    set_request_id,
    bind_user,
    bind_document,
)

from src.shared.models import (
    Document,
    Page,
    Section,
    SemanticChunk,
    ExtractedImage,
    ExtractedTable,
    NeuroEntity,
    EntityRelation,
    DocumentStatus,
    ChunkType,
    ImageType,
    EntityCategory,
    SearchResult,
    ImageSearchResult,
    ExtractionMetrics,
)

from .neuro_extractor import (
    NeuroExpertPatterns,
    NeuroExpertTextExtractor,
)

from .neuro_chunker import (
    NeuroSemanticChunker,
    TableAwareChunker,
    ChunkerConfig,
)

from .database import NeuroDatabase

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "RequestLoggingMiddleware",
    "get_request_id",
    "set_request_id",
    "bind_user",
    "bind_document",
    # Models
    "Document",
    "Page",
    "Section",
    "SemanticChunk",
    "ExtractedImage",
    "ExtractedTable",
    "NeuroEntity",
    "EntityRelation",
    "DocumentStatus",
    "ChunkType",
    "ImageType",
    "EntityCategory",
    "SearchResult",
    "ImageSearchResult",
    "ExtractionMetrics",
    # Extraction
    "NeuroExpertPatterns",
    "NeuroExpertTextExtractor",
    # Chunking
    "NeuroSemanticChunker",
    "TableAwareChunker",
    "ChunkerConfig",
    # Database
    "NeuroDatabase",
]
