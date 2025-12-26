"""
NeuroSynth Unified - API Models
================================

Pydantic models for request/response validation.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class SearchMode(str, Enum):
    """Search mode options."""
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"


class QuestionType(str, Enum):
    """Question type for prompt selection."""
    PROCEDURAL = "procedural"
    ANATOMICAL = "anatomical"
    CLINICAL = "clinical"
    DIFFERENTIAL = "differential"
    COMPARATIVE = "comparative"
    EDUCATIONAL = "educational"
    GENERAL = "general"


# =============================================================================
# Search Models
# =============================================================================

class SearchFilters(BaseModel):
    """Search filter options."""
    document_ids: List[str] = Field(default_factory=list, description="Filter by document IDs")
    chunk_types: List[str] = Field(default_factory=list, description="Filter by chunk types")
    specialties: List[str] = Field(default_factory=list, description="Filter by specialties")
    image_types: List[str] = Field(default_factory=list, description="Filter by image types")
    cuis: List[str] = Field(default_factory=list, description="Boost by UMLS CUIs")
    min_page: Optional[int] = Field(None, description="Minimum page number")
    max_page: Optional[int] = Field(None, description="Maximum page number")


class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    mode: SearchMode = Field(SearchMode.HYBRID, description="Search mode")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    include_images: bool = Field(True, description="Include linked images")
    rerank: bool = Field(True, description="Apply re-ranking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "retrosigmoid approach for acoustic neuroma",
                "mode": "hybrid",
                "top_k": 10,
                "filters": {
                    "chunk_types": ["PROCEDURE", "ANATOMY"],
                    "specialties": ["skull_base"]
                },
                "include_images": True
            }
        }


class SearchResultItem(BaseModel):
    """Single search result."""
    id: str
    content: str
    score: float
    result_type: str = "chunk"
    
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    specialty: Optional[str] = None
    image_type: Optional[str] = None
    cuis: List[str] = Field(default_factory=list)
    
    linked_images: List[Dict[str, Any]] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResultItem]
    total_candidates: int
    query: str
    mode: str
    search_time_ms: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "chunk-uuid",
                        "content": "The retrosigmoid approach...",
                        "score": 0.89,
                        "chunk_type": "PROCEDURE",
                        "page_number": 45
                    }
                ],
                "total_candidates": 50,
                "query": "retrosigmoid approach",
                "mode": "hybrid",
                "search_time_ms": 45
            }
        }


# =============================================================================
# RAG Models
# =============================================================================

class RAGRequest(BaseModel):
    """RAG question request."""
    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    question_type: Optional[QuestionType] = Field(None, description="Question type for prompt selection")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    include_citations: bool = Field(True, description="Include citations")
    include_images: bool = Field(True, description="Include linked images")
    max_context_chunks: int = Field(10, ge=1, le=20, description="Maximum context chunks")
    stream: bool = Field(False, description="Stream response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the retrosigmoid approach for acoustic neuroma?",
                "question_type": "procedural",
                "include_citations": True,
                "include_images": True,
                "max_context_chunks": 10
            }
        }


class CitationItem(BaseModel):
    """Citation reference."""
    index: int
    chunk_id: str
    snippet: str
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None


class ImageItem(BaseModel):
    """Linked image."""
    image_id: str
    file_path: str
    caption: str
    image_type: Optional[str] = None


class RAGResponse(BaseModel):
    """RAG response."""
    answer: str
    citations: List[CitationItem]
    used_citations: List[CitationItem]
    images: List[ImageItem]
    
    question: str
    context_chunks_used: int
    generation_time_ms: int
    search_time_ms: int
    total_time_ms: int
    model: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The retrosigmoid approach provides excellent exposure [1]...",
                "citations": [
                    {
                        "index": 1,
                        "chunk_id": "chunk-uuid",
                        "snippet": "The retrosigmoid approach...",
                        "page_number": 45
                    }
                ],
                "used_citations": [{"index": 1, "chunk_id": "chunk-uuid"}],
                "images": [],
                "question": "What is the retrosigmoid approach?",
                "context_chunks_used": 5,
                "generation_time_ms": 2500,
                "search_time_ms": 50,
                "total_time_ms": 2550,
                "model": "claude-sonnet-4-20250514"
            }
        }


# =============================================================================
# Conversation Models
# =============================================================================

class ConversationMessage(BaseModel):
    """Single conversation message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ConversationRequest(BaseModel):
    """Conversation request."""
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    filters: Optional[SearchFilters] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-uuid",
                "message": "Tell me more about the complications"
            }
        }


class ConversationResponse(BaseModel):
    """Conversation response."""
    conversation_id: str
    answer: str
    citations: List[CitationItem]
    history_length: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-uuid",
                "answer": "The main complications include...",
                "citations": [],
                "history_length": 3
            }
        }


# =============================================================================
# Document Models
# =============================================================================

class DocumentSummary(BaseModel):
    """Document summary."""
    id: str
    source_path: str
    title: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    total_images: int = 0
    created_at: Optional[datetime] = None


class DocumentDetail(DocumentSummary):
    """Document with statistics."""
    stats: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentSummary]
    total: int
    page: int
    page_size: int


class ChunkItem(BaseModel):
    """Chunk summary."""
    id: str
    content: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    chunk_type: Optional[str] = None
    specialty: Optional[str] = None
    cuis: List[str] = Field(default_factory=list)


class DocumentChunksResponse(BaseModel):
    """Document chunks."""
    document_id: str
    chunks: List[ChunkItem]
    total: int


# =============================================================================
# Health/Status Models
# =============================================================================

class ComponentStatus(BaseModel):
    """Status of a component."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    latency_ms: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, ComponentStatus]
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 5},
                    "faiss": {"status": "healthy", "details": {"text_size": 177}},
                    "search": {"status": "healthy"}
                },
                "timestamp": "2025-12-26T12:00:00Z"
            }
        }


class StatsResponse(BaseModel):
    """System statistics."""
    documents: int
    chunks: int
    images: int
    links: int
    faiss_indexes: Dict[str, int]
    database: Dict[str, Any]


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Query must not be empty",
                "code": "VALIDATION_ERROR"
            }
        }
