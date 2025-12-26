"""
NeuroSynth - API Models Unit Tests
===================================

Tests for Pydantic request/response models.
"""

import pytest
from pydantic import ValidationError


# =============================================================================
# Search Models Tests
# =============================================================================

class TestSearchRequest:
    """Tests for SearchRequest model."""
    
    def test_valid_request(self):
        """Valid search request."""
        from src.api.models import SearchRequest, SearchMode
        
        req = SearchRequest(
            query="retrosigmoid approach",
            mode=SearchMode.HYBRID,
            top_k=10
        )
        
        assert req.query == "retrosigmoid approach"
        assert req.mode == SearchMode.HYBRID
        assert req.top_k == 10
    
    def test_default_values(self):
        """Default values applied."""
        from src.api.models import SearchRequest, SearchMode
        
        req = SearchRequest(query="test query")
        
        assert req.mode == SearchMode.HYBRID
        assert req.top_k == 10
        assert req.include_images is True
        assert req.rerank is True
    
    def test_empty_query_rejected(self):
        """Empty query rejected."""
        from src.api.models import SearchRequest
        
        with pytest.raises(ValidationError):
            SearchRequest(query="")
    
    def test_top_k_bounds(self):
        """top_k must be 1-100."""
        from src.api.models import SearchRequest
        
        # Too low
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)
        
        # Too high
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=101)
        
        # Valid bounds
        req_min = SearchRequest(query="test", top_k=1)
        req_max = SearchRequest(query="test", top_k=100)
        assert req_min.top_k == 1
        assert req_max.top_k == 100
    
    def test_with_filters(self):
        """Request with filters."""
        from src.api.models import SearchRequest, SearchFilters
        
        req = SearchRequest(
            query="test",
            filters=SearchFilters(
                chunk_types=["PROCEDURE", "ANATOMY"],
                specialties=["skull_base"],
                cuis=["C0001418"]
            )
        )
        
        assert req.filters.chunk_types == ["PROCEDURE", "ANATOMY"]
        assert req.filters.cuis == ["C0001418"]


class TestSearchFilters:
    """Tests for SearchFilters model."""
    
    def test_empty_filters(self):
        """Empty filters allowed."""
        from src.api.models import SearchFilters
        
        filters = SearchFilters()
        
        assert filters.document_ids == []
        assert filters.chunk_types == []
        assert filters.cuis == []
    
    def test_page_range(self):
        """Page range filters."""
        from src.api.models import SearchFilters
        
        filters = SearchFilters(min_page=10, max_page=50)
        
        assert filters.min_page == 10
        assert filters.max_page == 50


class TestSearchResponse:
    """Tests for SearchResponse model."""
    
    def test_valid_response(self):
        """Valid response structure."""
        from src.api.models import SearchResponse, SearchResultItem
        
        response = SearchResponse(
            results=[
                SearchResultItem(
                    id="chunk-1",
                    content="Test content",
                    score=0.95
                )
            ],
            total_candidates=100,
            query="test",
            mode="hybrid",
            search_time_ms=25
        )
        
        assert len(response.results) == 1
        assert response.total_candidates == 100
        assert response.search_time_ms == 25


# =============================================================================
# RAG Models Tests
# =============================================================================

class TestRAGRequest:
    """Tests for RAGRequest model."""
    
    def test_valid_request(self):
        """Valid RAG request."""
        from src.api.models import RAGRequest
        
        req = RAGRequest(
            question="What is the retrosigmoid approach?"
        )
        
        assert "retrosigmoid" in req.question
        assert req.include_citations is True
    
    def test_empty_question_rejected(self):
        """Empty question rejected."""
        from src.api.models import RAGRequest
        
        with pytest.raises(ValidationError):
            RAGRequest(question="")
    
    def test_question_length_limit(self):
        """Question has length limit."""
        from src.api.models import RAGRequest
        
        # 2000 char limit
        long_question = "x" * 2001
        with pytest.raises(ValidationError):
            RAGRequest(question=long_question)
    
    def test_with_question_type(self):
        """Request with question type."""
        from src.api.models import RAGRequest, QuestionType
        
        req = RAGRequest(
            question="Describe the procedure steps",
            question_type=QuestionType.PROCEDURAL
        )
        
        assert req.question_type == QuestionType.PROCEDURAL
    
    def test_max_context_chunks(self):
        """Max context chunks bounds."""
        from src.api.models import RAGRequest
        
        # Valid range 1-20
        req = RAGRequest(question="test", max_context_chunks=15)
        assert req.max_context_chunks == 15
        
        # Too high
        with pytest.raises(ValidationError):
            RAGRequest(question="test", max_context_chunks=25)


class TestRAGResponse:
    """Tests for RAGResponse model."""
    
    def test_valid_response(self):
        """Valid RAG response."""
        from src.api.models import RAGResponse, CitationItem, ImageItem
        
        response = RAGResponse(
            answer="The retrosigmoid approach provides [1]...",
            citations=[
                CitationItem(
                    index=1,
                    chunk_id="chunk-1",
                    snippet="Provides exposure..."
                )
            ],
            used_citations=[
                CitationItem(
                    index=1,
                    chunk_id="chunk-1",
                    snippet="Provides exposure..."
                )
            ],
            images=[],
            question="What is the retrosigmoid approach?",
            context_chunks_used=5,
            generation_time_ms=2500,
            search_time_ms=50,
            total_time_ms=2550,
            model="claude-sonnet-4-20250514"
        )
        
        assert "[1]" in response.answer
        assert len(response.citations) == 1
        assert response.generation_time_ms == 2500


# =============================================================================
# Conversation Models Tests
# =============================================================================

class TestConversationRequest:
    """Tests for ConversationRequest model."""
    
    def test_new_conversation(self):
        """New conversation without ID."""
        from src.api.models import ConversationRequest
        
        req = ConversationRequest(message="Hello")
        
        assert req.conversation_id is None
        assert req.message == "Hello"
    
    def test_continue_conversation(self):
        """Continue existing conversation."""
        from src.api.models import ConversationRequest
        
        req = ConversationRequest(
            conversation_id="conv-123",
            message="Tell me more"
        )
        
        assert req.conversation_id == "conv-123"


class TestConversationResponse:
    """Tests for ConversationResponse model."""
    
    def test_valid_response(self):
        """Valid conversation response."""
        from src.api.models import ConversationResponse
        
        response = ConversationResponse(
            conversation_id="conv-123",
            answer="Here is more information...",
            citations=[],
            history_length=3
        )
        
        assert response.conversation_id == "conv-123"
        assert response.history_length == 3


# =============================================================================
# Document Models Tests
# =============================================================================

class TestDocumentModels:
    """Tests for document models."""
    
    def test_document_summary(self):
        """Document summary model."""
        from src.api.models import DocumentSummary
        
        doc = DocumentSummary(
            id="doc-123",
            source_path="/path/to/doc.pdf",
            title="Neurosurgery Atlas",
            total_pages=250,
            total_chunks=500,
            total_images=100
        )
        
        assert doc.total_pages == 250
    
    def test_document_list_response(self):
        """Document list response."""
        from src.api.models import DocumentListResponse, DocumentSummary
        
        response = DocumentListResponse(
            documents=[
                DocumentSummary(
                    id="doc-1",
                    source_path="/doc1.pdf"
                )
            ],
            total=100,
            page=1,
            page_size=20
        )
        
        assert len(response.documents) == 1
        assert response.total == 100


# =============================================================================
# Health Models Tests
# =============================================================================

class TestHealthModels:
    """Tests for health check models."""
    
    def test_component_status(self):
        """Component status model."""
        from src.api.models import ComponentStatus
        
        status = ComponentStatus(
            status="healthy",
            latency_ms=5,
            details={"pool_size": 10}
        )
        
        assert status.status == "healthy"
        assert status.latency_ms == 5
    
    def test_invalid_status_rejected(self):
        """Invalid status value rejected."""
        from src.api.models import ComponentStatus
        
        with pytest.raises(ValidationError):
            ComponentStatus(status="unknown")
    
    def test_health_response(self):
        """Health response model."""
        from src.api.models import HealthResponse, ComponentStatus
        from datetime import datetime
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            components={
                "database": ComponentStatus(status="healthy", latency_ms=5),
                "faiss": ComponentStatus(status="healthy")
            },
            timestamp=datetime.utcnow()
        )
        
        assert response.status == "healthy"
        assert len(response.components) == 2
    
    def test_stats_response(self):
        """Stats response model."""
        from src.api.models import StatsResponse
        
        response = StatsResponse(
            documents=10,
            chunks=500,
            images=100,
            links=200,
            faiss_indexes={"text": 500, "image": 100},
            database={"pool_size": 10}
        )
        
        assert response.documents == 10
        assert response.chunks == 500


# =============================================================================
# Error Models Tests
# =============================================================================

class TestErrorModels:
    """Tests for error models."""
    
    def test_error_response(self):
        """Error response model."""
        from src.api.models import ErrorResponse
        
        error = ErrorResponse(
            error="Not Found",
            detail="Document not found",
            code="NOT_FOUND"
        )
        
        assert error.error == "Not Found"
        assert error.code == "NOT_FOUND"
    
    def test_minimal_error(self):
        """Minimal error with just message."""
        from src.api.models import ErrorResponse
        
        error = ErrorResponse(error="Something went wrong")
        
        assert error.error == "Something went wrong"
        assert error.detail is None
        assert error.code is None


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for enum models."""
    
    def test_search_mode_values(self):
        """SearchMode enum values."""
        from src.api.models import SearchMode
        
        assert SearchMode.TEXT.value == "text"
        assert SearchMode.IMAGE.value == "image"
        assert SearchMode.HYBRID.value == "hybrid"
    
    def test_question_type_values(self):
        """QuestionType enum values."""
        from src.api.models import QuestionType
        
        assert QuestionType.PROCEDURAL.value == "procedural"
        assert QuestionType.ANATOMICAL.value == "anatomical"
        assert QuestionType.CLINICAL.value == "clinical"
