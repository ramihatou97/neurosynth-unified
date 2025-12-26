"""
NeuroSynth - API Integration Tests
===================================

Integration tests for FastAPI endpoints.
Uses TestClient for synchronous testing.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Get test client with mocked services."""
    from src.api.main import app
    from src.api.dependencies import ServiceContainer
    
    # Reset singleton
    ServiceContainer._instance = None
    
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def mock_container():
    """Create mock service container."""
    from src.api.dependencies import ServiceContainer
    
    container = MagicMock(spec=ServiceContainer)
    container._initialized = True
    container.is_healthy = True
    
    # Mock database
    container.database = AsyncMock()
    container.database.health_check = AsyncMock(return_value=True)
    container.database.get_stats = AsyncMock(return_value={"pool_size": 5})
    container.database.fetch = AsyncMock(return_value=[])
    
    # Mock repositories
    container.repositories = MagicMock()
    container.repositories.documents = MagicMock()
    container.repositories.documents.count = AsyncMock(return_value=10)
    container.repositories.chunks = MagicMock()
    container.repositories.chunks.get_statistics = AsyncMock(return_value={"total": 100})
    container.repositories.images = MagicMock()
    container.repositories.images.get_statistics = AsyncMock(return_value={"total": 50})
    container.repositories.links = MagicMock()
    container.repositories.links.get_statistics = AsyncMock(return_value={"total": 30})
    
    # Mock FAISS
    container.faiss = MagicMock()
    container.faiss.get_stats = MagicMock(return_value={
        "text": {"size": 100, "dimension": 1024},
        "image": {"size": 50, "dimension": 512},
        "caption": {"size": 50, "dimension": 1024}
    })
    
    # Mock search
    container.search = AsyncMock()
    
    # Mock RAG
    container.rag = AsyncMock()
    
    return container


# =============================================================================
# Root & Info Tests
# =============================================================================

class TestRootEndpoints:
    """Tests for root and info endpoints."""
    
    def test_root(self, client):
        """Root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["docs"] == "/docs"
    
    def test_info(self, client, mock_container):
        """Info endpoint returns configuration."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "models" in data


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client, mock_container):
        """Health check returns component status."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_liveness_probe(self, client):
        """Liveness probe returns alive."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_readiness_probe_healthy(self, client, mock_container):
        """Readiness probe when healthy."""
        mock_container.is_healthy = True
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/health/ready")
        
        assert response.status_code == 200
    
    def test_stats_endpoint(self, client, mock_container):
        """Stats endpoint returns statistics."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "chunks" in data
        assert "faiss_indexes" in data


# =============================================================================
# Search Endpoint Tests
# =============================================================================

class TestSearchEndpoints:
    """Tests for search endpoints."""
    
    def test_search_basic(self, client, mock_container):
        """Basic search request."""
        # Mock search response
        mock_response = MagicMock()
        mock_response.results = []
        mock_response.total_candidates = 0
        mock_response.query = "test query"
        mock_response.mode = "hybrid"
        mock_response.search_time_ms = 10
        
        mock_container.search.search = AsyncMock(return_value=mock_response)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/search",
                json={"query": "retrosigmoid approach"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_candidates" in data
        assert "search_time_ms" in data
    
    def test_search_with_filters(self, client, mock_container):
        """Search with filters."""
        mock_response = MagicMock()
        mock_response.results = []
        mock_response.total_candidates = 0
        mock_response.query = "test"
        mock_response.mode = "hybrid"
        mock_response.search_time_ms = 15
        
        mock_container.search.search = AsyncMock(return_value=mock_response)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/search",
                json={
                    "query": "test query",
                    "filters": {
                        "chunk_types": ["PROCEDURE"],
                        "specialties": ["skull_base"]
                    }
                }
            )
        
        assert response.status_code == 200
    
    def test_search_validation_error(self, client, mock_container):
        """Search with invalid request."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/search",
                json={"query": ""}  # Empty query
            )
        
        assert response.status_code == 422
    
    def test_quick_search(self, client, mock_container):
        """Quick search endpoint."""
        mock_response = MagicMock()
        mock_response.results = []
        mock_response.total_candidates = 0
        mock_response.query = "test"
        mock_response.mode = "text"
        mock_response.search_time_ms = 5
        
        mock_container.search.search = AsyncMock(return_value=mock_response)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/api/v1/search/quick?q=retrosigmoid&k=5")
        
        assert response.status_code == 200


# =============================================================================
# RAG Endpoint Tests
# =============================================================================

class TestRAGEndpoints:
    """Tests for RAG endpoints."""
    
    def test_rag_ask(self, client, mock_container):
        """RAG ask endpoint."""
        # Mock RAG response
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "The retrosigmoid approach [1]..."
        mock_rag_response.citations = []
        mock_rag_response.used_citations = []
        mock_rag_response.images = []
        mock_rag_response.question = "What is the retrosigmoid approach?"
        mock_rag_response.context_chunks_used = 5
        mock_rag_response.generation_time_ms = 2000
        mock_rag_response.search_time_ms = 50
        mock_rag_response.model = "claude-sonnet-4-20250514"
        
        mock_container.rag.ask = AsyncMock(return_value=mock_rag_response)
        mock_container.rag.assembler = MagicMock()
        mock_container.rag.system_prompt = ""
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/rag/ask",
                json={"question": "What is the retrosigmoid approach?"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "model" in data
    
    def test_rag_ask_validation_error(self, client, mock_container):
        """RAG ask with empty question."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/rag/ask",
                json={"question": ""}
            )
        
        assert response.status_code == 422
    
    def test_conversation_start(self, client, mock_container):
        """Start new conversation."""
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "Here is information..."
        mock_rag_response.used_citations = []
        
        # Mock conversation
        mock_conv = MagicMock()
        mock_conv.ask = AsyncMock(return_value=mock_rag_response)
        mock_conv.history = []
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            with patch("src.api.routes.rag.RAGConversation", return_value=mock_conv):
                response = client.post(
                    "/api/v1/rag/conversation",
                    json={"message": "What approaches exist?"}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "answer" in data
    
    def test_conversation_not_found(self, client, mock_container):
        """Continue non-existent conversation."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/rag/conversation",
                json={
                    "conversation_id": "non-existent",
                    "message": "Continue please"
                }
            )
        
        assert response.status_code == 404


# =============================================================================
# Document Endpoint Tests
# =============================================================================

class TestDocumentEndpoints:
    """Tests for document endpoints."""
    
    def test_list_documents(self, client, mock_container):
        """List documents."""
        mock_container.repositories.documents.list_with_counts = AsyncMock(return_value=[
            {
                "id": "doc-1",
                "source_path": "/doc1.pdf",
                "title": "Test Doc",
                "total_pages": 100
            }
        ])
        mock_container.repositories.documents.count = AsyncMock(return_value=1)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
    
    def test_list_documents_pagination(self, client, mock_container):
        """List documents with pagination."""
        mock_container.repositories.documents.list_with_counts = AsyncMock(return_value=[])
        mock_container.repositories.documents.count = AsyncMock(return_value=100)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/api/v1/documents?page=2&page_size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10
    
    def test_get_document_not_found(self, client, mock_container):
        """Get non-existent document."""
        mock_container.repositories.documents.get_with_stats = AsyncMock(return_value=None)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000")
        
        assert response.status_code == 404
    
    def test_get_document_invalid_id(self, client, mock_container):
        """Get document with invalid UUID."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.get("/api/v1/documents/invalid-uuid")
        
        assert response.status_code == 400
    
    def test_delete_document(self, client, mock_container):
        """Delete document."""
        mock_container.repositories.documents.exists = AsyncMock(return_value=True)
        mock_container.repositories.documents.delete_with_cascade = AsyncMock(return_value=True)
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.delete("/api/v1/documents/00000000-0000-0000-0000-000000000000")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_validation_error_format(self, client, mock_container):
        """Validation errors have proper format."""
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/search",
                json={"query": "", "top_k": 500}  # Multiple errors
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "detail" in data
        assert data["code"] == "VALIDATION_ERROR"
    
    def test_service_unavailable(self, client, mock_container):
        """Service unavailable when search not ready."""
        mock_container.search = None
        
        with patch("src.api.dependencies.ServiceContainer.get_instance", return_value=mock_container):
            response = client.post(
                "/api/v1/search",
                json={"query": "test"}
            )
        
        assert response.status_code == 503
