"""
NeuroSynth Unified - Test Configuration
========================================

Shared pytest fixtures for all tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4
import json

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock Data
# =============================================================================

@pytest.fixture
def sample_chunk():
    """Sample chunk data."""
    return {
        "id": str(uuid4()),
        "document_id": str(uuid4()),
        "content": "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle. Patient positioning is lateral with the head turned.",
        "page_number": 45,
        "chunk_index": 12,
        "chunk_type": "PROCEDURE",
        "specialty": "skull_base",
        "cuis": ["C0001418", "C0007776"],
        "embedding": np.random.randn(1024).astype(np.float32).tolist()
    }


@pytest.fixture
def sample_chunks():
    """Multiple sample chunks."""
    doc_id = str(uuid4())
    return [
        {
            "id": str(uuid4()),
            "document_id": doc_id,
            "content": "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle.",
            "page_number": 45,
            "chunk_type": "PROCEDURE",
            "specialty": "skull_base",
            "cuis": ["C0001418"]
        },
        {
            "id": str(uuid4()),
            "document_id": doc_id,
            "content": "Facial nerve preservation is a key goal in acoustic neuroma surgery.",
            "page_number": 47,
            "chunk_type": "PROCEDURE",
            "specialty": "skull_base",
            "cuis": ["C0015462", "C0027859"]
        },
        {
            "id": str(uuid4()),
            "document_id": doc_id,
            "content": "The tumor is typically found medial to the facial nerve at the internal auditory canal.",
            "page_number": 48,
            "chunk_type": "ANATOMY",
            "specialty": "skull_base",
            "cuis": ["C0027651", "C0015462"]
        }
    ]


@pytest.fixture
def sample_image():
    """Sample image data."""
    return {
        "id": str(uuid4()),
        "document_id": str(uuid4()),
        "file_path": "/images/fig_1.png",
        "page_number": 45,
        "image_type": "surgical_photo",
        "is_decorative": False,
        "vlm_caption": "Surgical corridor showing the cerebellopontine angle approach",
        "cuis": ["C0007776"],
        "embedding": np.random.randn(512).astype(np.float32).tolist(),
        "caption_embedding": np.random.randn(1024).astype(np.float32).tolist()
    }


@pytest.fixture
def sample_document():
    """Sample document data."""
    return {
        "id": str(uuid4()),
        "source_path": "/documents/neurosurgery_atlas.pdf",
        "title": "Neurosurgery Atlas - Skull Base Approaches",
        "total_pages": 250,
        "metadata": {
            "author": "Dr. Smith",
            "publisher": "Medical Press"
        }
    }


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for FAISS testing."""
    np.random.seed(42)
    n_chunks = 100
    n_images = 50
    
    return {
        "chunk_embeddings": np.random.randn(n_chunks, 1024).astype(np.float32),
        "chunk_ids": [f"chunk_{i}" for i in range(n_chunks)],
        "image_embeddings": np.random.randn(n_images, 512).astype(np.float32),
        "caption_embeddings": np.random.randn(n_images, 1024).astype(np.float32),
        "image_ids": [f"image_{i}" for i in range(n_images)]
    }


# =============================================================================
# Mock Services
# =============================================================================

class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self._call_count = 0
    
    async def embed(self, text: str) -> list:
        self._call_count += 1
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.dimension).astype(np.float32).tolist()
    
    async def embed_batch(self, texts: list) -> list:
        return [await self.embed(t) for t in texts]


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self._data = {
            "documents": {},
            "chunks": {},
            "images": {},
            "links": {}
        }
    
    async def fetch(self, query: str, *args):
        """Mock fetch - returns empty list."""
        return []
    
    async def fetchrow(self, query: str, *args):
        """Mock fetchrow - returns None."""
        return None
    
    async def execute(self, query: str, *args):
        """Mock execute."""
        pass
    
    async def health_check(self) -> bool:
        return True
    
    async def get_stats(self) -> dict:
        return {"pool_size": 5, "active": 1}


class MockSearchResult:
    """Mock search result."""
    
    def __init__(self, id: str, content: str, score: float, **kwargs):
        self.id = id
        self.content = content
        self.score = score
        self.result_type = kwargs.get("result_type", "chunk")
        self.document_id = kwargs.get("document_id")
        self.page_number = kwargs.get("page_number")
        self.chunk_type = kwargs.get("chunk_type")
        self.specialty = kwargs.get("specialty")
        self.cuis = kwargs.get("cuis", [])
        self.linked_images = kwargs.get("linked_images", [])


class MockSearchService:
    """Mock search service for testing."""
    
    def __init__(self, results: list = None):
        self._results = results or []
    
    async def search(self, query: str, **kwargs):
        """Mock search."""
        from dataclasses import dataclass
        
        @dataclass
        class MockResponse:
            results: list
            total_candidates: int
            query: str
            mode: str
            search_time_ms: int
        
        return MockResponse(
            results=self._results,
            total_candidates=len(self._results),
            query=query,
            mode=kwargs.get("mode", "hybrid"),
            search_time_ms=10
        )
    
    async def find_similar(self, item_id: str, **kwargs):
        return self._results


@pytest.fixture
def mock_embedder():
    """Get mock embedder."""
    return MockEmbedder()


@pytest.fixture
def mock_database():
    """Get mock database."""
    return MockDatabase()


@pytest.fixture
def mock_search_results(sample_chunks):
    """Get mock search results from sample chunks."""
    return [
        MockSearchResult(
            id=c["id"],
            content=c["content"],
            score=0.9 - i * 0.1,
            result_type="chunk",
            document_id=c["document_id"],
            page_number=c.get("page_number"),
            chunk_type=c.get("chunk_type"),
            specialty=c.get("specialty"),
            cuis=c.get("cuis", [])
        )
        for i, c in enumerate(sample_chunks)
    ]


@pytest.fixture
def mock_search_service(mock_search_results):
    """Get mock search service."""
    return MockSearchService(mock_search_results)


# =============================================================================
# API Testing
# =============================================================================

@pytest.fixture
def test_client():
    """Get FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Get async test client."""
    from httpx import AsyncClient, ASGITransport
    from src.api.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def temp_index_dir(tmp_path):
    """Temporary directory for FAISS indexes."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Environment
# =============================================================================

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


# =============================================================================
# Helpers
# =============================================================================

def assert_valid_uuid(value: str):
    """Assert value is valid UUID string."""
    from uuid import UUID
    UUID(value)  # Raises if invalid


def assert_response_ok(response, expected_status: int = 200):
    """Assert response has expected status."""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"


def make_search_request(
    query: str = "test query",
    mode: str = "hybrid",
    top_k: int = 10,
    **kwargs
) -> dict:
    """Create search request payload."""
    return {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        **kwargs
    }


def make_rag_request(
    question: str = "What is the retrosigmoid approach?",
    **kwargs
) -> dict:
    """Create RAG request payload."""
    return {
        "question": question,
        "include_citations": True,
        "include_images": True,
        **kwargs
    }
