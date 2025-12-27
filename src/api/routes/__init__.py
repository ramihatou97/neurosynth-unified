"""
NeuroSynth Unified - API Routes
================================

Route modules for the FastAPI application.
"""

from src.api.routes.search import router as search_router
from src.api.routes.rag import router as rag_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.api.routes.synthesis import router as synthesis_router
from src.api.routes.ingest import router as ingest_router

__all__ = [
    'search_router',
    'rag_router',
    'documents_router',
    'health_router',
    'synthesis_router',
    'ingest_router'
]
