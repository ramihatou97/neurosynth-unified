"""
NeuroSynth Unified - API Routes
================================

Route modules for the FastAPI application.

NOTE: V3 Synthesis features (web research enrichment) have been consolidated
into the main synthesis router. Use include_web_research=True to enable.
"""

from src.api.routes.search import router as search_router
from src.api.routes.rag import router as rag_router
from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.api.routes.synthesis import router as synthesis_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.entities import router as entities_router
from src.api.routes.indexes import router as indexes_router

# V2 Routes - Unified RAG with external search integration (consolidated from V1+V2+V3)
from src.api.routes.rag_unified import router as rag_unified_router

# NPRSS Learning Routes
from src.api.routes.learning import router as learning_router
from src.api.routes.learning_extended import router as learning_extended_router

__all__ = [
    'search_router',
    'rag_router',
    'documents_router',
    'health_router',
    'synthesis_router',
    'ingest_router',
    'entities_router',
    'indexes_router',
    # V2 Routes - Unified RAG (consolidated from V1+V2+V3)
    'rag_unified_router',
    # NPRSS Learning Routes
    'learning_router',
    'learning_extended_router',
]
