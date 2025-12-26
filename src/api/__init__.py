"""
NeuroSynth Unified - API Package
=================================

FastAPI REST API for NeuroSynth.

Components:
- main.py: FastAPI application
- models.py: Pydantic request/response models
- dependencies.py: Dependency injection
- routes/: API route handlers

Quick Start:
    # Run the API server
    uvicorn src.api.main:app --reload
    
    # Or with Python
    python -m src.api.main

Endpoints:
    GET  /health              - Health check
    GET  /stats               - System statistics
    
    POST /api/v1/search       - Semantic search
    GET  /api/v1/search/quick - Quick search
    
    POST /api/v1/rag/ask      - RAG question answering
    POST /api/v1/rag/conversation - Multi-turn conversation
    
    GET  /api/v1/documents    - List documents
    GET  /api/v1/documents/{id} - Get document details

Environment Variables:
    DATABASE_URL          - PostgreSQL connection string
    FAISS_INDEX_DIR       - Path to FAISS indexes
    VOYAGE_API_KEY        - Voyage AI API key
    ANTHROPIC_API_KEY     - Anthropic API key
    CLAUDE_MODEL          - Claude model to use (default: claude-sonnet-4-20250514)
    DEBUG                 - Enable debug mode (default: false)
"""

from src.api.main import app, create_app
from src.api.dependencies import (
    Settings,
    get_settings,
    ServiceContainer,
    get_container,
    get_database,
    get_repositories,
    get_search_service,
    get_rag_engine,
    get_faiss_manager
)
from src.api.models import (
    # Search
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchFilters,
    SearchMode,
    
    # RAG
    RAGRequest,
    RAGResponse,
    CitationItem,
    ImageItem,
    QuestionType,
    
    # Conversation
    ConversationRequest,
    ConversationResponse,
    
    # Documents
    DocumentSummary,
    DocumentDetail,
    DocumentListResponse,
    
    # Health
    HealthResponse,
    ComponentStatus,
    StatsResponse,
    
    # Error
    ErrorResponse
)

__all__ = [
    # Application
    'app',
    'create_app',
    
    # Settings
    'Settings',
    'get_settings',
    
    # Dependencies
    'ServiceContainer',
    'get_container',
    'get_database',
    'get_repositories',
    'get_search_service',
    'get_rag_engine',
    'get_faiss_manager',
    
    # Search Models
    'SearchRequest',
    'SearchResponse',
    'SearchResultItem',
    'SearchFilters',
    'SearchMode',
    
    # RAG Models
    'RAGRequest',
    'RAGResponse',
    'CitationItem',
    'ImageItem',
    'QuestionType',
    
    # Conversation Models
    'ConversationRequest',
    'ConversationResponse',
    
    # Document Models
    'DocumentSummary',
    'DocumentDetail',
    'DocumentListResponse',
    
    # Health Models
    'HealthResponse',
    'ComponentStatus',
    'StatsResponse',
    
    # Error
    'ErrorResponse'
]
