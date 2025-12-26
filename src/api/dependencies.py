"""
NeuroSynth Unified - API Dependencies
======================================

Dependency injection for FastAPI routes.
Manages service lifecycles and provides clean access to components.
"""

import logging
from typing import Optional, AsyncGenerator
from functools import lru_cache
from contextlib import asynccontextmanager
import os

from fastapi import Depends, HTTPException, status

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings from environment."""
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth"
    )
    db_min_connections: int = int(os.getenv("DB_MIN_CONNECTIONS", "2"))
    db_max_connections: int = int(os.getenv("DB_MAX_CONNECTIONS", "10"))
    
    # FAISS
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "./indexes")
    
    # Embeddings
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    voyage_model: str = os.getenv("VOYAGE_MODEL", "voyage-3")
    
    # Claude
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    
    # API
    api_version: str = "1.0.0"
    api_title: str = "NeuroSynth API"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Rate limiting
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    @property
    def cors_origins(self) -> list[str]:
        """
        Get CORS origins from environment variable.

        Returns:
            List of allowed origins. Defaults to ["*"] for development.
        """
        origins_str = os.getenv("CORS_ORIGINS", "*")
        return [origin.strip() for origin in origins_str.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()


# =============================================================================
# Service Container
# =============================================================================

class ServiceContainer:
    """
    Container for all services.
    
    Manages initialization and lifecycle of:
    - Database connection
    - FAISS indexes
    - Search service
    - RAG engine
    """
    
    _instance: Optional['ServiceContainer'] = None
    
    def __init__(self):
        self._database = None
        self._repositories = None
        self._faiss = None
        self._embedder = None
        self._search = None
        self._rag = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def initialize(self, settings: Settings = None) -> None:
        """Initialize all services."""
        if self._initialized:
            return
        
        settings = settings or get_settings()
        logger.info("Initializing services...")
        
        # 1. Database
        try:
            from src.database import init_database, get_repositories
            
            self._database = await init_database(
                settings.database_url,
                min_connections=settings.db_min_connections,
                max_connections=settings.db_max_connections
            )
            self._repositories = get_repositories(self._database)
            logger.info("✓ Database connected")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        
        # 2. FAISS indexes
        try:
            from src.retrieval import FAISSManager
            
            self._faiss = FAISSManager(settings.faiss_index_dir)
            stats = self._faiss.load()
            logger.info(f"✓ FAISS indexes loaded: {stats}")
        except FileNotFoundError:
            logger.warning("FAISS indexes not found - search will use pgvector only")
            self._faiss = None
        except Exception as e:
            logger.error(f"FAISS initialization failed: {e}")
            self._faiss = None
        
        # 3. Embedder
        try:
            from src.retrieval import VoyageEmbedder
            
            if settings.voyage_api_key:
                self._embedder = VoyageEmbedder(
                    api_key=settings.voyage_api_key,
                    model=settings.voyage_model
                )
                logger.info("✓ Voyage embedder initialized")
            else:
                logger.warning("VOYAGE_API_KEY not set - embedder disabled")
        except Exception as e:
            logger.error(f"Embedder initialization failed: {e}")
        
        # 4. Search service
        try:
            from src.retrieval import SearchService
            
            if self._embedder:
                self._search = SearchService(
                    database=self._database,
                    faiss_manager=self._faiss,
                    embedder=self._embedder
                )
                logger.info("✓ Search service initialized")
        except Exception as e:
            logger.error(f"Search service initialization failed: {e}")
        
        # 5. RAG engine
        try:
            from src.rag import RAGEngine, RAGConfig
            
            if self._search and settings.anthropic_api_key:
                config = RAGConfig(model=settings.claude_model)
                self._rag = RAGEngine(
                    search_service=self._search,
                    api_key=settings.anthropic_api_key,
                    config=config
                )
                logger.info("✓ RAG engine initialized")
            else:
                logger.warning("RAG engine not initialized - missing search or API key")
        except Exception as e:
            logger.error(f"RAG engine initialization failed: {e}")
        
        self._initialized = True
        logger.info("All services initialized")
    
    async def shutdown(self) -> None:
        """Shutdown all services."""
        logger.info("Shutting down services...")
        
        if self._database:
            from src.database import close_database
            await close_database()
        
        self._initialized = False
        logger.info("Services shut down")
    
    # Properties for service access
    @property
    def database(self):
        return self._database
    
    @property
    def repositories(self):
        return self._repositories
    
    @property
    def faiss(self):
        return self._faiss
    
    @property
    def embedder(self):
        return self._embedder
    
    @property
    def search(self):
        return self._search
    
    @property
    def rag(self):
        return self._rag
    
    @property
    def is_healthy(self) -> bool:
        return self._initialized and self._database is not None


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_container() -> ServiceContainer:
    """Get service container."""
    container = ServiceContainer.get_instance()
    if not container._initialized:
        await container.initialize()
    return container


async def get_database(
    container: ServiceContainer = Depends(get_container)
):
    """Get database connection."""
    if not container.database:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    return container.database


async def get_repositories(
    container: ServiceContainer = Depends(get_container)
):
    """Get repository container."""
    if not container.repositories:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    return container.repositories


async def get_search_service(
    container: ServiceContainer = Depends(get_container)
):
    """Get search service."""
    if not container.search:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not available"
        )
    return container.search


async def get_rag_engine(
    container: ServiceContainer = Depends(get_container)
):
    """Get RAG engine."""
    if not container.rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not available"
        )
    return container.rag


async def get_faiss_manager(
    container: ServiceContainer = Depends(get_container)
):
    """Get FAISS manager (optional)."""
    return container.faiss


# =============================================================================
# Conversation State (In-Memory)
# =============================================================================

class ConversationStore:
    """Simple in-memory conversation store."""
    
    _instance: Optional['ConversationStore'] = None
    
    def __init__(self, max_conversations: int = 1000):
        self._conversations: dict = {}
        self._max = max_conversations
    
    @classmethod
    def get_instance(cls) -> 'ConversationStore':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, conversation_id: str):
        return self._conversations.get(conversation_id)
    
    def set(self, conversation_id: str, conversation):
        # Simple LRU: remove oldest if at capacity
        if len(self._conversations) >= self._max:
            oldest = next(iter(self._conversations))
            del self._conversations[oldest]
        self._conversations[conversation_id] = conversation
    
    def delete(self, conversation_id: str):
        self._conversations.pop(conversation_id, None)


def get_conversation_store() -> ConversationStore:
    """Get conversation store."""
    return ConversationStore.get_instance()
