"""
NeuroSynth Unified - API Dependencies
======================================

Dependency injection for FastAPI routes.
Manages service lifecycles and provides clean access to components.
"""

import asyncio
import logging
import sys
from typing import Optional, AsyncGenerator, List, Dict, Any
from functools import lru_cache
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import os

from fastapi import Depends, HTTPException, status

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Validation
# =============================================================================

class EnvVarStatus(Enum):
    """Status of an environment variable."""
    PRESENT = "present"
    MISSING = "missing"
    INVALID = "invalid"
    DEFAULT = "using_default"


@dataclass
class EnvVarConfig:
    """Configuration for an environment variable."""
    name: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    validator: Optional[callable] = None
    sensitive: bool = False  # Don't log value if True


@dataclass
class ValidationResult:
    """Result of environment validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


# Validators
def _validate_url(value: str) -> bool:
    """Validate URL format."""
    return value.startswith(("http://", "https://", "postgresql://", "postgres://"))


def _validate_api_key(value: str) -> bool:
    """Validate API key format (non-empty, reasonable length)."""
    return len(value) >= 10 and len(value) <= 500  # Extended for longer session keys


def _validate_positive_int(value: str) -> bool:
    """Validate positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False


# Required environment variables
REQUIRED_ENV_VARS = [
    EnvVarConfig(
        name="DATABASE_URL",
        required=True,
        description="PostgreSQL connection URL",
        validator=_validate_url,
        default="postgresql://neurosynth:neurosynth@localhost:5432/neurosynth"
    ),
    EnvVarConfig(
        name="VOYAGE_API_KEY",
        required=True,
        description="Voyage AI API key for text embeddings",
        validator=_validate_api_key,
        sensitive=True
    ),
    EnvVarConfig(
        name="ANTHROPIC_API_KEY",
        required=True,
        description="Anthropic API key for Claude",
        validator=_validate_api_key,
        sensitive=True
    ),
    EnvVarConfig(
        name="VOYAGE_MODEL",
        required=False,
        default="voyage-3",
        description="Voyage embedding model name"
    ),
    EnvVarConfig(
        name="CLAUDE_MODEL",
        required=False,
        default="claude-sonnet-4-20250514",
        description="Claude model for RAG"
    ),
    EnvVarConfig(
        name="FAISS_INDEX_DIR",
        required=False,
        default="./indexes",
        description="Directory for FAISS indexes"
    ),
]


def validate_environment(
    env_vars: List[EnvVarConfig] = None,
    exit_on_error: bool = True
) -> ValidationResult:
    """
    Validate all required environment variables.

    Args:
        env_vars: List of EnvVarConfig to validate (default: REQUIRED_ENV_VARS)
        exit_on_error: If True, exit process on validation failure

    Returns:
        ValidationResult with errors, warnings, and loaded config
    """
    if env_vars is None:
        env_vars = REQUIRED_ENV_VARS

    result = ValidationResult(valid=True)

    print("=" * 60)
    print("Environment Variable Validation")
    print("=" * 60)

    for var in env_vars:
        value = os.getenv(var.name)
        status = EnvVarStatus.PRESENT

        # Check if present
        if value is None or value.strip() == "":
            if var.required and var.default is None:
                status = EnvVarStatus.MISSING
                result.valid = False
                result.errors.append(f"Missing required: {var.name} - {var.description}")
            elif var.default is not None:
                status = EnvVarStatus.DEFAULT
                value = var.default
                result.warnings.append(f"Using default for {var.name}: {var.default}")
            else:
                status = EnvVarStatus.MISSING
                result.warnings.append(f"Optional missing: {var.name}")

        # Validate format if present and validator exists
        elif var.validator and not var.validator(value):
            status = EnvVarStatus.INVALID
            result.valid = False
            result.errors.append(f"Invalid format: {var.name} - {var.description}")

        # Store in config
        result.config[var.name] = value

        # Log status
        if var.sensitive and value:
            display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
        else:
            display_value = value or "(not set)"

        status_icon = {
            EnvVarStatus.PRESENT: "[OK]",
            EnvVarStatus.MISSING: "[MISSING]",
            EnvVarStatus.INVALID: "[INVALID]",
            EnvVarStatus.DEFAULT: "[DEFAULT]"
        }[status]

        print(f"  {status_icon} {var.name}: {display_value}")

    print("=" * 60)

    # Print summary
    if result.errors:
        print("\n[ERRORS]:")
        for error in result.errors:
            print(f"   - {error}")

    if result.warnings:
        print("\n[WARNINGS]:")
        for warning in result.warnings:
            print(f"   - {warning}")

    if result.valid:
        print("\n[OK] Environment validation passed")
    else:
        print("\n[FAIL] Environment validation FAILED")
        print("\nTo fix, create a .env file with:")
        print("-" * 40)
        for var in env_vars:
            if var.required:
                example = "your-api-key-here" if var.sensitive else var.default or "value"
                print(f"{var.name}={example}")
        print("-" * 40)

        if exit_on_error:
            print("\nExiting due to missing configuration.")
            sys.exit(1)

    return result


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings from environment."""

    # Database (asyncpg requires plain postgresql:// scheme)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth"
    )
    db_min_connections: int = int(os.getenv("DB_MIN_CONNECTIONS", "2"))
    db_max_connections: int = int(os.getenv("DB_MAX_CONNECTIONS", "10"))

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Search Backend Feature Flags
    # ─────────────────────────────────────────────────────────────────────────
    # The system supports BOTH pgvector (PostgreSQL) and FAISS backends.
    # Choose based on your library size:
    #
    # SCALING THRESHOLDS:
    #   < 50K vectors   → pgvector HNSW (current default)
    #   50K - 500K      → Either works, benchmark both
    #   > 500K vectors  → FAISS IVFFlat recommended
    #
    # TO ENABLE FAISS AT PRODUCTION SCALE:
    #   export USE_FAISS=true
    #   export USE_PGVECTOR=false
    #   python scripts/build_indexes.py --faiss
    #
    # Current sample: ~1.3K vectors → pgvector optimal
    # Expected full scale: ~1M vectors → switch to FAISS at ~500K
    # ─────────────────────────────────────────────────────────────────────────
    use_faiss: bool = os.getenv("USE_FAISS", "false").lower() == "true"
    use_pgvector: bool = os.getenv("USE_PGVECTOR", "true").lower() == "true"
    enable_embedding_cache: bool = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    enable_conflict_detection: bool = os.getenv("ENABLE_CONFLICT_DETECTION", "true").lower() == "true"

    # FAISS (disabled by default, retained for rollback)
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
            List of allowed origins. Defaults to localhost in debug mode,
            empty list in production (blocking cross-origin requests).
        """
        origins_str = os.getenv("CORS_ORIGINS", "")

        if not origins_str:
            if self.debug:
                return ["http://localhost:3000", "http://127.0.0.1:3000"]
            else:
                logger.warning("CORS_ORIGINS not set - blocking cross-origin requests")
                return []

        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


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
    _init_lock: Optional['asyncio.Lock'] = None

    def __init__(self):
        self._database = None
        self._repositories = None
        self._faiss = None
        self._embedder = None
        self._search = None
        self._rag = None
        self._initialized = False
        self._initializing = False
    
    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._init_lock = asyncio.Lock()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for clean restart (hot reload support)."""
        cls._instance = None
        cls._init_lock = None
    
    async def initialize(self, settings: Settings = None) -> None:
        """Initialize all services with lock to prevent race conditions."""
        # Fast path: already initialized
        if self._initialized:
            return

        # Use lock to prevent concurrent initialization
        lock = ServiceContainer._init_lock
        if lock is None:
            lock = asyncio.Lock()
            ServiceContainer._init_lock = lock

        async with lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            if self._initializing:
                # Another task is initializing, wait for it
                while self._initializing and not self._initialized:
                    await asyncio.sleep(0.1)
                return

            self._initializing = True
            settings = settings or get_settings()
            logger.info("Initializing services...")

            try:
                # 1. Database (REQUIRED - fail if this fails)
                from src.database import init_database, get_repositories

                self._database = await init_database(
                    settings.database_url,
                    min_connections=settings.db_min_connections,
                    max_connections=settings.db_max_connections
                )
                self._repositories = get_repositories(self._database)
                logger.info("✓ Database connected")

                # 2. FAISS indexes (optional, disabled by default - using pgvector HNSW)
                if settings.use_faiss:
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
                else:
                    logger.info("⊘ FAISS disabled (USE_FAISS=false) - using pgvector HNSW indexes")
                    self._faiss = None

                # 3. Embedder (optional but needed for search)
                try:
                    from src.retrieval import VoyageEmbedder

                    if settings.voyage_api_key:
                        self._embedder = VoyageEmbedder(
                            api_key=settings.voyage_api_key,
                            model=settings.voyage_model
                        )
                        logger.info("✓ Voyage embedder initialized")
                    else:
                        logger.warning("⚠ VOYAGE_API_KEY not set - embedder disabled")
                        self._embedder = None
                except Exception as e:
                    logger.error(f"✗ Embedder initialization failed: {e}")
                    self._embedder = None

                # 4. Search service (optional)
                try:
                    from src.retrieval import SearchService

                    if self._embedder:
                        self._search = SearchService(
                            database=self._database,
                            faiss_manager=self._faiss,
                            embedder=self._embedder,
                            use_pgvector=settings.use_pgvector
                        )
                        search_mode = "pgvector HNSW" if settings.use_pgvector else "FAISS"
                        logger.info(f"✓ Search service initialized ({search_mode})")
                    else:
                        logger.warning("⚠ Search service not initialized - embedder required")
                        self._search = None
                except Exception as e:
                    logger.error(f"✗ Search service initialization failed: {e}")
                    self._search = None

                # 5. RAG engine (optional)
                try:
                    from src.rag import RAGEngine, RAGConfig

                    if self._search and settings.anthropic_api_key:
                        config = RAGConfig(
                            model=settings.claude_model,
                            use_graph_rag=False,  # Disabled: graph relations are imprecise
                        )
                        self._rag = RAGEngine(
                            search_service=self._search,
                            api_key=settings.anthropic_api_key,
                            config=config
                        )
                        logger.info("✓ RAG engine initialized")
                    else:
                        missing = []
                        if not self._search:
                            missing.append("search service")
                        if not settings.anthropic_api_key:
                            missing.append("ANTHROPIC_API_KEY")
                        logger.warning(f"⚠ RAG engine not initialized - missing: {', '.join(missing)}")
                        self._rag = None
                except Exception as e:
                    logger.error(f"✗ RAG engine initialization failed: {e}")
                    self._rag = None

                self._initialized = True
                logger.info("All services initialized")

            except Exception as e:
                logger.error(f"Service initialization failed: {e}")
                self._initializing = False
                raise

            finally:
                self._initializing = False
    
    async def shutdown(self) -> None:
        """Shutdown all services and reset singleton for clean restart."""
        logger.info("Shutting down services...")

        if self._database:
            from src.database import close_database
            await close_database()

        # Reset all state
        self._database = None
        self._repositories = None
        self._faiss = None
        self._embedder = None
        self._search = None
        self._rag = None
        self._initialized = False
        self._initializing = False

        # Reset class-level singleton for hot reload support
        ServiceContainer.reset_instance()
        logger.info("Services shut down and singleton reset")
    
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
    """Thread-safe in-memory conversation store with LRU eviction."""

    _instance: Optional['ConversationStore'] = None
    _instance_lock: Optional['asyncio.Lock'] = None

    def __init__(self, max_conversations: int = 1000):
        self._conversations: dict = {}
        self._max = max_conversations
        self._lock = asyncio.Lock()  # Thread safety for dict operations

    @classmethod
    def get_instance(cls) -> 'ConversationStore':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, conversation_id: str):
        """Get conversation (read-only, no lock needed for dict.get)."""
        return self._conversations.get(conversation_id)

    async def set_async(self, conversation_id: str, conversation):
        """Thread-safe set with LRU eviction (async version)."""
        async with self._lock:
            if len(self._conversations) >= self._max:
                oldest = next(iter(self._conversations))
                del self._conversations[oldest]
            self._conversations[conversation_id] = conversation

    def set(self, conversation_id: str, conversation):
        """Sync set (for backward compatibility - use set_async when possible)."""
        # Simple LRU: remove oldest if at capacity
        if len(self._conversations) >= self._max:
            oldest = next(iter(self._conversations))
            del self._conversations[oldest]
        self._conversations[conversation_id] = conversation

    async def delete_async(self, conversation_id: str):
        """Thread-safe delete (async version)."""
        async with self._lock:
            self._conversations.pop(conversation_id, None)

    def delete(self, conversation_id: str):
        """Sync delete (for backward compatibility)."""
        self._conversations.pop(conversation_id, None)

    def list_all(self) -> dict:
        """List all conversations as dict {conversation_id: conversation_data}."""
        return dict(self._conversations)

    def clear_all(self):
        """Clear all conversations."""
        self._conversations.clear()


def get_conversation_store() -> ConversationStore:
    """Get conversation store."""
    return ConversationStore.get_instance()
