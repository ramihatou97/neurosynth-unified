"""
NeuroSynth Unified - Authority Registry API
=============================================

API endpoints for managing the authority registry.

The authority registry controls how different neurosurgical sources
are weighted during synthesis. Higher-authority sources (e.g., Rhoton,
Lawton) are prioritized over general references.

Endpoints:
- GET  /api/v1/registry           - Get all sources with scores
- PUT  /api/v1/registry/score     - Update score for existing source
- POST /api/v1/registry/custom    - Add custom authority source
- DELETE /api/v1/registry/custom/{name} - Remove custom source
- POST /api/v1/registry/reset     - Reset to defaults

Database:
- Requires table 'authority_registry' (run migration 004_authority_registry.sql)
- Falls back to in-memory if table doesn't exist
"""

import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies import get_database
from src.synthesis.engine import (
    get_authority_registry,
    set_authority_registry,
    AuthorityRegistry,
    AuthoritySource
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/registry", tags=["Authority Registry"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AuthoritySourceResponse(BaseModel):
    """Single authority source with metadata."""
    name: str = Field(..., description="Source identifier (e.g., RHOTON, LAWTON)")
    score: float = Field(..., ge=0.0, le=1.0, description="Authority score (0.0-1.0)")
    keywords: List[str] = Field(default_factory=list, description="Detection keywords")
    is_custom: bool = Field(..., description="True if user-defined source")
    tier: Optional[int] = Field(None, description="Authority tier (1=Master, 2=Textbook, 3=Reference)")


class RegistryResponse(BaseModel):
    """Complete registry state."""
    sources: List[AuthoritySourceResponse]
    total: int
    persisted: bool = Field(..., description="True if loaded from database")


class UpdateScoreRequest(BaseModel):
    """Request to update a source's score."""
    source: str = Field(..., description="Source name (e.g., RHOTON)")
    score: float = Field(..., ge=0.0, le=1.0, description="New score")

    @field_validator('source')
    @classmethod
    def validate_source_name(cls, v: str) -> str:
        return v.upper().strip()


class AddCustomRequest(BaseModel):
    """Request to add a custom authority source."""
    name: str = Field(..., min_length=2, max_length=50, description="Source name")
    score: float = Field(..., ge=0.0, le=1.0, description="Authority score")
    keywords: List[str] = Field(..., min_length=1, description="Detection keywords")
    tier: int = Field(default=3, ge=1, le=3, description="Authority tier (1-3)")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        name = v.upper().strip().replace(' ', '_')
        # Check it's not a built-in source
        try:
            AuthoritySource(name)
            raise ValueError(f"Cannot use reserved source name: {name}")
        except ValueError:
            pass
        return name

    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        return [kw.lower().strip() for kw in v if kw.strip()]


class DetectRequest(BaseModel):
    """Request to detect authority from document title."""
    title: str = Field(..., min_length=1, description="Document title")


class DetectResponse(BaseModel):
    """Authority detection result."""
    source: str
    score: float
    matched_keyword: Optional[str] = None


# =============================================================================
# Schema Verification
# =============================================================================

_schema_verified = False
_registry_table_exists = False


async def _check_registry_table(database) -> bool:
    """Check if authority_registry table exists. Cached after first check."""
    global _schema_verified, _registry_table_exists

    if _schema_verified:
        return _registry_table_exists

    try:
        result = await database.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'authority_registry'
            )
        """)
        _registry_table_exists = bool(result)
        _schema_verified = True

        if not _registry_table_exists:
            logger.warning(
                "authority_registry table not found. "
                "Run migration 004_authority_registry.sql for persistence. "
                "Using in-memory registry."
            )
        return _registry_table_exists
    except Exception as e:
        logger.error(f"Error checking registry schema: {e}")
        return False


# =============================================================================
# Persistence Helpers
# =============================================================================

async def _persist_registry(database, registry: AuthorityRegistry) -> bool:
    """
    Save registry state to database.

    Returns True if persisted, False if table doesn't exist.
    """
    if not await _check_registry_table(database):
        logger.debug("Registry table not available, skipping persistence")
        return False

    try:
        config = registry.to_dict()
        await database.execute(
            "UPDATE authority_registry SET config = $1::jsonb WHERE id = 1",
            json.dumps(config)
        )
        logger.info("Authority registry persisted to database")
        return True
    except Exception as e:
        logger.error(f"Failed to persist registry: {e}")
        return False


async def load_registry_from_db(database) -> bool:
    """
    Load registry from database on startup.

    Call this during application startup to restore persisted configuration.

    Returns True if loaded from database, False if using defaults.
    """
    if not await _check_registry_table(database):
        return False

    try:
        row = await database.fetchrow(
            "SELECT config FROM authority_registry WHERE id = 1"
        )

        if row and row['config']:
            config = row['config']
            # Handle both dict and string
            if isinstance(config, str):
                config = json.loads(config)

            if config:
                registry = AuthorityRegistry.from_dict(config)
                set_authority_registry(registry)
                logger.info(
                    f"Loaded authority registry from database: "
                    f"{len(registry.list_sources())} sources"
                )
                return True

        logger.info("No registry config in database, using defaults")
        return False

    except Exception as e:
        logger.warning(f"Could not load authority registry from database: {e}")
        return False


# =============================================================================
# API Endpoints
# =============================================================================

@router.get(
    "",
    response_model=RegistryResponse,
    summary="Get authority registry",
    description="Get all authority sources with their scores and metadata"
)
async def get_registry(
    database=Depends(get_database)
):
    """Get current authority registry configuration."""
    registry = get_authority_registry()
    sources = registry.list_sources()

    persisted = await _check_registry_table(database)

    return RegistryResponse(
        sources=[AuthoritySourceResponse(**s) for s in sources],
        total=len(sources),
        persisted=persisted
    )


@router.put(
    "/score",
    summary="Update source score",
    description="Update the authority score for an existing source"
)
async def update_score(
    request: UpdateScoreRequest,
    database=Depends(get_database)
):
    """
    Update score for an existing authority source.

    Can be used to adjust scores for both built-in and custom sources.
    Changes are persisted to database if available.
    """
    registry = get_authority_registry()

    # Verify source exists
    sources = {s['name']: s for s in registry.list_sources()}
    if request.source not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Authority source not found: {request.source}"
        )

    old_score = sources[request.source]['score']

    # Update in-memory
    registry.set_score(request.source, request.score)

    # Persist to database
    persisted = await _persist_registry(database, registry)

    logger.info(
        f"Updated authority score: {request.source} {old_score:.2f} → {request.score:.2f}"
    )

    return {
        "status": "updated",
        "source": request.source,
        "old_score": old_score,
        "new_score": request.score,
        "persisted": persisted
    }


@router.post(
    "/custom",
    status_code=status.HTTP_201_CREATED,
    summary="Add custom source",
    description="Add a new custom authority source with keywords"
)
async def add_custom_source(
    request: AddCustomRequest,
    database=Depends(get_database)
):
    """
    Add a custom authority source.

    Custom sources allow you to define authority for sources not in the
    default list. Keywords are used to detect the source from document titles.

    Tiers:
    - 1 = Master (score typically 1.0)
    - 2 = Major textbook (score typically 0.85-0.95)
    - 3 = Reference (score typically 0.75-0.85)
    """
    registry = get_authority_registry()

    # Check if already exists
    sources = {s['name']: s for s in registry.list_sources()}
    if request.name in sources:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Source already exists: {request.name}"
        )

    # Add custom source
    registry.add_custom(
        name=request.name,
        score=request.score,
        keywords=request.keywords,
        tier=request.tier
    )

    # Persist
    persisted = await _persist_registry(database, registry)

    logger.info(
        f"Added custom authority source: {request.name} "
        f"(score={request.score}, tier={request.tier}, keywords={request.keywords})"
    )

    return {
        "status": "created",
        "name": request.name,
        "score": request.score,
        "keywords": request.keywords,
        "tier": request.tier,
        "persisted": persisted
    }


@router.delete(
    "/custom/{name}",
    summary="Remove custom source",
    description="Remove a custom authority source"
)
async def remove_custom_source(
    name: str,
    database=Depends(get_database)
):
    """
    Remove a custom authority source.

    Only custom sources can be removed. Built-in sources cannot be deleted
    (use PUT /score to set their score to 0 instead).
    """
    registry = get_authority_registry()
    name_upper = name.upper().strip()

    # Check if it's a built-in source
    try:
        AuthoritySource(name_upper)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot remove built-in source: {name_upper}. "
                   f"Use PUT /score to change its score instead."
        )
    except ValueError:
        pass  # Not a built-in, can proceed

    # Try to remove
    if not registry.remove_custom(name_upper):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Custom source not found: {name_upper}"
        )

    # Persist
    persisted = await _persist_registry(database, registry)

    logger.info(f"Removed custom authority source: {name_upper}")

    return {
        "status": "removed",
        "name": name_upper,
        "persisted": persisted
    }


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect authority from title",
    description="Detect the authority source from a document title"
)
async def detect_authority(
    request: DetectRequest
):
    """
    Detect authority source from a document title.

    Uses keyword matching to identify which authority source
    a document belongs to. Returns the matched source and its score.
    """
    registry = get_authority_registry()
    source, score = registry.detect_from_title(request.title)

    return DetectResponse(
        source=source,
        score=score
    )


@router.post(
    "/reset",
    summary="Reset to defaults",
    description="Reset registry to default configuration"
)
async def reset_registry(
    database=Depends(get_database)
):
    """
    Reset authority registry to default configuration.

    This removes all custom sources and resets all scores to defaults.
    Use with caution as this action cannot be undone.
    """
    # Create fresh registry with defaults
    new_registry = AuthorityRegistry()
    set_authority_registry(new_registry)

    # Persist empty config (will use defaults)
    if await _check_registry_table(database):
        await database.execute(
            "UPDATE authority_registry SET config = '{}'::jsonb WHERE id = 1"
        )

    logger.info("Authority registry reset to defaults")

    return {
        "status": "reset",
        "sources_count": len(new_registry.list_sources())
    }


@router.get(
    "/tiers",
    summary="Get tier definitions",
    description="Get descriptions of authority tiers"
)
async def get_tiers():
    """Get definitions of the authority tier system."""
    return {
        "tiers": [
            {
                "tier": 1,
                "name": "Master",
                "description": "Definitive works by recognized masters of the field",
                "typical_score_range": "0.95 - 1.00",
                "examples": ["Rhoton Cranial Anatomy", "Lawton Seven AVMs", "Samii Approaches"]
            },
            {
                "tier": 2,
                "name": "Major Textbook",
                "description": "Comprehensive multi-author reference textbooks",
                "typical_score_range": "0.85 - 0.95",
                "examples": ["Youmans Neurological Surgery", "Schmidek Operative", "Connolly"]
            },
            {
                "tier": 3,
                "name": "Reference",
                "description": "Specialized references and handbooks",
                "typical_score_range": "0.75 - 0.85",
                "examples": ["Greenberg Handbook", "Benzel Spine", "AO Spine"]
            }
        ]
    }
