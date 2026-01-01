"""
NeuroSynth Unified - Health Routes
===================================

Health check and status API endpoints.

WARNING: Rate limit tracking is in-memory only and will be lost on server restart.
The rate limits shown are approximate and not integrated with actual API call tracking.
For production, implement proper rate limiting middleware with Redis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.models import (
    HealthResponse,
    ComponentStatus,
    StatsResponse
)
from src.api.dependencies import (
    get_container,
    get_settings,
    ServiceContainer,
    Settings
)


# =============================================================================
# Rate Limit Models
# =============================================================================

class RateLimitInfo(BaseModel):
    """Rate limit information for a service."""
    used: int
    limit: int
    reset_at: datetime
    remaining: int


class RateLimitsResponse(BaseModel):
    """Rate limits for all services."""
    claude: RateLimitInfo
    voyage: RateLimitInfo


# In-memory rate tracking (WARNING: not integrated with actual API calls - see module docstring)
# These are placeholder values. Actual tracking requires middleware integration.
_rate_limits = {
    "claude": {"used": 0, "limit": 100, "reset_at": datetime.utcnow() + timedelta(hours=1)},
    "voyage": {"used": 0, "limit": 5000, "reset_at": datetime.utcnow() + timedelta(hours=1)}
}
_rate_limits_lock = asyncio.Lock()

logger = logging.getLogger(__name__)


async def _increment_rate_limit(service: str, count: int = 1):
    """
    Thread-safe increment of rate limit counter.

    Call this from API wrapper functions to track actual usage.
    Currently NOT integrated - this is a placeholder for future middleware.

    Args:
        service: "claude" or "voyage"
        count: Number to increment by (default 1)
    """
    async with _rate_limits_lock:
        now = datetime.utcnow()
        if service in _rate_limits:
            # Reset if past hour
            if now > _rate_limits[service]["reset_at"]:
                _rate_limits[service]["used"] = 0
                _rate_limits[service]["reset_at"] = now + timedelta(hours=1)
            _rate_limits[service]["used"] += count

router = APIRouter(tags=["Health"])


# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check health of all system components"
)
async def health_check(
    container: ServiceContainer = Depends(get_container),
    settings: Settings = Depends(get_settings)
):
    """
    Comprehensive health check.
    
    Checks:
    - Database connectivity
    - FAISS indexes loaded
    - Search service available
    - RAG engine available
    """
    components = {}
    overall_status = "healthy"
    
    # Database health
    if container.database:
        try:
            start = time.time()
            healthy = await container.database.health_check()
            latency = int((time.time() - start) * 1000)
            
            if healthy:
                pool_stats = await container.database.get_stats()
                components["database"] = ComponentStatus(
                    status="healthy",
                    latency_ms=latency,
                    details=pool_stats
                )
            else:
                components["database"] = ComponentStatus(
                    status="unhealthy",
                    details={"error": "Health check failed"}
                )
                overall_status = "degraded"
        except Exception as e:
            components["database"] = ComponentStatus(
                status="unhealthy",
                details={"error": str(e)}
            )
            overall_status = "unhealthy"
    else:
        components["database"] = ComponentStatus(
            status="unhealthy",
            details={"error": "Not initialized"}
        )
        overall_status = "unhealthy"
    
    # FAISS health
    if container.faiss:
        stats = container.faiss.get_stats()
        components["faiss"] = ComponentStatus(
            status="healthy",
            details=stats
        )
    else:
        components["faiss"] = ComponentStatus(
            status="degraded",
            details={"error": "Not loaded"}
        )
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Search service health
    if container.search:
        components["search"] = ComponentStatus(status="healthy")
    else:
        components["search"] = ComponentStatus(
            status="unhealthy",
            details={"error": "Not available"}
        )
        overall_status = "degraded"
    
    # RAG engine health
    if container.rag:
        components["rag"] = ComponentStatus(status="healthy")
    else:
        components["rag"] = ComponentStatus(
            status="degraded",
            details={"error": "Not available"}
        )
        if overall_status == "healthy":
            overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.api_version,
        components=components,
        timestamp=datetime.utcnow()
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes"
)
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Readiness check for Kubernetes"
)
async def readiness(
    container: ServiceContainer = Depends(get_container)
):
    """Kubernetes readiness probe."""
    if container.is_healthy:
        return {"status": "ready"}
    return {"status": "not_ready"}, 503


# =============================================================================
# Statistics
# =============================================================================

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Get system-wide statistics"
)
async def get_stats(
    container: ServiceContainer = Depends(get_container)
):
    """Get comprehensive system statistics."""
    stats = {
        "documents": 0,
        "chunks": 0,
        "images": 0,
        "links": 0,
        "faiss_indexes": {},
        "database": {}
    }
    
    # Database stats
    if container.repositories:
        try:
            stats["documents"] = await container.repositories.documents.count()
            
            chunk_stats = await container.repositories.chunks.get_statistics()
            stats["chunks"] = chunk_stats.get("total", 0)
            
            image_stats = await container.repositories.images.get_statistics()
            stats["images"] = image_stats.get("total", 0)
            
            link_stats = await container.repositories.links.get_statistics()
            stats["links"] = link_stats.get("total", 0)
            
        except Exception as e:
            logger.error(f"Error getting DB stats: {e}")
    
    # FAISS stats
    if container.faiss:
        faiss_stats = container.faiss.get_stats()
        stats["faiss_indexes"] = {
            "text": faiss_stats.get("text", {}).get("size", 0),
            "image": faiss_stats.get("image", {}).get("size", 0),
            "caption": faiss_stats.get("caption", {}).get("size", 0)
        }
    
    # Database connection stats
    if container.database:
        try:
            stats["database"] = await container.database.get_stats()
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
    
    return StatsResponse(**stats)


# =============================================================================
# Info
# =============================================================================

@router.get(
    "/info",
    summary="API information",
    description="Get API version and configuration info"
)
async def get_info(
    settings: Settings = Depends(get_settings)
):
    """Get API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "debug": settings.debug,
        "models": {
            "embedding": settings.voyage_model,
            "llm": settings.claude_model
        }
    }


# =============================================================================
# Metadata
# =============================================================================

@router.get(
    "/metadata",
    summary="Get metadata options",
    description="Get available filter options for UI (chunk types, specialties, image types)"
)
async def get_metadata():
    """
    Get metadata for UI filters.

    Returns available options for:
    - chunk_types: All valid ChunkType enum values
    - specialties: All neurosurgery specialties from chunker
    - image_types: All valid ImageType enum values

    This ensures frontend filter options stay synchronized with backend.
    """
    from src.shared.models import ChunkType, ImageType
    from src.core.neuro_chunker import EXPANDED_CATEGORY_SPECIALTY_MAP

    return {
        "chunk_types": [ct.value for ct in ChunkType],
        "specialties": sorted(set(EXPANDED_CATEGORY_SPECIALTY_MAP.values())),
        "image_types": [it.value for it in ImageType]
    }


# =============================================================================
# Rate Limits
# =============================================================================

@router.get(
    "/api/v1/rate-limits",
    response_model=RateLimitsResponse,
    summary="Get rate limits",
    description="Get current rate limit status for Claude and Voyage APIs"
)
async def get_rate_limits():
    """
    Get current rate limit status.

    Returns usage and limits for:
    - Claude API (LLM calls)
    - Voyage API (embedding calls)

    NOTE: These values are approximate and not integrated with actual API tracking.
    For production, implement proper rate limiting middleware with Redis.
    """
    async with _rate_limits_lock:
        now = datetime.utcnow()

        # Reset if past reset time
        for key in _rate_limits:
            if now > _rate_limits[key]["reset_at"]:
                _rate_limits[key]["used"] = 0
                _rate_limits[key]["reset_at"] = now + timedelta(hours=1)

        return RateLimitsResponse(
            claude=RateLimitInfo(
                used=_rate_limits["claude"]["used"],
                limit=_rate_limits["claude"]["limit"],
                reset_at=_rate_limits["claude"]["reset_at"],
                remaining=_rate_limits["claude"]["limit"] - _rate_limits["claude"]["used"]
            ),
            voyage=RateLimitInfo(
                used=_rate_limits["voyage"]["used"],
                limit=_rate_limits["voyage"]["limit"],
                reset_at=_rate_limits["voyage"]["reset_at"],
                remaining=_rate_limits["voyage"]["limit"] - _rate_limits["voyage"]["used"]
            )
        )
