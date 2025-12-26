"""
NeuroSynth Unified - Health Routes
===================================

Health check and status API endpoints.
"""

import logging
from datetime import datetime
import time

from fastapi import APIRouter, Depends

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

logger = logging.getLogger(__name__)

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
