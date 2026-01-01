"""
NeuroSynth Unified - Index Routes
==================================

Index management API endpoints for FAISS vector indexes.

WARNING: Job state is stored in-memory only and will be lost on server restart.
For production, migrate to database-backed job storage.
"""

import asyncio
import logging
from typing import Optional, Literal
from datetime import datetime
from enum import Enum
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.api.dependencies import get_container, get_faiss_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/indexes", tags=["Indexes"])


# =============================================================================
# MODELS
# =============================================================================

class IndexType(str, Enum):
    """Index types available for rebuild."""
    TEXT = "text"
    IMAGE = "image"
    CAPTION = "caption"
    ALL = "all"


class RebuildRequest(BaseModel):
    """Request to rebuild indexes."""
    index_type: IndexType = IndexType.ALL
    force: bool = False


class RebuildJobResponse(BaseModel):
    """Response when starting a rebuild job."""
    job_id: str
    status: str
    index_type: str


class RebuildStatus(BaseModel):
    """Status of a rebuild job."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float
    message: str
    index_type: str


class IndexStats(BaseModel):
    """Statistics about vector indexes."""
    text_vectors: int
    image_vectors: int
    caption_vectors: int
    total_vectors: int
    last_updated: Optional[datetime] = None


# In-memory job tracking (WARNING: lost on restart - see module docstring)
_rebuild_jobs: dict = {}
_rebuild_jobs_lock = asyncio.Lock()


# =============================================================================
# ROUTES
# =============================================================================

@router.post(
    "/rebuild",
    response_model=RebuildJobResponse,
    summary="Rebuild indexes",
    description="Start a background job to rebuild vector indexes"
)
async def rebuild_indexes(
    request: RebuildRequest,
    background_tasks: BackgroundTasks,
    container = Depends(get_container)
):
    """
    Start rebuilding vector indexes.

    Runs in background. Use /rebuild/status/{job_id} to check progress.

    Note: Job state is in-memory only and will be lost on server restart.
    """
    job_id = str(uuid.uuid4())
    async with _rebuild_jobs_lock:
        _rebuild_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Queued",
            "index_type": request.index_type.value
        }

    background_tasks.add_task(
        _do_rebuild,
        job_id,
        request.index_type,
        request.force,
        container
    )

    return RebuildJobResponse(
        job_id=job_id,
        status="pending",
        index_type=request.index_type.value
    )


@router.post(
    "/rebuild/{index_type}",
    response_model=RebuildJobResponse,
    summary="Rebuild specific index",
    description="Rebuild a specific index type (text, image, or caption)"
)
async def rebuild_specific_index(
    index_type: IndexType,
    background_tasks: BackgroundTasks,
    container = Depends(get_container)
):
    """Rebuild a specific index type. Job state is in-memory only."""
    job_id = str(uuid.uuid4())
    async with _rebuild_jobs_lock:
        _rebuild_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": f"Queued {index_type.value} rebuild",
            "index_type": index_type.value
        }

    background_tasks.add_task(
        _do_rebuild,
        job_id,
        index_type,
        False,
        container
    )

    return RebuildJobResponse(
        job_id=job_id,
        status="pending",
        index_type=index_type.value
    )


@router.get(
    "/rebuild/status/{job_id}",
    response_model=RebuildStatus,
    summary="Get rebuild status",
    description="Get the status of a rebuild job"
)
async def get_rebuild_status(job_id: str):
    """Get the status of a running rebuild job. Job state is in-memory only."""
    async with _rebuild_jobs_lock:
        if job_id not in _rebuild_jobs:
            raise HTTPException(status_code=404, detail="Job not found (may have been lost on restart)")
        job = _rebuild_jobs[job_id].copy()

    return RebuildStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        index_type=job["index_type"]
    )


@router.get(
    "/stats",
    response_model=IndexStats,
    summary="Get index statistics",
    description="Get statistics about vector indexes"
)
async def get_index_stats(
    faiss = Depends(get_faiss_manager)
):
    """Get statistics about loaded FAISS indexes."""
    if not faiss:
        return IndexStats(
            text_vectors=0,
            image_vectors=0,
            caption_vectors=0,
            total_vectors=0,
            last_updated=None
        )

    stats = faiss.get_stats()

    text_count = stats.get("text", {}).get("size", 0)
    image_count = stats.get("image", {}).get("size", 0)
    caption_count = stats.get("caption", {}).get("size", 0)

    return IndexStats(
        text_vectors=text_count,
        image_vectors=image_count,
        caption_vectors=caption_count,
        total_vectors=text_count + image_count + caption_count,
        last_updated=stats.get("last_updated")
    )


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def _update_job(job_id: str, **updates):
    """Thread-safe update of job state."""
    async with _rebuild_jobs_lock:
        if job_id in _rebuild_jobs:
            _rebuild_jobs[job_id].update(updates)


async def _do_rebuild(
    job_id: str,
    index_type: IndexType,
    force: bool,
    container
):
    """Background task to rebuild indexes from database."""
    try:
        await _update_job(job_id, status="running", message=f"Rebuilding {index_type.value} from database...")

        faiss = container.faiss
        db = container.database

        if not faiss:
            await _update_job(job_id, status="failed", message="FAISS manager not available")
            return

        if not db:
            await _update_job(job_id, status="failed", message="Database not available")
            return

        await _update_job(job_id, progress=10, message="Fetching embeddings from database...")

        # Use build_from_database to rebuild all indexes
        if hasattr(faiss, 'build_from_database'):
            stats = await faiss.build_from_database(db)
            await _update_job(job_id, progress=90, message=f"Saving indexes... Built: {stats}")

            # Save the rebuilt indexes
            faiss.save()

            await _update_job(job_id, status="completed", progress=100, message=f"Rebuild complete: {stats}")
        else:
            # Fallback: just reload from disk (build_from_database not implemented)
            await _update_job(job_id, message="build_from_database not available, reloading indexes from disk...")
            faiss.load()
            await _update_job(job_id, status="completed", progress=100, message="Reload complete (rebuild not implemented)")

    except Exception as e:
        logger.error(f"Rebuild failed for job {job_id}: {e}")
        await _update_job(job_id, status="failed", message=str(e))
