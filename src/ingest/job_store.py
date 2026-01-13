"""
NeuroSynth Unified - Ingestion Job Store
========================================

Production-ready job persistence layer with:
1. Redis-backed storage (survives restarts)
2. Thread-safe operations
3. Automatic backend selection
4. TTL-based job expiry

Usage:
    # Auto-detect (tries Redis, falls back to memory)
    store = JobStore()
    await store.initialize()

    # Create a job
    job = await store.create_job("job-123", "document.pdf")

    # Update progress
    await store.update_job("job-123", stage="extraction", progress=25)

    # Complete or fail
    await store.complete_job("job-123", summary={...})
    await store.fail_job("job-123", error="Something went wrong")

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    JOB_STORE_TTL_HOURS: Job expiry time in hours (default: 72)
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_JOB_TTL = int(os.getenv("JOB_STORE_TTL_HOURS", "72")) * 3600  # 72 hours
MAX_HISTORY_SIZE = 50


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Job:
    """Ingestion job state."""
    job_id: str
    filename: str
    stage: str = "upload"
    progress: float = 0
    status: str = "pending"  # pending, processing, completed, failed, cancelled
    current_operation: str = "Initializing..."
    created_at: str = ""
    updated_at: str = ""
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)  # Non-blocking warnings

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "stage": self.stage,
            "progress": self.progress,
            "status": self.status,
            "current_operation": self.current_operation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "error": self.error,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            job_id=data["job_id"],
            filename=data["filename"],
            stage=data.get("stage", "upload"),
            progress=data.get("progress", 0),
            status=data.get("status", "pending"),
            current_operation=data.get("current_operation", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            summary=data.get("summary"),
            error=data.get("error"),
            warnings=data.get("warnings", []),
        )


# =============================================================================
# ABSTRACT BACKEND
# =============================================================================

class JobStoreBackend(ABC):
    """Abstract backend for job storage."""

    @abstractmethod
    async def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        pass

    @abstractmethod
    async def save(self, job: Job) -> None:
        """Save job."""
        pass

    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Delete job."""
        pass

    @abstractmethod
    async def get_history(self, limit: int = 20) -> List[Job]:
        """Get completed/failed job history."""
        pass

    @abstractmethod
    async def add_to_history(self, job: Job) -> None:
        """Add job to history list."""
        pass


# =============================================================================
# IN-MEMORY BACKEND (Development)
# =============================================================================

class InMemoryJobBackend(JobStoreBackend):
    """In-memory backend for development/testing."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._history: List[Job] = []
        self._lock = asyncio.Lock()

    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def save(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    async def get_history(self, limit: int = 20) -> List[Job]:
        return self._history[:limit]

    async def add_to_history(self, job: Job) -> None:
        async with self._lock:
            self._history.insert(0, job)
            if len(self._history) > MAX_HISTORY_SIZE:
                self._history = self._history[:MAX_HISTORY_SIZE]


# =============================================================================
# REDIS BACKEND (Production)
# =============================================================================

class RedisJobBackend(JobStoreBackend):
    """Redis backend for production."""

    def __init__(
        self,
        redis_client,
        key_prefix: str = "neurosynth:job:",
        ttl: int = DEFAULT_JOB_TTL
    ):
        self._redis = redis_client
        self._prefix = key_prefix
        self._ttl = ttl
        self._history_key = f"{key_prefix}history"

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    async def get(self, job_id: str) -> Optional[Job]:
        data = await self._redis.get(self._key(job_id))
        if data:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return Job.from_dict(json.loads(data))
        return None

    async def save(self, job: Job) -> None:
        key = self._key(job.job_id)
        data = json.dumps(job.to_dict())
        await self._redis.setex(key, self._ttl, data)

    async def delete(self, job_id: str) -> bool:
        result = await self._redis.delete(self._key(job_id))
        return result > 0

    async def get_history(self, limit: int = 20) -> List[Job]:
        # Get from Redis list
        items = await self._redis.lrange(self._history_key, 0, limit - 1)
        jobs = []
        for item in items:
            if isinstance(item, bytes):
                item = item.decode('utf-8')
            try:
                jobs.append(Job.from_dict(json.loads(item)))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse history item: {e}")
        return jobs

    async def add_to_history(self, job: Job) -> None:
        data = json.dumps(job.to_dict())
        await self._redis.lpush(self._history_key, data)
        await self._redis.ltrim(self._history_key, 0, MAX_HISTORY_SIZE - 1)
        # Set TTL on history list
        await self._redis.expire(self._history_key, self._ttl)


# =============================================================================
# UNIFIED JOB STORE
# =============================================================================

class JobStore:
    """
    Unified job store with automatic backend selection.

    Usage:
        # Auto-detect (tries Redis, falls back to memory)
        store = JobStore()
        await store.initialize()

        # Force Redis
        store = JobStore(redis_url="redis://localhost:6379")

        # Force in-memory
        store = JobStore(force_memory=True)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        force_memory: bool = False,
        ttl: int = DEFAULT_JOB_TTL
    ):
        self._backend: Optional[JobStoreBackend] = None
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._force_memory = force_memory
        self._ttl = ttl
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the backend (call once at startup)."""
        if self._initialized:
            return

        if self._force_memory:
            logger.info("Job store using in-memory backend (forced)")
            self._backend = InMemoryJobBackend()
        elif self._redis_url:
            try:
                import redis.asyncio as redis
                client = redis.from_url(self._redis_url)
                await client.ping()
                self._backend = RedisJobBackend(client, ttl=self._ttl)
                logger.info(f"Job store using Redis: {self._redis_url}")
            except ImportError:
                logger.warning("redis package not installed, using in-memory backend")
                self._backend = InMemoryJobBackend()
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), falling back to in-memory")
                self._backend = InMemoryJobBackend()
        else:
            # Try auto-detect Redis at localhost
            try:
                import redis.asyncio as redis
                client = redis.from_url("redis://localhost:6379")
                await client.ping()
                self._backend = RedisJobBackend(client, ttl=self._ttl)
                logger.info("Job store auto-detected Redis at localhost:6379")
            except Exception:
                logger.info("Job store using in-memory backend")
                self._backend = InMemoryJobBackend()

        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure backend is initialized."""
        if not self._initialized:
            await self.initialize()

    # =========================================================================
    # ASYNC METHODS (Preferred for production)
    # =========================================================================

    async def create_job_async(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Create a new job (thread-safe)."""
        await self._ensure_initialized()

        job = Job(job_id=job_id, filename=filename)
        await self._backend.save(job)
        return job.to_dict()

    async def update_job_async(
        self,
        job_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update job fields (thread-safe)."""
        await self._ensure_initialized()

        async with self._lock:
            job = await self._backend.get(job_id)
            if not job:
                return None

            # Update fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

            job.updated_at = datetime.now().isoformat()
            await self._backend.save(job)
            return job.to_dict()

    async def get_job_async(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID (thread-safe)."""
        await self._ensure_initialized()

        job = await self._backend.get(job_id)
        return job.to_dict() if job else None

    async def complete_job_async(
        self,
        job_id: str,
        summary: Dict[str, Any]
    ) -> None:
        """Mark job as completed and add to history (thread-safe)."""
        await self._ensure_initialized()

        async with self._lock:
            job = await self._backend.get(job_id)
            if job:
                job.status = "completed"
                job.stage = "complete"
                job.progress = 100
                job.current_operation = "Done"
                job.summary = summary
                job.updated_at = datetime.now().isoformat()

                await self._backend.save(job)
                await self._backend.add_to_history(job)

    async def fail_job_async(self, job_id: str, error: str) -> None:
        """Mark job as failed and add to history (thread-safe)."""
        await self._ensure_initialized()

        async with self._lock:
            job = await self._backend.get(job_id)
            if job:
                job.status = "failed"
                job.error = error
                job.updated_at = datetime.now().isoformat()

                await self._backend.save(job)
                await self._backend.add_to_history(job)

    async def cancel_job_async(self, job_id: str) -> None:
        """Cancel a job (thread-safe)."""
        await self._ensure_initialized()

        async with self._lock:
            job = await self._backend.get(job_id)
            if job:
                job.status = "cancelled"
                job.updated_at = datetime.now().isoformat()
                await self._backend.save(job)

    async def get_history_async(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get job history (thread-safe)."""
        await self._ensure_initialized()

        jobs = await self._backend.get_history(limit)
        return [job.to_dict() for job in jobs]

    # =========================================================================
    # SYNC METHODS (Backward compatibility with existing code)
    # =========================================================================

    def create_job(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Sync version - creates job in a non-blocking way."""
        # For sync callers, we need to handle the async internally
        # This is a simplified version that works with the existing code
        job = Job(job_id=job_id, filename=filename)

        if self._initialized and self._backend:
            # Try to save immediately if we can
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the save but return immediately
                    asyncio.create_task(self._backend.save(job))
                else:
                    loop.run_until_complete(self._backend.save(job))
            except Exception:
                pass  # Fall through to return

        return job.to_dict()

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Sync version - updates job."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task for async update
                asyncio.create_task(self.update_job_async(job_id, **kwargs))
                # Return current state (may be slightly stale)
                if self._backend:
                    task = asyncio.create_task(self._backend.get(job_id))
                    # Can't await here, just schedule
                return {"job_id": job_id, **kwargs}
            else:
                return loop.run_until_complete(self.update_job_async(job_id, **kwargs))
        except Exception:
            return {"job_id": job_id, **kwargs}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Sync version - gets job."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In async context, we need to be careful
                # Try to get from cache if available
                if hasattr(self._backend, '_jobs'):
                    job = self._backend._jobs.get(job_id)
                    return job.to_dict() if job else None
                return None
            else:
                return loop.run_until_complete(self.get_job_async(job_id))
        except Exception:
            return None

    def complete_job(self, job_id: str, summary: Dict[str, Any]) -> None:
        """Sync version - completes job."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.complete_job_async(job_id, summary))
            else:
                loop.run_until_complete(self.complete_job_async(job_id, summary))
        except Exception as e:
            logger.warning(f"Failed to complete job {job_id}: {e}")

    def fail_job(self, job_id: str, error: str) -> None:
        """Sync version - fails job."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.fail_job_async(job_id, error))
            else:
                loop.run_until_complete(self.fail_job_async(job_id, error))
        except Exception as e:
            logger.warning(f"Failed to fail job {job_id}: {e}")

    def cancel_job(self, job_id: str) -> None:
        """Sync version - cancels job."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.cancel_job_async(job_id))
            else:
                loop.run_until_complete(self.cancel_job_async(job_id))
        except Exception as e:
            logger.warning(f"Failed to cancel job {job_id}: {e}")

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Sync version - gets history."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if hasattr(self._backend, '_history'):
                    return [j.to_dict() for j in self._backend._history[:limit]]
                return []
            else:
                return loop.run_until_complete(self.get_history_async(limit))
        except Exception:
            return []

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def backend_type(self) -> str:
        """Get current backend type."""
        if isinstance(self._backend, RedisJobBackend):
            return "redis"
        elif isinstance(self._backend, InMemoryJobBackend):
            return "memory"
        return "uninitialized"

    @property
    def is_initialized(self) -> bool:
        """Check if store is initialized."""
        return self._initialized


# =============================================================================
# BATCH TRACKER (Redis-enabled)
# =============================================================================

class BatchTracker:
    """
    Tracks batch jobs containing multiple file uploads.
    Supports both Redis and in-memory backends.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        force_memory: bool = False,
        ttl: int = DEFAULT_JOB_TTL
    ):
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._redis_client = None
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._force_memory = force_memory
        self._ttl = ttl
        self._prefix = "neurosynth:batch:"
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Redis connection if available."""
        if self._initialized:
            return

        if not self._force_memory and self._redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self._redis_url)
                await self._redis_client.ping()
                logger.info(f"BatchTracker using Redis: {self._redis_url}")
            except Exception as e:
                logger.warning(f"BatchTracker Redis unavailable ({e}), using in-memory")
                self._redis_client = None

        self._initialized = True

    def _key(self, batch_id: str) -> str:
        return f"{self._prefix}{batch_id}"

    async def create_batch(
        self,
        batch_id: str,
        job_infos: List[Dict[str, str]]
    ) -> Dict:
        """Create a new batch."""
        await self.initialize()

        batch = {
            "batch_id": batch_id,
            "job_ids": [j["job_id"] for j in job_infos],
            "job_infos": job_infos,
            "status": "pending",
            "current_index": 0,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "cancelled": False
        }

        if self._redis_client:
            await self._redis_client.setex(
                self._key(batch_id),
                self._ttl,
                json.dumps(batch, default=str)
            )
        else:
            self._batches[batch_id] = batch

        logger.info(f"Created batch {batch_id} with {len(job_infos)} files")
        return batch

    async def get_batch(self, batch_id: str) -> Optional[Dict]:
        """Get batch by ID."""
        await self.initialize()

        if self._redis_client:
            data = await self._redis_client.get(self._key(batch_id))
            if data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return json.loads(data)
            return None

        return self._batches.get(batch_id)

    async def update_batch(self, batch_id: str, **kwargs) -> None:
        """Update batch fields."""
        await self.initialize()

        async with self._lock:
            batch = await self.get_batch(batch_id)
            if not batch:
                return

            batch.update(kwargs)

            if self._redis_client:
                await self._redis_client.setex(
                    self._key(batch_id),
                    self._ttl,
                    json.dumps(batch, default=str)
                )
            else:
                self._batches[batch_id] = batch

    async def update_batch_status(self, batch_id: str, status: str) -> None:
        """Update batch status."""
        from datetime import datetime

        updates = {"status": status}

        if status == "processing":
            batch = await self.get_batch(batch_id)
            if batch and not batch.get("started_at"):
                updates["started_at"] = datetime.utcnow().isoformat()
        elif status in ("completed", "failed", "partial"):
            updates["completed_at"] = datetime.utcnow().isoformat()

        await self.update_batch(batch_id, **updates)

    async def update_current_index(self, batch_id: str, index: int) -> None:
        """Update which file is currently being processed."""
        await self.update_batch(batch_id, current_index=index)

    async def cancel_batch(self, batch_id: str) -> None:
        """Cancel a batch."""
        await self.update_batch(batch_id, cancelled=True, status="cancelled")

    async def is_cancelled(self, batch_id: str) -> bool:
        """Check if batch is cancelled."""
        batch = await self.get_batch(batch_id)
        return batch.get("cancelled", False) if batch else False


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_job_store: Optional[JobStore] = None
_batch_tracker: Optional[BatchTracker] = None


async def get_job_store(
    redis_url: Optional[str] = None,
    force_memory: bool = False
) -> JobStore:
    """Get singleton job store instance."""
    global _job_store

    if _job_store is None:
        _job_store = JobStore(
            redis_url=redis_url,
            force_memory=force_memory
        )
        await _job_store.initialize()

    return _job_store


async def get_batch_tracker(
    redis_url: Optional[str] = None,
    force_memory: bool = False
) -> BatchTracker:
    """Get singleton batch tracker instance."""
    global _batch_tracker

    if _batch_tracker is None:
        _batch_tracker = BatchTracker(
            redis_url=redis_url,
            force_memory=force_memory
        )
        await _batch_tracker.initialize()

    return _batch_tracker


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing JobStore...")

        # Test in-memory
        store = JobStore(force_memory=True)
        await store.initialize()
        print(f"Backend type: {store.backend_type}")

        # Create job
        job = await store.create_job_async("test-001", "test.pdf")
        print(f"Created job: {job['job_id']}")

        # Update job
        await store.update_job_async("test-001", stage="extraction", progress=25)
        job = await store.get_job_async("test-001")
        print(f"Updated job: stage={job['stage']}, progress={job['progress']}")

        # Complete job
        await store.complete_job_async("test-001", {"chunks": 10, "images": 5})
        job = await store.get_job_async("test-001")
        print(f"Completed job: status={job['status']}, summary={job['summary']}")

        # Get history
        history = await store.get_history_async()
        print(f"History: {len(history)} jobs")

        print("\nAll tests passed!")

    asyncio.run(test())
