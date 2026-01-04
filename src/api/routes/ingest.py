"""
NeuroSynth Unified - Ingest Routes
===================================

Document ingestion API endpoints for PDF upload and processing.

WARNING: Job state is stored in-memory only and will be lost on server restart.
For production, migrate to Redis or database-backed job storage.
"""

import asyncio
import gc
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


# =============================================================================
# IN-MEMORY JOB STORE (for demo - use Redis in production)
# WARNING: All state is lost on restart. Use database/Redis for production.
# =============================================================================

class JobStore:
    """
    Simple in-memory job store with thread-safe operations.

    WARNING: State is lost on server restart. For production deployments,
    migrate to Redis or database-backed storage.
    """

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def create_job_async(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Thread-safe job creation."""
        async with self._lock:
            return self._create_job_internal(job_id, filename)

    def _create_job_internal(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Internal job creation (call within lock or synchronously)."""
        job = {
            "job_id": job_id,
            "filename": filename,
            "stage": "upload",
            "progress": 0,
            "status": "pending",
            "current_operation": "Initializing...",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "summary": None,
            "error": None
        }
        self.jobs[job_id] = job
        return job

    def create_job(self, job_id: str, filename: str) -> Dict[str, Any]:
        """Sync version for backward compatibility. Use create_job_async in async context."""
        return self._create_job_internal(job_id, filename)

    async def update_job_async(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Thread-safe job update."""
        async with self._lock:
            return self._update_job_internal(job_id, **kwargs)

    def _update_job_internal(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Internal update (call within lock or synchronously)."""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
            return self.jobs[job_id]
        return None

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Sync version for backward compatibility."""
        return self._update_job_internal(job_id, **kwargs)

    async def get_job_async(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Thread-safe job retrieval."""
        async with self._lock:
            job = self.jobs.get(job_id)
            return job.copy() if job else None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Sync version for backward compatibility."""
        return self.jobs.get(job_id)

    async def complete_job_async(self, job_id: str, summary: Dict[str, Any]):
        """Thread-safe job completion."""
        async with self._lock:
            self._complete_job_internal(job_id, summary)

    def _complete_job_internal(self, job_id: str, summary: Dict[str, Any]):
        """Internal completion (call within lock or synchronously)."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job["status"] = "completed"
            job["stage"] = "complete"
            job["progress"] = 100
            job["current_operation"] = "Done"
            job["summary"] = summary
            job["updated_at"] = datetime.now().isoformat()
            self.history.insert(0, job.copy())
            if len(self.history) > 50:
                self.history = self.history[:50]

    def complete_job(self, job_id: str, summary: Dict[str, Any]):
        """Sync version for backward compatibility."""
        self._complete_job_internal(job_id, summary)

    async def fail_job_async(self, job_id: str, error: str):
        """Thread-safe job failure."""
        async with self._lock:
            self._fail_job_internal(job_id, error)

    def _fail_job_internal(self, job_id: str, error: str):
        """Internal failure (call within lock or synchronously)."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job["status"] = "failed"
            job["error"] = error
            job["updated_at"] = datetime.now().isoformat()
            self.history.insert(0, job.copy())

    def fail_job(self, job_id: str, error: str):
        """Sync version for backward compatibility."""
        self._fail_job_internal(job_id, error)

    def cancel_job(self, job_id: str):
        """Cancel a job. Note: not fully thread-safe in sync mode."""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "cancelled"
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get job history. Returns defensive copy to prevent external mutation."""
        return [job.copy() for job in self.history[:limit]]


# Global job store
_job_store = JobStore()


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class UploadResponse(BaseModel):
    """Response after upload."""
    job_id: str
    filename: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    filename: str
    stage: str
    progress: float
    status: str
    current_operation: Optional[str] = None
    created_at: str
    updated_at: str
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HistoryItem(BaseModel):
    """History item."""
    job_id: str
    filename: str
    status: str
    created_at: str
    summary: Optional[Dict[str, Any]] = None


# =============================================================================
# BACKGROUND PROCESSING
# =============================================================================

async def process_document_task(
    job_id: str,
    file_path: Path,
    title: str,
    config: Dict[str, Any]
):
    """Background task to process a document with real progress tracking."""
    from src.ingest.unified_pipeline import UnifiedPipeline
    from src.ingest.config import UnifiedPipelineConfig
    from src.database.connection import get_connection_string
    import os

    try:
        # Update to processing status
        await _job_store.update_job_async(
            job_id,
            stage="initializing",
            progress=5,
            status="processing",
            current_operation="Setting up pipeline..."
        )

        # Get database connection
        connection_string = get_connection_string()

        # Create pipeline config using the proper factory method
        pipeline_config = UnifiedPipelineConfig.for_database(
            connection_string=connection_string,
            enable_ocr=True,
            enable_tables=config.get("extract_tables", True),
            enable_knowledge_graph=True,  # Enable entity extraction
        )

        # Update embedding config with API keys from environment
        pipeline_config.embedding.text_api_key = os.getenv("VOYAGE_API_KEY", "")
        pipeline_config.embedding.vlm_api_key = os.getenv("ANTHROPIC_API_KEY", "")

        # REAL PROGRESS CALLBACK: Updates job store with actual pipeline progress
        # Handles both UnifiedPipeline (stage, current, total, message) and regular Pipeline (ProgressInfo)
        def progress_callback(stage_or_info, current=None, total=None, message=None):
            # Map ALL backend/Phase1 stage names to frontend-expected names
            # Frontend expects: upload, initializing, extraction, chunking, embedding, vlm, database, export, complete
            stage_name_map = {
                # Phase 1 pipeline stages
                "init": "initializing",
                "structure": "extraction",
                "pages": "extraction",
                "images": "extraction",
                "tables": "extraction",
                "chunking": "chunking",
                "chunk_summarization": "chunking",
                "umls_extraction": "chunking",
                "linking": "chunking",
                "text_embedding": "embedding",
                "image_embedding": "embedding",
                "caption_embedding": "embedding",
                "vlm_caption": "vlm",
                "caption_summarization": "vlm",
                "storage": "database",
                "complete": "complete",
                # Database writer stages
                "db_document": "database",
                "db_chunks": "database",
                "db_images": "database",
                "db_links": "database",
                "db_entities": "database",
            }

            # Stage weights for percentage calculation
            stage_weights = {
                "upload": (0, 5),
                "initializing": (5, 10),
                "extraction": (10, 30),
                "chunking": (30, 45),
                "embedding": (45, 65),
                "vlm": (65, 85),
                "database": (85, 95),
                "export": (95, 98),
                "complete": (98, 100),
            }

            # Handle ProgressInfo object from regular Pipeline
            if hasattr(stage_or_info, 'stage'):
                raw_stage = stage_or_info.stage.value if hasattr(stage_or_info.stage, 'value') else str(stage_or_info.stage)
                overall = int(stage_or_info.overall_progress * 100) if hasattr(stage_or_info, 'overall_progress') else 5
                msg = getattr(stage_or_info, 'message', '') or f"Processing..."
            else:
                # Handle UnifiedPipeline (stage, current, total, message)
                raw_stage = str(stage_or_info)
                start, end = stage_weights.get(raw_stage, (5, 95))
                if total and total > 0:
                    stage_progress = (current or 0) / total
                    overall = int(start + (end - start) * stage_progress)
                else:
                    overall = start
                msg = message or f"Processing..."

            # Normalize stage name for frontend
            stage_name = stage_name_map.get(raw_stage, raw_stage)

            # Use asyncio.create_task for thread-safe update from sync callback
            asyncio.create_task(_job_store.update_job_async(
                job_id,
                stage=stage_name,
                progress=overall,
                current_operation=msg
            ))

        # Create and initialize pipeline WITH progress callback
        pipeline = UnifiedPipeline(config=pipeline_config, on_progress=progress_callback)
        await pipeline.initialize()

        await _job_store.update_job_async(
            job_id,
            stage="extraction",
            progress=10,
            current_operation="Extracting content from PDF..."
        )

        # EXECUTE PIPELINE
        result = await pipeline.process_document(
            pdf_path=file_path,
            title=title
        )

        # Close pipeline
        await pipeline.close()

        # CRITICAL FIX 1: Check for explicit pipeline errors
        if result.error:
            logger.error(f"Job {job_id} pipeline error: {result.error}")
            await _job_store.fail_job_async(job_id, result.error)
            return

        # CRITICAL FIX 2: Validate content was actually produced
        if result.chunk_count == 0 and result.image_count == 0:
            msg = "Pipeline completed but produced 0 chunks and 0 images. Check PDF content/encryption."
            logger.warning(f"Job {job_id}: {msg}")
            await _job_store.fail_job_async(job_id, msg)
            return

        # SUCCESS: Create summary and complete job
        summary = {
            "document_id": str(result.document_id) if result.document_id else None,
            "chunks": result.chunk_count,
            "images": result.image_count,
            "entities": len(getattr(result, 'entities', [])),
            "links": result.link_count,
            "pages": result.total_pages
        }

        await _job_store.complete_job_async(job_id, summary)
        logger.info(f"Job {job_id} completed successfully: {summary}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed with exception: {e}")
        await _job_store.fail_job_async(job_id, str(e))

    finally:
        # Cleanup temp file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    extract_images: bool = Form(True),
    extract_tables: bool = Form(True),
    detect_sections: bool = Form(True),
    generate_embeddings: bool = Form(True),
    vlm_captioning: bool = Form(True)
):
    """
    Upload a PDF document for processing.

    The document will be processed asynchronously through the pipeline:
    1. Parse PDF
    2. Extract images
    3. Chunk text
    4. Extract entities
    5. Link images to chunks
    6. Generate text embeddings
    7. VLM captioning
    8. Generate image embeddings
    9. Build knowledge graph
    10. Write to database
    11. Update FAISS indexes
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save file temporarily
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp/neurosynth_uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{job_id}_{file.filename}"

    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create job
    _job_store.create_job(job_id, file.filename)

    # Config for processing
    config = {
        "extract_images": extract_images,
        "extract_tables": extract_tables,
        "detect_sections": detect_sections,
        "generate_embeddings": generate_embeddings,
        "vlm_captioning": vlm_captioning
    }

    # Start background processing
    background_tasks.add_task(
        process_document_task,
        job_id,
        file_path,
        title or file.filename.replace('.pdf', ''),
        config
    )

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="processing",
        message="Document upload started. Use /status/{job_id} to track progress."
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of an ingestion job."""
    job = _job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatus(**job)


@router.get("/history")
async def get_ingestion_history(limit: int = 20):
    """Get recent ingestion history."""
    history = _job_store.get_history(limit)
    return {
        "history": history,
        "total": len(history)
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel an in-progress ingestion job."""
    job = _job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] == "completed":
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")

    _job_store.cancel_job(job_id)

    return {"status": "cancelled", "job_id": job_id}


@router.post("/cancel/{job_id}")
async def cancel_job_alias(job_id: str):
    """Alias for /{job_id}/cancel (frontend compatibility)."""
    return await cancel_job(job_id)


# =============================================================================
# BATCH UPLOAD CONFIGURATION
# =============================================================================

class BatchConfig:
    """Batch upload limits and settings."""
    MAX_FILES = 10
    MAX_FILE_SIZE_MB = 100
    MAX_BATCH_SIZE_MB = 500
    ALLOWED_EXTENSIONS = {'.pdf'}
    DELAY_BETWEEN_FILES = 1.0  # seconds


# =============================================================================
# BATCH STATUS ENUM
# =============================================================================

class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some succeeded, some failed
    CANCELLED = "cancelled"


# =============================================================================
# BATCH RESPONSE MODELS
# =============================================================================

class BatchJobInfo(BaseModel):
    """Info about a single job in a batch."""
    job_id: str
    filename: str
    status: str
    progress: int = 0
    current_stage: Optional[str] = None
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """Response from batch upload endpoint."""
    batch_id: str
    jobs: List[BatchJobInfo]
    total_files: int
    status: str
    message: str


class BatchStatusResponse(BaseModel):
    """Response from batch status endpoint."""
    batch_id: str
    status: str
    total_files: int
    completed_files: int
    failed_files: int
    current_file_index: int
    current_file_name: Optional[str] = None
    jobs: List[BatchJobInfo]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None


# =============================================================================
# BATCH TRACKER
# =============================================================================

class BatchTracker:
    """
    Tracks batch jobs containing multiple file uploads.
    """

    def __init__(self):
        self.batches: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_batch(
        self,
        batch_id: str,
        job_infos: List[Dict[str, str]]
    ) -> Dict:
        """
        Create a new batch containing multiple jobs.

        Args:
            batch_id: Unique batch identifier
            job_infos: List of {job_id, filename, file_path} dicts
        """
        async with self._lock:
            batch = {
                "batch_id": batch_id,
                "job_ids": [j["job_id"] for j in job_infos],
                "job_infos": job_infos,
                "status": BatchStatus.PENDING,
                "current_index": 0,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "completed_at": None,
                "cancelled": False
            }
            self.batches[batch_id] = batch
            logger.info(f"Created batch {batch_id} with {len(job_infos)} files")
            return batch

    async def get_batch(self, batch_id: str) -> Optional[Dict]:
        """Get batch info by ID."""
        return self.batches.get(batch_id)

    async def update_batch_status(self, batch_id: str, status: BatchStatus):
        """Update batch status."""
        async with self._lock:
            if batch_id in self.batches:
                self.batches[batch_id]["status"] = status
                if status == BatchStatus.PROCESSING and not self.batches[batch_id]["started_at"]:
                    self.batches[batch_id]["started_at"] = datetime.utcnow()
                elif status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.PARTIAL):
                    self.batches[batch_id]["completed_at"] = datetime.utcnow()

    async def update_current_index(self, batch_id: str, index: int):
        """Update which file is currently being processed."""
        async with self._lock:
            if batch_id in self.batches:
                self.batches[batch_id]["current_index"] = index

    async def cancel_batch(self, batch_id: str):
        """Mark batch as cancelled."""
        async with self._lock:
            if batch_id in self.batches:
                self.batches[batch_id]["cancelled"] = True
                self.batches[batch_id]["status"] = BatchStatus.CANCELLED

    async def is_cancelled(self, batch_id: str) -> bool:
        """Check if batch was cancelled."""
        batch = self.batches.get(batch_id)
        return batch["cancelled"] if batch else False

    async def get_batch_status(
        self,
        batch_id: str,
        job_store  # Reference to main JobStore for individual job statuses
    ) -> Optional[BatchStatusResponse]:
        """
        Get comprehensive batch status with all job statuses.
        """
        batch = self.batches.get(batch_id)
        if not batch:
            return None

        jobs = []
        completed = 0
        failed = 0

        for info in batch["job_infos"]:
            job_id = info["job_id"]
            job_status = job_store.get_job(job_id)

            if job_status:
                job_info = BatchJobInfo(
                    job_id=job_id,
                    filename=info["filename"],
                    status=job_status.get("status", "pending"),
                    progress=job_status.get("progress", 0),
                    current_stage=job_status.get("stage"),
                    error=job_status.get("error")
                )

                if job_status.get("status") == "completed":
                    completed += 1
                elif job_status.get("status") == "failed":
                    failed += 1
            else:
                job_info = BatchJobInfo(
                    job_id=job_id,
                    filename=info["filename"],
                    status="pending"
                )

            jobs.append(job_info)

        # Calculate elapsed time
        elapsed = None
        if batch["started_at"]:
            end_time = batch["completed_at"] or datetime.utcnow()
            elapsed = (end_time - batch["started_at"]).total_seconds()

        # Determine current file name
        current_name = None
        idx = batch["current_index"]
        if 0 <= idx < len(batch["job_infos"]):
            current_name = batch["job_infos"][idx]["filename"]

        return BatchStatusResponse(
            batch_id=batch_id,
            status=batch["status"].value if isinstance(batch["status"], BatchStatus) else batch["status"],
            total_files=len(batch["job_ids"]),
            completed_files=completed,
            failed_files=failed,
            current_file_index=batch["current_index"],
            current_file_name=current_name,
            jobs=jobs,
            started_at=batch["started_at"],
            completed_at=batch["completed_at"],
            elapsed_seconds=elapsed
        )

    def cleanup_old_batches(self, max_age_hours: int = 24):
        """Remove old completed batches."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = [
            bid for bid, batch in self.batches.items()
            if batch.get("completed_at") and batch["completed_at"] < cutoff
        ]
        for bid in to_remove:
            del self.batches[bid]


# Global batch tracker
_batch_tracker = BatchTracker()


# =============================================================================
# BATCH PROCESSING TASK
# =============================================================================

async def process_batch_task(
    batch_id: str,
    job_infos: List[Dict[str, Any]]
):
    """
    Process files sequentially for consistent quality.

    This runs as a background task and processes one file at a time
    to avoid overwhelming external APIs and maintain quality.
    """
    logger.info(f"Starting batch {batch_id} with {len(job_infos)} files")

    await _batch_tracker.update_batch_status(batch_id, BatchStatus.PROCESSING)

    completed = 0
    failed = 0

    for index, job_info in enumerate(job_infos):
        # Check for cancellation
        if await _batch_tracker.is_cancelled(batch_id):
            logger.info(f"Batch {batch_id} was cancelled at file {index + 1}")
            break

        job_id = job_info["job_id"]
        filename = job_info["filename"]

        logger.info(f"[{index + 1}/{len(job_infos)}] Processing: {filename}")

        # Update current index
        await _batch_tracker.update_current_index(batch_id, index)

        try:
            # Use existing document processing function
            await process_document_task(
                job_id=job_id,
                file_path=Path(job_info["file_path"]),
                title=job_info["title"],
                config=job_info["config"]
            )

            completed += 1
            logger.info(f"[{index + 1}/{len(job_infos)}] Completed: {filename}")

        except Exception as e:
            failed += 1
            logger.error(f"[{index + 1}/{len(job_infos)}] Failed: {filename} - {e}")

            # Update job status to failed
            _job_store.update_job(job_id,
                status="failed",
                error=str(e),
                progress=0
            )

        # Memory cleanup between files
        gc.collect()

        # Brief delay to let APIs recover
        if index < len(job_infos) - 1:
            await asyncio.sleep(BatchConfig.DELAY_BETWEEN_FILES)

    # Determine final batch status
    if await _batch_tracker.is_cancelled(batch_id):
        final_status = BatchStatus.CANCELLED
    elif failed == len(job_infos):
        final_status = BatchStatus.FAILED
    elif failed > 0:
        final_status = BatchStatus.PARTIAL
    else:
        final_status = BatchStatus.COMPLETED

    await _batch_tracker.update_batch_status(batch_id, final_status)

    logger.info(
        f"Batch {batch_id} finished: {completed} completed, {failed} failed, "
        f"status: {final_status.value}"
    )


# =============================================================================
# BATCH UPLOAD ENDPOINTS
# =============================================================================

@router.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    title_prefix: Optional[str] = Form(None, description="Prefix for document titles"),
    extract_images: bool = Form(True),
    extract_tables: bool = Form(True),
    detect_sections: bool = Form(True),
    generate_embeddings: bool = Form(True),
    vlm_captioning: bool = Form(True),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """
    Upload multiple PDF documents for sequential processing.

    Files are processed one at a time to maintain quality and avoid
    rate limits on external APIs (VLM, embeddings).

    Returns a batch_id that can be used to poll status.
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(400, "No files provided")

    if len(files) > BatchConfig.MAX_FILES:
        raise HTTPException(
            400,
            f"Too many files. Maximum is {BatchConfig.MAX_FILES}, got {len(files)}"
        )

    # Validate files
    job_infos = []
    total_size = 0

    for file in files:
        # Check extension
        filename = file.filename or "unknown.pdf"
        ext = Path(filename).suffix.lower()

        if ext not in BatchConfig.ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Invalid file type: {filename}. Only PDF files are allowed."
            )

        # Check file size (if available)
        if hasattr(file, 'size') and file.size:
            if file.size > BatchConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    400,
                    f"File too large: {filename}. Maximum is {BatchConfig.MAX_FILE_SIZE_MB}MB"
                )
            total_size += file.size

    # Check total batch size
    if total_size > BatchConfig.MAX_BATCH_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            400,
            f"Total batch size exceeds {BatchConfig.MAX_BATCH_SIZE_MB}MB"
        )

    # Generate batch ID
    batch_id = str(uuid.uuid4())

    # Save files and create job entries
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp/neurosynth_uploads")) / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "extract_images": extract_images,
        "extract_tables": extract_tables,
        "detect_sections": detect_sections,
        "generate_embeddings": generate_embeddings,
        "vlm_captioning": vlm_captioning,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    try:
        for i, file in enumerate(files):
            job_id = str(uuid.uuid4())
            filename = file.filename or f"document_{i+1}.pdf"

            # Generate title
            title = Path(filename).stem
            if title_prefix:
                title = f"{title_prefix} - {title}"

            # Save file
            file_path = upload_dir / f"{job_id}_{filename}"
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            # Create job in JobStore
            _job_store.create_job(job_id, filename)

            job_infos.append({
                "job_id": job_id,
                "filename": filename,
                "file_path": str(file_path),
                "title": title,
                "config": config
            })

        # Create batch
        await _batch_tracker.create_batch(batch_id, job_infos)

        # Start background processing
        background_tasks.add_task(
            process_batch_task,
            batch_id,
            job_infos
        )

        # Build response
        response_jobs = [
            BatchJobInfo(
                job_id=info["job_id"],
                filename=info["filename"],
                status="pending"
            )
            for info in job_infos
        ]

        return BatchUploadResponse(
            batch_id=batch_id,
            jobs=response_jobs,
            total_files=len(files),
            status="pending",
            message=f"Batch created with {len(files)} files. Processing will start shortly."
        )

    except Exception as e:
        # Cleanup on failure
        if upload_dir.exists():
            import shutil
            shutil.rmtree(upload_dir, ignore_errors=True)

        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(500, f"Failed to create batch: {str(e)}")


@router.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get status of all jobs in a batch.

    Returns comprehensive status including:
    - Overall batch status
    - Per-file progress and status
    - Current file being processed
    - Completion counts
    """
    status = await _batch_tracker.get_batch_status(batch_id, _job_store)

    if not status:
        raise HTTPException(404, f"Batch not found: {batch_id}")

    return status


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch_endpoint(batch_id: str):
    """
    Cancel a running batch.

    Files currently processing will complete, but no new files will start.
    """
    batch = await _batch_tracker.get_batch(batch_id)

    if not batch:
        raise HTTPException(404, f"Batch not found: {batch_id}")

    if batch["status"] in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
        raise HTTPException(400, f"Batch already finished with status: {batch['status']}")

    await _batch_tracker.cancel_batch(batch_id)

    return {"message": f"Batch {batch_id} cancellation requested"}


@router.post("/batch/{batch_id}/retry-failed")
async def retry_failed_jobs(
    batch_id: str,
    background_tasks: BackgroundTasks
):
    """
    Retry only failed jobs in a batch.
    """
    batch = await _batch_tracker.get_batch(batch_id)

    if not batch:
        raise HTTPException(404, f"Batch not found: {batch_id}")

    if batch["status"] not in (BatchStatus.PARTIAL, BatchStatus.FAILED):
        raise HTTPException(
            400,
            f"Cannot retry batch with status: {batch['status']}. "
            f"Only 'partial' or 'failed' batches can be retried."
        )

    # Find failed jobs
    failed_jobs = []
    for info in batch["job_infos"]:
        job = _job_store.get_job(info["job_id"])
        if job and job.get("status") == "failed":
            # Reset job status
            _job_store.update_job(info["job_id"],
                status="pending",
                progress=0,
                error=None
            )
            failed_jobs.append(info)

    if not failed_jobs:
        return {"message": "No failed jobs to retry", "retry_count": 0}

    # Update batch status
    await _batch_tracker.update_batch_status(batch_id, BatchStatus.PROCESSING)
    batch["cancelled"] = False

    # Start background retry
    background_tasks.add_task(
        process_batch_task,
        batch_id,
        failed_jobs
    )

    return {
        "message": f"Retrying {len(failed_jobs)} failed jobs",
        "retry_count": len(failed_jobs),
        "job_ids": [j["job_id"] for j in failed_jobs]
    }


# =============================================================================
# MEMORY-SAFE PIPELINE ENDPOINTS
# =============================================================================

class MemorySafeUploadResponse(BaseModel):
    """Response for memory-safe upload endpoint."""
    job_id: str
    status: str
    config: Dict[str, Any]
    message: str


def _get_auto_pipeline_config():
    """
    Detect system resources and return a safe pipeline configuration.
    """
    import psutil
    from src.ingest.memory_safe_pipeline import PipelineConfig

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)

    config = PipelineConfig()

    # ALWAYS use subprocess for BiomedCLIP to prevent segfaults
    # when running SciSpacy + BiomedCLIP in the same process
    config.use_subprocess = True

    if available_gb < 4.0:
        # LOW MEMORY MODE (<4GB)
        logger.info(f"Low memory mode: {available_gb:.1f}GB available")
        config.image_batch_size = 2
        config.chunk_batch_size = 20
        config.min_available_memory_mb = 500
    elif available_gb < 8.0:
        # MEDIUM MEMORY MODE (4-8GB)
        logger.info(f"Medium memory mode: {available_gb:.1f}GB available")
        config.image_batch_size = 4
    else:
        # HIGH MEMORY MODE (>8GB)
        logger.info(f"High memory mode: {available_gb:.1f}GB available")
        config.image_batch_size = 8

    return config


async def _run_memory_safe_pipeline(
    job_id: str,
    file_path: Path,
    config,
    title: str
):
    """
    Background task to run memory-safe pipeline.

    WARNING: This pipeline processes documents but does NOT persist results to database.
    Results are returned in the job summary but not stored. For production use,
    use the regular /upload endpoint which writes to database and updates FAISS.
    """
    from src.ingest.memory_safe_pipeline import MemorySafePipeline

    try:
        _job_store.update_job(job_id, status="processing", stage="parsing", progress=10)

        pipeline = MemorySafePipeline(config)
        result = await pipeline.process(file_path)

        # Note: Database writing not yet implemented for memory-safe mode
        _job_store.update_job(job_id, stage="storing", progress=80)

        # Build summary with explicit warning about non-persistence
        summary = {
            "title": title,
            "chunks": result["stats"]["chunk_count"],
            "images": result["stats"]["image_count"],
            "entities": result["stats"]["total_entities"],
            "embedded_images": result["stats"]["embedded_images"],
            "warning": (
                "EXPERIMENTAL: Memory-safe mode does NOT persist results to database. "
                "Use regular /upload endpoint for production ingestion."
            ),
        }

        _job_store.complete_job(job_id, summary)
        logger.info(f"Memory-safe job {job_id} completed (NOT persisted): {summary}")

    except Exception as e:
        logger.error(f"Memory-safe job {job_id} failed: {e}")
        _job_store.fail_job(job_id, str(e))
    finally:
        # Cleanup uploaded file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


@router.post("/memory-safe/upload", response_model=MemorySafeUploadResponse)
async def memory_safe_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: Optional[str] = Form(None, description="Document title"),
    force_subprocess: bool = Form(False, description="Force subprocess isolation for embeddings"),
    enable_scispacy: bool = Form(True, description="Enable entity extraction"),
    enable_biomedclip: bool = Form(True, description="Enable image embeddings"),
):
    """
    Upload a PDF with memory-safe processing.

    This endpoint uses staged pipeline execution with explicit garbage collection
    between phases. Heavy models (SciSpacy, BiomedCLIP) are loaded on-demand and
    unloaded immediately after use.

    **Memory Modes (auto-detected):**
    - **Low (<4GB)**: Subprocess isolation, smaller batches
    - **Medium (4-8GB)**: Subprocess isolation, normal batches
    - **High (>8GB)**: In-process, larger batches

    Use `force_subprocess=true` to always use subprocess isolation for maximum
    memory safety (recommended for very large documents or constrained environments).
    """
    import psutil

    # Validate file
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Auto-configure pipeline
    config = _get_auto_pipeline_config()
    if force_subprocess:
        config.use_subprocess = True
    config.enable_scispacy = enable_scispacy
    config.enable_biomedclip = enable_biomedclip

    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp/neurosynth_uploads")) / "memory_safe"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{job_id}.pdf"

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Create job
    doc_title = title or Path(filename).stem
    _job_store.create_job(job_id, filename)

    # Queue background processing
    background_tasks.add_task(
        _run_memory_safe_pipeline,
        job_id,
        file_path,
        config,
        doc_title
    )

    return MemorySafeUploadResponse(
        job_id=job_id,
        status="processing",
        config={
            "mode": "subprocess" if config.use_subprocess else "in-process",
            "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "enable_scispacy": config.enable_scispacy,
            "enable_biomedclip": config.enable_biomedclip,
        },
        message=f"Processing started in {'subprocess' if config.use_subprocess else 'in-process'} mode"
    )


@router.get("/memory-safe/memory-stats")
async def get_memory_stats():
    """
    Get current memory usage statistics.

    Returns information about:
    - Process RSS (Resident Set Size)
    - Managed models and their memory usage
    - System available memory
    - Currently loaded models
    """
    import psutil
    from src.core.model_manager import model_manager

    process = psutil.Process()
    mem = psutil.virtual_memory()

    return {
        "process": {
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "vms_mb": round(process.memory_info().vms / (1024 * 1024), 2),
        },
        "system": {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "percent_used": mem.percent,
        },
        "model_manager": model_manager.get_memory_stats(),
    }


@router.post("/memory-safe/unload-all")
async def unload_all_models():
    """
    Force unload all loaded models.

    Use this endpoint to free memory if models are stuck loaded
    (e.g., due to an interrupted request).
    """
    from src.core.model_manager import model_manager

    before = model_manager.get_memory_stats()
    model_manager.unload_all()
    after = model_manager.get_memory_stats()

    gc.collect()

    return {
        "message": "All models unloaded",
        "before": before,
        "after": after,
    }
