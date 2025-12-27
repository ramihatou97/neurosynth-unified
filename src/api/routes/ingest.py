"""
NeuroSynth Unified - Ingest Routes
===================================

Document ingestion API endpoints for PDF upload and processing.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


# =============================================================================
# IN-MEMORY JOB STORE (for demo - use Redis in production)
# =============================================================================

class JobStore:
    """Simple in-memory job store."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

    def create_job(self, job_id: str, filename: str) -> Dict[str, Any]:
        job = {
            "job_id": job_id,
            "filename": filename,
            "stage": "upload",
            "progress": 0,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "summary": None,
            "error": None
        }
        self.jobs[job_id] = job
        return job

    def update_job(self, job_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
            return self.jobs[job_id]
        return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def complete_job(self, job_id: str, summary: Dict[str, Any]):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job["status"] = "completed"
            job["stage"] = "complete"
            job["progress"] = 100
            job["summary"] = summary
            job["updated_at"] = datetime.now().isoformat()
            self.history.insert(0, job.copy())
            if len(self.history) > 50:
                self.history = self.history[:50]

    def fail_job(self, job_id: str, error: str):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job["status"] = "failed"
            job["error"] = error
            job["updated_at"] = datetime.now().isoformat()
            self.history.insert(0, job.copy())

    def cancel_job(self, job_id: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "cancelled"
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.history[:limit]


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
    """Background task to process a document."""
    from src.ingest.unified_pipeline import UnifiedPipeline
    from src.ingest.config import UnifiedPipelineConfig
    from src.database.connection import get_connection_string
    import os

    try:
        # Update to parsing stage
        _job_store.update_job(job_id, stage="parse", progress=10, status="processing")

        # Get database connection
        connection_string = get_connection_string()

        # Create pipeline config using the proper factory method
        pipeline_config = UnifiedPipelineConfig.for_database(
            connection_string=connection_string,
            enable_ocr=True,
            enable_tables=config.get("extract_tables", True),
        )

        # Update embedding config with API keys from environment
        pipeline_config.embedding.text_api_key = os.getenv("VOYAGE_API_KEY", "")
        pipeline_config.embedding.vlm_api_key = os.getenv("ANTHROPIC_API_KEY", "")

        # Create and initialize pipeline
        pipeline = UnifiedPipeline(config=pipeline_config)
        await pipeline.initialize()

        # Stage updates during processing
        stages = [
            ("images", 20),
            ("chunk", 35),
            ("entities", 50),
            ("linking", 60),
            ("embed_text", 70),
            ("vlm", 80),
            ("embed_images", 85),
            ("graph", 90),
            ("database", 95),
        ]

        # Process with progress updates
        _job_store.update_job(job_id, stage="parse", progress=15)

        result = await pipeline.process_document(
            pdf_path=file_path,
            title=title
        )

        # Close pipeline
        await pipeline.close()

        # Update progress through stages (simplified)
        for stage, progress in stages:
            _job_store.update_job(job_id, stage=stage, progress=progress)
            await asyncio.sleep(0.1)  # Small delay for UI updates

        # Complete
        summary = {
            "document_id": str(result.document_id) if result.document_id else None,
            "chunks": result.chunk_count,
            "images": result.image_count,
            "entities": len(getattr(result, 'entities', [])),
            "links": result.link_count,
            "pages": result.total_pages
        }

        _job_store.complete_job(job_id, summary)
        logger.info(f"Job {job_id} completed: {summary}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        _job_store.fail_job(job_id, str(e))

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
