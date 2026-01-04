"""
NeuroSynth - Library Ingest Bridge
===================================

Bridges the library scanner to the existing ingest pipeline.
Handles:
- Converting library selections to ingest jobs
- Chapter-level (page range) ingestion via PyMuPDF
- Syncing ingestion status back to the catalog
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import fitz  # PyMuPDF for page extraction
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from .scanner import LibraryCatalog, ReferenceDocument
from .models import ChapterSelection

logger = logging.getLogger(__name__)


# =============================================================================
# Job Store Integration
# =============================================================================

def _get_job_store():
    """Get the global job store from ingest routes."""
    from src.api.routes.ingest import _job_store
    return _job_store


def _get_batch_id() -> str:
    """Generate a unique batch ID."""
    return f"lib-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


# =============================================================================
# Chapter Page Extraction
# =============================================================================

def extract_chapter_pages(
    source_pdf: Path,
    page_start: int,
    page_end: int,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Extract specific pages from a PDF to create a chapter-only PDF.

    Args:
        source_pdf: Path to the source PDF
        page_start: First page (1-indexed)
        page_end: Last page (1-indexed, inclusive)
        output_path: Optional output path. If None, creates temp file.

    Returns:
        Path to the extracted PDF
    """
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF (fitz) required for chapter extraction")

    if output_path is None:
        # Create temp file with meaningful name
        stem = source_pdf.stem
        suffix = f"_pages_{page_start}-{page_end}.pdf"
        fd, temp_path = tempfile.mkstemp(prefix=f"{stem}_", suffix=suffix)
        os.close(fd)
        output_path = Path(temp_path)

    logger.info(f"Extracting pages {page_start}-{page_end} from {source_pdf.name}")

    with fitz.open(source_pdf) as doc:
        # fitz uses 0-indexed pages
        start_idx = max(0, page_start - 1)
        end_idx = min(len(doc), page_end)  # end_idx is exclusive in select()

        # Create new document with selected pages
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx - 1)
        new_doc.save(output_path)
        new_doc.close()

    logger.info(f"Extracted {end_idx - start_idx} pages to {output_path}")
    return output_path


# =============================================================================
# Main Bridge Functions
# =============================================================================

async def create_ingest_jobs(
    catalog: LibraryCatalog,
    document_ids: List[str],
    chapter_selections: List[ChapterSelection],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create ingest jobs for selected documents and chapters.

    Args:
        catalog: The library catalog
        document_ids: List of full document IDs to ingest
        chapter_selections: List of chapter selections (doc_id + chapter_ids)
        config: Optional pipeline config overrides

    Returns:
        Tuple of (batch_id, document_jobs, chapter_jobs)
    """
    job_store = _get_job_store()
    batch_id = _get_batch_id()

    document_jobs = []
    chapter_jobs = []
    config = config or {}

    # Process full document selections
    for doc_id in document_ids:
        doc = catalog.get_document(doc_id)
        if not doc:
            logger.warning(f"Document not found in catalog: {doc_id}")
            continue

        if doc.is_ingested:
            logger.info(f"Document already ingested, skipping: {doc.title}")
            continue

        job_id = f"lib-doc-{uuid.uuid4().hex[:8]}"
        job_store.create_job(job_id, doc.file_name)

        # Queue for processing
        document_jobs.append({
            "job_id": job_id,
            "document_id": doc_id,
            "file_path": str(doc.file_path),
            "file_name": doc.file_name,
            "title": doc.title,
            "type": "full_document",
            "config": config,
        })

        logger.info(f"Queued full document: {doc.title} (job_id={job_id})")

    # Process chapter selections
    for selection in chapter_selections:
        doc = catalog.get_document(selection.document_id)
        if not doc:
            logger.warning(f"Document not found: {selection.document_id}")
            continue

        for chapter_id in selection.chapter_ids:
            # Find chapter in document
            chapter = None
            for ch in doc.chapters:
                if ch.id == chapter_id:
                    chapter = ch
                    break

            if not chapter:
                logger.warning(f"Chapter not found: {chapter_id} in {doc.title}")
                continue

            job_id = f"lib-ch-{uuid.uuid4().hex[:8]}"
            job_store.create_job(job_id, f"{doc.file_name}:{chapter.title}")

            chapter_jobs.append({
                "job_id": job_id,
                "document_id": selection.document_id,
                "chapter_id": chapter_id,
                "source_path": str(doc.file_path),
                "file_name": doc.file_name,
                "title": f"{doc.title} - {chapter.title}",
                "chapter_title": chapter.title,
                "page_start": chapter.page_start,
                "page_end": chapter.page_end,
                "type": "chapter",
                "config": config,
            })

            logger.info(
                f"Queued chapter: {chapter.title} "
                f"(pages {chapter.page_start}-{chapter.page_end}, job_id={job_id})"
            )

    # Start background processing
    if document_jobs or chapter_jobs:
        asyncio.create_task(
            _process_library_batch(batch_id, document_jobs, chapter_jobs)
        )

    return batch_id, document_jobs, chapter_jobs


async def _process_library_batch(
    batch_id: str,
    document_jobs: List[Dict[str, Any]],
    chapter_jobs: List[Dict[str, Any]],
):
    """
    Background task to process queued library items.

    Processes full documents directly, extracts chapter pages first.
    """
    from src.ingest.unified_pipeline import UnifiedPipeline
    from src.ingest.config import UnifiedPipelineConfig
    from src.database.connection import get_connection_string

    job_store = _get_job_store()
    connection_string = get_connection_string()

    logger.info(f"Starting library batch {batch_id}: {len(document_jobs)} docs, {len(chapter_jobs)} chapters")

    # Process full documents
    for job_info in document_jobs:
        job_id = job_info["job_id"]
        try:
            await _process_single_document(
                job_id=job_id,
                pdf_path=Path(job_info["file_path"]),
                title=job_info["title"],
                connection_string=connection_string,
                config=job_info.get("config", {}),
            )
        except Exception as e:
            logger.exception(f"Document job {job_id} failed: {e}")
            job_store.fail_job(job_id, str(e))

    # Process chapters (extract pages first)
    temp_files = []
    try:
        for job_info in chapter_jobs:
            job_id = job_info["job_id"]
            try:
                # Extract chapter pages to temp file
                await job_store.update_job_async(
                    job_id,
                    stage="extraction",
                    progress=5,
                    status="processing",
                    current_operation=f"Extracting pages {job_info['page_start']}-{job_info['page_end']}..."
                )

                chapter_pdf = extract_chapter_pages(
                    source_pdf=Path(job_info["source_path"]),
                    page_start=job_info["page_start"],
                    page_end=job_info["page_end"],
                )
                temp_files.append(chapter_pdf)

                # Process extracted chapter
                await _process_single_document(
                    job_id=job_id,
                    pdf_path=chapter_pdf,
                    title=job_info["title"],
                    connection_string=connection_string,
                    config=job_info.get("config", {}),
                    metadata={
                        "source_document": job_info["file_name"],
                        "chapter_title": job_info.get("chapter_title"),
                        "page_range": f"{job_info['page_start']}-{job_info['page_end']}",
                        "is_chapter_extract": True,
                    },
                )
            except Exception as e:
                logger.exception(f"Chapter job {job_id} failed: {e}")
                job_store.fail_job(job_id, str(e))

    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

    logger.info(f"Library batch {batch_id} completed")


async def _process_single_document(
    job_id: str,
    pdf_path: Path,
    title: str,
    connection_string: str,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Process a single document through the unified pipeline.

    Args:
        job_id: The job ID for progress tracking
        pdf_path: Path to PDF (full doc or extracted chapter)
        title: Document title
        connection_string: Database connection string
        config: Pipeline config overrides
        metadata: Optional metadata to attach
    """
    from src.ingest.unified_pipeline import UnifiedPipeline
    from src.ingest.config import UnifiedPipelineConfig

    job_store = _get_job_store()

    try:
        # Update to processing
        await job_store.update_job_async(
            job_id,
            stage="initializing",
            progress=10,
            status="processing",
            current_operation="Setting up pipeline..."
        )

        # Create pipeline config
        pipeline_config = UnifiedPipelineConfig.for_database(
            connection_string=connection_string,
            enable_ocr=config.get("enable_ocr", True),
            enable_tables=config.get("extract_tables", True),
            enable_knowledge_graph=config.get("enable_knowledge_graph", True),
        )

        # Set API keys from environment
        pipeline_config.embedding.text_api_key = os.getenv("VOYAGE_API_KEY", "")
        pipeline_config.embedding.vlm_api_key = os.getenv("ANTHROPIC_API_KEY", "")

        # Progress callback
        def progress_callback(stage, current=None, total=None, message=None):
            stage_map = {
                "init": ("initializing", 15),
                "structure": ("extraction", 25),
                "pages": ("extraction", 35),
                "images": ("extraction", 40),
                "chunking": ("chunking", 50),
                "text_embedding": ("embedding", 65),
                "image_embedding": ("embedding", 75),
                "vlm_caption": ("vlm", 85),
                "storage": ("database", 95),
            }

            mapped = stage_map.get(str(stage), (str(stage), 50))
            asyncio.create_task(job_store.update_job_async(
                job_id,
                stage=mapped[0],
                progress=mapped[1],
                current_operation=message or f"Processing {stage}..."
            ))

        # Initialize and run pipeline
        pipeline = UnifiedPipeline(config=pipeline_config, on_progress=progress_callback)
        await pipeline.initialize()

        result = await pipeline.process_document(
            pdf_path=pdf_path,
            title=title,
            metadata=metadata,
        )

        await pipeline.close()

        # Check for errors
        if result.error:
            await job_store.fail_job_async(job_id, result.error)
            return

        if result.chunk_count == 0 and result.image_count == 0:
            await job_store.fail_job_async(
                job_id,
                "Pipeline produced 0 chunks and 0 images"
            )
            return

        # Success
        summary = {
            "document_id": str(result.document_id) if result.document_id else None,
            "chunks": result.chunk_count,
            "images": result.image_count,
            "entities": len(getattr(result, 'entities', [])),
            "links": result.link_count,
            "pages": result.total_pages,
        }

        await job_store.complete_job_async(job_id, summary)
        logger.info(f"Job {job_id} completed: {summary}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await job_store.fail_job_async(job_id, str(e))


# =============================================================================
# Catalog Sync
# =============================================================================

async def sync_ingested_documents(catalog: LibraryCatalog) -> int:
    """
    Sync catalog with database to update ingested status.

    Queries the database for documents matching catalog entries
    and updates their is_ingested flag.

    Args:
        catalog: The library catalog to update

    Returns:
        Number of documents updated
    """
    from src.database.connection import get_async_pool

    pool = await get_async_pool()
    updated_count = 0

    async with pool.acquire() as conn:
        # Get all ingested document titles from database
        rows = await conn.fetch("""
            SELECT id, title, file_name, content_hash
            FROM documents
            WHERE status = 'processed'
        """)

        # Build lookup by content_hash and title
        db_docs = {}
        for row in rows:
            # Match by content hash (most reliable)
            if row['content_hash']:
                db_docs[row['content_hash']] = row['id']
            # Also index by title for fallback matching
            db_docs[row['title'].lower()] = row['id']

        # Update catalog entries
        for doc in catalog.documents:
            if doc.is_ingested:
                continue  # Already marked

            # Try to find in database
            doc_id = None

            # Match by content hash
            if doc.content_hash and doc.content_hash in db_docs:
                doc_id = db_docs[doc.content_hash]

            # Fallback: match by title
            if not doc_id and doc.title.lower() in db_docs:
                doc_id = db_docs[doc.title.lower()]

            if doc_id:
                doc.is_ingested = True
                doc.ingested_document_id = str(doc_id)
                doc.ingested_date = datetime.now().isoformat()
                updated_count += 1
                logger.info(f"Marked as ingested: {doc.title}")

    logger.info(f"Sync complete: {updated_count} documents updated")
    return updated_count


async def get_ingestion_progress(batch_id: str) -> Dict[str, Any]:
    """
    Get progress for a library batch.

    Args:
        batch_id: The batch ID returned from create_ingest_jobs

    Returns:
        Batch status with individual job statuses
    """
    job_store = _get_job_store()

    # Find all jobs for this batch (jobs are prefixed with lib-doc- or lib-ch-)
    batch_jobs = []
    for job_id, job in job_store.jobs.items():
        if job_id.startswith("lib-"):
            batch_jobs.append(job.copy())

    # Calculate overall progress
    if not batch_jobs:
        return {
            "batch_id": batch_id,
            "status": "not_found",
            "jobs": [],
            "progress": 0,
        }

    completed = sum(1 for j in batch_jobs if j["status"] == "completed")
    failed = sum(1 for j in batch_jobs if j["status"] == "failed")
    total = len(batch_jobs)

    if completed + failed == total:
        status = "completed" if failed == 0 else "completed_with_errors"
    else:
        status = "processing"

    return {
        "batch_id": batch_id,
        "status": status,
        "jobs": batch_jobs,
        "completed": completed,
        "failed": failed,
        "total": total,
        "progress": (completed + failed) / total * 100 if total > 0 else 0,
    }
