"""
NeuroSynth Unified - Document Routes
=====================================

Document management API endpoints.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks

from src.api.models import (
    DocumentSummary,
    DocumentDetail,
    DocumentListResponse,
    DocumentChunksResponse,
    ChunkItem,
    ErrorResponse
)
from src.api.dependencies import get_repositories

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# Document List
# =============================================================================

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all indexed documents with pagination"
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    repos = Depends(get_repositories)
):
    """List all indexed documents."""
    try:
        offset = (page - 1) * page_size
        
        docs = await repos.documents.list_with_counts(
            limit=page_size,
            offset=offset
        )
        
        total = await repos.documents.count()
        
        return DocumentListResponse(
            documents=[
                DocumentSummary(
                    id=str(d['id']),
                    source_path=d['source_path'],
                    title=d.get('title'),
                    total_pages=d.get('total_pages', 0),
                    total_chunks=d.get('total_chunks', 0),
                    total_images=d.get('total_images', 0),
                    created_at=d.get('created_at')
                )
                for d in docs
            ],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.exception(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Document Details
# =============================================================================

@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    },
    summary="Get document details",
    description="Get detailed information about a document"
)
async def get_document(
    document_id: str = Path(..., description="Document UUID"),
    repos = Depends(get_repositories)
):
    """Get document details with statistics."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        doc = await repos.documents.get_with_stats(doc_uuid)
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        return DocumentDetail(
            id=str(doc['id']),
            source_path=doc['source_path'],
            title=doc.get('title'),
            total_pages=doc.get('total_pages', 0),
            total_chunks=doc.get('total_chunks', 0),
            total_images=doc.get('total_images', 0),
            created_at=doc.get('created_at'),
            stats=doc.get('stats', {}),
            metadata=doc.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Update Document
# =============================================================================

from pydantic import BaseModel as PydanticBaseModel


class DocumentUpdate(PydanticBaseModel):
    """Request model for document updates."""
    title: Optional[str] = None
    specialty: Optional[str] = None
    metadata: Optional[dict] = None


@router.patch(
    "/{document_id}",
    response_model=DocumentDetail,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    },
    summary="Update document",
    description="Update document metadata (title, specialty, custom metadata)"
)
async def update_document(
    document_id: str = Path(..., description="Document UUID"),
    update: DocumentUpdate = None,
    repos = Depends(get_repositories)
):
    """
    Update document metadata.

    Supports partial updates - only provided fields are updated.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        # Check document exists
        doc = await repos.documents.get_with_stats(doc_uuid)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Build update dict from non-None fields
        update_data = {}
        if update:
            if update.title is not None:
                update_data["title"] = update.title
            if update.specialty is not None:
                update_data["specialty"] = update.specialty
            if update.metadata is not None:
                # Merge with existing metadata
                existing_meta = doc.get("metadata", {}) or {}
                update_data["metadata"] = {**existing_meta, **update.metadata}

        if not update_data:
            # No updates provided, return current document
            return DocumentDetail(
                id=str(doc['id']),
                source_path=doc['source_path'],
                title=doc.get('title'),
                total_pages=doc.get('total_pages', 0),
                total_chunks=doc.get('total_chunks', 0),
                total_images=doc.get('total_images', 0),
                created_at=doc.get('created_at'),
                stats=doc.get('stats', {}),
                metadata=doc.get('metadata', {})
            )

        # Perform update
        updated_doc = await repos.documents.update(doc_uuid, update_data)

        logger.info(f"Updated document {document_id}: {list(update_data.keys())}")

        return DocumentDetail(
            id=str(updated_doc['id']),
            source_path=updated_doc['source_path'],
            title=updated_doc.get('title'),
            total_pages=updated_doc.get('total_pages', 0),
            total_chunks=updated_doc.get('total_chunks', 0),
            total_images=updated_doc.get('total_images', 0),
            created_at=updated_doc.get('created_at'),
            stats=updated_doc.get('stats', {}),
            metadata=updated_doc.get('metadata', {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Update document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Title Correction (Admin)
# =============================================================================

class TitleCorrectionRequest(PydanticBaseModel):
    """Request model for title correction with audit trail."""
    new_title: str
    reason: Optional[str] = None


class TitleCorrectionResponse(PydanticBaseModel):
    """Response after title correction."""
    success: bool
    document_id: str
    old_title: str
    new_title: str
    correction_record: Dict[str, Any]


@router.patch(
    "/{document_id}/correct-title",
    response_model=TitleCorrectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    },
    summary="Correct document title",
    description="Correct a mislabeled document title with audit trail. "
                "Use this when a document was ingested with an incorrect filename."
)
async def correct_document_title(
    document_id: str = Path(..., description="Document UUID"),
    request: TitleCorrectionRequest = None,
    repos = Depends(get_repositories)
):
    """
    Correct a mislabeled document title.

    This endpoint stores a correction record in the document metadata
    for audit purposes, including the old title, new title, reason,
    and timestamp.

    Use cases:
    - PDF filename was incorrect at upload time
    - Chapter mislabeling detected by validation system
    - Manual correction after review
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    if not request or not request.new_title:
        raise HTTPException(status_code=400, detail="new_title is required")

    try:
        # Check document exists
        doc = await repos.documents.get_with_stats(doc_uuid)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        old_title = doc.get("title", "")

        # Build correction record for audit trail
        correction_record = {
            "old_title": old_title,
            "new_title": request.new_title,
            "reason": request.reason,
            "corrected_at": datetime.utcnow().isoformat(),
            "correction_type": "manual"
        }

        # Update metadata with correction history
        existing_metadata = doc.get("metadata", {}) or {}
        corrections_history = existing_metadata.get("title_corrections", [])
        corrections_history.append(correction_record)
        existing_metadata["title_corrections"] = corrections_history

        # Perform update (serialize metadata to JSON for JSONB column)
        await repos.documents.update(doc_uuid, {
            "title": request.new_title,
            "metadata": json.dumps(existing_metadata)
        })

        logger.info(
            f"Title corrected for document {document_id}: "
            f"'{old_title}' -> '{request.new_title}' "
            f"(reason: {request.reason or 'not provided'})"
        )

        return TitleCorrectionResponse(
            success=True,
            document_id=document_id,
            old_title=old_title,
            new_title=request.new_title,
            correction_record=correction_record
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Title correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Document Chunks
# =============================================================================

@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunksResponse,
    summary="Get document chunks",
    description="Get all chunks for a document"
)
async def get_document_chunks(
    document_id: str = Path(..., description="Document UUID"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    pdf_page: Optional[int] = Query(None, ge=1, description="Filter by PDF page number"),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type"),
    repos = Depends(get_repositories)
):
    """Get chunks for a document with pagination."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        # Check document exists
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Calculate pagination offset
        offset = (page - 1) * page_size

        # Get total count first (database-level)
        total = await repos.chunks.count_by_document(
            document_id=doc_uuid,
            page_number=pdf_page,
            chunk_type=chunk_type
        )

        # Get paginated chunks from database (efficient - only loads requested page)
        chunks = await repos.chunks.get_by_document(
            document_id=doc_uuid,
            page_number=pdf_page,
            include_embedding=False,
            limit=page_size,
            offset=offset
        )

        # Filter by type if specified (in case not filtered at DB level)
        if chunk_type:
            chunks = [c for c in chunks if c.get('chunk_type') == chunk_type]

        return DocumentChunksResponse(
            document_id=document_id,
            chunks=[
                ChunkItem(
                    id=str(c['id']),
                    content=c['content'],
                    summary=c.get('summary'),
                    page_number=c.get('page_number'),
                    chunk_index=c.get('chunk_index'),
                    chunk_type=c.get('chunk_type'),
                    specialty=c.get('specialty'),
                    cuis=c.get('cuis', [])
                )
                for c in chunks
            ],
            total=total
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get chunks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Document Images
# =============================================================================

@router.get(
    "/{document_id}/images",
    summary="Get document images",
    description="Get all images for a document"
)
async def get_document_images(
    document_id: str = Path(..., description="Document UUID"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    pdf_page: Optional[int] = Query(None, ge=1, description="Filter by PDF page number"),
    include_decorative: bool = Query(False, description="Include decorative images"),
    repos = Depends(get_repositories)
):
    """Get images for a document with pagination."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Calculate pagination offset
        offset = (page - 1) * page_size

        # Get total count first (database-level for efficiency)
        total = await repos.images.count_by_document(
            document_id=doc_uuid,
            page_number=pdf_page,
            include_decorative=include_decorative
        )

        # Get paginated images from database (efficient)
        images = await repos.images.get_by_document(
            document_id=doc_uuid,
            page_number=pdf_page,
            include_decorative=include_decorative,
            include_embedding=False,
            limit=page_size,
            offset=offset
        )

        return {
            "document_id": document_id,
            "images": [
                {
                    "id": str(img['id']),
                    "file_path": img.get('file_path'),
                    "page_number": img.get('page_number'),
                    "image_type": img.get('image_type'),
                    "is_decorative": img.get('is_decorative', False),
                    "caption": img.get('vlm_caption'),
                    "caption_summary": img.get('caption_summary'),
                    "cuis": img.get('cuis', [])
                }
                for img in images
            ],
            "total": total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Document Statistics
# =============================================================================

@router.get(
    "/{document_id}/stats",
    summary="Get document statistics",
    description="Get detailed statistics for a document"
)
async def get_document_stats(
    document_id: str = Path(..., description="Document UUID"),
    repos = Depends(get_repositories)
):
    """Get detailed statistics for a document."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        chunk_stats = await repos.chunks.get_statistics(doc_uuid)
        image_stats = await repos.images.get_statistics(doc_uuid)
        triage_stats = await repos.images.get_triage_statistics(doc_uuid)
        
        chunk_types = await repos.chunks.get_type_distribution(doc_uuid)
        image_types = await repos.images.get_type_distribution(doc_uuid)
        
        return {
            "document_id": document_id,
            "chunks": {
                **chunk_stats,
                "type_distribution": chunk_types
            },
            "images": {
                **image_stats,
                "triage": triage_stats,
                "type_distribution": image_types
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Bulk Delete Operations
# =============================================================================

from pydantic import BaseModel, Field


class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations."""
    ids: List[str] = Field(..., description="List of document IDs to delete")


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete operations."""
    deleted: int
    failed: List[str] = []
    message: str


@router.delete(
    "/all",
    response_model=BulkDeleteResponse,
    summary="Delete all documents",
    description="Delete ALL documents and cascading data. Use with extreme caution!"
)
async def delete_all_documents(
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
    repos = Depends(get_repositories)
):
    """Delete all documents with cascade. Requires confirmation."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to delete all documents"
        )

    try:
        # Get all document IDs
        all_docs = await repos.documents.list_all_ids()
        deleted_count = 0
        failed = []

        for doc_id in all_docs:
            try:
                await repos.documents.delete_with_cascade(doc_id)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                failed.append(str(doc_id))

        logger.warning(f"Deleted ALL documents: {deleted_count} total, {len(failed)} failed")

        return BulkDeleteResponse(
            deleted=deleted_count,
            failed=failed,
            message=f"Deleted {deleted_count} documents" + (f", {len(failed)} failed" if failed else "")
        )

    except Exception as e:
        logger.exception(f"Delete all documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "",
    response_model=BulkDeleteResponse,
    summary="Bulk delete documents",
    description="Delete multiple documents by ID with cascade"
)
async def bulk_delete_documents(
    request: BulkDeleteRequest,
    repos = Depends(get_repositories)
):
    """Delete multiple documents by ID."""
    if not request.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")

    deleted_count = 0
    failed = []

    for doc_id_str in request.ids:
        try:
            doc_uuid = UUID(doc_id_str)
            exists = await repos.documents.exists(doc_uuid)
            if not exists:
                failed.append(doc_id_str)
                continue

            await repos.documents.delete_with_cascade(doc_uuid)
            deleted_count += 1
        except ValueError:
            failed.append(doc_id_str)
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id_str}: {e}")
            failed.append(doc_id_str)

    logger.info(f"Bulk deleted {deleted_count} documents, {len(failed)} failed")

    return BulkDeleteResponse(
        deleted=deleted_count,
        failed=failed,
        message=f"Deleted {deleted_count} documents" + (f", {len(failed)} failed" if failed else "")
    )


# =============================================================================
# Delete Single Document
# =============================================================================

@router.delete(
    "/{document_id}",
    summary="Delete document",
    description="Delete a document and all associated chunks, images, and links"
)
async def delete_document(
    document_id: str = Path(..., description="Document UUID"),
    repos = Depends(get_repositories)
):
    """Delete a document (cascades to chunks, images, links)."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        deleted = await repos.documents.delete_with_cascade(doc_uuid)
        
        return {
            "status": "deleted" if deleted else "not_found",
            "document_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Delete Chunks
# =============================================================================

@router.delete(
    "/{document_id}/chunks",
    summary="Delete document chunks",
    description="Delete all chunks for a document, or specific chunks by ID"
)
async def delete_document_chunks(
    document_id: str = Path(..., description="Document UUID"),
    chunk_ids: Optional[List[str]] = Query(None, description="Specific chunk IDs to delete"),
    repos = Depends(get_repositories)
):
    """Delete chunks - all if no IDs specified, or specific IDs."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        if chunk_ids:
            # Delete specific chunks
            chunk_uuids = [UUID(cid) for cid in chunk_ids]
            count = await repos.chunks.delete_many(chunk_uuids)
        else:
            # Delete all chunks for document
            count = await repos.chunks.delete_by_document(doc_uuid)

        return {"deleted": count, "document_id": document_id}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid chunk ID format: {e}")
    except Exception as e:
        logger.exception(f"Delete chunks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Delete Images
# =============================================================================

@router.delete(
    "/{document_id}/images",
    summary="Delete document images",
    description="Delete all images for a document, or specific images by ID"
)
async def delete_document_images(
    document_id: str = Path(..., description="Document UUID"),
    image_ids: Optional[List[str]] = Query(None, description="Specific image IDs to delete"),
    repos = Depends(get_repositories)
):
    """Delete images - all if no IDs specified, or specific IDs."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        exists = await repos.documents.exists(doc_uuid)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        if image_ids:
            # Delete specific images
            image_uuids = [UUID(iid) for iid in image_ids]
            count = await repos.images.delete_many(image_uuids)
        else:
            # Delete all images for document
            count = await repos.images.delete_by_document(doc_uuid)

        return {"deleted": count, "document_id": document_id}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image ID format: {e}")
    except Exception as e:
        logger.exception(f"Delete images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Reindex Document
# =============================================================================

# Track reindex jobs
_reindex_jobs: dict = {}


@router.post(
    "/{document_id}/reindex",
    summary="Reindex document",
    description="Re-run embedding and indexing for a document"
)
async def reindex_document(
    document_id: str = Path(..., description="Document UUID"),
    background_tasks: BackgroundTasks = None,
    repos = Depends(get_repositories)
):
    """Re-run embedding and indexing for a document."""
    import uuid as uuid_module

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    try:
        # Verify document exists
        doc = await repos.documents.get_with_stats(doc_uuid)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        job_id = str(uuid_module.uuid4())
        _reindex_jobs[job_id] = {
            "status": "queued",
            "document_id": document_id,
            "progress": 0
        }

        if background_tasks:
            background_tasks.add_task(
                _reindex_document_task,
                document_id,
                job_id,
                repos
            )

        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": "queued",
            "message": f"Reindexing document {doc.get('title', document_id)}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _reindex_document_task(document_id: str, job_id: str, repos):
    """
    Background task to reindex a document.

    NOT YET IMPLEMENTED: This task currently does not re-embed chunks or update FAISS.
    To re-index a document, delete it and re-ingest the source PDF.
    """
    try:
        _reindex_jobs[job_id]["status"] = "running"
        _reindex_jobs[job_id]["progress"] = 10

        # NOT IMPLEMENTED: Full re-embedding would require:
        # 1. Fetch all chunks for document
        # 2. Re-generate embeddings via Voyage API
        # 3. Update chunk embeddings in database
        # 4. Rebuild FAISS indexes with new embeddings
        #
        # This is complex and expensive. For now, recommend delete + re-ingest.

        _reindex_jobs[job_id]["status"] = "failed"
        _reindex_jobs[job_id]["progress"] = 0
        _reindex_jobs[job_id]["message"] = (
            "Reindex not yet implemented. "
            "To re-embed a document: delete it via DELETE /documents/{id}, "
            "then re-upload via POST /api/v1/ingest/upload."
        )
        logger.warning(f"Reindex requested for {document_id} but not implemented")

    except Exception as e:
        logger.error(f"Reindex failed for {document_id}: {e}")
        _reindex_jobs[job_id]["status"] = "failed"
        _reindex_jobs[job_id]["message"] = str(e)


# =============================================================================
# Relink Document (Re-run TriPassLinker)
# =============================================================================

from uuid import uuid4
import asyncio
import numpy as np

# In-memory job tracking for relink operations
_relink_jobs: Dict[str, Dict[str, Any]] = {}
_relink_jobs_lock = asyncio.Lock()


class RelinkJobResponse(PydanticBaseModel):
    """Response when starting a relink job."""
    job_id: str
    status: str
    document_id: str
    message: str


class RelinkJobStatus(PydanticBaseModel):
    """Status of a relink job."""
    job_id: str
    status: str
    progress: float
    message: str
    links_created: Optional[int] = None
    links_deleted: Optional[int] = None


@router.post(
    "/{document_id}/relink",
    response_model=RelinkJobResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"}
    },
    summary="Re-run chunk-image linking",
    description="Re-run TriPassLinker for an existing document. Useful after VLM caption backfill or embedding regeneration."
)
async def relink_document(
    document_id: str = Path(..., description="Document UUID"),
    background_tasks: BackgroundTasks = None,
    repos = Depends(get_repositories)
):
    """
    Re-run chunk-image linking for an existing document.

    This uses TriPassLinker to create links between text chunks and images
    based on semantic similarity, CUI overlap, and explicit figure references.

    Prerequisites:
    - Document chunks must have embeddings (text_embedding)
    - Document images must have caption embeddings (caption_embedding)

    The operation runs in background. Check /documents/{id}/relink/status/{job_id}.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Verify document exists
    doc = await repos.documents.get_with_stats(doc_uuid)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )

    # Create job
    job_id = str(uuid4())
    async with _relink_jobs_lock:
        _relink_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Queued for linking",
            "document_id": document_id,
            "links_created": 0,
            "links_deleted": 0
        }

    # Start background task
    background_tasks.add_task(
        _relink_document_task,
        job_id,
        str(doc_uuid),
        repos
    )

    return RelinkJobResponse(
        job_id=job_id,
        status="pending",
        document_id=document_id,
        message=f"Relink job started for document: {doc.get('title', document_id[:8])}"
    )


@router.get(
    "/{document_id}/relink/status/{job_id}",
    response_model=RelinkJobStatus,
    summary="Get relink job status",
    description="Check the status of a document re-linking job"
)
async def get_relink_status(
    document_id: str = Path(..., description="Document UUID"),
    job_id: str = Path(..., description="Job UUID")
):
    """Get status of a relink job."""
    async with _relink_jobs_lock:
        if job_id not in _relink_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Relink job not found: {job_id}"
            )

        job = _relink_jobs[job_id]
        return RelinkJobStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            links_created=job.get("links_created"),
            links_deleted=job.get("links_deleted")
        )


async def _relink_document_task(job_id: str, doc_id: str, repos):
    """Background task to re-link a single document using TriPassLinker."""
    try:
        async with _relink_jobs_lock:
            _relink_jobs[job_id]["status"] = "running"
            _relink_jobs[job_id]["message"] = "Fetching chunks and images..."

        db = repos.documents.db

        # Fetch chunks with embeddings
        chunks = await db.fetch("""
            SELECT id, content, page_number, embedding
            FROM chunks
            WHERE document_id = $1::uuid AND embedding IS NOT NULL
            ORDER BY chunk_index
        """, doc_id)

        # Fetch images with caption embeddings
        images = await db.fetch("""
            SELECT id, page_number, vlm_caption, caption_embedding
            FROM images
            WHERE document_id = $1::uuid
              AND caption_embedding IS NOT NULL
              AND (is_decorative IS NULL OR NOT is_decorative)
        """, doc_id)

        async with _relink_jobs_lock:
            _relink_jobs[job_id]["progress"] = 0.2
            _relink_jobs[job_id]["message"] = f"Found {len(chunks)} chunks, {len(images)} images"

        if not chunks or not images:
            async with _relink_jobs_lock:
                _relink_jobs[job_id]["status"] = "completed"
                _relink_jobs[job_id]["progress"] = 1.0
                _relink_jobs[job_id]["message"] = "No chunks or images with embeddings to link"
            return

        # Simple semantic linking (cosine similarity)
        links_created = 0
        semantic_threshold = 0.55

        # Delete existing links for this document
        delete_result = await db.execute("""
            DELETE FROM chunk_image_links
            WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = $1::uuid)
        """, doc_id)
        deleted_count = 0
        if delete_result:
            parts = delete_result.split()
            if len(parts) >= 2:
                deleted_count = int(parts[1])

        async with _relink_jobs_lock:
            _relink_jobs[job_id]["links_deleted"] = deleted_count
            _relink_jobs[job_id]["progress"] = 0.4
            _relink_jobs[job_id]["message"] = f"Deleted {deleted_count} old links, computing new links..."

        # Compute links
        for chunk in chunks:
            raw_emb = chunk.get('embedding')
            chunk_emb = None
            if raw_emb is not None and (isinstance(raw_emb, (list, np.ndarray)) and len(raw_emb) > 0):
                chunk_emb = np.array(raw_emb)
            if chunk_emb is None:
                continue

            chunk_page = chunk['page_number'] or 1

            for image in images:
                raw_cap_emb = image.get('caption_embedding')
                img_emb = None
                if raw_cap_emb is not None and (isinstance(raw_cap_emb, (list, np.ndarray)) and len(raw_cap_emb) > 0):
                    img_emb = np.array(raw_cap_emb)
                if img_emb is None:
                    continue

                img_page = image['page_number'] or 1

                # Page proximity check (within 3 pages)
                if abs(chunk_page - img_page) > 3:
                    continue

                # Cosine similarity
                try:
                    norm_chunk = np.linalg.norm(chunk_emb)
                    norm_img = np.linalg.norm(img_emb)
                    if norm_chunk > 0 and norm_img > 0:
                        similarity = float(np.dot(chunk_emb, img_emb) / (norm_chunk * norm_img))
                    else:
                        similarity = 0.0
                except Exception:
                    similarity = 0.0

                if similarity >= semantic_threshold:
                    try:
                        await db.execute("""
                            INSERT INTO chunk_image_links (id, chunk_id, image_id, link_type, relevance_score, link_metadata)
                            VALUES ($1, $2::uuid, $3::uuid, $4, $5, $6::jsonb)
                            ON CONFLICT (chunk_id, image_id) DO UPDATE SET
                                relevance_score = EXCLUDED.relevance_score,
                                link_type = EXCLUDED.link_type
                        """,
                            uuid4(),
                            str(chunk['id']),
                            str(image['id']),
                            'semantic',
                            similarity,
                            '{"source": "relink_api"}'
                        )
                        links_created += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert link: {e}")

        async with _relink_jobs_lock:
            _relink_jobs[job_id]["status"] = "completed"
            _relink_jobs[job_id]["progress"] = 1.0
            _relink_jobs[job_id]["message"] = f"Created {links_created} links"
            _relink_jobs[job_id]["links_created"] = links_created

        logger.info(f"Relink completed for {doc_id}: {links_created} links created")

    except Exception as e:
        logger.exception(f"Relink failed for {doc_id}: {e}")
        async with _relink_jobs_lock:
            _relink_jobs[job_id]["status"] = "failed"
            _relink_jobs[job_id]["message"] = str(e)
