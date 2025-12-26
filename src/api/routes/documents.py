"""
NeuroSynth Unified - Document Routes
=====================================

Document management API endpoints.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path

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
    page: Optional[int] = Query(None, ge=1, description="Filter by page number"),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type"),
    repos = Depends(get_repositories)
):
    """Get chunks for a document."""
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
        
        chunks = await repos.chunks.get_by_document(
            document_id=doc_uuid,
            page_number=page,
            include_embedding=False
        )
        
        # Filter by type if specified
        if chunk_type:
            chunks = [c for c in chunks if c.get('chunk_type') == chunk_type]
        
        return DocumentChunksResponse(
            document_id=document_id,
            chunks=[
                ChunkItem(
                    id=str(c['id']),
                    content=c['content'],
                    page_number=c.get('page_number'),
                    chunk_index=c.get('chunk_index'),
                    chunk_type=c.get('chunk_type'),
                    specialty=c.get('specialty'),
                    cuis=c.get('cuis', [])
                )
                for c in chunks
            ],
            total=len(chunks)
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
    page: Optional[int] = Query(None, ge=1, description="Filter by page number"),
    include_decorative: bool = Query(False, description="Include decorative images"),
    repos = Depends(get_repositories)
):
    """Get images for a document."""
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
        
        images = await repos.images.get_by_document(
            document_id=doc_uuid,
            page_number=page,
            include_decorative=include_decorative,
            include_embedding=False
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
                    "cuis": img.get('cuis', [])
                }
                for img in images
            ],
            "total": len(images)
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
# Delete Document
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
