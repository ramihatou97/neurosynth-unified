"""
NeuroSynth Unified - Search Routes
===================================

Search API endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.models import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchFilters as SearchFiltersModel,
    ErrorResponse
)
from src.api.dependencies import get_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# =============================================================================
# Helper Functions
# =============================================================================

def convert_to_result_item(result) -> SearchResultItem:
    """
    Convert SearchResult (from search_service) to SearchResultItem (API response).

    Handles the new shared.models.SearchResult structure with synthesis fields.
    """
    # Serialize images to dicts for JSON response
    images_data = []
    if result.images:
        for img in result.images:
            images_data.append({
                'image_id': img.id,
                'file_path': str(img.file_path) if hasattr(img, 'file_path') else '',
                'caption': img.vlm_caption or img.caption or '',
                'image_type': img.image_type,
                'page_number': img.page_number
            })

    return SearchResultItem(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        content=result.content,
        title=result.title,
        chunk_type=result.chunk_type.value if hasattr(result.chunk_type, 'value') else str(result.chunk_type),
        page_start=result.page_start,
        entity_names=result.entity_names,
        image_ids=result.image_ids,
        authority_score=result.authority_score,
        keyword_score=result.keyword_score,
        semantic_score=result.semantic_score,
        final_score=result.final_score,
        document_title=result.document_title,
        cuis=result.cuis,
        images=images_data
    )


# =============================================================================
# Search Endpoints
# =============================================================================

@router.post(
    "",
    response_model=SearchResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Search error"}
    },
    summary="Semantic search",
    description="Search for relevant chunks and images using semantic similarity"
)
async def search(
    request: SearchRequest,
    search_service = Depends(get_search_service)
):
    """
    Perform semantic search over indexed content.
    
    Supports:
    - Text search (chunks only)
    - Image search (images via caption embeddings)
    - Hybrid search (combined text + image)
    
    Filters can be applied for document, chunk type, specialty, and CUIs.
    """
    try:
        # Convert API filters to service filters
        filters = None
        if request.filters:
            from src.retrieval import SearchFilters
            
            page_range = None
            if request.filters.min_page is not None or request.filters.max_page is not None:
                page_range = (
                    request.filters.min_page or 0,
                    request.filters.max_page or 9999
                )
            
            filters = SearchFilters(
                document_ids=request.filters.document_ids,
                chunk_types=request.filters.chunk_types,
                specialties=request.filters.specialties,
                image_types=request.filters.image_types,
                cuis=request.filters.cuis,
                page_range=page_range
            )
        
        # Execute search
        result = await search_service.search(
            query=request.query,
            mode=request.mode.value,
            top_k=request.top_k,
            filters=filters,
            include_images=request.include_images,
            rerank=request.rerank
        )
        
        # Convert to response model using helper function
        items = [convert_to_result_item(r) for r in result.results]
        
        return SearchResponse(
            results=items,
            total_candidates=result.total_candidates,
            query=result.query,
            mode=result.mode,
            search_time_ms=result.search_time_ms
        )
        
    except Exception as e:
        logger.exception(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/quick",
    response_model=SearchResponse,
    summary="Quick search",
    description="Simple GET-based search for autocomplete/quick lookups"
)
async def quick_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
    mode: str = Query("text", pattern="^(text|image|hybrid)$"),
    search_service = Depends(get_search_service)
):
    """Quick search endpoint for simple queries."""
    try:
        result = await search_service.search(
            query=q,
            mode=mode,
            top_k=k,
            include_images=False,
            rerank=False
        )
        
        # Convert to response model, truncating content for quick results
        items = []
        for r in result.results:
            item = convert_to_result_item(r)
            item.content = item.content[:500]  # Truncate for quick results
            items.append(item)
        
        return SearchResponse(
            results=items,
            total_candidates=result.total_candidates,
            query=q,
            mode=mode,
            search_time_ms=result.search_time_ms
        )
        
    except Exception as e:
        logger.exception(f"Quick search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/similar/{item_id}",
    response_model=SearchResponse,
    summary="Find similar items",
    description="Find items similar to a given chunk or image"
)
async def find_similar(
    item_id: str,
    item_type: str = Query("chunk", pattern="^(chunk|image)$"),
    k: int = Query(10, ge=1, le=50),
    search_service = Depends(get_search_service)
):
    """Find items similar to a given item."""
    try:
        results = await search_service.find_similar(
            item_id=item_id,
            item_type=item_type,
            top_k=k
        )
        
        # Convert to response model
        items = [convert_to_result_item(r) for r in results]
        
        return SearchResponse(
            results=items,
            total_candidates=len(results),
            query=f"similar to {item_id}",
            mode="similar",
            search_time_ms=0
        )
        
    except Exception as e:
        logger.exception(f"Similar search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
