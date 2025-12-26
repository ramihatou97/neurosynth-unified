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
        
        # Convert to response model
        items = []
        for r in result.results:
            items.append(SearchResultItem(
                id=r.id,
                content=r.content,
                score=r.score,
                result_type=r.result_type,
                document_id=r.document_id,
                page_number=r.page_number,
                chunk_type=r.chunk_type,
                specialty=r.specialty,
                image_type=r.image_type,
                cuis=r.cuis or [],
                linked_images=[
                    {
                        "image_id": img.get("image_id"),
                        "caption": img.get("caption"),
                        "link_score": img.get("link_score")
                    }
                    for img in (r.linked_images or [])
                ]
            ))
        
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
        
        items = [
            SearchResultItem(
                id=r.id,
                content=r.content[:500],  # Truncate for quick results
                score=r.score,
                result_type=r.result_type,
                page_number=r.page_number,
                chunk_type=r.chunk_type
            )
            for r in result.results
        ]
        
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
        
        items = [
            SearchResultItem(
                id=r.id,
                content=r.content,
                score=r.score,
                result_type=r.result_type,
                page_number=r.page_number,
                chunk_type=r.chunk_type
            )
            for r in results
        ]
        
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
