"""
NeuroSynth V3 - Unified RAG API Routes
=======================================

API routes for the Unified RAG Engine with tri-modal query processing.

This module provides ADDITIONAL routes that work alongside existing routes.
It does NOT modify the existing RAG routes.

New Endpoints:
    POST /api/rag/v3/ask         - Ask with automatic mode selection
    POST /api/rag/v3/ask/stream  - Streaming response
    GET  /api/rag/v3/modes       - List available query modes
    GET  /api/rag/v3/health      - Health check for V3 subsystem

Mount in src/api/main.py:
    from src.api.routes import rag_v3
    app.include_router(rag_v3.router)
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_container, get_search_service, ServiceContainer
from src.rag.unified_engine import (
    UnifiedRAGEngine,
    UnifiedRAGConfig,
    QueryMode,
    QueryComplexity,
    UnifiedResponse,
    Citation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag/v3", tags=["rag-v3"])

RAG_TIMEOUT_SECONDS = 120  # Extended for deep research mode


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class UnifiedAskRequest(BaseModel):
    """Request for unified RAG query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question to answer"
    )
    mode: str = Field(
        default="auto",
        description="Query mode: auto, standard, deep_research, external, or hybrid"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific documents to query (for deep research)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for multi-turn dialogue"
    )
    max_chunks: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum chunks for standard mode"
    )
    include_external: bool = Field(
        default=False,
        description="Include external sources in hybrid mode"
    )
    deep_analysis_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Complexity threshold for deep research routing"
    )


class CitationResponse(BaseModel):
    """Citation in the response."""
    index: int
    source_type: str  # internal, external
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str
    url: Optional[str] = None  # For external sources
    authority_score: Optional[float] = None


class UnifiedAskResponse(BaseModel):
    """Response from unified RAG query."""

    answer: str
    mode_used: str
    mode_reason: str
    citations: List[CitationResponse]

    # Query analysis
    complexity: str
    complexity_score: float

    # Performance
    processing_time_ms: int
    tokens_used: int

    # External enrichment (if applicable)
    external_sources_used: int = 0
    gaps_identified: int = 0

    # Conversation
    conversation_id: Optional[str] = None
    turn_number: int = 1


class QueryModeInfo(BaseModel):
    """Information about a query mode."""
    mode: str
    name: str
    description: str
    best_for: List[str]
    cost_tier: str  # low, medium, high
    avg_latency_ms: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine: str
    standard_mode: bool
    deep_research_mode: bool
    external_mode: bool
    gemini_available: bool
    perplexity_available: bool
    pages_table_exists: bool
    timestamp: str


# =============================================================================
# SINGLETON ENGINE
# =============================================================================

_unified_engine: Optional[UnifiedRAGEngine] = None


async def get_unified_engine(
    container: ServiceContainer = Depends(get_container)
) -> UnifiedRAGEngine:
    """Get or create the unified RAG engine using singleton pattern."""
    global _unified_engine

    if _unified_engine is not None:
        return _unified_engine

    try:
        # Get required services from container
        search_service = container.search
        if not search_service:
            raise HTTPException(
                status_code=503,
                detail="Search service not available"
            )

        # Initialize clients
        anthropic_client = None
        gemini_client = None
        perplexity_api_key = None

        # Anthropic client
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            from anthropic import AsyncAnthropic
            anthropic_client = AsyncAnthropic(api_key=anthropic_key)

        # Gemini client (for deep research mode)
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                gemini_client = genai.GenerativeModel("gemini-2.5-pro")
            except ImportError:
                logger.warning("google-generativeai not installed - deep research disabled")

        # Perplexity API key (for external mode)
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

        # Create engine
        _unified_engine = UnifiedRAGEngine(
            search_service=search_service,
            anthropic_client=anthropic_client,
            gemini_client=gemini_client,
            perplexity_api_key=perplexity_api_key,
            database=container.database,
        )

        logger.info("Unified RAG engine initialized")
        return _unified_engine

    except Exception as e:
        logger.error(f"Failed to initialize unified engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG engine: {str(e)}"
        )


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/modes", response_model=List[QueryModeInfo])
async def list_query_modes():
    """
    List available query modes with descriptions.

    Returns information about each mode's purpose, best use cases, and cost.
    """
    return [
        QueryModeInfo(
            mode="auto",
            name="Automatic",
            description="Intelligent routing based on query complexity analysis",
            best_for=["General questions", "Unknown complexity", "Default choice"],
            cost_tier="variable",
            avg_latency_ms=2000,
        ),
        QueryModeInfo(
            mode="standard",
            name="Standard RAG",
            description="Chunk-based retrieval with Claude synthesis",
            best_for=["Factual queries", "Definitions", "Quick lookups"],
            cost_tier="low",
            avg_latency_ms=1500,
        ),
        QueryModeInfo(
            mode="deep_research",
            name="Deep Research",
            description="Full-text analysis with Gemini 2M context",
            best_for=["Cross-document analysis", "Complex comparisons", "Literature review"],
            cost_tier="high",
            avg_latency_ms=15000,
        ),
        QueryModeInfo(
            mode="external",
            name="External Search",
            description="Web search with Perplexity for recent information",
            best_for=["Recent developments", "Current guidelines", "Latest trials"],
            cost_tier="medium",
            avg_latency_ms=3000,
        ),
        QueryModeInfo(
            mode="hybrid",
            name="Hybrid",
            description="Internal corpus + external sources combined",
            best_for=["Comprehensive answers", "Gap filling", "Verification"],
            cost_tier="high",
            avg_latency_ms=5000,
        ),
    ]


@router.post("/ask", response_model=UnifiedAskResponse)
async def unified_ask(
    request: UnifiedAskRequest,
    engine: UnifiedRAGEngine = Depends(get_unified_engine),
):
    """
    Ask a question using the unified RAG engine.

    The engine automatically routes queries to the appropriate processing mode
    based on complexity analysis:

    - **auto**: Let the router decide (recommended)
    - **standard**: Fast chunk-based RAG for simple queries
    - **deep_research**: Full-text Gemini analysis for complex queries
    - **external**: Web search for recent information
    - **hybrid**: Combined internal + external sources

    Example:
    ```json
    {
        "question": "What is the standard dose of dexamethasone for cerebral edema?",
        "mode": "auto"
    }
    ```
    """
    try:
        # Parse mode
        try:
            mode = QueryMode[request.mode.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Valid modes: auto, standard, deep_research, external, hybrid"
            )

        logger.info(f"Unified RAG query: '{request.question[:50]}...' mode={request.mode}")

        # Execute query with timeout
        # Note: UnifiedRAGEngine.ask() uses 'query' not 'question'
        ask_task = engine.ask(
            query=request.question,
            mode=mode,
            document_ids=[str(d) for d in request.document_ids] if request.document_ids else None,
            include_external=request.include_external,
        )

        response: UnifiedResponse = await asyncio.wait_for(
            ask_task,
            timeout=RAG_TIMEOUT_SECONDS
        )

        # Build citation responses
        # Note: unified_engine.Citation uses content_snippet, not snippet
        citations = [
            CitationResponse(
                index=c.index,
                source_type=c.source_type.value if hasattr(c.source_type, 'value') else str(c.source_type),
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                page_number=c.page_number,
                snippet=c.content_snippet,  # unified_engine uses content_snippet
                url=c.url,
                authority_score=getattr(c, 'authority_score', None),  # May not exist
            )
            for c in response.citations
        ]

        # Map UnifiedResponse fields to API response model
        # UnifiedResponse uses different field names
        query_analysis = response.query_analysis
        return UnifiedAskResponse(
            answer=response.answer,
            mode_used=response.mode_used.value,
            mode_reason=query_analysis.reasoning if query_analysis else "auto",
            citations=citations,
            complexity=query_analysis.complexity.value if query_analysis else "unknown",
            complexity_score=query_analysis.confidence if query_analysis else 0.0,
            processing_time_ms=response.total_time_ms,
            tokens_used=0,  # Not tracked in UnifiedResponse
            external_sources_used=response.external_sources,
            gaps_identified=0,  # Not tracked in current implementation
            conversation_id=None,  # Not tracked in current implementation
            turn_number=0,  # Not tracked in current implementation
        )

    except asyncio.TimeoutError:
        logger.error(f"Unified RAG timed out after {RAG_TIMEOUT_SECONDS}s")
        raise HTTPException(
            status_code=504,
            detail=f"Query timed out after {RAG_TIMEOUT_SECONDS} seconds"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unified RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def unified_ask_stream(
    request: UnifiedAskRequest,
    engine: UnifiedRAGEngine = Depends(get_unified_engine),
):
    """
    Stream a response from the unified RAG engine.

    Returns Server-Sent Events (SSE) with progressive answer generation.

    NOTE: V3 streaming is not yet implemented. Use the non-streaming /ask endpoint
    or the existing /api/v1/rag/ask/stream for streaming support.
    """
    # V3 UnifiedRAGEngine does not yet support streaming
    # Return 501 Not Implemented with helpful message
    raise HTTPException(
        status_code=501,
        detail={
            "error": "Streaming not yet implemented for V3 engine",
            "alternatives": [
                "Use POST /api/rag/v3/ask for non-streaming V3 responses",
                "Use POST /api/v1/rag/ask/stream for streaming with V1 engine",
            ],
            "status": "planned_feature"
        }
    )


@router.get("/health", response_model=HealthResponse)
async def unified_rag_health(
    container: ServiceContainer = Depends(get_container),
):
    """
    Health check for the unified RAG subsystem.

    Returns status of all components including:
    - Standard mode (Claude + pgvector)
    - Deep research mode (Gemini + pages table)
    - External mode (Perplexity)
    """
    try:
        # Check if engine exists
        engine_status = "initialized" if _unified_engine else "not initialized"

        # Check modes availability
        standard_available = container.search is not None and os.getenv("ANTHROPIC_API_KEY")
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        deep_research_available = bool(gemini_key)
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        external_available = bool(perplexity_key)

        # Check pages table
        pages_table_exists = False
        if container.database:
            try:
                result = await container.database.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'pages')"
                )
                pages_table_exists = bool(result)
            except Exception:
                pass

        return HealthResponse(
            status="healthy" if standard_available else "degraded",
            engine=engine_status,
            standard_mode=bool(standard_available),
            deep_research_mode=deep_research_available,
            external_mode=external_available,
            gemini_available=deep_research_available,
            perplexity_available=external_available,
            pages_table_exists=pages_table_exists,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            engine="error",
            standard_mode=False,
            deep_research_mode=False,
            external_mode=False,
            gemini_available=False,
            perplexity_available=False,
            pages_table_exists=False,
            timestamp=datetime.utcnow().isoformat(),
        )


@router.get("/complexity")
async def analyze_query_complexity(
    question: str = Query(..., min_length=3, description="Question to analyze"),
    engine: UnifiedRAGEngine = Depends(get_unified_engine),
):
    """
    Analyze query complexity without executing.

    Useful for understanding how the router will classify a query.
    """
    try:
        # Use the router's analyze method
        analysis = await engine.router.analyze(question)

        return {
            "question": question,
            "complexity": analysis.complexity.value,
            "recommended_mode": analysis.recommended_mode.value,
            "entity_count": analysis.entity_count,
            "requires_temporal": analysis.requires_temporal,
            "requires_comparison": analysis.requires_comparison,
            "document_scope": analysis.document_scope,
            "sub_queries": analysis.sub_queries,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
        }

    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
