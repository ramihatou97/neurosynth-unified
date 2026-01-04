"""
NeuroSynth V3 - Enhanced Synthesis API Routes
==============================================

Extended API routes that add Open-World research capability.

This module provides ADDITIONAL routes that work alongside existing routes.
It does NOT modify the existing synthesis routes.

New Endpoints:
    POST /api/synthesis/v3/generate     - Synthesis with optional web research
    GET  /api/synthesis/v3/capabilities - Get enrichment status/capabilities
    POST /api/synthesis/v3/analyze-gaps - Analyze gaps without full synthesis

Mount in src/api/main.py:
    from src.api.routes import synthesis_v3
    app.include_router(synthesis_v3.router)
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_container, ServiceContainer
from src.synthesis.engine import TemplateType, TEMPLATE_SECTIONS
from src.synthesis.enhanced_engine import (
    EnhancedSynthesisEngine,
    EnhancedSynthesisResult,
    create_enhanced_engine_from_env,
)
from src.synthesis.research_enricher import (
    EnrichmentConfig,
    EnrichmentResult,
    KnowledgeGap,
    GapType,
    GapPriority,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/synthesis/v3", tags=["synthesis-v3"])

SYNTHESIS_TIMEOUT_SECONDS = 180  # Extended for web research


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class EnhancedSynthesisRequest(BaseModel):
    """Request for synthesis with optional web research."""

    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Topic for synthesis"
    )
    template_type: str = Field(
        default="PROCEDURAL",
        description="Template type: PROCEDURAL, DISORDER, ANATOMY, or ENCYCLOPEDIA"
    )
    max_chunks: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum source chunks to retrieve"
    )
    include_verification: bool = Field(
        default=False,
        description="Run Gemini verification pass"
    )
    include_figures: bool = Field(
        default=True,
        description="Resolve figure placeholders"
    )
    # NEW V3 FIELDS
    include_web_research: bool = Field(
        default=False,
        description="Enable Open-World web research for recent sources (Perplexity + Gemini)"
    )
    web_research_sections: Optional[List[str]] = Field(
        default=None,
        description="Specific sections to enrich (None = all applicable sections)"
    )
    max_external_queries: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum external API queries per synthesis"
    )


class GapResponse(BaseModel):
    """A knowledge gap identified during analysis."""
    topic: str
    description: str
    gap_type: str
    priority: str
    suggested_query: str
    internal_coverage: str
    external_insight: str
    confidence: float


class EnrichmentSummary(BaseModel):
    """Summary of enrichment for a section."""
    section_name: str
    used_external: bool
    gaps_identified: int
    gaps_filled: int
    external_sources: int
    enrichment_time_ms: int


class EnhancedSynthesisResponse(BaseModel):
    """Response including enrichment metadata."""

    # Standard synthesis fields
    title: str
    abstract: str
    sections: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    total_words: int
    total_figures: int
    total_citations: int
    synthesis_time_ms: int

    # Verification fields
    verification_score: Optional[float] = None
    verification_issues: List[str] = []
    verified: bool = False

    # Conflict fields
    conflict_count: int = 0

    # V3 Enrichment fields
    enrichment_used: bool = False
    enrichment_summary: List[EnrichmentSummary] = []
    external_citations: List[str] = []
    total_external_sources: int = 0
    gaps_summary: Dict[str, int] = {}
    enrichment_time_ms: int = 0


class GapAnalysisRequest(BaseModel):
    """Request for gap analysis only (no full synthesis)."""
    topic: str = Field(..., min_length=3, max_length=500)
    section_name: Optional[str] = None
    max_chunks: int = Field(default=20, ge=5, le=100)


class GapAnalysisResponse(BaseModel):
    """Response from gap analysis."""
    topic: str
    section_name: Optional[str]
    internal_chunk_count: int
    gaps: List[GapResponse]
    external_overview: str
    analysis_time_ms: int


class EnrichmentCapabilities(BaseModel):
    """Current enrichment capabilities status."""
    enabled: bool
    perplexity_available: bool
    gemini_available: bool
    gap_analysis_available: bool
    max_queries_per_synthesis: int
    always_enrich_sections: List[str]


# =============================================================================
# SINGLETON ENGINE
# =============================================================================

_enhanced_engine: Optional[EnhancedSynthesisEngine] = None


async def get_enhanced_engine() -> EnhancedSynthesisEngine:
    """Get or create the enhanced synthesis engine."""
    global _enhanced_engine

    if _enhanced_engine is not None:
        return _enhanced_engine

    try:
        _enhanced_engine = await create_enhanced_engine_from_env()
        logger.info("Enhanced synthesis engine initialized")
        return _enhanced_engine
    except Exception as e:
        logger.error(f"Failed to initialize enhanced engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize synthesis engine: {str(e)}"
        )


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/capabilities", response_model=EnrichmentCapabilities)
async def get_enrichment_capabilities():
    """
    Get current enrichment capabilities.

    Returns status of:
    - Perplexity integration
    - Gemini grounding
    - Gap analysis
    - Configuration settings
    """
    try:
        engine = await get_enhanced_engine()
        enricher = engine.enricher

        if not enricher:
            return EnrichmentCapabilities(
                enabled=False,
                perplexity_available=False,
                gemini_available=False,
                gap_analysis_available=False,
                max_queries_per_synthesis=0,
                always_enrich_sections=[],
            )

        return EnrichmentCapabilities(
            enabled=enricher.config.enabled,
            perplexity_available=enricher.perplexity is not None and enricher.perplexity.is_available(),
            gemini_available=enricher.gemini is not None and enricher.gemini.is_available(),
            gap_analysis_available=enricher.gap_analyzer is not None,
            max_queries_per_synthesis=enricher.config.max_external_queries_per_synthesis,
            always_enrich_sections=enricher.config.always_enrich_sections,
        )

    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=EnhancedSynthesisResponse)
async def generate_enhanced_synthesis(
    request: EnhancedSynthesisRequest,
    container: ServiceContainer = Depends(get_container),
):
    """
    Generate synthesis with optional web research enrichment.

    When `include_web_research=true`:
    1. Each section is analyzed for knowledge gaps
    2. Gaps are filled with Perplexity (academic) and Gemini (grounding)
    3. External sources are cited as [Web: source name]
    4. Internal sources are cited as [Source N]

    Cost Considerations:
    - Web research adds ~2-5 external API calls per synthesis
    - Each call costs ~0.01-0.05 depending on provider
    - Use `max_external_queries` to control costs
    """
    try:
        # Validate template type
        try:
            template_type = TemplateType[request.template_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid template_type: {request.template_type}"
            )

        logger.info(
            f"Starting enhanced synthesis: topic='{request.topic}', "
            f"web_research={request.include_web_research}"
        )

        # Check search service from container (using singleton)
        if not container.search:
            raise HTTPException(
                status_code=503,
                detail="Search service not available"
            )

        # Retrieve chunks
        search_response = await container.search.search(
            query=request.topic,
            mode="hybrid",
            top_k=request.max_chunks,
            include_images=True,
            rerank=True,
        )

        if not search_response.results:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant content found for topic: {request.topic}"
            )

        logger.info(f"Retrieved {len(search_response.results)} chunks")

        # Get enhanced engine
        engine = await get_enhanced_engine()

        # Update enrichment config if custom limits specified
        if engine.enricher and request.max_external_queries != 5:
            engine.enricher.config.max_external_queries_per_synthesis = request.max_external_queries

        # Run synthesis with timeout
        synthesis_task = engine.synthesize(
            topic=request.topic,
            template_type=template_type,
            search_results=search_response.results,
            include_verification=request.include_verification,
            include_figures=request.include_figures,
            include_web_research=request.include_web_research,
        )

        result = await asyncio.wait_for(
            synthesis_task,
            timeout=SYNTHESIS_TIMEOUT_SECONDS
        )

        # Build response
        return _build_response(result)

    except asyncio.TimeoutError:
        logger.error(f"Enhanced synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS}s")
        raise HTTPException(
            status_code=504,
            detail=f"Synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS} seconds"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Enhanced synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-gaps", response_model=GapAnalysisResponse)
async def analyze_knowledge_gaps(
    request: GapAnalysisRequest,
    container: ServiceContainer = Depends(get_container),
):
    """
    Analyze knowledge gaps without running full synthesis.

    Useful for:
    - Pre-synthesis planning
    - Deciding whether web research is needed
    - Understanding corpus coverage

    This endpoint:
    1. Searches internal corpus
    2. Gets external overview
    3. Runs LLM gap analysis
    4. Returns identified gaps (does NOT fill them)
    """
    import time

    start = time.time()

    try:
        if not container.search:
            raise HTTPException(status_code=503, detail="Search service not available")

        # Get internal chunks
        search_response = await container.search.search(
            query=request.topic,
            mode="hybrid",
            top_k=request.max_chunks,
        )

        internal_chunks = [
            {
                "id": r.chunk_id,
                "content": r.content,
                "document_title": r.document_title,
                "authority_score": r.authority_score,
                "entity_names": r.entity_names,
            }
            for r in search_response.results
        ]

        # Get engine and enricher
        engine = await get_enhanced_engine()
        if not engine.enricher or not engine.enricher.gap_analyzer:
            raise HTTPException(
                status_code=503,
                detail="Gap analysis not available - missing API keys"
            )

        # Get external overview
        external_result = await engine.enricher._get_external_overview(
            topic=request.topic,
            section_name=request.section_name,
        )

        # Run gap analysis
        gaps = await engine.enricher.gap_analyzer.analyze_gaps(
            topic=request.topic,
            section_name=request.section_name,
            internal_chunks=internal_chunks,
            external_overview=external_result.summary,
        )

        elapsed = int((time.time() - start) * 1000)

        return GapAnalysisResponse(
            topic=request.topic,
            section_name=request.section_name,
            internal_chunk_count=len(internal_chunks),
            gaps=[
                GapResponse(
                    topic=g.topic,
                    description=g.description,
                    gap_type=g.gap_type.value,
                    priority=g.priority.value,
                    suggested_query=g.suggested_query,
                    internal_coverage=g.internal_coverage,
                    external_insight=g.external_insight,
                    confidence=g.confidence,
                )
                for g in gaps
            ],
            external_overview=external_result.summary[:1000],
            analysis_time_ms=elapsed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gap analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def enhanced_synthesis_health():
    """Health check for enhanced synthesis subsystem."""
    try:
        engine = await get_enhanced_engine()
        enricher = engine.enricher

        return {
            "status": "healthy",
            "engine": "initialized",
            "enricher": "available" if enricher else "not configured",
            "perplexity": enricher.perplexity.is_available() if enricher and enricher.perplexity else False,
            "gemini": enricher.gemini.is_available() if enricher and enricher.gemini else False,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_response(result: EnhancedSynthesisResult) -> EnhancedSynthesisResponse:
    """Convert EnhancedSynthesisResult to API response."""

    # Build enrichment summaries
    enrichment_summaries = []
    for section_name, enrichment in result.enrichment_results.items():
        enrichment_summaries.append(EnrichmentSummary(
            section_name=section_name,
            used_external=enrichment.used_external,
            gaps_identified=len(enrichment.gaps_identified),
            gaps_filled=len(enrichment.gaps_filled),
            external_sources=enrichment.external_source_count,
            enrichment_time_ms=enrichment.enrichment_time_ms,
        ))

    return EnhancedSynthesisResponse(
        title=result.title,
        abstract=result.abstract,
        sections=[
            {
                "title": s.title,
                "content": s.content,
                "level": s.level,
                "sources": s.sources,
                "word_count": s.word_count,
            }
            for s in result.sections
        ],
        references=[
            ref if isinstance(ref, dict) else {"source": str(ref)}
            for ref in result.references
        ],
        total_words=result.total_words,
        total_figures=result.total_figures,
        total_citations=result.total_citations,
        synthesis_time_ms=result.synthesis_time_ms,
        verification_score=result.verification_score,
        verification_issues=result.verification_issues,
        verified=result.verified,
        conflict_count=result.conflict_report.count if result.conflict_report else 0,
        # V3 fields
        enrichment_used=result.enrichment_used,
        enrichment_summary=enrichment_summaries,
        external_citations=result.external_citations,
        total_external_sources=result.total_external_sources,
        gaps_summary=result.gaps_summary,
        enrichment_time_ms=result.enrichment_time_ms,
    )
