"""
Synthesis API Routes

Integrates with existing Phase 1 infrastructure:
- Uses ServiceContainer from src.api.dependencies
- Uses SearchService from src.retrieval.search_service
- SearchResult already has all fields needed for synthesis

V3 Features (consolidated):
- Optional web research via Perplexity + Gemini (include_web_research=True)
- Gap analysis for knowledge coverage
- External citations tracked separately

Mount in src/api/main.py:
    from src.api.routes import synthesis
    app.include_router(synthesis.router)
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Any, Dict

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.models import ValidationResult  # Enhanced pipeline model

from src.synthesis.engine import (
    SynthesisEngine,
    TemplateType,
    SynthesisResult,
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
)
from src.synthesis.enhanced_engine import (
    EnhancedSynthesisEngine,
    EnhancedSynthesisResult,
    create_enhanced_engine_from_env,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/synthesis", tags=["synthesis"])

SYNTHESIS_TIMEOUT_SECONDS = 300  # Base timeout for synthesis without web research
SYNTHESIS_TIMEOUT_WITH_RESEARCH = 600  # Extended timeout for web research (10 minutes)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    """Request for content synthesis with optional web research."""
    topic: str = Field(..., min_length=3, max_length=500, description="Topic for synthesis")
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
    # Web research options (V3 feature)
    include_web_research: bool = Field(
        default=False,
        description="Enable Open-World web research for recent sources (Perplexity + Gemini)"
    )
    web_research_sections: Optional[List[str]] = Field(
        default=None,
        description="Specific sections to enrich with web research (None = all applicable sections)"
    )
    max_external_queries: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum external API queries per synthesis"
    )
    # 14-Stage Gap Detection (V4 feature)
    gap_fill_strategy: Optional[str] = Field(
        default="high_priority",
        description="Gap filling strategy: 'none' (report only), 'high_priority' (CRITICAL/HIGH gaps), "
                    "'all' (all gaps with fallback), 'external' (always fetch external)"
    )


class SectionResponse(BaseModel):
    """A section in the synthesis."""
    title: str
    content: str
    level: int
    sources: List[str]
    figures: List[dict]
    word_count: int


class ReferenceResponse(BaseModel):
    """Source reference."""
    source: str
    document_id: str
    authority: str
    chunks_used: int


class EnrichmentSummaryResponse(BaseModel):
    """Summary of web research enrichment for a section."""
    section_name: str
    used_external: bool
    gaps_identified: int
    gaps_filled: int
    external_sources: int
    enrichment_time_ms: int


class GapReportResponse(BaseModel):
    """14-stage neurosurgical gap detection report."""
    total_gaps: int = Field(default=0, description="Total gaps detected")
    critical_gaps: int = Field(default=0, description="Safety-critical gaps requiring expert review")
    high_gaps: int = Field(default=0, description="High priority gaps")
    medium_gaps: int = Field(default=0, description="Medium priority gaps")
    low_gaps: int = Field(default=0, description="Low priority gaps")
    safety_flags: List[str] = Field(default_factory=list, description="Safety-critical gap descriptions")
    gaps_by_type: Dict[str, int] = Field(default_factory=dict, description="Gap counts by type")
    gaps_by_priority: Dict[str, int] = Field(default_factory=dict, description="Gap counts by priority")
    subspecialty_detected: Optional[str] = Field(None, description="Detected neurosurgical subspecialty")
    requires_expert_review: bool = Field(default=False, description="True if safety-critical gaps require expert review")
    top_gaps: List[dict] = Field(default_factory=list, description="Top priority gaps with details")


class SynthesisResponse(BaseModel):
    """Complete synthesis response with optional web research enrichment."""
    title: str
    abstract: str
    sections: List[SectionResponse]
    references: List[ReferenceResponse]
    figure_requests: List[dict]
    resolved_figures: List[dict]
    total_words: int
    total_figures: int
    total_citations: int
    synthesis_time_ms: int
    verification_score: Optional[float] = None
    verification_issues: List[str] = []
    verified: bool = False
    conflict_count: int = 0
    conflict_report: Optional[dict] = None
    # Enhanced pipeline fields
    all_cuis: List[str] = Field(default_factory=list, description="All CUIs found across all source chunks")
    chunk_type_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of chunks by type (PROCEDURE, ANATOMY, etc.)")
    validation_result: Optional[ValidationResult] = Field(None, description="Hallucination check results")
    quality_summary: Dict[str, float] = Field(default_factory=dict, description="Aggregate quality stats of used sources")
    # Web research enrichment fields (V3)
    enrichment_used: bool = Field(default=False, description="Whether web research was used")
    enrichment_summary: List[EnrichmentSummaryResponse] = Field(default_factory=list, description="Per-section enrichment details")
    external_citations: List[str] = Field(default_factory=list, description="Citations from external web sources")
    total_external_sources: int = Field(default=0, description="Total external sources consulted")
    gaps_summary: Dict[str, int] = Field(default_factory=dict, description="Summary of knowledge gaps by type")
    enrichment_time_ms: int = Field(default=0, description="Time spent on web research")
    # 14-Stage Gap Detection fields (V4)
    gap_report: Optional[GapReportResponse] = Field(None, description="14-stage neurosurgical gap detection report")
    critical_gap_count: int = Field(default=0, description="Number of safety-critical gaps")
    requires_expert_review: bool = Field(default=False, description="True if safety-critical gaps require expert review")


class TemplateInfo(BaseModel):
    """Template information."""
    type: str
    description: str
    sections: List[str]
    min_words: int
    min_figures: int


class ExportRequest(BaseModel):
    """Request model for export."""
    synthesis_id: Optional[str] = None
    topic: Optional[str] = None
    template_type: str = "PROCEDURAL"
    format: Literal["pdf", "html", "docx", "markdown"] = "pdf"
    author: str = "NeuroSynth"
    image_quality: Literal["high", "medium", "low"] = "high"
    include_toc: bool = True
    include_abstract: bool = True
    include_references: bool = True


class SynthesisProgress(BaseModel):
    """Progress update for streaming."""
    stage: str
    progress: float
    message: str
    section: Optional[str] = None


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

_synthesis_engine: Optional[EnhancedSynthesisEngine] = None


async def get_synthesis_engine(container=None) -> EnhancedSynthesisEngine:
    """Get or create EnhancedSynthesisEngine singleton.

    The enhanced engine supports both standard synthesis and optional
    web research enrichment via Perplexity and Gemini.
    """
    global _synthesis_engine

    if _synthesis_engine is not None:
        return _synthesis_engine

    try:
        # Use factory function that reads from environment
        _synthesis_engine = await create_enhanced_engine_from_env()

        # Log capabilities
        enricher = _synthesis_engine.enricher
        if enricher:
            perplexity_ok = enricher.perplexity and enricher.perplexity.is_available()
            gemini_ok = enricher.gemini and enricher.gemini.is_available()
            logger.info(
                f"EnhancedSynthesisEngine initialized - "
                f"Perplexity: {'✓' if perplexity_ok else '✗'}, "
                f"Gemini: {'✓' if gemini_ok else '✗'}"
            )
        else:
            logger.info("EnhancedSynthesisEngine initialized (no enricher - web research unavailable)")

        return _synthesis_engine

    except Exception as e:
        logger.error(f"Failed to initialize enhanced engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize synthesis engine: {str(e)}"
        )


async def get_exporter():
    """Get synthesis exporter instance."""
    from src.synthesis.export import SynthesisExporter, ExportConfig

    config = ExportConfig()
    return SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )


class SynthesisEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles synthesis-specific objects."""
    def default(self, obj):
        # Handle dataclass instances (like FigureRequest, SynthesisSection)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Fallback for other types
        return str(obj)


def _sse_event(data) -> str:
    """Format data as Server-Sent Event."""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    return f"data: {json.dumps(data, cls=SynthesisEncoder)}\n\n"


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/templates", response_model=List[TemplateInfo])
async def list_templates():
    """List available synthesis templates with their requirements."""
    templates = []
    
    descriptions = {
        "PROCEDURAL": "Operative technique synthesis (Schmidek + Rhoton style)",
        "DISORDER": "Disease-focused synthesis (Youmans + Greenberg style)",
        "ANATOMY": "Neuroanatomy synthesis (Rhoton Microsurgical style)",
        "ENCYCLOPEDIA": "Comprehensive integration (all styles combined)",
    }
    
    for template_type in TemplateType:
        sections = TEMPLATE_SECTIONS.get(template_type, [])
        requirements = TEMPLATE_REQUIREMENTS.get(template_type, {})
        
        templates.append(TemplateInfo(
            type=template_type.value,
            description=descriptions.get(template_type.value, ""),
            sections=[s[0] for s in sections],
            min_words=requirements.get("min_words", 0),
            min_figures=requirements.get("min_figures", 0),
        ))
    
    return templates


@router.post("/generate", response_model=SynthesisResponse)
async def generate_synthesis(request: SynthesisRequest):
    """
    Generate textbook-quality synthesis from NeuroSynth retrieval.
    
    Flow:
    1. SearchService retrieves relevant chunks (already enriched with all fields)
    2. SynthesisEngine generates structured content
    3. FigureResolver matches image placeholders to SearchResult.images
    
    SearchResult already contains:
    - document_id, document_title, chunk_type
    - authority_score, page_start, entity_names
    - images (List[ExtractedImage])
    
    NO ADDITIONAL DATABASE QUERIES NEEDED.
    """
    # Import here to avoid circular imports at module load
    from src.api.dependencies import ServiceContainer
    
    try:
        # Get container (existing Phase 1 DI)
        container = ServiceContainer()
        await container.initialize()
        
        # Validate template type
        try:
            template_type = TemplateType[request.template_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid template_type: {request.template_type}. "
                       f"Must be one of: {[t.value for t in TemplateType]}"
            )
        
        logger.info(
            f"Starting synthesis: topic='{request.topic}', template={template_type.value}, "
            f"web_research={request.include_web_research}"
        )

        # Check if search service is available
        if not container.search:
            raise HTTPException(
                status_code=503,
                detail="Search service not available - cannot generate synthesis. Check VOYAGE_API_KEY configuration."
            )

        # Stage 1: Retrieval using existing SearchService
        # SearchResult ALREADY has all fields needed - no enrichment step!
        search_response = await container.search.search(
            query=request.topic,
            mode="hybrid",
            top_k=request.max_chunks,
            include_images=True,  # Populates SearchResult.images
            rerank=True,
        )

        if not search_response.results:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant content found for topic: {request.topic}"
            )

        logger.info(f"Retrieved {len(search_response.results)} SearchResults")

        # Stage 2+3+4: Synthesis with timeout
        engine = await get_synthesis_engine(container)

        # Configure enricher limits if specified
        if engine.enricher and request.max_external_queries != 5:
            engine.enricher.config.max_external_queries_per_synthesis = request.max_external_queries

        synthesis_task = engine.synthesize(
            topic=request.topic,
            template_type=template_type,
            search_results=search_response.results,  # List[SearchResult] - already complete!
            include_verification=request.include_verification,
            include_figures=request.include_figures,
            include_web_research=request.include_web_research,  # V3: optional web research
            gap_fill_strategy=request.gap_fill_strategy,  # V4: 14-stage gap filling
        )
        
        # Use longer timeout for web research
        timeout = SYNTHESIS_TIMEOUT_WITH_RESEARCH if request.include_web_research else SYNTHESIS_TIMEOUT_SECONDS
        result = await asyncio.wait_for(
            synthesis_task,
            timeout=timeout
        )
        
        logger.info(
            f"Synthesis complete: {result.total_words} words, "
            f"{result.total_figures} figures, {result.synthesis_time_ms}ms"
        )

        # Register synthesis context for chat (non-blocking)
        try:
            from src.chat.routes import get_stores
            from src.chat.engine import SynthesisContext
            from uuid import uuid4

            _, synthesis_store = await get_stores()
            if synthesis_store:
                # Extract chunk IDs from references
                chunk_ids = []
                document_ids = set()
                for ref in result.references:
                    if isinstance(ref, dict):
                        if ref.get('chunk_id'):
                            chunk_ids.append(ref['chunk_id'])
                        if ref.get('document_id'):
                            document_ids.add(ref['document_id'])

                # Also extract from search results used
                for sr in search_response.results[:request.max_chunks]:
                    if hasattr(sr, 'chunk_id') and sr.chunk_id:
                        chunk_ids.append(sr.chunk_id)
                    if hasattr(sr, 'document_id') and sr.document_id:
                        document_ids.add(sr.document_id)

                context = SynthesisContext(
                    synthesis_id=str(uuid4()),
                    topic=request.topic,
                    template_type=request.template_type,
                    chunk_ids=list(set(chunk_ids)),  # Deduplicate
                    document_ids=list(document_ids),
                    section_summaries={
                        s.title: s.content[:200] + "..." if len(s.content) > 200 else s.content
                        for s in result.sections
                    },
                    created_at=datetime.utcnow()
                )
                await synthesis_store.save(context)
                logger.info(f"Registered synthesis context {context.synthesis_id} for chat")
        except Exception as e:
            logger.warning(f"Failed to register synthesis context for chat: {e}")

        # Convert to response model
        # Extract enhanced fields from result context
        context = getattr(result, 'context', {}) or {}

        # Build validation result if available
        validation_result_data = None
        if hasattr(result, 'validation_report') and result.validation_report:
            vr = result.validation_report
            validation_result_data = ValidationResult(
                validated=getattr(vr, 'validated', True),
                hallucination_risk=getattr(vr, 'hallucination_risk', False),
                generated_cuis=getattr(vr, 'generated_cuis', 0),
                source_cuis=getattr(vr, 'source_cuis', 0),
                unsupported_cuis=getattr(vr, 'unsupported_cuis', []),
                issues=getattr(vr, 'issues', []),
            )

        # Build enrichment summaries if available (V3 feature)
        enrichment_summaries = []
        if hasattr(result, 'enrichment_results') and result.enrichment_results:
            for section_name, enrichment in result.enrichment_results.items():
                enrichment_summaries.append(EnrichmentSummaryResponse(
                    section_name=section_name,
                    used_external=enrichment.used_external,
                    gaps_identified=len(enrichment.gaps_identified),
                    gaps_filled=len(enrichment.gaps_filled),
                    external_sources=enrichment.external_source_count,
                    enrichment_time_ms=enrichment.enrichment_time_ms,
                ))

        # Build gap report response if available
        gap_report_response = None
        if hasattr(result, 'gap_report') and result.gap_report:
            gr = result.gap_report
            gap_report_response = GapReportResponse(
                total_gaps=gr.total_gaps,
                critical_gaps=gr.critical_gaps,
                high_gaps=gr.gaps_by_priority.get('high', 0),
                medium_gaps=gr.gaps_by_priority.get('medium', 0),
                low_gaps=gr.gaps_by_priority.get('low', 0),
                safety_flags=gr.safety_flags,
                gaps_by_type=gr.gaps_by_type,
                gaps_by_priority=gr.gaps_by_priority,
                subspecialty_detected=gr.subspecialty_detected,
                requires_expert_review=gr.requires_expert_review,
                top_gaps=[g.to_dict() for g in gr.top_gaps[:10]] if gr.top_gaps else [],
            )

        return SynthesisResponse(
            title=result.title,
            abstract=result.abstract,
            sections=[
                SectionResponse(
                    title=s.title,
                    content=s.content,
                    level=s.level,
                    sources=s.sources,
                    figures=s.figures,
                    word_count=s.word_count,
                )
                for s in result.sections
            ],
            references=[
                ReferenceResponse(**ref) if isinstance(ref, dict) else ReferenceResponse(
                    source=str(ref),
                    document_id="",
                    authority="GENERAL",
                    chunks_used=1,
                )
                for ref in result.references
            ],
            figure_requests=[f.to_dict() for f in result.figure_requests],
            resolved_figures=result.resolved_figures,
            total_words=result.total_words,
            total_figures=result.total_figures,
            total_citations=result.total_citations,
            synthesis_time_ms=result.synthesis_time_ms,
            verification_score=result.verification_score,
            verification_issues=result.verification_issues,
            verified=result.verified,
            conflict_count=result.conflict_count,
            conflict_report=result.conflict_report.to_dict() if result.conflict_report else None,
            # Enhanced pipeline fields
            all_cuis=context.get('all_cuis', []),
            chunk_type_distribution=context.get('chunk_type_distribution', {}),
            validation_result=validation_result_data,
            quality_summary=context.get('quality_summary', {}),
            # V3 enrichment fields
            enrichment_used=getattr(result, 'enrichment_used', False),
            enrichment_summary=enrichment_summaries,
            external_citations=getattr(result, 'external_citations', []),
            total_external_sources=getattr(result, 'total_external_sources', 0),
            gaps_summary=getattr(result, 'gaps_summary', {}),
            enrichment_time_ms=getattr(result, 'enrichment_time_ms', 0),
            # V4 gap detection fields
            gap_report=gap_report_response,
            critical_gap_count=result.critical_gap_count if hasattr(result, 'critical_gap_count') else 0,
            requires_expert_review=result.requires_expert_review if hasattr(result, 'requires_expert_review') else False,
        )
    
    except asyncio.TimeoutError:
        logger.error(f"Synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS}s")
        raise HTTPException(
            status_code=504,
            detail=f"Synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS} seconds. "
                   f"Try reducing max_chunks or using a simpler template."
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis error: {str(e)}"
        )


@router.get("/health")
async def synthesis_health():
    """Check synthesis subsystem health."""
    engine = _synthesis_engine
    enricher = engine.enricher if engine else None
    # Access base engine properties through wrapper
    base_engine = engine._base_engine if engine else None

    return {
        "status": "available",
        "engine_initialized": engine is not None,
        "verification_available": base_engine.has_verification if base_engine else False,
        # V3 enrichment status
        "enrichment_available": enricher is not None,
        "perplexity_available": enricher.perplexity.is_available() if enricher and enricher.perplexity else False,
        "gemini_available": enricher.gemini.is_available() if enricher and enricher.gemini else False,
        # V4 gap detection status
        "gap_detection_available": base_engine.has_gap_detection if base_engine else False,
    }


@router.get("/capabilities", response_model=EnrichmentCapabilities)
async def get_enrichment_capabilities():
    """
    Get current web research enrichment capabilities.

    Returns status of:
    - Perplexity integration (academic web search)
    - Gemini grounding (Google search + reasoning)
    - Gap analysis (knowledge coverage analysis)
    - Configuration settings
    """
    try:
        engine = await get_synthesis_engine()
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


# =============================================================================
# STREAMING SYNTHESIS
# =============================================================================

@router.post("/generate/stream")
async def generate_synthesis_stream(request: SynthesisRequest):
    """
    Generate synthesis with streaming progress updates.

    Returns Server-Sent Events with progress updates and final result.
    """
    from src.api.dependencies import ServiceContainer

    async def event_generator():
        try:
            # Initialize
            container = ServiceContainer()
            await container.initialize()

            # Stage 1: Search
            yield _sse_event(SynthesisProgress(
                stage="search",
                progress=10,
                message="Searching knowledge base..."
            ))

            try:
                template_type = TemplateType[request.template_type.upper()]
            except KeyError:
                yield _sse_event({
                    "stage": "error",
                    "progress": 0,
                    "message": f"Invalid template_type: {request.template_type}"
                })
                return

            # Check if search service is available
            if not container.search:
                yield _sse_event({
                    "stage": "error",
                    "progress": 0,
                    "message": "Search service not available - check VOYAGE_API_KEY configuration"
                })
                return

            search_response = await container.search.search(
                query=request.topic,
                mode="hybrid",
                top_k=request.max_chunks,
                include_images=True
            )

            if not search_response.results:
                yield _sse_event({
                    "stage": "error",
                    "progress": 0,
                    "message": f"No relevant content found for: {request.topic}"
                })
                return

            yield _sse_event(SynthesisProgress(
                stage="search",
                progress=20,
                message=f"Found {len(search_response.results)} relevant chunks"
            ))

            # Stage 2: Context preparation
            yield _sse_event(SynthesisProgress(
                stage="prepare",
                progress=30,
                message="Preparing context..."
            ))

            # Stage 2.5: Web research (if enabled)
            if request.include_web_research:
                yield _sse_event(SynthesisProgress(
                    stage="web_research",
                    progress=35,
                    message="Enriching with web research..."
                ))

            # Stage 3: Generation
            engine = await get_synthesis_engine(container)

            # Configure enricher limits if specified
            if engine.enricher and request.max_external_queries != 5:
                engine.enricher.config.max_external_queries_per_synthesis = request.max_external_queries

            # Use longer timeout for web research
            timeout = SYNTHESIS_TIMEOUT_WITH_RESEARCH if request.include_web_research else SYNTHESIS_TIMEOUT_SECONDS
            result = await asyncio.wait_for(
                engine.synthesize(
                    topic=request.topic,
                    template_type=template_type,
                    search_results=search_response.results,
                    include_verification=request.include_verification,
                    include_figures=request.include_figures,
                    include_web_research=request.include_web_research,  # V3: optional web research
                    gap_fill_strategy=request.gap_fill_strategy,  # V4: 14-stage gap filling
                ),
                timeout=timeout
            )

            # Report sections generated
            for i, section in enumerate(result.sections):
                progress = 40 + (i / max(len(result.sections), 1)) * 40
                yield _sse_event(SynthesisProgress(
                    stage="generate",
                    progress=progress,
                    message=f"Generated: {section.title}",
                    section=section.title
                ))

            # Stage 4: Figure resolution
            yield _sse_event(SynthesisProgress(
                stage="figures",
                progress=85,
                message=f"Resolving {len(result.figure_requests)} figures..."
            ))

            # Stage 5: Finalization
            yield _sse_event(SynthesisProgress(
                stage="finalize",
                progress=95,
                message="Finalizing synthesis..."
            ))

            # Final result - serialize all objects properly
            # Convert FigureRequest objects to dicts
            figure_requests = [
                fr.to_dict() if hasattr(fr, 'to_dict') else {
                    "placeholder_id": getattr(fr, 'placeholder_id', ''),
                    "figure_type": getattr(fr, 'figure_type', ''),
                    "topic": getattr(fr, 'topic', ''),
                    "context": getattr(fr, 'context', ''),
                    "resolved_id": getattr(fr, 'resolved_id', None),
                    "resolved_path": getattr(fr, 'resolved_path', None),
                }
                for fr in (result.figure_requests or [])
            ]

            # resolved_figures should already be dicts, but ensure they are
            resolved_figures = []
            for rf in (result.resolved_figures or []):
                if isinstance(rf, dict):
                    resolved_figures.append(rf)
                elif hasattr(rf, '__dict__'):
                    resolved_figures.append(rf.__dict__)
                else:
                    resolved_figures.append(str(rf))

            # Helper to serialize figures in sections
            def serialize_figures(figs):
                if not figs:
                    return []
                result_figs = []
                for f in figs:
                    if isinstance(f, dict):
                        result_figs.append(f)
                    elif hasattr(f, 'to_dict'):
                        result_figs.append(f.to_dict())
                    elif hasattr(f, '__dict__'):
                        result_figs.append(f.__dict__)
                    else:
                        result_figs.append(str(f))
                return result_figs

            # Build enrichment summaries for streaming response
            enrichment_summaries = []
            if hasattr(result, 'enrichment_results') and result.enrichment_results:
                for section_name, enrichment in result.enrichment_results.items():
                    enrichment_summaries.append({
                        "section_name": section_name,
                        "used_external": enrichment.used_external,
                        "gaps_identified": len(enrichment.gaps_identified),
                        "gaps_filled": len(enrichment.gaps_filled),
                        "external_sources": enrichment.external_source_count,
                        "enrichment_time_ms": enrichment.enrichment_time_ms,
                    })

            # Build gap report for streaming response
            gap_report_data = None
            if hasattr(result, 'gap_report') and result.gap_report:
                gr = result.gap_report
                gap_report_data = {
                    "total_gaps": gr.total_gaps,
                    "critical_gaps": gr.critical_gaps,
                    "safety_flags": gr.safety_flags,
                    "gaps_by_type": gr.gaps_by_type,
                    "gaps_by_priority": gr.gaps_by_priority,
                    "subspecialty_detected": gr.subspecialty_detected,
                    "requires_expert_review": gr.requires_expert_review,
                    "top_gaps": [g.to_dict() for g in gr.top_gaps[:5]] if gr.top_gaps else [],
                }

            yield _sse_event({
                "stage": "complete",
                "progress": 100,
                "result": {
                    "title": result.title,
                    "abstract": result.abstract,
                    "sections": [
                        {
                            "title": s.title,
                            "content": s.content,
                            "level": s.level,
                            "sources": s.sources,
                            "figures": serialize_figures(s.figures),
                            "word_count": s.word_count
                        }
                        for s in result.sections
                    ],
                    "figure_requests": figure_requests,
                    "resolved_figures": resolved_figures,
                    "total_words": result.total_words,
                    "total_figures": result.total_figures,
                    "total_citations": result.total_citations,
                    "synthesis_time_ms": result.synthesis_time_ms,
                    "conflict_count": result.conflict_count,
                    "conflict_report": result.conflict_report.to_dict() if result.conflict_report else None,
                    # V3 enrichment fields
                    "enrichment_used": getattr(result, 'enrichment_used', False),
                    "enrichment_summary": enrichment_summaries,
                    "external_citations": getattr(result, 'external_citations', []),
                    "total_external_sources": getattr(result, 'total_external_sources', 0),
                    "gaps_summary": getattr(result, 'gaps_summary', {}),
                    "enrichment_time_ms": getattr(result, 'enrichment_time_ms', 0),
                    # V4 gap detection fields
                    "gap_report": gap_report_data,
                    "critical_gap_count": result.critical_gap_count if hasattr(result, 'critical_gap_count') else 0,
                    "requires_expert_review": result.requires_expert_review if hasattr(result, 'requires_expert_review') else False,
                }
            })

        except asyncio.TimeoutError:
            yield _sse_event({
                "stage": "error",
                "progress": 0,
                "message": f"Synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS} seconds"
            })
        except Exception as e:
            logger.exception("Streaming synthesis failed")
            yield _sse_event({
                "stage": "error",
                "progress": 0,
                "message": str(e)
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================

@router.post("/export/pdf")
async def export_synthesis_pdf(
    request: ExportRequest,
    background_tasks: BackgroundTasks
):
    """
    Export synthesis to PDF.

    Returns PDF file with:
    - Professional typesetting (ReportLab)
    - Embedded images (native, not base64)
    - Table of contents
    - Clinical callouts (pearl/hazard boxes)
    - Figure captions and references
    """
    from src.synthesis.export import SynthesisExporter, ExportConfig
    from src.api.dependencies import ServiceContainer

    if not request.topic:
        raise HTTPException(status_code=400, detail="Topic required for synthesis")

    # Generate synthesis first
    synth_request = SynthesisRequest(
        topic=request.topic,
        template_type=request.template_type,
        include_figures=True
    )

    synthesis = await generate_synthesis(synth_request)

    # Configure exporter
    config = ExportConfig(
        title=synthesis.title,
        author=request.author,
        image_quality=request.image_quality,
        include_toc=request.include_toc,
        include_abstract=request.include_abstract,
        include_references=request.include_references
    )

    exporter = SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        output_path = Path(tmp.name)

    # Generate PDF
    synthesis_dict = synthesis.model_dump()
    exporter.to_pdf(synthesis_dict, output_path)

    # Schedule cleanup
    def cleanup():
        try:
            os.unlink(output_path)
        except Exception:
            pass

    background_tasks.add_task(cleanup)

    # Return file
    filename = f"{synthesis.title[:50].replace(' ', '_')}.pdf"

    return FileResponse(
        path=output_path,
        filename=filename,
        media_type="application/pdf"
    )


@router.post("/export/html")
async def export_synthesis_html(request: ExportRequest):
    """Export synthesis to self-contained HTML with embedded images."""
    from src.synthesis.export import SynthesisExporter, ExportConfig

    if not request.topic:
        raise HTTPException(status_code=400, detail="Topic required")

    # Generate synthesis
    synth_request = SynthesisRequest(
        topic=request.topic,
        template_type=request.template_type,
        include_figures=True
    )

    synthesis = await generate_synthesis(synth_request)

    # Configure and export
    config = ExportConfig(
        title=synthesis.title,
        author=request.author,
        include_toc=request.include_toc,
        include_abstract=request.include_abstract,
        include_references=request.include_references
    )

    exporter = SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )

    html = exporter.to_html(synthesis.model_dump(), embed_images=True)

    return StreamingResponse(
        iter([html]),
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{synthesis.title[:50]}.html"'
        }
    )


@router.post("/export/docx")
async def export_synthesis_docx(
    request: ExportRequest,
    background_tasks: BackgroundTasks
):
    """Export synthesis to Microsoft Word document."""
    from src.synthesis.export import SynthesisExporter, ExportConfig

    if not request.topic:
        raise HTTPException(status_code=400, detail="Topic required")

    # Generate synthesis
    synth_request = SynthesisRequest(
        topic=request.topic,
        template_type=request.template_type,
        include_figures=True
    )

    synthesis = await generate_synthesis(synth_request)

    # Configure and export
    config = ExportConfig(
        title=synthesis.title,
        author=request.author
    )

    exporter = SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        output_path = Path(tmp.name)

    exporter.to_docx(synthesis.model_dump(), output_path)

    # Schedule cleanup
    background_tasks.add_task(lambda: os.unlink(output_path))

    filename = f"{synthesis.title[:50].replace(' ', '_')}.docx"

    return FileResponse(
        path=output_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@router.post("/export/markdown")
async def export_synthesis_markdown(request: ExportRequest):
    """Export synthesis to Markdown with resolved figure links."""
    from src.synthesis.export import SynthesisExporter, ExportConfig

    if not request.topic:
        raise HTTPException(status_code=400, detail="Topic required")

    # Generate synthesis
    synth_request = SynthesisRequest(
        topic=request.topic,
        template_type=request.template_type,
        include_figures=True
    )

    synthesis = await generate_synthesis(synth_request)

    # Configure and export
    config = ExportConfig(title=synthesis.title, author=request.author)
    exporter = SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )

    markdown = exporter.to_markdown(synthesis.model_dump())

    return StreamingResponse(
        iter([markdown]),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{synthesis.title[:50]}.md"'
        }
    )


# =============================================================================
# GALLERY ENDPOINTS - Persistent Storage for Synthesis Results
# =============================================================================

class GallerySaveRequest(BaseModel):
    """Request to save synthesis to gallery."""
    title: str
    topic: str
    template_type: str = "PROCEDURAL"
    abstract: Optional[str] = None
    sections: List[dict] = []
    source_references: List[dict] = []
    resolved_figures: List[dict] = []
    figure_requests: List[dict] = []
    markdown_content: Optional[str] = None
    total_words: int = 0
    total_figures: int = 0
    total_citations: int = 0
    synthesis_time_ms: int = 0
    verification_score: Optional[float] = None
    verified: bool = False
    conflict_count: int = 0
    conflict_report: Optional[dict] = None
    tags: List[str] = []


class GalleryItemResponse(BaseModel):
    """Gallery item response."""
    id: str
    topic: str
    template_type: str
    title: str
    abstract: Optional[str] = None
    total_words: int
    total_figures: int
    total_citations: int
    synthesis_time_ms: int
    created_at: str
    is_favorite: bool = False
    tags: List[str] = []
    # Preview data
    preview_images: List[dict] = []  # First 3 resolved figures


class GalleryListResponse(BaseModel):
    """Gallery list response."""
    items: List[GalleryItemResponse]
    total: int


class GalleryDetailResponse(GalleryItemResponse):
    """Full gallery item with content."""
    sections: List[dict] = []
    source_references: List[dict] = []
    resolved_figures: List[dict] = []
    figure_requests: List[dict] = []
    markdown_content: Optional[str] = None
    verification_score: Optional[float] = None
    verified: bool = False
    conflict_count: int = 0
    conflict_report: Optional[dict] = None


@router.post("/gallery/save", response_model=GalleryItemResponse)
async def save_to_gallery(request: GallerySaveRequest):
    """
    Save a synthesis result to the gallery for persistent storage.

    This enables:
    - Viewing past syntheses with full content and images
    - Quick preview of resolved figures
    - Favoriting and tagging for organization
    """
    from src.api.dependencies import ServiceContainer

    try:
        container = ServiceContainer()
        await container.initialize()

        # Insert into gallery
        query = """
            INSERT INTO synthesis_gallery (
                topic, template_type, title, abstract,
                sections, source_references, resolved_figures, figure_requests,
                markdown_content, total_words, total_figures, total_citations,
                synthesis_time_ms, verification_score, verified,
                conflict_count, conflict_report, tags
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            RETURNING id, created_at
        """

        row = await container.database.fetchrow(
            query,
            request.topic,
            request.template_type,
            request.title,
            request.abstract,
            json.dumps(request.sections),
            json.dumps(request.source_references),
            json.dumps(request.resolved_figures),
            json.dumps(request.figure_requests),
            request.markdown_content,
            request.total_words,
            request.total_figures,
            request.total_citations,
            request.synthesis_time_ms,
            request.verification_score,
            request.verified,
            request.conflict_count,
            json.dumps(request.conflict_report) if request.conflict_report else None,
            request.tags
        )

        logger.info(f"Saved synthesis to gallery: {row['id']}")

        # Return preview
        preview_images = request.resolved_figures[:3] if request.resolved_figures else []

        return GalleryItemResponse(
            id=str(row['id']),
            topic=request.topic,
            template_type=request.template_type,
            title=request.title,
            abstract=request.abstract,
            total_words=request.total_words,
            total_figures=request.total_figures,
            total_citations=request.total_citations,
            synthesis_time_ms=request.synthesis_time_ms,
            created_at=row['created_at'].isoformat(),
            is_favorite=False,
            tags=request.tags,
            preview_images=preview_images
        )

    except Exception as e:
        logger.exception("Failed to save to gallery")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gallery", response_model=GalleryListResponse)
async def list_gallery(
    limit: int = 20,
    offset: int = 0,
    template_type: Optional[str] = None,
    favorites_only: bool = False,
    search: Optional[str] = None
):
    """
    List saved syntheses in the gallery.

    Supports:
    - Pagination (limit/offset)
    - Filter by template type
    - Filter by favorites
    - Full-text search on topic
    """
    from src.api.dependencies import ServiceContainer

    try:
        container = ServiceContainer()
        await container.initialize()

        # Build query with filters
        conditions = []
        params = []
        param_idx = 1

        if template_type:
            conditions.append(f"template_type = ${param_idx}")
            params.append(template_type)
            param_idx += 1

        if favorites_only:
            conditions.append("is_favorite = TRUE")

        if search:
            conditions.append(f"to_tsvector('english', topic) @@ plainto_tsquery('english', ${param_idx})")
            params.append(search)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Count total
        count_query = f"SELECT COUNT(*) FROM synthesis_gallery {where_clause}"
        total = await container.database.fetchval(count_query, *params)

        # Fetch items
        query = f"""
            SELECT
                id, topic, template_type, title, abstract,
                total_words, total_figures, total_citations,
                synthesis_time_ms, created_at, is_favorite, tags,
                resolved_figures
            FROM synthesis_gallery
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await container.database.fetch(query, *params)

        items = []
        for row in rows:
            resolved_figs = row['resolved_figures'] or []
            if isinstance(resolved_figs, str):
                resolved_figs = json.loads(resolved_figs)

            items.append(GalleryItemResponse(
                id=str(row['id']),
                topic=row['topic'],
                template_type=row['template_type'],
                title=row['title'],
                abstract=row['abstract'],
                total_words=row['total_words'],
                total_figures=row['total_figures'],
                total_citations=row['total_citations'],
                synthesis_time_ms=row['synthesis_time_ms'],
                created_at=row['created_at'].isoformat(),
                is_favorite=row['is_favorite'],
                tags=row['tags'] or [],
                preview_images=resolved_figs[:3]
            ))

        return GalleryListResponse(items=items, total=total)

    except Exception as e:
        logger.exception("Failed to list gallery")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gallery/{synthesis_id}", response_model=GalleryDetailResponse)
async def get_gallery_item(synthesis_id: str):
    """
    Get full details of a saved synthesis including all content and images.
    """
    from src.api.dependencies import ServiceContainer
    from uuid import UUID

    try:
        container = ServiceContainer()
        await container.initialize()

        query = """
            SELECT *
            FROM synthesis_gallery
            WHERE id = $1
        """

        row = await container.database.fetchrow(query, UUID(synthesis_id))

        if not row:
            raise HTTPException(status_code=404, detail="Synthesis not found")

        # Parse JSONB fields
        sections = row['sections'] or []
        source_refs = row['source_references'] or []
        resolved_figs = row['resolved_figures'] or []
        figure_reqs = row['figure_requests'] or []
        conflict_rep = row['conflict_report']

        if isinstance(sections, str):
            sections = json.loads(sections)
        if isinstance(source_refs, str):
            source_refs = json.loads(source_refs)
        if isinstance(resolved_figs, str):
            resolved_figs = json.loads(resolved_figs)
        if isinstance(figure_reqs, str):
            figure_reqs = json.loads(figure_reqs)
        if isinstance(conflict_rep, str):
            conflict_rep = json.loads(conflict_rep)

        return GalleryDetailResponse(
            id=str(row['id']),
            topic=row['topic'],
            template_type=row['template_type'],
            title=row['title'],
            abstract=row['abstract'],
            total_words=row['total_words'],
            total_figures=row['total_figures'],
            total_citations=row['total_citations'],
            synthesis_time_ms=row['synthesis_time_ms'],
            created_at=row['created_at'].isoformat(),
            is_favorite=row['is_favorite'],
            tags=row['tags'] or [],
            preview_images=resolved_figs[:3],
            sections=sections,
            source_references=source_refs,
            resolved_figures=resolved_figs,
            figure_requests=figure_reqs,
            markdown_content=row['markdown_content'],
            verification_score=row['verification_score'],
            verified=row['verified'],
            conflict_count=row['conflict_count'],
            conflict_report=conflict_rep
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get gallery item")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/gallery/{synthesis_id}")
async def delete_gallery_item(synthesis_id: str):
    """Delete a synthesis from the gallery."""
    from src.api.dependencies import ServiceContainer
    from uuid import UUID

    try:
        container = ServiceContainer()
        await container.initialize()

        query = "DELETE FROM synthesis_gallery WHERE id = $1 RETURNING id"
        row = await container.database.fetchrow(query, UUID(synthesis_id))

        if not row:
            raise HTTPException(status_code=404, detail="Synthesis not found")

        logger.info(f"Deleted synthesis from gallery: {synthesis_id}")
        return {"deleted": True, "id": synthesis_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete gallery item")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/gallery/{synthesis_id}/favorite")
async def toggle_favorite(synthesis_id: str, is_favorite: bool = True):
    """Toggle favorite status of a gallery item."""
    from src.api.dependencies import ServiceContainer
    from uuid import UUID

    try:
        container = ServiceContainer()
        await container.initialize()

        query = """
            UPDATE synthesis_gallery
            SET is_favorite = $2
            WHERE id = $1
            RETURNING id, is_favorite
        """
        row = await container.database.fetchrow(query, UUID(synthesis_id), is_favorite)

        if not row:
            raise HTTPException(status_code=404, detail="Synthesis not found")

        return {"id": str(row['id']), "is_favorite": row['is_favorite']}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to toggle favorite")
        raise HTTPException(status_code=500, detail=str(e))
