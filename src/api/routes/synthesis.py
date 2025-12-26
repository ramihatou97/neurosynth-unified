"""
Synthesis API Routes

Integrates with existing Phase 1 infrastructure:
- Uses ServiceContainer from src.api.dependencies
- Uses SearchService from src.retrieval.search_service
- SearchResult already has all fields needed for synthesis

Mount in src/api/main.py:
    from src.api.routes import synthesis
    app.include_router(synthesis.router)
"""

import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.synthesis.engine import (
    SynthesisEngine,
    TemplateType,
    SynthesisResult,
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/synthesis", tags=["synthesis"])

SYNTHESIS_TIMEOUT_SECONDS = 120


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SynthesisRequest(BaseModel):
    """Request for content synthesis."""
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


class SynthesisResponse(BaseModel):
    """Complete synthesis response."""
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


class TemplateInfo(BaseModel):
    """Template information."""
    type: str
    description: str
    sections: List[str]
    min_words: int
    min_figures: int


# =============================================================================
# SINGLETON ENGINE
# =============================================================================

_synthesis_engine: Optional[SynthesisEngine] = None


async def get_synthesis_engine(container) -> SynthesisEngine:
    """Get or create SynthesisEngine singleton."""
    global _synthesis_engine
    
    if _synthesis_engine is not None:
        return _synthesis_engine
    
    # Get Anthropic API key from settings
    anthropic_key = getattr(container.settings, 'anthropic_api_key', None)
    if not anthropic_key:
        import os
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not anthropic_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured"
        )
    
    # Initialize Anthropic client
    from anthropic import AsyncAnthropic
    anthropic_client = AsyncAnthropic(api_key=anthropic_key)
    
    # Optional: Gemini for verification
    verification_client = None
    google_key = getattr(container.settings, 'google_api_key', None)
    if not google_key:
        import os
        google_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            verification_client = genai.GenerativeModel("gemini-1.5-pro")
            logger.info("Gemini verification client initialized")
        except ImportError:
            logger.warning("google-generativeai not installed, verification disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
    
    _synthesis_engine = SynthesisEngine(
        anthropic_client=anthropic_client,
        verification_client=verification_client,
    )
    
    logger.info("SynthesisEngine initialized")
    return _synthesis_engine


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
        
        logger.info(f"Starting synthesis: topic='{request.topic}', template={template_type.value}")
        
        # Stage 1: Retrieval using existing SearchService
        # SearchResult ALREADY has all fields needed - no enrichment step!
        search_response = await container.search_service.search(
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
        
        synthesis_task = engine.synthesize(
            topic=request.topic,
            template_type=template_type,
            search_results=search_response.results,  # List[SearchResult] - already complete!
            include_verification=request.include_verification,
            include_figures=request.include_figures,
        )
        
        result = await asyncio.wait_for(
            synthesis_task,
            timeout=SYNTHESIS_TIMEOUT_SECONDS
        )
        
        logger.info(
            f"Synthesis complete: {result.total_words} words, "
            f"{result.total_figures} figures, {result.synthesis_time_ms}ms"
        )
        
        # Convert to response model
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
    return {
        "status": "available",
        "engine_initialized": _synthesis_engine is not None,
        "verification_available": _synthesis_engine.has_verification if _synthesis_engine else False,
    }
