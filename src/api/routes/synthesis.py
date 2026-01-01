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
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
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


async def get_exporter():
    """Get synthesis exporter instance."""
    from src.synthesis.export import SynthesisExporter, ExportConfig

    config = ExportConfig()
    return SynthesisExporter(
        image_base_path=Path(os.getenv("IMAGE_BASE_PATH", "data/images")),
        config=config
    )


def _sse_event(data) -> str:
    """Format data as Server-Sent Event."""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    return f"data: {json.dumps(data)}\n\n"


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

            # Stage 3: Generation
            engine = await get_synthesis_engine(container)

            result = await asyncio.wait_for(
                engine.synthesize(
                    topic=request.topic,
                    template_type=template_type,
                    search_results=search_response.results,
                    include_verification=request.include_verification,
                    include_figures=request.include_figures
                ),
                timeout=SYNTHESIS_TIMEOUT_SECONDS
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

            # Final result
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
                            "figures": s.figures,
                            "word_count": s.word_count
                        }
                        for s in result.sections
                    ],
                    "resolved_figures": result.resolved_figures,
                    "total_words": result.total_words,
                    "total_figures": result.total_figures,
                    "total_citations": result.total_citations,
                    "synthesis_time_ms": result.synthesis_time_ms
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
