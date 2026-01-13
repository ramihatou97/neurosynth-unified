"""
NeuroSynth Unified - Unified RAG API Routes
============================================

V2 RAG API endpoints with external search integration.

Endpoints:
- POST /api/v2/rag/ask: Ask with mode selection
- POST /api/v2/rag/ask/stream: Streaming response with mode
- GET /api/v2/rag/modes: List available search modes

These endpoints extend the existing /api/v1/rag/* endpoints with:
- Mode selection (standard, hybrid, deep_research, external)
- Dual-source citations (internal vs external)
- Gap analysis reports
- Source composition metrics
"""

import logging
import json
import time
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/rag", tags=["Unified RAG"])


# =============================================================================
# Request/Response Models
# =============================================================================

class RAGFilters(BaseModel):
    """Filters for internal search."""
    document_ids: Optional[List[str]] = None
    chunk_types: Optional[List[str]] = None
    specialties: Optional[List[str]] = None
    cuis: Optional[List[str]] = None
    min_page: Optional[int] = None
    max_page: Optional[int] = None


class UnifiedRAGRequest(BaseModel):
    """Request model for unified RAG with mode selection."""
    question: str = Field(..., description="Question to answer")
    mode: Optional[str] = Field(
        "hybrid",
        description="Search mode: standard, hybrid, deep_research, external"
    )
    filters: Optional[RAGFilters] = Field(
        None,
        description="Filters for internal search"
    )
    include_citations: bool = Field(True, description="Include citation tracking")
    include_images: bool = Field(True, description="Include related images")
    include_gap_analysis: bool = Field(True, description="Include gap analysis")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the current guidelines for DBS infection management?",
                "mode": "hybrid",
                "include_citations": True,
                "include_gap_analysis": True
            }
        }


class InternalCitationItem(BaseModel):
    """Internal citation from database."""
    index: int
    chunk_id: str
    snippet: str
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    source_document: Optional[str] = None
    authority_score: Optional[float] = None


class ExternalCitationItem(BaseModel):
    """External citation from web search."""
    index: str  # "W1", "W2", etc.
    source_url: str
    source_title: str
    snippet: str
    provider: str  # "perplexity", "gemini"
    publication_date: Optional[str] = None


class ConflictItem(BaseModel):
    """Conflict between internal and external sources."""
    topic: str
    internal_claim: str
    external_claim: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    recommendation: Optional[str] = None


class GapReportResponse(BaseModel):
    """
    Gap analysis report from the "Judge" architecture.

    Compares GROUND TRUTH (user's database) vs WORLD TRUTH (current guidelines).
    """
    agreements: List[str] = Field(
        default=[],
        description="Points where internal and external sources agree (validates internal data)"
    )
    conflicts: List[ConflictItem] = Field(
        default=[],
        description="Points where internal contradicts external (potential clinical risks)"
    )
    gaps: List[str] = Field(
        default=[],
        description="Information in external sources missing from internal database"
    )
    unique_internal: List[str] = Field(
        default=[],
        description="Specialized information in internal notes not found externally"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the gap analysis (0.0-1.0)"
    )
    alert_level: str = Field(
        default="none",
        description="Overall alert level: none, caution, warning, critical"
    )
    has_conflicts: bool = Field(
        default=False,
        description="Whether any conflicts were detected"
    )
    has_critical_conflicts: bool = Field(
        default=False,
        description="Whether any CRITICAL severity conflicts exist"
    )
    has_high_severity_conflicts: bool = Field(
        default=False,
        description="Whether HIGH or CRITICAL severity conflicts exist"
    )
    conflict_count: int = Field(
        default=0,
        description="Total number of conflicts detected"
    )
    gap_count: int = Field(
        default=0,
        description="Total number of knowledge gaps identified"
    )
    highest_severity: Optional[str] = Field(
        default=None,
        description="Highest conflict severity: low, medium, high, critical"
    )


class ImageItem(BaseModel):
    """Image metadata."""
    image_id: str
    file_path: str  # Coerced from PosixPath via validator
    caption: Optional[str] = None  # Can be None if VLM captioning not run
    image_type: Optional[str] = None

    @field_validator('file_path', mode='before')
    @classmethod
    def coerce_path_to_str(cls, v):
        """Convert PosixPath to string."""
        return str(v) if v is not None else ""


class UnifiedRAGResponse(BaseModel):
    """Response from unified RAG with dual citations."""
    answer: str
    question: str

    # Dual citations
    internal_citations: List[InternalCitationItem] = []
    external_citations: List[ExternalCitationItem] = []

    # Gap analysis
    gap_report: Optional[GapReportResponse] = None

    # Mode info
    mode_used: str
    internal_ratio: float = Field(
        ...,
        description="Ratio of internal to total sources (0.0-1.0)"
    )

    # Images
    images: List[ImageItem] = []

    # Timing
    search_time_ms: int = 0
    generation_time_ms: int = 0
    total_time_ms: int = 0

    # Model
    model: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Current guidelines for DBS infection management [1] recommend...",
                "question": "What are the current guidelines for DBS infection management?",
                "internal_citations": [
                    {
                        "index": 1,
                        "chunk_id": "chunk-123",
                        "snippet": "DBS infection rates range from...",
                        "source_document": "Greenberg"
                    }
                ],
                "external_citations": [
                    {
                        "index": "W1",
                        "source_url": "https://uptodate.com/...",
                        "source_title": "Deep brain stimulation: Complications",
                        "snippet": "2024 guidelines recommend...",
                        "provider": "perplexity"
                    }
                ],
                "mode_used": "hybrid",
                "internal_ratio": 0.67
            }
        }


class ModeInfoResponse(BaseModel):
    """Information about a search mode."""
    id: str
    name: str
    description: str
    icon: str
    cost: str
    latency: str
    available: bool = True
    requires_api_key: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# Query Analysis Models (from V3)
# =============================================================================

class QueryComplexityRequest(BaseModel):
    """Request for query complexity analysis."""
    query: str = Field(..., description="Query to analyze")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to consider")


class QueryComplexityResponse(BaseModel):
    """Response with query complexity analysis."""
    complexity: str = Field(..., description="Complexity level: simple, moderate, complex, research")
    recommended_mode: str = Field(..., description="Recommended search mode")
    entity_count: int = Field(0, description="Estimated medical entity count")
    requires_temporal: bool = Field(False, description="Whether query needs recent information")
    requires_comparison: bool = Field(False, description="Whether query involves comparison")
    sub_queries: List[str] = Field(default=[], description="Decomposed sub-queries for complex questions")
    confidence: float = Field(0.0, description="Confidence in routing decision")
    reasoning: str = Field("", description="Explanation for the routing decision")


# =============================================================================
# V1 Feature Models (Summarize, Compare, Context)
# =============================================================================

class SummarizeResponse(BaseModel):
    """Document summary response."""
    document_id: str
    title: str
    summary: str
    word_count: int
    generation_time_ms: int


class CompareRequest(BaseModel):
    """Request for comparing surgical approaches."""
    approach_a: str = Field(..., description="First approach to compare")
    approach_b: str = Field(..., description="Second approach to compare")
    aspects: Optional[List[str]] = Field(
        default=["indications", "technique", "complications", "outcomes"],
        description="Aspects to compare"
    )
    max_chunks: int = Field(30, description="Maximum context chunks")


class CompareResponse(BaseModel):
    """Comparison response."""
    approach_a: str
    approach_b: str
    comparison: str
    aspects_covered: List[str]
    citations: List[InternalCitationItem]
    generation_time_ms: int


class ConversationHistoryItem(BaseModel):
    """Single conversation entry."""
    id: str
    created_at: float
    updated_at: float
    mode: str
    message_count: int


class ConversationHistoryResponse(BaseModel):
    """List of conversations."""
    conversations: List[ConversationHistoryItem]
    total: int
    page: int
    page_size: int


class ClearConversationsRequest(BaseModel):
    """Request to clear conversations."""
    before: Optional[float] = Field(None, description="Clear conversations before this timestamp")
    conversation_ids: Optional[List[str]] = Field(None, description="Specific conversation IDs to clear")


class ContextChunk(BaseModel):
    """Context chunk for RAG."""
    index: int
    content: str
    chunk_id: str
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    score: float = 0.0


class ContextResponse(BaseModel):
    """RAG context without answer."""
    query: str
    chunks: List[ContextChunk]
    total_chunks: int
    search_time_ms: int


# =============================================================================
# Dependencies
# =============================================================================

async def get_unified_rag_engine():
    """Get unified RAG engine from container."""
    from src.api.dependencies import ServiceContainer

    container = ServiceContainer.get_instance()
    if not container._initialized:
        await container.initialize()

    # Check if unified engine exists
    if hasattr(container, '_unified_rag') and container._unified_rag:
        return container._unified_rag

    # Try to create it from existing components
    if container._rag:
        from src.rag.unified_rag_engine import UnifiedRAGEngine
        from src.research import create_enricher_from_env
        import os

        # Get anthropic client if available
        anthropic_client = None
        if hasattr(container._rag, '_async_client'):
            anthropic_client = container._rag._async_client

        # Create client directly if not found but API key exists
        if not anthropic_client and os.getenv('ANTHROPIC_API_KEY'):
            import anthropic
            anthropic_client = anthropic.AsyncAnthropic()

        # Create enricher
        enricher = None
        if container._search:
            enricher = create_enricher_from_env(
                container._search,
                anthropic_client
            )

        # Create unified engine
        container._unified_rag = UnifiedRAGEngine(
            rag_engine=container._rag,
            enricher=enricher,
            anthropic_client=anthropic_client
        )

        return container._unified_rag

    raise HTTPException(
        status_code=503,
        detail="Unified RAG service not available"
    )


async def get_conversation_store():
    """Get conversation store."""
    from src.api.dependencies import ConversationStore
    return ConversationStore.get_instance()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/ask",
    response_model=UnifiedRAGResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Generation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    summary="Ask with mode selection",
    description="Ask a question with search mode selection for internal/external search integration"
)
async def ask_unified(
    request: UnifiedRAGRequest,
    engine = Depends(get_unified_rag_engine),
    conversation_store = Depends(get_conversation_store)
):
    """
    Ask a question using unified RAG with mode selection.

    Modes:
    - **standard**: Internal database only (fastest, $0)
    - **hybrid**: Internal + Perplexity web search (recommended)
    - **deep_research**: Internal + Gemini reasoning (complex queries)
    - **external**: Web search only (non-corpus questions)

    The response includes:
    - Answer with dual citations ([N] for internal, [WN] for external)
    - Gap analysis identifying conflicts between sources
    - Source composition metrics
    """
    start_time = time.time()

    try:
        # Parse mode
        from src.research.models import SearchMode
        try:
            mode = SearchMode(request.mode) if request.mode else SearchMode.HYBRID
        except ValueError:
            mode = SearchMode.HYBRID

        # Build filters if provided
        filters = None
        if request.filters:
            from src.retrieval import SearchFilters

            page_range = None
            if request.filters.min_page or request.filters.max_page:
                page_range = (
                    request.filters.min_page or 0,
                    request.filters.max_page or 9999
                )

            filters = SearchFilters(
                document_ids=request.filters.document_ids,
                chunk_types=request.filters.chunk_types,
                specialties=request.filters.specialties,
                cuis=request.filters.cuis,
                page_range=page_range
            )

        # Get conversation history if conversation_id provided
        conversation_history = None
        if request.conversation_id:
            conv = conversation_store.get(request.conversation_id)
            if conv:
                conversation_history = conv.get('history', [])

        # Execute unified RAG
        result = await engine.ask(
            question=request.question,
            mode=mode,
            filters=filters,
            include_citations=request.include_citations,
            include_images=request.include_images,
            conversation_history=conversation_history
        )

        # Convert to response model
        internal_citations = [
            InternalCitationItem(
                index=c.index,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_type=c.chunk_type.value if hasattr(c.chunk_type, 'value') else str(c.chunk_type) if c.chunk_type else None
            )
            for c in result.internal_citations
        ]

        external_citations = [
            ExternalCitationItem(
                index=c.index,
                source_url=c.source_url,
                source_title=c.source_title,
                snippet=c.snippet,
                provider=c.provider
            )
            for c in result.external_citations
        ]

        # Convert gap report
        gap_report = None
        if result.gap_report:
            gap_report = GapReportResponse(
                agreements=result.gap_report.agreements,
                conflicts=[
                    ConflictItem(
                        topic=c.topic,
                        internal_claim=c.internal_claim,
                        external_claim=c.external_claim,
                        severity=c.severity.value,
                        recommendation=c.recommendation
                    )
                    for c in result.gap_report.conflicts
                ],
                gaps=result.gap_report.gaps,
                unique_internal=result.gap_report.unique_internal,
                confidence=result.gap_report.confidence,
                alert_level=result.gap_report.alert_level or result.gap_report.computed_alert_level,
                has_conflicts=result.gap_report.has_conflicts,
                has_critical_conflicts=result.gap_report.has_critical_conflicts,
                has_high_severity_conflicts=result.gap_report.has_high_severity_conflicts,
                conflict_count=result.gap_report.conflict_count,
                gap_count=result.gap_report.gap_count,
                highest_severity=result.gap_report.highest_severity.value if result.gap_report.highest_severity else None
            )

        # Convert images
        images = [
            ImageItem(
                image_id=getattr(img, 'image_id', str(i)),
                file_path=getattr(img, 'file_path', ''),
                caption=getattr(img, 'caption', ''),
                image_type=getattr(img, 'image_type', None)
            )
            for i, img in enumerate(result.images)
        ]

        total_time = int((time.time() - start_time) * 1000)

        # Update conversation if ID provided
        if request.conversation_id:
            conv = conversation_store.get(request.conversation_id) or {'history': []}
            conv['history'].append({"role": "user", "content": request.question})
            conv['history'].append({"role": "assistant", "content": result.answer})
            conversation_store.set(request.conversation_id, conv)

        return UnifiedRAGResponse(
            answer=result.answer,
            question=result.question,
            internal_citations=internal_citations,
            external_citations=external_citations,
            gap_report=gap_report,
            mode_used=result.mode_used.value,
            internal_ratio=result.internal_ratio,
            images=images,
            search_time_ms=result.search_time_ms,
            generation_time_ms=result.generation_time_ms,
            total_time_ms=total_time,
            model=result.model
        )

    except Exception as e:
        logger.exception(f"Unified RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ask/stream",
    summary="Ask with streaming response",
    description="Ask a question and stream the response tokens"
)
async def ask_unified_stream(
    request: UnifiedRAGRequest,
    engine = Depends(get_unified_rag_engine)
):
    """
    Ask a question with streaming response.

    Returns Server-Sent Events (SSE) stream with:
    - Token chunks as they're generated
    - Final metadata packet with citations and gap report
    """
    from src.research.models import SearchMode

    try:
        mode = SearchMode(request.mode) if request.mode else SearchMode.HYBRID
    except ValueError:
        mode = SearchMode.HYBRID

    async def event_generator():
        """Generate SSE events."""
        try:
            # Get streaming response
            stream = await engine.ask(
                question=request.question,
                mode=mode,
                include_citations=request.include_citations,
                stream=True
            )

            async for event in stream:
                if event["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': event['content']})}\n\n"
                elif event["type"] == "done":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get(
    "/modes",
    response_model=List[ModeInfoResponse],
    summary="List available search modes",
    description="Get information about available search modes and their requirements"
)
async def list_modes(
    engine = Depends(get_unified_rag_engine)
):
    """
    List available search modes with descriptions.

    Returns information about each mode including:
    - Name and description
    - Cost and latency estimates
    - Required API keys
    - Availability status
    """
    modes = engine.get_available_modes()

    return [
        ModeInfoResponse(
            id=m.id,
            name=m.name,
            description=m.description,
            icon=m.icon,
            cost=m.cost,
            latency=m.latency,
            available=True,
            requires_api_key=m.requires_api_key
        )
        for m in modes
    ]


@router.get(
    "/health",
    summary="Check unified RAG health",
    description="Check the health of unified RAG components"
)
async def check_health():
    """Check health of unified RAG components."""
    from src.api.dependencies import ServiceContainer

    container = ServiceContainer.get_instance()

    status = {
        "internal_rag": container._rag is not None,
        "search_service": container._search is not None,
        "unified_rag": hasattr(container, '_unified_rag') and container._unified_rag is not None
    }

    # Check external search availability
    if hasattr(container, '_unified_rag') and container._unified_rag:
        if container._unified_rag.enricher:
            status["perplexity"] = container._unified_rag.enricher.has_perplexity
            status["gemini"] = container._unified_rag.enricher.has_gemini
            status["external_search"] = container._unified_rag.has_external

    return {
        "status": "healthy" if status.get("unified_rag") else "degraded",
        "components": status
    }


# =============================================================================
# Conversation Management (Optional)
# =============================================================================

@router.post(
    "/conversation",
    summary="Create new conversation",
    description="Create a new multi-turn conversation"
)
async def create_conversation(
    mode: str = Query("hybrid", description="Default search mode for conversation"),
    conversation_store = Depends(get_conversation_store)
):
    """Create a new conversation with specified default mode."""
    conversation_id = str(uuid4())

    conversation_store.set(conversation_id, {
        "history": [],
        "mode": mode,
        "created_at": time.time()
    })

    return {
        "conversation_id": conversation_id,
        "mode": mode
    }


@router.delete(
    "/conversation/{conversation_id}",
    summary="Delete conversation",
    description="Delete a conversation and its history"
)
async def delete_conversation(
    conversation_id: str,
    conversation_store = Depends(get_conversation_store)
):
    """Delete a conversation."""
    conversation_store.delete(conversation_id)
    return {"deleted": True}


# =============================================================================
# Query Complexity Analysis (from V3)
# =============================================================================

@router.post(
    "/complexity",
    response_model=QueryComplexityResponse,
    summary="Analyze query complexity",
    description="Analyze a query to determine optimal search mode and complexity"
)
async def analyze_complexity(
    request: QueryComplexityRequest,
    engine = Depends(get_unified_rag_engine)
):
    """
    Analyze query complexity and get recommended search mode.

    Returns:
    - Complexity level (simple, moderate, complex, research)
    - Recommended search mode
    - Entity count and temporal requirements
    - Sub-queries for complex questions
    - Reasoning for the routing decision
    """
    try:
        analysis = engine.analyze_complexity(
            query=request.query,
            document_ids=request.document_ids
        )

        return QueryComplexityResponse(
            complexity=analysis.complexity.value,
            recommended_mode=analysis.recommended_mode.value,
            entity_count=analysis.entity_count,
            requires_temporal=analysis.requires_temporal,
            requires_comparison=analysis.requires_comparison,
            sub_queries=analysis.sub_queries,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning
        )

    except Exception as e:
        logger.exception(f"Complexity analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# V1 Feature Endpoints (Summarize, Compare, Context, History, Clear)
# =============================================================================

@router.get(
    "/summarize/{document_id}",
    response_model=SummarizeResponse,
    summary="Summarize document",
    description="Generate a summary of a specific document"
)
async def summarize_document(
    document_id: str,
    max_length: int = Query(500, description="Maximum summary length in words"),
    engine = Depends(get_unified_rag_engine)
):
    """
    Generate a summary of a document.

    Uses the internal RAG engine to summarize document content.
    """
    start_time = time.time()

    try:
        # Get internal engine
        internal = engine.internal_engine

        # Call internal summarize if available
        if hasattr(internal, 'summarize_document'):
            result = await internal.summarize_document(document_id, max_length)
            return SummarizeResponse(
                document_id=document_id,
                title=result.get('title', 'Unknown'),
                summary=result.get('summary', ''),
                word_count=len(result.get('summary', '').split()),
                generation_time_ms=int((time.time() - start_time) * 1000)
            )

        # Fallback: use search and generate
        from src.api.dependencies import ServiceContainer
        container = ServiceContainer.get_instance()

        # Get document info
        doc = await container.database.fetchrow(
            "SELECT id, title FROM documents WHERE id = $1",
            document_id
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get chunks for this document
        chunks = await container.database.fetch(
            """SELECT content FROM chunks
               WHERE document_id = $1
               ORDER BY page_number, chunk_index
               LIMIT 20""",
            document_id
        )

        if not chunks:
            raise HTTPException(status_code=404, detail="No content found for document")

        # Generate summary using Claude
        content = "\n\n".join([c['content'] for c in chunks])

        response = await engine._anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_length * 2,
            messages=[{
                "role": "user",
                "content": f"Summarize this neurosurgical content in {max_length} words or less:\n\n{content[:15000]}"
            }]
        )

        summary = response.content[0].text

        return SummarizeResponse(
            document_id=document_id,
            title=doc['title'] or 'Unknown',
            summary=summary,
            word_count=len(summary.split()),
            generation_time_ms=int((time.time() - start_time) * 1000)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare surgical approaches",
    description="Compare two surgical approaches across multiple aspects"
)
async def compare_approaches(
    request: CompareRequest,
    engine = Depends(get_unified_rag_engine)
):
    """
    Compare two surgical approaches.

    Searches for information about both approaches and generates
    a structured comparison across specified aspects.
    """
    start_time = time.time()

    try:
        # Build comparison query
        query = f"Compare {request.approach_a} and {request.approach_b} in terms of: {', '.join(request.aspects)}"

        # Use internal engine with STANDARD mode for comparison
        from src.research.models import SearchMode
        result = await engine.ask(
            question=query,
            mode=SearchMode.STANDARD,
            include_citations=True
        )

        citations = [
            InternalCitationItem(
                index=c.index,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_type=c.chunk_type.value if hasattr(c.chunk_type, 'value') else str(c.chunk_type) if c.chunk_type else None
            )
            for c in result.internal_citations
        ]

        return CompareResponse(
            approach_a=request.approach_a,
            approach_b=request.approach_b,
            comparison=result.answer,
            aspects_covered=request.aspects,
            citations=citations,
            generation_time_ms=int((time.time() - start_time) * 1000)
        )

    except Exception as e:
        logger.exception(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="List all conversations with pagination"
)
async def get_conversation_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    conversation_store = Depends(get_conversation_store)
):
    """
    Get paginated conversation history.
    """
    try:
        all_convs = conversation_store.list_all()

        # Sort by updated time
        sorted_convs = sorted(
            all_convs.items(),
            key=lambda x: x[1].get('updated_at', x[1].get('created_at', 0)),
            reverse=True
        )

        total = len(sorted_convs)
        start = (page - 1) * page_size
        end = start + page_size

        conversations = [
            ConversationHistoryItem(
                id=conv_id,
                created_at=conv.get('created_at', 0),
                updated_at=conv.get('updated_at', conv.get('created_at', 0)),
                mode=conv.get('mode', 'hybrid'),
                message_count=len(conv.get('history', []))
            )
            for conv_id, conv in sorted_convs[start:end]
        ]

        return ConversationHistoryResponse(
            conversations=conversations,
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.exception(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/clear",
    summary="Clear conversations",
    description="Clear conversation history"
)
async def clear_conversations(
    request: Optional[ClearConversationsRequest] = None,
    conversation_store = Depends(get_conversation_store)
):
    """
    Clear conversations.

    Options:
    - Clear all conversations (no body)
    - Clear specific conversation IDs
    - Clear conversations before a timestamp
    """
    try:
        cleared = 0

        if request and request.conversation_ids:
            for conv_id in request.conversation_ids:
                conversation_store.delete(conv_id)
                cleared += 1
        elif request and request.before:
            all_convs = conversation_store.list_all()
            for conv_id, conv in all_convs.items():
                created_at = conv.get('created_at', 0)
                if created_at < request.before:
                    conversation_store.delete(conv_id)
                    cleared += 1
        else:
            all_convs = conversation_store.list_all()
            cleared = len(all_convs)
            conversation_store.clear_all()

        return {
            "cleared": cleared,
            "message": f"Cleared {cleared} conversation(s)"
        }

    except Exception as e:
        logger.exception(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/context",
    response_model=ContextResponse,
    summary="Get RAG context",
    description="Get RAG context chunks without generating an answer"
)
async def get_context(
    question: str = Query(..., description="Question to get context for"),
    max_chunks: int = Query(20, description="Maximum number of chunks"),
    document_ids: Optional[str] = Query(None, description="Comma-separated document IDs"),
    engine = Depends(get_unified_rag_engine)
):
    """
    Get RAG context chunks without generating an answer.

    Useful for:
    - Debugging search results
    - Building custom prompts
    - Understanding what context is available
    """
    start_time = time.time()

    try:
        from src.api.dependencies import ServiceContainer
        container = ServiceContainer.get_instance()

        # Parse document IDs
        doc_ids = None
        if document_ids:
            doc_ids = [d.strip() for d in document_ids.split(',')]

        # Build filters
        filters = None
        if doc_ids:
            from src.retrieval import SearchFilters
            filters = SearchFilters(document_ids=doc_ids)

        # Search
        search_result = await container._search.search(
            query=question,
            mode="hybrid",
            top_k=max_chunks,
            filters=filters,
            rerank=True
        )

        search_time = int((time.time() - start_time) * 1000)

        chunks = [
            ContextChunk(
                index=i + 1,
                content=getattr(r, 'content', '')[:500],
                chunk_id=getattr(r, 'id', str(i)),
                document_id=getattr(r, 'document_id', None),
                page_number=getattr(r, 'page_number', None),
                score=getattr(r, 'score', 0.0)
            )
            for i, r in enumerate(search_result.results)
        ]

        return ContextResponse(
            query=question,
            chunks=chunks,
            total_chunks=len(chunks),
            search_time_ms=search_time
        )

    except Exception as e:
        logger.exception(f"Context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/conversation/{conversation_id}",
    summary="Get conversation details",
    description="Get full conversation history and metadata"
)
async def get_conversation(
    conversation_id: str,
    conversation_store = Depends(get_conversation_store)
):
    """Get a specific conversation with full history."""
    conv = conversation_store.get(conversation_id)

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "mode": conv.get('mode', 'hybrid'),
        "created_at": conv.get('created_at', 0),
        "updated_at": conv.get('updated_at', conv.get('created_at', 0)),
        "history": conv.get('history', []),
        "message_count": len(conv.get('history', []))
    }
