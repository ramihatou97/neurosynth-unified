"""
NeuroSynth Production Chat Routes
==================================

Production-ready Chat API endpoints using:
- ConversationStore (Redis-backed persistence)
- SynthesisContextStore (synthesis linking)
- Token budget management
- Proper concurrency handling

Endpoints:
- POST /api/v1/chat/ask - Ask with optional conversation context
- POST /api/v1/chat/ask/stream - Stream response via SSE
- GET  /api/v1/chat/conversations - List recent conversations
- GET  /api/v1/chat/conversations/{id} - Get conversation history
- DELETE /api/v1/chat/conversations/{id} - Delete conversation
- POST /api/v1/chat/link-synthesis - Link conversation to synthesis
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.chat.store import (
    ConversationStore,
    SynthesisContextStore,
    Conversation,
    get_conversation_store,
    get_synthesis_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatFilters(BaseModel):
    """Filters for chat context retrieval."""
    document_ids: Optional[List[str]] = None
    chunk_types: Optional[List[str]] = None
    cuis: Optional[List[str]] = None
    min_authority: Optional[float] = Field(None, ge=0, le=1)


class ChatRequest(BaseModel):
    """Enhanced chat request."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    synthesis_id: Optional[str] = Field(None, description="Link to synthesis result")
    filters: Optional[ChatFilters] = None
    include_citations: bool = Field(default=True)
    include_images: bool = Field(default=True)
    max_context_chunks: int = Field(default=10, ge=1, le=30)
    max_history_tokens: int = Field(default=4000, ge=0, le=16000)
    stream: bool = Field(default=False, description="Stream response via SSE")


class CitationResponse(BaseModel):
    """Rich citation in response."""
    index: int
    chunk_id: str
    snippet: str
    document_title: str
    page_number: int
    chunk_type: str
    authority_score: float
    cuis: List[str] = []


class ImageResponse(BaseModel):
    """Image in response."""
    id: str
    caption: str
    file_path: str
    image_type: str = "unknown"


class ChatResponse(BaseModel):
    """Enhanced chat response."""
    answer: str
    conversation_id: str

    # Citations
    citations: List[CitationResponse]
    used_citation_indices: List[int]

    # Images
    images: List[ImageResponse] = []

    # Metadata
    context_chunks_used: int
    synthesis_context_id: Optional[str] = None

    # Token tracking
    history_tokens_used: int = 0
    context_tokens_used: int = 0

    # Timing
    search_time_ms: int
    generation_time_ms: int
    total_time_ms: int

    # Follow-ups
    follow_up_questions: List[str] = []


class ConversationInfo(BaseModel):
    """Conversation metadata."""
    conversation_id: str
    created_at: float
    updated_at: float
    turn_count: int
    total_tokens: int
    synthesis_id: Optional[str] = None
    topic: Optional[str] = None


class ConversationHistory(BaseModel):
    """Full conversation history."""
    conversation_id: str
    turns: List[Dict[str, Any]]
    all_citations: List[CitationResponse]
    synthesis_context: Optional[Dict] = None
    total_tokens: int


class SynthesisLinkRequest(BaseModel):
    """Request to link conversation to synthesis."""
    conversation_id: str
    synthesis_id: str


# =============================================================================
# STORE SINGLETONS
# =============================================================================

_conversation_store: Optional[ConversationStore] = None
_synthesis_store: Optional[SynthesisContextStore] = None


async def get_stores():
    """Get conversation and synthesis stores (singleton)."""
    global _conversation_store, _synthesis_store

    redis_url = os.getenv("REDIS_URL")

    if _conversation_store is None:
        _conversation_store = await get_conversation_store(redis_url)

    if _synthesis_store is None:
        _synthesis_store = await get_synthesis_store(redis_url)

    return _conversation_store, _synthesis_store


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/ask", response_model=ChatResponse)
async def chat_ask(request: ChatRequest):
    """
    Ask a question with optional conversation context.

    Features:
    - Multi-turn conversation support with token tracking
    - Synthesis context linking
    - Rich citations with medical concepts
    - Follow-up question suggestions
    - Redis-backed persistence (production)
    """
    from src.api.dependencies import ServiceContainer

    start_time = time.time()
    conv_store, synth_store = await get_stores()

    try:
        # Initialize services
        container = ServiceContainer.get_instance()
        if not container._initialized:
            await container.initialize()

        if not container.search:
            raise HTTPException(503, "Search service not available")

        # Get or create conversation
        conv = await conv_store.get_or_create(request.conversation_id)
        conv_id = conv.conversation_id

        # Link synthesis if provided
        synthesis_context = None
        if request.synthesis_id:
            synthesis_context = await synth_store.get(request.synthesis_id)
            if synthesis_context and not conv.synthesis_id:
                await conv_store.link_synthesis(
                    conv_id,
                    request.synthesis_id,
                    synthesis_context
                )
        elif conv.synthesis_id:
            synthesis_context = conv.synthesis_context

        # Add user turn
        await conv_store.add_turn(conv_id, 'user', request.message)

        # Build search filters
        filters = None
        if request.filters or synthesis_context:
            from src.retrieval.search_service import SearchFilters

            filter_dict = {}
            if request.filters:
                if request.filters.document_ids:
                    filter_dict['document_ids'] = request.filters.document_ids
                if request.filters.chunk_types:
                    filter_dict['chunk_types'] = request.filters.chunk_types
                if request.filters.cuis:
                    filter_dict['cuis'] = request.filters.cuis

            # Add synthesis context filters
            if synthesis_context and not filter_dict.get('document_ids'):
                doc_ids = synthesis_context.get('document_ids', [])
                if doc_ids:
                    filter_dict['document_ids'] = doc_ids[:5]

            if filter_dict:
                filters = SearchFilters(**filter_dict)

        # Search
        search_start = time.time()
        search_response = await container.search.search(
            query=request.message,
            top_k=request.max_context_chunks * 2,
            include_images=request.include_images,
            filters=filters
        )
        search_time = int((time.time() - search_start) * 1000)

        if not search_response.results:
            # Still add assistant turn for consistency
            no_result_answer = "I couldn't find relevant information to answer your question. Could you rephrase or ask about a different topic?"
            await conv_store.add_turn(conv_id, 'assistant', no_result_answer)

            return ChatResponse(
                answer=no_result_answer,
                conversation_id=conv_id,
                citations=[],
                used_citation_indices=[],
                images=[],
                context_chunks_used=0,
                search_time_ms=search_time,
                generation_time_ms=0,
                total_time_ms=int((time.time() - start_time) * 1000)
            )

        # Build context
        from src.rag.context import ContextAssembler
        assembler = ContextAssembler(
            max_context_tokens=8000,
            max_chunks=request.max_context_chunks
        )
        context = assembler.assemble(search_response.results, request.message)

        # Build enhanced citations
        citations = []
        for i, result in enumerate(search_response.results[:context.chunks_used], 1):
            content = getattr(result, 'content', '')
            snippet = content[:150] + "..." if len(content) > 150 else content

            citations.append(CitationResponse(
                index=i,
                chunk_id=getattr(result, 'chunk_id', str(i)),
                snippet=snippet,
                document_title=getattr(result, 'document_title', 'Unknown'),
                page_number=getattr(result, 'page_start', 0) or 0,
                chunk_type=str(getattr(result, 'chunk_type', 'GENERAL')),
                authority_score=getattr(result, 'authority_score', 0.7),
                cuis=getattr(result, 'cuis', []) or []
            ))

        # Build prompt
        prompt_parts = []

        if synthesis_context:
            prompt_parts.append(
                f"Context: User is asking about a chapter on '{synthesis_context.get('topic', 'Unknown')}'"
            )
            prompt_parts.append("")

        prompt_parts.append("Retrieved Context:")
        prompt_parts.append(context.text)
        prompt_parts.append("")
        prompt_parts.append(f"Question: {request.message}")
        prompt_parts.append("")
        prompt_parts.append("Provide a comprehensive answer using citations [1], [2], etc.")

        prompt = "\n".join(prompt_parts)

        # Get conversation history with token budget
        history = await conv_store.get_history_for_claude(
            conv_id,
            max_tokens=request.max_history_tokens
        )
        history_tokens = sum(len(m['content']) // 4 for m in history)

        # Generate with Claude
        gen_start = time.time()

        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        messages = history[:-1] + [{'role': 'user', 'content': prompt}]  # Replace last user msg with full prompt

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.3,
            system="You are a neurosurgical knowledge assistant. Answer based only on the provided context. Use [N] citations.",
            messages=messages
        )
        answer = response.content[0].text

        gen_time = int((time.time() - gen_start) * 1000)

        # Extract used citations
        import re
        used_indices = list(set(int(m) for m in re.findall(r'\[(\d+)\]', answer)))

        # Build images response
        images = []
        for img in context.images[:5]:
            images.append(ImageResponse(
                id=getattr(img, 'image_id', str(uuid4())),
                caption=getattr(img, 'caption', ''),
                file_path=getattr(img, 'file_path', ''),
                image_type=getattr(img, 'image_type', 'unknown')
            ))

        # Generate follow-up questions
        follow_ups = _generate_simple_follow_ups(request.message, answer)

        # Add assistant turn with citations
        used_citations_data = [c.model_dump() for c in citations if c.index in used_indices]
        await conv_store.add_turn(conv_id, 'assistant', answer, used_citations_data)

        total_time = int((time.time() - start_time) * 1000)

        return ChatResponse(
            answer=answer,
            conversation_id=conv_id,
            citations=citations,
            used_citation_indices=used_indices,
            images=images,
            context_chunks_used=context.chunks_used,
            synthesis_context_id=request.synthesis_id or conv.synthesis_id,
            history_tokens_used=history_tokens,
            context_tokens_used=context.total_tokens,
            search_time_ms=search_time,
            generation_time_ms=gen_time,
            total_time_ms=total_time,
            follow_up_questions=follow_ups
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(500, f"Chat error: {str(e)}")


@router.post("/ask/stream")
async def chat_ask_stream(request: ChatRequest):
    """
    Stream chat response via Server-Sent Events.

    Events:
    - search_complete: Search finished
    - text: Token stream
    - complete: Final metadata with citations
    - error: Error occurred
    """
    from src.api.dependencies import ServiceContainer

    async def event_generator():
        start_time = time.time()
        conv_store, synth_store = await get_stores()

        try:
            container = ServiceContainer.get_instance()
            if not container._initialized:
                await container.initialize()

            # Get or create conversation
            conv = await conv_store.get_or_create(request.conversation_id)
            conv_id = conv.conversation_id

            await conv_store.add_turn(conv_id, 'user', request.message)

            # Search
            search_start = time.time()
            search_response = await container.search.search(
                query=request.message,
                top_k=request.max_context_chunks * 2,
                include_images=request.include_images
            )
            search_time = int((time.time() - search_start) * 1000)

            yield f"data: {json.dumps({'type': 'search_complete', 'results': len(search_response.results), 'time_ms': search_time})}\n\n"

            if not search_response.results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant results found'})}\n\n"
                return

            # Build context
            from src.rag.context import ContextAssembler
            assembler = ContextAssembler(max_chunks=request.max_context_chunks)
            context = assembler.assemble(search_response.results, request.message)

            # Build prompt
            prompt = f"Context:\n{context.text}\n\nQuestion: {request.message}\n\nAnswer with [N] citations:"

            # Get history
            history = await conv_store.get_history_for_claude(conv_id, max_tokens=request.max_history_tokens)

            # Stream from Claude
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            messages = history[:-1] + [{'role': 'user', 'content': prompt}]

            gen_start = time.time()
            full_answer = []

            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                temperature=0.3,
                system="You are a neurosurgical knowledge assistant. Use [N] citations.",
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    full_answer.append(text)
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

            gen_time = int((time.time() - gen_start) * 1000)
            answer = ''.join(full_answer)

            # Build citations
            citations = []
            for i, result in enumerate(search_response.results[:context.chunks_used], 1):
                citations.append({
                    'index': i,
                    'chunk_id': getattr(result, 'chunk_id', str(i)),
                    'snippet': getattr(result, 'content', '')[:100] + "...",
                    'document_title': getattr(result, 'document_title', 'Unknown'),
                    'page_number': getattr(result, 'page_start', 0)
                })

            # Extract used citations
            import re
            used_indices = list(set(int(m) for m in re.findall(r'\[(\d+)\]', answer)))

            # Save assistant turn
            used_citations_data = [c for c in citations if c['index'] in used_indices]
            await conv_store.add_turn(conv_id, 'assistant', answer, used_citations_data)

            total_time = int((time.time() - start_time) * 1000)

            # Send final metadata
            yield f"data: {json.dumps({'type': 'complete', 'conversation_id': conv_id, 'citations': citations, 'used_indices': used_indices, 'search_time_ms': search_time, 'generation_time_ms': gen_time, 'total_time_ms': total_time})}\n\n"

        except Exception as e:
            logger.exception("Streaming chat failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.post("/link-synthesis")
async def link_conversation_to_synthesis(request: SynthesisLinkRequest):
    """Link an existing conversation to a synthesis result."""
    conv_store, synth_store = await get_stores()

    conv = await conv_store.get(request.conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    synthesis_context = await synth_store.get(request.synthesis_id)
    if not synthesis_context:
        raise HTTPException(404, "Synthesis context not found. Generate synthesis first.")

    await conv_store.link_synthesis(
        request.conversation_id,
        request.synthesis_id,
        synthesis_context
    )

    return {
        "status": "linked",
        "conversation_id": request.conversation_id,
        "synthesis_id": request.synthesis_id
    }


@router.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations(limit: int = Query(default=20, le=100)):
    """List recent conversations."""
    conv_store, synth_store = await get_stores()

    conversations = await conv_store.list_recent(limit)

    result = []
    for conv in conversations:
        topic = None
        if conv.synthesis_context:
            topic = conv.synthesis_context.get('topic')

        result.append(ConversationInfo(
            conversation_id=conv.conversation_id,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            turn_count=len(conv.turns),
            total_tokens=conv.total_tokens,
            synthesis_id=conv.synthesis_id,
            topic=topic
        ))

    return result


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """Get full conversation history."""
    conv_store, _ = await get_stores()

    conv = await conv_store.get(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    # Get all unique citations
    all_citations = await conv_store.get_all_citations(conversation_id)
    citation_responses = []
    for cit in all_citations:
        required_keys = ['index', 'chunk_id', 'snippet', 'document_title', 'page_number', 'chunk_type', 'authority_score']
        if all(k in cit for k in required_keys):
            citation_responses.append(CitationResponse(**cit))

    return ConversationHistory(
        conversation_id=conversation_id,
        turns=[t.to_dict() for t in conv.turns],
        all_citations=citation_responses,
        synthesis_context=conv.synthesis_context,
        total_tokens=conv.total_tokens
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conv_store, _ = await get_stores()

    deleted = await conv_store.delete(conversation_id)
    if not deleted:
        raise HTTPException(404, "Conversation not found")

    return {"status": "deleted", "conversation_id": conversation_id}


@router.post("/conversations/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """Clear conversation history but keep metadata."""
    conv_store, _ = await get_stores()

    cleared = await conv_store.clear(conversation_id)
    if not cleared:
        raise HTTPException(404, "Conversation not found")

    return {"status": "cleared", "conversation_id": conversation_id}


@router.get("/store-info")
async def get_store_info():
    """Get information about the conversation store backend."""
    conv_store, synth_store = await get_stores()

    # Force initialization
    await conv_store._ensure_initialized()

    return {
        "backend": conv_store.backend_type,
        "redis_url": os.getenv("REDIS_URL", "not configured"),
        "conversation_ttl_days": 7
    }


@router.get("/health")
async def chat_health():
    """Chat subsystem health check."""
    conv_store, synth_store = await get_stores()
    await conv_store._ensure_initialized()

    return {
        "status": "healthy",
        "backend": conv_store.backend_type,
        "features": [
            "multi_turn_conversations",
            "synthesis_linking",
            "token_tracking",
            "enhanced_citations",
            "follow_up_questions",
            "streaming"
        ]
    }


# =============================================================================
# HELPERS
# =============================================================================

def _generate_simple_follow_ups(question: str, answer: str) -> List[str]:
    """Generate simple follow-up question suggestions."""
    follow_ups = []

    question_lower = question.lower()

    if "approach" in question_lower:
        follow_ups.append("What are the potential complications of this approach?")
        follow_ups.append("What anatomical structures are at risk?")
    elif "complication" in question_lower:
        follow_ups.append("How can these complications be avoided?")
        follow_ups.append("What is the management if complications occur?")
    elif "anatomy" in question_lower:
        follow_ups.append("What surgical approaches access this region?")
        follow_ups.append("What are the key anatomical landmarks?")
    elif "technique" in question_lower:
        follow_ups.append("What are the indications for this technique?")
        follow_ups.append("What are alternative techniques?")
    else:
        follow_ups.append("Can you provide more details?")
        follow_ups.append("What are the key considerations?")

    return follow_ups[:3]
