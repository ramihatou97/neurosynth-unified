"""
NeuroSynth Unified - RAG Routes
================================

RAG (Retrieval-Augmented Generation) API endpoints.
"""

import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import json

from pydantic import BaseModel
from typing import List, Optional

from src.api.models import (
    RAGRequest,
    RAGResponse,
    CitationItem,
    ImageItem,
    ConversationRequest,
    ConversationResponse,
    ErrorResponse
)
from src.api.dependencies import (
    get_rag_engine,
    get_search_service,
    get_conversation_store
)


class CompareRequest(BaseModel):
    """Request model for comparing surgical approaches."""
    approach_a: str
    approach_b: str
    aspects: Optional[List[str]] = ["indications", "technique", "complications", "outcomes"]
    max_chunks: int = 20

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# =============================================================================
# RAG Endpoints
# =============================================================================

@router.post(
    "/ask",
    response_model=RAGResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Generation error"}
    },
    summary="Ask a question",
    description="Ask a question and get an answer with citations from indexed content"
)
async def ask_question(
    request: RAGRequest,
    rag_engine = Depends(get_rag_engine)
):
    """
    Ask a question using RAG.
    
    The engine will:
    1. Search for relevant context chunks
    2. Assemble context within token budget
    3. Generate answer with Claude
    4. Extract and link citations
    5. Return answer with full provenance
    """
    import time
    start_time = time.time()
    
    try:
        # Build filters if provided
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
                cuis=request.filters.cuis,
                page_range=page_range
            )
        
        # Update config if needed
        if request.max_context_chunks:
            rag_engine.assembler.max_chunks = request.max_context_chunks
        
        # Set question type prompt if specified
        if request.question_type:
            from src.rag import PromptLibrary, QuestionType
            library = PromptLibrary()
            qt = QuestionType(request.question_type.value)
            rag_engine.system_prompt = library.get_system_prompt(qt)
        
        # Generate answer
        result = await rag_engine.ask(
            question=request.question,
            filters=filters,
            include_citations=request.include_citations,
            include_images=request.include_images
        )
        
        total_time = int((time.time() - start_time) * 1000)
        
        # Convert to response model
        citations = [
            CitationItem(
                index=c.index,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_type=c.chunk_type
            )
            for c in result.citations
        ]
        
        used_citations = [
            CitationItem(
                index=c.index,
                chunk_id=c.chunk_id,
                snippet=c.snippet,
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_type=c.chunk_type
            )
            for c in result.used_citations
        ]
        
        images = [
            ImageItem(
                image_id=img.image_id,
                file_path=img.file_path,
                caption=img.caption,
                image_type=img.image_type
            )
            for img in result.images
        ]
        
        return RAGResponse(
            answer=result.answer,
            citations=citations,
            used_citations=used_citations,
            images=images,
            question=result.question,
            context_chunks_used=result.context_chunks_used,
            generation_time_ms=result.generation_time_ms,
            search_time_ms=result.search_time_ms,
            total_time_ms=total_time,
            model=result.model
        )
        
    except Exception as e:
        logger.exception(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ask/stream",
    summary="Ask with streaming response",
    description="Ask a question and stream the response tokens"
)
async def ask_question_stream(
    request: RAGRequest,
    rag_engine = Depends(get_rag_engine)
):
    """
    Ask a question with streaming response.
    
    Returns Server-Sent Events (SSE) stream of tokens.
    """
    try:
        # Build filters
        filters = None
        if request.filters:
            from src.retrieval import SearchFilters
            filters = SearchFilters(
                document_ids=request.filters.document_ids,
                chunk_types=request.filters.chunk_types,
                specialties=request.filters.specialties,
                cuis=request.filters.cuis
            )
        
        async def generate():
            async for chunk in await rag_engine.ask(
                question=request.question,
                filters=filters,
                include_citations=request.include_citations,
                include_images=request.include_images,
                stream=True
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
            }
        )
        
    except Exception as e:
        logger.exception(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Conversation Endpoints
# =============================================================================

@router.post(
    "/conversation",
    response_model=ConversationResponse,
    summary="Multi-turn conversation",
    description="Continue or start a multi-turn conversation"
)
async def conversation(
    request: ConversationRequest,
    rag_engine = Depends(get_rag_engine),
    store = Depends(get_conversation_store)
):
    """
    Multi-turn conversation with context persistence.
    
    If conversation_id is provided, continues existing conversation.
    Otherwise, starts a new conversation.
    """
    try:
        from src.rag import RAGConversation
        
        # Get or create conversation
        if request.conversation_id:
            conv = store.get(request.conversation_id)
            if not conv:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation not found: {request.conversation_id}"
                )
        else:
            conv = RAGConversation(rag_engine)
            request.conversation_id = str(uuid4())
        
        # Build filters
        filters = None
        if request.filters:
            from src.retrieval import SearchFilters
            filters = SearchFilters(
                document_ids=request.filters.document_ids,
                chunk_types=request.filters.chunk_types,
                specialties=request.filters.specialties,
                cuis=request.filters.cuis
            )
        
        # Get response
        result = await conv.ask(
            question=request.message,
            filters=filters if filters else None
        )
        
        # Save conversation
        store.set(request.conversation_id, conv)
        
        return ConversationResponse(
            conversation_id=request.conversation_id,
            answer=result.answer,
            citations=[
                CitationItem(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    snippet=c.snippet,
                    page_number=c.page_number
                )
                for c in result.used_citations
            ],
            history_length=len(conv.history)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/conversation/{conversation_id}",
    summary="End conversation",
    description="End and delete a conversation"
)
async def end_conversation(
    conversation_id: str,
    store = Depends(get_conversation_store)
):
    """End a conversation and free resources."""
    store.delete(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


# =============================================================================
# Specialized Endpoints
# =============================================================================

@router.get(
    "/summarize/{document_id}",
    response_model=RAGResponse,
    summary="Summarize document",
    description="Generate a summary of a document"
)
async def summarize_document(
    document_id: str,
    max_chunks: int = Query(20, ge=5, le=50),
    rag_engine = Depends(get_rag_engine)
):
    """Generate a summary of a document."""
    try:
        result = await rag_engine.summarize_document(
            document_id=document_id,
            max_chunks=max_chunks
        )
        
        return RAGResponse(
            answer=result.answer,
            citations=[
                CitationItem(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    snippet=c.snippet,
                    page_number=c.page_number
                )
                for c in result.citations
            ],
            used_citations=[
                CitationItem(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    snippet=c.snippet,
                    page_number=c.page_number
                )
                for c in result.used_citations
            ],
            images=[],
            question=f"Summarize document {document_id}",
            context_chunks_used=result.context_chunks_used,
            generation_time_ms=result.generation_time_ms,
            search_time_ms=result.search_time_ms,
            total_time_ms=result.generation_time_ms + result.search_time_ms,
            model=result.model
        )
        
    except Exception as e:
        logger.exception(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/compare",
    response_model=RAGResponse,
    summary="Compare approaches",
    description="Compare two surgical approaches"
)
async def compare_approaches(
    request: CompareRequest,
    rag_engine = Depends(get_rag_engine)
):
    """Compare two surgical approaches."""
    try:
        result = await rag_engine.compare_approaches(
            approach1=request.approach_a,
            approach2=request.approach_b
        )

        return RAGResponse(
            answer=result.answer,
            citations=[
                CitationItem(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    snippet=c.snippet,
                    page_number=c.page_number
                )
                for c in result.citations
            ],
            used_citations=[
                CitationItem(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    snippet=c.snippet,
                    page_number=c.page_number
                )
                for c in result.used_citations
            ],
            images=[],
            question=f"Compare {request.approach_a} vs {request.approach_b}",
            context_chunks_used=result.context_chunks_used,
            generation_time_ms=result.generation_time_ms,
            search_time_ms=result.search_time_ms,
            total_time_ms=result.generation_time_ms + result.search_time_ms,
            model=result.model
        )

    except Exception as e:
        logger.exception(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# History & Context Endpoints (Frontend Parity)
# =============================================================================

@router.get(
    "/history",
    summary="Get conversation history",
    description="Get list of past conversations with pagination"
)
async def get_conversation_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    store = Depends(get_conversation_store)
):
    """
    Get paginated list of past conversations.

    Returns conversation metadata (id, created_at, message_count) without full history.
    """
    try:
        # Get all conversations from store
        all_conversations = store.list_all() if hasattr(store, 'list_all') else []

        # Paginate
        total = len(all_conversations)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_conversations[start:end]

        return {
            "conversations": [
                {
                    "id": conv.get("id", ""),
                    "created_at": conv.get("created_at", ""),
                    "updated_at": conv.get("updated_at", ""),
                    "message_count": conv.get("message_count", 0),
                    "preview": conv.get("preview", "")
                }
                for conv in page_items
            ],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0
        }

    except Exception as e:
        logger.exception(f"History error: {e}")
        # Return empty list on error rather than failing
        return {
            "conversations": [],
            "page": page,
            "page_size": page_size,
            "total": 0,
            "total_pages": 0
        }


@router.post(
    "/clear",
    summary="Clear conversations",
    description="Clear conversation history"
)
async def clear_conversations(
    conversation_ids: Optional[List[str]] = None,
    store = Depends(get_conversation_store)
):
    """
    Clear conversation history.

    If conversation_ids provided, clears only those.
    Otherwise clears all conversations.
    """
    try:
        cleared_count = 0

        if conversation_ids:
            # Clear specific conversations
            for conv_id in conversation_ids:
                if store.delete(conv_id):
                    cleared_count += 1
        else:
            # Clear all
            if hasattr(store, 'clear_all'):
                cleared_count = store.clear_all()
            else:
                # Fallback: delete one by one
                all_ids = store.list_all() if hasattr(store, 'list_all') else []
                for conv in all_ids:
                    if store.delete(conv.get("id", "")):
                        cleared_count += 1

        return {
            "status": "cleared",
            "cleared_count": cleared_count
        }

    except Exception as e:
        logger.exception(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/context",
    summary="Get RAG context",
    description="Get context chunks without generating an answer"
)
async def get_rag_context(
    question: str = Query(..., min_length=1, description="Question to find context for"),
    max_chunks: int = Query(20, ge=1, le=50, description="Maximum context chunks"),
    rag_engine = Depends(get_rag_engine),
    search_service = Depends(get_search_service)
):
    """
    Get RAG context chunks for a question without generating an answer.

    Useful for:
    - Previewing what context will be used
    - Debugging retrieval quality
    - Building custom prompts
    """
    try:
        # Use search service to find relevant chunks
        search_result = await search_service.search(
            query=question,
            top_k=max_chunks,
            mode="hybrid"
        )

        # Format context chunks
        chunks = []
        for i, result in enumerate(search_result.results):
            chunks.append({
                "index": i + 1,
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
                "document_id": result.document_id,
                "page_number": result.metadata.get("page_number") if result.metadata else None,
                "chunk_type": result.metadata.get("chunk_type") if result.metadata else None
            })

        return {
            "question": question,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "search_time_ms": search_result.search_time_ms if hasattr(search_result, 'search_time_ms') else 0
        }

    except Exception as e:
        logger.exception(f"Context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
