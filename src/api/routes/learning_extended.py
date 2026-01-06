# src/api/routes/learning_extended.py
"""
Extended Learning API Routes

Additional endpoints for:
- Content-based flashcard generation (relations, UMLS, tables)
- Hybrid flashcard generation
- Socratic learning mode
- Interleaved quiz generation
"""

from typing import Optional, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Import NPRSS dependencies
from src.learning.nprss.dependencies import (
    get_flashcard_generator,
    get_hybrid_generator,
    get_socratic_engine,
    get_chunk_repository,
    get_learning_item_repository,
)


router = APIRouter(prefix="/api/v1/learning", tags=["learning-extended"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

# Content Generation
class GenerateFlashcardsRequest(BaseModel):
    """Request to generate content-based flashcards"""
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    content: Optional[str] = None
    strategies: List[str] = Field(
        default=['relation', 'umls'],
        description="Generation strategies: relation, umls, table"
    )
    max_cards: int = Field(default=50, ge=1, le=200)
    chunk_types: Optional[List[str]] = None


class FlashcardResponse(BaseModel):
    """Response for generated flashcard"""
    id: str
    card_type: str
    prompt: str
    answer: str
    options: Optional[List[dict]] = None
    explanation: Optional[str] = None
    difficulty: float
    tags: List[str]
    source_page: Optional[int] = None
    generation_method: str
    quality_score: float


class GenerateFlashcardsResponse(BaseModel):
    """Response from flashcard generation"""
    cards: List[FlashcardResponse]
    total_generated: int
    strategies_used: List[str]
    by_strategy: dict


# Interleaved Quiz
class InterleaveQuizRequest(BaseModel):
    """Request for interleaved quiz"""
    topic: Optional[str] = None
    chunk_types: List[str] = Field(
        default=['ANATOMY', 'PROCEDURE', 'PATHOLOGY']
    )
    num_questions: int = Field(default=10, ge=5, le=50)
    difficulty_distribution: Optional[dict] = None


class QuizQuestionResponse(BaseModel):
    """Single quiz question"""
    item_id: str
    question: str
    correct_answer: str
    options: Optional[List[dict]] = None
    chunk_type: str
    difficulty: str


class InterleaveQuizResponse(BaseModel):
    """Response for interleaved quiz"""
    quiz_id: str
    questions: List[QuizQuestionResponse]
    total_questions: int
    interleaved: bool = True


# Socratic Mode
class SocraticAskRequest(BaseModel):
    """Request to start Socratic conversation"""
    question: str
    content_type: Optional[str] = None
    user_id: str = "default"


class SocraticRespondRequest(BaseModel):
    """Request to continue Socratic conversation"""
    student_response: str


class SocraticResponse(BaseModel):
    """Response from Socratic engine"""
    response: str
    conversation_id: str
    mode: str
    state: str
    content_type: Optional[str] = None
    is_correct: Optional[bool] = None
    sources: Optional[List[dict]] = None


class SocraticHintResponse(BaseModel):
    """Response with hint"""
    hint: str
    conversation_id: str
    hints_remaining: int
    state: str


class SocraticRevealResponse(BaseModel):
    """Response with revealed answer"""
    answer: str
    conversation_id: str
    state: str
    sources: List[dict]
    total_attempts: int
    hints_used: int


# =============================================================================
# CONTENT-BASED FLASHCARD ENDPOINTS
# =============================================================================

@router.post("/flashcards/generate", response_model=GenerateFlashcardsResponse)
async def generate_flashcards(
    request: GenerateFlashcardsRequest,
    generator=Depends(get_hybrid_generator),
    chunk_repo=Depends(get_chunk_repository),
    learning_repo=Depends(get_learning_item_repository)
):
    """
    Generate flashcards using content-based strategies.

    Strategies:
    - **relation**: Anatomical relationship cards (uses NeuroRelationExtractor)
    - **umls**: Medical terminology definitions (uses UMLSExtractor)
    - **table**: MCQs from grading scales and classifications (uses TableExtractor)

    Can generate from:
    - document_id: All chunks in a document
    - chunk_id: Single specific chunk
    - content: Raw text content
    """
    cards = []

    if request.chunk_id:
        # Generate from single chunk
        chunk = await chunk_repo.get_by_id(request.chunk_id)
        if not chunk:
            raise HTTPException(404, "Chunk not found")

        chunk_dict = chunk if isinstance(chunk, dict) else chunk.to_dict()
        cards = await generator.generate_from_chunk(
            chunk_dict,
            strategies=request.strategies
        )

    elif request.document_id:
        # Generate from document
        cards = await generator.generate_from_document(
            request.document_id,
            chunk_repository=chunk_repo,
            strategies=request.strategies,
            chunk_types=request.chunk_types
        )

    elif request.content:
        # Generate from raw content
        chunk = {
            'content': request.content,
            'id': str(uuid4()),
            'chunk_type': 'GENERAL'
        }
        cards = await generator.generate_from_chunk(
            chunk,
            strategies=request.strategies
        )

    else:
        raise HTTPException(400, "Provide document_id, chunk_id, or content")

    # Limit results
    cards = cards[:request.max_cards]

    # Save to database
    saved_count = 0
    if learning_repo and cards:
        try:
            for card in cards:
                await learning_repo.create(card)
                saved_count += 1
        except Exception as e:
            print(f"Failed to save cards: {e}")

    # Format response
    stats = generator.get_stats()

    return GenerateFlashcardsResponse(
        cards=[
            FlashcardResponse(
                id=str(c.id) if hasattr(c, 'id') and c.id else str(uuid4()),
                card_type=c.card_type.value if hasattr(c.card_type, 'value') else str(c.card_type),
                prompt=c.prompt,
                answer=c.answer,
                options=c.options if hasattr(c, 'options') else None,
                explanation=c.explanation if hasattr(c, 'explanation') else None,
                difficulty=c.difficulty_preset if hasattr(c, 'difficulty_preset') else 0.5,
                tags=c.tags if hasattr(c, 'tags') else [],
                source_page=c.source_page if hasattr(c, 'source_page') else None,
                generation_method=c.generation_method if hasattr(c, 'generation_method') else 'unknown',
                quality_score=c.quality_score if hasattr(c, 'quality_score') else 0.8
            )
            for c in cards
        ],
        total_generated=len(cards),
        strategies_used=request.strategies,
        by_strategy=stats.get('by_strategy', {})
    )


@router.get("/flashcards/strategies")
async def get_available_strategies():
    """
    Get available flashcard generation strategies.
    """
    return {
        'strategies': [
            {
                'name': 'relation',
                'description': 'Anatomical relationship cards (supplies, innervates, etc.)',
                'source': 'NeuroRelationExtractor',
                'card_types': ['forward', 'reverse'],
                'example_prompt': 'What does the facial nerve innervate?'
            },
            {
                'name': 'umls',
                'description': 'Medical terminology definition cards',
                'source': 'UMLSExtractor',
                'card_types': ['definition'],
                'example_prompt': 'Define: Astrocytoma'
            },
            {
                'name': 'table',
                'description': 'MCQs from tables (grading scales, classifications)',
                'source': 'TableExtractor',
                'card_types': ['mcq'],
                'example_prompt': 'Hunt-Hess Scale: What are the criteria for Grade 3?'
            }
        ],
        'combinations': {
            'comprehensive': ['relation', 'umls', 'table'],
            'anatomy_focus': ['relation'],
            'terminology_focus': ['umls'],
            'clinical_focus': ['table', 'umls']
        }
    }


# =============================================================================
# INTERLEAVED QUIZ ENDPOINTS
# =============================================================================

@router.post("/quiz/interleaved", response_model=InterleaveQuizResponse)
async def generate_interleaved_quiz(
    request: InterleaveQuizRequest,
    generator=Depends(get_hybrid_generator),
    user_id: str = Query(default="default")
):
    """
    Generate an interleaved quiz.

    Interleaving mixes question types (anatomy, procedure, pathology)
    to improve long-term retention by forcing discrimination between
    similar concepts.

    Research shows g = 0.42-0.67 improvement over blocked practice.

    Difficulty distribution defaults to research-optimal:
    - 30% easy (build confidence)
    - 50% medium (zone of proximal development)
    - 20% hard (challenge and extend)
    """
    difficulty_dist = request.difficulty_distribution or {
        'easy': 0.3,
        'medium': 0.5,
        'hard': 0.2
    }

    # Generate quiz using hybrid generator
    quiz_id = str(uuid4())

    # For now, return placeholder - full implementation would use database
    return InterleaveQuizResponse(
        quiz_id=quiz_id,
        questions=[],
        total_questions=0,
        interleaved=True
    )


# =============================================================================
# SOCRATIC MODE ENDPOINTS
# =============================================================================

@router.post("/socratic/ask", response_model=SocraticResponse)
async def socratic_ask(
    request: SocraticAskRequest,
    engine=Depends(get_socratic_engine)
):
    """
    Start a Socratic learning conversation.

    Instead of giving direct answers, the system guides the learner
    to discover answers through questioning.

    Research shows 77.8% learner preference for guided discovery.

    Content types:
    - anatomy: Anatomical structures, relationships
    - procedure: Surgical approaches, techniques
    - pathology: Disease processes, mechanisms
    - clinical: Diagnosis, management
    - imaging: Radiology interpretation
    """
    result = await engine.ask(
        question=request.question,
        user_id=request.user_id,
        content_type=request.content_type
    )

    return SocraticResponse(
        response=result['response'],
        conversation_id=result['conversation_id'],
        mode=result['mode'],
        state=result['state'],
        content_type=result.get('content_type'),
        sources=result.get('sources')
    )


@router.post("/socratic/{conversation_id}/respond", response_model=SocraticResponse)
async def socratic_respond(
    conversation_id: str,
    request: SocraticRespondRequest,
    engine=Depends(get_socratic_engine)
):
    """
    Continue Socratic conversation with student's response.

    The system will either:
    - Acknowledge correct reasoning and ask follow-up questions
    - Guide toward correct answer with more specific questions
    - Provide feedback on partial understanding
    """
    try:
        result = await engine.respond(
            conversation_id=conversation_id,
            student_response=request.student_response
        )

        return SocraticResponse(
            response=result['response'],
            conversation_id=result['conversation_id'],
            mode='socratic',
            state=result['state'],
            is_correct=result.get('is_correct')
        )

    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/socratic/{conversation_id}/hint", response_model=SocraticHintResponse)
async def socratic_hint(
    conversation_id: str,
    engine=Depends(get_socratic_engine)
):
    """
    Get a hint when stuck.

    Hints progressively guide toward the answer without revealing it.
    After max hints (default: 3), the full answer is revealed.
    """
    try:
        result = await engine.get_hint(conversation_id)

        return SocraticHintResponse(
            hint=result['hint'],
            conversation_id=result['conversation_id'],
            hints_remaining=result['hints_remaining'],
            state=result['state']
        )

    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/socratic/{conversation_id}/reveal", response_model=SocraticRevealResponse)
async def socratic_reveal(
    conversation_id: str,
    engine=Depends(get_socratic_engine)
):
    """
    Reveal the answer after attempts.

    Provides:
    - The correct answer
    - Explanation of key concepts
    - Feedback on student's attempts
    - Related concepts to explore
    """
    try:
        result = await engine.reveal(conversation_id)

        return SocraticRevealResponse(
            answer=result['answer'],
            conversation_id=result['conversation_id'],
            state=result['state'],
            sources=result['sources'],
            total_attempts=result['total_attempts'],
            hints_used=result['hints_used']
        )

    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/socratic/{conversation_id}")
async def get_socratic_conversation(
    conversation_id: str,
    engine=Depends(get_socratic_engine)
):
    """
    Get details of a Socratic conversation.
    """
    conversation = engine.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(404, "Conversation not found")

    return conversation


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/generation-stats")
async def get_generation_stats(
    generator=Depends(get_flashcard_generator)
):
    """
    Get flashcard generation statistics.
    """
    return generator.get_stats()


@router.get("/analytics/learning-progress")
async def get_learning_progress(
    user_id: str,
    period_days: int = Query(default=30, ge=7, le=365)
):
    """
    Get learning progress analytics.

    Includes:
    - Cards reviewed over time
    - Retention rates by chunk type
    - Mastery progression
    - Interleaving effectiveness
    """
    # This would query actual analytics data
    return {
        'user_id': user_id,
        'period_days': period_days,
        'message': 'Analytics endpoint - implement with actual data'
    }
