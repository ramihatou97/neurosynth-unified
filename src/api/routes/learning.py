# src/api/routes/learning.py
"""
NPRSS Learning API Routes

Endpoints for procedural learning:
- /enrich - Enrich procedure with learning features
- /schedule - R1-R7 retrieval schedules
- /cards - FSRS flashcard management
- /mastery - Mastery tracking
- /surgical-card - Surgical card management
- /csps - Critical Safety Points
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Import NPRSS dependencies
from src.learning.nprss.dependencies import (
    get_review_service,
    get_mastery_service,
    get_learning_services,
    get_user_state_repository,
    get_learning_item_repository,
)
from src.database.connection import get_database


router = APIRouter(prefix="/api/v1/learning", tags=["learning"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

# Enrichment
class EnrichProcedureRequest(BaseModel):
    """Request to enrich a procedure"""
    procedure_id: Optional[str] = None
    synthesis_id: Optional[str] = None
    generate_surgical_card: bool = True
    generate_fsrs_cards: bool = True
    create_retrieval_schedule: bool = False
    user_id: Optional[str] = None


class EnrichProcedureResponse(BaseModel):
    """Response from procedure enrichment"""
    procedure_id: str
    phases_mapped: int
    csps_extracted: int
    anchors_generated: int
    phase_gates_created: int
    cards_generated: int
    surgical_card_id: Optional[str] = None
    schedule_id: Optional[str] = None


# Schedule
class CreateScheduleRequest(BaseModel):
    """Request to create retrieval schedule"""
    user_id: str
    procedure_id: str
    target_retention_days: int = Field(default=180, ge=30, le=365)
    encoding_date: Optional[str] = None


class RetrievalSessionResponse(BaseModel):
    """Response for retrieval session"""
    id: str
    session_number: int
    scheduled_date: str
    days_from_encoding: int
    retrieval_task: str
    task_type: str
    estimated_duration_min: int
    procedure_id: str
    procedure_name: str
    completed: bool = False
    is_overdue: bool = False


class CompleteSessionRequest(BaseModel):
    """Request to complete a session"""
    self_assessment_score: int = Field(ge=1, le=4)
    notes: Optional[str] = None


# Cards
class CardReviewRequest(BaseModel):
    """Request to review a card"""
    user_id: str
    rating: int = Field(ge=1, le=4)  # 1=AGAIN, 2=HARD, 3=GOOD, 4=EASY
    response_time_ms: Optional[int] = None
    session_id: Optional[str] = None


class DueCardResponse(BaseModel):
    """Response for due card"""
    id: str
    procedure_id: Optional[str] = None
    procedure_name: Optional[str] = None
    card_type: str
    prompt: str
    answer: Optional[str] = None
    due_date: Optional[str] = None
    state: str
    difficulty: float
    days_overdue: float = 0


class CardReviewResponse(BaseModel):
    """Response after reviewing a card"""
    card_id: str
    new_state: str
    new_due_date: str
    new_stability: float
    new_difficulty: float
    next_interval_days: float


# Mastery
class MasteryResponse(BaseModel):
    """Response for mastery state"""
    user_id: str
    procedure_id: str
    procedure_name: Optional[str] = None
    current_level: int
    level_name: str
    phase_scores: dict = {}
    weak_csps: List[int] = []
    weak_phases: List[str] = []
    predicted_retention_score: float = 0.5
    next_optimal_review: Optional[str] = None
    total_retrieval_sessions: int = 0


# Surgical Card
class SurgicalCardResponse(BaseModel):
    """Response for surgical card"""
    id: str
    procedure_id: str
    title: str
    subtitle: Optional[str] = None
    approach: Optional[str] = None
    corridor: Optional[str] = None
    card_rows: List[dict]
    csp_summary: List[dict]
    dictation_template: Optional[str] = None
    mantra: str


# CSP
class CSPResponse(BaseModel):
    """Response for Critical Safety Point"""
    id: str
    csp_number: int
    phase_type: Optional[str] = None
    when_action: str
    stop_if_trigger: str
    structure_at_risk: str
    mechanism_of_injury: Optional[str] = None
    if_violated_action: Optional[str] = None
    retrieval_cue: Optional[str] = None


class CSPQuizItem(BaseModel):
    """Quiz item for CSP rapid-fire"""
    csp_number: int
    question_type: str  # "trigger_to_structure", "when_to_stop", "mechanism"
    prompt: str
    answer: str


# =============================================================================
# CARD ENDPOINTS
# =============================================================================

@router.get("/cards/due", response_model=List[DueCardResponse])
async def get_due_cards(
    user_id: str,
    procedure_id: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    review_service=Depends(get_review_service)
):
    """
    Get FSRS cards due for review.

    Returns cards sorted by due date (most overdue first).
    """
    due_cards = await review_service.get_due_cards(user_id, limit)

    return [
        DueCardResponse(
            id=str(card.get('card_id', card.get('id'))),
            procedure_id=str(card['procedure_id']) if card.get('procedure_id') else None,
            procedure_name=card.get('procedure_name'),
            card_type=card.get('card_type', 'mcq'),
            prompt=card.get('prompt', ''),
            answer=card.get('answer'),
            due_date=card['due_date'].isoformat() if card.get('due_date') else None,
            state=card.get('state', 'new'),
            difficulty=float(card.get('difficulty', 0.3)),
            days_overdue=float(card.get('days_overdue', 0))
        )
        for card in due_cards
    ]


@router.post("/cards/{card_id}/review", response_model=CardReviewResponse)
async def review_card(
    card_id: str,
    request: CardReviewRequest,
    review_service=Depends(get_review_service)
):
    """
    Submit card review and get next schedule.

    Ratings:
    - 1: AGAIN - Complete failure
    - 2: HARD - Recalled with significant difficulty
    - 3: GOOD - Recalled with moderate effort
    - 4: EASY - Effortless recall
    """
    result = await review_service.submit_review(
        user_id=request.user_id,
        card_id=card_id,
        rating=request.rating,
        response_time_ms=request.response_time_ms,
        session_id=request.session_id
    )

    return CardReviewResponse(
        card_id=card_id,
        new_state=result['new_state'],
        new_due_date=result['next_review'].isoformat() if result.get('next_review') else "",
        new_stability=result['new_stability'],
        new_difficulty=result['new_difficulty'],
        next_interval_days=result['interval_days']
    )


@router.get("/cards/forecast")
async def get_review_forecast(
    user_id: str,
    days: int = Query(default=7, ge=1, le=30),
    review_service=Depends(get_review_service)
):
    """Get forecast of upcoming reviews."""
    forecast = await review_service.get_forecast(user_id, days)
    return {"forecast": forecast}


@router.get("/cards/statistics")
async def get_card_statistics(
    user_id: str,
    procedure_id: Optional[str] = None,
    state_repo=Depends(get_user_state_repository)
):
    """Get learning statistics for user."""
    from uuid import UUID as UUIDType
    proc_uuid = UUIDType(procedure_id) if procedure_id else None
    stats = await state_repo.get_user_statistics(user_id, proc_uuid)
    return stats


# =============================================================================
# MASTERY ENDPOINTS
# =============================================================================

@router.get("/mastery/{user_id}", response_model=List[MasteryResponse])
async def get_user_mastery(
    user_id: str,
    service=Depends(get_mastery_service)
):
    """Get mastery overview for all user procedures."""
    results = await service.get_user_mastery_overview(user_id)
    return [MasteryResponse(**r) for r in results]


@router.get("/mastery/{user_id}/{procedure_id}", response_model=MasteryResponse)
async def get_procedure_mastery(
    user_id: str,
    procedure_id: str,
    service=Depends(get_mastery_service)
):
    """
    Get detailed mastery for a specific procedure.

    Includes:
    - Current mastery level (1-4)
    - Phase-level scores
    - Weak CSPs and phases
    - Predicted retention score
    - Next optimal review date
    """
    result = await service.get_mastery(user_id, procedure_id)
    return MasteryResponse(**result)


# =============================================================================
# SURGICAL CARD ENDPOINTS
# =============================================================================

@router.get("/surgical-card/{procedure_id}", response_model=SurgicalCardResponse)
async def get_surgical_card(
    procedure_id: str,
    db=Depends(get_database)
):
    """Get surgical card for a procedure."""
    query = "SELECT * FROM surgical_cards WHERE procedure_id = $1"
    row = await db.fetchrow(query, procedure_id)

    if not row:
        raise HTTPException(status_code=404, detail="Surgical card not found")

    import json

    return SurgicalCardResponse(
        id=str(row['id']),
        procedure_id=str(row['procedure_id']),
        title=row['title'],
        subtitle=row['subtitle'],
        approach=row['approach'],
        corridor=row['corridor'],
        card_rows=json.loads(row['card_rows']) if isinstance(row['card_rows'], str) else row['card_rows'],
        csp_summary=json.loads(row['csp_summary']) if isinstance(row['csp_summary'], str) else (row['csp_summary'] or []),
        dictation_template=row['dictation_template'],
        mantra=row['mantra']
    )


# =============================================================================
# CSP ENDPOINTS
# =============================================================================

@router.get("/csps/{procedure_id}", response_model=List[CSPResponse])
async def get_csps(
    procedure_id: str,
    db=Depends(get_database)
):
    """Get all Critical Safety Points for a procedure."""
    query = """
        SELECT * FROM critical_safety_points
        WHERE procedure_id = $1
        ORDER BY csp_number
    """
    rows = await db.fetch(query, procedure_id)

    return [
        CSPResponse(
            id=str(row['id']),
            csp_number=row['csp_number'],
            phase_type=row['phase_type'],
            when_action=row['when_action'],
            stop_if_trigger=row['stop_if_trigger'],
            structure_at_risk=row['structure_at_risk'],
            mechanism_of_injury=row['mechanism_of_injury'],
            if_violated_action=row['if_violated_action'],
            retrieval_cue=row['retrieval_cue']
        )
        for row in rows
    ]


@router.get("/csps/{procedure_id}/quiz", response_model=List[CSPQuizItem])
async def get_csp_quiz(
    procedure_id: str,
    count: int = Query(default=5, ge=1, le=20),
    db=Depends(get_database)
):
    """
    Get rapid-fire CSP quiz items.

    Three question types:
    1. trigger_to_structure: See trigger -> Name structure at risk
    2. when_to_stop: When action -> Stop if...
    3. mechanism: How would injury occur?
    """
    import random

    query = """
        SELECT * FROM critical_safety_points
        WHERE procedure_id = $1
        ORDER BY csp_number
    """
    rows = await db.fetch(query, procedure_id)

    if not rows:
        return []

    quiz_items = []

    for row in rows:
        # Type 1: Trigger -> Structure
        quiz_items.append(CSPQuizItem(
            csp_number=row['csp_number'],
            question_type="trigger_to_structure",
            prompt=f"CSP #{row['csp_number']}: {row['stop_if_trigger']} -> What structure?",
            answer=row['structure_at_risk']
        ))

        # Type 2: When -> Stop
        quiz_items.append(CSPQuizItem(
            csp_number=row['csp_number'],
            question_type="when_to_stop",
            prompt=f"WHEN: {row['when_action'][:50]}... -> STOP IF?",
            answer=row['stop_if_trigger']
        ))

        # Type 3: Mechanism
        if row['mechanism_of_injury']:
            quiz_items.append(CSPQuizItem(
                csp_number=row['csp_number'],
                question_type="mechanism",
                prompt=f"How would {row['structure_at_risk']} be injured?",
                answer=row['mechanism_of_injury']
            ))

    # Shuffle and limit
    random.shuffle(quiz_items)
    return quiz_items[:count]
