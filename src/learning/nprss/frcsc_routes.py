"""
FRCSC Learning Enhancement Routes

New API endpoints for:
- Question bank with IRT
- Exam simulation
- Enhanced gamification
- CSP quizzes

Mount at: /api/v1/frcsc
"""

from datetime import date
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS
# ============================================================================

# Question Bank
class AdaptiveQuestionRequest(BaseModel):
    category: Optional[str] = None
    exclude_ids: List[str] = []


class QuestionResponse(BaseModel):
    question_id: str
    stem: str
    options: Optional[List[dict]] = None
    format: str
    category: str
    yield_rating: int
    time_limit_seconds: int = 120


class AnswerSubmission(BaseModel):
    question_id: str
    answer: str
    time_spent_seconds: int


class AnswerResult(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: Optional[str] = None
    key_points: List[str] = []
    ability_percentile: int


# Exam Simulation
class ExamStartRequest(BaseModel):
    exam_type: str = "written"  # written, oral, mock_full


class ExamQuestionResponse(BaseModel):
    question_number: int
    total_questions: int
    question_id: str
    stem: str
    options: Optional[List[dict]] = None
    time_remaining: int
    allocated_seconds: int


class ExamAnswerSubmission(BaseModel):
    question_id: str
    answer: str
    time_spent_seconds: int


class ExamResult(BaseModel):
    exam_id: str
    score_percentage: float
    passed: bool
    pass_threshold: float
    total_questions: int
    correct_answers: int
    duration_minutes: int
    category_scores: dict
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


# Pass Prediction
class PassPrediction(BaseModel):
    probability: float
    confidence: str
    factors: dict


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter()


# Import actual dependencies from neurosynth-unified
from .dependencies import (
    get_question_bank_service,
    get_exam_service
)

# User auth dependency - using placeholder for now
# TODO: Replace with actual auth when implemented
async def get_current_user_id():
    """Get current user ID from auth context."""
    from uuid import UUID
    # Placeholder - in production this would come from JWT/session
    return UUID("00000000-0000-0000-0000-000000000001")


# ============================================================================
# QUESTION BANK ENDPOINTS
# ============================================================================

@router.post("/questions/adaptive", response_model=QuestionResponse)
async def get_adaptive_question(
    request: AdaptiveQuestionRequest,
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_question_bank_service)
):
    """Get next optimal question based on IRT ability estimate."""
    question = await service.get_adaptive_question(
        user_id=user_id,
        category=request.category,
        exclude_ids=[UUID(id) for id in request.exclude_ids]
    )
    
    if not question:
        raise HTTPException(status_code=404, detail="No questions available")
    
    return QuestionResponse(
        question_id=str(question.id),
        stem=question.stem,
        options=question.options,
        format=question.format.value,
        category=str(question.category_id),
        yield_rating=question.yield_rating,
        time_limit_seconds=120 if question.format.value == "short_answer" else 90
    )


@router.post("/questions/answer", response_model=AnswerResult)
async def submit_answer(
    submission: AnswerSubmission,
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_question_bank_service)
):
    """Submit answer and get result with updated ability."""
    result = await service.submit_answer(
        user_id=user_id,
        question_id=UUID(submission.question_id),
        user_answer=submission.answer,
        time_spent=submission.time_spent_seconds
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return AnswerResult(
        is_correct=result["is_correct"],
        correct_answer=result["correct_answer"],
        explanation=result.get("explanation"),
        key_points=result.get("key_points", []),
        ability_percentile=result.get("ability_percentile", 50)
    )


@router.get("/questions/performance")
async def get_question_performance(
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_question_bank_service)
):
    """Get performance breakdown by category."""
    return await service.get_category_performance(user_id)


@router.get("/questions/ability")
async def get_ability_estimate(
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_question_bank_service)
):
    """Get current ability estimate."""
    ability = await service.get_user_ability(user_id)
    return {
        "theta": ability.theta,
        "standard_error": ability.standard_error,
        "questions_answered": ability.questions_answered,
        "confidence_interval": ability.confidence_interval
    }


# ============================================================================
# EXAM SIMULATION ENDPOINTS
# ============================================================================

@router.post("/exam/start")
async def start_exam(
    request: ExamStartRequest,
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_exam_service)
):
    """Start a new exam simulation."""
    from .services.exam_simulation import ExamType
    
    try:
        exam_type = ExamType(request.exam_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid exam type: {request.exam_type}")
    
    session = await service.start_exam(user_id, exam_type)
    
    return {
        "exam_id": str(session.id),
        "exam_type": session.exam_type.value,
        "total_questions": len(session.questions),
        "time_limit_minutes": session.time_limit_minutes,
        "started_at": session.started_at.isoformat()
    }


@router.get("/exam/{exam_id}/question", response_model=ExamQuestionResponse)
async def get_exam_question(
    exam_id: str,
    service = Depends(get_exam_service)
):
    """Get current question in exam."""
    question = await service.get_current_question(UUID(exam_id))
    
    if not question:
        raise HTTPException(status_code=404, detail="No more questions or exam not found")
    
    return ExamQuestionResponse(**question)


@router.post("/exam/{exam_id}/answer")
async def submit_exam_answer(
    exam_id: str,
    submission: ExamAnswerSubmission,
    service = Depends(get_exam_service)
):
    """Submit answer for current exam question."""
    result = await service.submit_answer(
        exam_id=UUID(exam_id),
        question_id=UUID(submission.question_id),
        answer=submission.answer,
        time_spent=submission.time_spent_seconds
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/exam/{exam_id}/complete", response_model=ExamResult)
async def complete_exam(
    exam_id: str,
    service = Depends(get_exam_service)
):
    """Complete exam and get results."""
    try:
        result = await service.complete_exam(UUID(exam_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return ExamResult(
        exam_id=str(result.exam_id),
        score_percentage=result.score_percentage,
        passed=result.passed,
        pass_threshold=result.pass_threshold,
        total_questions=result.total_questions,
        correct_answers=result.correct_answers,
        duration_minutes=result.duration_minutes,
        category_scores=result.category_scores,
        strengths=result.strengths,
        weaknesses=result.weaknesses,
        recommendations=result.recommendations
    )


@router.delete("/exam/{exam_id}")
async def abandon_exam(
    exam_id: str,
    service = Depends(get_exam_service)
):
    """Abandon exam without completing."""
    success = await service.abandon_exam(UUID(exam_id))
    if not success:
        raise HTTPException(status_code=404, detail="Exam not found")
    return {"success": True}


@router.get("/exam/history")
async def get_exam_history(
    limit: int = Query(10, le=50),
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_exam_service)
):
    """Get exam history."""
    return await service.get_exam_history(user_id, limit)


@router.get("/exam/predict", response_model=PassPrediction)
async def predict_pass_probability(
    user_id: UUID = Depends(get_current_user_id),
    service = Depends(get_exam_service)
):
    """Predict probability of passing FRCSC exam."""
    return await service.predict_pass_probability(user_id)


# ============================================================================
# FSRS CONFIGURATION ENDPOINTS
# ============================================================================

@router.post("/fsrs/exam-date")
async def set_exam_date(
    exam_date: str,  # ISO format: YYYY-MM-DD
    user_id: UUID = Depends(get_current_user_id)
):
    """Set exam date for accelerated scheduling."""
    from datetime import datetime
    
    try:
        parsed = datetime.strptime(exam_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Would store in user settings
    return {
        "exam_date": exam_date,
        "days_until": (parsed - date.today()).days,
        "scheduling_mode": "accelerated"
    }


@router.get("/fsrs/forecast")
async def get_review_forecast(
    days: int = Query(30, le=90),
    user_id: UUID = Depends(get_current_user_id)
):
    """Get review load forecast for upcoming days."""
    # Would call FSRSEnhanced.forecast_reviews
    return {
        "forecast": [
            {"date": (date.today()).isoformat(), "due_count": 25},
            # ... more days
        ],
        "total_cards": 500,
        "average_daily": 20
    }
