"""
Exam Simulation Service

FRCSC exam simulation with:
- Written exam mode (timed MCQ)
- Oral exam mode (case-based Q&A)
- Performance analytics
- Pass prediction
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum


class ExamType(str, Enum):
    WRITTEN = "written"
    ORAL = "oral"
    MOCK_FULL = "mock_full"


class ExamStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class ExamConfig:
    """Exam configuration."""
    exam_type: ExamType
    time_limit_minutes: int
    total_questions: int
    pass_threshold: float
    category_weights: Dict[str, float]


@dataclass
class ExamQuestion:
    """Question in an exam context."""
    question_id: UUID
    stem: str
    options: Optional[List[Dict]]
    category: str
    allocated_seconds: int


@dataclass
class ExamResponse:
    """User's response to an exam question."""
    question_id: UUID
    user_answer: str
    time_spent_seconds: int
    submitted_at: datetime


@dataclass
class ExamSession:
    """Active exam session."""
    id: UUID
    user_id: UUID
    exam_type: ExamType
    status: ExamStatus
    
    started_at: datetime
    time_limit_minutes: int
    
    questions: List[ExamQuestion]
    responses: List[ExamResponse] = field(default_factory=list)
    current_index: int = 0
    
    @property
    def time_remaining(self) -> int:
        """Seconds remaining."""
        elapsed = (datetime.now() - self.started_at).total_seconds()
        remaining = self.time_limit_minutes * 60 - elapsed
        return max(0, int(remaining))
    
    @property
    def is_expired(self) -> bool:
        return self.time_remaining <= 0


@dataclass
class ExamResult:
    """Final exam results."""
    exam_id: UUID
    user_id: UUID
    exam_type: ExamType
    
    total_questions: int
    correct_answers: int
    score_percentage: float
    pass_threshold: float
    passed: bool
    
    duration_minutes: int
    category_scores: Dict[str, Dict]
    
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


# FRCSC exam configurations
EXAM_CONFIGS = {
    ExamType.WRITTEN: ExamConfig(
        exam_type=ExamType.WRITTEN,
        time_limit_minutes=180,  # 3 hours
        total_questions=100,
        pass_threshold=0.70,
        category_weights={
            "neuro_oncology": 0.20,
            "vascular": 0.17,
            "trauma": 0.13,
            "spine": 0.13,
            "pediatrics": 0.10,
            "functional": 0.08,
            "peripheral_nerve": 0.05,
            "anatomy": 0.04,
            "neurology": 0.04,
            "radiology": 0.03,
            "pathology": 0.02,
            "research": 0.01,
        }
    ),
    ExamType.ORAL: ExamConfig(
        exam_type=ExamType.ORAL,
        time_limit_minutes=45,
        total_questions=6,  # 6 cases
        pass_threshold=0.65,
        category_weights={
            "neuro_oncology": 0.20,
            "vascular": 0.20,
            "trauma": 0.15,
            "spine": 0.15,
            "pediatrics": 0.15,
            "functional": 0.15,
        }
    ),
    ExamType.MOCK_FULL: ExamConfig(
        exam_type=ExamType.MOCK_FULL,
        time_limit_minutes=240,
        total_questions=150,
        pass_threshold=0.70,
        category_weights={
            "neuro_oncology": 0.20,
            "vascular": 0.17,
            "trauma": 0.13,
            "spine": 0.13,
            "pediatrics": 0.10,
            "functional": 0.08,
            "peripheral_nerve": 0.05,
            "anatomy": 0.04,
            "neurology": 0.04,
            "radiology": 0.03,
            "pathology": 0.02,
            "research": 0.01,
        }
    ),
}


class ExamSimulationService:
    """Service for managing exam simulations."""
    
    def __init__(self, repository=None, question_bank=None):
        self.repo = repository
        self.qb = question_bank
        
        # Active sessions cache
        self._sessions: Dict[str, ExamSession] = {}
    
    async def start_exam(
        self,
        user_id: UUID,
        exam_type: ExamType = ExamType.WRITTEN
    ) -> ExamSession:
        """Start a new exam session."""
        config = EXAM_CONFIGS.get(exam_type)
        if not config:
            raise ValueError(f"Unknown exam type: {exam_type}")
        
        # Get questions
        if self.qb:
            questions = await self.qb.get_exam_questions(
                user_id=user_id,
                exam_type=exam_type.value,
                count=config.total_questions
            )
        else:
            # Mock questions for testing
            questions = self._generate_mock_questions(config)
        
        # Build exam questions
        seconds_per_question = (config.time_limit_minutes * 60) // config.total_questions
        
        exam_questions = [
            ExamQuestion(
                question_id=q.id if hasattr(q, 'id') else uuid4(),
                stem=q.stem if hasattr(q, 'stem') else q.get('stem', ''),
                options=q.options if hasattr(q, 'options') else q.get('options'),
                category=str(q.category_id) if hasattr(q, 'category_id') else q.get('category', ''),
                allocated_seconds=seconds_per_question
            )
            for q in questions
        ]
        
        # Create session
        session = ExamSession(
            id=uuid4(),
            user_id=user_id,
            exam_type=exam_type,
            status=ExamStatus.IN_PROGRESS,
            started_at=datetime.now(),
            time_limit_minutes=config.time_limit_minutes,
            questions=exam_questions
        )
        
        # Cache and persist
        self._sessions[str(session.id)] = session
        
        if self.repo:
            await self.repo.create_exam_session(session)
        
        return session
    
    async def get_current_question(
        self,
        exam_id: UUID
    ) -> Optional[Dict]:
        """Get current question in exam."""
        session = self._get_session(exam_id)
        if not session:
            return None
        
        if session.is_expired:
            await self.complete_exam(exam_id)
            return None
        
        if session.current_index >= len(session.questions):
            return None
        
        question = session.questions[session.current_index]
        
        return {
            "question_number": session.current_index + 1,
            "total_questions": len(session.questions),
            "question_id": str(question.question_id),
            "stem": question.stem,
            "options": question.options,
            "time_remaining": session.time_remaining,
            "allocated_seconds": question.allocated_seconds
        }
    
    async def submit_answer(
        self,
        exam_id: UUID,
        question_id: UUID,
        answer: str,
        time_spent: int
    ) -> Dict:
        """Submit answer for current question."""
        session = self._get_session(exam_id)
        if not session:
            return {"error": "Exam not found"}
        
        if session.is_expired:
            await self.complete_exam(exam_id)
            return {"error": "Exam time expired"}
        
        # Record response
        response = ExamResponse(
            question_id=question_id,
            user_answer=answer,
            time_spent_seconds=time_spent,
            submitted_at=datetime.now()
        )
        session.responses.append(response)
        
        # Move to next question
        session.current_index += 1
        
        # Check if exam complete
        if session.current_index >= len(session.questions):
            result = await self.complete_exam(exam_id)
            return {
                "exam_complete": True,
                "result": result
            }
        
        return {
            "exam_complete": False,
            "next_question": session.current_index + 1,
            "questions_remaining": len(session.questions) - session.current_index,
            "time_remaining": session.time_remaining
        }
    
    async def complete_exam(self, exam_id: UUID) -> ExamResult:
        """Complete exam and calculate results."""
        session = self._get_session(exam_id)
        if not session:
            raise ValueError("Exam not found")
        
        session.status = ExamStatus.COMPLETED
        config = EXAM_CONFIGS[session.exam_type]
        
        # Grade responses
        correct = 0
        category_scores = {}
        
        for i, question in enumerate(session.questions):
            response = next(
                (r for r in session.responses if r.question_id == question.question_id),
                None
            )
            
            if response:
                is_correct = await self._check_answer(question.question_id, response.user_answer)
                if is_correct:
                    correct += 1
                
                # Track by category
                cat = question.category
                if cat not in category_scores:
                    category_scores[cat] = {"total": 0, "correct": 0}
                category_scores[cat]["total"] += 1
                if is_correct:
                    category_scores[cat]["correct"] += 1
        
        # Calculate scores
        score = correct / len(session.questions) if session.questions else 0
        passed = score >= config.pass_threshold
        
        duration = int((datetime.now() - session.started_at).total_seconds() / 60)
        
        # Analyze strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for cat, stats in category_scores.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["accuracy"] = accuracy
            
            if accuracy >= 0.80:
                strengths.append(cat)
            elif accuracy < 0.60:
                weaknesses.append(cat)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(weaknesses, score, passed)
        
        result = ExamResult(
            exam_id=session.id,
            user_id=session.user_id,
            exam_type=session.exam_type,
            total_questions=len(session.questions),
            correct_answers=correct,
            score_percentage=round(score * 100, 1),
            pass_threshold=config.pass_threshold,
            passed=passed,
            duration_minutes=duration,
            category_scores=category_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
        
        # Persist result
        if self.repo:
            await self.repo.save_exam_result(result)
        
        # Clean up session
        del self._sessions[str(exam_id)]
        
        return result
    
    async def abandon_exam(self, exam_id: UUID) -> bool:
        """Abandon exam without completing."""
        session = self._get_session(exam_id)
        if not session:
            return False
        
        session.status = ExamStatus.ABANDONED
        
        if self.repo:
            await self.repo.update_exam_status(exam_id, ExamStatus.ABANDONED)
        
        del self._sessions[str(exam_id)]
        return True
    
    async def get_exam_history(
        self,
        user_id: UUID,
        limit: int = 10
    ) -> List[Dict]:
        """Get user's exam history."""
        if self.repo:
            return await self.repo.get_exam_history(user_id, limit)
        return []
    
    async def predict_pass_probability(self, user_id: UUID) -> Dict:
        """Predict probability of passing based on performance."""
        if not self.repo:
            return {"probability": 0.5, "confidence": "low"}
        
        # Get historical data
        history = await self.repo.get_exam_history(user_id, limit=5)
        category_perf = await self.repo.get_category_performance(user_id)
        
        if not history:
            return {"probability": 0.5, "confidence": "low", "reason": "No exam history"}
        
        # Calculate weighted average
        recent_scores = [h.get("score_percentage", 0) for h in history]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Factor in category weaknesses
        weights = EXAM_CONFIGS[ExamType.WRITTEN].category_weights
        weighted_score = 0
        
        for cat, weight in weights.items():
            cat_accuracy = category_perf.get(cat, {}).get("accuracy", 0.5)
            weighted_score += cat_accuracy * weight
        
        # Combine metrics
        probability = 0.4 * (avg_score / 100) + 0.6 * weighted_score
        
        # Determine confidence
        if len(history) >= 3:
            confidence = "high"
        elif len(history) >= 1:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "probability": round(probability, 2),
            "confidence": confidence,
            "factors": {
                "average_score": round(avg_score, 1),
                "weighted_category_score": round(weighted_score * 100, 1),
                "exams_taken": len(history)
            }
        }
    
    def _get_session(self, exam_id: UUID) -> Optional[ExamSession]:
        """Get session from cache."""
        return self._sessions.get(str(exam_id))
    
    async def _check_answer(self, question_id: UUID, answer: str) -> bool:
        """Check if answer is correct."""
        if self.qb:
            result = await self.qb.submit_answer(
                user_id=uuid4(),  # Not tracking for grading
                question_id=question_id,
                user_answer=answer,
                time_spent=0
            )
            return result.get("is_correct", False)
        
        # Mock: 70% correct for testing
        return random.random() < 0.7
    
    def _generate_recommendations(
        self,
        weaknesses: List[str],
        score: float,
        passed: bool
    ) -> List[str]:
        """Generate study recommendations."""
        recs = []
        
        if not passed:
            recs.append("Focus on weak categories before retaking exam")
        
        for cat in weaknesses[:3]:
            recs.append(f"Review {cat} - complete additional practice questions")
        
        if score < 0.60:
            recs.append("Consider comprehensive review of core topics")
        elif score < 0.70:
            recs.append("Target specific weak areas with focused study")
        elif not passed:
            recs.append("You're close! Focus on your 2-3 weakest categories")
        
        if len(weaknesses) > 3:
            recs.append("Consider structured study schedule covering all categories")
        
        return recs
    
    def _generate_mock_questions(self, config: ExamConfig) -> List[Dict]:
        """Generate mock questions for testing."""
        questions = []
        for i in range(config.total_questions):
            questions.append({
                "id": uuid4(),
                "stem": f"Mock question {i+1}",
                "options": [
                    {"label": "A", "text": "Option A", "is_correct": i % 4 == 0},
                    {"label": "B", "text": "Option B", "is_correct": i % 4 == 1},
                    {"label": "C", "text": "Option C", "is_correct": i % 4 == 2},
                    {"label": "D", "text": "Option D", "is_correct": i % 4 == 3},
                ],
                "category": list(config.category_weights.keys())[i % len(config.category_weights)]
            })
        return questions
