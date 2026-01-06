"""
Question Bank Service

FRCSC question bank with:
- IRT-based adaptive question selection
- Category-weighted sampling
- Performance analytics
- Exam simulation
"""

import math
import random
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from uuid import UUID
from dataclasses import dataclass
from enum import Enum


class QuestionFormat(str, Enum):
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    ORAL = "oral"


@dataclass
class Question:
    """Question with IRT parameters."""
    id: UUID
    question_code: str
    category_id: UUID
    stem: str
    options: Optional[List[Dict]]  # MCQ options
    answer: str
    explanation: Optional[str]
    format: QuestionFormat
    yield_rating: int  # 1-3
    cognitive_level: int  # 1-6 (Bloom's)
    
    # IRT parameters
    difficulty: float = 0.0  # b parameter (-3 to +3)
    discrimination: float = 1.0  # a parameter (0.5 to 2.5)
    guessing: float = 0.25  # c parameter for MCQ
    
    year_asked: Optional[int] = None
    key_points: List[str] = None


@dataclass
class QuestionAttempt:
    """Record of a question attempt."""
    question_id: UUID
    user_answer: str
    is_correct: bool
    time_spent_seconds: int
    ability_estimate: float  # User's ability after this attempt


@dataclass
class AbilityEstimate:
    """User's estimated ability level."""
    theta: float  # Ability parameter (-3 to +3)
    standard_error: float
    questions_answered: int
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        return (
            self.theta - 1.96 * self.standard_error,
            self.theta + 1.96 * self.standard_error
        )


class IRTEngine:
    """
    Item Response Theory engine for adaptive testing.
    Uses 3-parameter logistic model (3PL).
    """
    
    @staticmethod
    def probability_correct(theta: float, a: float, b: float, c: float) -> float:
        """
        Calculate probability of correct response.
        
        P(X=1|θ) = c + (1-c) / (1 + exp(-a(θ-b)))
        
        Args:
            theta: Ability parameter
            a: Discrimination
            b: Difficulty
            c: Guessing parameter
        """
        exponent = -a * (theta - b)
        return c + (1 - c) / (1 + math.exp(exponent))
    
    @staticmethod
    def information(theta: float, a: float, b: float, c: float) -> float:
        """
        Calculate Fisher information at ability level.
        
        Higher information = more precise measurement at this ability.
        """
        p = IRTEngine.probability_correct(theta, a, b, c)
        q = 1 - p
        
        if p <= c or p >= 1:
            return 0.0
        
        # Information formula for 3PL
        numerator = (a ** 2) * ((p - c) ** 2) * q
        denominator = ((1 - c) ** 2) * p
        
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def update_ability(
        theta: float,
        response: bool,
        a: float,
        b: float,
        c: float,
        learning_rate: float = 0.3
    ) -> float:
        """
        Update ability estimate after a response (EAP estimation simplified).
        """
        p = IRTEngine.probability_correct(theta, a, b, c)
        
        # Gradient of log-likelihood
        if response:
            gradient = a * (1 - p) * (p - c) / (p * (1 - c))
        else:
            gradient = -a * (p - c) / ((1 - p) * (1 - c))
        
        # Update with learning rate and prior pull toward 0
        new_theta = theta + learning_rate * gradient - 0.1 * theta
        
        # Bound to reasonable range
        return max(-3.0, min(3.0, new_theta))
    
    @staticmethod
    def standard_error(theta: float, questions: List[Question]) -> float:
        """Calculate standard error of ability estimate."""
        total_info = sum(
            IRTEngine.information(theta, q.discrimination, q.difficulty, q.guessing)
            for q in questions
        )
        
        if total_info <= 0:
            return 1.0
        
        return 1.0 / math.sqrt(total_info)


class QuestionSelector:
    """Selects optimal questions for adaptive testing."""
    
    def __init__(self, category_weights: Dict[str, float] = None):
        self.category_weights = category_weights or {}
    
    def select_next(
        self,
        available: List[Question],
        theta: float,
        answered_ids: set,
        target_category: Optional[str] = None,
        prioritize_high_yield: bool = True
    ) -> Optional[Question]:
        """
        Select next question maximizing information.
        
        Strategy:
        1. Filter out already answered
        2. Prioritize target category if specified
        3. Select question with maximum information at current ability
        4. Tie-break by yield rating
        """
        candidates = [q for q in available if q.id not in answered_ids]
        
        if not candidates:
            return None
        
        if target_category:
            category_candidates = [q for q in candidates if str(q.category_id) == target_category]
            if category_candidates:
                candidates = category_candidates
        
        # Score each candidate
        scored = []
        for q in candidates:
            info = IRTEngine.information(theta, q.discrimination, q.difficulty, q.guessing)
            
            # Boost high-yield questions
            yield_boost = q.yield_rating * 0.1 if prioritize_high_yield else 0
            
            score = info + yield_boost
            scored.append((score, q))
        
        # Select best
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Add slight randomization among top candidates
        top_n = min(5, len(scored))
        return random.choice([q for _, q in scored[:top_n]])
    
    def select_balanced_set(
        self,
        available: List[Question],
        count: int,
        category_weights: Dict[str, float] = None
    ) -> List[Question]:
        """
        Select questions balanced across categories per FRCSC weights.
        """
        weights = category_weights or self.category_weights
        
        # Group by category
        by_category = {}
        for q in available:
            cat = str(q.category_id)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)
        
        # Allocate questions per category
        selected = []
        for cat, cat_weight in weights.items():
            n_cat = max(1, int(count * cat_weight))
            
            if cat in by_category:
                cat_questions = by_category[cat]
                random.shuffle(cat_questions)
                selected.extend(cat_questions[:n_cat])
        
        # Fill remaining slots
        remaining = count - len(selected)
        if remaining > 0:
            unused = [q for q in available if q not in selected]
            random.shuffle(unused)
            selected.extend(unused[:remaining])
        
        random.shuffle(selected)
        return selected[:count]


class QuestionBankService:
    """
    Main service for question bank operations.
    
    Integrates with repository for database operations.
    """
    
    def __init__(self, repository=None):
        self.repo = repository
        self.irt = IRTEngine()
        self.selector = QuestionSelector()
        
        # User ability cache
        self._ability_cache: Dict[str, AbilityEstimate] = {}
    
    async def get_adaptive_question(
        self,
        user_id: UUID,
        category: Optional[str] = None,
        exclude_ids: List[UUID] = None
    ) -> Optional[Question]:
        """Get next optimal question for user."""
        # Get user's current ability
        ability = await self.get_user_ability(user_id)
        
        # Get available questions
        if self.repo:
            questions = await self.repo.get_questions(
                category=category,
                exclude_ids=exclude_ids or []
            )
        else:
            questions = []
        
        # Select optimal question
        return self.selector.select_next(
            available=questions,
            theta=ability.theta,
            answered_ids=set(exclude_ids or []),
            target_category=category
        )
    
    async def submit_answer(
        self,
        user_id: UUID,
        question_id: UUID,
        user_answer: str,
        time_spent: int
    ) -> Dict:
        """
        Submit answer and update ability estimate.
        
        Returns result with correctness, explanation, new ability.
        """
        # Get question
        if self.repo:
            question = await self.repo.get_question(question_id)
        else:
            return {"error": "Repository not configured"}
        
        if not question:
            return {"error": "Question not found"}
        
        # Check correctness
        is_correct = self._check_answer(question, user_answer)
        
        # Update ability estimate
        ability = await self.get_user_ability(user_id)
        new_theta = self.irt.update_ability(
            theta=ability.theta,
            response=is_correct,
            a=question.discrimination,
            b=question.difficulty,
            c=question.guessing
        )
        
        # Update cache and database
        new_ability = AbilityEstimate(
            theta=new_theta,
            standard_error=ability.standard_error * 0.95,  # SE decreases with more data
            questions_answered=ability.questions_answered + 1
        )
        self._ability_cache[str(user_id)] = new_ability
        
        # Record attempt
        if self.repo:
            await self.repo.record_attempt(
                user_id=user_id,
                question_id=question_id,
                user_answer=user_answer,
                is_correct=is_correct,
                time_spent=time_spent,
                ability_estimate=new_theta
            )
        
        return {
            "is_correct": is_correct,
            "correct_answer": question.answer,
            "explanation": question.explanation,
            "key_points": question.key_points,
            "new_ability": new_theta,
            "ability_percentile": self._theta_to_percentile(new_theta)
        }
    
    async def get_user_ability(self, user_id: UUID) -> AbilityEstimate:
        """Get user's current ability estimate."""
        cache_key = str(user_id)
        
        if cache_key in self._ability_cache:
            return self._ability_cache[cache_key]
        
        # Load from database or initialize
        if self.repo:
            history = await self.repo.get_user_attempts(user_id, limit=100)
            if history:
                # Use most recent estimate
                theta = history[0].ability_estimate
                se = 1.0 / math.sqrt(max(1, len(history)))
                return AbilityEstimate(theta=theta, standard_error=se, questions_answered=len(history))
        
        # Default for new users
        return AbilityEstimate(theta=0.0, standard_error=1.0, questions_answered=0)
    
    async def get_exam_questions(
        self,
        user_id: UUID,
        exam_type: str = "written",
        count: int = 100
    ) -> List[Question]:
        """Generate exam simulation questions."""
        # FRCSC category weights
        weights = {
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
        
        if self.repo:
            available = await self.repo.get_all_questions()
        else:
            available = []
        
        return self.selector.select_balanced_set(
            available=available,
            count=count,
            category_weights=weights
        )
    
    async def get_category_performance(
        self,
        user_id: UUID
    ) -> Dict[str, Dict]:
        """Get performance breakdown by category."""
        if not self.repo:
            return {}
        
        attempts = await self.repo.get_user_attempts(user_id, limit=1000)
        
        # Group by category
        by_category = {}
        for attempt in attempts:
            cat = attempt.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "correct": 0}
            by_category[cat]["total"] += 1
            if attempt.is_correct:
                by_category[cat]["correct"] += 1
        
        # Calculate accuracy
        result = {}
        for cat, stats in by_category.items():
            result[cat] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
        
        return result
    
    def _check_answer(self, question: Question, user_answer: str) -> bool:
        """Check if answer is correct."""
        if question.format == QuestionFormat.MCQ:
            # MCQ: check selected option
            if question.options:
                for opt in question.options:
                    if opt.get("is_correct") and opt.get("label") == user_answer.upper():
                        return True
            return False
        else:
            # Short answer: fuzzy match
            return user_answer.lower().strip() == question.answer.lower().strip()
    
    def _theta_to_percentile(self, theta: float) -> int:
        """Convert ability to percentile (assuming normal distribution)."""
        # Standard normal CDF approximation
        z = theta
        if z < -4:
            return 0
        if z > 4:
            return 100
        
        # Approximation
        t = 1 / (1 + 0.2316419 * abs(z))
        d = 0.3989423 * math.exp(-z * z / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        
        if z > 0:
            return int((1 - p) * 100)
        return int(p * 100)
