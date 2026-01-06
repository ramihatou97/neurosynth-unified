# src/learning/nprss/fsrs.py
"""
Free Spaced Repetition Scheduler (FSRS) v4 Implementation

20-30% more efficient than SM-2 algorithm.
Based on: https://github.com/open-spaced-repetition/fsrs4anki

Key Concepts:
- Stability (S): Expected time to forget to ~90% retrievability
- Difficulty (D): Inherent difficulty of the card (1-10)
- Retrievability (R): Current probability of recall (0-1)

Medical Optimization:
- Higher target retention (90%) for patient safety
- Integration with R1-R7 macro schedule for procedural learning
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Optional, List, Tuple, Dict, Any
from copy import deepcopy


class Rating(IntEnum):
    """User rating for a review"""
    AGAIN = 1   # Complete failure - couldn't recall
    HARD = 2    # Significant difficulty - recalled with effort
    GOOD = 3    # Correct with moderate effort
    EASY = 4    # Effortless recall


class State(IntEnum):
    """Card learning state"""
    NEW = 0
    LEARNING = 1
    REVIEW = 2
    RELEARNING = 3


@dataclass
class FSRSParameters:
    """
    FSRS v4 parameters - can be optimized per user

    Default values from FSRS v4 research paper trained on 700M reviews.
    """
    # 17 weight parameters
    w: List[float] = field(default_factory=lambda: [
        0.4,    # w[0]: Initial stability for AGAIN
        0.6,    # w[1]: Initial stability for HARD
        2.4,    # w[2]: Initial stability for GOOD
        5.8,    # w[3]: Initial stability for EASY
        4.93,   # w[4]: Initial difficulty mean
        0.94,   # w[5]: Difficulty adjustment based on rating
        0.86,   # w[6]: Difficulty change rate
        0.01,   # w[7]: Difficulty mean reversion rate
        1.49,   # w[8]: Stability increase base
        0.14,   # w[9]: Stability decrease rate with higher S
        0.94,   # w[10]: Stability increase from retrievability
        2.18,   # w[11]: Stability after lapse base
        0.05,   # w[12]: Stability after lapse difficulty factor
        0.34,   # w[13]: Stability after lapse stability factor
        1.26,   # w[14]: Stability after lapse retrievability factor
        0.29,   # w[15]: Hard penalty
        2.61    # w[16]: Easy bonus
    ])

    # Target retention rate (90% recommended for medical content)
    request_retention: float = 0.90

    # Maximum interval in days
    maximum_interval: int = 365

    # Learning/relearning steps (in minutes)
    learning_steps: List[int] = field(default_factory=lambda: [1, 10])
    relearning_steps: List[int] = field(default_factory=lambda: [10])


@dataclass
class MemoryState:
    """Current memory state for a card"""
    difficulty: float = 0.3
    stability: float = 1.0
    state: State = State.NEW
    step: int = 0  # Current step in learning/relearning
    due: Optional[datetime] = None
    last_review: Optional[datetime] = None
    reps: int = 0
    lapses: int = 0

    def copy(self) -> 'MemoryState':
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "stability": self.stability,
            "state": self.state.value if isinstance(self.state, State) else self.state,
            "step": self.step,
            "due": self.due.isoformat() if self.due else None,
            "last_review": self.last_review.isoformat() if self.last_review else None,
            "reps": self.reps,
            "lapses": self.lapses,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryState':
        return cls(
            difficulty=data.get("difficulty", 0.3),
            stability=data.get("stability", 1.0),
            state=State(data.get("state", 0)),
            step=data.get("step", 0),
            due=datetime.fromisoformat(data["due"]) if data.get("due") else None,
            last_review=datetime.fromisoformat(data["last_review"]) if data.get("last_review") else None,
            reps=data.get("reps", 0),
            lapses=data.get("lapses", 0),
        )


@dataclass
class ReviewLog:
    """Log of a single review"""
    card_id: str
    rating: Rating
    state: State
    due: datetime
    stability: float
    difficulty: float
    elapsed_days: int
    scheduled_days: int
    review_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "rating": self.rating.value,
            "state": self.state.value,
            "due": self.due.isoformat(),
            "stability": self.stability,
            "difficulty": self.difficulty,
            "elapsed_days": self.elapsed_days,
            "scheduled_days": self.scheduled_days,
            "review_time": self.review_time.isoformat(),
        }


class FSRS:
    """
    FSRS v4 Algorithm Implementation

    Key concepts:
    - Stability (S): Expected time to forget to ~90% retrievability
    - Difficulty (D): Inherent difficulty of the card (1-10)
    - Retrievability (R): Current probability of recall (0-1)

    Usage:
        fsrs = FSRS()
        memory = MemoryState()

        # Review a card
        new_memory, log = fsrs.review(memory, Rating.GOOD)

        # Check retrievability
        r = fsrs.get_retrievability(new_memory)
    """

    def __init__(self, params: Optional[FSRSParameters] = None):
        self.p = params or FSRSParameters()

    def review(
        self,
        memory: MemoryState,
        rating: Rating,
        review_time: Optional[datetime] = None
    ) -> Tuple[MemoryState, ReviewLog]:
        """
        Process a review and return updated memory state + log

        Args:
            memory: Current memory state
            rating: User's rating (AGAIN, HARD, GOOD, EASY)
            review_time: When the review occurred (default: now)

        Returns:
            Tuple of (new_memory_state, review_log)
        """
        review_time = review_time or datetime.now()
        old_state = memory.copy()

        if memory.state == State.NEW:
            new_memory = self._review_new(memory, rating, review_time)
        elif memory.state == State.LEARNING or memory.state == State.RELEARNING:
            new_memory = self._review_learning(memory, rating, review_time)
        else:  # State.REVIEW
            new_memory = self._review_review(memory, rating, review_time)

        # Calculate elapsed days
        elapsed_days = 0
        if old_state.last_review:
            elapsed_days = (review_time - old_state.last_review).days

        # Calculate scheduled days
        scheduled_days = 0
        if new_memory.due:
            scheduled_days = (new_memory.due - review_time).days

        log = ReviewLog(
            card_id="",  # Set by caller
            rating=rating,
            state=old_state.state,
            due=old_state.due or review_time,
            stability=new_memory.stability,
            difficulty=new_memory.difficulty,
            elapsed_days=elapsed_days,
            scheduled_days=scheduled_days,
            review_time=review_time
        )

        return new_memory, log

    def _review_new(
        self,
        memory: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> MemoryState:
        """Handle first review of a new card"""
        new = memory.copy()

        # Initialize difficulty
        new.difficulty = self._init_difficulty(rating)

        # Initialize stability
        new.stability = self._init_stability(rating)

        new.reps = 1
        new.last_review = review_time

        if rating == Rating.AGAIN:
            new.state = State.LEARNING
            new.step = 0
            new.lapses += 1
            # Schedule for first learning step
            new.due = review_time + timedelta(minutes=self.p.learning_steps[0])
        elif rating == Rating.HARD:
            new.state = State.LEARNING
            new.step = 0
            new.due = review_time + timedelta(minutes=self.p.learning_steps[0])
        elif rating == Rating.GOOD:
            new.state = State.LEARNING
            new.step = len(self.p.learning_steps) - 1
            # Graduate to review
            if len(self.p.learning_steps) <= 1:
                new.state = State.REVIEW
                new.due = review_time + timedelta(days=self._next_interval(new.stability))
            else:
                new.due = review_time + timedelta(minutes=self.p.learning_steps[-1])
        else:  # EASY
            new.state = State.REVIEW
            new.due = review_time + timedelta(days=self._next_interval(new.stability))

        return new

    def _review_learning(
        self,
        memory: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> MemoryState:
        """Handle review during learning/relearning phase"""
        new = memory.copy()
        new.last_review = review_time
        new.reps += 1

        steps = self.p.learning_steps if memory.state == State.LEARNING else self.p.relearning_steps

        if rating == Rating.AGAIN:
            new.step = 0
            new.lapses += 1
            new.due = review_time + timedelta(minutes=steps[0])
        elif rating == Rating.HARD:
            # Stay at current step
            new.due = review_time + timedelta(minutes=steps[min(new.step, len(steps)-1)])
        elif rating == Rating.GOOD:
            new.step += 1
            if new.step >= len(steps):
                # Graduate to review
                new.state = State.REVIEW
                new.stability = self._next_stability_success(
                    new.stability, new.difficulty, 1.0, rating
                )
                new.due = review_time + timedelta(days=self._next_interval(new.stability))
            else:
                new.due = review_time + timedelta(minutes=steps[new.step])
        else:  # EASY
            # Immediate graduation
            new.state = State.REVIEW
            new.stability = self._next_stability_success(
                new.stability, new.difficulty, 1.0, rating
            )
            new.due = review_time + timedelta(days=self._next_interval(new.stability))

        return new

    def _review_review(
        self,
        memory: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> MemoryState:
        """Handle review of graduated card"""
        new = memory.copy()
        new.last_review = review_time
        new.reps += 1

        # Calculate elapsed time and retrievability
        elapsed_days = (review_time - memory.last_review).days if memory.last_review else 0
        retrievability = self._retrievability(memory.stability, elapsed_days)

        # Update difficulty
        new.difficulty = self._next_difficulty(memory.difficulty, rating)

        if rating == Rating.AGAIN:
            # Lapse
            new.lapses += 1
            new.state = State.RELEARNING
            new.step = 0
            new.stability = self._next_stability_lapse(
                memory.stability, memory.difficulty, retrievability
            )
            new.due = review_time + timedelta(minutes=self.p.relearning_steps[0])
        else:
            # Successful recall
            new.stability = self._next_stability_success(
                memory.stability, memory.difficulty, retrievability, rating
            )
            new.due = review_time + timedelta(days=self._next_interval(new.stability))

        return new

    # =========================================================================
    # FSRS Formulas
    # =========================================================================

    def _init_difficulty(self, rating: Rating) -> float:
        """Calculate initial difficulty based on first rating"""
        d = self.p.w[4] - (rating.value - 3) * self.p.w[5]
        return min(max(d, 1.0), 10.0)

    def _init_stability(self, rating: Rating) -> float:
        """Calculate initial stability based on first rating"""
        return self.p.w[rating.value - 1]

    def _retrievability(self, stability: float, elapsed_days: int) -> float:
        """
        Calculate current retrievability (forgetting curve)
        R(t) = (1 + t/(9*S))^(-1)
        """
        if elapsed_days <= 0:
            return 1.0
        return math.pow(1 + elapsed_days / (9 * stability), -1)

    def _next_difficulty(self, d: float, rating: Rating) -> float:
        """
        Calculate next difficulty
        D' = D - w6 * (R - 3)
        With mean reversion: D'' = w7 * D0 + (1 - w7) * D'
        """
        d_prime = d - self.p.w[6] * (rating.value - 3)
        d_prime_prime = self.p.w[7] * self.p.w[4] + (1 - self.p.w[7]) * d_prime
        return min(max(d_prime_prime, 1.0), 10.0)

    def _next_stability_success(
        self,
        s: float,
        d: float,
        r: float,
        rating: Rating
    ) -> float:
        """
        Calculate next stability after successful recall

        S' = S * (1 + e^w8 * (11-D) * S^(-w9) * (e^((1-R)*w10) - 1) * hard_penalty * easy_bonus)
        """
        hard_penalty = self.p.w[15] if rating == Rating.HARD else 1.0
        easy_bonus = self.p.w[16] if rating == Rating.EASY else 1.0

        new_s = s * (
            1 +
            math.exp(self.p.w[8]) *
            (11 - d) *
            math.pow(s, -self.p.w[9]) *
            (math.exp((1 - r) * self.p.w[10]) - 1) *
            hard_penalty *
            easy_bonus
        )

        return max(new_s, 0.1)  # Minimum stability

    def _next_stability_lapse(
        self,
        s: float,
        d: float,
        r: float
    ) -> float:
        """
        Calculate next stability after lapse (AGAIN rating)

        S' = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^((1-R)*w14)
        """
        new_s = (
            self.p.w[11] *
            math.pow(d, -self.p.w[12]) *
            (math.pow(s + 1, self.p.w[13]) - 1) *
            math.exp((1 - r) * self.p.w[14])
        )

        return max(new_s, 0.1)

    def _next_interval(self, stability: float) -> int:
        """
        Calculate next interval in days from stability

        I = 9 * S * (1/R - 1) where R is target retention
        """
        interval = 9 * stability * (1 / self.p.request_retention - 1)
        interval = int(round(interval))
        interval = max(1, min(interval, self.p.maximum_interval))
        return interval

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_retrievability(
        self,
        memory: MemoryState,
        at_time: Optional[datetime] = None
    ) -> float:
        """Get current retrievability at a given time"""
        at_time = at_time or datetime.now()
        if memory.last_review is None:
            return 0.0
        elapsed = (at_time - memory.last_review).days
        return self._retrievability(memory.stability, elapsed)

    def predict_forgetting_date(
        self,
        memory: MemoryState,
        target_retrievability: float = 0.5
    ) -> Optional[datetime]:
        """Predict when retrievability will drop to target level"""
        if memory.last_review is None:
            return None

        # Solve for t: R(t) = target
        # target = (1 + t/(9*S))^(-1)
        # t = 9 * S * (1/target - 1)
        days_to_forget = 9 * memory.stability * (1 / target_retrievability - 1)
        return memory.last_review + timedelta(days=int(days_to_forget))

    def get_intervals_for_all_ratings(
        self,
        memory: MemoryState,
        review_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """Get predicted intervals for each possible rating"""
        review_time = review_time or datetime.now()
        intervals = {}

        for rating in Rating:
            new_memory, _ = self.review(memory.copy(), rating, review_time)
            if new_memory.due:
                intervals[rating.name] = (new_memory.due - review_time).days
            else:
                intervals[rating.name] = 0

        return intervals


# =============================================================================
# HYBRID SCHEDULER: FSRS + R1-R7
# =============================================================================

class HybridScheduler:
    """
    Combines FSRS for card-level scheduling with R1-R7 macro schedule

    - R1-R7: Procedure-level retrieval sessions at expanding intervals
    - FSRS: Individual card scheduling within and between sessions

    This creates a two-tier learning system:
    1. Macro: When to do comprehensive procedure review (R1-R7)
    2. Micro: Which cards to review within each session (FSRS)
    """

    R1_R7_TEMPLATE = [
        (1, "R1", "Dictate operative note from memory", "free_recall", 15),
        (3, "R2", "Write Surgical Card from memory -> compare", "free_recall", 20),
        (7, "R3", "Mental rehearsal (full procedure)", "rehearsal", 20),
        (14, "R4", "Verbal teach-back (explain to peer/recorder)", "elaboration", 25),
        (30, "R5", "CSP rapid-fire quiz + case variation", "cued_recall", 15),
        (60, "R6", "Interleaved review (mix with similar procedures)", "interleaved", 30),
        (120, "R7", "Full simulation or cadaver lab if available", "application", 60),
    ]

    def __init__(self, params: Optional[FSRSParameters] = None):
        self.fsrs = FSRS(params)

    def create_hybrid_schedule(
        self,
        procedure_id: str,
        user_id: str,
        encoding_date: Optional[datetime] = None,
        target_retention_days: int = 180
    ) -> Dict[str, Any]:
        """
        Create schedule that combines R1-R7 with FSRS

        Args:
            procedure_id: UUID of the procedure
            user_id: User identifier
            encoding_date: When learning started (default: now)
            target_retention_days: Target retention period (default: 180 for Royal College)

        Returns:
            Schedule dict with macro sessions and metadata
        """
        encoding_date = encoding_date or datetime.now()

        macro_sessions = []
        for day, label, task, task_type, duration in self.R1_R7_TEMPLATE:
            # Apply adaptive multiplier if retention target differs
            adjusted_day = int(day * (target_retention_days / 180))

            macro_sessions.append({
                "session_number": int(label[1]),
                "label": label,
                "scheduled_date": (encoding_date + timedelta(days=adjusted_day)).isoformat(),
                "days_from_encoding": adjusted_day,
                "retrieval_task": task,
                "task_type": task_type,
                "estimated_duration_min": duration,
                "completed": False,
                "self_assessment_score": None,
            })

        return {
            "procedure_id": procedure_id,
            "user_id": user_id,
            "encoding_date": encoding_date.isoformat(),
            "target_retention_days": target_retention_days,
            "macro_schedule": macro_sessions,
        }

    def get_cards_for_session(
        self,
        session_number: int,
        all_cards: List[Dict[str, Any]],
        session_date: datetime
    ) -> List[str]:
        """
        Determine which FSRS cards should be reviewed in a given R session

        Logic:
        - R1-R2: All cards (comprehensive review)
        - R3: Cards with retrievability < 0.9
        - R4-R5: Focus on weak cards (retrievability < 0.8)
        - R6-R7: Only significantly forgotten cards (retrievability < 0.7)

        Args:
            session_number: Which R session (1-7)
            all_cards: List of card dicts with memory_state
            session_date: When the session is scheduled

        Returns:
            List of card IDs to review
        """
        if session_number <= 2:
            return [c["id"] for c in all_cards]

        threshold_map = {
            3: 0.9,
            4: 0.85,
            5: 0.8,
            6: 0.75,
            7: 0.7
        }
        threshold = threshold_map.get(session_number, 0.8)

        due_cards = []
        for card in all_cards:
            memory_data = card.get("memory_state", {})
            if memory_data:
                memory = MemoryState.from_dict(memory_data)
                r = self.fsrs.get_retrievability(memory, session_date)
                if r < threshold:
                    due_cards.append(card["id"])
            else:
                # New cards always included
                due_cards.append(card["id"])

        return due_cards

    def adjust_schedule_from_performance(
        self,
        schedule: Dict[str, Any],
        session_number: int,
        self_assessment: int  # 1-4
    ) -> Dict[str, Any]:
        """
        Adjust remaining schedule based on session performance

        - Score 1-2: Compress intervals (add extra sessions)
        - Score 3: Keep as is
        - Score 4: Can expand intervals slightly

        Args:
            schedule: Current schedule dict
            session_number: Which session was just completed
            self_assessment: User's self-assessment score (1-4)

        Returns:
            Updated schedule dict
        """
        multiplier = 1.0

        if self_assessment == 1:
            multiplier = 0.5  # Half the intervals
        elif self_assessment == 2:
            multiplier = 0.75
        elif self_assessment == 4:
            multiplier = 1.25  # Slightly expand

        # Adjust future sessions
        encoding = datetime.fromisoformat(schedule["encoding_date"])

        for session in schedule["macro_schedule"]:
            if session["session_number"] > session_number:
                original_days = session["days_from_encoding"]
                new_days = int(original_days * multiplier)

                session["days_from_encoding"] = new_days
                session["scheduled_date"] = (encoding + timedelta(days=new_days)).isoformat()

        return schedule

    def get_session_focus_areas(
        self,
        session_number: int,
        cards: List[Dict[str, Any]],
        session_date: datetime
    ) -> Dict[str, Any]:
        """
        Identify focus areas for a session based on card performance

        Returns:
            Dict with weak_phases, weak_csps, and recommendations
        """
        weak_cards = []
        phase_performance: Dict[str, List[float]] = {}
        csp_performance: Dict[int, List[float]] = {}

        for card in cards:
            memory_data = card.get("memory_state", {})
            if not memory_data:
                continue

            memory = MemoryState.from_dict(memory_data)
            r = self.fsrs.get_retrievability(memory, session_date)

            # Track by phase
            for tag in card.get("tags", []):
                if tag in ["architecture", "approach", "target", "closure"]:
                    if tag not in phase_performance:
                        phase_performance[tag] = []
                    phase_performance[tag].append(r)

                # Track CSPs
                if tag.startswith("csp_"):
                    try:
                        csp_num = int(tag.split("_")[1])
                        if csp_num not in csp_performance:
                            csp_performance[csp_num] = []
                        csp_performance[csp_num].append(r)
                    except (ValueError, IndexError):
                        pass

            if r < 0.8:
                weak_cards.append(card["id"])

        # Calculate averages
        weak_phases = []
        for phase, scores in phase_performance.items():
            avg = sum(scores) / len(scores) if scores else 1.0
            if avg < 0.8:
                weak_phases.append({"phase": phase, "avg_retention": avg})

        weak_csps = []
        for csp_num, scores in csp_performance.items():
            avg = sum(scores) / len(scores) if scores else 1.0
            if avg < 0.8:
                weak_csps.append({"csp_number": csp_num, "avg_retention": avg})

        return {
            "weak_phases": sorted(weak_phases, key=lambda x: x["avg_retention"]),
            "weak_csps": sorted(weak_csps, key=lambda x: x["avg_retention"]),
            "weak_card_count": len(weak_cards),
            "total_cards": len(cards),
            "recommendations": self._generate_recommendations(session_number, weak_phases, weak_csps),
        }

    def _generate_recommendations(
        self,
        session_number: int,
        weak_phases: List[Dict],
        weak_csps: List[Dict]
    ) -> List[str]:
        """Generate learning recommendations based on weak areas"""
        recommendations = []

        if weak_phases:
            phase_names = [p["phase"].upper() for p in weak_phases[:2]]
            recommendations.append(f"Focus extra time on {', '.join(phase_names)} phase(s)")

        if weak_csps:
            csp_nums = [f"CSP-{c['csp_number']}" for c in weak_csps[:3]]
            recommendations.append(f"Review safety points: {', '.join(csp_nums)}")

        # Session-specific recommendations
        session_tips = {
            1: "Dictate the full operative note without looking at references",
            2: "Draw the surgical card from memory, then compare with reference",
            3: "Close your eyes and mentally walk through each phase",
            4: "Explain the procedure to a colleague or record yourself",
            5: "Quiz yourself rapidly on CSP triggers",
            6: "Compare this procedure with similar approaches",
            7: "If possible, practice on simulation or cadaver",
        }

        if session_number in session_tips:
            recommendations.append(session_tips[session_number])

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_fsrs_card(
    card_id: str,
    difficulty_preset: float = 0.3
) -> Dict[str, Any]:
    """Create a new FSRS card with initial memory state"""
    return {
        "id": card_id,
        "memory_state": {
            "difficulty": difficulty_preset,
            "stability": 1.0,
            "state": State.NEW.value,
            "step": 0,
            "due": None,
            "last_review": None,
            "reps": 0,
            "lapses": 0,
        }
    }


def review_card(
    card: Dict[str, Any],
    rating: int,
    review_time: Optional[datetime] = None,
    params: Optional[FSRSParameters] = None
) -> Dict[str, Any]:
    """
    Convenience function to review a card

    Args:
        card: Card dict with memory_state
        rating: 1=AGAIN, 2=HARD, 3=GOOD, 4=EASY
        review_time: When review occurred
        params: FSRS parameters (uses defaults if None)

    Returns:
        Updated card dict
    """
    fsrs = FSRS(params)
    memory = MemoryState.from_dict(card["memory_state"])

    new_memory, log = fsrs.review(memory, Rating(rating), review_time)

    card["memory_state"] = new_memory.to_dict()

    return card


def calculate_retention_score(
    cards: List[Dict[str, Any]],
    at_time: Optional[datetime] = None
) -> float:
    """
    Calculate overall retention score for a set of cards

    Args:
        cards: List of card dicts with memory_state
        at_time: Time to calculate retention at (default: now)

    Returns:
        Average retrievability (0-1)
    """
    at_time = at_time or datetime.now()
    fsrs = FSRS()

    total_r = 0.0
    count = 0

    for card in cards:
        memory_data = card.get("memory_state", {})
        if memory_data and memory_data.get("last_review"):
            memory = MemoryState.from_dict(memory_data)
            total_r += fsrs.get_retrievability(memory, at_time)
            count += 1

    return total_r / count if count > 0 else 0.5
