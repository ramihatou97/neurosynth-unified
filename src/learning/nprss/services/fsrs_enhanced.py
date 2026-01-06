"""
Enhanced FSRS Service

Extends existing NeuroSynth FSRS with:
- Dual-track scheduling (factual/procedural/CSP)
- Exam-aware acceleration
- Load balancing
- Retention target optimization

INTEGRATION: Import and extend your existing FSRS class
"""

import math
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict

# Try to import existing FSRS from NeuroSynth
# Adjust import path based on your actual structure
try:
    from src.learning.fsrs import FSRS as BaseFSRS
    HAS_BASE_FSRS = True
except ImportError:
    HAS_BASE_FSRS = False
    BaseFSRS = object


class Rating(IntEnum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


class CardTrack(str):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CSP = "csp"


class CardState(str):
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    RELEARNING = "relearning"


@dataclass
class FSRSConfig:
    """Configuration for enhanced FSRS."""
    # Retention targets by track
    retention_factual: float = 0.90
    retention_procedural: float = 0.92
    retention_csp: float = 0.95
    
    # Maximum intervals (days)
    max_interval_factual: int = 365
    max_interval_procedural: int = 180
    max_interval_csp: int = 90
    
    # Daily limits
    daily_review_capacity: int = 100
    new_cards_per_day: int = 20
    
    # Exam-aware settings
    exam_date: Optional[date] = None
    min_reviews_before_exam: int = 3
    
    # Load balancing
    enable_load_balancing: bool = True
    max_daily_variance: float = 0.2  # Allow ±20% from average


@dataclass
class MemoryState:
    """Card memory state compatible with existing system."""
    state: str = CardState.NEW
    stability: float = 0.0
    difficulty: float = 0.0
    due_date: datetime = field(default_factory=datetime.now)
    last_review: Optional[datetime] = None
    reps: int = 0
    lapses: int = 0
    card_track: str = CardTrack.FACTUAL


@dataclass 
class ReviewResult:
    """Result of FSRS calculation."""
    new_state: str
    new_stability: float
    new_difficulty: float
    interval_days: float
    next_due: datetime
    retrievability: float = 0.0


class FSRSEnhanced:
    """
    Enhanced FSRS with dual-track scheduling.
    
    If base FSRS exists, extends it. Otherwise, standalone implementation.
    """
    
    # FSRS v4 parameters optimized for medical education
    WEIGHTS = [
        0.4072,   # w0: initial stability for Again
        1.1829,   # w1: initial stability for Hard  
        3.1262,   # w2: initial stability for Good
        15.4722,  # w3: initial stability for Easy
        7.2102,   # w4: difficulty weight
        0.5316,   # w5: stability decay
        1.0651,   # w6: stability factor
        0.0046,   # w7: hard penalty
        1.5352,   # w8: easy bonus
        0.1192,   # w9: difficulty factor
        1.0507,   # w10: stability after lapse
        0.0028,   # w11: lapse stability decay
        2.4028,   # w12: short-term factor
        0.1131,   # w13: short-term decay
        0.2909,   # w14: difficulty regression
        2.2271,   # w15: difficulty mean
        0.0272,   # w16: review count factor
    ]
    
    def __init__(self, config: Optional[FSRSConfig] = None):
        self.config = config or FSRSConfig()
        self.w = self.WEIGHTS
        
        # Track-specific retention targets
        self._retention = {
            CardTrack.FACTUAL: self.config.retention_factual,
            CardTrack.PROCEDURAL: self.config.retention_procedural,
            CardTrack.CSP: self.config.retention_csp,
        }
        
        # Track-specific max intervals
        self._max_interval = {
            CardTrack.FACTUAL: self.config.max_interval_factual,
            CardTrack.PROCEDURAL: self.config.max_interval_procedural,
            CardTrack.CSP: self.config.max_interval_csp,
        }
        
        # Initialize base FSRS if available
        if HAS_BASE_FSRS:
            self._base = BaseFSRS()
        else:
            self._base = None
    
    # =========================================================================
    # CORE REVIEW LOGIC
    # =========================================================================
    
    def review(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: Optional[datetime] = None
    ) -> ReviewResult:
        """Process a review and return new scheduling state."""
        review_time = review_time or datetime.now()
        
        if state.state == CardState.NEW:
            return self._review_new(state, rating, review_time)
        elif state.state == CardState.LEARNING:
            return self._review_learning(state, rating, review_time)
        elif state.state == CardState.RELEARNING:
            return self._review_relearning(state, rating, review_time)
        else:
            return self._review_review(state, rating, review_time)
    
    def _review_new(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> ReviewResult:
        """First review of a new card."""
        stability = self._init_stability(rating)
        difficulty = self._init_difficulty(rating)
        
        if rating == Rating.AGAIN:
            return ReviewResult(
                new_state=CardState.LEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=1/1440,  # 1 minute
                next_due=review_time + timedelta(minutes=1)
            )
        elif rating == Rating.HARD:
            return ReviewResult(
                new_state=CardState.LEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=6/1440,  # 6 minutes
                next_due=review_time + timedelta(minutes=6)
            )
        else:  # Good or Easy
            interval = self._calculate_interval(stability, state.card_track)
            return ReviewResult(
                new_state=CardState.REVIEW,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=interval,
                next_due=review_time + timedelta(days=interval)
            )
    
    def _review_learning(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> ReviewResult:
        """Review in learning state."""
        difficulty = self._next_difficulty(state.difficulty, rating)
        
        if rating == Rating.AGAIN:
            stability = self._init_stability(Rating.AGAIN)
            return ReviewResult(
                new_state=CardState.LEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=1/1440,
                next_due=review_time + timedelta(minutes=1)
            )
        elif rating == Rating.HARD:
            stability = state.stability * 1.2
            return ReviewResult(
                new_state=CardState.LEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=10/1440,
                next_due=review_time + timedelta(minutes=10)
            )
        else:
            stability = self._next_stability(state.stability, difficulty, rating, 0)
            interval = self._calculate_interval(stability, state.card_track)
            return ReviewResult(
                new_state=CardState.REVIEW,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=interval,
                next_due=review_time + timedelta(days=interval)
            )
    
    def _review_relearning(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> ReviewResult:
        """Review in relearning state (after lapse)."""
        difficulty = self._next_difficulty(state.difficulty, rating)
        
        if rating == Rating.AGAIN:
            stability = max(0.1, state.stability * 0.5)
            return ReviewResult(
                new_state=CardState.RELEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=1/1440,
                next_due=review_time + timedelta(minutes=1)
            )
        else:
            stability = self._next_stability(state.stability, difficulty, rating, state.lapses)
            interval = self._calculate_interval(stability, state.card_track)
            return ReviewResult(
                new_state=CardState.REVIEW,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=interval,
                next_due=review_time + timedelta(days=interval)
            )
    
    def _review_review(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: datetime
    ) -> ReviewResult:
        """Review in review state."""
        difficulty = self._next_difficulty(state.difficulty, rating)
        
        # Calculate elapsed time and retrievability
        if state.last_review:
            elapsed = (review_time - state.last_review).total_seconds() / 86400
        else:
            elapsed = state.stability
        
        retrievability = self._retrievability(elapsed, state.stability)
        
        if rating == Rating.AGAIN:
            # Lapse
            stability = self._stability_after_lapse(state.stability, difficulty, retrievability)
            return ReviewResult(
                new_state=CardState.RELEARNING,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=10/1440,
                next_due=review_time + timedelta(minutes=10),
                retrievability=retrievability
            )
        else:
            stability = self._next_stability(
                state.stability, difficulty, rating, state.lapses, retrievability
            )
            interval = self._calculate_interval(stability, state.card_track)
            
            # Apply exam-aware acceleration
            interval = self._apply_exam_acceleration(interval, state.card_track)
            
            return ReviewResult(
                new_state=CardState.REVIEW,
                new_stability=stability,
                new_difficulty=difficulty,
                interval_days=interval,
                next_due=review_time + timedelta(days=interval),
                retrievability=retrievability
            )
    
    # =========================================================================
    # FSRS CALCULATIONS
    # =========================================================================
    
    def _init_stability(self, rating: Rating) -> float:
        """Initial stability based on first rating."""
        return self.w[rating.value - 1]
    
    def _init_difficulty(self, rating: Rating) -> float:
        """Initial difficulty based on first rating."""
        return self.w[4] - math.exp(self.w[5] * (rating.value - 1)) + 1
    
    def _next_difficulty(self, d: float, rating: Rating) -> float:
        """Calculate next difficulty."""
        delta = (rating.value - 3) * self.w[14]
        mean_reversion = self.w[14] * (self.w[15] - d)
        return max(1.0, min(10.0, d + delta + mean_reversion))
    
    def _next_stability(
        self,
        s: float,
        d: float,
        rating: Rating,
        lapses: int,
        r: float = 0.9
    ) -> float:
        """Calculate next stability."""
        # Hard penalty / easy bonus
        if rating == Rating.HARD:
            modifier = self.w[7]
        elif rating == Rating.EASY:
            modifier = self.w[8]
        else:
            modifier = 1.0
        
        # Stability increase factor
        s_increase = math.exp(self.w[6]) * (11 - d) * (s ** -self.w[5])
        s_increase *= math.exp(self.w[9] * (1 - r)) * modifier
        
        # Lapse penalty
        if lapses > 0:
            s_increase *= math.exp(-self.w[16] * lapses)
        
        return s * (1 + s_increase)
    
    def _stability_after_lapse(self, s: float, d: float, r: float) -> float:
        """Calculate stability after a lapse."""
        return max(
            0.1,
            s * math.exp(self.w[10]) * (d ** -self.w[11]) * 
            ((s + 1) ** self.w[12] - 1) * math.exp(self.w[13] * (1 - r))
        )
    
    def _retrievability(self, elapsed: float, stability: float) -> float:
        """Calculate probability of recall."""
        if stability <= 0:
            return 0.0
        return math.exp(-elapsed / stability * math.log(0.9))
    
    def _calculate_interval(self, stability: float, track: str) -> float:
        """Calculate interval based on track-specific retention target."""
        retention = self._retention.get(track, 0.9)
        max_ivl = self._max_interval.get(track, 365)
        
        # I = S * ln(R) / ln(0.9)
        interval = stability * math.log(retention) / math.log(0.9)
        
        return max(1.0, min(interval, max_ivl))
    
    # =========================================================================
    # EXAM-AWARE SCHEDULING
    # =========================================================================
    
    def _apply_exam_acceleration(self, interval: float, track: str) -> float:
        """Shorten intervals if exam is approaching."""
        if not self.config.exam_date:
            return interval
        
        days_until_exam = (self.config.exam_date - date.today()).days
        if days_until_exam <= 0:
            return interval
        
        # Calculate acceleration factor
        # More aggressive for CSP, less for factual
        if track == CardTrack.CSP:
            # CSP: ensure at least 3 reviews before exam
            max_allowed = days_until_exam / self.config.min_reviews_before_exam
            return min(interval, max_allowed)
        elif track == CardTrack.PROCEDURAL:
            # Procedural: moderate acceleration
            if days_until_exam < 30:
                return min(interval, days_until_exam / 2)
            return interval
        else:
            # Factual: light acceleration only in final weeks
            if days_until_exam < 14:
                return min(interval, days_until_exam / 1.5)
            return interval
    
    def set_exam_date(self, exam_date: date):
        """Set exam date for acceleration."""
        self.config.exam_date = exam_date
    
    # =========================================================================
    # LOAD BALANCING
    # =========================================================================
    
    def balance_load(
        self,
        cards: List[Tuple[str, MemoryState]],
        forecast_days: int = 30
    ) -> Dict[str, datetime]:
        """
        Redistribute due dates to balance daily review load.
        
        Returns dict of card_id -> new_due_date
        """
        if not self.config.enable_load_balancing:
            return {}
        
        # Count cards per day
        daily_counts = defaultdict(int)
        card_days = {}
        
        for card_id, state in cards:
            due_day = state.due_date.date()
            daily_counts[due_day] += 1
            card_days[card_id] = due_day
        
        # Calculate average
        total = sum(daily_counts.values())
        avg = total / max(1, len(daily_counts))
        max_per_day = avg * (1 + self.config.max_daily_variance)
        
        # Find overloaded days and redistribute
        adjustments = {}
        
        for day, count in sorted(daily_counts.items()):
            if count > max_per_day:
                excess = int(count - max_per_day)
                
                # Find cards to move (lowest stability first)
                day_cards = [
                    (cid, s) for cid, s in cards
                    if card_days.get(cid) == day
                ]
                day_cards.sort(key=lambda x: x[1].stability)
                
                for i, (card_id, state) in enumerate(day_cards[:excess]):
                    # Move to next day with capacity
                    new_day = day + timedelta(days=1)
                    while daily_counts.get(new_day, 0) >= max_per_day:
                        new_day += timedelta(days=1)
                    
                    adjustments[card_id] = datetime.combine(
                        new_day, state.due_date.time()
                    )
                    daily_counts[new_day] += 1
                    daily_counts[day] -= 1
        
        return adjustments
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def preview_intervals(self, state: MemoryState) -> Dict[str, float]:
        """Preview intervals for each rating."""
        return {
            "again": self.review(state, Rating.AGAIN).interval_days,
            "hard": self.review(state, Rating.HARD).interval_days,
            "good": self.review(state, Rating.GOOD).interval_days,
            "easy": self.review(state, Rating.EASY).interval_days,
        }
    
    def forecast_reviews(
        self,
        states: List[MemoryState],
        days: int = 30
    ) -> List[Dict]:
        """Forecast review counts for upcoming days."""
        forecast = []
        today = date.today()
        
        for day_offset in range(days):
            target = today + timedelta(days=day_offset)
            
            due = sum(
                1 for s in states
                if s.due_date.date() <= target
            )
            
            forecast.append({
                "date": target.isoformat(),
                "due_count": due,
                "day_offset": day_offset
            })
        
        return forecast


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_fsrs(
    exam_date: Optional[date] = None,
    retention_factual: float = 0.90,
    retention_csp: float = 0.95,
    enable_load_balancing: bool = True
) -> FSRSEnhanced:
    """Create configured FSRS instance."""
    config = FSRSConfig(
        exam_date=exam_date,
        retention_factual=retention_factual,
        retention_csp=retention_csp,
        enable_load_balancing=enable_load_balancing
    )
    return FSRSEnhanced(config)
