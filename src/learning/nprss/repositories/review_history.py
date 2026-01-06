# src/learning/nprss/repositories/review_history.py
"""
Review History Repository

Audit trail for all review sessions.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class ReviewRecord:
    """Single review record."""
    id: UUID = None
    user_id: str = ""
    card_id: UUID = None
    session_id: UUID = None

    # Review outcome
    rating: int = 3  # 1=Again, 2=Hard, 3=Good, 4=Easy
    response_time_ms: int = None
    reviewed_at: datetime = None

    # FSRS snapshots
    difficulty_before: float = None
    difficulty_after: float = None
    stability_before: float = None
    stability_after: float = None
    retrievability_before: float = None
    retrievability_after: float = None

    # Scheduling
    interval_before_days: float = None
    interval_after_days: float = None

    # Context
    review_mode: str = "standard"
    device_type: str = None

    metadata: Dict[str, Any] = None


class ReviewHistoryRepository(BaseRepository[ReviewRecord]):
    """
    Repository for review history.

    Provides audit trail and analytics for all reviews.
    """

    @property
    def table_name(self) -> str:
        return "review_history"

    def _row_to_entity(self, row: Dict[str, Any]) -> ReviewRecord:
        """Convert database row to ReviewRecord."""
        return ReviewRecord(
            id=row.get('id'),
            user_id=row.get('user_id', ''),
            card_id=row.get('card_id'),
            session_id=row.get('session_id'),
            rating=row.get('rating', 3),
            response_time_ms=row.get('response_time_ms'),
            reviewed_at=row.get('reviewed_at'),
            difficulty_before=row.get('difficulty_before'),
            difficulty_after=row.get('difficulty_after'),
            stability_before=row.get('stability_before'),
            stability_after=row.get('stability_after'),
            retrievability_before=row.get('retrievability_before'),
            retrievability_after=row.get('retrievability_after'),
            interval_before_days=row.get('interval_before_days'),
            interval_after_days=row.get('interval_after_days'),
            review_mode=row.get('review_mode', 'standard'),
            device_type=row.get('device_type'),
            metadata=row.get('metadata', {})
        )

    def _entity_to_dict(self, entity: ReviewRecord) -> Dict[str, Any]:
        """Convert ReviewRecord to dict."""
        return {
            'id': entity.id,
            'user_id': entity.user_id,
            'card_id': entity.card_id,
            'session_id': entity.session_id,
            'rating': entity.rating,
            'response_time_ms': entity.response_time_ms,
            'reviewed_at': entity.reviewed_at or datetime.now(),
            'difficulty_before': entity.difficulty_before,
            'difficulty_after': entity.difficulty_after,
            'stability_before': entity.stability_before,
            'stability_after': entity.stability_after,
            'retrievability_before': entity.retrievability_before,
            'retrievability_after': entity.retrievability_after,
            'interval_before_days': entity.interval_before_days,
            'interval_after_days': entity.interval_after_days,
            'review_mode': entity.review_mode,
            'device_type': entity.device_type,
            'metadata': entity.metadata or {}
        }

    # =========================================================================
    # Record Creation
    # =========================================================================

    async def record_review(
        self,
        user_id: str,
        card_id: UUID,
        rating: int,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        response_time_ms: int = None,
        session_id: UUID = None,
        review_mode: str = "standard",
        device_type: str = None
    ) -> ReviewRecord:
        """
        Record a review with full state snapshots.

        Args:
            user_id: User identifier
            card_id: Card UUID
            rating: User rating (1-4)
            state_before: FSRS state before review
            state_after: FSRS state after review
            response_time_ms: Time to answer
            session_id: Optional session UUID
            review_mode: standard, socratic_guided, etc.
            device_type: web, mobile, tablet

        Returns:
            Created ReviewRecord
        """
        record = ReviewRecord(
            user_id=user_id,
            card_id=card_id,
            session_id=session_id,
            rating=rating,
            response_time_ms=response_time_ms,
            reviewed_at=datetime.now(),
            difficulty_before=state_before.get('difficulty'),
            difficulty_after=state_after.get('difficulty'),
            stability_before=state_before.get('stability'),
            stability_after=state_after.get('stability'),
            retrievability_before=state_before.get('retrievability'),
            retrievability_after=state_after.get('retrievability'),
            interval_before_days=state_before.get('scheduled_days'),
            interval_after_days=state_after.get('scheduled_days'),
            review_mode=review_mode,
            device_type=device_type
        )

        return await self.create(record)

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        card_id: UUID = None
    ) -> List[ReviewRecord]:
        """
        Get review history for a user.

        Args:
            user_id: User identifier
            limit: Maximum records
            offset: Pagination offset
            card_id: Optional filter by card

        Returns:
            List of review records (newest first)
        """
        card_filter = "AND card_id = $4" if card_id else ""

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1
              {card_filter}
            ORDER BY reviewed_at DESC
            LIMIT $2 OFFSET $3
        """

        params = [user_id, limit, offset]
        if card_id:
            params.append(card_id)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_session_history(self, session_id: UUID) -> List[ReviewRecord]:
        """Get all reviews in a session."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE session_id = $1
            ORDER BY reviewed_at
        """
        rows = await self.db.fetch(query, session_id)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_card_history(
        self,
        card_id: UUID,
        user_id: str = None
    ) -> List[ReviewRecord]:
        """
        Get review history for a card.

        Args:
            card_id: Card UUID
            user_id: Optional user filter

        Returns:
            Review records (oldest first)
        """
        user_filter = "AND user_id = $2" if user_id else ""

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE card_id = $1
              {user_filter}
            ORDER BY reviewed_at
        """

        params = [card_id]
        if user_id:
            params.append(user_id)

        rows = await self.db.fetch(query, *params)
        return [self._row_to_entity(dict(row)) for row in rows]

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_daily_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get daily review statistics.

        Args:
            user_id: User identifier
            days: Days to look back

        Returns:
            List of daily stat dicts
        """
        query = f"""
            SELECT
                DATE(reviewed_at) as review_date,
                COUNT(*) as total_reviews,
                COUNT(*) FILTER (WHERE rating >= 3) as correct,
                COUNT(*) FILTER (WHERE rating < 3) as incorrect,
                AVG(response_time_ms) as avg_response_time,
                AVG(interval_after_days) as avg_interval
            FROM {self.table_name}
            WHERE user_id = $1
              AND reviewed_at >= NOW() - INTERVAL '{days} days'
            GROUP BY DATE(reviewed_at)
            ORDER BY review_date DESC
        """

        rows = await self.db.fetch(query, user_id)
        return [dict(row) for row in rows]

    async def get_rating_distribution(
        self,
        user_id: str,
        days: int = 30,
        card_id: UUID = None
    ) -> Dict[int, int]:
        """
        Get distribution of ratings.

        Returns:
            {1: count, 2: count, 3: count, 4: count}
        """
        card_filter = "AND card_id = $3" if card_id else ""

        query = f"""
            SELECT rating, COUNT(*) as count
            FROM {self.table_name}
            WHERE user_id = $1
              AND reviewed_at >= NOW() - INTERVAL '{days} days'
              {card_filter}
            GROUP BY rating
        """

        params = [user_id, days]
        if card_id:
            params.append(card_id)

        rows = await self.db.fetch(query, *params)

        distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        for row in rows:
            distribution[row['rating']] = row['count']

        return distribution

    async def get_accuracy_trend(
        self,
        user_id: str,
        days: int = 30,
        window_size: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get rolling accuracy trend.

        Args:
            user_id: User identifier
            days: Days to look back
            window_size: Rolling window size

        Returns:
            List of {date, accuracy} dicts
        """
        query = f"""
            WITH daily_reviews AS (
                SELECT
                    DATE(reviewed_at) as review_date,
                    COUNT(*) as total,
                    SUM(CASE WHEN rating >= 3 THEN 1 ELSE 0 END) as correct
                FROM {self.table_name}
                WHERE user_id = $1
                  AND reviewed_at >= NOW() - INTERVAL '{days} days'
                GROUP BY DATE(reviewed_at)
            )
            SELECT
                review_date,
                SUM(correct) OVER w::FLOAT / NULLIF(SUM(total) OVER w, 0) as rolling_accuracy
            FROM daily_reviews
            WINDOW w AS (ORDER BY review_date ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW)
            ORDER BY review_date
        """

        rows = await self.db.fetch(query, user_id)
        return [dict(row) for row in rows]

    async def get_response_time_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, float]:
        """
        Get response time statistics.

        Returns:
            {avg, p50, p90, p99} in milliseconds
        """
        query = f"""
            SELECT
                AVG(response_time_ms) as avg,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as p50,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY response_time_ms) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99
            FROM {self.table_name}
            WHERE user_id = $1
              AND reviewed_at >= NOW() - INTERVAL '{days} days'
              AND response_time_ms IS NOT NULL
        """

        row = await self.db.fetchrow(query, user_id)
        return dict(row) if row else {'avg': 0, 'p50': 0, 'p90': 0, 'p99': 0}

    async def get_stability_progression(
        self,
        user_id: str,
        card_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get stability progression over time for a card.

        Returns:
            List of {date, stability_before, stability_after} dicts
        """
        query = f"""
            SELECT
                reviewed_at,
                rating,
                stability_before,
                stability_after,
                difficulty_before,
                difficulty_after
            FROM {self.table_name}
            WHERE user_id = $1 AND card_id = $2
            ORDER BY reviewed_at
        """

        rows = await self.db.fetch(query, user_id, card_id)
        return [dict(row) for row in rows]
