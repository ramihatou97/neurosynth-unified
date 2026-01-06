# src/learning/nprss/repositories/user_state.py
"""
User Learning State Repository

Database access for FSRS learning state per user per card.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class UserLearningState:
    """User's learning state for a card."""
    id: UUID = None
    user_id: str = ""
    card_id: UUID = None

    # FSRS parameters
    difficulty: float = 0.3
    stability: float = 0.0
    retrievability: float = 1.0

    # Scheduling
    state: str = "new"  # new, learning, review, relearning
    due_date: datetime = None
    last_review: datetime = None
    scheduled_days: float = 0.0

    # Counts
    reps: int = 0
    lapses: int = 0

    # R-level tracking
    current_r_level: int = 0
    r1_completed: bool = False
    r2_completed: bool = False
    r3_completed: bool = False
    r4_completed: bool = False
    r5_completed: bool = False
    r6_completed: bool = False
    r7_completed: bool = False

    created_at: datetime = None
    updated_at: datetime = None


class UserLearningStateRepository(BaseRepository[UserLearningState]):
    """
    Repository for user learning state (FSRS parameters).

    Maps to: nprss_card_memory_state table
    """

    @property
    def table_name(self) -> str:
        return "nprss_card_memory_state"

    def _row_to_entity(self, row: Dict[str, Any]) -> UserLearningState:
        """Convert database row to UserLearningState."""
        return UserLearningState(
            id=row.get('id'),
            user_id=row.get('user_id', ''),
            card_id=row.get('card_id'),
            difficulty=row.get('difficulty', 0.3),
            stability=row.get('stability', 0.0),
            retrievability=row.get('retrievability', 1.0),
            state=row.get('state', 'new'),
            due_date=row.get('due_date'),
            last_review=row.get('last_review'),
            scheduled_days=row.get('scheduled_days', 0.0),
            reps=row.get('reps', 0),
            lapses=row.get('lapses', 0),
            current_r_level=row.get('current_r_level', 0),
            r1_completed=row.get('r1_completed', False),
            r2_completed=row.get('r2_completed', False),
            r3_completed=row.get('r3_completed', False),
            r4_completed=row.get('r4_completed', False),
            r5_completed=row.get('r5_completed', False),
            r6_completed=row.get('r6_completed', False),
            r7_completed=row.get('r7_completed', False),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )

    def _entity_to_dict(self, entity: UserLearningState) -> Dict[str, Any]:
        """Convert UserLearningState to dict."""
        return {
            'id': entity.id,
            'user_id': entity.user_id,
            'card_id': entity.card_id,
            'difficulty': entity.difficulty,
            'stability': entity.stability,
            'retrievability': entity.retrievability,
            'state': entity.state,
            'due_date': entity.due_date,
            'last_review': entity.last_review,
            'scheduled_days': entity.scheduled_days,
            'reps': entity.reps,
            'lapses': entity.lapses,
            'current_r_level': entity.current_r_level,
            'r1_completed': entity.r1_completed,
            'r2_completed': entity.r2_completed,
            'r3_completed': entity.r3_completed,
            'r4_completed': entity.r4_completed,
            'r5_completed': entity.r5_completed,
            'r6_completed': entity.r6_completed,
            'r7_completed': entity.r7_completed,
        }

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def get_user_state(
        self,
        user_id: str,
        card_id: UUID
    ) -> Optional[UserLearningState]:
        """
        Get learning state for user-card pair.

        Args:
            user_id: User identifier
            card_id: Card UUID

        Returns:
            UserLearningState or None
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1 AND card_id = $2
        """
        row = await self.db.fetchrow(query, user_id, card_id)
        return self._row_to_entity(dict(row)) if row else None

    async def get_or_create(
        self,
        user_id: str,
        card_id: UUID,
        initial_difficulty: float = 0.3
    ) -> UserLearningState:
        """
        Get existing state or create new one.

        Args:
            user_id: User identifier
            card_id: Card UUID
            initial_difficulty: Initial difficulty for new state

        Returns:
            UserLearningState (existing or newly created)
        """
        existing = await self.get_user_state(user_id, card_id)
        if existing:
            return existing

        # Create new state
        state = UserLearningState(
            user_id=user_id,
            card_id=card_id,
            difficulty=initial_difficulty,
            state='new',
            due_date=datetime.now()
        )
        return await self.create(state)

    async def update_after_review(
        self,
        user_id: str,
        card_id: UUID,
        new_difficulty: float,
        new_stability: float,
        new_state: str,
        next_review: datetime,
        scheduled_days: float,
        r_level: int = None
    ) -> UserLearningState:
        """
        Update state after a review.

        Args:
            user_id: User identifier
            card_id: Card UUID
            new_difficulty: Updated difficulty
            new_stability: Updated stability
            new_state: New state (learning, review, relearning)
            next_review: Next review datetime
            scheduled_days: Days until next review
            r_level: R-level achieved (1-7)

        Returns:
            Updated state
        """
        # Build R-level updates
        r_updates = {}
        if r_level:
            r_updates[f'r{r_level}_completed'] = True
            r_updates['current_r_level'] = r_level

        query = f"""
            UPDATE {self.table_name}
            SET difficulty = $3,
                stability = $4,
                state = $5,
                due_date = $6,
                scheduled_days = $7,
                last_review = NOW(),
                reps = reps + 1,
                current_r_level = COALESCE($8, current_r_level),
                r1_completed = COALESCE($9, r1_completed),
                r2_completed = COALESCE($10, r2_completed),
                r3_completed = COALESCE($11, r3_completed),
                r4_completed = COALESCE($12, r4_completed),
                r5_completed = COALESCE($13, r5_completed),
                r6_completed = COALESCE($14, r6_completed),
                r7_completed = COALESCE($15, r7_completed),
                updated_at = NOW()
            WHERE user_id = $1 AND card_id = $2
            RETURNING *
        """

        row = await self.db.fetchrow(
            query,
            user_id, card_id,
            new_difficulty, new_stability, new_state,
            next_review, scheduled_days,
            r_updates.get('current_r_level'),
            r_updates.get('r1_completed'),
            r_updates.get('r2_completed'),
            r_updates.get('r3_completed'),
            r_updates.get('r4_completed'),
            r_updates.get('r5_completed'),
            r_updates.get('r6_completed'),
            r_updates.get('r7_completed')
        )

        return self._row_to_entity(dict(row)) if row else None

    async def increment_lapses(self, user_id: str, card_id: UUID) -> None:
        """Increment lapse count for a card."""
        query = f"""
            UPDATE {self.table_name}
            SET lapses = lapses + 1, updated_at = NOW()
            WHERE user_id = $1 AND card_id = $2
        """
        await self.db.execute(query, user_id, card_id)

    # =========================================================================
    # Due Card Queries
    # =========================================================================

    async def get_due_cards(
        self,
        user_id: str,
        limit: int = 20,
        include_new: bool = True,
        procedure_id: UUID = None
    ) -> List[Dict[str, Any]]:
        """
        Get cards due for review with card details.

        Args:
            user_id: User identifier
            limit: Maximum cards to return
            include_new: Include new (unreviewed) cards
            procedure_id: Optional procedure filter

        Returns:
            List of dicts with state + card info
        """
        # Build dynamic query parts
        params = [user_id, limit]
        param_idx = 3

        state_filter = ""
        if not include_new:
            state_filter = f"AND uls.state != 'new'"

        proc_filter = ""
        if procedure_id:
            proc_filter = f"AND lc.procedure_id = ${param_idx}"
            params.append(procedure_id)
            param_idx += 1

        query = f"""
            SELECT
                uls.*,
                lc.card_type,
                lc.prompt,
                lc.answer,
                lc.procedure_id,
                p.name as procedure_name,
                EXTRACT(EPOCH FROM (NOW() - uls.due_date))/86400 AS days_overdue
            FROM {self.table_name} uls
            JOIN nprss_learning_cards lc ON lc.id = uls.card_id
            LEFT JOIN procedures p ON p.id = lc.procedure_id
            WHERE uls.user_id = $1
              AND (uls.due_date <= NOW() OR uls.due_date IS NULL)
              {state_filter}
              {proc_filter}
            ORDER BY
                CASE WHEN uls.state = 'relearning' THEN 0
                     WHEN uls.state = 'learning' THEN 1
                     WHEN uls.state = 'new' THEN 2
                     ELSE 3 END,
                uls.due_date ASC NULLS LAST
            LIMIT $2
        """

        rows = await self.db.fetch(query, *params)
        return [dict(row) for row in rows]

    async def get_due_count(
        self,
        user_id: str,
        procedure_id: UUID = None
    ) -> Dict[str, int]:
        """
        Get count of due cards by state.

        Returns:
            {'new': N, 'learning': N, 'review': N, 'relearning': N, 'total': N}
        """
        proc_filter = "AND lc.procedure_id = $2" if procedure_id else ""

        query = f"""
            SELECT
                state,
                COUNT(*) as count
            FROM {self.table_name} uls
            JOIN nprss_learning_cards lc ON lc.id = uls.card_id
            WHERE uls.user_id = $1
              AND uls.due_date <= NOW()
              {proc_filter}
            GROUP BY state
        """

        params = [user_id]
        if procedure_id:
            params.append(procedure_id)

        rows = await self.db.fetch(query, *params)

        counts = {'new': 0, 'learning': 0, 'review': 0, 'relearning': 0}
        for row in rows:
            counts[row['state']] = row['count']

        counts['total'] = sum(counts.values())
        return counts

    async def get_forecast(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get review forecast for next N days.

        Args:
            user_id: User identifier
            days: Days to forecast

        Returns:
            List of {date, count} dicts
        """
        query = f"""
            SELECT
                DATE(due_date) as review_date,
                COUNT(*) as count
            FROM {self.table_name}
            WHERE user_id = $1
              AND due_date BETWEEN NOW() AND NOW() + INTERVAL '{days} days'
              AND state != 'new'
            GROUP BY DATE(due_date)
            ORDER BY review_date
        """

        rows = await self.db.fetch(query, user_id)
        return [dict(row) for row in rows]

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_user_statistics(
        self,
        user_id: str,
        procedure_id: UUID = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics for user.

        Returns:
            Statistics dict with counts, averages, R-level progress
        """
        proc_filter = "AND lc.procedure_id = $2" if procedure_id else ""

        query = f"""
            SELECT
                COUNT(*) as total_cards,
                COUNT(*) FILTER (WHERE state = 'new') as new_cards,
                COUNT(*) FILTER (WHERE state = 'learning') as learning_cards,
                COUNT(*) FILTER (WHERE state = 'review') as review_cards,
                COUNT(*) FILTER (WHERE state = 'relearning') as relearning_cards,
                AVG(stability) as avg_stability,
                AVG(difficulty) as avg_difficulty,
                SUM(reps) as total_reviews,
                SUM(lapses) as total_lapses,
                COUNT(*) FILTER (WHERE r7_completed = true) as r7_mastered,
                COUNT(*) FILTER (WHERE r6_completed = true) as r6_completed,
                COUNT(*) FILTER (WHERE r5_completed = true) as r5_completed,
                COUNT(*) FILTER (WHERE r4_completed = true) as r4_completed,
                COUNT(*) FILTER (WHERE r3_completed = true) as r3_completed,
                COUNT(*) FILTER (WHERE r2_completed = true) as r2_completed,
                COUNT(*) FILTER (WHERE r1_completed = true) as r1_completed
            FROM {self.table_name} uls
            JOIN nprss_learning_cards lc ON lc.id = uls.card_id
            WHERE uls.user_id = $1
              {proc_filter}
        """

        params = [user_id]
        if procedure_id:
            params.append(procedure_id)

        row = await self.db.fetchrow(query, *params)
        return dict(row) if row else {}

    async def get_retention_rate(
        self,
        user_id: str,
        days: int = 30
    ) -> float:
        """
        Calculate retention rate over recent period.

        Args:
            user_id: User identifier
            days: Days to look back

        Returns:
            Retention rate (0-1)
        """
        query = """
            SELECT
                CASE WHEN COUNT(*) > 0
                     THEN SUM(CASE WHEN rating >= 3 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
                     ELSE 0.0
                END as retention
            FROM review_history
            WHERE user_id = $1
              AND reviewed_at >= NOW() - INTERVAL '$2 days'
        """

        result = await self.db.fetchval(query, user_id, days)
        return result or 0.0
