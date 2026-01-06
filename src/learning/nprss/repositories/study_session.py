# src/learning/nprss/repositories/study_session.py
"""
Study Session Repository

Manages study session tracking and analytics.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base import BaseRepository


@dataclass
class StudySession:
    """Study session record."""
    id: UUID = None
    user_id: str = ""

    # Timing
    started_at: datetime = None
    ended_at: datetime = None
    duration_seconds: int = None

    # Stats
    cards_reviewed: int = 0
    cards_correct: int = 0
    cards_incorrect: int = 0
    cards_skipped: int = 0

    # Averages
    avg_response_time_ms: float = None
    avg_difficulty: float = None

    # Type and focus
    session_type: str = "daily_review"
    focus_procedure_id: UUID = None
    focus_specialty: str = None

    # Progress
    r_levels_achieved: List[str] = field(default_factory=list)

    device_type: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StudySessionRepository(BaseRepository[StudySession]):
    """
    Repository for study sessions.

    Tracks session analytics and engagement.
    """

    @property
    def table_name(self) -> str:
        return "study_sessions"

    def _row_to_entity(self, row: Dict[str, Any]) -> StudySession:
        """Convert database row to StudySession."""
        return StudySession(
            id=row.get('id'),
            user_id=row.get('user_id', ''),
            started_at=row.get('started_at'),
            ended_at=row.get('ended_at'),
            duration_seconds=row.get('duration_seconds'),
            cards_reviewed=row.get('cards_reviewed', 0),
            cards_correct=row.get('cards_correct', 0),
            cards_incorrect=row.get('cards_incorrect', 0),
            cards_skipped=row.get('cards_skipped', 0),
            avg_response_time_ms=row.get('avg_response_time_ms'),
            avg_difficulty=row.get('avg_difficulty'),
            session_type=row.get('session_type', 'daily_review'),
            focus_procedure_id=row.get('focus_procedure_id'),
            focus_specialty=row.get('focus_specialty'),
            r_levels_achieved=row.get('r_levels_achieved', []),
            device_type=row.get('device_type'),
            metadata=row.get('metadata', {})
        )

    def _entity_to_dict(self, entity: StudySession) -> Dict[str, Any]:
        """Convert StudySession to dict."""
        return {
            'id': entity.id,
            'user_id': entity.user_id,
            'started_at': entity.started_at or datetime.now(),
            'ended_at': entity.ended_at,
            'duration_seconds': entity.duration_seconds,
            'cards_reviewed': entity.cards_reviewed,
            'cards_correct': entity.cards_correct,
            'cards_incorrect': entity.cards_incorrect,
            'cards_skipped': entity.cards_skipped,
            'avg_response_time_ms': entity.avg_response_time_ms,
            'avg_difficulty': entity.avg_difficulty,
            'session_type': entity.session_type,
            'focus_procedure_id': entity.focus_procedure_id,
            'focus_specialty': entity.focus_specialty,
            'r_levels_achieved': entity.r_levels_achieved,
            'device_type': entity.device_type,
            'metadata': entity.metadata
        }

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    async def start_session(
        self,
        user_id: str,
        session_type: str = "daily_review",
        focus_procedure_id: UUID = None,
        focus_specialty: str = None,
        device_type: str = None
    ) -> StudySession:
        """
        Start a new study session.

        Args:
            user_id: User identifier
            session_type: Type of session (daily_review, procedure_focus, etc.)
            focus_procedure_id: Optional procedure focus
            focus_specialty: Optional specialty focus
            device_type: Client device type

        Returns:
            Created session
        """
        session = StudySession(
            user_id=user_id,
            started_at=datetime.now(),
            session_type=session_type,
            focus_procedure_id=focus_procedure_id,
            focus_specialty=focus_specialty,
            device_type=device_type
        )

        return await self.create(session)

    async def end_session(
        self,
        session_id: UUID,
        cards_reviewed: int = None,
        cards_correct: int = None,
        cards_incorrect: int = None,
        cards_skipped: int = None,
        avg_response_time_ms: float = None,
        avg_difficulty: float = None,
        r_levels_achieved: List[str] = None
    ) -> StudySession:
        """
        End a study session with final stats.

        Args:
            session_id: Session UUID
            cards_reviewed: Total cards reviewed
            cards_correct: Cards answered correctly
            cards_incorrect: Cards answered incorrectly
            cards_skipped: Cards skipped
            avg_response_time_ms: Average response time
            avg_difficulty: Average card difficulty
            r_levels_achieved: R-levels achieved this session

        Returns:
            Updated session
        """
        now = datetime.now()

        # Get session to calculate duration
        session = await self.get_by_id(session_id)
        duration = None
        if session and session.started_at:
            duration = int((now - session.started_at).total_seconds())

        updates = {
            'ended_at': now,
            'duration_seconds': duration,
            'cards_reviewed': cards_reviewed,
            'cards_correct': cards_correct,
            'cards_incorrect': cards_incorrect,
            'cards_skipped': cards_skipped,
            'avg_response_time_ms': avg_response_time_ms,
            'avg_difficulty': avg_difficulty,
            'r_levels_achieved': r_levels_achieved or []
        }

        return await self.update(session_id, updates)

    async def increment_stats(
        self,
        session_id: UUID,
        correct: bool = False,
        incorrect: bool = False,
        skipped: bool = False
    ) -> None:
        """
        Increment session stats during review.

        Args:
            session_id: Session UUID
            correct: Increment correct count
            incorrect: Increment incorrect count
            skipped: Increment skipped count
        """
        updates = []
        if correct:
            updates.append("cards_correct = cards_correct + 1")
        if incorrect:
            updates.append("cards_incorrect = cards_incorrect + 1")
        if skipped:
            updates.append("cards_skipped = cards_skipped + 1")

        updates.append("cards_reviewed = cards_reviewed + 1")

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(updates)}
            WHERE id = $1
        """

        await self.db.execute(query, session_id)

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_active_session(self, user_id: str) -> Optional[StudySession]:
        """
        Get user's active (not ended) session.

        Args:
            user_id: User identifier

        Returns:
            Active session or None
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1 AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
        """

        row = await self.db.fetchrow(query, user_id)
        return self._row_to_entity(dict(row)) if row else None

    async def get_recent_sessions(
        self,
        user_id: str,
        days: int = 30,
        limit: int = 50
    ) -> List[StudySession]:
        """
        Get user's recent sessions.

        Args:
            user_id: User identifier
            days: Days to look back
            limit: Maximum sessions

        Returns:
            List of sessions (newest first)
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1
              AND started_at >= NOW() - INTERVAL '{days} days'
            ORDER BY started_at DESC
            LIMIT $2
        """

        rows = await self.db.fetch(query, user_id, limit)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_sessions_by_type(
        self,
        user_id: str,
        session_type: str,
        limit: int = 20
    ) -> List[StudySession]:
        """Get sessions by type."""
        return await self.find_by(
            {'user_id': user_id, 'session_type': session_type},
            limit=limit,
            order_by='started_at DESC'
        )

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_streak(self, user_id: str) -> int:
        """
        Get user's current study streak (consecutive days).

        Args:
            user_id: User identifier

        Returns:
            Streak in days
        """
        # Use the database function
        query = "SELECT get_user_streak($1)"
        result = await self.db.fetchval(query, user_id)
        return result or 0

    async def get_weekly_summary(
        self,
        user_id: str,
        weeks: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Get weekly study summary.

        Args:
            user_id: User identifier
            weeks: Weeks to look back

        Returns:
            List of weekly summary dicts
        """
        query = f"""
            SELECT
                DATE_TRUNC('week', started_at) as week_start,
                COUNT(*) as sessions,
                SUM(cards_reviewed) as total_cards,
                SUM(cards_correct) as total_correct,
                SUM(duration_seconds) / 60.0 as total_minutes,
                AVG(cards_correct::FLOAT / NULLIF(cards_reviewed, 0)) as avg_accuracy
            FROM {self.table_name}
            WHERE user_id = $1
              AND started_at >= NOW() - INTERVAL '{weeks} weeks'
              AND ended_at IS NOT NULL
            GROUP BY DATE_TRUNC('week', started_at)
            ORDER BY week_start DESC
        """

        rows = await self.db.fetch(query, user_id)
        return [dict(row) for row in rows]

    async def get_session_type_distribution(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, int]:
        """
        Get distribution of session types.

        Returns:
            {session_type: count}
        """
        query = f"""
            SELECT session_type, COUNT(*) as count
            FROM {self.table_name}
            WHERE user_id = $1
              AND started_at >= NOW() - INTERVAL '{days} days'
            GROUP BY session_type
        """

        rows = await self.db.fetch(query, user_id)
        return {row['session_type']: row['count'] for row in rows}

    async def get_average_session_length(
        self,
        user_id: str,
        days: int = 30
    ) -> float:
        """
        Get average session length in minutes.

        Returns:
            Average minutes per session
        """
        query = f"""
            SELECT AVG(duration_seconds) / 60.0 as avg_minutes
            FROM {self.table_name}
            WHERE user_id = $1
              AND started_at >= NOW() - INTERVAL '{days} days'
              AND ended_at IS NOT NULL
        """

        result = await self.db.fetchval(query, user_id)
        return result or 0.0

    async def get_best_time_of_day(
        self,
        user_id: str,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze best performance by time of day.

        Returns:
            {hour: {accuracy, avg_response_time}}
        """
        query = f"""
            SELECT
                EXTRACT(HOUR FROM started_at) as hour,
                AVG(cards_correct::FLOAT / NULLIF(cards_reviewed, 0)) as accuracy,
                AVG(avg_response_time_ms) as avg_response_time,
                COUNT(*) as sessions
            FROM {self.table_name}
            WHERE user_id = $1
              AND started_at >= NOW() - INTERVAL '{days} days'
              AND ended_at IS NOT NULL
              AND cards_reviewed > 0
            GROUP BY EXTRACT(HOUR FROM started_at)
            HAVING COUNT(*) >= 3
            ORDER BY accuracy DESC
        """

        rows = await self.db.fetch(query, user_id)
        return {int(row['hour']): dict(row) for row in rows}
