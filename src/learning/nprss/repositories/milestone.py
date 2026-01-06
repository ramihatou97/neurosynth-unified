# src/learning/nprss/repositories/milestone.py
"""
Milestone Repository

Achievement and milestone tracking for gamification.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base import BaseRepository


@dataclass
class Milestone:
    """Achievement milestone."""
    id: UUID = None
    user_id: str = ""

    # Milestone identification
    milestone_type: str = ""  # procedure_mastery, streak_achievement, card_count, etc.
    milestone_name: str = ""
    milestone_level: int = 1

    # Achievement
    achieved_at: datetime = None

    # Context
    procedure_id: UUID = None
    specialty: str = None

    # Metrics at achievement
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Notification
    notified: bool = False
    notified_at: datetime = None


class MilestoneRepository(BaseRepository[Milestone]):
    """
    Repository for achievement milestones.

    Provides gamification tracking and achievement management.
    """

    @property
    def table_name(self) -> str:
        return "mastery_milestones"

    def _row_to_entity(self, row: Dict[str, Any]) -> Milestone:
        """Convert database row to Milestone."""
        return Milestone(
            id=row.get('id'),
            user_id=row.get('user_id', ''),
            milestone_type=row.get('milestone_type', ''),
            milestone_name=row.get('milestone_name', ''),
            milestone_level=row.get('milestone_level', 1),
            achieved_at=row.get('achieved_at'),
            procedure_id=row.get('procedure_id'),
            specialty=row.get('specialty'),
            metrics=row.get('metrics', {}),
            notified=row.get('notified', False),
            notified_at=row.get('notified_at')
        )

    def _entity_to_dict(self, entity: Milestone) -> Dict[str, Any]:
        """Convert Milestone to dict."""
        return {
            'id': entity.id,
            'user_id': entity.user_id,
            'milestone_type': entity.milestone_type,
            'milestone_name': entity.milestone_name,
            'milestone_level': entity.milestone_level,
            'achieved_at': entity.achieved_at or datetime.now(),
            'procedure_id': entity.procedure_id,
            'specialty': entity.specialty,
            'metrics': entity.metrics,
            'notified': entity.notified,
            'notified_at': entity.notified_at
        }

    # =========================================================================
    # Milestone Management
    # =========================================================================

    async def award_milestone(
        self,
        user_id: str,
        milestone_type: str,
        milestone_name: str,
        metrics: Dict[str, Any] = None,
        procedure_id: UUID = None,
        specialty: str = None
    ) -> Optional[Milestone]:
        """
        Award a milestone (idempotent - won't duplicate).

        Args:
            user_id: User identifier
            milestone_type: Type of milestone
            milestone_name: Name of milestone
            metrics: Metrics at achievement time
            procedure_id: Optional procedure context
            specialty: Optional specialty context

        Returns:
            Milestone if newly awarded, None if already exists
        """
        # Check if already exists
        existing = await self.find_one_by({
            'user_id': user_id,
            'milestone_type': milestone_type,
            'milestone_name': milestone_name,
            'procedure_id': procedure_id
        })

        if existing:
            return None

        milestone = Milestone(
            user_id=user_id,
            milestone_type=milestone_type,
            milestone_name=milestone_name,
            achieved_at=datetime.now(),
            procedure_id=procedure_id,
            specialty=specialty,
            metrics=metrics or {}
        )

        return await self.create(milestone)

    async def check_and_award(self, user_id: str) -> List[Milestone]:
        """
        Check all milestone conditions and award eligible ones.

        Uses database function check_milestones().

        Args:
            user_id: User identifier

        Returns:
            List of newly awarded milestones
        """
        query = "SELECT * FROM check_milestones($1)"
        rows = await self.db.fetch(query, user_id)

        # Return only newly achieved
        return [
            Milestone(
                user_id=user_id,
                milestone_type=row['milestone_type'],
                milestone_name=row['milestone_name'],
                achieved_at=datetime.now()
            )
            for row in rows
            if row.get('newly_achieved', False)
        ]

    async def mark_notified(self, milestone_id: UUID) -> None:
        """Mark milestone as notified."""
        await self.update(milestone_id, {
            'notified': True,
            'notified_at': datetime.now()
        })

    async def mark_all_notified(self, user_id: str) -> int:
        """Mark all user's milestones as notified."""
        query = f"""
            UPDATE {self.table_name}
            SET notified = true, notified_at = NOW()
            WHERE user_id = $1 AND notified = false
        """
        result = await self.db.execute(query, user_id)
        return int(result.split()[-1]) if result else 0

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_user_milestones(
        self,
        user_id: str,
        milestone_type: str = None,
        limit: int = 100
    ) -> List[Milestone]:
        """
        Get user's milestones.

        Args:
            user_id: User identifier
            milestone_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of milestones (newest first)
        """
        conditions = {'user_id': user_id}
        if milestone_type:
            conditions['milestone_type'] = milestone_type

        return await self.find_by(
            conditions,
            limit=limit,
            order_by='achieved_at DESC'
        )

    async def get_unnotified(self, user_id: str) -> List[Milestone]:
        """
        Get milestones not yet shown to user.

        Args:
            user_id: User identifier

        Returns:
            Unnotified milestones
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1 AND notified = false
            ORDER BY achieved_at DESC
        """

        rows = await self.db.fetch(query, user_id)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_recent_achievements(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Milestone]:
        """Get milestones achieved in last N days."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1
              AND achieved_at >= NOW() - INTERVAL '{days} days'
            ORDER BY achieved_at DESC
        """

        rows = await self.db.fetch(query, user_id)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_procedure_milestones(
        self,
        user_id: str,
        procedure_id: UUID
    ) -> List[Milestone]:
        """Get milestones for a specific procedure."""
        return await self.find_by({
            'user_id': user_id,
            'procedure_id': procedure_id
        })

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_milestone_counts(
        self,
        user_id: str
    ) -> Dict[str, int]:
        """
        Get count of milestones by type.

        Returns:
            {type: count}
        """
        query = f"""
            SELECT milestone_type, COUNT(*) as count
            FROM {self.table_name}
            WHERE user_id = $1
            GROUP BY milestone_type
        """

        rows = await self.db.fetch(query, user_id)
        return {row['milestone_type']: row['count'] for row in rows}

    async def get_total_achievements(self, user_id: str) -> int:
        """Get total number of achievements."""
        return await self.count({'user_id': user_id})

    async def get_leaderboard(
        self,
        milestone_type: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get achievement leaderboard.

        Args:
            milestone_type: Optional filter
            limit: Top N users

        Returns:
            List of {user_id, count, latest_achievement}
        """
        type_filter = "WHERE milestone_type = $2" if milestone_type else ""

        query = f"""
            SELECT
                user_id,
                COUNT(*) as total_achievements,
                MAX(achieved_at) as latest_achievement
            FROM {self.table_name}
            {type_filter}
            GROUP BY user_id
            ORDER BY total_achievements DESC
            LIMIT $1
        """

        params = [limit]
        if milestone_type:
            params.append(milestone_type)

        rows = await self.db.fetch(query, *params)
        return [dict(row) for row in rows]


# =============================================================================
# MILESTONE DEFINITIONS
# =============================================================================

MILESTONE_DEFINITIONS = {
    # Card count milestones
    'card_count': [
        {'name': 'First Steps', 'threshold': 10, 'level': 1},
        {'name': 'Getting Started', 'threshold': 50, 'level': 2},
        {'name': 'Century Reviewer', 'threshold': 100, 'level': 3},
        {'name': 'Dedicated Learner', 'threshold': 500, 'level': 4},
        {'name': 'Millennium Scholar', 'threshold': 1000, 'level': 5},
        {'name': 'Master Reviewer', 'threshold': 5000, 'level': 6},
    ],

    # Streak milestones
    'streak_achievement': [
        {'name': 'Three Day Streak', 'threshold': 3, 'level': 1},
        {'name': 'Week Warrior', 'threshold': 7, 'level': 2},
        {'name': 'Two Week Champion', 'threshold': 14, 'level': 3},
        {'name': 'Monthly Master', 'threshold': 30, 'level': 4},
        {'name': 'Quarterly Quest', 'threshold': 90, 'level': 5},
        {'name': 'Year of Dedication', 'threshold': 365, 'level': 6},
    ],

    # Accuracy milestones
    'accuracy_achievement': [
        {'name': 'Sharp Mind', 'threshold': 0.8, 'level': 1},
        {'name': 'Precision Expert', 'threshold': 0.9, 'level': 2},
        {'name': 'Near Perfect', 'threshold': 0.95, 'level': 3},
        {'name': 'Flawless Memory', 'threshold': 0.99, 'level': 4},
    ],

    # R-level milestones
    'r_level_complete': [
        {'name': 'R1 Foundation', 'threshold': 1, 'level': 1},
        {'name': 'R3 Consolidator', 'threshold': 3, 'level': 2},
        {'name': 'R5 Long-Term', 'threshold': 5, 'level': 3},
        {'name': 'R7 Master', 'threshold': 7, 'level': 4},
    ],

    # Procedure milestones
    'procedure_mastery': [
        {'name': 'Procedure Familiar', 'threshold': 'developing', 'level': 1},
        {'name': 'Procedure Competent', 'threshold': 'competent', 'level': 2},
        {'name': 'Procedure Expert', 'threshold': 'mastery', 'level': 3},
    ],

    # Session milestones
    'session_milestone': [
        {'name': 'First Session', 'threshold': 1, 'level': 1},
        {'name': '10 Sessions', 'threshold': 10, 'level': 2},
        {'name': '50 Sessions', 'threshold': 50, 'level': 3},
        {'name': '100 Sessions', 'threshold': 100, 'level': 4},
    ],
}
