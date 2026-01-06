# src/learning/nprss/repositories/socratic.py
"""
Socratic Repositories

Database access for Socratic prompts and responses.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, field

from .base import BaseRepository


# =============================================================================
# MODELS
# =============================================================================

@dataclass
class SocraticPrompt:
    """Pre-generated Socratic question."""
    id: UUID = None
    card_id: UUID = None

    # Level
    level: str = "guided"  # guided, reflective, challenging

    # Content
    prompt_text: str = ""
    hint_text: str = None
    followup_prompts: List[str] = field(default_factory=list)

    # Expected response
    expected_concepts: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    min_response_length: int = 20

    # Quality metrics
    quality_score: float = 0.5
    usage_count: int = 0
    avg_user_score: float = None

    created_at: datetime = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocraticResponse:
    """User response to Socratic prompt."""
    id: UUID = None
    user_id: str = ""
    prompt_id: UUID = None
    session_id: UUID = None

    # Response
    response_text: str = ""
    response_time_ms: int = None

    # Evaluation
    concepts_covered: List[str] = field(default_factory=list)
    key_terms_used: List[str] = field(default_factory=list)
    completeness_score: float = None
    accuracy_score: float = None

    # AI feedback
    ai_feedback: str = None
    ai_score: float = None

    # Self-assessment
    user_confidence: int = None

    created_at: datetime = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# REPOSITORIES
# =============================================================================

class SocraticPromptRepository(BaseRepository[SocraticPrompt]):
    """
    Repository for Socratic prompts.

    Pre-generated questions for guided learning.
    """

    @property
    def table_name(self) -> str:
        return "socratic_prompts"

    def _row_to_entity(self, row: Dict[str, Any]) -> SocraticPrompt:
        """Convert database row to SocraticPrompt."""
        return SocraticPrompt(
            id=row.get('id'),
            card_id=row.get('card_id'),
            level=row.get('level', 'guided'),
            prompt_text=row.get('prompt_text', ''),
            hint_text=row.get('hint_text'),
            followup_prompts=row.get('followup_prompts', []),
            expected_concepts=row.get('expected_concepts', []),
            key_terms=row.get('key_terms', []),
            min_response_length=row.get('min_response_length', 20),
            quality_score=row.get('quality_score', 0.5),
            usage_count=row.get('usage_count', 0),
            avg_user_score=row.get('avg_user_score'),
            created_at=row.get('created_at'),
            metadata=row.get('metadata', {})
        )

    def _entity_to_dict(self, entity: SocraticPrompt) -> Dict[str, Any]:
        """Convert SocraticPrompt to dict."""
        return {
            'id': entity.id,
            'card_id': entity.card_id,
            'level': entity.level,
            'prompt_text': entity.prompt_text,
            'hint_text': entity.hint_text,
            'followup_prompts': entity.followup_prompts,
            'expected_concepts': entity.expected_concepts,
            'key_terms': entity.key_terms,
            'min_response_length': entity.min_response_length,
            'quality_score': entity.quality_score,
            'usage_count': entity.usage_count,
            'avg_user_score': entity.avg_user_score,
            'metadata': entity.metadata
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_prompts_for_card(
        self,
        card_id: UUID,
        level: str = None
    ) -> List[SocraticPrompt]:
        """
        Get Socratic prompts for a card.

        Args:
            card_id: Card UUID
            level: Optional level filter

        Returns:
            List of prompts
        """
        conditions = {'card_id': card_id}
        if level:
            conditions['level'] = level

        return await self.find_by(conditions, order_by='quality_score DESC')

    async def get_random_prompt(
        self,
        card_id: UUID,
        level: str = None
    ) -> Optional[SocraticPrompt]:
        """
        Get random prompt for a card.

        Args:
            card_id: Card UUID
            level: Optional level filter

        Returns:
            Random prompt or None
        """
        level_filter = "AND level = $2" if level else ""

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE card_id = $1
              {level_filter}
            ORDER BY RANDOM()
            LIMIT 1
        """

        params = [card_id]
        if level:
            params.append(level)

        row = await self.db.fetchrow(query, *params)
        return self._row_to_entity(dict(row)) if row else None

    async def increment_usage(self, prompt_id: UUID) -> None:
        """Increment prompt usage count."""
        query = f"""
            UPDATE {self.table_name}
            SET usage_count = usage_count + 1
            WHERE id = $1
        """
        await self.db.execute(query, prompt_id)

    async def update_avg_score(
        self,
        prompt_id: UUID,
        new_score: float
    ) -> None:
        """
        Update average user score with new observation.

        Uses running average formula.
        """
        query = f"""
            UPDATE {self.table_name}
            SET avg_user_score = CASE
                WHEN avg_user_score IS NULL THEN $2
                ELSE (avg_user_score * usage_count + $2) / (usage_count + 1)
            END
            WHERE id = $1
        """
        await self.db.execute(query, prompt_id, new_score)

    async def get_by_level(
        self,
        level: str,
        min_quality: float = 0.5,
        limit: int = 100
    ) -> List[SocraticPrompt]:
        """Get prompts by level with quality filter."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE level = $1 AND quality_score >= $2
            ORDER BY quality_score DESC
            LIMIT $3
        """

        rows = await self.db.fetch(query, level, min_quality, limit)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def bulk_create_for_card(
        self,
        card_id: UUID,
        prompts: List[Dict[str, Any]]
    ) -> List[SocraticPrompt]:
        """
        Create multiple prompts for a card.

        Args:
            card_id: Card UUID
            prompts: List of prompt data dicts

        Returns:
            Created prompts
        """
        created = []
        for prompt_data in prompts:
            prompt = SocraticPrompt(
                card_id=card_id,
                level=prompt_data.get('level', 'guided'),
                prompt_text=prompt_data['prompt_text'],
                hint_text=prompt_data.get('hint_text'),
                followup_prompts=prompt_data.get('followup_prompts', []),
                expected_concepts=prompt_data.get('expected_concepts', []),
                key_terms=prompt_data.get('key_terms', []),
                quality_score=prompt_data.get('quality_score', 0.5)
            )
            created.append(await self.create(prompt))

        return created


class SocraticResponseRepository(BaseRepository[SocraticResponse]):
    """
    Repository for Socratic responses.

    User responses with evaluation and feedback.
    """

    @property
    def table_name(self) -> str:
        return "socratic_responses"

    def _row_to_entity(self, row: Dict[str, Any]) -> SocraticResponse:
        """Convert database row to SocraticResponse."""
        return SocraticResponse(
            id=row.get('id'),
            user_id=row.get('user_id', ''),
            prompt_id=row.get('prompt_id'),
            session_id=row.get('session_id'),
            response_text=row.get('response_text', ''),
            response_time_ms=row.get('response_time_ms'),
            concepts_covered=row.get('concepts_covered', []),
            key_terms_used=row.get('key_terms_used', []),
            completeness_score=row.get('completeness_score'),
            accuracy_score=row.get('accuracy_score'),
            ai_feedback=row.get('ai_feedback'),
            ai_score=row.get('ai_score'),
            user_confidence=row.get('user_confidence'),
            created_at=row.get('created_at'),
            metadata=row.get('metadata', {})
        )

    def _entity_to_dict(self, entity: SocraticResponse) -> Dict[str, Any]:
        """Convert SocraticResponse to dict."""
        return {
            'id': entity.id,
            'user_id': entity.user_id,
            'prompt_id': entity.prompt_id,
            'session_id': entity.session_id,
            'response_text': entity.response_text,
            'response_time_ms': entity.response_time_ms,
            'concepts_covered': entity.concepts_covered,
            'key_terms_used': entity.key_terms_used,
            'completeness_score': entity.completeness_score,
            'accuracy_score': entity.accuracy_score,
            'ai_feedback': entity.ai_feedback,
            'ai_score': entity.ai_score,
            'user_confidence': entity.user_confidence,
            'metadata': entity.metadata
        }

    # =========================================================================
    # Recording Methods
    # =========================================================================

    async def record_response(
        self,
        user_id: str,
        prompt_id: UUID,
        response_text: str,
        response_time_ms: int = None,
        session_id: UUID = None
    ) -> SocraticResponse:
        """
        Record a user's response to a Socratic prompt.

        Evaluation can be added later.

        Args:
            user_id: User identifier
            prompt_id: Prompt UUID
            response_text: User's response
            response_time_ms: Time to respond
            session_id: Optional session

        Returns:
            Created response (without evaluation yet)
        """
        response = SocraticResponse(
            user_id=user_id,
            prompt_id=prompt_id,
            session_id=session_id,
            response_text=response_text,
            response_time_ms=response_time_ms
        )

        return await self.create(response)

    async def add_evaluation(
        self,
        response_id: UUID,
        concepts_covered: List[str],
        key_terms_used: List[str],
        completeness_score: float,
        accuracy_score: float,
        ai_feedback: str = None,
        ai_score: float = None
    ) -> SocraticResponse:
        """
        Add evaluation to a response.

        Called after AI evaluation of the response.
        """
        return await self.update(response_id, {
            'concepts_covered': concepts_covered,
            'key_terms_used': key_terms_used,
            'completeness_score': completeness_score,
            'accuracy_score': accuracy_score,
            'ai_feedback': ai_feedback,
            'ai_score': ai_score
        })

    async def add_self_assessment(
        self,
        response_id: UUID,
        user_confidence: int
    ) -> SocraticResponse:
        """Add user's self-assessment."""
        return await self.update(response_id, {
            'user_confidence': user_confidence
        })

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_user_responses(
        self,
        user_id: str,
        prompt_id: UUID = None,
        limit: int = 50
    ) -> List[SocraticResponse]:
        """
        Get user's Socratic responses.

        Args:
            user_id: User identifier
            prompt_id: Optional prompt filter
            limit: Maximum results

        Returns:
            List of responses (newest first)
        """
        conditions = {'user_id': user_id}
        if prompt_id:
            conditions['prompt_id'] = prompt_id

        return await self.find_by(
            conditions,
            limit=limit,
            order_by='created_at DESC'
        )

    async def get_session_responses(
        self,
        session_id: UUID
    ) -> List[SocraticResponse]:
        """Get all Socratic responses in a session."""
        return await self.find_by(
            {'session_id': session_id},
            order_by='created_at'
        )

    async def get_responses_for_prompt(
        self,
        prompt_id: UUID,
        limit: int = 100
    ) -> List[SocraticResponse]:
        """Get all responses to a specific prompt."""
        return await self.find_by(
            {'prompt_id': prompt_id},
            limit=limit,
            order_by='created_at DESC'
        )

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_user_socratic_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get user's Socratic mode statistics.

        Returns:
            Stats dict with counts, averages, etc.
        """
        query = f"""
            SELECT
                COUNT(*) as total_responses,
                AVG(completeness_score) as avg_completeness,
                AVG(accuracy_score) as avg_accuracy,
                AVG(ai_score) as avg_ai_score,
                AVG(user_confidence) as avg_confidence,
                AVG(response_time_ms) as avg_response_time
            FROM {self.table_name}
            WHERE user_id = $1
              AND created_at >= NOW() - INTERVAL '{days} days'
        """

        row = await self.db.fetchrow(query, user_id)
        return dict(row) if row else {}

    async def get_prompt_effectiveness(
        self,
        prompt_id: UUID
    ) -> Dict[str, Any]:
        """
        Get effectiveness metrics for a prompt.

        Returns:
            {total_uses, avg_accuracy, avg_completeness, avg_time}
        """
        query = f"""
            SELECT
                COUNT(*) as total_uses,
                AVG(completeness_score) as avg_completeness,
                AVG(accuracy_score) as avg_accuracy,
                AVG(ai_score) as avg_ai_score,
                AVG(response_time_ms) as avg_response_time
            FROM {self.table_name}
            WHERE prompt_id = $1
        """

        row = await self.db.fetchrow(query, prompt_id)
        return dict(row) if row else {}

    async def get_level_comparison(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare user performance across Socratic levels.

        Returns:
            {level: {avg_accuracy, avg_completeness}}
        """
        query = f"""
            SELECT
                sp.level,
                AVG(sr.completeness_score) as avg_completeness,
                AVG(sr.accuracy_score) as avg_accuracy,
                COUNT(*) as count
            FROM {self.table_name} sr
            JOIN socratic_prompts sp ON sr.prompt_id = sp.id
            WHERE sr.user_id = $1
              AND sr.created_at >= NOW() - INTERVAL '{days} days'
            GROUP BY sp.level
        """

        rows = await self.db.fetch(query, user_id)
        return {row['level']: dict(row) for row in rows}
