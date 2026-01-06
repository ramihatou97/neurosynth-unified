# src/learning/nprss/repositories/learning_item.py
"""
Learning Item Repository

Database access for learning cards (flashcards, MCQs, etc.).
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import asdict

from .base import BaseRepository
from ..models import LearningCard, CardType


class LearningItemRepository(BaseRepository[LearningCard]):
    """
    Repository for learning cards.

    Maps to: nprss_learning_cards table
    """

    @property
    def table_name(self) -> str:
        return "nprss_learning_cards"

    def _row_to_entity(self, row: Dict[str, Any]) -> LearningCard:
        """Convert database row to LearningCard."""
        return LearningCard(
            id=row.get('id'),
            procedure_id=row.get('procedure_id'),
            element_id=row.get('element_id'),
            csp_id=row.get('csp_id'),
            card_type=CardType(row['card_type']) if row.get('card_type') else CardType.MCQ,
            prompt=row.get('prompt', ''),
            answer=row.get('answer', ''),
            options=row.get('options'),
            explanation=row.get('explanation'),
            difficulty_preset=row.get('difficulty_preset', 0.3),
            tags=row.get('tags', []),
            cuis=row.get('cuis', []),
            source_chunk_id=row.get('source_chunk_id'),
            source_document_id=row.get('source_document_id'),
            source_page=row.get('source_page'),
            generation_method=row.get('generation_method'),
            quality_score=row.get('quality_score', 0.5),
            active=row.get('active', True),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )

    def _entity_to_dict(self, entity: LearningCard) -> Dict[str, Any]:
        """Convert LearningCard to dict for database."""
        data = asdict(entity) if hasattr(entity, '__dataclass_fields__') else entity.__dict__.copy()

        # Convert enum to string
        if 'card_type' in data and hasattr(data['card_type'], 'value'):
            data['card_type'] = data['card_type'].value

        return data

    # =========================================================================
    # Specialized Queries
    # =========================================================================

    async def get_by_procedure(
        self,
        procedure_id: UUID,
        card_types: List[str] = None,
        active_only: bool = True
    ) -> List[LearningCard]:
        """
        Get cards for a procedure.

        Args:
            procedure_id: Procedure UUID
            card_types: Optional filter by card types
            active_only: Only return active cards

        Returns:
            List of cards
        """
        conditions = ['procedure_id = $1']
        values = [procedure_id]
        param_idx = 2

        if active_only:
            conditions.append(f'active = ${param_idx}')
            values.append(True)
            param_idx += 1

        if card_types:
            conditions.append(f'card_type = ANY(${param_idx})')
            values.append(card_types)
            param_idx += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at
        """

        rows = await self.db.fetch(query, *values)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_element(self, element_id: UUID) -> List[LearningCard]:
        """Get cards for a procedural element."""
        return await self.find_by({'element_id': element_id})

    async def get_by_csp(self, csp_id: UUID) -> List[LearningCard]:
        """Get cards for a CSP."""
        return await self.find_by({'csp_id': csp_id})

    async def get_by_source_chunk(self, chunk_id: UUID) -> List[LearningCard]:
        """Get cards generated from a specific chunk."""
        return await self.find_by({'source_chunk_id': chunk_id})

    async def get_by_document(
        self,
        document_id: UUID,
        limit: int = 200
    ) -> List[LearningCard]:
        """Get all cards from a document."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE source_document_id = $1
            ORDER BY created_at
            LIMIT $2
        """
        rows = await self.db.fetch(query, document_id, limit)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_generation_method(
        self,
        method: str,
        min_quality: float = 0.0,
        limit: int = 100
    ) -> List[LearningCard]:
        """
        Get cards by generation method.

        Args:
            method: Generation method (relation_based, umls_based, table_based, etc.)
            min_quality: Minimum quality score
            limit: Maximum cards to return

        Returns:
            List of cards
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE generation_method = $1
              AND quality_score >= $2
              AND active = true
            ORDER BY quality_score DESC
            LIMIT $3
        """
        rows = await self.db.fetch(query, method, min_quality, limit)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 100
    ) -> List[LearningCard]:
        """
        Search cards by tags.

        Args:
            tags: Tags to search for
            match_all: True = AND, False = OR
            limit: Maximum results

        Returns:
            Matching cards
        """
        if match_all:
            # All tags must be present
            operator = '@>'
        else:
            # Any tag matches
            operator = '&&'

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE tags {operator} $1
              AND active = true
            LIMIT $2
        """
        rows = await self.db.fetch(query, tags, limit)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_statistics(self, procedure_id: UUID = None) -> Dict[str, Any]:
        """
        Get card statistics.

        Args:
            procedure_id: Optional procedure filter

        Returns:
            Statistics dict
        """
        where_clause = "WHERE procedure_id = $1" if procedure_id else ""
        values = [procedure_id] if procedure_id else []

        query = f"""
            SELECT
                COUNT(*) as total_cards,
                COUNT(*) FILTER (WHERE active = true) as active_cards,
                COUNT(DISTINCT card_type) as card_types,
                COUNT(DISTINCT generation_method) as generation_methods,
                AVG(quality_score) as avg_quality,
                COUNT(DISTINCT procedure_id) as procedures_covered
            FROM {self.table_name}
            {where_clause}
        """

        row = await self.db.fetchrow(query, *values)
        return dict(row) if row else {}

    async def deactivate_by_source(self, source_chunk_id: UUID) -> int:
        """
        Deactivate all cards from a source chunk.

        Args:
            source_chunk_id: Source chunk UUID

        Returns:
            Number of cards deactivated
        """
        query = f"""
            UPDATE {self.table_name}
            SET active = false, updated_at = NOW()
            WHERE source_chunk_id = $1
              AND active = true
        """
        result = await self.db.execute(query, source_chunk_id)
        return int(result.split()[-1]) if result else 0

    async def bulk_update_quality(
        self,
        card_ids: List[UUID],
        quality_delta: float
    ) -> int:
        """
        Bulk update quality scores.

        Args:
            card_ids: Card UUIDs to update
            quality_delta: Amount to add to quality (can be negative)

        Returns:
            Number of cards updated
        """
        query = f"""
            UPDATE {self.table_name}
            SET quality_score = GREATEST(0, LEAST(1, quality_score + $1)),
                updated_at = NOW()
            WHERE id = ANY($2)
        """
        result = await self.db.execute(query, quality_delta, card_ids)
        return int(result.split()[-1]) if result else 0
