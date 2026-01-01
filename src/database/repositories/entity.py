"""
NeuroSynth Unified - Entity Repository
=======================================

Repository for UMLS entity operations.
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from dataclasses import dataclass

from src.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity data class."""
    id: UUID
    cui: str
    name: str
    semantic_type: Optional[str] = None
    tui: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class EntityRepository(BaseRepository[Entity]):
    """Repository for UMLS entities."""

    @property
    def table_name(self) -> str:
        return "entities"

    def _to_entity(self, row: dict) -> Entity:
        """Convert database row to Entity object."""
        return Entity(
            id=row['id'],
            cui=row['cui'],
            name=row['name'],
            semantic_type=row.get('semantic_type'),
            tui=row.get('tui'),
            chunk_count=row.get('chunk_count', 0) or 0,
            image_count=row.get('image_count', 0) or 0,
            metadata=row.get('metadata')
        )

    def _to_record(self, entity: Entity) -> Dict[str, Any]:
        """Convert Entity object to database record."""
        return {
            'id': entity.id,
            'cui': entity.cui,
            'name': entity.name,
            'semantic_type': entity.semantic_type,
            'tui': entity.tui,
            'chunk_count': entity.chunk_count,
            'image_count': entity.image_count,
            'metadata': entity.metadata or {}
        }

    async def upsert(
        self,
        cui: str,
        name: str,
        semantic_type: Optional[str] = None,
        tui: Optional[str] = None,
        chunk_count_increment: int = 1
    ) -> UUID:
        """
        Insert or update entity, incrementing chunk count.

        Args:
            cui: UMLS Concept Unique Identifier
            name: Entity name
            semantic_type: Semantic type name
            tui: Type Unique Identifier
            chunk_count_increment: Amount to increment chunk_count by

        Returns:
            UUID of the entity
        """
        query = """
            INSERT INTO entities (cui, name, semantic_type, tui, chunk_count)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (cui) DO UPDATE SET
                name = COALESCE(NULLIF(EXCLUDED.name, ''), entities.name),
                semantic_type = COALESCE(EXCLUDED.semantic_type, entities.semantic_type),
                tui = COALESCE(EXCLUDED.tui, entities.tui),
                chunk_count = entities.chunk_count + EXCLUDED.chunk_count
            RETURNING id
        """
        return await self.db.fetchval(query, cui, name, semantic_type, tui, chunk_count_increment)

    async def upsert_many(self, entities_data: List[Dict[str, Any]]) -> int:
        """
        Bulk upsert entities.

        Args:
            entities_data: List of dicts with keys: cui, name, semantic_type, tui, chunk_count_increment

        Returns:
            Number of entities upserted
        """
        if not entities_data:
            return 0

        # Group by CUI and sum counts
        cui_data = {}
        for e in entities_data:
            cui = e['cui']
            if cui not in cui_data:
                cui_data[cui] = {
                    'cui': cui,
                    'name': e.get('name', cui),
                    'semantic_type': e.get('semantic_type'),
                    'tui': e.get('tui'),
                    'chunk_count': 0
                }
            cui_data[cui]['chunk_count'] += e.get('chunk_count_increment', 1)
            # Update name if current is better
            if e.get('name') and e['name'] != cui:
                cui_data[cui]['name'] = e['name']

        # Batch insert/update
        query = """
            INSERT INTO entities (cui, name, semantic_type, tui, chunk_count)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (cui) DO UPDATE SET
                name = COALESCE(NULLIF(EXCLUDED.name, ''), entities.name),
                semantic_type = COALESCE(EXCLUDED.semantic_type, entities.semantic_type),
                tui = COALESCE(EXCLUDED.tui, entities.tui),
                chunk_count = entities.chunk_count + EXCLUDED.chunk_count
        """

        for data in cui_data.values():
            await self.db.execute(
                query,
                data['cui'],
                data['name'],
                data['semantic_type'],
                data['tui'],
                data['chunk_count']
            )

        return len(cui_data)

    async def get_by_cui(self, cui: str) -> Optional[Entity]:
        """Get entity by CUI."""
        query = "SELECT * FROM entities WHERE cui = $1"
        row = await self.db.fetchrow(query, cui)
        return self._to_entity(dict(row)) if row else None

    async def search_by_name(self, query: str, limit: int = 20) -> List[Entity]:
        """Search entities by name."""
        sql = """
            SELECT * FROM entities
            WHERE name ILIKE '%' || $1 || '%'
            ORDER BY chunk_count DESC
            LIMIT $2
        """
        rows = await self.db.fetch(sql, query, limit)
        return [self._to_entity(dict(row)) for row in rows]

    async def get_top_entities(self, limit: int = 100) -> List[Entity]:
        """Get entities sorted by occurrence count."""
        query = """
            SELECT * FROM entities
            ORDER BY chunk_count DESC
            LIMIT $1
        """
        rows = await self.db.fetch(query, limit)
        return [self._to_entity(dict(row)) for row in rows]
