# src/learning/nprss/repositories/base.py
"""
Base Repository Pattern

Abstract base class following NeuroSynth's existing repository pattern.
Provides common CRUD operations and query building.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic
from uuid import UUID
from datetime import datetime


T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common CRUD operations.

    Subclasses must implement:
    - table_name: Database table name
    - _row_to_entity: Convert database row to entity
    - _entity_to_dict: Convert entity to dict for insert/update
    """

    def __init__(self, db_connection):
        """
        Initialize repository with database connection.

        Args:
            db_connection: AsyncPG connection pool or connection
        """
        self.db = db_connection

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Database table name."""
        pass

    @property
    def primary_key(self) -> str:
        """Primary key column name. Override if not 'id'."""
        return 'id'

    @abstractmethod
    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Convert database row to entity object."""
        pass

    @abstractmethod
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dict for database operations."""
        pass

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """
        Get entity by primary key.

        Args:
            entity_id: Entity UUID

        Returns:
            Entity or None if not found
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {self.primary_key} = $1
        """
        row = await self.db.fetchrow(query, entity_id)
        return self._row_to_entity(dict(row)) if row else None

    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = None
    ) -> List[T]:
        """
        Get all entities with pagination.

        Args:
            limit: Maximum entities to return
            offset: Number of entities to skip
            order_by: Column to sort by (e.g., 'created_at DESC')

        Returns:
            List of entities
        """
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        query = f"""
            SELECT * FROM {self.table_name}
            {order_clause}
            LIMIT $1 OFFSET $2
        """
        rows = await self.db.fetch(query, limit, offset)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def create(self, entity: T) -> T:
        """
        Create new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity with generated ID
        """
        data = self._entity_to_dict(entity)

        # Remove None values and auto-generated fields
        data = {k: v for k, v in data.items() if v is not None}
        data.pop('id', None)
        data.pop('created_at', None)
        data.pop('updated_at', None)

        columns = list(data.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        values = list(data.values())

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING *
        """

        row = await self.db.fetchrow(query, *values)
        return self._row_to_entity(dict(row))

    async def update(self, entity_id: UUID, updates: Dict[str, Any]) -> Optional[T]:
        """
        Update entity by ID.

        Args:
            entity_id: Entity UUID
            updates: Dict of column:value updates

        Returns:
            Updated entity or None if not found
        """
        # Filter out None values and protected fields
        updates = {k: v for k, v in updates.items()
                   if v is not None and k not in ('id', 'created_at')}

        if not updates:
            return await self.get_by_id(entity_id)

        # Add updated_at if column exists
        updates['updated_at'] = datetime.now()

        set_clauses = [f"{k} = ${i+2}" for i, k in enumerate(updates.keys())]
        values = [entity_id] + list(updates.values())

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE {self.primary_key} = $1
            RETURNING *
        """

        row = await self.db.fetchrow(query, *values)
        return self._row_to_entity(dict(row)) if row else None

    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete entity by ID.

        Args:
            entity_id: Entity UUID

        Returns:
            True if deleted, False if not found
        """
        query = f"""
            DELETE FROM {self.table_name}
            WHERE {self.primary_key} = $1
            RETURNING {self.primary_key}
        """
        result = await self.db.fetchrow(query, entity_id)
        return result is not None

    # =========================================================================
    # Query Helpers
    # =========================================================================

    async def find_by(
        self,
        conditions: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
        order_by: str = None
    ) -> List[T]:
        """
        Find entities by conditions.

        Args:
            conditions: Dict of column:value conditions (AND'd together)
            limit: Maximum entities to return
            offset: Number to skip
            order_by: Sort column

        Returns:
            List of matching entities
        """
        if not conditions:
            return await self.get_all(limit, offset, order_by)

        where_clauses = [f"{k} = ${i+1}" for i, k in enumerate(conditions.keys())]
        values = list(conditions.values())

        order_clause = f"ORDER BY {order_by}" if order_by else ""

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(where_clauses)}
            {order_clause}
            LIMIT ${len(values)+1} OFFSET ${len(values)+2}
        """

        rows = await self.db.fetch(query, *values, limit, offset)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def find_one_by(self, conditions: Dict[str, Any]) -> Optional[T]:
        """
        Find single entity by conditions.

        Args:
            conditions: Dict of column:value conditions

        Returns:
            Entity or None
        """
        results = await self.find_by(conditions, limit=1)
        return results[0] if results else None

    async def count(self, conditions: Dict[str, Any] = None) -> int:
        """
        Count entities matching conditions.

        Args:
            conditions: Optional filter conditions

        Returns:
            Count of matching entities
        """
        if not conditions:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            result = await self.db.fetchval(query)
        else:
            where_clauses = [f"{k} = ${i+1}" for i, k in enumerate(conditions.keys())]
            values = list(conditions.values())
            query = f"""
                SELECT COUNT(*) FROM {self.table_name}
                WHERE {' AND '.join(where_clauses)}
            """
            result = await self.db.fetchval(query, *values)

        return result or 0

    async def exists(self, conditions: Dict[str, Any]) -> bool:
        """
        Check if entity exists matching conditions.

        Args:
            conditions: Filter conditions

        Returns:
            True if exists, False otherwise
        """
        count = await self.count(conditions)
        return count > 0

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def create_batch(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in single transaction.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities
        """
        if not entities:
            return []

        created = []
        async with self.db.transaction():
            for entity in entities:
                created.append(await self.create(entity))

        return created

    async def delete_by(self, conditions: Dict[str, Any]) -> int:
        """
        Delete entities matching conditions.

        Args:
            conditions: Filter conditions

        Returns:
            Number of deleted entities
        """
        if not conditions:
            raise ValueError("Conditions required for batch delete")

        where_clauses = [f"{k} = ${i+1}" for i, k in enumerate(conditions.keys())]
        values = list(conditions.values())

        query = f"""
            DELETE FROM {self.table_name}
            WHERE {' AND '.join(where_clauses)}
        """

        result = await self.db.execute(query, *values)
        # Parse "DELETE N" result
        return int(result.split()[-1]) if result else 0
