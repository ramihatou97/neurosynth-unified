"""
NeuroSynth Unified - Base Repository
=====================================

Abstract base class for repository pattern with common CRUD operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Type, Set
from uuid import UUID
from datetime import datetime

from asyncpg import Connection
import numpy as np

logger = logging.getLogger(__name__)

# Generic type for entity classes
T = TypeVar('T')

# Whitelist of allowed ORDER BY columns for SQL injection prevention
ALLOWED_ORDER_COLUMNS: Set[str] = {
    'created_at', 'updated_at', 'id', 'name', 'title',
    'score', 'chunk_count', 'page_number', 'sequence_in_doc',
    'ingested_at', 'occurrence_count', 'file_name'
}

# Whitelist of allowed filter columns for SQL injection prevention
ALLOWED_FILTER_COLUMNS: Set[str] = {
    'document_id', 'chunk_type', 'specialty', 'image_type',
    'page_number', 'is_decorative', 'link_type', 'status',
    'chunk_id', 'image_id'
}

# Whitelist of allowed update columns for SQL injection prevention
ALLOWED_UPDATE_COLUMNS: Set[str] = {
    'title', 'content', 'status', 'score', 'embedding',
    'metadata', 'caption', 'vlm_caption', 'is_decorative',
    'authority_score', 'chunk_type', 'image_type', 'processed',
    'specialty_relevance', 'topic_tags', 'entity_mentions'
}

# Whitelist of allowed condition columns for find_by() SQL injection prevention
ALLOWED_CONDITION_COLUMNS: Set[str] = {
    # Primary keys and foreign keys
    'id', 'document_id', 'chunk_id', 'image_id', 'entity_id',
    # Common filter fields
    'cui', 'name', 'title', 'file_path', 'source_path',
    'chunk_type', 'specialty', 'page_number', 'start_page',
    'image_type', 'is_decorative', 'semantic_type',
    'link_type', 'relation_type', 'tui', 'source',
    # Graph fields
    'source_entity_id', 'target_entity_id',
    # Status fields
    'status', 'content_hash'
}


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common CRUD operations.
    
    Subclasses must implement:
    - table_name: str
    - _to_entity(row) -> T
    - _to_record(entity) -> dict
    """
    
    def __init__(self, connection):
        """
        Initialize repository with database connection.
        
        Args:
            connection: DatabaseConnection instance
        """
        self.db = connection
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Table name for this repository."""
        pass

    @property
    def updatable_columns(self) -> Set[str]:
        """
        Columns that can be updated via update() method.

        Override in subclasses to restrict which columns can be modified.
        Empty set means use global ALLOWED_UPDATE_COLUMNS (backward compatible).
        """
        return set()

    @abstractmethod
    def _to_entity(self, row: dict) -> T:
        """Convert database row to entity object."""
        pass
    
    @abstractmethod
    def _to_record(self, entity: T) -> Dict[str, Any]:
        """Convert entity object to database record."""
        pass

    def _validate_order_by(self, order_by: str) -> str:
        """Validate and sanitize ORDER BY clause to prevent SQL injection."""
        if not order_by:
            return "created_at DESC"

        parts = order_by.strip().split()
        if len(parts) == 1:
            column, direction = parts[0], "ASC"
        elif len(parts) == 2:
            column, direction = parts
        else:
            logger.warning(f"Invalid order_by format: '{order_by}'. Using fallback.")
            return "created_at DESC"

        if column.lower() not in ALLOWED_ORDER_COLUMNS:
            logger.warning(f"Invalid sort column: '{column}'. Using fallback.")
            return "created_at DESC"

        if direction.upper() not in ('ASC', 'DESC'):
            logger.warning(f"Invalid sort direction: '{direction}'. Defaulting to ASC.")
            return f"{column} ASC"

        return f"{column} {direction.upper()}"

    def _validate_condition_column(self, column: str) -> bool:
        """
        Validate that a column name is allowed in WHERE conditions.

        Prevents SQL injection via dynamic column names in find_by().

        Args:
            column: Column name to validate

        Returns:
            True if column is allowed, False otherwise
        """
        if column.lower() not in ALLOWED_CONDITION_COLUMNS:
            logger.warning(
                f"Invalid condition column: '{column}' for table {self.table_name}. "
                f"Skipping this condition. Add to ALLOWED_CONDITION_COLUMNS if legitimate."
            )
            return False
        return True

    # =========================================================================
    # Read Operations
    # =========================================================================
    
    async def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        row = await self.db.fetchrow(query, id)
        return self._to_entity(dict(row)) if row else None
    
    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC"
    ) -> List[T]:
        """Get all entities with pagination."""
        safe_order_by = self._validate_order_by(order_by)
        query = f"""
            SELECT * FROM {self.table_name}
            ORDER BY {safe_order_by}
            LIMIT $1 OFFSET $2
        """
        rows = await self.db.fetch(query, limit, offset)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_ids(self, ids: List[UUID]) -> List[T]:
        """
        Get multiple entities by IDs.

        Optimized with CTE for better query planning with large ID lists.
        """
        if not ids:
            return []

        # Use CTE for better performance with large ID lists
        query = f"""
            WITH candidate_ids AS (
                SELECT unnest($1::uuid[]) AS id
            )
            SELECT t.*
            FROM {self.table_name} t
            INNER JOIN candidate_ids ci ON t.id = ci.id
        """

        rows = await self.db.fetch(query, ids)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def exists(self, id: UUID) -> bool:
        """Check if entity exists."""
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE id = $1)"
        return await self.db.fetchval(query, id)
    
    async def count(self) -> int:
        """Count total entities."""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        return await self.db.fetchval(query)
    
    # =========================================================================
    # Write Operations
    # =========================================================================
    
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        record = self._to_record(entity)
        columns = list(record.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(record.values())
        
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING *
        """
        
        row = await self.db.fetchrow(query, *values)
        return self._to_entity(dict(row))
    
    async def create_many(self, entities: List[T]) -> int:
        """Create multiple entities. Returns count of created entities."""
        if not entities:
            return 0
        
        records = [self._to_record(e) for e in entities]
        columns = list(records[0].keys())
        
        # Build batch insert
        async with self.db.transaction() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self.table_name} ({', '.join(columns)})
                VALUES ({', '.join(f'${i+1}' for i in range(len(columns)))})
                """,
                [tuple(r.values()) for r in records]
            )
        
        return len(entities)
    
    async def update(self, id: UUID, updates: Dict[str, Any]) -> Optional[T]:
        """
        Update an entity.

        Security: Only columns in updatable_columns (or ALLOWED_UPDATE_COLUMNS
        if updatable_columns is empty) can be modified.
        """
        if not updates:
            return await self.get_by_id(id)

        # Determine allowed columns
        allowed = self.updatable_columns
        if not allowed:
            # Backward compatible: use global whitelist
            allowed = ALLOWED_UPDATE_COLUMNS

        # Filter to valid columns only
        set_parts = []
        values = []
        param_idx = 1

        for key, value in updates.items():
            if key.lower() not in allowed:
                logger.warning(
                    f"Column '{key}' not allowed for update on {self.table_name}. Skipping."
                )
                continue

            set_parts.append(f"{key} = ${param_idx}")
            values.append(value)
            param_idx += 1

        if not set_parts:
            logger.warning("No valid columns to update. Returning current entity.")
            return await self.get_by_id(id)

        values.append(id)

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_parts)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ${len(values)}
            RETURNING *
        """

        row = await self.db.fetchrow(query, *values)
        return self._to_entity(dict(row)) if row else None
    
    async def delete(self, id: UUID) -> bool:
        """Delete an entity. Returns True if deleted."""
        query = f"DELETE FROM {self.table_name} WHERE id = $1"
        result = await self.db.execute(query, id)
        return result == "DELETE 1"
    
    async def delete_many(self, ids: List[UUID]) -> int:
        """Delete multiple entities. Returns count of deleted."""
        if not ids:
            return 0
        
        query = f"DELETE FROM {self.table_name} WHERE id = ANY($1)"
        result = await self.db.execute(query, ids)
        # Parse "DELETE N" to get count
        return int(result.split()[1])
    
    # =========================================================================
    # Query Helpers
    # =========================================================================
    
    async def find_by(
        self,
        conditions: Dict[str, Any],
        limit: int = 100,
        order_by: str = "created_at DESC"
    ) -> List[T]:
        """
        Find entities matching conditions.

        Security: Only columns in ALLOWED_CONDITION_COLUMNS will be used.
        Invalid columns are logged and skipped.
        """
        if not conditions:
            return await self.get_all(limit=limit, order_by=order_by)

        safe_order_by = self._validate_order_by(order_by)
        where_parts = []
        values = []
        param_idx = 1

        for key, value in conditions.items():
            if not self._validate_condition_column(key):
                continue  # Skip invalid columns

            if isinstance(value, list):
                where_parts.append(f"{key} = ANY(${param_idx})")
            else:
                where_parts.append(f"{key} = ${param_idx}")
            values.append(value)
            param_idx += 1

        if not where_parts:
            logger.warning("All conditions were invalid. Returning empty result.")
            return []

        values.append(limit)

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(where_parts)}
            ORDER BY {safe_order_by}
            LIMIT ${len(values)}
        """

        rows = await self.db.fetch(query, *values)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def find_one_by(self, conditions: Dict[str, Any]) -> Optional[T]:
        """Find single entity matching conditions."""
        results = await self.find_by(conditions, limit=1)
        return results[0] if results else None


# =============================================================================
# Vector Search Mixin
# =============================================================================

class VectorSearchMixin:
    """
    Mixin for repositories that support vector similarity search.
    
    Requires:
    - table_name property
    - embedding_column property
    - db connection
    """
    
    @property
    @abstractmethod
    def embedding_column(self) -> str:
        """Column name for vector embedding."""
        pass
    
    async def search_by_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search by vector similarity using pgvector.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            min_similarity: Minimum similarity threshold (0-1)
            filters: Additional WHERE conditions
        
        Returns:
            List of dicts with entity data and similarity score
        """
        # Encode embedding
        from src.database.connection import DatabaseConnection
        embedding_str = DatabaseConnection._encode_vector(query_embedding)
        
        # Build WHERE clause
        where_parts = [f"{self.embedding_column} IS NOT NULL"]
        params = [embedding_str, top_k]
        param_idx = 3
        
        if filters:
            for key, value in filters.items():
                # Validate filter column to prevent SQL injection
                if key.lower() not in ALLOWED_FILTER_COLUMNS:
                    logger.warning(f"Invalid filter column: '{key}'. Skipping.")
                    continue
                if isinstance(value, list):
                    where_parts.append(f"{key} = ANY(${param_idx})")
                else:
                    where_parts.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1
        
        where_clause = " AND ".join(where_parts)
        
        query = f"""
            SELECT *,
                   1 - ({self.embedding_column} <=> $1::vector) AS similarity
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.embedding_column} <=> $1::vector
            LIMIT $2
        """
        
        rows = await self.db.fetch(query, *params)
        
        # Filter by minimum similarity
        results = []
        for row in rows:
            if row['similarity'] >= min_similarity:
                results.append(dict(row))
        
        return results
    
    async def search_by_vector_hybrid(
        self,
        query_embedding: np.ndarray,
        query_cuis: List[str] = None,
        top_k: int = 10,
        cui_boost: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and CUI matching.
        
        Args:
            query_embedding: Query vector
            query_cuis: List of UMLS CUIs to boost
            top_k: Number of results
            cui_boost: Multiplier for results with matching CUIs
        
        Returns:
            List of dicts with entity data and combined score
        """
        from src.database.connection import DatabaseConnection
        embedding_str = DatabaseConnection._encode_vector(query_embedding)
        
        query_cuis = query_cuis or []
        
        query = f"""
            SELECT *,
                   1 - ({self.embedding_column} <=> $1::vector) AS similarity,
                   COALESCE(array_length(cuis & $2, 1), 0) AS cui_overlap,
                   (1 - ({self.embedding_column} <=> $1::vector)) * 
                       CASE WHEN array_length(cuis & $2, 1) > 0 
                            THEN $4 
                            ELSE 1.0 
                       END AS combined_score
            FROM {self.table_name}
            WHERE {self.embedding_column} IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT $3
        """
        
        rows = await self.db.fetch(query, embedding_str, query_cuis, top_k, cui_boost)
        return [dict(row) for row in rows]
