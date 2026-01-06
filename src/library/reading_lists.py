"""
NeuroSynth - Reading List Manager
=================================

CRUD operations for user-curated document collections.
Supports the "Flight Plan" workflow for procedure preparation.

Usage:
    manager = ReadingListManager(db_pool)
    lists = await manager.get_lists()
    new_list = await manager.create_list("Pterional Essentials", procedure_slug="pterional-craniotomy")
    await manager.add_item(new_list["id"], "path/to/document.pdf", priority=1)
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

logger = logging.getLogger("neurosynth.library.reading_lists")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ReadingListCreate(BaseModel):
    """Request model for creating a reading list."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = ""
    procedure_slug: Optional[str] = None
    specialty: Optional[str] = None


class ReadingListUpdate(BaseModel):
    """Request model for updating a reading list."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    procedure_slug: Optional[str] = None
    specialty: Optional[str] = None
    is_shared: Optional[bool] = None


class ReadingListItemAdd(BaseModel):
    """Request model for adding an item to a list."""
    document_id: str = Field(..., min_length=1)  # file_path from catalog
    priority: int = Field(2, ge=1, le=3)  # 1=Essential, 2=Recommended, 3=Optional
    notes: Optional[str] = ""


class ReadingListItemUpdate(BaseModel):
    """Request model for updating an item."""
    priority: Optional[int] = Field(None, ge=1, le=3)
    notes: Optional[str] = None
    position: Optional[int] = None


# =============================================================================
# READING LIST MANAGER
# =============================================================================

class ReadingListManager:
    """
    Manages reading list CRUD operations.

    Reading lists are user-curated collections of library documents,
    optionally linked to a procedure for case preparation.
    """

    def __init__(self, db_pool):
        """
        Initialize with database pool.

        Args:
            db_pool: asyncpg connection pool from ServiceContainer
        """
        self.pool = db_pool

    # =========================================================================
    # LIST CRUD
    # =========================================================================

    async def create_list(self, data: ReadingListCreate) -> Dict[str, Any]:
        """
        Create a new reading list.

        Returns:
            Dict with id, name, created_at
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO reading_lists (name, description, procedure_slug, specialty)
                VALUES ($1, $2, $3, $4)
                RETURNING id, name, description, procedure_slug, specialty,
                          created_at, updated_at, is_shared, item_count
            """, data.name, data.description, data.procedure_slug, data.specialty)

            logger.info(f"Created reading list: {row['name']} (id={row['id']})")
            return self._row_to_dict(row)

    async def get_lists(
        self,
        procedure_slug: Optional[str] = None,
        specialty: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get all reading lists, optionally filtered.

        Args:
            procedure_slug: Filter by linked procedure
            specialty: Filter by specialty
            include_shared: Include shared lists from other users

        Returns:
            List of reading list dicts
        """
        query = "SELECT * FROM reading_lists WHERE 1=1"
        params = []
        param_idx = 1

        if procedure_slug:
            query += f" AND (procedure_slug = ${param_idx} OR procedure_slug IS NULL)"
            params.append(procedure_slug)
            param_idx += 1

        if specialty:
            query += f" AND (specialty = ${param_idx} OR specialty IS NULL)"
            params.append(specialty)
            param_idx += 1

        query += " ORDER BY updated_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_dict(row) for row in rows]

    async def get_list(self, list_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single reading list by ID.

        Returns:
            Reading list dict or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reading_lists WHERE id = $1",
                list_id if isinstance(list_id, UUID) else UUID(list_id)
            )
            return self._row_to_dict(row) if row else None

    async def get_list_with_items(self, list_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a reading list with all its items.

        Returns:
            Dict with list metadata and items array
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        async with self.pool.acquire() as conn:
            # Get list metadata
            list_row = await conn.fetchrow(
                "SELECT * FROM reading_lists WHERE id = $1",
                list_uuid
            )

            if not list_row:
                return None

            # Get items
            items = await conn.fetch("""
                SELECT document_id, position, priority, notes, added_at
                FROM reading_list_items
                WHERE list_id = $1
                ORDER BY position ASC, added_at DESC
            """, list_uuid)

            result = self._row_to_dict(list_row)
            result["items"] = [self._row_to_dict(item) for item in items]

            return result

    async def update_list(self, list_id: str, data: ReadingListUpdate) -> Optional[Dict[str, Any]]:
        """
        Update a reading list's metadata.

        Returns:
            Updated list dict or None if not found
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        # Build dynamic update query
        updates = []
        params = []
        param_idx = 1

        if data.name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(data.name)
            param_idx += 1

        if data.description is not None:
            updates.append(f"description = ${param_idx}")
            params.append(data.description)
            param_idx += 1

        if data.procedure_slug is not None:
            updates.append(f"procedure_slug = ${param_idx}")
            params.append(data.procedure_slug)
            param_idx += 1

        if data.specialty is not None:
            updates.append(f"specialty = ${param_idx}")
            params.append(data.specialty)
            param_idx += 1

        if data.is_shared is not None:
            updates.append(f"is_shared = ${param_idx}")
            params.append(data.is_shared)
            param_idx += 1

        if not updates:
            return await self.get_list(list_id)

        updates.append("updated_at = NOW()")
        params.append(list_uuid)

        query = f"""
            UPDATE reading_lists
            SET {', '.join(updates)}
            WHERE id = ${param_idx}
            RETURNING *
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return self._row_to_dict(row) if row else None

    async def delete_list(self, list_id: str) -> bool:
        """
        Delete a reading list and all its items.

        Returns:
            True if deleted, False if not found
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM reading_lists WHERE id = $1",
                list_uuid
            )
            deleted = result.split()[-1] != '0'

            if deleted:
                logger.info(f"Deleted reading list: {list_id}")

            return deleted

    # =========================================================================
    # ITEM OPERATIONS
    # =========================================================================

    async def add_item(
        self,
        list_id: str,
        data: ReadingListItemAdd,
    ) -> bool:
        """
        Add a document to a reading list.

        Returns:
            True if added, False if already exists
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        async with self.pool.acquire() as conn:
            # Check if already exists
            exists = await conn.fetchval("""
                SELECT 1 FROM reading_list_items
                WHERE list_id = $1 AND document_id = $2
            """, list_uuid, data.document_id)

            if exists:
                logger.debug(f"Document already in list: {data.document_id}")
                return False

            # Get next position
            next_pos = await conn.fetchval("""
                SELECT COALESCE(MAX(position), 0) + 1
                FROM reading_list_items
                WHERE list_id = $1
            """, list_uuid)

            # Insert item
            await conn.execute("""
                INSERT INTO reading_list_items (list_id, document_id, priority, notes, position)
                VALUES ($1, $2, $3, $4, $5)
            """, list_uuid, data.document_id, data.priority, data.notes, next_pos)

            logger.debug(f"Added document to list {list_id}: {data.document_id}")
            return True

    async def add_items_batch(
        self,
        list_id: str,
        document_ids: List[str],
        priority: int = 2,
    ) -> int:
        """
        Add multiple documents to a reading list.

        Returns:
            Number of items actually added (skips duplicates)
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)
        added_count = 0

        async with self.pool.acquire() as conn:
            # Get current max position
            next_pos = await conn.fetchval("""
                SELECT COALESCE(MAX(position), 0) + 1
                FROM reading_list_items
                WHERE list_id = $1
            """, list_uuid)

            for doc_id in document_ids:
                try:
                    await conn.execute("""
                        INSERT INTO reading_list_items (list_id, document_id, priority, position)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (list_id, document_id) DO NOTHING
                    """, list_uuid, doc_id, priority, next_pos)
                    added_count += 1
                    next_pos += 1
                except Exception as e:
                    logger.warning(f"Failed to add {doc_id}: {e}")

        logger.info(f"Added {added_count} documents to list {list_id}")
        return added_count

    async def update_item(
        self,
        list_id: str,
        document_id: str,
        data: ReadingListItemUpdate,
    ) -> bool:
        """
        Update an item's priority, notes, or position.

        Returns:
            True if updated, False if not found
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        updates = []
        params = []
        param_idx = 1

        if data.priority is not None:
            updates.append(f"priority = ${param_idx}")
            params.append(data.priority)
            param_idx += 1

        if data.notes is not None:
            updates.append(f"notes = ${param_idx}")
            params.append(data.notes)
            param_idx += 1

        if data.position is not None:
            updates.append(f"position = ${param_idx}")
            params.append(data.position)
            param_idx += 1

        if not updates:
            return True

        params.extend([list_uuid, document_id])

        query = f"""
            UPDATE reading_list_items
            SET {', '.join(updates)}
            WHERE list_id = ${param_idx} AND document_id = ${param_idx + 1}
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result.split()[-1] != '0'

    async def remove_item(self, list_id: str, document_id: str) -> bool:
        """
        Remove a document from a reading list.

        Returns:
            True if removed, False if not found
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM reading_list_items
                WHERE list_id = $1 AND document_id = $2
            """, list_uuid, document_id)

            return result.split()[-1] != '0'

    async def reorder_items(self, list_id: str, document_ids: List[str]) -> None:
        """
        Reorder items in a list based on provided order.

        Args:
            list_id: Reading list ID
            document_ids: List of document IDs in desired order
        """
        list_uuid = list_id if isinstance(list_id, UUID) else UUID(list_id)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for position, doc_id in enumerate(document_ids, start=1):
                    await conn.execute("""
                        UPDATE reading_list_items
                        SET position = $1
                        WHERE list_id = $2 AND document_id = $3
                    """, position, list_uuid, doc_id)

        logger.debug(f"Reordered {len(document_ids)} items in list {list_id}")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert asyncpg Record to dict with JSON-serializable values."""
        if row is None:
            return {}

        result = {}
        for key, value in row.items():
            if isinstance(value, UUID):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value

        return result
