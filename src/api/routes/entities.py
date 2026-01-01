"""
NeuroSynth Unified - Entity Routes
===================================

Medical entity browser API endpoints.

REQUIRED DATABASE SCHEMA:
- entities table: id (UUID), cui, name, semantic_type, tui, chunk_count, image_count
- entity_chunk_links: entity_id, chunk_id (optional - graceful fallback if missing)

Run migration 003_entity_relations.sql to create required tables.
If tables don't exist, endpoints will return 503 with helpful error message.
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/entities", tags=["Entities"])

# Schema verification cache
_schema_verified = False
_entities_table_exists = False


async def _check_entities_schema(database) -> bool:
    """Check if entities table exists. Cached after first check."""
    global _schema_verified, _entities_table_exists

    if _schema_verified:
        return _entities_table_exists

    try:
        result = await database.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'entities'
            )
        """)
        _entities_table_exists = bool(result)
        _schema_verified = True
        if not _entities_table_exists:
            logger.warning("entities table not found - run migration 003_entity_relations.sql")
        return _entities_table_exists
    except Exception as e:
        logger.error(f"Error checking entities schema: {e}")
        return False


def _raise_schema_error():
    """Raise HTTP 503 for missing schema."""
    raise HTTPException(
        status_code=503,
        detail="Entity features require the 'entities' table. Run migration 003_entity_relations.sql"
    )


# =============================================================================
# MODELS
# =============================================================================

class EntityBase(BaseModel):
    """Base entity model."""
    cui: str = Field(..., description="Entity ID (UUID as string)")
    name: str = Field(..., description="Entity text/name")
    semantic_type: str = Field(..., description="Entity category")
    semantic_type_name: Optional[str] = None
    definition: Optional[str] = None


class EntityDetail(EntityBase):
    """Extended entity with related data."""
    aliases: List[str] = []
    occurrence_count: int = 0
    related_chunks: List[Dict[str, Any]] = []
    related_entities: List[Dict[str, Any]] = []


class EntityListResponse(BaseModel):
    """Paginated entity list."""
    entities: List[EntityBase]
    total: int
    page: int
    page_size: int
    has_more: bool


class EntitySearchResult(BaseModel):
    """Search result item."""
    cui: str
    name: str
    semantic_type: str
    score: float
    occurrence_count: int


class GraphNode(BaseModel):
    """Knowledge graph node."""
    id: str
    label: str
    type: str
    size: int = 10


class GraphEdge(BaseModel):
    """Knowledge graph edge."""
    source: str
    target: str
    relationship: str
    weight: float = 1.0


class KnowledgeGraph(BaseModel):
    """Knowledge graph response."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_cui: str


# =============================================================================
# ROUTES
# =============================================================================

@router.get(
    "",
    response_model=EntityListResponse,
    summary="List entities",
    description="List all medical entities with pagination and filtering"
)
async def list_entities(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=10, le=200, description="Items per page"),
    semantic_type: Optional[str] = Query(None, description="Filter by category"),
    min_occurrences: int = Query(0, ge=0, description="Minimum mention count"),
    sort_by: str = Query("mention_count", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    database = Depends(get_database)
):
    """
    List all medical entities with pagination and filtering.

    Requires 'entities' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    try:
        offset = (page - 1) * page_size

        # Build WHERE clauses with parameterized queries (SQL injection safe)
        params = []
        where_parts = []

        if semantic_type:
            params.append(semantic_type)
            where_parts.append(f"semantic_type = ${len(params)}")
        if min_occurrences > 0:
            params.append(min_occurrences)
            where_parts.append(f"chunk_count >= ${len(params)}")

        where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        order_dir = "DESC" if sort_order == "desc" else "ASC"

        # Map frontend sort fields to actual columns (whitelist validation)
        sort_map = {
            "occurrence_count": "chunk_count",
            "mention_count": "chunk_count",
            "chunk_count": "chunk_count",
            "name": "name",
            "cui": "cui",
            "semantic_type": "semantic_type"
        }
        actual_sort = sort_map.get(sort_by, "chunk_count")

        # Add pagination parameters
        params.append(page_size)
        limit_param = len(params)
        params.append(offset)
        offset_param = len(params)

        # Get entities - use actual schema column names
        query = f"""
            SELECT cui, name, semantic_type,
                   semantic_type as semantic_type_name, NULL as definition, chunk_count
            FROM entities
            {where_sql}
            ORDER BY {actual_sort} {order_dir}
            LIMIT ${limit_param} OFFSET ${offset_param}
        """

        rows = await database.fetch(query, *params)

        entities = [
            EntityBase(
                cui=row['cui'],
                name=row['name'],
                semantic_type=row['semantic_type'],
                semantic_type_name=row.get('semantic_type_name'),
                definition=row.get('definition')
            )
            for row in rows
        ]

        # Get total count (use only filter params, not pagination params)
        filter_params = params[:-2] if len(params) >= 2 else []
        count_query = f"SELECT COUNT(*) FROM entities {where_sql}"
        total = await database.fetchval(count_query, *filter_params)

        return EntityListResponse(
            entities=entities,
            total=total or 0,
            page=page,
            page_size=page_size,
            has_more=(page * page_size) < (total or 0)
        )

    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search",
    response_model=List[EntitySearchResult],
    summary="Search entities",
    description="Search entities by name or ID"
)
async def search_entities(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    semantic_type: Optional[str] = Query(None, description="Filter by category"),
    database = Depends(get_database)
):
    """
    Search entities by name or ID.

    Requires 'entities' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    try:
        # Check if searching by CUI pattern or UUID
        is_cui_or_uuid = q.startswith('C') or (len(q) >= 8 and '-' in q)

        if is_cui_or_uuid:
            query = """
                SELECT cui, name, semantic_type,
                       chunk_count, 1.0 as score
                FROM entities
                WHERE cui ILIKE $1 OR id::text ILIKE $1
                LIMIT 1
            """
            rows = await database.fetch(query, f"%{q}%")
        else:
            # Text search with ILIKE (parameterized to prevent SQL injection)
            params = [q, limit]
            type_filter = ""
            if semantic_type:
                params.append(semantic_type)
                type_filter = f"AND semantic_type = ${len(params)}"

            query = f"""
                SELECT cui, name, semantic_type,
                       chunk_count,
                       CASE
                           WHEN LOWER(name) = LOWER($1) THEN 1.0
                           WHEN LOWER(name) LIKE LOWER($1) || '%' THEN 0.8
                           ELSE 0.5
                       END as score
                FROM entities
                WHERE name ILIKE '%' || $1 || '%'
                {type_filter}
                ORDER BY score DESC, chunk_count DESC
                LIMIT $2
            """
            rows = await database.fetch(query, *params)

        return [
            EntitySearchResult(
                cui=row['cui'],
                name=row['name'],
                semantic_type=row['semantic_type'],
                score=float(row['score']),
                occurrence_count=row['chunk_count'] or 0
            )
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/graph",
    response_model=KnowledgeGraph,
    summary="Get entity graph",
    description="Get knowledge graph centered on an entity"
)
async def get_entity_graph(
    cui: str = Query(..., description="Center entity ID"),
    depth: int = Query(1, ge=1, le=3, description="Graph depth"),
    max_nodes: int = Query(50, ge=10, le=200, description="Max nodes"),
    database = Depends(get_database)
):
    """
    Get knowledge graph centered on an entity.

    Requires 'entities' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    try:
        # Get center entity
        center = await database.fetchrow(
            """
            SELECT cui, name, semantic_type, chunk_count
            FROM entities
            WHERE cui = $1 OR name ILIKE $1
            LIMIT 1
            """,
            cui
        )

        if not center:
            raise HTTPException(status_code=404, detail=f"Entity {cui} not found")

        nodes = [
            GraphNode(
                id=center['cui'],
                label=center['name'],
                type=center['semantic_type'],
                size=20
            )
        ]
        edges = []

        # Get related entities (co-occurring in chunks via entity_chunk_links)
        # Note: This may return empty if entity_chunk_links table doesn't exist
        try:
            # Get center entity's UUID
            center_id = await database.fetchval(
                "SELECT id FROM entities WHERE cui = $1",
                center['cui']
            )

            related_rows = await database.fetch(
                """
                SELECT DISTINCT e2.cui, e2.name, e2.semantic_type,
                       COUNT(*) as co_occurrence
                FROM entity_chunk_links ecl1
                JOIN entity_chunk_links ecl2 ON ecl1.chunk_id = ecl2.chunk_id
                JOIN entities e2 ON ecl2.entity_id = e2.id
                WHERE ecl1.entity_id = $1 AND ecl2.entity_id != $1
                GROUP BY e2.cui, e2.name, e2.semantic_type
                ORDER BY co_occurrence DESC
                LIMIT $2
                """,
                center_id, max_nodes - 1
            )
        except Exception:
            # entity_chunk_links table may not exist
            related_rows = []

        for row in related_rows:
            nodes.append(
                GraphNode(
                    id=row['cui'],
                    label=row['name'],
                    type=row['semantic_type'],
                    size=min(10 + row['co_occurrence'], 18)
                )
            )
            edges.append(
                GraphEdge(
                    source=center['cui'],
                    target=row['cui'],
                    relationship="co_occurs_with",
                    weight=float(row['co_occurrence'])
                )
            )

        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            center_cui=center['cui']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building entity graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DELETE OPERATIONS
# =============================================================================

class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete operations."""
    ids: List[str] = Field(..., description="List of entity IDs to delete")


class DeleteResponse(BaseModel):
    """Response for delete operations."""
    deleted: int
    message: str


@router.delete(
    "/all",
    response_model=DeleteResponse,
    summary="Delete all entities",
    description="Delete ALL entities and their chunk links. Use with caution!"
)
async def delete_all_entities(
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
    database = Depends(get_database)
):
    """Delete all entities. Requires confirmation. Needs 'entities' table."""
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to delete all entities"
        )

    try:
        # First delete all entity-chunk links
        await database.execute("DELETE FROM entity_chunk_links")

        # Then delete all entities
        result = await database.execute("DELETE FROM entities")

        # Get count (asyncpg returns status string like "DELETE 170")
        count = int(result.split()[-1]) if result else 0

        logger.info(f"Deleted all entities: {count} total")
        return DeleteResponse(deleted=count, message=f"Deleted all {count} entities")

    except Exception as e:
        logger.error(f"Error deleting all entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Bulk delete entities",
    description="Delete multiple entities by their IDs"
)
async def bulk_delete_entities(
    request: BulkDeleteRequest,
    database = Depends(get_database)
):
    """Delete multiple entities by ID. Needs 'entities' table."""
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    if not request.ids:
        raise HTTPException(status_code=400, detail="No IDs provided")

    try:
        # Convert string IDs to UUIDs for validation
        from uuid import UUID
        try:
            uuids = [str(UUID(id_str)) for id_str in request.ids]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")

        # Delete entity-chunk links first
        await database.execute(
            "DELETE FROM entity_chunk_links WHERE entity_id = ANY($1::uuid[])",
            uuids
        )

        # Delete entities
        result = await database.execute(
            "DELETE FROM entities WHERE id = ANY($1::uuid[])",
            uuids
        )

        count = int(result.split()[-1]) if result else 0

        logger.info(f"Bulk deleted {count} entities")
        return DeleteResponse(deleted=count, message=f"Deleted {count} entities")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{entity_id}",
    response_model=DeleteResponse,
    summary="Delete single entity",
    description="Delete a specific entity by ID"
)
async def delete_entity(
    entity_id: str,
    database = Depends(get_database)
):
    """Delete a single entity by ID. Needs 'entities' table."""
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    try:
        from uuid import UUID
        try:
            uuid_str = str(UUID(entity_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid entity ID format")

        # Check if entity exists
        exists = await database.fetchval(
            "SELECT 1 FROM entities WHERE id = $1::uuid",
            uuid_str
        )

        if not exists:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        # Delete entity-chunk links first
        await database.execute(
            "DELETE FROM entity_chunk_links WHERE entity_id = $1::uuid",
            uuid_str
        )

        # Delete entity
        await database.execute(
            "DELETE FROM entities WHERE id = $1::uuid",
            uuid_str
        )

        logger.info(f"Deleted entity {entity_id}")
        return DeleteResponse(deleted=1, message=f"Deleted entity {entity_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENTITY DETAILS
# =============================================================================

@router.get(
    "/{cui}",
    response_model=EntityDetail,
    summary="Get entity details",
    description="Get detailed information about a specific entity"
)
async def get_entity(
    cui: str,
    database = Depends(get_database)
):
    """
    Get detailed information about a specific entity.

    Requires 'entities' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_entities_schema(database):
        _raise_schema_error()

    try:
        # Get entity - search by CUI or name
        row = await database.fetchrow(
            """
            SELECT id, cui, name, semantic_type,
                   semantic_type as semantic_type_name, chunk_count
            FROM entities
            WHERE cui = $1 OR name ILIKE $1
            LIMIT 1
            """,
            cui
        )

        if not row:
            raise HTTPException(status_code=404, detail=f"Entity {cui} not found")

        entity_uuid = row['id']

        # Get related chunks (where entity appears via CUI in chunks.cuis array)
        try:
            chunks_rows = await database.fetch(
                """
                SELECT c.id, c.content, c.chunk_type,
                       d.title as document_title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE $1 = ANY(c.cuis)
                LIMIT 10
                """,
                row['cui']
            )
        except Exception:
            # Fall back to entity_chunk_links if it exists
            try:
                chunks_rows = await database.fetch(
                    """
                    SELECT c.id, c.content, c.chunk_type,
                           d.title as document_title
                    FROM entity_chunk_links ecl
                    JOIN chunks c ON ecl.chunk_id = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE ecl.entity_id = $1
                    LIMIT 10
                    """,
                    entity_uuid
                )
            except Exception:
                chunks_rows = []

        related_chunks = [
            {
                "chunk_id": str(r['id']),
                "content": r['content'][:200] + "..." if r['content'] and len(r['content']) > 200 else r['content'],
                "summary": None,
                "chunk_type": r.get('chunk_type'),
                "document_title": r.get('document_title')
            }
            for r in chunks_rows
        ]

        # Get related entities (co-occurring in chunks via shared CUIs)
        try:
            # Find entities that co-occur in chunks with this entity's CUI
            related_rows = await database.fetch(
                """
                WITH entity_chunks AS (
                    SELECT DISTINCT c.id as chunk_id
                    FROM chunks c
                    WHERE $1 = ANY(c.cuis)
                )
                SELECT e.cui, e.name, e.semantic_type,
                       COUNT(DISTINCT ec.chunk_id) as co_occurrence
                FROM entity_chunks ec
                JOIN chunks c ON ec.chunk_id = c.id
                CROSS JOIN LATERAL unnest(c.cuis) as co_cui
                JOIN entities e ON e.cui = co_cui
                WHERE e.cui != $1
                GROUP BY e.cui, e.name, e.semantic_type
                ORDER BY co_occurrence DESC
                LIMIT 10
                """,
                row['cui']
            )
        except Exception:
            related_rows = []

        related_entities = [
            {
                "cui": r['cui'],
                "name": r['name'],
                "semantic_type": r['semantic_type'],
                "co_occurrence": r['co_occurrence']
            }
            for r in related_rows
        ]

        return EntityDetail(
            cui=row['cui'],
            name=row['name'],
            semantic_type=row['semantic_type'],
            semantic_type_name=row.get('semantic_type_name'),
            definition=None,
            aliases=[],  # Could populate from metadata if available
            occurrence_count=row['chunk_count'] or 0,
            related_chunks=related_chunks,
            related_entities=related_entities
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity {cui}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
