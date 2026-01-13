"""
Knowledge Graph API endpoints.

Provides endpoints for:
- Entity lookup with relationships
- Graph traversal
- Cytoscape.js visualization format
- Graph statistics

REQUIRED DATABASE SCHEMA:
- entities table: id (UUID), name, aliases (text[])
- entity_relations table: source_entity_id, target_entity_id, relation_type, confidence, context_snippet, chunk_ids

Run migration 003_entity_relations.sql to create required tables.
If tables don't exist, endpoints will return 503 with helpful error message.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-graph", tags=["knowledge-graph"])

# Schema verification cache
_schema_verified = False
_relations_table_exists = False


async def _check_graph_schema(pool) -> bool:
    """Check if entity_relations table exists. Cached after first check."""
    global _schema_verified, _relations_table_exists

    if _schema_verified:
        return _relations_table_exists

    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'entity_relations'
                )
            """)
            _relations_table_exists = bool(result)
            _schema_verified = True
            if not _relations_table_exists:
                logger.warning("entity_relations table not found - run migration 003_entity_relations.sql")
            return _relations_table_exists
    except Exception as e:
        logger.error(f"Error checking graph schema: {e}")
        return False


def _raise_schema_error():
    """Raise HTTP 503 for missing schema."""
    raise HTTPException(
        status_code=503,
        detail="Knowledge graph features require the 'entity_relations' table. Run migration 003_entity_relations.sql"
    )


# =============================================================================
# Models
# =============================================================================

class EntityRelation(BaseModel):
    target: str
    relation_type: str
    confidence: float
    context_snippet: Optional[str] = None


class EntityResponse(BaseModel):
    name: str
    normalized: str
    aliases: List[str] = []
    relations: List[EntityRelation] = []
    relation_count: int = 0


class TraversalRequest(BaseModel):
    entities: List[str]
    hop_limit: int = 2
    relation_types: Optional[List[str]] = None
    min_confidence: float = 0.0


class EdgeData(BaseModel):
    source: str
    target: str
    relation: str
    confidence: float
    context: Optional[str] = None
    hop: int
    is_negated: bool = False
    negation_cue: Optional[str] = None
    extraction_method: Optional[str] = None


class TraversalResponse(BaseModel):
    edges: List[EdgeData]
    entities: List[str]
    stats: dict


class GraphStats(BaseModel):
    total_entities: int
    total_relations: int
    relation_types: int
    types: List[str]


class CytoscapeNode(BaseModel):
    data: dict


class CytoscapeEdge(BaseModel):
    data: dict


class CytoscapeElements(BaseModel):
    nodes: List[CytoscapeNode]
    edges: List[CytoscapeEdge]


class VisualizationResponse(BaseModel):
    elements: CytoscapeElements
    center: str


# =============================================================================
# Dependencies
# =============================================================================

from src.api.dependencies import get_container, ServiceContainer


async def get_db_pool(container: ServiceContainer = Depends(get_container)):
    """Get database pool from ServiceContainer."""
    if not container.database:
        raise HTTPException(500, "Database not initialized")
    return container.database.pool


async def get_graph_context():
    """Get GraphRAGContext or create normalizer fallback."""
    try:
        from src.core.relation_extractor import NeuroRelationExtractor
        return NeuroRelationExtractor()
    except ImportError:
        # Return a simple fallback
        class SimpleNormalizer:
            def normalize_entity(self, name: str) -> str:
                return name.lower().strip()
        return SimpleNormalizer()


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/entity/{name}", response_model=EntityResponse)
async def get_entity(
    name: str,
    db_pool=Depends(get_db_pool),
    graph_ctx=Depends(get_graph_context),
):
    """
    Get entity with its relationships.

    Returns the entity's normalized name, aliases, and all outgoing relations.
    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    # Normalize entity name
    normalized = name.lower().strip()
    if hasattr(graph_ctx, 'normalize_entity'):
        normalized = graph_ctx.normalize_entity(name)

    async with db_pool.acquire() as conn:
        # Get entity (aliases column doesn't exist in current schema)
        entity_row = await conn.fetchrow("""
            SELECT id, name FROM entities WHERE name = $1
        """, normalized)

        if not entity_row:
            # Try case-insensitive search
            entity_row = await conn.fetchrow("""
                SELECT id, name FROM entities WHERE LOWER(name) = LOWER($1)
            """, name)

        if not entity_row:
            raise HTTPException(404, f"Entity not found: {name}")

        # Get relations
        relations = await conn.fetch("""
            SELECT
                e2.name as target,
                er.relation_type,
                er.confidence,
                er.context_snippet
            FROM entity_relations er
            JOIN entities e2 ON er.target_entity_id = e2.id
            WHERE er.source_entity_id = $1
            ORDER BY er.confidence DESC
        """, entity_row["id"])

        return EntityResponse(
            name=entity_row["name"],
            normalized=normalized,
            aliases=[],  # aliases column not in current schema
            relations=[
                EntityRelation(
                    target=r["target"],
                    relation_type=r["relation_type"],
                    confidence=r["confidence"],
                    context_snippet=r["context_snippet"],
                )
                for r in relations
            ],
            relation_count=len(relations),
        )


@router.post("/traverse", response_model=TraversalResponse)
async def traverse_graph(
    request: TraversalRequest,
    db_pool=Depends(get_db_pool),
    graph_ctx=Depends(get_graph_context),
):
    """
    Traverse graph from seed entities.

    Performs breadth-first traversal up to hop_limit depth.
    Optionally filter by relation types and minimum confidence.
    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    # Normalize entities
    entities = []
    for e in request.entities:
        if hasattr(graph_ctx, 'normalize_entity'):
            entities.append(graph_ctx.normalize_entity(e))
        else:
            entities.append(e.lower().strip())

    edges = []
    current_layer = set(entities)
    visited_edges = set()

    async with db_pool.acquire() as conn:
        for hop in range(request.hop_limit):
            if not current_layer:
                break

            # Build query with optional filters
            query = """
                SELECT
                    e1.name as source_name,
                    e2.name as target_name,
                    er.relation_type,
                    er.confidence,
                    er.context_snippet,
                    COALESCE(er.is_negated, FALSE) as is_negated,
                    er.negation_cue,
                    er.extraction_method
                FROM entity_relations er
                JOIN entities e1 ON er.source_entity_id = e1.id
                JOIN entities e2 ON er.target_entity_id = e2.id
                WHERE e1.name = ANY($1)
                AND er.confidence >= $2
            """
            params = [list(current_layer), request.min_confidence]

            if request.relation_types:
                query += " AND er.relation_type = ANY($3)"
                params.append(request.relation_types)

            rows = await conn.fetch(query, *params)

            next_layer = set()

            for row in rows:
                edge_key = (row['source_name'], row['target_name'], row['relation_type'])

                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                edges.append(EdgeData(
                    source=row['source_name'],
                    target=row['target_name'],
                    relation=row['relation_type'],
                    confidence=row['confidence'],
                    context=row['context_snippet'],
                    hop=hop + 1,
                    is_negated=row['is_negated'],
                    negation_cue=row['negation_cue'],
                    extraction_method=row['extraction_method'],
                ))

                if row['target_name'] not in entities:
                    next_layer.add(row['target_name'])

            current_layer = next_layer

    # Collect all entities
    all_entities = set(entities)
    for e in edges:
        all_entities.add(e.source)
        all_entities.add(e.target)

    return TraversalResponse(
        edges=edges,
        entities=list(all_entities),
        stats={
            "total_edges": len(edges),
            "total_entities": len(all_entities),
            "max_hop": max((e.hop for e in edges), default=0),
            "seed_entities": len(entities),
        },
    )


@router.get("/visualization/{name}", response_model=VisualizationResponse)
async def get_visualization_data(
    name: str,
    hop_limit: int = Query(default=1, le=3, ge=1),
    db_pool=Depends(get_db_pool),
    graph_ctx=Depends(get_graph_context),
):
    """
    Get graph data in Cytoscape.js format.

    Returns nodes and edges formatted for Cytoscape.js visualization.
    The center node is marked with type="center".
    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first (traverse_graph also checks, but fail fast here)
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    # Normalize
    normalized = name.lower().strip()
    if hasattr(graph_ctx, 'normalize_entity'):
        normalized = graph_ctx.normalize_entity(name)

    # Use traverse logic
    request = TraversalRequest(entities=[normalized], hop_limit=hop_limit)
    traversal = await traverse_graph(request, db_pool, graph_ctx)

    # Build Cytoscape elements
    nodes = {}
    cyto_edges = []

    # Add center node
    nodes[normalized] = CytoscapeNode(
        data={
            "id": normalized,
            "label": normalized,
            "type": "center",
        }
    )

    for edge in traversal.edges:
        # Add source node
        if edge.source not in nodes:
            nodes[edge.source] = CytoscapeNode(
                data={
                    "id": edge.source,
                    "label": edge.source,
                    "type": "entity",
                    "hop": edge.hop,
                }
            )

        # Add target node
        if edge.target not in nodes:
            nodes[edge.target] = CytoscapeNode(
                data={
                    "id": edge.target,
                    "label": edge.target,
                    "type": "entity",
                    "hop": edge.hop,
                }
            )

        # Add edge
        edge_id = f"{edge.source}-{edge.relation}-{edge.target}"
        cyto_edges.append(CytoscapeEdge(
            data={
                "id": edge_id,
                "source": edge.source,
                "target": edge.target,
                "label": edge.relation.replace("_", " "),
                "confidence": edge.confidence,
                "relation_type": edge.relation,
                "is_negated": edge.is_negated,
                "negation_cue": edge.negation_cue,
                "extraction_method": edge.extraction_method,
            }
        ))

    return VisualizationResponse(
        elements=CytoscapeElements(
            nodes=list(nodes.values()),
            edges=cyto_edges,
        ),
        center=normalized,
    )


@router.get("/stats", response_model=GraphStats)
async def get_graph_stats(db_pool=Depends(get_db_pool)):
    """
    Get knowledge graph statistics.

    Returns counts of entities, relations, and unique relation types.
    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM entities) as total_entities,
                (SELECT COUNT(*) FROM entity_relations) as total_relations,
                (SELECT COUNT(DISTINCT relation_type) FROM entity_relations) as relation_types,
                (SELECT array_agg(DISTINCT relation_type) FROM entity_relations) as types
        """)

        return GraphStats(
            total_entities=stats["total_entities"] or 0,
            total_relations=stats["total_relations"] or 0,
            relation_types=stats["relation_types"] or 0,
            types=stats["types"] or [],
        )


@router.get("/search")
async def search_entities(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, le=50),
    db_pool=Depends(get_db_pool),
):
    """
    Search entities by name.

    Supports prefix matching and returns entities with relation counts.
    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    async with db_pool.acquire() as conn:
        # aliases column not in current schema
        rows = await conn.fetch("""
            SELECT
                e.id,
                e.name,
                COUNT(er.id) as relation_count
            FROM entities e
            LEFT JOIN entity_relations er ON e.id = er.source_entity_id
            WHERE e.name ILIKE $1 || '%'
            GROUP BY e.id, e.name
            ORDER BY relation_count DESC, e.name
            LIMIT $2
        """, q, limit)

        return {
            "results": [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "aliases": [],
                    "relation_count": row["relation_count"],
                }
                for row in rows
            ],
            "total": len(rows),
        }


@router.get("/relation-types")
async def get_relation_types(db_pool=Depends(get_db_pool)):
    """
    Get all unique relation types with counts.

    Requires 'entity_relations' table (run migration 003_entity_relations.sql).
    """
    # Check schema first
    if not await _check_graph_schema(db_pool):
        _raise_schema_error()

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                relation_type,
                COUNT(*) as count
            FROM entity_relations
            GROUP BY relation_type
            ORDER BY count DESC
        """)

        return {
            "relation_types": [
                {
                    "type": row["relation_type"],
                    "count": row["count"],
                    "label": row["relation_type"].replace("_", " ").title(),
                }
                for row in rows
            ]
        }
