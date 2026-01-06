"""
NeuroSynth - Procedure-Centric Library API
==========================================

FastAPI routes for procedure-centric library browsing.
The "Flight Plan" experience for surgical preparation.

Mount at: /api/v1/procedures
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/procedures", tags=["Procedures"])


# =============================================================================
# Models
# =============================================================================

class SurgicalPhase(str, Enum):
    """Surgical phase enum matching database."""
    PLANNING = "PLANNING"
    POSITIONING = "POSITIONING"
    EXPOSURE = "EXPOSURE"
    INTRADURAL = "INTRADURAL"
    CLOSURE = "CLOSURE"
    POSTOPERATIVE = "POSTOPERATIVE"


class ContentType(str, Enum):
    """Content classification types."""
    ANATOMY = "anatomy"
    TECHNIQUE = "technique"
    COMPLICATION = "complication"
    EVIDENCE = "evidence"


class ProcedureSummary(BaseModel):
    """Summary of a procedure for listing."""
    id: int
    slug: str
    name: str
    specialty: str
    complexity_level: Optional[int] = None
    parent_id: Optional[int] = None
    parent_name: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    has_steps: bool = False


class ProcedureStep(BaseModel):
    """A step within a procedure."""
    id: int
    step_order: int
    phase: SurgicalPhase
    name: str
    description: Optional[str] = None
    criticality_score: Optional[int] = None
    danger_structures: List[str] = []


class ProcedureDetail(BaseModel):
    """Full procedure details with steps."""
    id: int
    slug: str
    name: str
    description: Optional[str] = None
    specialty: str
    acgme_category: Optional[str] = None
    complexity_level: Optional[int] = None
    anatomy_tags: List[str] = []
    pathology_tags: List[str] = []
    approach_tags: List[str] = []
    steps: List[ProcedureStep] = []
    content_stats: Dict[str, int] = {}


class ChunkContent(BaseModel):
    """A chunk of content linked to a procedure."""
    chunk_id: UUID
    content: str
    page_number: Optional[int] = None
    document_id: UUID
    document_title: str
    authority_score: Optional[float] = None
    relevance_score: float
    confidence: float
    content_type: str
    surgical_phase: Optional[str] = None
    is_pearl: bool = False
    is_pitfall: bool = False
    is_critical: bool = False


class PhaseContent(BaseModel):
    """Content organized by surgical phase."""
    phase: SurgicalPhase
    step_name: Optional[str] = None
    step_order: Optional[int] = None
    criticality_score: Optional[int] = None
    danger_structures: List[str] = []
    chunks: List[ChunkContent] = []
    pearls_count: int = 0
    pitfalls_count: int = 0


class ProcedureContentResponse(BaseModel):
    """Organized content for a procedure."""
    procedure: ProcedureSummary
    phases: List[PhaseContent]
    total_chunks: int
    total_pearls: int
    total_pitfalls: int


class ImageContent(BaseModel):
    """An image linked to a procedure."""
    image_id: UUID
    file_path: str
    caption: Optional[str] = None
    document_title: str
    relevance_score: float
    image_role: Optional[str] = None
    surgical_phase: Optional[str] = None
    is_hero_image: bool = False


class ClinicalEntity(BaseModel):
    """Clinical entity (pathology, variant) for case prep."""
    id: int
    entity_type: str
    name: str
    slug: str
    typical_location: Optional[str] = None
    procedures: List[ProcedureSummary] = []


class MappingStats(BaseModel):
    """Statistics from procedure mapping."""
    chunks_processed: int
    mappings_created: int
    pearls_found: int
    pitfalls_found: int
    by_content_type: Dict[str, int] = {}
    by_phase: Dict[str, int] = {}


# =============================================================================
# Database Dependency
# =============================================================================

_db_pool = None

async def get_db():
    """Get database connection pool."""
    global _db_pool
    if _db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized. Call init_procedure_routes() first."
        )
    return _db_pool


def init_procedure_routes(db_pool):
    """Initialize routes with database pool."""
    global _db_pool
    _db_pool = db_pool
    logger.info("Procedure routes initialized with database pool")


# =============================================================================
# Procedure Listing Endpoints
# =============================================================================

@router.get("/", response_model=List[ProcedureSummary])
async def list_procedures(
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    level: Optional[int] = Query(None, description="Filter by hierarchy level (1=category, 2=procedure)"),
    parent_id: Optional[int] = Query(None, description="Filter by parent procedure"),
    include_stats: bool = Query(True, description="Include content counts"),
    db=Depends(get_db)
):
    """
    List all procedures in the taxonomy.

    Returns procedures organized by hierarchy with optional content statistics.
    """
    params = []
    where_parts = []
    param_idx = 0

    if specialty:
        param_idx += 1
        params.append(specialty)
        where_parts.append(f"pt.specialty = ${param_idx}")

    if level is not None:
        param_idx += 1
        params.append(level)
        where_parts.append(f"pt.level = ${param_idx}")

    if parent_id is not None:
        param_idx += 1
        params.append(parent_id)
        where_parts.append(f"pt.parent_id = ${param_idx}")

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    if include_stats:
        query = f"""
            SELECT
                pt.id,
                pt.slug,
                pt.name,
                pt.specialty,
                pt.complexity_level,
                pt.parent_id,
                parent.name as parent_name,
                COALESCE(chunk_stats.chunk_count, 0) as chunk_count,
                COALESCE(img_stats.image_count, 0) as image_count,
                EXISTS(SELECT 1 FROM procedure_steps ps WHERE ps.procedure_id = pt.id) as has_steps
            FROM procedure_taxonomy pt
            LEFT JOIN procedure_taxonomy parent ON parent.id = pt.parent_id
            LEFT JOIN (
                SELECT procedure_id, COUNT(*) as chunk_count
                FROM chunk_procedure_relevance
                GROUP BY procedure_id
            ) chunk_stats ON chunk_stats.procedure_id = pt.id
            LEFT JOIN (
                SELECT procedure_id, COUNT(*) as image_count
                FROM image_procedure_relevance
                GROUP BY procedure_id
            ) img_stats ON img_stats.procedure_id = pt.id
            {where_clause}
            ORDER BY pt.level, pt.specialty, pt.name
        """
    else:
        query = f"""
            SELECT
                pt.id,
                pt.slug,
                pt.name,
                pt.specialty,
                pt.complexity_level,
                pt.parent_id,
                parent.name as parent_name,
                0 as chunk_count,
                0 as image_count,
                FALSE as has_steps
            FROM procedure_taxonomy pt
            LEFT JOIN procedure_taxonomy parent ON parent.id = pt.parent_id
            {where_clause}
            ORDER BY pt.level, pt.specialty, pt.name
        """

    rows = await db.fetch(query, *params)

    return [
        ProcedureSummary(
            id=row['id'],
            slug=row['slug'],
            name=row['name'],
            specialty=row['specialty'],
            complexity_level=row['complexity_level'],
            parent_id=row['parent_id'],
            parent_name=row['parent_name'],
            chunk_count=row['chunk_count'],
            image_count=row['image_count'],
            has_steps=row['has_steps'],
        )
        for row in rows
    ]


# =============================================================================
# Static Path Routes (MUST be before /{slug} to avoid path matching issues)
# =============================================================================

@router.get("/mapping-stats", response_model=Dict[str, Any])
async def get_mapping_stats(
    db=Depends(get_db)
):
    """
    Get statistics about procedure-chunk mappings.
    """
    stats_query = """
        SELECT
            (SELECT COUNT(*) FROM chunk_procedure_relevance) as total_mappings,
            (SELECT COUNT(DISTINCT chunk_id) FROM chunk_procedure_relevance) as unique_chunks,
            (SELECT COUNT(DISTINCT procedure_id) FROM chunk_procedure_relevance) as procedures_with_content,
            (SELECT COUNT(*) FROM chunk_procedure_relevance WHERE is_pearl = TRUE) as total_pearls,
            (SELECT COUNT(*) FROM chunk_procedure_relevance WHERE is_pitfall = TRUE) as total_pitfalls,
            (SELECT COUNT(*) FROM chunk_procedure_relevance WHERE is_critical = TRUE) as critical_content
    """
    row = await db.fetchrow(stats_query)

    # Content by type
    type_query = """
        SELECT content_type, COUNT(*) as count
        FROM chunk_procedure_relevance
        GROUP BY content_type
        ORDER BY count DESC
    """
    type_rows = await db.fetch(type_query)

    # Content by phase
    phase_query = """
        SELECT surgical_phase, COUNT(*) as count
        FROM chunk_procedure_relevance
        WHERE surgical_phase IS NOT NULL
        GROUP BY surgical_phase
        ORDER BY count DESC
    """
    phase_rows = await db.fetch(phase_query)

    return {
        "total_mappings": row['total_mappings'],
        "unique_chunks": row['unique_chunks'],
        "procedures_with_content": row['procedures_with_content'],
        "total_pearls": row['total_pearls'],
        "total_pitfalls": row['total_pitfalls'],
        "critical_content": row['critical_content'],
        "by_content_type": {r['content_type']: r['count'] for r in type_rows},
        "by_phase": {r['surgical_phase']: r['count'] for r in phase_rows if r['surgical_phase']},
    }


# =============================================================================
# Dynamic Procedure Routes
# =============================================================================

@router.get("/{slug}", response_model=ProcedureDetail)
async def get_procedure(
    slug: str,
    db=Depends(get_db)
):
    """
    Get detailed procedure information including steps.

    Returns full procedure details with surgical steps organized by phase.
    """
    # Get procedure
    proc_query = """
        SELECT
            id, slug, name, description, specialty, acgme_category,
            complexity_level, anatomy_tags, pathology_tags, approach_tags
        FROM procedure_taxonomy
        WHERE slug = $1
    """
    proc_row = await db.fetchrow(proc_query, slug)

    if not proc_row:
        raise HTTPException(status_code=404, detail=f"Procedure '{slug}' not found")

    # Get steps
    steps_query = """
        SELECT id, step_order, phase, name, description,
               criticality_score, danger_structures
        FROM procedure_steps
        WHERE procedure_id = $1
        ORDER BY step_order
    """
    step_rows = await db.fetch(steps_query, proc_row['id'])

    steps = [
        ProcedureStep(
            id=row['id'],
            step_order=row['step_order'],
            phase=SurgicalPhase(row['phase']),
            name=row['name'],
            description=row['description'],
            criticality_score=row['criticality_score'],
            danger_structures=row['danger_structures'] or [],
        )
        for row in step_rows
    ]

    # Get content stats
    stats_query = """
        SELECT content_type, COUNT(*) as count
        FROM chunk_procedure_relevance
        WHERE procedure_id = $1
        GROUP BY content_type
    """
    stats_rows = await db.fetch(stats_query, proc_row['id'])
    content_stats = {row['content_type']: row['count'] for row in stats_rows}

    return ProcedureDetail(
        id=proc_row['id'],
        slug=proc_row['slug'],
        name=proc_row['name'],
        description=proc_row['description'],
        specialty=proc_row['specialty'],
        acgme_category=proc_row['acgme_category'],
        complexity_level=proc_row['complexity_level'],
        anatomy_tags=proc_row['anatomy_tags'] or [],
        pathology_tags=proc_row['pathology_tags'] or [],
        approach_tags=proc_row['approach_tags'] or [],
        steps=steps,
        content_stats=content_stats,
    )


# =============================================================================
# Procedure Content Endpoints (Flight Plan)
# =============================================================================

@router.get("/{slug}/content", response_model=ProcedureContentResponse)
async def get_procedure_content(
    slug: str,
    phase: Optional[SurgicalPhase] = Query(None, description="Filter by surgical phase"),
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    pearls_only: bool = Query(False, description="Only show pearls"),
    pitfalls_only: bool = Query(False, description="Only show pitfalls"),
    critical_only: bool = Query(False, description="Only show critical content"),
    min_relevance: float = Query(0.3, ge=0, le=1, description="Minimum relevance score"),
    limit_per_phase: int = Query(20, ge=1, le=100, description="Max chunks per phase"),
    db=Depends(get_db)
):
    """
    Get organized content for a procedure (The "Flight Plan").

    Returns content organized by surgical phase, with steps and danger zones.
    This is the main endpoint for surgical preparation.

    Content is sorted by:
    1. Critical/pitfall content first
    2. Relevance score
    3. Source authority
    """
    # Get procedure ID
    proc_query = "SELECT id, slug, name, specialty, complexity_level FROM procedure_taxonomy WHERE slug = $1"
    proc_row = await db.fetchrow(proc_query, slug)

    if not proc_row:
        raise HTTPException(status_code=404, detail=f"Procedure '{slug}' not found")

    proc_id = proc_row['id']

    # Build content query
    params = [proc_id, min_relevance, limit_per_phase]
    where_parts = ["cpr.procedure_id = $1", "cpr.relevance_score >= $2"]
    param_idx = 3

    if phase:
        param_idx += 1
        params.insert(-1, phase.value)
        where_parts.append(f"cpr.surgical_phase = ${param_idx}::surgical_phase_enum")

    if content_type:
        param_idx += 1
        params.insert(-1, content_type.value)
        where_parts.append(f"cpr.content_type = ${param_idx}")

    if pearls_only:
        where_parts.append("cpr.is_pearl = TRUE")

    if pitfalls_only:
        where_parts.append("cpr.is_pitfall = TRUE")

    if critical_only:
        where_parts.append("cpr.is_critical = TRUE")

    content_query = f"""
        WITH ranked_content AS (
            SELECT
                c.id as chunk_id,
                c.content,
                c.page_number,
                c.document_id,
                d.title as document_title,
                d.authority_score,
                cpr.relevance_score,
                cpr.confidence,
                cpr.content_type,
                cpr.surgical_phase,
                cpr.is_pearl,
                cpr.is_pitfall,
                cpr.is_critical,
                ps.id as step_id,
                ps.name as step_name,
                ps.step_order,
                ps.criticality_score as step_criticality,
                ps.danger_structures,
                ROW_NUMBER() OVER (
                    PARTITION BY cpr.surgical_phase
                    ORDER BY cpr.is_critical DESC, cpr.is_pitfall DESC,
                             cpr.relevance_score DESC, d.authority_score DESC NULLS LAST
                ) as rn
            FROM chunk_procedure_relevance cpr
            JOIN chunks c ON c.id = cpr.chunk_id
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN procedure_steps ps ON ps.id = cpr.step_id
            WHERE {' AND '.join(where_parts)}
        )
        SELECT * FROM ranked_content
        WHERE rn <= ${len(params)}
        ORDER BY
            CASE surgical_phase
                WHEN 'PLANNING' THEN 1
                WHEN 'POSITIONING' THEN 2
                WHEN 'EXPOSURE' THEN 3
                WHEN 'INTRADURAL' THEN 4
                WHEN 'CLOSURE' THEN 5
                WHEN 'POSTOPERATIVE' THEN 6
            END,
            step_order NULLS LAST,
            is_critical DESC,
            is_pitfall DESC,
            relevance_score DESC
    """

    rows = await db.fetch(content_query, *params)

    # Organize by phase
    phases_dict: Dict[str, PhaseContent] = {}
    total_pearls = 0
    total_pitfalls = 0

    for row in rows:
        phase_key = row['surgical_phase'] or 'UNKNOWN'

        if phase_key not in phases_dict:
            phases_dict[phase_key] = PhaseContent(
                phase=SurgicalPhase(phase_key) if phase_key in SurgicalPhase.__members__ else SurgicalPhase.PLANNING,
                step_name=row['step_name'],
                step_order=row['step_order'],
                criticality_score=row['step_criticality'],
                danger_structures=row['danger_structures'] or [],
                chunks=[],
                pearls_count=0,
                pitfalls_count=0,
            )

        chunk = ChunkContent(
            chunk_id=row['chunk_id'],
            content=row['content'],
            page_number=row['page_number'],
            document_id=row['document_id'],
            document_title=row['document_title'],
            authority_score=row['authority_score'],
            relevance_score=row['relevance_score'],
            confidence=row['confidence'],
            content_type=row['content_type'],
            surgical_phase=row['surgical_phase'],
            is_pearl=row['is_pearl'],
            is_pitfall=row['is_pitfall'],
            is_critical=row['is_critical'],
        )

        phases_dict[phase_key].chunks.append(chunk)

        if row['is_pearl']:
            phases_dict[phase_key].pearls_count += 1
            total_pearls += 1
        if row['is_pitfall']:
            phases_dict[phase_key].pitfalls_count += 1
            total_pitfalls += 1

    # Sort phases in surgical order
    phase_order = ['PLANNING', 'POSITIONING', 'EXPOSURE', 'INTRADURAL', 'CLOSURE', 'POSTOPERATIVE']
    phases = [
        phases_dict[p] for p in phase_order if p in phases_dict
    ]

    return ProcedureContentResponse(
        procedure=ProcedureSummary(
            id=proc_row['id'],
            slug=proc_row['slug'],
            name=proc_row['name'],
            specialty=proc_row['specialty'],
            complexity_level=proc_row['complexity_level'],
            chunk_count=sum(len(p.chunks) for p in phases),
        ),
        phases=phases,
        total_chunks=sum(len(p.chunks) for p in phases),
        total_pearls=total_pearls,
        total_pitfalls=total_pitfalls,
    )


@router.get("/{slug}/images", response_model=List[ImageContent])
async def get_procedure_images(
    slug: str,
    phase: Optional[SurgicalPhase] = Query(None, description="Filter by surgical phase"),
    hero_only: bool = Query(False, description="Only hero images"),
    limit: int = Query(50, ge=1, le=200, description="Max images"),
    db=Depends(get_db)
):
    """
    Get images linked to a procedure.

    Returns images sorted by display priority (hero images first).
    """
    # Get procedure ID
    proc_row = await db.fetchrow(
        "SELECT id FROM procedure_taxonomy WHERE slug = $1", slug
    )
    if not proc_row:
        raise HTTPException(status_code=404, detail=f"Procedure '{slug}' not found")

    params = [proc_row['id'], limit]
    where_parts = ["ipr.procedure_id = $1"]

    if phase:
        params.insert(-1, phase.value)
        where_parts.append(f"ipr.surgical_phase = ${len(params)}::surgical_phase_enum")

    if hero_only:
        where_parts.append("ipr.is_hero_image = TRUE")

    query = f"""
        SELECT
            i.id as image_id,
            i.file_path,
            i.caption,
            d.title as document_title,
            ipr.relevance_score,
            ipr.image_role,
            ipr.surgical_phase,
            ipr.is_hero_image
        FROM image_procedure_relevance ipr
        JOIN images i ON i.id = ipr.image_id
        JOIN documents d ON d.id = i.document_id
        WHERE {' AND '.join(where_parts)}
        ORDER BY ipr.is_hero_image DESC, ipr.display_priority DESC, ipr.relevance_score DESC
        LIMIT ${len(params)}
    """

    rows = await db.fetch(query, *params)

    return [
        ImageContent(
            image_id=row['image_id'],
            file_path=row['file_path'],
            caption=row['caption'],
            document_title=row['document_title'],
            relevance_score=row['relevance_score'],
            image_role=row['image_role'],
            surgical_phase=row['surgical_phase'],
            is_hero_image=row['is_hero_image'],
        )
        for row in rows
    ]


# =============================================================================
# Clinical Entity Endpoints (Case Prep Mode)
# =============================================================================

@router.get("/clinical-entities", response_model=List[ClinicalEntity])
async def list_clinical_entities(
    entity_type: Optional[str] = Query(None, description="Filter by type (pathology, anatomy_variant)"),
    search: Optional[str] = Query(None, description="Search in name/synonyms"),
    db=Depends(get_db)
):
    """
    List clinical entities for case prep mode.

    Clinical entities represent real-world scenarios (pathologies, anatomical variants)
    that map to surgical procedures.
    """
    params = []
    where_parts = []

    if entity_type:
        params.append(entity_type)
        where_parts.append(f"ce.entity_type = ${len(params)}")

    if search:
        params.append(f"%{search}%")
        where_parts.append(f"(ce.name ILIKE ${len(params)} OR EXISTS (SELECT 1 FROM unnest(ce.synonyms) s WHERE s ILIKE ${len(params)}))")

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    query = f"""
        SELECT
            ce.id,
            ce.entity_type,
            ce.name,
            ce.slug,
            ce.typical_location
        FROM clinical_entities ce
        {where_clause}
        ORDER BY ce.entity_type, ce.name
    """

    rows = await db.fetch(query, *params)

    results = []
    for row in rows:
        # Get linked procedures
        proc_query = """
            SELECT pt.id, pt.slug, pt.name, pt.specialty, pt.complexity_level, cep.is_primary
            FROM clinical_entity_procedures cep
            JOIN procedure_taxonomy pt ON pt.id = cep.procedure_id
            WHERE cep.clinical_entity_id = $1
            ORDER BY cep.is_primary DESC, cep.relevance_score DESC
        """
        proc_rows = await db.fetch(proc_query, row['id'])

        procedures = [
            ProcedureSummary(
                id=pr['id'],
                slug=pr['slug'],
                name=pr['name'],
                specialty=pr['specialty'],
                complexity_level=pr['complexity_level'],
            )
            for pr in proc_rows
        ]

        results.append(ClinicalEntity(
            id=row['id'],
            entity_type=row['entity_type'],
            name=row['name'],
            slug=row['slug'],
            typical_location=row['typical_location'],
            procedures=procedures,
        ))

    return results


@router.get("/for-pathology/{pathology_slug}", response_model=List[ProcedureSummary])
async def get_procedures_for_pathology(
    pathology_slug: str,
    db=Depends(get_db)
):
    """
    Get recommended procedures for a clinical pathology.

    Use this for case prep: "I have an MCA aneurysm, what procedures should I review?"
    """
    query = """
        SELECT
            pt.id,
            pt.slug,
            pt.name,
            pt.specialty,
            pt.complexity_level,
            cep.is_primary
        FROM clinical_entities ce
        JOIN clinical_entity_procedures cep ON ce.id = cep.clinical_entity_id
        JOIN procedure_taxonomy pt ON pt.id = cep.procedure_id
        WHERE ce.slug = $1
        ORDER BY cep.is_primary DESC, cep.relevance_score DESC
    """

    rows = await db.fetch(query, pathology_slug)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No procedures found for pathology '{pathology_slug}'"
        )

    return [
        ProcedureSummary(
            id=row['id'],
            slug=row['slug'],
            name=row['name'],
            specialty=row['specialty'],
            complexity_level=row['complexity_level'],
        )
        for row in rows
    ]


# =============================================================================
# Mapping Management Endpoints
# =============================================================================

@router.post("/map-chunks", response_model=MappingStats)
async def trigger_chunk_mapping(
    document_ids: Optional[List[UUID]] = Query(None, description="Limit to specific documents"),
    db=Depends(get_db)
):
    """
    Trigger chunk-to-procedure mapping.

    This populates the chunk_procedure_relevance table by analyzing
    all chunks and linking them to relevant procedures.
    """
    from .procedure_mapper import ProcedureMapper, MapperConfig

    config = MapperConfig(
        min_relevance_score=0.3,
        min_confidence=0.4,
    )

    mapper = ProcedureMapper(db, config)
    stats = await mapper.map_all_chunks(document_ids=document_ids)

    return MappingStats(
        chunks_processed=stats.get('chunks_processed', 0),
        mappings_created=stats.get('mappings_created', 0),
        pearls_found=stats.get('pearls_found', 0),
        pitfalls_found=stats.get('pitfalls_found', 0),
        by_content_type=stats.get('by_content_type', {}),
        by_phase=stats.get('by_phase', {}),
    )
