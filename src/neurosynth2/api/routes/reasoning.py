"""
NeuroSynth 2.0 - Reasoning API Routes
======================================

REST API endpoints for surgical reasoning and simulation.

Endpoints:
- POST /api/v1/reasoning/assess-risk - Assess risks of a surgical action
- POST /api/v1/reasoning/simulate - Simulate complete surgical approach
- GET  /api/v1/reasoning/principles - List clinical principles
- GET  /api/v1/reasoning/corridors - List surgical corridors
- GET  /api/v1/reasoning/entity/{name}/physics - Get entity physics
- GET  /api/v1/reasoning/stats - Get reasoning engine stats

REQUIRED DATABASE SCHEMA:
- anatomical_entities table
- surgical_corridors table
- clinical_principles table

Run migration 009_neurosynth2_physics_schema.sql to create required tables.
"""

import logging
import time
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# NS2 Dependencies
from src.neurosynth2.dependencies import (
    get_ns2_db_pool,
    get_clinical_reasoner,
    get_surgical_simulator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reasoning", tags=["reasoning"])

# Schema verification cache
_schema_verified = False
_reasoning_schema_available = False


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def _check_reasoning_schema() -> bool:
    """Check if reasoning schema tables exist."""
    global _schema_verified, _reasoning_schema_available

    if _schema_verified:
        return _reasoning_schema_available

    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'clinical_principles'
                )
            """)
            _reasoning_schema_available = bool(result)
            _schema_verified = True

            if not _reasoning_schema_available:
                logger.warning(
                    "Reasoning schema not found. "
                    "Run migration 009_neurosynth2_physics_schema.sql to enable reasoning features."
                )
            return _reasoning_schema_available
    except Exception as e:
        logger.error(f"Schema check failed: {e}")
        return False


def _raise_schema_error():
    """Raise HTTP 503 for missing schema."""
    raise HTTPException(
        status_code=503,
        detail=(
            "Reasoning features require the physics schema. "
            "Run migration 009_neurosynth2_physics_schema.sql"
        )
    )


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""
    action: str = Field(
        ...,
        description="Surgical action: retract, dissect, coagulate, sacrifice, mobilize, clip, resect, etc."
    )
    structure: str = Field(
        ...,
        description="Target anatomical structure"
    )
    approach: Optional[str] = Field(
        None,
        description="Surgical approach context (e.g., 'pterional', 'retrosigmoid')"
    )
    patient_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient-specific factors (age, comorbidities, etc.)"
    )
    include_alternatives: bool = Field(
        default=True,
        description="Include alternative approaches in response"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "action": "retract",
                "structure": "temporal_lobe",
                "approach": "pterional",
                "patient_factors": {"age": 65}
            }]
        }
    }


class RiskItem(BaseModel):
    """Individual risk assessment."""
    level: str
    structure: str
    action: str
    principle_id: str
    principle_name: str
    mechanism: str
    mitigation: Optional[str]
    confidence: float
    evidence_level: str


class RiskAssessmentResponse(BaseModel):
    """Risk assessment result."""
    structure: str
    action: str
    risks: List[RiskItem]
    highest_risk_level: str
    total_risks: int
    recommendations: List[str]
    alternatives: List[str]
    data_gaps: List[str]
    assessment_time_ms: int


class SimulationRequest(BaseModel):
    """Request for surgical simulation."""
    approach: str = Field(
        ...,
        description="Surgical approach name (e.g., 'pterional', 'retrosigmoid')"
    )
    target: str = Field(
        ...,
        description="Target pathology/structure"
    )
    patient_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient-specific factors"
    )
    max_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum simulation steps"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "approach": "pterional",
                "target": "MCA_aneurysm",
                "patient_factors": {"age": 55, "hypertension": True}
            }]
        }
    }


class SimulationStepResponse(BaseModel):
    """A single simulation step."""
    step: int
    action: str
    structure: str
    highest_risk: Optional[str]
    outcome: str
    complications: List[Dict[str, Any]]
    is_decision_point: bool
    alternatives: List[str]


class PatientStateResponse(BaseModel):
    """Patient state snapshot."""
    condition: str
    visibility: float
    brain_relaxation: float
    active_bleeding: bool
    bleeding_rate: str
    brain_swelling: bool
    monitoring: Dict[str, float]
    cumulative_risk_score: float
    complications: List[Dict[str, Any]]


class SimulationResponse(BaseModel):
    """Complete simulation result."""
    id: str
    approach: str
    target: str
    verdict: str
    confidence: float
    steps: List[SimulationStepResponse]
    final_state: PatientStateResponse
    critical_steps: List[int]
    highest_risk_structure: Optional[str]
    data_gaps: List[str]
    warnings: List[str]
    recommendations: List[str]
    simulation_time_ms: int


class PrincipleResponse(BaseModel):
    """Clinical principle summary."""
    id: str
    name: str
    statement: str
    domain: str
    category: str
    severity: str
    evidence_level: str


class PrincipleListResponse(BaseModel):
    """List of principles."""
    principles: List[PrincipleResponse]
    total: int
    domains: List[str]


class CorridorResponse(BaseModel):
    """Surgical corridor summary."""
    name: str
    display_name: str
    approach_type: str
    category: str
    step_count: int
    structures_at_risk: List[str]
    required_monitoring: List[str]
    evidence_level: str


class CorridorListResponse(BaseModel):
    """List of corridors."""
    corridors: List[CorridorResponse]
    total: int
    categories: List[str]


class EntityPhysicsResponse(BaseModel):
    """Entity physics properties."""
    name: str
    canonical_name: str
    mobility: str
    consistency: str
    is_end_artery: bool
    collateral_capacity: Optional[str]
    retraction_tolerance: str
    sacrifice_safety: str
    eloquence_grade: str
    confidence: float
    spatial_context: Dict[str, Any]


class ReasoningStatsResponse(BaseModel):
    """Reasoning engine statistics."""
    reasoner_initialized: bool
    simulator_initialized: bool
    principles_loaded: int
    corridors_loaded: int
    entities_cached: int
    domains: List[str]
    schema_available: bool


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/assess-risk", response_model=RiskAssessmentResponse)
async def assess_surgical_risk(
    request: RiskAssessmentRequest,
    reasoner = Depends(get_clinical_reasoner)
):
    """
    Assess risks of a specific surgical action.

    Evaluates the action against clinical principles and entity physics
    to identify potential risks, their severity, and mitigations.

    Returns:
        Risk assessment with recommendations and alternatives
    """
    start_time = time.time()

    # Check schema
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        # Build surgical context
        from src.neurosynth2.reasoning.models import SurgicalContext
        context = SurgicalContext(
            approach=request.approach or "unknown",
            target_structure=request.structure,
            patient_factors=request.patient_factors
        )

        # Assess risks
        risks = await reasoner.assess_action(
            action=request.action,
            structure=request.structure,
            context=context
        )

        # Process results
        risk_items = [
            RiskItem(
                level=r.level.value,
                structure=r.structure,
                action=r.action,
                principle_id=r.principle_id,
                principle_name=r.principle_name,
                mechanism=r.mechanism,
                mitigation=r.mitigation,
                confidence=round(r.confidence, 3),
                evidence_level=r.evidence_level
            )
            for r in risks
        ]

        # Get highest risk level (ordered: minimal < low < moderate < high < critical)
        RISK_ORDER = {"minimal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
        highest = max(
            (r.level.value for r in risks),
            key=lambda x: RISK_ORDER.get(x, 0),
            default="minimal"
        )

        # Collect recommendations (unique mitigations)
        recommendations = list(set(
            r.mitigation for r in risks
            if r.mitigation and r.level.value in ("high", "critical")
        ))

        # Collect alternatives
        alternatives = []
        if request.include_alternatives:
            for r in risks:
                if r.level.value in ("high", "critical"):
                    if r.mitigation:
                        alternatives.append(r.mitigation)
        alternatives = list(set(alternatives))[:5]  # Top 5

        # Identify data gaps
        data_gaps = [
            r.principle_id for r in risks
            if r.principle_id == "DATA_GAP"
        ]

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RiskAssessmentResponse(
            structure=request.structure,
            action=request.action,
            risks=risk_items,
            highest_risk_level=highest,
            total_risks=len(risks),
            recommendations=recommendations,
            alternatives=alternatives,
            data_gaps=data_gaps,
            assessment_time_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_approach(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    simulator = Depends(get_surgical_simulator)
):
    """
    Simulate a complete surgical approach.

    Runs step-by-step simulation with dynamic patient state tracking,
    evaluating risks at each step and propagating complications.

    Returns:
        Complete simulation result with verdict and recommendations
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        # Run simulation
        result = await simulator.simulate_approach(
            approach=request.approach,
            target=request.target,
            patient_factors=request.patient_factors,
            max_steps=request.max_steps
        )

        # Convert to response format
        steps = [
            SimulationStepResponse(
                step=s.step_number,
                action=s.action,
                structure=s.structure,
                highest_risk=s.highest_risk.value if s.highest_risk else None,
                outcome=s.outcome,
                complications=s.complications_occurred,
                is_decision_point=s.is_decision_point,
                alternatives=s.alternatives
            )
            for s in result.steps
        ]

        final_state = None
        if result.final_state:
            state_dict = result.final_state.to_dict()
            final_state = PatientStateResponse(
                condition=state_dict["condition"],
                visibility=state_dict["visibility"],
                brain_relaxation=state_dict["brain_relaxation"],
                active_bleeding=state_dict["active_bleeding"],
                bleeding_rate=state_dict["bleeding_rate"],
                brain_swelling=state_dict["brain_swelling"],
                monitoring=state_dict["monitoring"],
                cumulative_risk_score=state_dict["cumulative_risk_score"],
                complications=state_dict["complications"]
            )

        # Log simulation for audit (background task)
        background_tasks.add_task(
            _log_simulation,
            result.to_dict()
        )

        return SimulationResponse(
            id=str(result.id),
            approach=result.approach,
            target=result.target,
            verdict=result.verdict.value,
            confidence=round(result.confidence, 3),
            steps=steps,
            final_state=final_state,
            critical_steps=result.critical_steps,
            highest_risk_structure=result.highest_risk_structure,
            data_gaps=result.data_gaps,
            warnings=result.warnings,
            recommendations=result.recommendations,
            simulation_time_ms=result.simulation_time_ms
        )

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _log_simulation(result_dict: Dict):
    """Log simulation to database (background task)."""
    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            # Check if simulation_sessions table exists
            has_table = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'simulation_sessions'
                )
            """)

            if not has_table:
                return

            import json
            await conn.execute("""
                INSERT INTO simulation_sessions (
                    approach, target_pathology, verdict, confidence,
                    steps, final_state, data_gaps, warnings, recommendations,
                    execution_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                result_dict.get("approach"),
                result_dict.get("target"),
                result_dict.get("verdict"),
                result_dict.get("confidence"),
                json.dumps(result_dict.get("steps", [])),
                json.dumps(result_dict.get("final_state", {})),
                result_dict.get("data_gaps", []),
                result_dict.get("warnings", []),
                result_dict.get("recommendations", []),
                result_dict.get("simulation_time_ms", 0)
            )
    except Exception as e:
        logger.warning(f"Failed to log simulation: {e}")


@router.get("/principles", response_model=PrincipleListResponse)
async def list_clinical_principles(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    List available clinical principles.

    Principles are the IF-THEN rules that power surgical reasoning.
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            # Build query
            query = """
                SELECT id, name, statement, domain, category, severity, evidence_level
                FROM clinical_principles
                WHERE is_active = TRUE
            """
            params = []

            if domain:
                params.append(domain)
                query += f" AND domain = ${len(params)}"
            if category:
                params.append(category)
                query += f" AND category = ${len(params)}"
            if severity:
                params.append(severity)
                query += f" AND severity = ${len(params)}"

            query += f" ORDER BY severity DESC, domain, id LIMIT ${len(params) + 1}"
            params.append(limit)

            rows = await conn.fetch(query, *params)

            # Get domains
            domains = await conn.fetch("""
                SELECT DISTINCT domain FROM clinical_principles WHERE is_active = TRUE
            """)

            principles = [
                PrincipleResponse(
                    id=r["id"],
                    name=r["name"],
                    statement=r["statement"],
                    domain=r["domain"],
                    category=r["category"] or "",
                    severity=r["severity"],
                    evidence_level=r["evidence_level"]
                )
                for r in rows
            ]

            return PrincipleListResponse(
                principles=principles,
                total=len(principles),
                domains=[d["domain"] for d in domains]
            )

    except Exception as e:
        logger.error(f"Failed to list principles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/corridors", response_model=CorridorListResponse)
async def list_surgical_corridors(
    category: Optional[str] = Query(None, description="Filter by category"),
    subspecialty: Optional[str] = Query(None, description="Filter by subspecialty"),
    limit: int = Query(50, ge=1, le=100)
):
    """
    List available surgical corridors.

    Corridors define the sequence of structures encountered in surgical approaches.
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT name, display_name, approach_type, category,
                       array_length(structure_sequence, 1) as step_count,
                       structures_at_risk, required_monitoring, evidence_level
                FROM surgical_corridors
                WHERE 1=1
            """
            params = []

            if category:
                params.append(category)
                query += f" AND category = ${len(params)}"
            if subspecialty:
                params.append(subspecialty)
                query += f" AND subspecialty = ${len(params)}"

            query += f" ORDER BY category, approach_type LIMIT ${len(params) + 1}"
            params.append(limit)

            rows = await conn.fetch(query, *params)

            # Get categories
            categories = await conn.fetch("""
                SELECT DISTINCT category FROM surgical_corridors
            """)

            corridors = [
                CorridorResponse(
                    name=r["name"],
                    display_name=r["display_name"],
                    approach_type=r["approach_type"] or "",
                    category=r["category"] or "",
                    step_count=r["step_count"] or 0,
                    structures_at_risk=r["structures_at_risk"] or [],
                    required_monitoring=r["required_monitoring"] or [],
                    evidence_level=r["evidence_level"]
                )
                for r in rows
            ]

            return CorridorListResponse(
                corridors=corridors,
                total=len(corridors),
                categories=[c["category"] for c in categories if c["category"]]
            )

    except Exception as e:
        logger.error(f"Failed to list corridors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{name}/physics", response_model=EntityPhysicsResponse)
async def get_entity_physics(
    name: str,
    reasoner = Depends(get_clinical_reasoner)
):
    """
    Get physics properties for an anatomical entity.

    Returns the physical and surgical properties that drive reasoning.
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        entity = await reasoner.get_entity_physics(name)

        if entity is None:
            raise HTTPException(
                status_code=404,
                detail=f"No physics data found for entity: {name}"
            )

        return EntityPhysicsResponse(
            name=entity.name,
            canonical_name=entity.canonical_name,
            mobility=entity.mobility,
            consistency=entity.consistency,
            is_end_artery=entity.is_end_artery,
            collateral_capacity=entity.collateral_capacity,
            retraction_tolerance=entity.retraction_tolerance,
            sacrifice_safety=entity.sacrifice_safety,
            eloquence_grade=entity.eloquence_grade,
            confidence=round(entity.confidence, 3),
            spatial_context=entity.spatial_context
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity physics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ReasoningStatsResponse)
async def get_reasoning_stats():
    """
    Get reasoning engine statistics.

    Useful for monitoring and debugging the reasoning system.
    """
    schema_available = await _check_reasoning_schema()

    stats = {
        "reasoner_initialized": False,
        "simulator_initialized": False,
        "principles_loaded": 0,
        "corridors_loaded": 0,
        "entities_cached": 0,
        "domains": [],
        "schema_available": schema_available
    }

    try:
        reasoner = await get_clinical_reasoner()
        if reasoner is not None:
            reasoner_stats = reasoner.stats
            stats.update({
                "reasoner_initialized": reasoner_stats.get("initialized", False),
                "principles_loaded": reasoner_stats.get("principles_loaded", 0),
                "entities_cached": reasoner_stats.get("entities_cached", 0),
                "domains": reasoner_stats.get("domains", [])
            })
    except Exception:
        pass

    try:
        simulator = await get_surgical_simulator()
        if simulator is not None:
            simulator_stats = simulator.stats
            stats.update({
                "simulator_initialized": simulator_stats.get("initialized", False),
                "corridors_loaded": simulator_stats.get("corridors_cached", 0)
            })
    except Exception:
        pass

    return ReasoningStatsResponse(**stats)


@router.get("/corridor/{name}")
async def get_corridor_detail(name: str):
    """
    Get detailed corridor definition.

    Returns complete structure sequence and critical steps.
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM surgical_corridors
                WHERE LOWER(name) = LOWER($1) OR LOWER(approach_type) = LOWER($1)
            """, name)

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Corridor not found: {name}"
                )

            return dict(row)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get corridor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/principle/{principle_id}")
async def get_principle_detail(principle_id: str):
    """
    Get detailed principle definition.

    Returns full antecedent, consequent, and examples.
    """
    if not await _check_reasoning_schema():
        _raise_schema_error()

    try:
        pool = get_ns2_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM clinical_principles
                WHERE id = $1
            """, principle_id)

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Principle not found: {principle_id}"
                )

            return dict(row)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get principle: {e}")
        raise HTTPException(status_code=500, detail=str(e))
