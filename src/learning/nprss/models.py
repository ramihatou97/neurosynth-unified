# src/learning/nprss/models.py
"""
Unified Data Models for NPRSS Procedural Learning System

Combines:
- 6-level procedural hierarchy (Procedure -> Phase -> Step -> Substep -> Task -> Motion)
- 4-Phase learning framework (Architecture -> Approach -> Target -> Closure)
- Critical Safety Points (CSPs)
- FSRS learning cards
- R1-R7 retrieval schedules
- Mastery tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ACGMESubspecialty(str, Enum):
    """ACGME Neurosurgery subspecialty domains"""
    BRAIN_TUMOR = "brain_tumor"
    SPINE = "spine"
    CEREBROVASCULAR = "cerebrovascular"
    FUNCTIONAL = "functional"
    PEDIATRIC = "pediatric"
    PAIN_PERIPHERAL = "pain_peripheral"
    TRAUMA = "trauma"
    CRITICAL_CARE = "critical_care"


class Complexity(str, Enum):
    """Procedure complexity level"""
    ROUTINE = "routine"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class ElementType(str, Enum):
    """Procedural element types (6-level hierarchy)"""
    PHASE = "phase"      # Level 4
    STEP = "step"        # Level 3
    SUBSTEP = "substep"  # Level 2
    TASK = "task"        # Level 1
    MOTION = "motion"    # Level 0


class PhaseType(str, Enum):
    """
    Universal 4-Phase Framework for Learning

    Based on Cowan's 4-chunk working memory capacity:
    - ARCHITECTURE: Before the incision
    - APPROACH: From incision to dural opening
    - TARGET: The "WHY" of surgery
    - CLOSURE: Reverse of approach
    """
    ARCHITECTURE = "architecture"  # Positioning, fixation, registration
    APPROACH = "approach"          # Incision, bone work, dural opening
    TARGET = "target"              # Pathology-specific actions
    CLOSURE = "closure"            # Reconstruction, closure


class MasteryLevel(IntEnum):
    """
    4-level mastery progression (Dreyfus-inspired)

    1 = NOT_YET: <50% recall
    2 = DEVELOPING: Gaps in substeps/CSPs
    3 = COMPETENT: Accurate with minor omissions
    4 = MASTERY: Complete + can teach + variations
    """
    NOT_YET = 1
    DEVELOPING = 2
    COMPETENT = 3
    MASTERY = 4


class EntrustmentLevel(IntEnum):
    """
    Zwisch Scale for Entrustable Professional Activities (EPAs)

    Used for workplace-based assessment of surgical competence.
    """
    SHOW_TELL = 1       # Instructor performs, learner observes
    ACTIVE_ASSIST = 2   # Learner assists, instructor performs critical parts
    PASSIVE_ASSIST = 3  # Learner performs, instructor assists as needed
    SUPERVISION = 4     # Learner performs independently, instructor supervises


class MillerLevel(str, Enum):
    """Miller's Pyramid levels for assessment"""
    KNOWS = "knows"           # Knowledge quiz
    KNOWS_HOW = "knows_how"   # Clinical reasoning
    SHOWS_HOW = "shows_how"   # Demonstrated skill (simulation)
    DOES = "does"             # Workplace performance


class CardType(str, Enum):
    """Types of FSRS learning cards"""
    SEQUENCE = "sequence"       # Order the phases/steps
    IMAGE = "image"             # Identify anatomy in image
    MCQ = "mcq"                 # Multiple choice
    SCENARIO = "scenario"       # Decision scenario
    CSP_TRIGGER = "csp_trigger" # CSP rapid-fire
    DICTATION = "dictation"     # Dictation cue
    SAFE_ZONE = "safe_zone"     # Safe zone identification


class FSRSState(str, Enum):
    """FSRS card learning state"""
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    RELEARNING = "relearning"


# =============================================================================
# CORE PROCEDURAL MODELS
# =============================================================================

@dataclass
class SafeEntryZone:
    """
    Neurosurgery-critical: Quantitative safe zone measurements

    Example: Posterior Median Sulcus safe zone for myelotomy
    - mean_safe_depth_mm: 6.0
    - boundaries: dorsal columns laterally, central gray anteriorly
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    anatomical_region: str = ""

    # Quantitative Measurements
    mean_safe_depth_mm: Optional[float] = None
    min_safe_depth_mm: Optional[float] = None
    max_safe_depth_mm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None

    # Boundaries
    superior_boundary: Optional[str] = None
    inferior_boundary: Optional[str] = None
    lateral_boundary: Optional[str] = None
    medial_boundary: Optional[str] = None

    # Critical Distances
    distance_to_critical_structures: List[Dict[str, Any]] = field(default_factory=list)

    # Source
    source_reference: Optional[str] = None
    evidence_grade: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "anatomical_region": self.anatomical_region,
            "mean_safe_depth_mm": self.mean_safe_depth_mm,
            "min_safe_depth_mm": self.min_safe_depth_mm,
            "max_safe_depth_mm": self.max_safe_depth_mm,
            "boundaries": {
                "superior": self.superior_boundary,
                "inferior": self.inferior_boundary,
                "lateral": self.lateral_boundary,
                "medial": self.medial_boundary,
            },
            "source_reference": self.source_reference,
        }


@dataclass
class DangerZone:
    """Structures at risk during procedure"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    structures_at_risk: List[str] = field(default_factory=list)
    mechanism_of_injury: Optional[str] = None
    prevention_strategy: Optional[str] = None
    management_if_violated: Optional[str] = None
    anatomical_region: Optional[str] = None
    fma_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "structures_at_risk": self.structures_at_risk,
            "mechanism_of_injury": self.mechanism_of_injury,
            "prevention_strategy": self.prevention_strategy,
            "management_if_violated": self.management_if_violated,
        }


@dataclass
class DecisionBranch:
    """Intraoperative decision point"""
    id: UUID = field(default_factory=uuid4)
    source_element_id: Optional[UUID] = None
    condition_type: str = ""  # anatomical_variant, complication, finding
    condition_criteria: Dict[str, Any] = field(default_factory=dict)
    alternative_path_id: Optional[UUID] = None
    evidence_grade: Optional[str] = None
    evidence_reference: Optional[str] = None


@dataclass
class VisualDescription:
    """Impeccable format visual description for a step"""
    expected_view: str = ""
    landmarks: List[str] = field(default_factory=list)
    color_cues: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_view": self.expected_view,
            "landmarks": self.landmarks,
            "color_cues": self.color_cues,
        }


@dataclass
class ProcedureElement:
    """
    Unified element for levels 4-0 in hierarchy

    Level 4: Phase (ARCHITECTURE, APPROACH, TARGET, CLOSURE)
    Level 3: Step (major surgical action)
    Level 2: Substep (Impeccable 7-element format)
    Level 1: Task (elemental action)
    Level 0: Motion (atomic movement)
    """
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None
    parent_id: Optional[UUID] = None

    # Link to NeuroSynth chunk (provenance)
    source_chunk_id: Optional[UUID] = None

    # Hierarchy
    element_type: ElementType = ElementType.STEP
    granularity_level: int = 3  # 4=phase, 3=step, 2=substep, 1=task, 0=motion
    sequence_order: int = 0

    # Content
    name: str = ""
    description: Optional[str] = None
    critical_step: bool = False

    # Phase Classification (Learning System)
    phase_type: Optional[PhaseType] = None

    # Anatomical References
    anatomical_structure_fma: Optional[int] = None
    action_verb: Optional[str] = None
    safe_zone_refs: List[UUID] = field(default_factory=list)
    danger_zone_refs: List[UUID] = field(default_factory=list)

    # Technical Details
    instrument_sequence: List[Dict[str, Any]] = field(default_factory=list)
    ionm_requirements: List[str] = field(default_factory=list)

    # Impeccable 7-Element Fields (for substeps)
    standard_measurements: Optional[str] = None
    trajectory_specification: Optional[str] = None
    instrument_specification: Optional[str] = None
    the_maneuver: Optional[str] = None
    bailout_protocol: Optional[str] = None
    visual_description: Optional[VisualDescription] = None

    # Versioning
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "procedure_id": str(self.procedure_id) if self.procedure_id else None,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "element_type": self.element_type.value if isinstance(self.element_type, ElementType) else self.element_type,
            "granularity_level": self.granularity_level,
            "sequence_order": self.sequence_order,
            "name": self.name,
            "description": self.description,
            "critical_step": self.critical_step,
            "phase_type": self.phase_type.value if isinstance(self.phase_type, PhaseType) else self.phase_type,
            "the_maneuver": self.the_maneuver,
            "visual_description": self.visual_description.to_dict() if self.visual_description else None,
        }


@dataclass
class Procedure:
    """Level 5: Complete surgical procedure"""
    id: UUID = field(default_factory=uuid4)

    # Link to NeuroSynth content
    source_document_id: Optional[UUID] = None
    source_synthesis_id: Optional[UUID] = None

    # Identification
    snomed_ct_code: Optional[str] = None
    icd10_pcs_code: Optional[str] = None
    cpt_code: Optional[str] = None
    name: str = ""
    description: Optional[str] = None

    # ACGME Classification
    subspecialty_domain: ACGMESubspecialty = ACGMESubspecialty.BRAIN_TUMOR
    complexity: Complexity = Complexity.ROUTINE

    # Milestone Targets
    milestone_pc_target: Optional[int] = None  # Patient Care 1-5
    milestone_mk_target: Optional[int] = None  # Medical Knowledge 1-5

    # Anatomical Context
    primary_target_structure_fma: Optional[int] = None
    surgical_approach: Optional[str] = None

    # Versioning
    version: int = 1
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[UUID] = None

    # Relationships (loaded separately)
    elements: List[ProcedureElement] = field(default_factory=list)
    safe_zones: List[SafeEntryZone] = field(default_factory=list)
    danger_zones: List[DangerZone] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "subspecialty_domain": self.subspecialty_domain.value if isinstance(self.subspecialty_domain, ACGMESubspecialty) else self.subspecialty_domain,
            "complexity": self.complexity.value if isinstance(self.complexity, Complexity) else self.complexity,
            "surgical_approach": self.surgical_approach,
            "status": self.status,
            "element_count": len(self.elements),
        }


# =============================================================================
# LEARNING ENRICHMENT MODELS
# =============================================================================

@dataclass
class PhaseGate:
    """Verification checkpoint between phases"""
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None

    from_phase: PhaseType = PhaseType.ARCHITECTURE
    to_phase: PhaseType = PhaseType.APPROACH

    verification_questions: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class CriticalSafetyPoint:
    """
    CSP: Trigger-action safety circuit

    Prevents autopilot errors through visual cue -> mandatory action.

    Example:
        WHEN drilling posterior wall of IAC
        STOP IF dura turns blue (Bill's Bar)
        BECAUSE facial nerve lies immediately deep
    """
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None
    element_id: Optional[UUID] = None

    csp_number: int = 1
    phase_type: Optional[PhaseType] = None

    # Trigger-Action Circuit
    when_action: str = ""           # "Drilling posterior wall of IAC"
    stop_if_trigger: str = ""       # "Dura turns blue"
    visual_cue: str = ""            # Additional visual details

    # Consequence
    structure_at_risk: str = ""     # "Facial nerve"
    mechanism_of_injury: str = ""   # "Nerve lies immediately deep"

    # Recovery
    if_violated_action: Optional[str] = None

    # Source Tracing
    derived_from_danger_zone_id: Optional[UUID] = None
    derived_from_safe_zone_id: Optional[UUID] = None

    # Learning Metadata
    retrieval_cue: str = ""         # For spaced repetition testing
    common_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "csp_number": self.csp_number,
            "phase_type": self.phase_type.value if isinstance(self.phase_type, PhaseType) else self.phase_type,
            "when_action": self.when_action,
            "stop_if_trigger": self.stop_if_trigger,
            "structure_at_risk": self.structure_at_risk,
            "mechanism_of_injury": self.mechanism_of_injury,
            "if_violated_action": self.if_violated_action,
            "retrieval_cue": self.retrieval_cue,
        }


@dataclass
class VisuospatialAnchor:
    """
    3D-anchored step description for spatial memory encoding

    Enhances procedural memory by linking steps to visual landmarks
    and spatial relationships.
    """
    id: UUID = field(default_factory=uuid4)
    element_id: Optional[UUID] = None

    # From Visual Description
    expected_view: str = ""
    landmarks: List[str] = field(default_factory=list)
    color_cues: str = ""

    # Learning Enhancements
    mental_rotation_prompt: str = ""   # "What structure lies immediately deep?"
    spatial_relationship: str = ""      # "M1 runs inferior to..."
    depth_reference: str = ""           # "2cm from cortical surface"
    viewing_angle: str = ""             # "lateral", "superior"

    # 3D Coordinates (for AR/VR)
    coordinates_3d: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "element_id": str(self.element_id) if self.element_id else None,
            "expected_view": self.expected_view,
            "landmarks": self.landmarks,
            "color_cues": self.color_cues,
            "mental_rotation_prompt": self.mental_rotation_prompt,
        }


@dataclass
class SurgicalCardRow:
    """Single row in surgical card table"""
    phase: PhaseType
    phase_label: str              # "I. ARCHITECTURE"
    key_actions: List[str]        # Bullet points
    anchor_or_csp: str            # "ANCHOR: Malar eminence" or "CSP: Fat pad"
    anchor_type: str              # "anchor" | "csp" | "decision" | "verify"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value if isinstance(self.phase, PhaseType) else self.phase,
            "phase_label": self.phase_label,
            "key_actions": self.key_actions,
            "anchor_or_csp": self.anchor_or_csp,
            "anchor_type": self.anchor_type,
        }


@dataclass
class SurgicalCard:
    """
    One-page procedure summary for rapid reference

    Designed to fit on a single page for OR reference
    and exam preparation.
    """
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None

    # Header
    title: str = ""
    subtitle: str = ""
    approach: str = ""
    corridor: str = ""
    exam_relevance: str = "Royal College Core"

    # Content
    card_rows: List[SurgicalCardRow] = field(default_factory=list)
    csp_summary: List[Dict[str, str]] = field(default_factory=list)

    # Dictation Template
    dictation_template: str = ""

    # Mantra
    mantra: str = "4 folders -> 3-5 substeps -> visual anchors -> CSP triggers"

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "procedure_id": str(self.procedure_id) if self.procedure_id else None,
            "title": self.title,
            "subtitle": self.subtitle,
            "approach": self.approach,
            "corridor": self.corridor,
            "exam_relevance": self.exam_relevance,
            "card_rows": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.card_rows],
            "csp_summary": self.csp_summary,
            "dictation_template": self.dictation_template,
            "mantra": self.mantra,
        }


# =============================================================================
# LEARNING SYSTEM MODELS (FSRS + Spaced Repetition)
# =============================================================================

@dataclass
class LearningCard:
    """FSRS learning card generated from procedure content"""
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None
    element_id: Optional[UUID] = None
    csp_id: Optional[UUID] = None

    card_type: CardType = CardType.MCQ

    # Content
    prompt: str = ""
    answer: str = ""
    options: Optional[List[str]] = None  # For MCQ
    image_asset_id: Optional[UUID] = None

    # Metadata
    difficulty_preset: float = 0.3
    tags: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "procedure_id": str(self.procedure_id) if self.procedure_id else None,
            "card_type": self.card_type.value if isinstance(self.card_type, CardType) else self.card_type,
            "prompt": self.prompt,
            "answer": self.answer,
            "options": self.options,
            "difficulty_preset": self.difficulty_preset,
            "tags": self.tags,
        }


@dataclass
class CardMemoryState:
    """FSRS memory state per user per card"""
    id: UUID = field(default_factory=uuid4)
    card_id: Optional[UUID] = None
    user_id: Optional[str] = None

    # FSRS Parameters
    difficulty: float = 0.3       # D parameter (1-10)
    stability: float = 1.0        # S in days
    retrievability: float = 1.0   # R (0-1)

    # State
    state: FSRSState = FSRSState.NEW
    step: int = 0
    due_date: Optional[datetime] = None

    # History
    review_count: int = 0
    lapses: int = 0
    last_review: Optional[datetime] = None


@dataclass
class RetrievalSchedule:
    """R1-R7 expansion schedule for a procedure"""
    id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    procedure_id: Optional[UUID] = None

    encoding_date: datetime = field(default_factory=datetime.now)
    target_retention_days: int = 180  # 6 months for Royal College

    # Adaptive Parameters
    interval_multiplier: float = 1.0

    created_at: datetime = field(default_factory=datetime.now)

    # Sessions (loaded separately)
    sessions: List['RetrievalSession'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "procedure_id": str(self.procedure_id) if self.procedure_id else None,
            "encoding_date": self.encoding_date.isoformat() if self.encoding_date else None,
            "target_retention_days": self.target_retention_days,
            "session_count": len(self.sessions),
        }


@dataclass
class RetrievalSession:
    """Individual retrieval session (R1, R2, etc.)"""
    id: UUID = field(default_factory=uuid4)
    schedule_id: Optional[UUID] = None

    session_number: int = 1  # R1, R2, R3...
    scheduled_date: Optional[datetime] = None
    days_from_encoding: int = 1

    # Task
    retrieval_task: str = ""
    task_type: str = "free_recall"  # free_recall, cued_recall, rehearsal, elaboration, interleaved, application
    estimated_duration_min: int = 15

    # Focus Areas
    focus_phases: List[PhaseType] = field(default_factory=list)
    focus_csps: List[int] = field(default_factory=list)

    # Completion
    completed: bool = False
    completed_at: Optional[datetime] = None
    self_assessment_score: Optional[int] = None  # 1-4 mastery level
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "session_number": self.session_number,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "days_from_encoding": self.days_from_encoding,
            "retrieval_task": self.retrieval_task,
            "task_type": self.task_type,
            "estimated_duration_min": self.estimated_duration_min,
            "completed": self.completed,
            "self_assessment_score": self.self_assessment_score,
        }


@dataclass
class ProcedureMastery:
    """User mastery state for a procedure"""
    id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    procedure_id: Optional[UUID] = None

    # Current State
    current_level: MasteryLevel = MasteryLevel.NOT_YET

    # Phase-Level Granularity
    phase_scores: Dict[str, float] = field(default_factory=dict)

    # Weak Points
    weak_csps: List[int] = field(default_factory=list)
    weak_phases: List[str] = field(default_factory=list)

    # History
    assessment_history: List[Dict[str, Any]] = field(default_factory=list)
    total_retrieval_sessions: int = 0
    last_session_date: Optional[datetime] = None

    # Predictions
    predicted_retention_score: float = 0.5
    next_optimal_review: Optional[datetime] = None

    # Entrustment (EPA)
    entrustment_level: Optional[EntrustmentLevel] = None

    def to_dict(self) -> Dict[str, Any]:
        level_names = {1: "NOT_YET", 2: "DEVELOPING", 3: "COMPETENT", 4: "MASTERY"}
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "procedure_id": str(self.procedure_id) if self.procedure_id else None,
            "current_level": self.current_level,
            "current_level_name": level_names.get(int(self.current_level), "UNKNOWN"),
            "phase_scores": self.phase_scores,
            "weak_csps": self.weak_csps,
            "predicted_retention_score": self.predicted_retention_score,
            "next_optimal_review": self.next_optimal_review.isoformat() if self.next_optimal_review else None,
        }


# =============================================================================
# ASSESSMENT MODELS
# =============================================================================

@dataclass
class AssessmentItem:
    """Miller's Pyramid assessment item"""
    id: UUID = field(default_factory=uuid4)
    procedure_id: Optional[UUID] = None
    element_id: Optional[UUID] = None

    miller_level: MillerLevel = MillerLevel.KNOWS
    item_type: str = "mcq"  # mcq, sequence, scenario, labeled_anatomy

    prompt: str = ""
    correct_answer: str = ""
    options: Optional[List[str]] = None
    scoring_rubric: Optional[Dict[str, Any]] = None

    difficulty_level: str = "medium"
    tags: List[str] = field(default_factory=list)


@dataclass
class EntrustmentAssessment:
    """Workplace-based entrustment assessment"""
    id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    procedure_id: Optional[UUID] = None
    assessor_id: Optional[str] = None

    assessment_date: datetime = field(default_factory=datetime.now)

    # Zwisch Scale
    entrustment_level: EntrustmentLevel = EntrustmentLevel.SHOW_TELL

    # Milestone Mapping
    milestone_pc_level: Optional[int] = None
    milestone_mk_level: Optional[int] = None

    narrative_feedback: Optional[str] = None
    verified: bool = False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

class RetrievalScheduleFactory:
    """Factory for creating R1-R7 schedules"""

    R1_R7_TEMPLATE = [
        (1, "R1", "Dictate operative note from memory", "free_recall", 15),
        (3, "R2", "Write Surgical Card from memory -> compare", "free_recall", 20),
        (7, "R3", "Mental rehearsal (full procedure)", "rehearsal", 20),
        (14, "R4", "Verbal teach-back (explain to peer/recorder)", "elaboration", 25),
        (30, "R5", "CSP rapid-fire quiz + case variation", "cued_recall", 15),
        (60, "R6", "Interleaved review (mix with similar procedures)", "interleaved", 30),
        (120, "R7", "Full simulation or cadaver lab if available", "application", 60),
    ]

    @classmethod
    def create(
        cls,
        user_id: str,
        procedure_id: UUID,
        encoding_date: Optional[datetime] = None,
        target_retention_days: int = 180
    ) -> RetrievalSchedule:
        """Create schedule with R1-R7 sessions"""
        encoding_date = encoding_date or datetime.now()

        schedule = RetrievalSchedule(
            user_id=user_id,
            procedure_id=procedure_id,
            encoding_date=encoding_date,
            target_retention_days=target_retention_days
        )

        for day, label, task, task_type, duration in cls.R1_R7_TEMPLATE:
            session = RetrievalSession(
                schedule_id=schedule.id,
                session_number=int(label[1]),
                scheduled_date=encoding_date + timedelta(days=day),
                days_from_encoding=day,
                retrieval_task=task,
                task_type=task_type,
                estimated_duration_min=duration
            )
            schedule.sessions.append(session)

        return schedule
