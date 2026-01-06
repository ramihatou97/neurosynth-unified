"""
NeuroSynth 2.0 - Reasoning Models
==================================

Core data structures for surgical reasoning and simulation.

This module defines:
- Risk levels and assessments
- Patient state tracking
- Surgical context
- Simulation results

These models are shared across:
- ClinicalReasoner (deductive reasoning)
- SurgicalSimulator (causal simulation)
- API responses (serialization)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from uuid import UUID, uuid4
import json


# =============================================================================
# ENUMS
# =============================================================================

class RiskLevel(Enum):
    """Surgical risk classification with clinical meaning."""
    MINIMAL = "minimal"      # No significant concern
    LOW = "low"              # Standard surgical risk
    MODERATE = "moderate"    # Requires attention/modification
    HIGH = "high"            # Significant injury potential
    CRITICAL = "critical"    # Life or function threatening


class PatientCondition(Enum):
    """Overall patient state during simulation."""
    STABLE = "stable"
    MILDLY_COMPROMISED = "mildly_compromised"
    MODERATELY_COMPROMISED = "moderately_compromised"
    SEVERELY_COMPROMISED = "severely_compromised"
    CRITICAL = "critical"


class SimulationVerdict(Enum):
    """Final simulation verdict."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH_RISK"
    CONTRAINDICATED = "CONTRAINDICATED"


class ComplicationType(Enum):
    """Types of intraoperative complications."""
    BLEEDING = "bleeding"
    ISCHEMIA = "ischemia"
    NEURAL_INJURY = "neural_injury"
    BRAIN_SWELLING = "brain_swelling"
    RETRACTION_INJURY = "retraction_injury"
    THERMAL_INJURY = "thermal_injury"
    CSF_LEAK = "csf_leak"
    MONITORING_CHANGE = "monitoring_change"


class SurgicalAction(Enum):
    """Standard surgical actions for reasoning."""
    EXPOSE = "expose"
    DISSECT = "dissect"
    RETRACT = "retract"
    MOBILIZE = "mobilize"
    COAGULATE = "coagulate"
    CLIP = "clip"
    CUT = "cut"
    SACRIFICE = "sacrifice"
    PRESERVE = "preserve"
    DRILL = "drill"
    RESECT = "resect"
    DEBULK = "debulk"
    REPAIR = "repair"
    CLOSE = "close"


# =============================================================================
# ENTITY PHYSICS MODEL
# =============================================================================

@dataclass
class EntityPhysics:
    """
    Physics properties of an anatomical entity.
    
    This is the key data structure that enables reasoning:
    - mobility determines retraction safety
    - consistency determines manipulation risk
    - vascular properties determine sacrifice consequences
    """
    name: str
    canonical_name: str
    
    # Physics
    mobility: str = "fixed"
    consistency: str = "soft_brain"
    
    # Vascular
    is_end_artery: bool = False
    has_collaterals: Optional[bool] = None
    collateral_capacity: Optional[str] = None
    vessel_diameter_mm: Optional[float] = None
    territory_supplied: List[str] = field(default_factory=list)
    
    # Surgical
    retraction_tolerance: str = "minimal"
    sacrifice_safety: str = "never"
    coagulation_tolerance: str = "minimal"
    eloquence_grade: str = "non_eloquent"
    
    # Spatial
    spatial_context: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence
    confidence: float = 0.5
    source_chunk_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "mobility": self.mobility,
            "consistency": self.consistency,
            "is_end_artery": self.is_end_artery,
            "has_collaterals": self.has_collaterals,
            "collateral_capacity": self.collateral_capacity,
            "retraction_tolerance": self.retraction_tolerance,
            "sacrifice_safety": self.sacrifice_safety,
            "eloquence_grade": self.eloquence_grade,
            "spatial_context": self.spatial_context,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "EntityPhysics":
        """Create from database row."""
        import json
        
        # Handle spatial_context which might be JSON string
        spatial_context = row.get("spatial_context", {})
        if isinstance(spatial_context, str):
            try:
                spatial_context = json.loads(spatial_context)
            except json.JSONDecodeError:
                spatial_context = {}
        
        return cls(
            name=row["name"],
            canonical_name=row.get("canonical_name", row["name"]),
            mobility=row.get("mobility", "fixed"),
            consistency=row.get("consistency", "soft_brain"),
            is_end_artery=row.get("is_end_artery", False),
            has_collaterals=row.get("has_collaterals"),
            collateral_capacity=row.get("collateral_capacity"),
            vessel_diameter_mm=row.get("vessel_diameter_mm"),
            territory_supplied=row.get("territory_supplied", []),
            retraction_tolerance=row.get("retraction_tolerance", "minimal"),
            sacrifice_safety=row.get("sacrifice_safety", "never"),
            coagulation_tolerance=row.get("coagulation_tolerance", "minimal"),
            eloquence_grade=row.get("eloquence_grade", "non_eloquent"),
            spatial_context=spatial_context,
            confidence=row.get("confidence", 0.5),
            source_chunk_ids=row.get("source_chunk_ids", [])
        )


# =============================================================================
# RISK ASSESSMENT MODEL
# =============================================================================

@dataclass
class RiskAssessment:
    """
    Result of evaluating a surgical action against clinical principles.
    
    Each assessment represents one identified risk with:
    - The risk level (how serious)
    - The triggering principle (why)
    - The mechanism (what happens)
    - Possible mitigation (how to avoid)
    """
    level: RiskLevel
    structure: str
    action: str
    principle_id: str
    principle_name: str
    mechanism: str
    mitigation: Optional[str] = None
    confidence: float = 0.5
    evidence_level: str = "IV"
    
    # For traceability
    supporting_chunks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "level": self.level.value,
            "structure": self.structure,
            "action": self.action,
            "principle_id": self.principle_id,
            "principle_name": self.principle_name,
            "mechanism": self.mechanism,
            "mitigation": self.mitigation,
            "confidence": round(self.confidence, 3),
            "evidence_level": self.evidence_level
        }
    
    def __lt__(self, other: "RiskAssessment") -> bool:
        """Compare by risk level for sorting."""
        level_order = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        return level_order[self.level] < level_order[other.level]


# =============================================================================
# SURGICAL CONTEXT MODEL
# =============================================================================

@dataclass
class SurgicalContext:
    """
    Current surgical state for reasoning.
    
    Tracks what has happened in the surgery so far,
    allowing context-aware risk assessment.
    """
    approach: str
    target_structure: str
    current_step: int = 0
    
    # Structures affected
    structures_exposed: Set[str] = field(default_factory=set)
    structures_mobilized: Set[str] = field(default_factory=set)
    structures_retracted: Set[str] = field(default_factory=set)
    vessels_sacrificed: Set[str] = field(default_factory=set)
    vessels_temporarily_occluded: Set[str] = field(default_factory=set)
    
    # Patient factors
    patient_factors: Dict[str, Any] = field(default_factory=dict)
    # Example: {"age": 65, "comorbidities": ["diabetes"], "previous_surgery": True}
    
    # Monitoring state
    monitoring_active: Set[str] = field(default_factory=set)
    monitoring_alerts: List[Dict] = field(default_factory=list)
    
    # Timing
    total_retraction_time_min: float = 0.0
    temporary_occlusion_time_min: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "approach": self.approach,
            "target": self.target_structure,
            "step": self.current_step,
            "exposed": list(self.structures_exposed),
            "mobilized": list(self.structures_mobilized),
            "sacrificed": list(self.vessels_sacrificed),
            "patient_factors": self.patient_factors
        }


# =============================================================================
# PATIENT STATE MODEL (Dynamic Simulation State)
# =============================================================================

@dataclass
class PatientState:
    """
    Dynamic patient state during simulation.
    
    THIS IS THE KEY INNOVATION:
    Surgery is NOT a linear sequence. Events at step N affect options at step N+1.
    
    This class tracks:
    - Physiological state (bleeding, swelling)
    - Visibility/exposure quality
    - Neurophysiological monitoring
    - Accumulated complications
    """
    # Overall condition
    condition: PatientCondition = PatientCondition.STABLE
    
    # =========================================================================
    # Visibility and Exposure
    # =========================================================================
    visibility: float = 1.0        # 1.0 = perfect, 0.0 = no visibility
    exposure_quality: float = 1.0  # 1.0 = excellent, 0.0 = none
    brain_relaxation: float = 1.0  # 1.0 = slack brain, 0.0 = tight/swollen
    
    # =========================================================================
    # Active Complications
    # =========================================================================
    active_bleeding: bool = False
    bleeding_source: Optional[str] = None
    bleeding_rate: str = "none"  # none, minimal, moderate, severe, massive
    
    brain_swelling: bool = False
    swelling_severity: str = "none"  # none, mild, moderate, severe
    
    # =========================================================================
    # Neurophysiological Monitoring
    # =========================================================================
    # 1.0 = baseline, 0.0 = lost
    motor_evoked_potentials: float = 1.0
    somatosensory_evoked_potentials: float = 1.0
    brainstem_auditory_evoked: float = 1.0
    facial_nerve_emg: float = 1.0
    
    # Specific alerts
    monitoring_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # =========================================================================
    # Structure Tracking
    # =========================================================================
    structures_sacrificed: List[str] = field(default_factory=list)
    structures_injured: List[str] = field(default_factory=list)
    structures_preserved: List[str] = field(default_factory=list)
    
    # Vessels with temporary occlusion
    temporary_occlusions: Dict[str, float] = field(default_factory=dict)
    # {vessel_name: time_in_minutes}
    
    # =========================================================================
    # Accumulated Metrics
    # =========================================================================
    cumulative_risk_score: float = 0.0
    total_blood_loss_ml: float = 0.0
    total_retraction_time_min: float = 0.0
    
    # Complication log
    complications: List[Dict[str, Any]] = field(default_factory=list)
    
    def apply_complication(
        self, 
        complication_type: str, 
        severity: str, 
        source: str,
        details: Optional[Dict] = None
    ):
        """
        Apply a complication to patient state.
        
        This modifies the state and propagates consequences.
        """
        comp_record = {
            "type": complication_type,
            "severity": severity,
            "source": source,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.complications.append(comp_record)
        
        # =====================================================================
        # Bleeding Effects
        # =====================================================================
        if complication_type == ComplicationType.BLEEDING.value or complication_type == "bleeding":
            self.active_bleeding = True
            self.bleeding_source = source
            self.bleeding_rate = severity
            
            # Bleeding affects visibility
            visibility_impact = {
                "minimal": 0.9,
                "moderate": 0.6,
                "severe": 0.3,
                "massive": 0.1
            }
            self.visibility *= visibility_impact.get(severity, 0.5)
            
            # Estimate blood loss
            blood_loss = {
                "minimal": 50,
                "moderate": 200,
                "severe": 500,
                "massive": 1000
            }
            self.total_blood_loss_ml += blood_loss.get(severity, 100)
            
            # Severe bleeding degrades condition
            if severity in ("severe", "massive"):
                self.condition = self._degrade_condition()
                
        # =====================================================================
        # Brain Swelling Effects
        # =====================================================================
        elif complication_type == ComplicationType.BRAIN_SWELLING.value or complication_type == "swelling":
            self.brain_swelling = True
            self.swelling_severity = severity
            
            # Swelling affects exposure and relaxation
            relaxation_impact = {
                "mild": 0.8,
                "moderate": 0.5,
                "severe": 0.2
            }
            self.brain_relaxation *= relaxation_impact.get(severity, 0.5)
            self.exposure_quality *= relaxation_impact.get(severity, 0.5)
            
            if severity == "severe":
                self.condition = self._degrade_condition()
                
        # =====================================================================
        # Neural Injury Effects
        # =====================================================================
        elif complication_type == ComplicationType.NEURAL_INJURY.value or complication_type == "neural_injury":
            self.structures_injured.append(source)
            
            # Update monitoring if relevant
            if "motor" in source.lower():
                self.motor_evoked_potentials *= 0.5
            if "sensory" in source.lower():
                self.somatosensory_evoked_potentials *= 0.5
            if "facial" in source.lower():
                self.facial_nerve_emg *= 0.5
                
            self.cumulative_risk_score += 0.2
            
        # =====================================================================
        # Monitoring Changes
        # =====================================================================
        elif complication_type == ComplicationType.MONITORING_CHANGE.value or complication_type == "monitoring_change":
            self.monitoring_alerts.append({
                "modality": details.get("modality", "unknown"),
                "change": details.get("change", "decrease"),
                "magnitude": severity,
                "source": source
            })
            
            # Update specific monitoring values
            modality = details.get("modality", "").lower()
            magnitude = {"mild": 0.8, "moderate": 0.5, "severe": 0.2}.get(severity, 0.5)
            
            if "mep" in modality or "motor" in modality:
                self.motor_evoked_potentials *= magnitude
            if "ssep" in modality or "sensory" in modality:
                self.somatosensory_evoked_potentials *= magnitude
            if "baep" in modality:
                self.brainstem_auditory_evoked *= magnitude
            if "emg" in modality or "facial" in modality:
                self.facial_nerve_emg *= magnitude
    
    def record_sacrifice(self, structure: str, is_critical: bool = False):
        """Record structure sacrifice and consequences."""
        self.structures_sacrificed.append(structure)
        
        if is_critical:
            self.cumulative_risk_score += 0.3
            self.condition = self._degrade_condition()
    
    def add_temporary_occlusion(self, vessel: str, duration_min: float):
        """Track temporary vessel occlusion."""
        if vessel in self.temporary_occlusions:
            self.temporary_occlusions[vessel] += duration_min
        else:
            self.temporary_occlusions[vessel] = duration_min
            
        # Check for prolonged occlusion
        if self.temporary_occlusions[vessel] > 15:  # 15 min threshold
            self.cumulative_risk_score += 0.1
    
    def _degrade_condition(self) -> PatientCondition:
        """Worsen patient condition by one level."""
        progression = [
            PatientCondition.STABLE,
            PatientCondition.MILDLY_COMPROMISED,
            PatientCondition.MODERATELY_COMPROMISED,
            PatientCondition.SEVERELY_COMPROMISED,
            PatientCondition.CRITICAL
        ]
        current_idx = progression.index(self.condition)
        return progression[min(current_idx + 1, len(progression) - 1)]
    
    def get_risk_multiplier(self) -> float:
        """
        Calculate risk multiplier based on current state.
        
        Compromised states make further actions riskier.
        """
        multiplier = 1.0
        
        # Bleeding increases risk
        if self.active_bleeding:
            bleeding_multiplier = {
                "minimal": 1.1,
                "moderate": 1.5,
                "severe": 2.0,
                "massive": 3.0
            }
            multiplier *= bleeding_multiplier.get(self.bleeding_rate, 1.5)
        
        # Poor visibility increases risk
        if self.visibility < 0.5:
            multiplier *= (2.0 - self.visibility)
        
        # Brain swelling increases risk
        if self.brain_swelling:
            swelling_multiplier = {
                "mild": 1.2,
                "moderate": 1.5,
                "severe": 2.0
            }
            multiplier *= swelling_multiplier.get(self.swelling_severity, 1.5)
        
        return multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response and logging."""
        return {
            "condition": self.condition.value,
            "visibility": round(self.visibility, 2),
            "exposure_quality": round(self.exposure_quality, 2),
            "brain_relaxation": round(self.brain_relaxation, 2),
            "active_bleeding": self.active_bleeding,
            "bleeding_source": self.bleeding_source,
            "bleeding_rate": self.bleeding_rate,
            "brain_swelling": self.brain_swelling,
            "swelling_severity": self.swelling_severity,
            "monitoring": {
                "MEP": round(self.motor_evoked_potentials, 2),
                "SSEP": round(self.somatosensory_evoked_potentials, 2),
                "BAEP": round(self.brainstem_auditory_evoked, 2),
                "facial_EMG": round(self.facial_nerve_emg, 2)
            },
            "monitoring_alerts": self.monitoring_alerts,
            "structures_sacrificed": self.structures_sacrificed,
            "structures_injured": self.structures_injured,
            "temporary_occlusions": {k: round(v, 1) for k, v in self.temporary_occlusions.items()},
            "cumulative_risk_score": round(self.cumulative_risk_score, 3),
            "total_blood_loss_ml": round(self.total_blood_loss_ml, 0),
            "complications": self.complications
        }
    
    def copy(self) -> "PatientState":
        """Create a deep copy for simulation branching."""
        import copy
        return copy.deepcopy(self)


# =============================================================================
# SIMULATION STEP MODEL
# =============================================================================

@dataclass
class SimulationStep:
    """A single step in the surgical simulation."""
    step_number: int
    action: str
    structure: str
    
    # Risk analysis
    risks_assessed: List[RiskAssessment] = field(default_factory=list)
    highest_risk: Optional[RiskLevel] = None
    
    # Outcome
    outcome: str = "completed"
    complications_occurred: List[Dict] = field(default_factory=list)
    
    # State after this step
    state_snapshot: Optional[Dict] = None
    
    # Decision points
    is_decision_point: bool = False
    alternatives: List[str] = field(default_factory=list)
    
    # Reasoning trace
    principles_triggered: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "step": self.step_number,
            "action": self.action,
            "structure": self.structure,
            "risks": [r.to_dict() for r in self.risks_assessed],
            "highest_risk": self.highest_risk.value if self.highest_risk else None,
            "outcome": self.outcome,
            "complications": self.complications_occurred,
            "state": self.state_snapshot,
            "is_decision_point": self.is_decision_point,
            "alternatives": self.alternatives,
            "principles_triggered": self.principles_triggered
        }


# =============================================================================
# SIMULATION RESULT MODEL
# =============================================================================

@dataclass
class SimulationResult:
    """Complete simulation result."""
    id: UUID = field(default_factory=uuid4)
    
    # Input
    approach: str = ""
    target: str = ""
    patient_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Steps
    steps: List[SimulationStep] = field(default_factory=list)
    
    # Final state
    final_state: Optional[PatientState] = None
    
    # Verdict
    verdict: SimulationVerdict = SimulationVerdict.CAUTION
    confidence: float = 0.5
    
    # Analysis
    data_gaps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Critical findings
    critical_steps: List[int] = field(default_factory=list)
    highest_risk_structure: Optional[str] = None
    
    # Timing
    simulation_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "id": str(self.id),
            "approach": self.approach,
            "target": self.target,
            "patient_factors": self.patient_factors,
            "steps": [s.to_dict() for s in self.steps],
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "data_gaps": self.data_gaps,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "critical_steps": self.critical_steps,
            "highest_risk_structure": self.highest_risk_structure,
            "simulation_time_ms": self.simulation_time_ms,
            "created_at": self.created_at.isoformat()
        }


# =============================================================================
# CORRIDOR MODEL
# =============================================================================

@dataclass
class SurgicalCorridor:
    """Surgical corridor definition from database."""
    id: UUID
    name: str
    display_name: str
    approach_type: str
    category: str
    
    # Corridor definition
    structure_sequence: List[str]
    structures_at_risk: List[str]
    critical_steps: List[Dict]
    
    # Requirements
    patient_position: str
    required_monitoring: List[str]
    required_equipment: List[str]
    
    # Indications
    primary_indications: List[str]
    contraindications: List[str]
    
    # Evidence
    evidence_level: str
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "SurgicalCorridor":
        """Create from database row."""
        import json
        
        # Handle critical_steps which might be JSON string
        critical_steps = row.get("critical_steps", [])
        if isinstance(critical_steps, str):
            try:
                critical_steps = json.loads(critical_steps)
            except json.JSONDecodeError:
                critical_steps = []
        
        return cls(
            id=row["id"],
            name=row["name"],
            display_name=row.get("display_name", row["name"]),
            approach_type=row["approach_type"],
            category=row["category"],
            structure_sequence=row.get("structure_sequence", []),
            structures_at_risk=row.get("structures_at_risk", []),
            critical_steps=critical_steps,
            patient_position=row.get("patient_position", "supine"),
            required_monitoring=row.get("required_monitoring", []),
            required_equipment=row.get("required_equipment", []),
            primary_indications=row.get("primary_indications", []),
            contraindications=row.get("contraindications", []),
            evidence_level=row.get("evidence_level", "expert_opinion")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "id": str(self.id),
            "name": self.name,
            "display_name": self.display_name,
            "approach_type": self.approach_type,
            "category": self.category,
            "structure_sequence": self.structure_sequence,
            "structures_at_risk": self.structures_at_risk,
            "critical_steps": self.critical_steps,
            "patient_position": self.patient_position,
            "required_monitoring": self.required_monitoring,
            "required_equipment": self.required_equipment,
            "primary_indications": self.primary_indications,
            "contraindications": self.contraindications,
            "evidence_level": self.evidence_level
        }


# =============================================================================
# CLINICAL PRINCIPLE MODEL
# =============================================================================

@dataclass
class ClinicalPrinciple:
    """Clinical principle from database."""
    id: str
    name: str
    statement: str
    antecedent: str
    consequent: str
    mechanism: Optional[str]
    domain: str
    category: str
    severity: str
    exceptions: List[str]
    examples: List[Dict]
    evidence_level: str
    trigger_entities: List[str]
    trigger_actions: List[str]
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "ClinicalPrinciple":
        """Create from database row."""
        return cls(
            id=row["id"],
            name=row["name"],
            statement=row["statement"],
            antecedent=row["antecedent"],
            consequent=row["consequent"],
            mechanism=row.get("mechanism"),
            domain=row["domain"],
            category=row["category"],
            severity=row.get("severity", "warning"),
            exceptions=row.get("exceptions", []),
            examples=row.get("examples", []),
            evidence_level=row.get("evidence_level", "IV"),
            trigger_entities=row.get("trigger_entities", []),
            trigger_actions=row.get("trigger_actions", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "id": self.id,
            "name": self.name,
            "statement": self.statement,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "mechanism": self.mechanism,
            "domain": self.domain,
            "category": self.category,
            "severity": self.severity,
            "evidence_level": self.evidence_level
        }
