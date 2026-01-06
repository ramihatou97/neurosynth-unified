"""
NeuroSynth 2.0 - Surgical Simulator
====================================

Simulates surgical procedures step-by-step with:
1. Dynamic patient state tracking
2. Causal consequence propagation
3. Counterfactual reasoning ("what if we had...")

This is NOT a visual simulation - it's a logical state machine
that tracks surgical consequences.

Architecture:
- Loads corridor definitions (surgical approach sequences)
- Simulates each step using ClinicalReasoner
- Maintains PatientState across steps
- Propagates complications through causal edges
- Generates actionable recommendations

Usage:
    simulator = SurgicalSimulator(reasoner, database)
    await simulator.initialize()
    
    result = await simulator.simulate_approach(
        approach="pterional",
        target="MCA_aneurysm",
        patient_factors={"age": 65}
    )
"""

import asyncio
import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID, uuid4

from src.neurosynth2.reasoning.models import (
    RiskLevel,
    RiskAssessment,
    PatientState,
    PatientCondition,
    SurgicalContext,
    SimulationStep,
    SimulationResult,
    SimulationVerdict,
    SurgicalCorridor,
    EntityPhysics,
    ComplicationType,
)
from src.neurosynth2.reasoning.clinical_reasoner import ClinicalReasoner

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION INFERENCE
# =============================================================================

class ActionInference:
    """
    Infers appropriate surgical actions for structures in a corridor.
    
    Uses structure properties and position in corridor to determine
    what action is likely to be performed.
    """
    
    # Default actions for structure types
    STRUCTURE_ACTIONS = {
        # Surface structures
        "skin": "incise",
        "scalp": "incise",
        "muscle": "dissect",
        "temporalis": "dissect",
        "periosteum": "elevate",
        
        # Bone
        "bone": "drill",
        "skull": "drill",
        "mastoid": "drill",
        "petrous": "drill",
        "craniotomy": "drill",
        
        # Dura
        "dura": "open",
        "dural": "open",
        
        # Brain/Neural
        "brain": "retract",
        "frontal_lobe": "retract",
        "temporal_lobe": "retract",
        "cerebellum": "retract",
        "cortex": "retract",
        
        # CSF spaces
        "cistern": "open",
        "arachnoid": "dissect",
        "sylvian_fissure": "dissect",
        
        # Vascular
        "artery": "dissect",
        "vein": "preserve",
        "aneurysm": "clip",
        "tumor": "resect",
        "lesion": "resect",
    }
    
    # Actions for structures at risk
    AT_RISK_ACTIONS = {
        "nerve": "preserve",
        "cranial_nerve": "preserve",
        "vein": "preserve",
        "sinus": "preserve",
        "eloquent": "preserve",
    }
    
    @classmethod
    def infer_action(
        cls,
        structure: str,
        target: str,
        corridor: SurgicalCorridor,
        position: int
    ) -> str:
        """
        Infer the surgical action for a structure.
        
        Args:
            structure: Structure name
            target: Target pathology
            corridor: Surgical corridor definition
            position: Position in corridor sequence
            
        Returns:
            Inferred action string
        """
        structure_lower = structure.lower()
        
        # If this is the target, determine treatment action
        if structure_lower == target.lower() or target.lower() in structure_lower:
            if "aneurysm" in target.lower():
                return "clip"
            elif "tumor" in target.lower() or "meningioma" in target.lower():
                return "resect"
            elif "schwannoma" in target.lower():
                return "dissect"
            else:
                return "treat"
        
        # If structure is at risk, default to preserve
        if structure in corridor.structures_at_risk:
            for keyword, action in cls.AT_RISK_ACTIONS.items():
                if keyword in structure_lower:
                    return action
            return "preserve"
        
        # Check critical steps for specific actions
        for step in corridor.critical_steps:
            if step.get("structure", "").lower() == structure_lower:
                return step.get("action", "dissect")
        
        # Match by structure type
        for keyword, action in cls.STRUCTURE_ACTIONS.items():
            if keyword in structure_lower:
                return action
        
        # Default based on position
        if position < len(corridor.structure_sequence) // 3:
            return "expose"  # Early in corridor
        elif position > 2 * len(corridor.structure_sequence) // 3:
            return "dissect"  # Deep in corridor
        else:
            return "mobilize"  # Middle of corridor


# =============================================================================
# COMPLICATION SIMULATOR
# =============================================================================

class ComplicationSimulator:
    """
    Simulates whether complications occur based on risks and state.
    
    Uses deterministic rules rather than random sampling for
    reproducible simulations.
    """
    
    @staticmethod
    def should_complication_occur(
        risk: RiskAssessment,
        state: PatientState,
        context: SurgicalContext
    ) -> Tuple[bool, str]:
        """
        Determine if a complication should occur in simulation.
        
        Uses threshold-based logic for reproducibility.
        
        Args:
            risk: The risk being evaluated
            state: Current patient state
            context: Surgical context
            
        Returns:
            (should_occur, complication_type)
        """
        # Get effective risk level
        multiplier = state.get_risk_multiplier()
        effective_confidence = min(1.0, risk.confidence * multiplier)
        
        # Critical risks with high confidence always manifest in simulation
        if risk.level == RiskLevel.CRITICAL and effective_confidence >= 0.8:
            return True, ComplicationSimulator._determine_complication_type(risk)
        
        # High risks manifest when confidence exceeds threshold
        if risk.level == RiskLevel.HIGH and effective_confidence >= 0.7:
            return True, ComplicationSimulator._determine_complication_type(risk)
        
        # Moderate risks only manifest in compromised states
        if risk.level == RiskLevel.MODERATE:
            if state.condition != PatientCondition.STABLE and effective_confidence >= 0.6:
                return True, ComplicationSimulator._determine_complication_type(risk)
        
        return False, ""
    
    @staticmethod
    def _determine_complication_type(risk: RiskAssessment) -> str:
        """Map risk to complication type."""
        mechanism = risk.mechanism.lower()
        
        if any(word in mechanism for word in ["bleed", "hemorrh", "avulsion"]):
            return ComplicationType.BLEEDING.value
        if any(word in mechanism for word in ["infarct", "ischemia", "stroke"]):
            return ComplicationType.ISCHEMIA.value
        if any(word in mechanism for word in ["nerve", "neural", "palsy"]):
            return ComplicationType.NEURAL_INJURY.value
        if any(word in mechanism for word in ["swell", "edema", "herniat"]):
            return ComplicationType.BRAIN_SWELLING.value
        if any(word in mechanism for word in ["retract", "contusion"]):
            return ComplicationType.RETRACTION_INJURY.value
        if any(word in mechanism for word in ["thermal", "heat", "burn"]):
            return ComplicationType.THERMAL_INJURY.value
        if any(word in mechanism for word in ["csf", "leak"]):
            return ComplicationType.CSF_LEAK.value
            
        return ComplicationType.NEURAL_INJURY.value  # Default


# =============================================================================
# SURGICAL SIMULATOR
# =============================================================================

class SurgicalSimulator:
    """
    Simulates surgical procedures with dynamic state tracking.
    
    Key innovations:
    1. PatientState evolves across steps (bleeding affects visibility)
    2. Complications propagate through causal chains
    3. State-aware risk assessment (risks increase in compromised state)
    4. Generates actionable recommendations
    
    Usage:
        simulator = SurgicalSimulator(reasoner, database)
        await simulator.initialize()
        
        result = await simulator.simulate_approach(
            approach="pterional",
            target="MCA_aneurysm",
            patient_factors={"age": 65}
        )
    """
    
    def __init__(
        self,
        reasoner: ClinicalReasoner,
        database
    ):
        """
        Initialize the simulator.
        
        Args:
            reasoner: ClinicalReasoner instance
            database: Async database connection pool
        """
        self.reasoner = reasoner
        self.db = database
        
        # Corridor cache
        self.corridor_cache: Dict[str, SurgicalCorridor] = {}
        
        # Causal edge cache
        self.causal_edges: Dict[str, List[Dict]] = {}
        
        # State
        self._initialized = False
        self._schema_available = False
    
    async def initialize(self) -> bool:
        """Initialize the simulator."""
        try:
            self._schema_available = await self._check_schema()
            
            if self._schema_available:
                # Pre-load common corridors
                await self._load_corridors()
                
                # Load causal edges
                await self._load_causal_edges()
            
            self._initialized = True
            logger.info(f"SurgicalSimulator initialized: schema_available={self._schema_available}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SurgicalSimulator: {e}")
            self._initialized = True
            return False
    
    async def _check_schema(self) -> bool:
        """Check if simulation schema tables exist."""
        try:
            result = await self.db.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'surgical_corridors'
                )
            """)
            return bool(result)
        except Exception as e:
            logger.warning(f"Schema check failed: {e}")
            return False
    
    async def _load_corridors(self):
        """Load surgical corridors from database."""
        if not self._schema_available:
            return
            
        try:
            rows = await self.db.fetch("""
                SELECT * FROM surgical_corridors
                ORDER BY approach_type, name
            """)
            
            for row in rows:
                corridor = SurgicalCorridor.from_db_row(dict(row))
                self.corridor_cache[corridor.name.lower()] = corridor
                self.corridor_cache[corridor.approach_type.lower()] = corridor
                
            logger.info(f"Loaded {len(rows)} surgical corridors")
            
        except Exception as e:
            logger.error(f"Failed to load corridors: {e}")
    
    async def _load_causal_edges(self):
        """Load causal edges for consequence propagation."""
        if not self._schema_available:
            return
            
        try:
            # Check if causal_edges table exists
            has_table = await self.db.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'causal_edges'
                )
            """)
            
            if not has_table:
                return
                
            rows = await self.db.fetch("""
                SELECT source_entity, target_entity, relation_type,
                       mechanism_chain, probability, effect_magnitude
                FROM causal_edges
                WHERE is_active = TRUE
                ORDER BY probability DESC NULLS LAST
            """)
            
            for row in rows:
                source = row["source_entity"].lower()
                if source not in self.causal_edges:
                    self.causal_edges[source] = []
                self.causal_edges[source].append(dict(row))
                
            logger.info(f"Loaded causal edges for {len(self.causal_edges)} source entities")
            
        except Exception as e:
            logger.warning(f"Failed to load causal edges: {e}")
    
    async def get_corridor(self, approach: str) -> Optional[SurgicalCorridor]:
        """
        Get surgical corridor definition.
        
        Args:
            approach: Approach name or type
            
        Returns:
            SurgicalCorridor if found
        """
        approach_lower = approach.lower()
        
        # Check cache
        if approach_lower in self.corridor_cache:
            return self.corridor_cache[approach_lower]
        
        # Query database
        if self._schema_available:
            try:
                row = await self.db.fetchrow("""
                    SELECT * FROM surgical_corridors
                    WHERE LOWER(name) = $1 OR LOWER(approach_type) = $1
                """, approach_lower)
                
                if row:
                    corridor = SurgicalCorridor.from_db_row(dict(row))
                    self.corridor_cache[approach_lower] = corridor
                    return corridor
                    
            except Exception as e:
                logger.warning(f"Corridor lookup failed for '{approach}': {e}")
        
        return None
    
    async def simulate_approach(
        self,
        approach: str,
        target: str,
        patient_factors: Optional[Dict[str, Any]] = None,
        max_steps: int = 50
    ) -> SimulationResult:
        """
        Simulate a complete surgical approach.
        
        This is the main entry point for simulation.
        
        Args:
            approach: Surgical approach name
            target: Target pathology/structure
            patient_factors: Patient-specific variables
            max_steps: Maximum simulation steps (safety limit)
            
        Returns:
            Complete simulation result with step-by-step analysis
        """
        start_time = time.time()
        
        # Initialize result
        result = SimulationResult(
            approach=approach,
            target=target,
            patient_factors=patient_factors or {}
        )
        
        # Load corridor definition
        corridor = await self.get_corridor(approach)
        
        if corridor is None:
            return self._create_no_corridor_result(approach, target, patient_factors)
        
        # Initialize patient state
        state = PatientState()
        
        # Initialize surgical context
        context = SurgicalContext(
            approach=approach,
            target_structure=target,
            current_step=0,
            patient_factors=patient_factors or {}
        )
        
        # Add required monitoring to context
        context.monitoring_active = set(corridor.required_monitoring)
        
        # Track data gaps
        data_gaps: List[str] = []
        
        # =================================================================
        # SIMULATE EACH STEP
        # =================================================================
        
        steps: List[SimulationStep] = []
        
        for i, structure in enumerate(corridor.structure_sequence[:max_steps]):
            context.current_step = i
            
            # Determine action for this structure
            action = ActionInference.infer_action(
                structure, target, corridor, i
            )
            
            # Get entity physics
            entity = await self.reasoner.get_entity_physics(structure)
            
            # Track data gaps
            if entity is None:
                data_gaps.append(f"entity_physics:{structure}")
            elif entity.confidence < 0.5:
                data_gaps.append(f"low_confidence:{structure}:{entity.confidence:.2f}")
            
            # =============================================================
            # ASSESS RISKS (with current state)
            # =============================================================
            
            risks = await self.reasoner.assess_action(
                action=action,
                structure=structure,
                context=context,
                state=state
            )
            
            # Get highest risk
            highest_risk = max(risks, key=lambda r: r.level.value) if risks else None
            highest_level = highest_risk.level if highest_risk else None
            
            # Track triggered principles
            triggered_principles = [r.principle_id for r in risks]
            
            # =============================================================
            # SIMULATE OUTCOME
            # =============================================================
            
            outcome = "completed"
            complications = []
            
            for risk in risks:
                should_occur, comp_type = ComplicationSimulator.should_complication_occur(
                    risk, state, context
                )
                
                if should_occur:
                    # Determine severity
                    severity = self._determine_complication_severity(risk, state)
                    
                    # Apply complication to state
                    state.apply_complication(
                        complication_type=comp_type,
                        severity=severity,
                        source=structure,
                        details={"risk": risk.to_dict()}
                    )
                    
                    complications.append({
                        "type": comp_type,
                        "severity": severity,
                        "source": structure,
                        "principle": risk.principle_id
                    })
                    
                    outcome = f"COMPLICATION: {comp_type} ({severity})"
            
            # Update context based on action
            self._update_context_after_action(context, action, structure)
            
            # =============================================================
            # RECORD STEP
            # =============================================================
            
            step = SimulationStep(
                step_number=i,
                action=action,
                structure=structure,
                risks_assessed=risks,
                highest_risk=highest_level,
                outcome=outcome,
                complications_occurred=complications,
                state_snapshot=state.to_dict(),
                is_decision_point=(
                    highest_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) if highest_level else False
                ),
                alternatives=self._get_alternatives(action, structure, risks),
                principles_triggered=triggered_principles
            )
            
            steps.append(step)
            
            # =============================================================
            # CHECK ABORT CONDITIONS
            # =============================================================
            
            if state.condition == PatientCondition.CRITICAL:
                result.warnings.append(
                    f"Simulation aborted at step {i}: Patient condition CRITICAL"
                )
                break
            
            # Check for simulation-ending complications
            if self._should_abort_simulation(state, complications):
                result.warnings.append(
                    f"Simulation ended at step {i} due to severe complication"
                )
                break
        
        # =================================================================
        # FINALIZE RESULT
        # =================================================================
        
        result.steps = steps
        result.final_state = state
        result.data_gaps = data_gaps
        
        # Generate verdict
        result.verdict, result.confidence = self._generate_verdict(steps, state, data_gaps)
        
        # Identify critical steps
        result.critical_steps = [
            s.step_number for s in steps 
            if s.is_decision_point or s.complications_occurred
        ]
        
        # Risk level ordering for comparison
        risk_order = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
            None: -1
        }
        
        # Find highest risk structure
        max_risk_step = max(
            steps, 
            key=lambda s: risk_order.get(s.highest_risk, -1),
            default=None
        )
        if max_risk_step and max_risk_step.highest_risk:
            result.highest_risk_structure = max_risk_step.structure
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(steps, state, corridor)
        
        # Generate warnings
        result.warnings.extend(self._generate_warnings(steps, state, data_gaps))
        
        # Timing
        result.simulation_time_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    def _create_no_corridor_result(
        self,
        approach: str,
        target: str,
        patient_factors: Optional[Dict]
    ) -> SimulationResult:
        """Create result when corridor definition is missing."""
        return SimulationResult(
            approach=approach,
            target=target,
            patient_factors=patient_factors or {},
            verdict=SimulationVerdict.CAUTION,
            confidence=0.3,
            data_gaps=[f"corridor_definition:{approach}"],
            warnings=[
                f"No corridor definition found for approach '{approach}'. "
                "Simulation cannot proceed without structure sequence."
            ],
            recommendations=[
                "Add corridor definition to surgical_corridors table",
                "Consider using a known approach with existing definition"
            ]
        )
    
    def _determine_complication_severity(
        self,
        risk: RiskAssessment,
        state: PatientState
    ) -> str:
        """Determine complication severity based on risk and state."""
        base_severity_map = {
            RiskLevel.CRITICAL: "severe",
            RiskLevel.HIGH: "moderate",
            RiskLevel.MODERATE: "minimal",
            RiskLevel.LOW: "minimal",
            RiskLevel.MINIMAL: "minimal"
        }
        
        severity = base_severity_map.get(risk.level, "moderate")
        
        # Upgrade severity if state is compromised
        if state.condition in (PatientCondition.MODERATELY_COMPROMISED, 
                               PatientCondition.SEVERELY_COMPROMISED):
            if severity == "minimal":
                severity = "moderate"
            elif severity == "moderate":
                severity = "severe"
        
        return severity
    
    def _update_context_after_action(
        self,
        context: SurgicalContext,
        action: str,
        structure: str
    ):
        """Update surgical context after performing an action."""
        if action in ("expose", "open", "incise"):
            context.structures_exposed.add(structure)
        elif action in ("mobilize", "retract"):
            context.structures_mobilized.add(structure)
            context.structures_retracted.add(structure)
        elif action == "sacrifice":
            context.vessels_sacrificed.add(structure)
        elif action == "temporary_occlusion":
            context.vessels_temporarily_occluded.add(structure)
    
    def _get_alternatives(
        self,
        action: str,
        structure: str,
        risks: List[RiskAssessment]
    ) -> List[str]:
        """Get alternative approaches when risks are high."""
        alternatives = []
        
        high_risks = [r for r in risks if r.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        
        for risk in high_risks:
            if risk.mitigation:
                alternatives.append(risk.mitigation)
                
            # Action-specific alternatives
            if action == "retract":
                alternatives.append("Consider releasing tethering structures first")
                alternatives.append("Use dynamic hand-held retraction instead of fixed")
            elif action == "sacrifice":
                alternatives.append("Evaluate collateral capacity with ICG/Doppler")
                alternatives.append("Consider alternative corridor that preserves this structure")
            elif action == "coagulate":
                alternatives.append("Use irrigation and lowest effective power")
                alternatives.append("Consider mechanical hemostasis instead")
        
        # Deduplicate
        return list(set(alternatives))
    
    def _should_abort_simulation(
        self,
        state: PatientState,
        complications: List[Dict]
    ) -> bool:
        """Determine if simulation should abort due to complications."""
        # Abort on critical complications
        for comp in complications:
            if comp.get("severity") == "severe" and comp.get("type") in (
                ComplicationType.ISCHEMIA.value,
                ComplicationType.NEURAL_INJURY.value
            ):
                return True
        
        # Abort on massive blood loss
        if state.total_blood_loss_ml > 1500:
            return True
        
        # Abort on complete visibility loss
        if state.visibility < 0.1:
            return True
        
        return False
    
    def _generate_verdict(
        self,
        steps: List[SimulationStep],
        final_state: PatientState,
        data_gaps: List[str]
    ) -> Tuple[SimulationVerdict, float]:
        """Generate overall verdict with confidence."""
        
        # Count critical events
        critical_steps = sum(
            1 for s in steps 
            if s.highest_risk == RiskLevel.CRITICAL
        )
        high_risk_steps = sum(
            1 for s in steps 
            if s.highest_risk == RiskLevel.HIGH
        )
        complication_steps = sum(
            1 for s in steps 
            if s.complications_occurred
        )
        
        # Base verdict on outcomes
        if final_state.condition == PatientCondition.CRITICAL:
            verdict = SimulationVerdict.CONTRAINDICATED
            confidence = 0.9
        elif final_state.condition == PatientCondition.SEVERELY_COMPROMISED:
            verdict = SimulationVerdict.HIGH_RISK
            confidence = 0.85
        elif complication_steps >= 2 or critical_steps >= 1:
            verdict = SimulationVerdict.HIGH_RISK
            confidence = 0.8
        elif high_risk_steps >= 3 or complication_steps >= 1:
            verdict = SimulationVerdict.CAUTION
            confidence = 0.75
        elif high_risk_steps >= 1:
            verdict = SimulationVerdict.CAUTION
            confidence = 0.7
        else:
            verdict = SimulationVerdict.SAFE
            confidence = 0.85
        
        # Reduce confidence for data gaps
        gap_penalty = len(data_gaps) * 0.03
        confidence = max(0.3, confidence - gap_penalty)
        
        return verdict, confidence
    
    def _generate_recommendations(
        self,
        steps: List[SimulationStep],
        state: PatientState,
        corridor: SurgicalCorridor
    ) -> List[str]:
        """Generate surgical recommendations from simulation."""
        recommendations = []
        
        # Monitoring recommendations
        if state.motor_evoked_potentials < 1.0:
            recommendations.append("MEP monitoring essential - baseline changes detected")
        if state.facial_nerve_emg < 1.0:
            recommendations.append("Facial nerve EMG monitoring essential")
            
        # Add required monitoring from corridor
        if corridor.required_monitoring:
            recommendations.append(
                f"Required monitoring: {', '.join(corridor.required_monitoring)}"
            )
        
        # Decision point summaries
        decision_points = [s for s in steps if s.is_decision_point]
        for step in decision_points[:3]:  # Top 3 critical points
            recommendations.append(
                f"CRITICAL STEP {step.step_number}: {step.action} {step.structure} - "
                f"verify before proceeding"
            )
        
        # State-based recommendations
        if state.cumulative_risk_score > 0.5:
            recommendations.append(
                "High cumulative risk - consider staged procedure or alternative approach"
            )
        
        # Specific mitigations from high-risk steps
        for step in steps:
            for risk in step.risks_assessed:
                if risk.level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and risk.mitigation:
                    recommendations.append(f"Step {step.step_number}: {risk.mitigation}")
        
        # Deduplicate and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]
    
    def _generate_warnings(
        self,
        steps: List[SimulationStep],
        state: PatientState,
        data_gaps: List[str]
    ) -> List[str]:
        """Generate warnings about simulation limitations."""
        warnings = []
        
        # Data gap warnings
        entity_gaps = [g for g in data_gaps if g.startswith("entity_physics:")]
        if entity_gaps:
            warnings.append(
                f"Missing physics data for {len(entity_gaps)} structures. "
                "Risk assessment may be incomplete."
            )
        
        confidence_gaps = [g for g in data_gaps if g.startswith("low_confidence:")]
        if confidence_gaps:
            warnings.append(
                f"{len(confidence_gaps)} structures have low-confidence physics data. "
                "Consider expert verification."
            )
        
        # State warnings
        if state.active_bleeding:
            warnings.append(
                f"Active bleeding from {state.bleeding_source}. "
                "Address hemostasis before proceeding."
            )
        
        if state.brain_swelling:
            warnings.append(
                f"Brain swelling detected ({state.swelling_severity}). "
                "Consider additional relaxation measures."
            )
        
        return warnings
    
    async def simulate_counterfactual(
        self,
        base_result: SimulationResult,
        modification: Dict[str, Any]
    ) -> SimulationResult:
        """
        Simulate a counterfactual scenario.
        
        "What if we had taken a different action at step X?"
        
        Args:
            base_result: Original simulation result
            modification: {
                "step": step_number,
                "action": alternative_action,
                "structure": optional_alternative_structure
            }
            
        Returns:
            New simulation result with modified trajectory
        """
        # Get original corridor
        corridor = await self.get_corridor(base_result.approach)
        if corridor is None:
            return base_result
        
        # Create modified structure sequence
        mod_step = modification.get("step", 0)
        mod_structure = modification.get("structure")
        
        if mod_structure:
            # Replace structure at step
            new_sequence = list(corridor.structure_sequence)
            if mod_step < len(new_sequence):
                new_sequence[mod_step] = mod_structure
                
            # Create modified corridor
            modified_corridor = SurgicalCorridor(
                id=corridor.id,
                name=f"{corridor.name}_counterfactual",
                display_name=f"{corridor.display_name} (Counterfactual)",
                approach_type=corridor.approach_type,
                category=corridor.category,
                structure_sequence=new_sequence,
                structures_at_risk=corridor.structures_at_risk,
                critical_steps=corridor.critical_steps,
                patient_position=corridor.patient_position,
                required_monitoring=corridor.required_monitoring,
                required_equipment=corridor.required_equipment,
                primary_indications=corridor.primary_indications,
                contraindications=corridor.contraindications,
                evidence_level=corridor.evidence_level
            )
            
            # Cache temporarily
            temp_name = f"_cf_{base_result.approach}_{mod_step}"
            self.corridor_cache[temp_name] = modified_corridor
            
            # Run simulation
            result = await self.simulate_approach(
                approach=temp_name,
                target=base_result.target,
                patient_factors=base_result.patient_factors
            )
            
            # Clean up
            del self.corridor_cache[temp_name]
            
            return result
        
        return base_result
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get simulator statistics."""
        return {
            "initialized": self._initialized,
            "schema_available": self._schema_available,
            "corridors_cached": len(self.corridor_cache),
            "causal_edges_loaded": len(self.causal_edges),
            "corridor_names": list(self.corridor_cache.keys())
        }
