"""
NeuroSynth 2.0 - Clinical Reasoner
===================================

Rule-based clinical reasoning engine that evaluates surgical actions
against clinical principles and entity physics properties.

This is NOT a probabilistic model - it's a deterministic expert system
that applies medical axioms to surgical scenarios.

Architecture:
- Loads clinical principles from database
- Caches entity physics properties
- Evaluates actions against applicable principles
- Returns risk assessments with evidence trails

Usage:
    reasoner = ClinicalReasoner(database)
    await reasoner.initialize()
    
    risks = await reasoner.assess_action(
        action="retract",
        structure="temporal_lobe",
        context=SurgicalContext(approach="pterional", ...)
    )
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from src.neurosynth2.reasoning.models import (
    RiskLevel,
    RiskAssessment,
    SurgicalContext,
    EntityPhysics,
    ClinicalPrinciple,
    PatientState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BUILT-IN RULES (Always Active)
# =============================================================================

class BuiltInRules:
    """
    Hard-coded rules that are always evaluated.
    
    These capture fundamental surgical safety principles that
    should never be disabled or modified.
    """
    
    @staticmethod
    def assess_tethered_retraction(
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> Optional[RiskAssessment]:
        """
        Rule: Tethered structures cannot be safely retracted.
        
        Applies when:
        - Entity mobility is 'tethered_by_*'
        - Action is 'retract' or 'mobilize'
        """
        if action not in ("retract", "mobilize", "displace"):
            return None
            
        if not entity.mobility.startswith("tethered"):
            return None
            
        # Determine tethering type for specific guidance
        tether_type = entity.mobility.replace("tethered_by_", "")
        
        mechanism_map = {
            "nerve": f"{entity.name} is tethered by neural structures. Retraction may cause stretch injury to the tethering nerve.",
            "vessel": f"{entity.name} is tethered by vascular structures. Retraction risks vessel avulsion.",
            "perforators": f"{entity.name} is tethered by perforating arteries. Mobilization risks perforator avulsion causing stroke."
        }
        
        mitigation_map = {
            "nerve": "Identify and release the tethering nerve before mobilization.",
            "vessel": "Dissect and release the tethering vessel with sharp technique.",
            "perforators": "Perform careful arachnoid dissection to release perforators before parent vessel mobilization."
        }
        
        return RiskAssessment(
            level=RiskLevel.HIGH,
            structure=entity.name,
            action=action,
            principle_id="BUILTIN_TETHER",
            principle_name="Tethered Structure Retraction Risk",
            mechanism=mechanism_map.get(tether_type, f"{entity.name} is tethered ({entity.mobility}). Retraction risks avulsion injury."),
            mitigation=mitigation_map.get(tether_type, "Identify and release tethering structures before mobilization."),
            confidence=0.95
        )
    
    @staticmethod
    def assess_end_artery_sacrifice(
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> Optional[RiskAssessment]:
        """
        Rule: End artery sacrifice causes infarction.
        
        Applies when:
        - Entity is marked as end artery
        - Action is 'sacrifice', 'coagulate', 'clip', or 'occlude'
        """
        if action not in ("sacrifice", "coagulate", "clip", "occlude", "ligate"):
            return None
            
        if not entity.is_end_artery:
            return None
            
        # Check collateral status
        collaterals = entity.collateral_capacity or "none"
        
        if collaterals in ("none", "poor"):
            level = RiskLevel.CRITICAL
            mechanism = (
                f"{entity.name} is an end artery with {collaterals} collateral capacity. "
                f"Sacrifice will cause infarction in the supplied territory "
                f"({', '.join(entity.territory_supplied) if entity.territory_supplied else 'downstream region'})."
            )
            mitigation = "Preserve at all costs. Consider alternative surgical approach or corridor."
        elif collaterals == "variable":
            level = RiskLevel.HIGH
            mechanism = (
                f"{entity.name} has variable collateral capacity. "
                "Sacrifice may cause infarction depending on patient's collateral development."
            )
            mitigation = "Perform intraoperative flow assessment (ICG, Doppler) before any sacrifice decision."
        else:  # moderate or rich
            level = RiskLevel.MODERATE
            mechanism = (
                f"{entity.name} has {collaterals} collaterals. "
                "Sacrifice may be tolerated but carries risk of territory compromise."
            )
            mitigation = "Consider temporary occlusion test before permanent sacrifice."
        
        return RiskAssessment(
            level=level,
            structure=entity.name,
            action=action,
            principle_id="BUILTIN_ENDARTERY",
            principle_name="End-Artery Vulnerability",
            mechanism=mechanism,
            mitigation=mitigation,
            confidence=0.95 if collaterals in ("none", "poor") else 0.8
        )
    
    @staticmethod
    def assess_eloquent_resection(
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> Optional[RiskAssessment]:
        """
        Rule: Eloquent cortex resection causes permanent deficit.
        
        Applies when:
        - Entity eloquence_grade is 'eloquent'
        - Action is 'resect', 'ablate', or 'destroy'
        """
        if action not in ("resect", "ablate", "destroy", "remove"):
            return None
            
        if entity.eloquence_grade != "eloquent":
            return None
            
        functional_role = entity.spatial_context.get("functional_role", "functional tissue")
        
        return RiskAssessment(
            level=RiskLevel.CRITICAL,
            structure=entity.name,
            action=action,
            principle_id="BUILTIN_ELOQUENT",
            principle_name="Eloquent Cortex Preservation",
            mechanism=(
                f"{entity.name} is eloquent cortex ({functional_role}). "
                "Resection will cause permanent neurological deficit."
            ),
            mitigation=(
                "Avoid resection. Consider: (1) Subtotal resection sparing eloquent tissue, "
                "(2) Awake craniotomy with mapping, (3) Alternative treatment (radiosurgery, chemotherapy)."
            ),
            confidence=0.98
        )
    
    @staticmethod
    def assess_thermal_injury_risk(
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> Optional[RiskAssessment]:
        """
        Rule: Coagulation near neural tissue risks thermal injury.
        
        Applies when:
        - Action is 'coagulate' or 'cauterize'
        - Entity is neural or near neural structures
        """
        if action not in ("coagulate", "cauterize", "bipolar"):
            return None
            
        if entity.coagulation_tolerance not in ("none", "minimal"):
            return None
            
        if entity.consistency != "neural":
            return None
            
        return RiskAssessment(
            level=RiskLevel.MODERATE,
            structure=entity.name,
            action=action,
            principle_id="BUILTIN_THERMAL",
            principle_name="Thermal Spread in Bipolar Coagulation",
            mechanism=(
                f"{entity.name} is neural tissue with minimal coagulation tolerance. "
                "Bipolar heat spreads beyond forceps tips, risking thermal injury."
            ),
            mitigation=(
                "Use lowest effective power setting. Apply copious irrigation. "
                "Brief intermittent application. Avoid prolonged coagulation near neural structures."
            ),
            confidence=0.85
        )
    
    @staticmethod
    def assess_poor_visibility_risk(
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext,
        state: Optional[PatientState] = None
    ) -> Optional[RiskAssessment]:
        """
        Rule: Actions in poor visibility carry increased risk.
        
        Applies when:
        - Patient state indicates poor visibility (bleeding, inadequate exposure)
        - Any surgical action is being performed
        """
        if state is None:
            return None
            
        if state.visibility >= 0.7:
            return None
            
        # Only warn for precision actions
        precision_actions = {
            "dissect", "coagulate", "clip", "cut", "resect",
            "mobilize", "dissect_capsule"
        }
        if action not in precision_actions:
            return None
            
        if state.visibility < 0.3:
            level = RiskLevel.HIGH
            visibility_desc = "severely compromised"
        elif state.visibility < 0.5:
            level = RiskLevel.MODERATE
            visibility_desc = "compromised"
        else:
            level = RiskLevel.LOW
            visibility_desc = "suboptimal"
            
        cause = ""
        if state.active_bleeding:
            cause = f" due to active bleeding from {state.bleeding_source}"
        elif state.brain_swelling:
            cause = " due to brain swelling"
            
        return RiskAssessment(
            level=level,
            structure=entity.name,
            action=action,
            principle_id="BUILTIN_VISIBILITY",
            principle_name="Poor Visibility Risk Amplification",
            mechanism=(
                f"Visibility is {visibility_desc} ({state.visibility:.0%}){cause}. "
                f"Precise {action} of {entity.name} carries increased risk of unintended injury."
            ),
            mitigation=(
                "Address visibility first: control bleeding, improve exposure, "
                "consider temporary pause for hemostasis."
            ),
            confidence=0.9
        )


# =============================================================================
# CLINICAL REASONER
# =============================================================================

class ClinicalReasoner:
    """
    Rule-based clinical reasoning engine.
    
    Evaluates surgical actions against clinical principles
    and entity physics properties.
    
    Architecture:
    1. Built-in rules: Always-on safety rules
    2. Database principles: Loaded from clinical_principles table
    3. Entity physics: Loaded from anatomical_entities table
    
    Usage:
        reasoner = ClinicalReasoner(database)
        await reasoner.initialize()
        
        risks = await reasoner.assess_action(
            action="retract",
            structure="temporal_lobe",
            context=SurgicalContext(...)
        )
    """
    
    def __init__(self, database):
        """
        Initialize the reasoner.
        
        Args:
            database: Async database connection pool
        """
        self.db = database
        
        # Principle storage
        self.principles: Dict[str, ClinicalPrinciple] = {}
        self.principles_by_domain: Dict[str, List[str]] = defaultdict(list)
        self.principles_by_trigger: Dict[str, Set[str]] = defaultdict(set)
        
        # Entity cache
        self.entity_cache: Dict[str, EntityPhysics] = {}
        self.entity_aliases: Dict[str, str] = {}  # alias -> canonical name
        
        # Initialization state
        self._initialized = False
        self._principles_loaded = 0
        self._schema_available = False
        
    async def initialize(self) -> bool:
        """
        Initialize the reasoner by loading principles and checking schema.
        
        Returns:
            True if initialization successful
        """
        try:
            # Check schema availability
            self._schema_available = await self._check_schema()
            
            if self._schema_available:
                # Load clinical principles
                await self._load_principles()
                
                # Pre-warm entity cache with common structures
                await self._prewarm_entity_cache()
                
            self._initialized = True
            logger.info(
                f"ClinicalReasoner initialized: "
                f"{self._principles_loaded} principles loaded, "
                f"schema_available={self._schema_available}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ClinicalReasoner: {e}")
            self._initialized = True  # Allow degraded operation
            return False
    
    async def _check_schema(self) -> bool:
        """Check if reasoning schema tables exist."""
        try:
            result = await self.db.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'clinical_principles'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'anatomical_entities'
                )
            """)
            return bool(result)
        except Exception as e:
            logger.warning(f"Schema check failed: {e}")
            return False
    
    async def _load_principles(self):
        """Load all active clinical principles from database."""
        if not self._schema_available:
            return
            
        try:
            rows = await self.db.fetch("""
                SELECT id, name, statement, antecedent, consequent, mechanism,
                       domain, category, severity, exceptions, examples,
                       evidence_level, trigger_entities, trigger_actions
                FROM clinical_principles
                WHERE is_active = TRUE
                ORDER BY severity DESC, id
            """)
            
            for row in rows:
                principle = ClinicalPrinciple.from_db_row(dict(row))
                self.principles[principle.id] = principle
                
                # Index by domain
                self.principles_by_domain[principle.domain].append(principle.id)
                
                # Index by trigger entities
                for entity in principle.trigger_entities:
                    self.principles_by_trigger[entity.lower()].add(principle.id)
                    
                # Index by trigger actions
                for action in principle.trigger_actions:
                    self.principles_by_trigger[f"action:{action.lower()}"].add(principle.id)
            
            self._principles_loaded = len(self.principles)
            logger.info(f"Loaded {self._principles_loaded} clinical principles")
            
        except Exception as e:
            logger.error(f"Failed to load principles: {e}")
    
    async def _prewarm_entity_cache(self):
        """Pre-load commonly referenced entities."""
        if not self._schema_available:
            return
            
        try:
            # Load high-confidence entities
            rows = await self.db.fetch("""
                SELECT * FROM anatomical_entities
                WHERE confidence >= 0.7
                ORDER BY confidence DESC
                LIMIT 100
            """)
            
            for row in rows:
                entity = EntityPhysics.from_db_row(dict(row))
                self.entity_cache[entity.name.lower()] = entity
                
                # Also cache by canonical name
                if entity.canonical_name != entity.name:
                    self.entity_cache[entity.canonical_name.lower()] = entity
                    
            logger.info(f"Pre-warmed entity cache with {len(rows)} entities")
            
        except Exception as e:
            logger.warning(f"Entity cache prewarm failed: {e}")
    
    async def get_entity_physics(self, entity_name: str) -> Optional[EntityPhysics]:
        """
        Get physics properties for an entity (with caching).
        
        Args:
            entity_name: Name of the anatomical entity
            
        Returns:
            EntityPhysics if found, None otherwise
        """
        # Normalize name
        name_lower = entity_name.lower().strip()
        
        # Check cache first
        if name_lower in self.entity_cache:
            return self.entity_cache[name_lower]
            
        # Check aliases
        if name_lower in self.entity_aliases:
            canonical = self.entity_aliases[name_lower]
            if canonical in self.entity_cache:
                return self.entity_cache[canonical]
                
        # Query database
        if not self._schema_available:
            return None
            
        try:
            row = await self.db.fetchrow("""
                SELECT * FROM anatomical_entities
                WHERE LOWER(name) = $1 
                   OR LOWER(canonical_name) = $1
                   OR $1 = ANY(SELECT LOWER(unnest(aliases)))
            """, name_lower)
            
            if row:
                entity = EntityPhysics.from_db_row(dict(row))
                self.entity_cache[name_lower] = entity
                return entity
                
            return None
            
        except Exception as e:
            logger.warning(f"Entity lookup failed for '{entity_name}': {e}")
            return None
    
    async def assess_action(
        self,
        action: str,
        structure: str,
        context: Optional[SurgicalContext] = None,
        state: Optional[PatientState] = None
    ) -> List[RiskAssessment]:
        """
        Assess risks of performing an action on a structure.
        
        This is the main entry point for reasoning. It:
        1. Retrieves entity physics properties
        2. Applies built-in safety rules
        3. Evaluates applicable clinical principles
        4. Considers current patient state
        
        Args:
            action: Surgical action (retract, dissect, coagulate, etc.)
            structure: Anatomical structure name
            context: Current surgical context (optional)
            state: Current patient state for state-aware assessment (optional)
            
        Returns:
            List of RiskAssessment objects, sorted by severity (highest first)
        """
        risks: List[RiskAssessment] = []
        
        # Get entity physics
        entity = await self.get_entity_physics(structure)
        
        if not entity:
            logger.debug(f"No physics data for entity: {structure}")
            # Create minimal entity for rule evaluation
            entity = EntityPhysics(
                name=structure,
                canonical_name=structure,
                confidence=0.0
            )
            risks.append(RiskAssessment(
                level=RiskLevel.LOW,
                structure=structure,
                action=action,
                principle_id="DATA_GAP",
                principle_name="Incomplete Entity Data",
                mechanism=f"No physics properties available for {structure}. Risk assessment may be incomplete.",
                mitigation="Consider adding physics properties for this structure to improve reasoning.",
                confidence=0.3
            ))
        
        # Create default context if not provided
        if context is None:
            context = SurgicalContext(
                approach="unknown",
                target_structure=structure
            )
        
        # =================================================================
        # 1. Apply Built-in Rules (always evaluated)
        # =================================================================
        
        builtin_assessments = [
            BuiltInRules.assess_tethered_retraction(entity, action, context),
            BuiltInRules.assess_end_artery_sacrifice(entity, action, context),
            BuiltInRules.assess_eloquent_resection(entity, action, context),
            BuiltInRules.assess_thermal_injury_risk(entity, action, context),
            BuiltInRules.assess_poor_visibility_risk(entity, action, context, state),
        ]
        
        for assessment in builtin_assessments:
            if assessment is not None:
                risks.append(assessment)
        
        # =================================================================
        # 2. Evaluate Database Principles
        # =================================================================
        
        # Find potentially applicable principles
        candidate_principles = self._find_candidate_principles(
            entity, action, context
        )
        
        for principle_id in candidate_principles:
            principle = self.principles.get(principle_id)
            if principle is None:
                continue
                
            # Check if principle applies
            if self._principle_applies(principle, entity, action, context):
                assessment = self._create_risk_from_principle(
                    principle, entity, action
                )
                risks.append(assessment)
        
        # =================================================================
        # 3. Apply State-Based Risk Amplification
        # =================================================================
        
        if state is not None:
            risks = self._apply_state_modifiers(risks, state)
        
        # =================================================================
        # 4. Deduplicate and Sort
        # =================================================================
        
        # Remove duplicates (same principle_id)
        seen_principles = set()
        unique_risks = []
        for risk in risks:
            if risk.principle_id not in seen_principles:
                seen_principles.add(risk.principle_id)
                unique_risks.append(risk)
        
        # Sort by severity (highest first)
        unique_risks.sort(reverse=True)
        
        return unique_risks
    
    def _find_candidate_principles(
        self,
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> Set[str]:
        """
        Find principles that might apply based on triggers.
        
        Uses the trigger indexes built during principle loading
        for efficient lookup.
        """
        candidates = set()
        
        # Match by entity name
        candidates.update(self.principles_by_trigger.get(entity.name.lower(), set()))
        candidates.update(self.principles_by_trigger.get(entity.canonical_name.lower(), set()))
        
        # Match by action
        candidates.update(self.principles_by_trigger.get(f"action:{action.lower()}", set()))
        
        # Match by entity properties
        if entity.is_end_artery:
            candidates.update(self.principles_by_trigger.get("end_artery", set()))
            candidates.update(self.principles_by_trigger.get("perforator", set()))
            
        if entity.consistency == "vascular":
            candidates.update(self.principles_by_trigger.get("artery", set()))
            candidates.update(self.principles_by_trigger.get("vein", set()))
            candidates.update(self.principles_by_trigger.get("vessel", set()))
            
        if entity.consistency == "neural":
            candidates.update(self.principles_by_trigger.get("nerve", set()))
            candidates.update(self.principles_by_trigger.get("cranial_nerve", set()))
            
        return candidates
    
    def _principle_applies(
        self,
        principle: ClinicalPrinciple,
        entity: EntityPhysics,
        action: str,
        context: SurgicalContext
    ) -> bool:
        """
        Check if a principle's antecedent is satisfied.
        
        Uses pattern matching on the antecedent string.
        Production systems might use a proper rule engine here.
        """
        antecedent = principle.antecedent.lower()
        
        # Check action matches
        if "action" in antecedent:
            # Extract actions from patterns like "action IN ('sacrifice', 'coagulate')"
            action_match = re.search(r"action\s*(?:IN|=)\s*\(?'?([^')\s]+)'?\)?", antecedent, re.IGNORECASE)
            if action_match:
                allowed_actions = [a.strip().strip("'\"") for a in action_match.group(1).split(",")]
                if action.lower() not in [a.lower() for a in allowed_actions]:
                    return False
                    
        # Check entity property matches
        property_patterns = [
            (r"mobility\s*=\s*'([^']+)'", "mobility"),
            (r"consistency\s*=\s*'([^']+)'", "consistency"),
            (r"is_end_artery\s*=\s*(true|false)", "is_end_artery"),
            (r"eloquence_grade\s*=\s*'([^']+)'", "eloquence_grade"),
        ]
        
        for pattern, prop_name in property_patterns:
            match = re.search(pattern, antecedent, re.IGNORECASE)
            if match:
                expected_value = match.group(1).lower()
                actual_value = str(getattr(entity, prop_name, "")).lower()
                
                if expected_value == "true":
                    if not getattr(entity, prop_name, False):
                        return False
                elif expected_value == "false":
                    if getattr(entity, prop_name, False):
                        return False
                elif expected_value not in actual_value and actual_value not in expected_value:
                    return False
        
        # Check for exclusions
        for exception in principle.exceptions:
            exception_lower = exception.lower()
            
            # Check if patient has exception condition
            patient_conditions = context.patient_factors.get("conditions", [])
            if exception_lower in [c.lower() for c in patient_conditions]:
                return False
                
            # Check if exception matches context
            if exception_lower in context.approach.lower():
                return False
        
        return True
    
    def _create_risk_from_principle(
        self,
        principle: ClinicalPrinciple,
        entity: EntityPhysics,
        action: str
    ) -> RiskAssessment:
        """Create a RiskAssessment from a matched principle."""
        
        # Map severity to RiskLevel
        severity_map = {
            "advisory": RiskLevel.LOW,
            "warning": RiskLevel.MODERATE,
            "critical": RiskLevel.HIGH,
            "absolute": RiskLevel.CRITICAL
        }
        level = severity_map.get(principle.severity, RiskLevel.MODERATE)
        
        # Build mechanism text
        mechanism = principle.mechanism or principle.consequent
        
        # Substitute entity name if present
        mechanism = mechanism.replace("{structure}", entity.name)
        mechanism = mechanism.replace("{entity}", entity.name)
        
        # Derive mitigation from examples if available
        mitigation = None
        for example in principle.examples:
            if "mitigation" in example:
                mitigation = example["mitigation"]
                break
        
        return RiskAssessment(
            level=level,
            structure=entity.name,
            action=action,
            principle_id=principle.id,
            principle_name=principle.name,
            mechanism=mechanism,
            mitigation=mitigation,
            confidence=0.85 if principle.evidence_level in ("Ia", "Ib", "IIa") else 0.7,
            evidence_level=principle.evidence_level
        )
    
    def _apply_state_modifiers(
        self,
        risks: List[RiskAssessment],
        state: PatientState
    ) -> List[RiskAssessment]:
        """
        Modify risk assessments based on current patient state.
        
        Compromised states increase effective risk levels.
        """
        multiplier = state.get_risk_multiplier()
        
        if multiplier <= 1.0:
            return risks
            
        modified_risks = []
        
        for risk in risks:
            # Increase confidence based on state compromise
            new_confidence = min(1.0, risk.confidence * multiplier)
            
            # Potentially upgrade risk level for severe state compromise
            new_level = risk.level
            if multiplier >= 2.0 and risk.level == RiskLevel.MODERATE:
                new_level = RiskLevel.HIGH
            elif multiplier >= 2.0 and risk.level == RiskLevel.HIGH:
                new_level = RiskLevel.CRITICAL
            elif multiplier >= 1.5 and risk.level == RiskLevel.LOW:
                new_level = RiskLevel.MODERATE
            
            # Create modified assessment
            modified = RiskAssessment(
                level=new_level,
                structure=risk.structure,
                action=risk.action,
                principle_id=risk.principle_id,
                principle_name=risk.principle_name,
                mechanism=risk.mechanism,
                mitigation=risk.mitigation,
                confidence=new_confidence,
                evidence_level=risk.evidence_level,
                supporting_chunks=risk.supporting_chunks
            )
            
            # Add note about state-modified assessment
            if new_level != risk.level:
                modified.mechanism += (
                    f" [Risk elevated from {risk.level.value} due to "
                    f"compromised patient state (visibility: {state.visibility:.0%})]"
                )
            
            modified_risks.append(modified)
        
        return modified_risks
    
    async def get_applicable_principles(
        self,
        entity_name: str,
        action: Optional[str] = None
    ) -> List[ClinicalPrinciple]:
        """
        Get all principles that could apply to an entity/action combination.
        
        Useful for educational display and principle browsing.
        """
        entity = await self.get_entity_physics(entity_name)
        if entity is None:
            entity = EntityPhysics(name=entity_name, canonical_name=entity_name)
        
        context = SurgicalContext(approach="unknown", target_structure=entity_name)
        candidates = self._find_candidate_principles(entity, action or "", context)
        
        return [
            self.principles[pid] 
            for pid in candidates 
            if pid in self.principles
        ]
    
    def get_principles_by_domain(self, domain: str) -> List[ClinicalPrinciple]:
        """Get all principles in a domain."""
        principle_ids = self.principles_by_domain.get(domain, [])
        return [self.principles[pid] for pid in principle_ids if pid in self.principles]
    
    def get_all_domains(self) -> List[str]:
        """Get list of all domains with principles."""
        return list(self.principles_by_domain.keys())
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "initialized": self._initialized,
            "schema_available": self._schema_available,
            "principles_loaded": self._principles_loaded,
            "entities_cached": len(self.entity_cache),
            "domains": list(self.principles_by_domain.keys()),
            "principles_by_domain": {
                domain: len(ids) 
                for domain, ids in self.principles_by_domain.items()
            }
        }
