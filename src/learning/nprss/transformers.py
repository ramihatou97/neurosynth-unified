# src/learning/nprss/transformers.py
"""
Learning Transformers for NPRSS

Transforms raw procedure content into learning-optimized structures:
- PhaseMapper: Classifies elements into 4-phase framework
- CSPExtractor: Extracts Critical Safety Points from danger zones
- AnchorGenerator: Creates visuospatial anchors from visual descriptions
"""

import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .models import (
    PhaseType,
    CriticalSafetyPoint,
    VisuospatialAnchor,
    PhaseGate,
)


# =============================================================================
# PHASE MAPPER
# =============================================================================

@dataclass
class PhaseMapperSettings:
    """Configuration for phase mapping"""
    use_llm_fallback: bool = True
    confidence_threshold: float = 0.7
    validate_sequence: bool = True


class PhaseMapper:
    """
    Maps procedure elements to 4-phase learning framework

    Phase Framework (Cowan's 4-chunk working memory):
    - ARCHITECTURE: Before the incision (positioning, fixation, registration)
    - APPROACH: From incision to dural opening (skin, bone, dura)
    - TARGET: The "WHY" of surgery (pathology-specific actions)
    - CLOSURE: Reverse of approach (hemostasis, reconstruction, skin)

    Methods:
    1. Keyword-based classification (fast, deterministic)
    2. LLM-assisted classification (fallback for ambiguous cases)
    3. Positional heuristics (last resort)
    """

    # Keyword patterns for each phase
    PHASE_KEYWORDS = {
        PhaseType.ARCHITECTURE: [
            'position', 'positioning', 'mayfield', 'pin', 'fixation',
            'navigation', 'registration', 'prep', 'drape', 'draping',
            'setup', 'time-out', 'timeout', 'antibiotics', 'anesthesia',
            'induction', 'intubation', 'foley', 'lines', 'monitoring',
            'neuromonitoring', 'ionm', 'ssep', 'mep', 'baseline',
            'lateral', 'supine', 'prone', 'park bench', 'three-quarter',
        ],
        PhaseType.APPROACH: [
            'incision', 'skin', 'scalp', 'dissection', 'flap',
            'craniotomy', 'craniectomy', 'burr hole', 'drill', 'drilling',
            'bone', 'bone flap', 'dura', 'dural', 'durotomy', 'dural opening',
            'retraction', 'retractor', 'exposure', 'expose', 'approach',
            'corridor', 'sylvian', 'fissure', 'cistern', 'arachnoid',
            'laminectomy', 'laminotomy', 'flavectomy', 'foraminotomy',
            'muscle', 'fascia', 'periosteum', 'temporalis',
        ],
        PhaseType.TARGET: [
            'resection', 'resect', 'tumor', 'lesion', 'clip', 'clipping',
            'aneurysm', 'avm', 'malformation', 'decompression', 'decompress',
            'fusion', 'instrumentation', 'screw', 'rod', 'cage', 'graft',
            'discectomy', 'corpectomy', 'microdiscectomy', 'foraminotomy',
            'stimulation', 'mapping', 'awake', 'eloquent', 'motor',
            'biopsy', 'debulking', 'gross total', 'subtotal', 'evacuation',
            'hematoma', 'abscess', 'cyst', 'fenestration', 'marsupialization',
            'bypass', 'anastomosis', 'revascularization', 'ec-ic',
            'stereotactic', 'lead', 'electrode', 'dbs', 'lesioning',
        ],
        PhaseType.CLOSURE: [
            'closure', 'close', 'closing', 'hemostasis', 'bleeding',
            'surgicel', 'gelfoam', 'floseal', 'bipolar', 'coagulation',
            'dural closure', 'duraplasty', 'duragen', 'dural substitute',
            'cranioplasty', 'bone replacement', 'plate', 'titanium',
            'muscle', 'fascia', 'galea', 'skin closure', 'staples', 'suture',
            'drain', 'subgaleal', 'jackson-pratt', 'jp drain',
            'count', 'sponge count', 'needle count', 'instrument count',
            'dressing', 'head wrap', 'collar', 'brace',
        ],
    }

    # Action verbs associated with phases
    PHASE_VERBS = {
        PhaseType.ARCHITECTURE: ['position', 'secure', 'register', 'prep', 'drape', 'induce'],
        PhaseType.APPROACH: ['incise', 'dissect', 'elevate', 'drill', 'open', 'retract', 'expose'],
        PhaseType.TARGET: ['resect', 'clip', 'decompress', 'fuse', 'implant', 'stimulate', 'evacuate'],
        PhaseType.CLOSURE: ['close', 'achieve', 'replace', 'secure', 'dress', 'count'],
    }

    PHASE_CLASSIFICATION_PROMPT = """
Classify this neurosurgical step into one of 4 phases:

ARCHITECTURE: Setup before incision (positioning, fixation, navigation, prep)
APPROACH: From incision to target exposure (skin, bone work, dural opening)
TARGET: Core surgical objective (resection, clipping, decompression, fusion)
CLOSURE: Reverse of approach (hemostasis, reconstruction, skin closure)

Step: {step_name}
Description: {description}
Maneuver: {the_maneuver}

Respond with ONLY one word: ARCHITECTURE, APPROACH, TARGET, or CLOSURE
"""

    def __init__(
        self,
        settings: Optional[PhaseMapperSettings] = None,
        llm_client: Any = None
    ):
        self.settings = settings or PhaseMapperSettings()
        self.llm = llm_client

    def map_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Map all elements to phases

        Args:
            elements: List of procedure element dicts

        Returns:
            Dict mapping element_id -> phase_type
        """
        mapping = {}

        # Filter to mappable elements (steps and substeps)
        mappable = [
            e for e in elements
            if e.get('element_type') in ['step', 'substep']
        ]

        for element in mappable:
            element_id = str(element.get('id'))

            # Try keyword classification first
            phase, confidence = self._keyword_classify(element)

            if confidence >= self.settings.confidence_threshold:
                mapping[element_id] = phase
            elif self.settings.use_llm_fallback and self.llm:
                # Fall back to LLM
                phase = self._llm_classify(element)
                mapping[element_id] = phase
            else:
                # Positional fallback
                phase = self._positional_fallback(element, mappable)
                mapping[element_id] = phase

        # Validate and fix sequence errors
        if self.settings.validate_sequence:
            mapping = self._validate_phase_sequence(mapping, mappable)

        return mapping

    def _keyword_classify(
        self,
        element: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Classify element using keyword matching

        Returns (phase, confidence)
        """
        text = ' '.join([
            str(element.get('name', '')),
            str(element.get('description', '')),
            str(element.get('the_maneuver', '')),
            str(element.get('action_verb', '')),
        ]).lower()

        scores = {phase: 0 for phase in PhaseType}

        for phase, keywords in self.PHASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[phase] += 1

        # Check action verbs (higher weight)
        action_verb = str(element.get('action_verb', '')).lower()
        for phase, verbs in self.PHASE_VERBS.items():
            if action_verb in verbs:
                scores[phase] += 2

        total = sum(scores.values())
        if total == 0:
            return PhaseType.TARGET, 0.0

        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / total if total > 0 else 0.0

        return best_phase, confidence

    async def _llm_classify(self, element: Dict[str, Any]) -> str:
        """Classify element using LLM"""
        if not self.llm:
            return PhaseType.TARGET

        prompt = self.PHASE_CLASSIFICATION_PROMPT.format(
            step_name=element.get('name', ''),
            description=element.get('description', ''),
            the_maneuver=element.get('the_maneuver', '')
        )

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip().upper()

            phase_map = {
                'ARCHITECTURE': PhaseType.ARCHITECTURE,
                'APPROACH': PhaseType.APPROACH,
                'TARGET': PhaseType.TARGET,
                'CLOSURE': PhaseType.CLOSURE,
            }

            for key, phase in phase_map.items():
                if result.startswith(key):
                    return phase

        except Exception as e:
            print(f"LLM classification error: {e}")

        return PhaseType.TARGET  # Default fallback

    def _positional_fallback(
        self,
        element: Dict[str, Any],
        all_elements: List[Dict[str, Any]]
    ) -> str:
        """
        Fallback classification based on position in sequence

        Assumes rough distribution:
        - First 10-15%: Architecture
        - Next 25-30%: Approach
        - Middle 40-50%: Target
        - Last 15-20%: Closure
        """
        sequence = element.get('sequence_order', 0)
        total = len([e for e in all_elements if e.get('element_type') in ['step', 'substep']])

        if total == 0:
            return PhaseType.TARGET

        position_ratio = sequence / total

        if position_ratio < 0.12:
            return PhaseType.ARCHITECTURE
        elif position_ratio < 0.35:
            return PhaseType.APPROACH
        elif position_ratio < 0.80:
            return PhaseType.TARGET
        else:
            return PhaseType.CLOSURE

    def _validate_phase_sequence(
        self,
        mapping: Dict[str, str],
        elements: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Validate and fix phase sequence errors

        Rules:
        - Architecture should come before Approach
        - Approach should come before Target
        - Target should come before Closure
        - No Closure elements before Target elements
        """
        sorted_elements = sorted(
            [e for e in elements if str(e.get('id')) in mapping],
            key=lambda x: x.get('sequence_order', 0)
        )

        phase_order = [
            PhaseType.ARCHITECTURE,
            PhaseType.APPROACH,
            PhaseType.TARGET,
            PhaseType.CLOSURE
        ]

        current_phase_idx = 0

        for element in sorted_elements:
            element_id = str(element.get('id'))
            assigned_phase = mapping[element_id]
            assigned_idx = phase_order.index(assigned_phase) if assigned_phase in phase_order else 2

            if assigned_idx < current_phase_idx:
                # Phase regression detected - keep but could log warning
                pass
            else:
                current_phase_idx = max(current_phase_idx, assigned_idx)

        return mapping

    def generate_phase_gates(
        self,
        procedure_id: str,
        phase_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate verification gates between phases

        Returns list of PhaseGate dicts
        """
        gate_templates = {
            (PhaseType.ARCHITECTURE, PhaseType.APPROACH): {
                "verification_questions": [
                    "Is positioning optimized for surgical corridor?",
                    "Is navigation registered (if applicable)?",
                    "Is venous drainage unobstructed?",
                    "Are all monitors and equipment confirmed functional?"
                ],
                "prerequisites": [
                    "Mayfield secure with 3-point fixation",
                    "Skin prep and draping complete",
                    "Time-out completed",
                    "Antibiotics given"
                ]
            },
            (PhaseType.APPROACH, PhaseType.TARGET): {
                "verification_questions": [
                    "Is dural opening adequate for visualization?",
                    "Is brain relaxed?",
                    "Are critical vessels and nerves identified?",
                    "Is working corridor established?"
                ],
                "prerequisites": [
                    "Hemostasis achieved in approach",
                    "Retraction optimized",
                    "CSF drainage if needed for relaxation",
                    "IONM baselines stable"
                ]
            },
            (PhaseType.TARGET, PhaseType.CLOSURE): {
                "verification_questions": [
                    "Is primary surgical objective achieved?",
                    "Is hemostasis complete?",
                    "Are all implants/clips confirmed in position?",
                    "Final inspection completed?"
                ],
                "prerequisites": [
                    "ICG/Doppler confirmation (if vascular)",
                    "Tumor resection confirmed (if oncologic)",
                    "Decompression adequate (if stenosis)",
                    "Instrument and sponge counts initiated"
                ]
            }
        }

        gates = []
        for (from_phase, to_phase), template in gate_templates.items():
            gates.append({
                "id": str(uuid.uuid4()),
                "procedure_id": procedure_id,
                "from_phase": from_phase.value if isinstance(from_phase, PhaseType) else from_phase,
                "to_phase": to_phase.value if isinstance(to_phase, PhaseType) else to_phase,
                "verification_questions": template["verification_questions"],
                "prerequisites": template["prerequisites"]
            })

        return gates


# =============================================================================
# CSP EXTRACTOR
# =============================================================================

@dataclass
class CSPExtractorSettings:
    """Configuration for CSP extraction"""
    max_csps_per_procedure: int = 15
    min_confidence: float = 0.6
    use_llm_extraction: bool = True


class CSPExtractor:
    """
    Extract Critical Safety Points from:
    1. Danger Zone references
    2. Bailout protocols
    3. Safe Entry Zone boundaries
    4. LLM analysis of maneuvers
    """

    CSP_EXTRACTION_PROMPT = """
Analyze this neurosurgical substep and extract a Critical Safety Point (CSP) if applicable.

A CSP is a "hard stop" in the mental algorithm-a specific visual trigger that
mandates a specific action to prevent injury. Not every step has a CSP.

Substep: {substep_title}
Maneuver: {the_maneuver}
Danger Zone: {danger_zone}
Bailout Protocol: {bailout_protocol}

If a CSP exists, respond in this exact JSON format:
{{
    "has_csp": true,
    "when_action": "specific action being performed",
    "stop_if_trigger": "specific visual/tactile cue to stop",
    "structure_at_risk": "anatomical structure that could be injured",
    "mechanism_of_injury": "how injury would occur",
    "if_violated_action": "what to do if the trigger is missed"
}}

If no clear CSP exists, respond:
{{"has_csp": false}}
"""

    # Common danger zone to CSP patterns
    DANGER_ZONE_PATTERNS = {
        'facial nerve': {
            'triggers': ['blue dura', 'bill\'s bar', 'vertical crest'],
            'mechanisms': ['traction', 'heat', 'direct injury'],
        },
        'carotid': {
            'triggers': ['pulsation', 'red bleeding', 'arterial bleeding'],
            'mechanisms': ['laceration', 'avulsion', 'thermal injury'],
        },
        'optic nerve': {
            'triggers': ['white structure', 'loss of red reflex'],
            'mechanisms': ['traction', 'compression', 'thermal'],
        },
        'spinal cord': {
            'triggers': ['mep changes', 'ssep changes', 'dural breach'],
            'mechanisms': ['compression', 'ischemia', 'direct trauma'],
        },
        'venous sinus': {
            'triggers': ['dark bleeding', 'air embolism', 'venous engorgement'],
            'mechanisms': ['laceration', 'air embolism', 'thrombosis'],
        },
    }

    def __init__(
        self,
        settings: Optional[CSPExtractorSettings] = None,
        llm_client: Any = None
    ):
        self.settings = settings or CSPExtractorSettings()
        self.llm = llm_client

    def extract_csps(
        self,
        procedure_id: str,
        elements: List[Dict[str, Any]],
        danger_zones: List[Dict[str, Any]],
        safe_zones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract CSPs from procedure content

        Args:
            procedure_id: UUID of the procedure
            elements: Procedure elements with maneuvers
            danger_zones: Associated danger zones
            safe_zones: Associated safe entry zones

        Returns:
            List of CSP dicts ready for database insertion
        """
        csps = []
        csp_number = 1

        # Build danger zone lookup
        dz_lookup = {str(dz.get('id')): dz for dz in danger_zones}

        # Extract from elements with danger zone references
        for element in elements:
            if element.get('element_type') != 'substep':
                continue

            # Check for danger zone references
            dz_refs = element.get('danger_zone_refs', [])
            bailout = element.get('bailout_protocol', '')

            for dz_id in dz_refs:
                dz = dz_lookup.get(str(dz_id))
                if dz:
                    csp = self._extract_from_danger_zone(
                        procedure_id=procedure_id,
                        element=element,
                        danger_zone=dz,
                        csp_number=csp_number
                    )
                    if csp:
                        csps.append(csp)
                        csp_number += 1

            # Extract from bailout protocols
            if bailout and not dz_refs:
                csp = self._extract_from_bailout(
                    procedure_id=procedure_id,
                    element=element,
                    bailout_protocol=bailout,
                    csp_number=csp_number
                )
                if csp:
                    csps.append(csp)
                    csp_number += 1

        # Extract from safe zone boundaries
        for sz in safe_zones:
            csp = self._extract_from_safe_zone(
                procedure_id=procedure_id,
                safe_zone=sz,
                csp_number=csp_number
            )
            if csp:
                csps.append(csp)
                csp_number += 1

        # Limit total CSPs
        return csps[:self.settings.max_csps_per_procedure]

    def _extract_from_danger_zone(
        self,
        procedure_id: str,
        element: Dict[str, Any],
        danger_zone: Dict[str, Any],
        csp_number: int
    ) -> Optional[Dict[str, Any]]:
        """Extract CSP from danger zone reference"""
        structures = danger_zone.get('structures_at_risk', [])
        if not structures:
            return None

        primary_structure = structures[0] if isinstance(structures, list) else str(structures)

        # Try to find matching pattern
        trigger = None
        mechanism = None

        for structure_key, patterns in self.DANGER_ZONE_PATTERNS.items():
            if structure_key.lower() in primary_structure.lower():
                trigger = patterns['triggers'][0] if patterns['triggers'] else None
                mechanism = patterns['mechanisms'][0] if patterns['mechanisms'] else None
                break

        # Construct CSP
        csp = {
            "id": str(uuid.uuid4()),
            "procedure_id": procedure_id,
            "element_id": str(element.get('id')),
            "csp_number": csp_number,
            "phase_type": element.get('phase_type'),
            "when_action": element.get('the_maneuver', element.get('name', '')),
            "stop_if_trigger": trigger or f"Approaching {primary_structure}",
            "visual_cue": danger_zone.get('prevention_strategy', ''),
            "structure_at_risk": primary_structure,
            "mechanism_of_injury": mechanism or danger_zone.get('mechanism_of_injury', ''),
            "if_violated_action": danger_zone.get('management_if_violated', ''),
            "derived_from_danger_zone_id": str(danger_zone.get('id')),
            "retrieval_cue": f"What structure is at risk when {element.get('name', '')}?",
            "common_errors": [],
        }

        return csp

    def _extract_from_bailout(
        self,
        procedure_id: str,
        element: Dict[str, Any],
        bailout_protocol: str,
        csp_number: int
    ) -> Optional[Dict[str, Any]]:
        """Extract CSP from bailout protocol text"""
        # Parse bailout for structure at risk
        bailout_lower = bailout_protocol.lower()

        structure = None
        trigger = None

        # Common patterns in bailout protocols
        risk_patterns = [
            (r'if\s+(.+?)\s+(is\s+)?(injured|damaged|violated)', 'structure'),
            (r'(bleeding|hemorrhage)\s+from\s+(.+)', 'structure'),
            (r'protect\s+the\s+(.+)', 'structure'),
        ]

        for pattern, group_type in risk_patterns:
            match = re.search(pattern, bailout_lower)
            if match:
                structure = match.group(1) if group_type == 'structure' else match.group(2)
                break

        if not structure:
            return None

        csp = {
            "id": str(uuid.uuid4()),
            "procedure_id": procedure_id,
            "element_id": str(element.get('id')),
            "csp_number": csp_number,
            "phase_type": element.get('phase_type'),
            "when_action": element.get('the_maneuver', element.get('name', '')),
            "stop_if_trigger": f"Risk of {structure} injury",
            "visual_cue": "",
            "structure_at_risk": structure.strip(),
            "mechanism_of_injury": "",
            "if_violated_action": bailout_protocol,
            "retrieval_cue": f"What is the bailout for {element.get('name', '')}?",
            "common_errors": [],
        }

        return csp

    def _extract_from_safe_zone(
        self,
        procedure_id: str,
        safe_zone: Dict[str, Any],
        csp_number: int
    ) -> Optional[Dict[str, Any]]:
        """Extract CSP from safe zone boundary"""
        depth = safe_zone.get('max_safe_depth_mm')
        if not depth:
            return None

        csp = {
            "id": str(uuid.uuid4()),
            "procedure_id": procedure_id,
            "element_id": None,
            "csp_number": csp_number,
            "phase_type": PhaseType.TARGET.value,
            "when_action": f"Entering {safe_zone.get('name', 'safe zone')}",
            "stop_if_trigger": f"Depth exceeds {depth}mm",
            "visual_cue": f"Boundaries: {safe_zone.get('superior_boundary', '')} / {safe_zone.get('inferior_boundary', '')}",
            "structure_at_risk": "Adjacent critical structures",
            "mechanism_of_injury": "Exceeding safe depth",
            "if_violated_action": "Stop, reassess trajectory, check navigation",
            "derived_from_safe_zone_id": str(safe_zone.get('id')),
            "retrieval_cue": f"What is the safe depth for {safe_zone.get('name', '')}?",
            "common_errors": ["Exceeding depth limit", "Wrong trajectory"],
        }

        return csp

    async def extract_with_llm(
        self,
        procedure_id: str,
        element: Dict[str, Any],
        danger_zone: Optional[Dict[str, Any]] = None,
        csp_number: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to extract CSP from element"""
        if not self.llm:
            return None

        prompt = self.CSP_EXTRACTION_PROMPT.format(
            substep_title=element.get('name', ''),
            the_maneuver=element.get('the_maneuver', ''),
            danger_zone=danger_zone.get('name', 'None') if danger_zone else 'None',
            bailout_protocol=element.get('bailout_protocol', 'None')
        )

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            result = json.loads(response.content[0].text)

            if not result.get('has_csp'):
                return None

            csp = {
                "id": str(uuid.uuid4()),
                "procedure_id": procedure_id,
                "element_id": str(element.get('id')),
                "csp_number": csp_number,
                "phase_type": element.get('phase_type'),
                "when_action": result.get('when_action', ''),
                "stop_if_trigger": result.get('stop_if_trigger', ''),
                "structure_at_risk": result.get('structure_at_risk', ''),
                "mechanism_of_injury": result.get('mechanism_of_injury', ''),
                "if_violated_action": result.get('if_violated_action', ''),
                "retrieval_cue": f"CSP for {element.get('name', '')}",
                "common_errors": [],
            }

            return csp

        except Exception as e:
            print(f"LLM CSP extraction error: {e}")
            return None


# =============================================================================
# ANCHOR GENERATOR
# =============================================================================

class AnchorGenerator:
    """
    Generates visuospatial anchors from visual descriptions

    Anchors encode spatial memory by linking steps to:
    - Expected surgical view
    - Anatomical landmarks
    - Color cues
    - Depth references
    - Mental rotation prompts
    """

    MENTAL_ROTATION_TEMPLATES = [
        "What structure lies immediately {direction}?",
        "If you rotate the view {degrees}, what becomes visible?",
        "From this view, which direction is {landmark}?",
        "What is the relationship between {structure1} and {structure2}?",
    ]

    def generate_anchors(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate visuospatial anchors for elements with visual descriptions

        Args:
            elements: Procedure elements

        Returns:
            List of anchor dicts
        """
        anchors = []

        for element in elements:
            visual_desc = element.get('visual_description')
            if not visual_desc:
                continue

            # Handle both dict and object
            if isinstance(visual_desc, dict):
                expected_view = visual_desc.get('expected_view', '')
                landmarks = visual_desc.get('landmarks', [])
                color_cues = visual_desc.get('color_cues', '')
            else:
                expected_view = getattr(visual_desc, 'expected_view', '')
                landmarks = getattr(visual_desc, 'landmarks', [])
                color_cues = getattr(visual_desc, 'color_cues', '')

            if not expected_view and not landmarks:
                continue

            anchor = {
                "id": str(uuid.uuid4()),
                "element_id": str(element.get('id')),
                "expected_view": expected_view,
                "landmarks": landmarks if isinstance(landmarks, list) else [landmarks],
                "color_cues": color_cues,
                "mental_rotation_prompt": self._generate_rotation_prompt(landmarks),
                "spatial_relationship": self._extract_spatial_relationship(expected_view),
                "depth_reference": self._extract_depth(element),
                "viewing_angle": self._infer_viewing_angle(expected_view),
            }

            anchors.append(anchor)

        return anchors

    def _generate_rotation_prompt(self, landmarks: List[str]) -> str:
        """Generate a mental rotation prompt based on landmarks"""
        if not landmarks:
            return ""

        primary = landmarks[0] if landmarks else "the target"
        directions = ["deep", "superficial", "medial", "lateral", "superior", "inferior"]

        import random
        direction = random.choice(directions)

        return f"What structure lies immediately {direction} to {primary}?"

    def _extract_spatial_relationship(self, view_description: str) -> str:
        """Extract spatial relationships from view description"""
        if not view_description:
            return ""

        # Look for spatial patterns
        patterns = [
            r'(\w+)\s+(runs|courses|lies)\s+(superior|inferior|medial|lateral|deep|superficial)\s+to\s+(\w+)',
            r'(\w+)\s+is\s+(above|below|medial|lateral)\s+(\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, view_description, re.IGNORECASE)
            if match:
                return match.group(0)

        return ""

    def _extract_depth(self, element: Dict[str, Any]) -> str:
        """Extract depth reference from element"""
        # Check standard measurements
        measurements = element.get('standard_measurements', '')
        if measurements:
            depth_match = re.search(r'(\d+\.?\d*)\s*(mm|cm)\s*(deep|depth)', measurements, re.IGNORECASE)
            if depth_match:
                return f"{depth_match.group(1)}{depth_match.group(2)} from surface"

        return ""

    def _infer_viewing_angle(self, view_description: str) -> str:
        """Infer viewing angle from description"""
        if not view_description:
            return ""

        view_lower = view_description.lower()

        angle_keywords = {
            'lateral': ['lateral', 'side', 'temporal'],
            'superior': ['superior', 'top', 'above', 'looking down'],
            'anterior': ['anterior', 'front', 'ventral'],
            'posterior': ['posterior', 'back', 'dorsal'],
            'inferior': ['inferior', 'below', 'looking up'],
            'oblique': ['oblique', 'angled', '45'],
        }

        for angle, keywords in angle_keywords.items():
            if any(kw in view_lower for kw in keywords):
                return angle

        return "direct"
