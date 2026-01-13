"""
Disease Stage Context Detection
================================

P2 Enhancement: Acute vs Chronic Management Differentiation

This module detects disease stage context to ensure recommendations are
appropriate for the clinical phase (acute vs chronic, early vs late).

Clinical Safety Rationale:
- Acute SAH management differs from chronic hydrocephalus
- Acute TBI ICP management differs from chronic ICP (tumor)
- Early vs late stroke management (TPA window)
- Acute vs chronic spinal cord injury

Detection Approach:
1. Identify disease stage markers in content
2. Map stage to appropriate management paradigms
3. Flag mismatched stage-treatment combinations
4. Provide stage-specific recommendations
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DiseaseStage(Enum):
    """Clinical phase of disease."""

    # Temporal stages
    HYPERACUTE = "hyperacute"      # Minutes to hours (stroke, trauma)
    ACUTE = "acute"                # Hours to days
    SUBACUTE = "subacute"          # Days to weeks
    CHRONIC = "chronic"            # Weeks to months+
    RECURRENT = "recurrent"        # Recurrent episode

    # Severity/progression stages
    EARLY = "early"                # Early in disease course
    ADVANCED = "advanced"          # Advanced/late stage
    END_STAGE = "end_stage"        # Terminal phase

    # Treatment stages
    PREOPERATIVE = "preoperative"
    INTRAOPERATIVE = "intraoperative"
    POSTOPERATIVE = "postoperative"

    # Unknown
    UNKNOWN = "unknown"


class ManagementParadigm(Enum):
    """Treatment paradigm for disease stage."""

    EMERGENCY = "emergency"            # Life-saving intervention
    URGENT = "urgent"                  # Requires prompt treatment
    ELECTIVE = "elective"              # Can be scheduled
    CONSERVATIVE = "conservative"      # Non-surgical management
    PALLIATIVE = "palliative"          # Comfort-focused care
    REHABILITATIVE = "rehabilitative"  # Recovery and rehab


@dataclass
class StageProfile:
    """Profile for a disease at a specific stage."""

    disease: str
    stage: DiseaseStage
    typical_paradigm: ManagementParadigm
    time_frame: str
    key_interventions: List[str]
    contraindicated_interventions: List[str]
    stage_specific_goals: List[str]
    danger_flags: List[str]


@dataclass
class StageContextWarning:
    """Warning about stage-inappropriate recommendation."""

    warning_id: str
    detected_stage: DiseaseStage
    intervention: str
    issue: str
    recommendation: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW


# =============================================================================
# DISEASE-STAGE PROFILES
# =============================================================================

DISEASE_STAGE_PROFILES: Dict[str, Dict[DiseaseStage, StageProfile]] = {
    # =========================================================================
    # STROKE
    # =========================================================================
    "ischemic_stroke": {
        DiseaseStage.HYPERACUTE: StageProfile(
            disease="ischemic_stroke",
            stage=DiseaseStage.HYPERACUTE,
            typical_paradigm=ManagementParadigm.EMERGENCY,
            time_frame="0-4.5 hours (TPA window), 0-24 hours (thrombectomy)",
            key_interventions=[
                "TPA (within 4.5h window)",
                "Mechanical thrombectomy (within 24h for LVO)",
                "Blood pressure management (permissive HTN until reperfusion)",
                "Glucose control",
                "Aspirin after hemorrhage excluded",
            ],
            contraindicated_interventions=[
                "Aggressive blood pressure lowering before reperfusion",
                "Anticoagulation before 24-48h",
                "Surgical decompression (unless malignant MCA)",
            ],
            stage_specific_goals=[
                "Reperfusion (TPA or thrombectomy)",
                "Prevent hemorrhagic conversion",
                "Maintain perfusion to penumbra",
            ],
            danger_flags=[
                "TPA contraindications",
                "Hemorrhagic conversion",
                "Malignant edema",
            ],
        ),
        DiseaseStage.ACUTE: StageProfile(
            disease="ischemic_stroke",
            stage=DiseaseStage.ACUTE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="24 hours to 7 days",
            key_interventions=[
                "Aspirin or dual antiplatelet",
                "Statin therapy",
                "DVT prophylaxis",
                "Swallow evaluation",
                "Early mobilization",
                "Blood pressure control (target <180/105)",
            ],
            contraindicated_interventions=[
                "TPA (outside window)",
                "Full anticoagulation (unless specific indication)",
            ],
            stage_specific_goals=[
                "Secondary prevention",
                "Early rehabilitation",
                "Prevent complications",
            ],
            danger_flags=[
                "Hemorrhagic transformation",
                "Aspiration",
                "DVT/PE",
            ],
        ),
        DiseaseStage.CHRONIC: StageProfile(
            disease="ischemic_stroke",
            stage=DiseaseStage.CHRONIC,
            typical_paradigm=ManagementParadigm.REHABILITATIVE,
            time_frame="> 3 months",
            key_interventions=[
                "Long-term antiplatelet or anticoagulation",
                "Statin therapy",
                "Blood pressure control (<130/80)",
                "Rehabilitation",
                "Risk factor modification",
            ],
            contraindicated_interventions=[
                "Acute reperfusion therapy",
            ],
            stage_specific_goals=[
                "Maximize recovery",
                "Prevent recurrence",
                "Quality of life",
            ],
            danger_flags=[
                "Recurrent stroke",
                "Post-stroke depression",
                "Falls",
            ],
        ),
    },

    # =========================================================================
    # SUBARACHNOID HEMORRHAGE
    # =========================================================================
    "sah": {
        DiseaseStage.HYPERACUTE: StageProfile(
            disease="sah",
            stage=DiseaseStage.HYPERACUTE,
            typical_paradigm=ManagementParadigm.EMERGENCY,
            time_frame="0-24 hours",
            key_interventions=[
                "Secure airway",
                "Blood pressure control (SBP <160 until aneurysm secured)",
                "CT/CTA diagnosis",
                "EVD if hydrocephalus",
                "Aneurysm securing (coiling or clipping)",
                "Nimodipine 60mg q4h",
            ],
            contraindicated_interventions=[
                "Aggressive blood pressure lowering before aneurysm secured",
                "Lumbar puncture (if CT positive)",
                "Anticoagulation",
            ],
            stage_specific_goals=[
                "Secure aneurysm",
                "Prevent rebleed",
                "Treat hydrocephalus",
                "Manage ICP",
            ],
            danger_flags=[
                "Rebleed (highest in first 24h)",
                "Hydrocephalus",
                "Herniation",
            ],
        ),
        DiseaseStage.ACUTE: StageProfile(
            disease="sah",
            stage=DiseaseStage.ACUTE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="Days 1-14 (vasospasm window)",
            key_interventions=[
                "Nimodipine continuation",
                "Euvolemia (avoid hypovolemia)",
                "TCD monitoring for vasospasm",
                "Triple-H therapy if vasospasm",
                "Angioplasty/intra-arterial verapamil if refractory",
                "Salt supplementation",
            ],
            contraindicated_interventions=[
                "Fluid restriction",
                "Diuretics (unless specific indication)",
            ],
            stage_specific_goals=[
                "Prevent and treat vasospasm",
                "Maintain cerebral perfusion",
                "Avoid hyponatremia",
            ],
            danger_flags=[
                "Vasospasm (peak days 7-10)",
                "Delayed cerebral ischemia",
                "Hyponatremia/cerebral salt wasting",
            ],
        ),
        DiseaseStage.SUBACUTE: StageProfile(
            disease="sah",
            stage=DiseaseStage.SUBACUTE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="Days 14-21",
            key_interventions=[
                "Continue nimodipine (total 21 days)",
                "Mobilization",
                "Hydrocephalus evaluation",
                "VP shunt if persistent hydrocephalus",
            ],
            contraindicated_interventions=[],
            stage_specific_goals=[
                "Complete vasospasm prophylaxis",
                "Early rehabilitation",
                "Assess for shunt need",
            ],
            danger_flags=[
                "Persistent hydrocephalus",
                "Shunt dependence",
            ],
        ),
    },

    # =========================================================================
    # TRAUMATIC BRAIN INJURY
    # =========================================================================
    "tbi": {
        DiseaseStage.HYPERACUTE: StageProfile(
            disease="tbi",
            stage=DiseaseStage.HYPERACUTE,
            typical_paradigm=ManagementParadigm.EMERGENCY,
            time_frame="0-6 hours",
            key_interventions=[
                "Airway management",
                "C-spine immobilization",
                "Hemorrhage control",
                "CT head",
                "ICP monitor if indicated (GCS ≤8)",
                "Craniotomy if surgical lesion",
                "Osmotherapy for herniation",
            ],
            contraindicated_interventions=[
                "Corticosteroids (CRASH trial)",
                "Prophylactic hyperventilation",
                "Hypotonic fluids",
            ],
            stage_specific_goals=[
                "Prevent secondary injury",
                "Evacuate surgical lesions",
                "Maintain CPP",
            ],
            danger_flags=[
                "Herniation",
                "Expanding hematoma",
                "Hypotension",
                "Hypoxia",
            ],
        ),
        DiseaseStage.ACUTE: StageProfile(
            disease="tbi",
            stage=DiseaseStage.ACUTE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="6 hours to 7 days",
            key_interventions=[
                "ICP management (target <22 mmHg)",
                "CPP maintenance (60-70 mmHg)",
                "Sedation protocol",
                "Seizure prophylaxis (7 days)",
                "VTE prophylaxis",
                "Nutrition",
                "Fever control",
            ],
            contraindicated_interventions=[
                "Steroids",
                "Aggressive hyperventilation",
            ],
            stage_specific_goals=[
                "ICP/CPP optimization",
                "Prevent secondary insults",
                "Early prognostication",
            ],
            danger_flags=[
                "Refractory ICP",
                "Progression of contusions",
                "Post-traumatic seizures",
            ],
        ),
        DiseaseStage.SUBACUTE: StageProfile(
            disease="tbi",
            stage=DiseaseStage.SUBACUTE,
            typical_paradigm=ManagementParadigm.REHABILITATIVE,
            time_frame="1-4 weeks",
            key_interventions=[
                "Wean ICP monitor",
                "Cranioplasty planning if DC performed",
                "Early rehabilitation",
                "Tracheostomy/PEG if needed",
                "Neuropsychological evaluation",
            ],
            contraindicated_interventions=[],
            stage_specific_goals=[
                "Transition to rehabilitation",
                "Functional recovery",
            ],
            danger_flags=[
                "Post-traumatic hydrocephalus",
                "Persistent disorders of consciousness",
            ],
        ),
    },

    # =========================================================================
    # SPINAL CORD INJURY
    # =========================================================================
    "sci": {
        DiseaseStage.HYPERACUTE: StageProfile(
            disease="sci",
            stage=DiseaseStage.HYPERACUTE,
            typical_paradigm=ManagementParadigm.EMERGENCY,
            time_frame="0-24 hours",
            key_interventions=[
                "Spinal immobilization",
                "MAP goals (85-90 mmHg for 7 days)",
                "Surgical decompression if indicated",
                "ICU admission",
                "Baseline ASIA examination",
            ],
            contraindicated_interventions=[
                "High-dose methylprednisolone (NASCIS protocol)",
                "Delayed decompression if indicated",
            ],
            stage_specific_goals=[
                "Prevent secondary injury",
                "Optimize spinal cord perfusion",
                "Surgical decompression when indicated",
            ],
            danger_flags=[
                "Neurogenic shock",
                "Respiratory failure (cervical)",
                "Ascending injury",
            ],
        ),
        DiseaseStage.ACUTE: StageProfile(
            disease="sci",
            stage=DiseaseStage.ACUTE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="1-7 days",
            key_interventions=[
                "MAP goals",
                "DVT prophylaxis (mechanical then pharmacologic)",
                "Bowel/bladder management",
                "Skin protection",
                "Respiratory support (C5 or above)",
            ],
            contraindicated_interventions=[
                "Steroids",
            ],
            stage_specific_goals=[
                "Maintain perfusion",
                "Prevent complications",
                "Plan stabilization surgery if needed",
            ],
            danger_flags=[
                "Autonomic dysreflexia (T6 or above)",
                "DVT/PE",
                "Respiratory decompensation",
            ],
        ),
        DiseaseStage.CHRONIC: StageProfile(
            disease="sci",
            stage=DiseaseStage.CHRONIC,
            typical_paradigm=ManagementParadigm.REHABILITATIVE,
            time_frame="> 6 months",
            key_interventions=[
                "Rehabilitation",
                "Chronic pain management",
                "Spasticity management",
                "Neurogenic bladder management",
                "Pressure ulcer prevention",
            ],
            contraindicated_interventions=[],
            stage_specific_goals=[
                "Maximize independence",
                "Prevent complications",
                "Quality of life",
            ],
            danger_flags=[
                "Autonomic dysreflexia",
                "Syringomyelia",
                "Chronic pain",
            ],
        ),
    },

    # =========================================================================
    # BRAIN TUMOR
    # =========================================================================
    "brain_tumor": {
        DiseaseStage.PREOPERATIVE: StageProfile(
            disease="brain_tumor",
            stage=DiseaseStage.PREOPERATIVE,
            typical_paradigm=ManagementParadigm.ELECTIVE,
            time_frame="Days to weeks before surgery",
            key_interventions=[
                "Dexamethasone for vasogenic edema",
                "Seizure prophylaxis if indicated",
                "Surgical planning (navigation, functional mapping)",
                "Embolization if vascular",
            ],
            contraindicated_interventions=[
                "Steroids in suspected lymphoma (before biopsy)",
            ],
            stage_specific_goals=[
                "Reduce edema",
                "Optimize for surgery",
                "Establish diagnosis",
            ],
            danger_flags=[
                "Mass effect",
                "Herniation",
                "Steroid-responsive lymphoma",
            ],
        ),
        DiseaseStage.POSTOPERATIVE: StageProfile(
            disease="brain_tumor",
            stage=DiseaseStage.POSTOPERATIVE,
            typical_paradigm=ManagementParadigm.URGENT,
            time_frame="0-7 days after surgery",
            key_interventions=[
                "Steroid taper",
                "Seizure prophylaxis",
                "MRI within 24-48h",
                "DVT prophylaxis",
                "Plan adjuvant therapy",
            ],
            contraindicated_interventions=[
                "Abrupt steroid discontinuation",
            ],
            stage_specific_goals=[
                "Verify resection extent",
                "Prevent complications",
                "Plan adjuvant therapy",
            ],
            danger_flags=[
                "Hemorrhage",
                "Edema",
                "Infection",
                "New deficits",
            ],
        ),
    },
}

# =============================================================================
# STAGE DETECTION PATTERNS
# =============================================================================

STAGE_DETECTION_PATTERNS: Dict[DiseaseStage, List[str]] = {
    DiseaseStage.HYPERACUTE: [
        r"\bhyper-?acute\b",
        r"\bwithin\s+\d+\s*(?:minutes?|hours?)\b",
        r"\bimmediate(?:ly)?\b",
        r"\bemergent\b",
        r"\bfirst\s+(?:24|48)\s*hours?\b",
        r"\bacute\s+presentation\b",
        r"\b(?:tpa|alteplase)\s+window\b",
        r"\btime\s+(?:to|is)\s+brain\b",
    ],
    DiseaseStage.ACUTE: [
        r"\bacute\s+phase\b",
        r"\bday(?:s)?\s+[1-7]\b",
        r"\bfirst\s+week\b",
        r"\bicu\s+management\b",
        r"\bin-?patient\b",
        r"\bhospitalized\b",
    ],
    DiseaseStage.SUBACUTE: [
        r"\bsub-?acute\b",
        r"\bweek(?:s)?\s+[2-4]\b",
        r"\b(?:second|third|fourth)\s+week\b",
        r"\bstep-?down\b",
        r"\btransition(?:ing)?\s+to\b",
    ],
    DiseaseStage.CHRONIC: [
        r"\bchronic\b",
        r"\blong-?term\b",
        r"\boutpatient\b",
        r"\bmonths?\s+(?:after|since|post)\b",
        r"\byears?\s+(?:after|since|post)\b",
        r"\bmaintenance\b",
        r"\bfollow-?up\b",
    ],
    DiseaseStage.PREOPERATIVE: [
        r"\bpre-?op(?:erative)?\b",
        r"\bbefore\s+surgery\b",
        r"\bprior\s+to\s+(?:surgery|operation)\b",
        r"\bsurgical\s+planning\b",
    ],
    DiseaseStage.INTRAOPERATIVE: [
        r"\bintra-?op(?:erative)?\b",
        r"\bduring\s+surgery\b",
        r"\bin\s+the\s+(?:operating|or)\b",
    ],
    DiseaseStage.POSTOPERATIVE: [
        r"\bpost-?op(?:erative)?\b",
        r"\bafter\s+surgery\b",
        r"\bfollowing\s+(?:surgery|operation|resection)\b",
        r"\bpod\s*\d+\b",  # Post-operative day
    ],
}


class DiseaseStageAnalyzer:
    """
    Analyzes content for disease stage and validates stage-appropriate recommendations.

    Usage:
        analyzer = DiseaseStageAnalyzer()
        stages = analyzer.detect_stage(content)
        profile = analyzer.get_stage_profile("tbi", DiseaseStage.ACUTE)
        warnings = analyzer.check_stage_appropriateness(content, "tbi")
    """

    def __init__(self):
        self.disease_profiles = DISEASE_STAGE_PROFILES
        self._compiled_patterns: Dict[DiseaseStage, List[re.Pattern]] = {}

        # Compile detection patterns
        for stage, patterns in STAGE_DETECTION_PATTERNS.items():
            self._compiled_patterns[stage] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self.logger = logging.getLogger(__name__)

    def detect_stage(
        self,
        content: str,
    ) -> List[Tuple[DiseaseStage, float]]:
        """
        Detect disease stage(s) from content.

        Returns list of (stage, confidence) tuples, sorted by confidence.
        """
        content_lower = content.lower()
        stage_scores: Dict[DiseaseStage, float] = {}

        for stage, patterns in self._compiled_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(content_lower)
                score += len(matches) * 0.3  # Weight per match

            if score > 0:
                stage_scores[stage] = min(score, 1.0)  # Cap at 1.0

        # Sort by confidence
        sorted_stages = sorted(
            stage_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_stages if sorted_stages else [(DiseaseStage.UNKNOWN, 0.0)]

    def detect_disease(self, content: str) -> List[str]:
        """Detect diseases mentioned in content."""
        content_lower = content.lower()
        detected = []

        disease_patterns = {
            "ischemic_stroke": [
                r"\bischemic\s+stroke\b",
                r"\bacute\s+stroke\b",
                r"\bcerebral\s+infarct(?:ion)?\b",
            ],
            "sah": [
                r"\bsubarachnoid\s+hemorrhage\b",
                r"\bsah\b",
                r"\baneurysm(?:al)?\s+(?:rupture|hemorrhage)\b",
            ],
            "tbi": [
                r"\btraumatic\s+brain\s+injury\b",
                r"\btbi\b",
                r"\bhead\s+(?:trauma|injury)\b",
            ],
            "sci": [
                r"\bspinal\s+cord\s+injury\b",
                r"\bsci\b",
                r"\bcervical\s+(?:cord\s+)?injury\b",
            ],
            "brain_tumor": [
                r"\bbrain\s+tumor\b",
                r"\bglioma\b",
                r"\bglioblastoma\b",
                r"\bmeningioma\b",
                r"\bintracranial\s+(?:mass|tumor|lesion)\b",
            ],
        }

        for disease, patterns in disease_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    detected.append(disease)
                    break

        return detected

    def get_stage_profile(
        self,
        disease: str,
        stage: DiseaseStage,
    ) -> Optional[StageProfile]:
        """Get the stage profile for a specific disease and stage."""
        disease_profiles = self.disease_profiles.get(disease)
        if disease_profiles:
            return disease_profiles.get(stage)
        return None

    def check_stage_appropriateness(
        self,
        content: str,
        disease: Optional[str] = None,
    ) -> List[StageContextWarning]:
        """
        Check if content recommendations are appropriate for detected stage.

        Returns warnings for potential stage-inappropriate recommendations.
        """
        warnings: List[StageContextWarning] = []
        content_lower = content.lower()

        # Detect disease if not provided
        diseases = [disease] if disease else self.detect_disease(content)
        if not diseases:
            return warnings

        # Detect stage
        stages = self.detect_stage(content)
        if stages[0][0] == DiseaseStage.UNKNOWN:
            return warnings

        detected_stage = stages[0][0]

        for disease_name in diseases:
            profile = self.get_stage_profile(disease_name, detected_stage)
            if not profile:
                continue

            # Check for contraindicated interventions
            for intervention in profile.contraindicated_interventions:
                intervention_terms = intervention.lower().split()
                key_terms = [t for t in intervention_terms if len(t) > 3]

                mentioned = all(term in content_lower for term in key_terms[:3])

                if mentioned:
                    # Check if it's being recommended (not just mentioned as contraindicated)
                    recommend_patterns = [
                        r"recommend\w*",
                        r"administer",
                        r"give",
                        r"initiate",
                        r"start",
                        r"consider(?!ing\s+(?:not|avoid))",
                    ]

                    for pattern in recommend_patterns:
                        # Find pattern near intervention mention
                        combined_pattern = f"{pattern}.*{key_terms[0]}"
                        if re.search(combined_pattern, content_lower):
                            warnings.append(StageContextWarning(
                                warning_id=f"STAGE_{disease_name}_{detected_stage.value}",
                                detected_stage=detected_stage,
                                intervention=intervention,
                                issue=f"'{intervention}' may be contraindicated in {detected_stage.value} {disease_name}",
                                recommendation=f"Verify appropriateness for {detected_stage.value} phase",
                                severity="HIGH",
                            ))
                            break

        return warnings

    def get_stage_appropriate_interventions(
        self,
        disease: str,
        stage: DiseaseStage,
    ) -> Dict[str, Any]:
        """Get appropriate interventions for a disease at a specific stage."""
        profile = self.get_stage_profile(disease, stage)
        if not profile:
            return {}

        return {
            "disease": disease,
            "stage": stage.value,
            "paradigm": profile.typical_paradigm.value,
            "time_frame": profile.time_frame,
            "key_interventions": profile.key_interventions,
            "contraindicated": profile.contraindicated_interventions,
            "goals": profile.stage_specific_goals,
            "danger_flags": profile.danger_flags,
        }

    def get_all_stages_for_disease(
        self,
        disease: str
    ) -> List[Dict[str, Any]]:
        """Get all stage profiles for a disease."""
        disease_profiles = self.disease_profiles.get(disease)
        if not disease_profiles:
            return []

        return [
            self.get_stage_appropriate_interventions(disease, stage)
            for stage in disease_profiles.keys()
        ]
