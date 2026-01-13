"""
Patient Context Models for Demographic-Aware Gap Detection
==========================================================

P0 Enhancement: Patient-Context Injection

This module provides demographic-aware gap detection by adjusting:
- Measurement thresholds (pediatric vs adult ICP targets)
- Drug dosing warnings (weight-based, age-appropriate)
- Contraindication flags (pregnancy, renal function)
- Anatomy considerations (pediatric bone development)

Clinical Rationale:
- Pediatric ICP: 3-7 mmHg normal (vs adult 7-15 mmHg)
- Pediatric treatment threshold: 20 mmHg (vs adult 22 mmHg per BTF)
- Mannitol contraindicated in renal failure (GFR <30)
- Certain AEDs contraindicated in pregnancy (valproic acid)
- Drug dosing must be weight-based in pediatrics
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgeGroup(Enum):
    """Age group classification for threshold adjustment."""

    NEONATE = "neonate"          # 0-28 days
    INFANT = "infant"            # 1-12 months
    TODDLER = "toddler"          # 1-3 years
    CHILD = "child"              # 3-12 years
    ADOLESCENT = "adolescent"    # 12-18 years
    ADULT = "adult"              # 18-65 years
    ELDERLY = "elderly"          # >65 years


class PregnancyStatus(Enum):
    """Pregnancy status for contraindication checking."""

    NOT_PREGNANT = "not_pregnant"
    FIRST_TRIMESTER = "first_trimester"
    SECOND_TRIMESTER = "second_trimester"
    THIRD_TRIMESTER = "third_trimester"
    POSTPARTUM = "postpartum"
    NOT_APPLICABLE = "not_applicable"  # Male patients


class RenalFunction(Enum):
    """Renal function category based on GFR."""

    NORMAL = "normal"              # GFR >= 90
    MILD_IMPAIRMENT = "mild"       # GFR 60-89
    MODERATE_IMPAIRMENT = "moderate"  # GFR 30-59
    SEVERE_IMPAIRMENT = "severe"   # GFR 15-29
    END_STAGE = "end_stage"        # GFR < 15 or dialysis


@dataclass
class PatientContext:
    """
    Patient demographic context for gap detection adjustment.

    This context modifies gap detection behavior based on patient
    demographics and comorbidities. All fields are optional.

    Example Usage:
        context = PatientContext(
            age_years=8,
            weight_kg=25.0,
            is_female=True,
            known_conditions={"epilepsy", "renal_failure"}
        )

        # Gap detection will now:
        # - Use pediatric ICP thresholds
        # - Flag mannitol as contraindicated (renal failure)
        # - Require weight-based drug dosing
    """

    # Core demographics
    age_years: Optional[float] = None
    age_months: Optional[int] = None  # For infants/neonates
    weight_kg: Optional[float] = None
    is_female: Optional[bool] = None

    # Clinical status
    pregnancy_status: PregnancyStatus = PregnancyStatus.NOT_APPLICABLE
    gfr: Optional[float] = None  # Glomerular filtration rate

    # Known conditions (lowercase, underscored)
    known_conditions: Set[str] = field(default_factory=set)

    # Current medications (for interaction checking)
    current_medications: Set[str] = field(default_factory=set)

    # Allergies
    allergies: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Normalize and validate inputs."""
        # Normalize conditions to lowercase
        if self.known_conditions:
            self.known_conditions = {c.lower().replace(" ", "_") for c in self.known_conditions}
        if self.current_medications:
            self.current_medications = {m.lower() for m in self.current_medications}
        if self.allergies:
            self.allergies = {a.lower() for a in self.allergies}

    @property
    def age_group(self) -> AgeGroup:
        """Determine age group from age."""
        if self.age_years is None and self.age_months is None:
            return AgeGroup.ADULT  # Default assumption

        # Convert to months for fine-grained classification
        total_months = (self.age_months or 0)
        if self.age_years is not None:
            total_months += int(self.age_years * 12)

        if total_months < 1:
            return AgeGroup.NEONATE
        elif total_months < 12:
            return AgeGroup.INFANT
        elif total_months < 36:
            return AgeGroup.TODDLER
        elif total_months < 144:  # 12 years
            return AgeGroup.CHILD
        elif total_months < 216:  # 18 years
            return AgeGroup.ADOLESCENT
        elif total_months < 780:  # 65 years
            return AgeGroup.ADULT
        else:
            return AgeGroup.ELDERLY

    @property
    def is_pediatric(self) -> bool:
        """True if patient is under 18 years."""
        return self.age_group in (
            AgeGroup.NEONATE,
            AgeGroup.INFANT,
            AgeGroup.TODDLER,
            AgeGroup.CHILD,
            AgeGroup.ADOLESCENT,
        )

    @property
    def is_pregnant(self) -> bool:
        """True if currently pregnant."""
        return self.pregnancy_status in (
            PregnancyStatus.FIRST_TRIMESTER,
            PregnancyStatus.SECOND_TRIMESTER,
            PregnancyStatus.THIRD_TRIMESTER,
        )

    @property
    def renal_function(self) -> RenalFunction:
        """Categorize renal function from GFR."""
        if self.gfr is None:
            return RenalFunction.NORMAL  # Assume normal if unknown

        if self.gfr >= 90:
            return RenalFunction.NORMAL
        elif self.gfr >= 60:
            return RenalFunction.MILD_IMPAIRMENT
        elif self.gfr >= 30:
            return RenalFunction.MODERATE_IMPAIRMENT
        elif self.gfr >= 15:
            return RenalFunction.SEVERE_IMPAIRMENT
        else:
            return RenalFunction.END_STAGE

    @property
    def has_renal_impairment(self) -> bool:
        """True if GFR indicates significant renal impairment."""
        return self.renal_function in (
            RenalFunction.MODERATE_IMPAIRMENT,
            RenalFunction.SEVERE_IMPAIRMENT,
            RenalFunction.END_STAGE,
        )

    def has_condition(self, condition: str) -> bool:
        """Check if patient has a specific condition."""
        normalized = condition.lower().replace(" ", "_")
        return normalized in self.known_conditions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "age_years": self.age_years,
            "age_months": self.age_months,
            "age_group": self.age_group.value,
            "is_pediatric": self.is_pediatric,
            "weight_kg": self.weight_kg,
            "is_female": self.is_female,
            "pregnancy_status": self.pregnancy_status.value,
            "gfr": self.gfr,
            "renal_function": self.renal_function.value,
            "known_conditions": list(self.known_conditions),
            "current_medications": list(self.current_medications),
            "allergies": list(self.allergies),
        }


# =============================================================================
# AGE-SPECIFIC MEASUREMENT PROFILES
# =============================================================================

PEDIATRIC_MEASUREMENT_PROFILES: Dict[AgeGroup, Dict[str, Any]] = {
    AgeGroup.NEONATE: {
        "icp": {
            "normal_range": {"min": 0, "max": 6, "unit": "mmHg"},
            "treatment_threshold": {"value": 10, "unit": "mmHg"},
        },
        "cpp": {
            "normal_range": {"min": 40, "max": 50, "unit": "mmHg"},
            "minimum_target": {"value": 40, "unit": "mmHg"},
        },
        "fontanelle_status": "open",
        "skull_sutures": "unfused",
        "source": "Pediatric TBI Guidelines 2019",
    },
    AgeGroup.INFANT: {
        "icp": {
            "normal_range": {"min": 2, "max": 8, "unit": "mmHg"},
            "treatment_threshold": {"value": 15, "unit": "mmHg"},
        },
        "cpp": {
            "normal_range": {"min": 45, "max": 55, "unit": "mmHg"},
            "minimum_target": {"value": 45, "unit": "mmHg"},
        },
        "fontanelle_status": "closing",
        "skull_sutures": "partially_fused",
        "source": "Pediatric TBI Guidelines 2019",
    },
    AgeGroup.TODDLER: {
        "icp": {
            "normal_range": {"min": 3, "max": 10, "unit": "mmHg"},
            "treatment_threshold": {"value": 18, "unit": "mmHg"},
        },
        "cpp": {
            "normal_range": {"min": 50, "max": 60, "unit": "mmHg"},
            "minimum_target": {"value": 50, "unit": "mmHg"},
        },
        "skull_sutures": "fusing",
        "source": "Pediatric TBI Guidelines 2019",
    },
    AgeGroup.CHILD: {
        "icp": {
            "normal_range": {"min": 3, "max": 12, "unit": "mmHg"},
            "treatment_threshold": {"value": 20, "unit": "mmHg"},
        },
        "cpp": {
            "normal_range": {"min": 55, "max": 65, "unit": "mmHg"},
            "minimum_target": {"value": 55, "unit": "mmHg"},
        },
        "skull_sutures": "fused",
        "source": "Pediatric TBI Guidelines 2019",
    },
    AgeGroup.ADOLESCENT: {
        "icp": {
            "normal_range": {"min": 5, "max": 15, "unit": "mmHg"},
            "treatment_threshold": {"value": 20, "unit": "mmHg"},
        },
        "cpp": {
            "normal_range": {"min": 60, "max": 70, "unit": "mmHg"},
            "minimum_target": {"value": 60, "unit": "mmHg"},
        },
        "source": "Transition to adult guidelines",
    },
}

ADULT_MEASUREMENT_PROFILE: Dict[str, Any] = {
    "icp": {
        "normal_range": {"min": 7, "max": 15, "unit": "mmHg"},
        "treatment_threshold": {"value": 22, "unit": "mmHg", "source": "BTF 4th Edition 2016"},
    },
    "cpp": {
        "normal_range": {"min": 60, "max": 70, "unit": "mmHg"},
        "minimum_target": {"value": 60, "unit": "mmHg"},
        "maximum_target": {"value": 70, "unit": "mmHg", "note": "Avoid >70 (ARDS risk)"},
    },
    "source": "BTF 4th Edition 2016",
}

ELDERLY_MEASUREMENT_ADJUSTMENTS: Dict[str, Any] = {
    "cpp_considerations": "May tolerate lower CPP due to chronic hypertension",
    "anticoagulation_reversal": "Higher bleeding risk, lower threshold for reversal",
    "osmotherapy_caution": "Increased cardiac and renal comorbidities",
}


# =============================================================================
# CONTRAINDICATION MATRICES
# =============================================================================

DRUG_CONTRAINDICATIONS: Dict[str, Dict[str, Any]] = {
    "mannitol": {
        "contraindicated_conditions": [
            "renal_failure",
            "end_stage_renal_disease",
            "dialysis",
            "anuria",
            "severe_dehydration",
            "pulmonary_edema",
            "intracranial_hemorrhage_active",
        ],
        "relative_contraindications": [
            "congestive_heart_failure",
            "hypernatremia",
        ],
        "renal_gfr_threshold": 30,  # Contraindicated if GFR < 30
        "alternative": "hypertonic_saline",
    },
    "valproic_acid": {
        "contraindicated_conditions": [
            "pregnancy",
            "hepatic_failure",
            "urea_cycle_disorder",
            "mitochondrial_disease",
        ],
        "pregnancy_category": "X",
        "teratogenic_effects": ["neural_tube_defects", "craniofacial_abnormalities"],
        "alternative": "levetiracetam",
    },
    "phenytoin": {
        "contraindicated_conditions": [
            "sinus_bradycardia",
            "sinoatrial_block",
            "second_degree_av_block",
            "third_degree_av_block",
            "adams_stokes_syndrome",
        ],
        "relative_contraindications": [
            "pregnancy",  # Category D
            "hepatic_impairment",
        ],
        "pregnancy_category": "D",
        "alternative": "levetiracetam",
    },
    "methylprednisolone_high_dose": {
        "contraindicated_conditions": [
            "traumatic_brain_injury",  # CRASH trial
            "spinal_cord_injury",  # NASCIS now not recommended
        ],
        "evidence": {
            "TBI": "CRASH trial (2004): Increased mortality",
            "SCI": "NASCIS protocols no longer recommended",
        },
    },
    "dexamethasone": {
        "contraindicated_conditions": [
            "traumatic_brain_injury",  # No benefit, potential harm
        ],
        "indicated_conditions": [
            "vasogenic_edema",
            "brain_tumor",
            "brain_abscess",
        ],
    },
    "aspirin": {
        "contraindicated_conditions": [
            "active_bleeding",
            "hemorrhagic_stroke",
            "intracranial_hemorrhage",
            "platelet_count_below_100k",
        ],
    },
    "heparin": {
        "contraindicated_conditions": [
            "heparin_induced_thrombocytopenia",
            "active_bleeding",
            "intracranial_hemorrhage",
            "severe_thrombocytopenia",
        ],
    },
}

PREGNANCY_CONTRAINDICATED_DRUGS: Set[str] = {
    "valproic_acid",
    "carbamazepine",
    "phenobarbital",
    "topiramate",
    "warfarin",
    "methotrexate",
    "isotretinoin",
}

RENAL_DOSE_ADJUSTMENTS: Dict[str, Dict[RenalFunction, str]] = {
    "levetiracetam": {
        RenalFunction.NORMAL: "Standard dosing",
        RenalFunction.MILD_IMPAIRMENT: "500-1000 mg BID",
        RenalFunction.MODERATE_IMPAIRMENT: "250-750 mg BID",
        RenalFunction.SEVERE_IMPAIRMENT: "250-500 mg BID",
        RenalFunction.END_STAGE: "500-1000 mg daily, supplement after dialysis",
    },
    "gabapentin": {
        RenalFunction.NORMAL: "300-1200 mg TID",
        RenalFunction.MILD_IMPAIRMENT: "300-700 mg TID",
        RenalFunction.MODERATE_IMPAIRMENT: "200-700 mg BID",
        RenalFunction.SEVERE_IMPAIRMENT: "100-300 mg daily",
        RenalFunction.END_STAGE: "125-350 mg after dialysis",
    },
}


# =============================================================================
# CONTEXT-AWARE GAP DETECTION HELPER
# =============================================================================

class PatientContextAnalyzer:
    """
    Analyzes patient context to provide gap detection adjustments.

    Usage:
        analyzer = PatientContextAnalyzer()
        adjustments = analyzer.get_measurement_adjustments(patient_context)
        contraindications = analyzer.check_contraindications(
            patient_context,
            drugs_mentioned=["mannitol", "valproic_acid"]
        )
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_measurement_profile(
        self,
        patient_context: Optional[PatientContext]
    ) -> Dict[str, Any]:
        """
        Get appropriate measurement thresholds for patient.

        Returns adjusted ICP/CPP targets based on age group.
        """
        if patient_context is None:
            return ADULT_MEASUREMENT_PROFILE

        age_group = patient_context.age_group

        if age_group in PEDIATRIC_MEASUREMENT_PROFILES:
            profile = PEDIATRIC_MEASUREMENT_PROFILES[age_group].copy()
            profile["age_group"] = age_group.value
            profile["is_pediatric"] = True
            return profile

        if age_group == AgeGroup.ELDERLY:
            profile = ADULT_MEASUREMENT_PROFILE.copy()
            profile.update(ELDERLY_MEASUREMENT_ADJUSTMENTS)
            profile["age_group"] = "elderly"
            return profile

        profile = ADULT_MEASUREMENT_PROFILE.copy()
        profile["age_group"] = "adult"
        return profile

    def check_drug_contraindications(
        self,
        patient_context: Optional[PatientContext],
        drugs_mentioned: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Check for contraindicated drugs based on patient context.

        Returns list of contraindication warnings.
        """
        if patient_context is None:
            return []

        warnings = []

        for drug in drugs_mentioned:
            drug_lower = drug.lower().replace(" ", "_")

            if drug_lower not in DRUG_CONTRAINDICATIONS:
                continue

            contraindication_info = DRUG_CONTRAINDICATIONS[drug_lower]

            # Check absolute contraindications
            for condition in contraindication_info.get("contraindicated_conditions", []):
                if patient_context.has_condition(condition):
                    warnings.append({
                        "drug": drug,
                        "severity": "CRITICAL",
                        "reason": f"Contraindicated with {condition.replace('_', ' ')}",
                        "alternative": contraindication_info.get("alternative"),
                        "evidence": contraindication_info.get("evidence", {}).get(
                            condition, "Clinical guidelines"
                        ),
                    })

            # Check pregnancy contraindication
            if patient_context.is_pregnant:
                if drug_lower in PREGNANCY_CONTRAINDICATED_DRUGS:
                    warnings.append({
                        "drug": drug,
                        "severity": "CRITICAL",
                        "reason": "Contraindicated in pregnancy",
                        "pregnancy_category": contraindication_info.get(
                            "pregnancy_category", "X"
                        ),
                        "teratogenic_effects": contraindication_info.get(
                            "teratogenic_effects", []
                        ),
                        "alternative": contraindication_info.get("alternative"),
                    })

            # Check renal contraindication
            if patient_context.has_renal_impairment:
                gfr_threshold = contraindication_info.get("renal_gfr_threshold")
                if gfr_threshold and patient_context.gfr:
                    if patient_context.gfr < gfr_threshold:
                        warnings.append({
                            "drug": drug,
                            "severity": "CRITICAL",
                            "reason": f"Contraindicated with GFR < {gfr_threshold}",
                            "patient_gfr": patient_context.gfr,
                            "alternative": contraindication_info.get("alternative"),
                        })

        return warnings

    def get_dosing_warnings(
        self,
        patient_context: Optional[PatientContext],
        drugs_mentioned: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get dosing adjustment warnings based on patient context.
        """
        if patient_context is None:
            return []

        warnings = []

        # Pediatric weight-based dosing requirement
        if patient_context.is_pediatric:
            for drug in drugs_mentioned:
                warnings.append({
                    "drug": drug,
                    "severity": "HIGH",
                    "reason": "Pediatric patient requires weight-based dosing",
                    "patient_weight": patient_context.weight_kg,
                    "age_group": patient_context.age_group.value,
                })

        # Renal dose adjustments
        for drug in drugs_mentioned:
            drug_lower = drug.lower()
            if drug_lower in RENAL_DOSE_ADJUSTMENTS:
                renal_status = patient_context.renal_function
                if renal_status != RenalFunction.NORMAL:
                    adjustment = RENAL_DOSE_ADJUSTMENTS[drug_lower].get(renal_status)
                    if adjustment:
                        warnings.append({
                            "drug": drug,
                            "severity": "MEDIUM",
                            "reason": f"Renal adjustment needed ({renal_status.value})",
                            "adjusted_dose": adjustment,
                            "patient_gfr": patient_context.gfr,
                        })

        return warnings

    def get_anatomy_considerations(
        self,
        patient_context: Optional[PatientContext],
    ) -> List[Dict[str, Any]]:
        """
        Get anatomy-related considerations for pediatric patients.
        """
        if patient_context is None or not patient_context.is_pediatric:
            return []

        considerations = []
        age_group = patient_context.age_group

        if age_group in PEDIATRIC_MEASUREMENT_PROFILES:
            profile = PEDIATRIC_MEASUREMENT_PROFILES[age_group]

            if "fontanelle_status" in profile:
                considerations.append({
                    "structure": "Anterior fontanelle",
                    "status": profile["fontanelle_status"],
                    "clinical_relevance": "ICP assessment, surgical planning",
                })

            if "skull_sutures" in profile:
                considerations.append({
                    "structure": "Skull sutures",
                    "status": profile["skull_sutures"],
                    "clinical_relevance": "Cranial vault expansion, growing skull fracture risk",
                })

        if age_group in (AgeGroup.NEONATE, AgeGroup.INFANT, AgeGroup.TODDLER):
            considerations.append({
                "structure": "Developing brain",
                "consideration": "Higher plasticity, different injury patterns",
                "clinical_relevance": "Long-term neurodevelopmental outcomes",
            })

        return considerations

    def generate_context_summary(
        self,
        patient_context: Optional[PatientContext],
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of context-specific adjustments.
        """
        if patient_context is None:
            return {"status": "no_context", "using_defaults": True}

        return {
            "status": "context_applied",
            "patient_summary": {
                "age_group": patient_context.age_group.value,
                "is_pediatric": patient_context.is_pediatric,
                "is_pregnant": patient_context.is_pregnant,
                "renal_function": patient_context.renal_function.value,
            },
            "measurement_profile": self.get_measurement_profile(patient_context),
            "anatomy_considerations": self.get_anatomy_considerations(patient_context),
            "active_flags": [
                f"PEDIATRIC: {patient_context.age_group.value}"
                if patient_context.is_pediatric else None,
                "PREGNANT" if patient_context.is_pregnant else None,
                f"RENAL_IMPAIRMENT: {patient_context.renal_function.value}"
                if patient_context.has_renal_impairment else None,
            ],
        }
