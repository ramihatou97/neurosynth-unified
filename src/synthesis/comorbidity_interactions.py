"""
Comorbidity Interaction Awareness
==================================

P1 Enhancement: Comorbidity-Aware Gap Detection

This module detects dangerous drug-drug, drug-condition, and multi-factor
interactions that may not be obvious from single-factor analysis.

Clinical Safety Rationale:
- CKD patient + mannitol recommendation → flag contraindication
- Anticoagulation + recent craniotomy + TPA consideration → flag bleeding risk
- Diabetes + steroids + infection risk → compound risk awareness
- Hepatic failure + phenytoin metabolism → toxicity risk

Detection Approach:
1. Drug-Drug Interactions (pharmacokinetic & pharmacodynamic)
2. Drug-Disease Interactions (contraindicated combinations)
3. Cascade Risk Detection (multi-factor compounding risks)
4. Syndrome-Specific Alerts (e.g., serotonin syndrome precursors)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class InteractionSeverity(Enum):
    """Severity of drug/condition interaction."""

    CRITICAL = "critical"      # Absolute contraindication, potential fatality
    HIGH = "high"              # Major interaction, requires intervention
    MODERATE = "moderate"      # May need dose adjustment or monitoring
    LOW = "low"                # Minor interaction, monitor only
    INFORMATIONAL = "info"     # Theoretical concern, low clinical significance


class InteractionType(Enum):
    """Type of interaction detected."""

    DRUG_DRUG = "drug_drug"                # Two drugs interact
    DRUG_DISEASE = "drug_disease"          # Drug contraindicated with condition
    DRUG_PROCEDURE = "drug_procedure"      # Drug timing around surgery
    CASCADE_RISK = "cascade_risk"          # Multiple factors compound risk
    SYNDROME_PRECURSOR = "syndrome"        # Combination may cause syndrome


@dataclass
class DrugInteraction:
    """Represents a drug-drug interaction."""

    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    recommendation: str
    evidence_level: str = "established"


@dataclass
class DrugDiseaseInteraction:
    """Represents a drug-disease contraindication."""

    drug: str
    condition: str
    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    alternative: Optional[str] = None


@dataclass
class CascadeRisk:
    """Represents a multi-factor compounding risk."""

    factors: List[str]
    combined_risk: str
    severity: InteractionSeverity
    mechanism: str
    recommendation: str


@dataclass
class InteractionWarning:
    """Warning about detected interaction."""

    warning_id: str
    interaction_type: InteractionType
    severity: InteractionSeverity
    involved_elements: List[str]
    description: str
    mechanism: str
    recommendation: str
    evidence: str = ""


# =============================================================================
# DRUG-DRUG INTERACTIONS DATABASE (Neurosurgery-Focused)
# =============================================================================

DRUG_DRUG_INTERACTIONS: List[DrugInteraction] = [
    # Anticoagulation interactions
    DrugInteraction(
        drug_a="warfarin",
        drug_b="aspirin",
        severity=InteractionSeverity.HIGH,
        mechanism="Additive anticoagulant/antiplatelet effect",
        clinical_effect="Significantly increased bleeding risk, ICH risk elevated",
        recommendation="Avoid combination in neurosurgical patients unless absolutely necessary. If required, ensure INR <2.0 and monitor closely.",
        evidence_level="established",
    ),
    DrugInteraction(
        drug_a="heparin",
        drug_b="aspirin",
        severity=InteractionSeverity.HIGH,
        mechanism="Additive anticoagulant/antiplatelet effect",
        clinical_effect="Increased bleeding risk",
        recommendation="Minimize concurrent use perioperatively. Consider timing separation.",
    ),
    DrugInteraction(
        drug_a="clopidogrel",
        drug_b="aspirin",
        severity=InteractionSeverity.MODERATE,
        mechanism="Dual antiplatelet therapy",
        clinical_effect="Increased bleeding risk, but may be indicated post-stent",
        recommendation="Hold both 7-10 days before elective cranial surgery if safe.",
    ),

    # AED interactions
    DrugInteraction(
        drug_a="phenytoin",
        drug_b="warfarin",
        severity=InteractionSeverity.HIGH,
        mechanism="Phenytoin induces CYP2C9, increasing warfarin metabolism",
        clinical_effect="Reduced warfarin efficacy → thrombosis risk",
        recommendation="Monitor INR closely (q2-3 days initially). May need warfarin dose increase 30-50%.",
    ),
    DrugInteraction(
        drug_a="phenytoin",
        drug_b="dexamethasone",
        severity=InteractionSeverity.MODERATE,
        mechanism="Phenytoin induces dexamethasone metabolism via CYP3A4",
        clinical_effect="Reduced steroid efficacy for vasogenic edema",
        recommendation="May need to increase dexamethasone dose by 50-100%. Consider alternative AED (levetiracetam).",
    ),
    DrugInteraction(
        drug_a="carbamazepine",
        drug_b="warfarin",
        severity=InteractionSeverity.HIGH,
        mechanism="CYP enzyme induction",
        clinical_effect="Reduced anticoagulation",
        recommendation="Monitor INR closely. Consider levetiracetam as alternative.",
    ),
    DrugInteraction(
        drug_a="valproic_acid",
        drug_b="phenytoin",
        severity=InteractionSeverity.MODERATE,
        mechanism="Valproate displaces phenytoin from protein binding",
        clinical_effect="Transient increase in free phenytoin → toxicity risk",
        recommendation="Monitor phenytoin free levels. Watch for toxicity signs.",
    ),
    DrugInteraction(
        drug_a="valproic_acid",
        drug_b="lamotrigine",
        severity=InteractionSeverity.MODERATE,
        mechanism="Valproate inhibits lamotrigine glucuronidation",
        clinical_effect="Doubled lamotrigine levels → SJS risk increased",
        recommendation="Reduce lamotrigine dose by 50% when adding valproate.",
    ),

    # Osmotherapy interactions
    DrugInteraction(
        drug_a="mannitol",
        drug_b="furosemide",
        severity=InteractionSeverity.MODERATE,
        mechanism="Additive diuretic effect",
        clinical_effect="Severe hypovolemia, electrolyte derangement",
        recommendation="Monitor volume status closely. Check Na, K, osmolality q4-6h.",
    ),
    DrugInteraction(
        drug_a="mannitol",
        drug_b="lithium",
        severity=InteractionSeverity.HIGH,
        mechanism="Mannitol-induced diuresis increases lithium concentration",
        clinical_effect="Lithium toxicity (tremor, confusion, arrhythmia)",
        recommendation="Hold lithium during osmotherapy. Monitor lithium levels.",
    ),

    # Sedation/Anesthesia interactions
    DrugInteraction(
        drug_a="propofol",
        drug_b="fentanyl",
        severity=InteractionSeverity.MODERATE,
        mechanism="Synergistic CNS depression",
        clinical_effect="Enhanced sedation, respiratory depression",
        recommendation="Reduce propofol dose by 20-30% when combined with fentanyl.",
    ),
    DrugInteraction(
        drug_a="midazolam",
        drug_b="phenytoin",
        severity=InteractionSeverity.MODERATE,
        mechanism="Phenytoin induces midazolam metabolism",
        clinical_effect="Reduced sedation efficacy",
        recommendation="May need 50-100% higher midazolam doses in chronic phenytoin users.",
    ),

    # Serotonin syndrome precursors
    DrugInteraction(
        drug_a="tramadol",
        drug_b="ssri",
        severity=InteractionSeverity.HIGH,
        mechanism="Both increase serotonin levels",
        clinical_effect="Serotonin syndrome risk (agitation, hyperthermia, clonus)",
        recommendation="Avoid combination. Use alternative analgesic (morphine, hydromorphone).",
    ),
    DrugInteraction(
        drug_a="fentanyl",
        drug_b="ssri",
        severity=InteractionSeverity.MODERATE,
        mechanism="Fentanyl has weak serotonergic activity",
        clinical_effect="Low risk of serotonin syndrome but possible",
        recommendation="Monitor for serotonin syndrome signs. Prefer morphine if concerned.",
    ),
    DrugInteraction(
        drug_a="linezolid",
        drug_b="ssri",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Linezolid is an MAO inhibitor",
        clinical_effect="High risk of serotonin syndrome",
        recommendation="AVOID combination. Use alternative antibiotic if possible. If linezolid required, hold SSRI for 2 weeks.",
    ),

    # Contrast-related
    DrugInteraction(
        drug_a="metformin",
        drug_b="contrast",
        severity=InteractionSeverity.MODERATE,
        mechanism="Contrast-induced nephropathy + metformin accumulation",
        clinical_effect="Lactic acidosis risk",
        recommendation="Hold metformin 48h before and after contrast. Check creatinine before resuming.",
    ),
]


# =============================================================================
# DRUG-DISEASE INTERACTIONS DATABASE
# =============================================================================

DRUG_DISEASE_INTERACTIONS: List[DrugDiseaseInteraction] = [
    # Renal impairment
    DrugDiseaseInteraction(
        drug="mannitol",
        condition="renal_failure",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Mannitol accumulates, cannot be excreted",
        clinical_effect="Volume overload, pulmonary edema, worsening renal failure",
        alternative="hypertonic_saline (23.4%)",
    ),
    DrugDiseaseInteraction(
        drug="gadolinium",
        condition="renal_failure",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Impaired gadolinium excretion",
        clinical_effect="Nephrogenic systemic fibrosis (NSF)",
        alternative="Non-contrast MRI or CT",
    ),
    DrugDiseaseInteraction(
        drug="levetiracetam",
        condition="renal_failure",
        severity=InteractionSeverity.MODERATE,
        mechanism="Renal excretion reduced",
        clinical_effect="Accumulation, CNS side effects",
        alternative="Dose reduce per GFR table",
    ),
    DrugDiseaseInteraction(
        drug="gabapentin",
        condition="renal_failure",
        severity=InteractionSeverity.MODERATE,
        mechanism="Renal excretion reduced",
        clinical_effect="Accumulation, sedation, myoclonus",
        alternative="Dose reduce per GFR table",
    ),

    # Hepatic impairment
    DrugDiseaseInteraction(
        drug="phenytoin",
        condition="hepatic_failure",
        severity=InteractionSeverity.HIGH,
        mechanism="Hepatic metabolism reduced, protein binding altered",
        clinical_effect="Toxicity at 'therapeutic' total levels (check free level)",
        alternative="levetiracetam",
    ),
    DrugDiseaseInteraction(
        drug="valproic_acid",
        condition="hepatic_failure",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Hepatic metabolism and protein binding affected",
        clinical_effect="Hepatotoxicity, hyperammonemia, coagulopathy",
        alternative="levetiracetam",
    ),
    DrugDiseaseInteraction(
        drug="carbamazepine",
        condition="hepatic_failure",
        severity=InteractionSeverity.HIGH,
        mechanism="Hepatic metabolism reduced",
        clinical_effect="Accumulation, hepatotoxicity",
        alternative="levetiracetam",
    ),

    # Cardiac conditions
    DrugDiseaseInteraction(
        drug="phenytoin",
        condition="cardiac_arrhythmia",
        severity=InteractionSeverity.HIGH,
        mechanism="Phenytoin has class Ib antiarrhythmic properties",
        clinical_effect="Bradycardia, AV block, hypotension with IV loading",
        alternative="levetiracetam",
    ),
    DrugDiseaseInteraction(
        drug="mannitol",
        condition="congestive_heart_failure",
        severity=InteractionSeverity.HIGH,
        mechanism="Initial volume expansion before diuresis",
        clinical_effect="Acute decompensation, pulmonary edema",
        alternative="hypertonic_saline with close monitoring",
    ),

    # Coagulation disorders
    DrugDiseaseInteraction(
        drug="aspirin",
        condition="coagulopathy",
        severity=InteractionSeverity.HIGH,
        mechanism="Further impairs platelet function",
        clinical_effect="Uncontrollable bleeding",
        alternative="Correct coagulopathy first",
    ),
    DrugDiseaseInteraction(
        drug="heparin",
        condition="heparin_induced_thrombocytopenia",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Antibody-mediated platelet activation",
        clinical_effect="Paradoxical thrombosis, limb-threatening ischemia",
        alternative="argatroban or bivalirudin",
    ),
    DrugDiseaseInteraction(
        drug="enoxaparin",
        condition="heparin_induced_thrombocytopenia",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Cross-reactivity with HIT antibodies",
        clinical_effect="Can trigger HIT",
        alternative="argatroban or bivalirudin",
    ),

    # Infection risk compounding
    DrugDiseaseInteraction(
        drug="dexamethasone",
        condition="active_infection",
        severity=InteractionSeverity.MODERATE,
        mechanism="Immunosuppression masks infection signs",
        clinical_effect="Progression of uncontrolled infection",
        alternative="Ensure adequate antimicrobial coverage before steroids",
    ),
    DrugDiseaseInteraction(
        drug="dexamethasone",
        condition="diabetes",
        severity=InteractionSeverity.MODERATE,
        mechanism="Glucocorticoid-induced hyperglycemia",
        clinical_effect="Poor glucose control → infection risk → poor wound healing",
        alternative="Intensive glucose monitoring and insulin adjustment",
    ),

    # Intracranial conditions
    DrugDiseaseInteraction(
        drug="dexamethasone",
        condition="traumatic_brain_injury",
        severity=InteractionSeverity.CRITICAL,
        mechanism="CRASH trial evidence",
        clinical_effect="Increased mortality",
        alternative="Supportive care, osmotherapy for edema",
    ),
    DrugDiseaseInteraction(
        drug="methylprednisolone",
        condition="spinal_cord_injury",
        severity=InteractionSeverity.CRITICAL,
        mechanism="NASCIS protocols now not recommended",
        clinical_effect="Infection, GI bleed, hyperglycemia without benefit",
        alternative="Supportive care, surgical decompression if indicated",
    ),
]


# =============================================================================
# CASCADE RISK PATTERNS
# =============================================================================

CASCADE_RISKS: List[CascadeRisk] = [
    CascadeRisk(
        factors=["anticoagulation", "recent_craniotomy", "tpa_consideration"],
        combined_risk="Catastrophic intracranial hemorrhage",
        severity=InteractionSeverity.CRITICAL,
        mechanism="TPA in recently operated brain is absolutely contraindicated",
        recommendation="TPA contraindicated within 3 months of craniotomy. Consider mechanical thrombectomy.",
    ),
    CascadeRisk(
        factors=["diabetes", "steroids", "recent_surgery"],
        combined_risk="Compound infection and wound healing risk",
        severity=InteractionSeverity.HIGH,
        mechanism="Steroids worsen glucose control → immunosuppression → surgical site infection",
        recommendation="Tight glucose control (140-180 mg/dL). Minimize steroid duration. Prophylactic antibiotics.",
    ),
    CascadeRisk(
        factors=["chronic_kidney_disease", "contrast", "nsaids"],
        combined_risk="Acute kidney injury",
        severity=InteractionSeverity.HIGH,
        mechanism="Triple hit: CKD baseline + contrast nephropathy + NSAID vasoconstriction",
        recommendation="Avoid NSAIDs. Pre-hydrate for contrast. Consider non-contrast imaging.",
    ),
    CascadeRisk(
        factors=["elderly", "anticoagulation", "fall_risk"],
        combined_risk="Intracranial hemorrhage from minor trauma",
        severity=InteractionSeverity.HIGH,
        mechanism="Age-related brain atrophy + anticoagulation = SDH/ICH risk",
        recommendation="Fall prevention. Consider reversal thresholds. Lower INR targets.",
    ),
    CascadeRisk(
        factors=["cirrhosis", "phenytoin", "coagulopathy"],
        combined_risk="Phenytoin toxicity and bleeding",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Liver disease → reduced phenytoin metabolism + low albumin + impaired clotting",
        recommendation="Use levetiracetam instead. Monitor free phenytoin if must use. Correct coagulopathy.",
    ),
    CascadeRisk(
        factors=["renal_failure", "mannitol", "hypertonic_saline"],
        combined_risk="Osmolar derangement without ICP benefit",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Cannot excrete osmolar load → no gradient established",
        recommendation="Avoid mannitol in ESRD. Hypertonic saline only with ICU monitoring. Consider decompressive surgery.",
    ),
    CascadeRisk(
        factors=["pregnancy", "anticoagulation", "craniotomy"],
        combined_risk="Maternal and fetal hemorrhage risk",
        severity=InteractionSeverity.CRITICAL,
        mechanism="Pregnancy hypercoagulable but also at surgical bleeding risk",
        recommendation="Multidisciplinary planning. Bridging with LMWH. Delivery planning if viable.",
    ),
    CascadeRisk(
        factors=["spinal_cord_injury", "immobility", "dvt_risk"],
        combined_risk="Venous thromboembolism",
        severity=InteractionSeverity.HIGH,
        mechanism="SCI + paralysis → stasis → DVT/PE",
        recommendation="Mechanical prophylaxis immediately. Pharmacologic prophylaxis when hemostasis assured (24-72h).",
    ),
]


# =============================================================================
# COMORBIDITY INTERACTION ANALYZER
# =============================================================================

class ComorbidityInteractionAnalyzer:
    """
    Analyzes patient context and content for dangerous comorbidity interactions.

    Usage:
        analyzer = ComorbidityInteractionAnalyzer()
        warnings = analyzer.analyze(
            patient_conditions={'diabetes', 'chronic_kidney_disease'},
            drugs_in_content=['mannitol', 'dexamethasone'],
            patient_context=patient_context
        )
    """

    def __init__(
        self,
        custom_drug_interactions: Optional[List[DrugInteraction]] = None,
        custom_disease_interactions: Optional[List[DrugDiseaseInteraction]] = None,
    ):
        self.drug_interactions = DRUG_DRUG_INTERACTIONS.copy()
        self.disease_interactions = DRUG_DISEASE_INTERACTIONS.copy()
        self.cascade_risks = CASCADE_RISKS.copy()

        if custom_drug_interactions:
            self.drug_interactions.extend(custom_drug_interactions)
        if custom_disease_interactions:
            self.disease_interactions.extend(custom_disease_interactions)

        # Build lookup indices
        self._build_indices()
        self.logger = logging.getLogger(__name__)

    def _build_indices(self):
        """Build fast lookup indices for interactions."""
        # Drug-drug interaction index
        self._drug_pair_index: Dict[Tuple[str, str], DrugInteraction] = {}
        for interaction in self.drug_interactions:
            key1 = (interaction.drug_a.lower(), interaction.drug_b.lower())
            key2 = (interaction.drug_b.lower(), interaction.drug_a.lower())
            self._drug_pair_index[key1] = interaction
            self._drug_pair_index[key2] = interaction

        # Drug-disease interaction index
        self._drug_disease_index: Dict[Tuple[str, str], DrugDiseaseInteraction] = {}
        for interaction in self.disease_interactions:
            key = (interaction.drug.lower(), interaction.condition.lower())
            self._drug_disease_index[key] = interaction

    def analyze(
        self,
        patient_conditions: Set[str],
        drugs_in_content: List[str],
        additional_factors: Optional[Set[str]] = None,
    ) -> List[InteractionWarning]:
        """
        Analyze for all types of interactions.

        Args:
            patient_conditions: Set of patient conditions/comorbidities
            drugs_in_content: List of drugs mentioned in content
            additional_factors: Additional risk factors (e.g., 'recent_craniotomy')

        Returns:
            List of interaction warnings
        """
        warnings: List[InteractionWarning] = []

        # Normalize inputs
        conditions_normalized = {c.lower().replace(" ", "_") for c in patient_conditions}
        drugs_normalized = [d.lower().replace(" ", "_") for d in drugs_in_content]
        factors = additional_factors or set()
        factors_normalized = {f.lower().replace(" ", "_") for f in factors}

        # 1. Check drug-drug interactions
        drug_drug_warnings = self._check_drug_drug_interactions(drugs_normalized)
        warnings.extend(drug_drug_warnings)

        # 2. Check drug-disease interactions
        drug_disease_warnings = self._check_drug_disease_interactions(
            drugs_normalized, conditions_normalized
        )
        warnings.extend(drug_disease_warnings)

        # 3. Check cascade risks
        all_factors = conditions_normalized | factors_normalized | set(drugs_normalized)
        cascade_warnings = self._check_cascade_risks(all_factors)
        warnings.extend(cascade_warnings)

        # Sort by severity
        severity_order = {
            InteractionSeverity.CRITICAL: 0,
            InteractionSeverity.HIGH: 1,
            InteractionSeverity.MODERATE: 2,
            InteractionSeverity.LOW: 3,
            InteractionSeverity.INFORMATIONAL: 4,
        }
        warnings.sort(key=lambda w: severity_order[w.severity])

        return warnings

    def _check_drug_drug_interactions(
        self, drugs: List[str]
    ) -> List[InteractionWarning]:
        """Check for drug-drug interactions among drugs in content."""
        warnings = []

        # Check all pairs
        for i, drug_a in enumerate(drugs):
            for drug_b in drugs[i+1:]:
                key = (drug_a, drug_b)
                if key in self._drug_pair_index:
                    interaction = self._drug_pair_index[key]
                    warnings.append(InteractionWarning(
                        warning_id=f"DDI_{drug_a}_{drug_b}",
                        interaction_type=InteractionType.DRUG_DRUG,
                        severity=interaction.severity,
                        involved_elements=[drug_a, drug_b],
                        description=f"Drug interaction: {drug_a} + {drug_b}",
                        mechanism=interaction.mechanism,
                        recommendation=interaction.recommendation,
                        evidence=interaction.evidence_level,
                    ))

        return warnings

    def _check_drug_disease_interactions(
        self,
        drugs: List[str],
        conditions: Set[str],
    ) -> List[InteractionWarning]:
        """Check for drug-disease contraindications."""
        warnings = []

        for drug in drugs:
            for condition in conditions:
                key = (drug, condition)
                if key in self._drug_disease_index:
                    interaction = self._drug_disease_index[key]
                    warnings.append(InteractionWarning(
                        warning_id=f"DCI_{drug}_{condition}",
                        interaction_type=InteractionType.DRUG_DISEASE,
                        severity=interaction.severity,
                        involved_elements=[drug, condition],
                        description=f"{drug.title()} contraindicated with {condition.replace('_', ' ')}",
                        mechanism=interaction.mechanism,
                        recommendation=interaction.alternative or "Avoid this drug",
                        evidence="Clinical guidelines",
                    ))

        return warnings

    def _check_cascade_risks(
        self, all_factors: Set[str]
    ) -> List[InteractionWarning]:
        """Check for cascade risk patterns."""
        warnings = []

        for cascade in self.cascade_risks:
            cascade_factors = {f.lower().replace(" ", "_") for f in cascade.factors}

            # Check if all cascade factors are present
            if cascade_factors.issubset(all_factors):
                warnings.append(InteractionWarning(
                    warning_id=f"CASCADE_{'_'.join(cascade.factors[:2])}",
                    interaction_type=InteractionType.CASCADE_RISK,
                    severity=cascade.severity,
                    involved_elements=cascade.factors,
                    description=cascade.combined_risk,
                    mechanism=cascade.mechanism,
                    recommendation=cascade.recommendation,
                ))
            # Also check partial matches (2 of 3 factors) for HIGH/CRITICAL cascades
            elif cascade.severity in (InteractionSeverity.CRITICAL, InteractionSeverity.HIGH):
                matching_factors = cascade_factors.intersection(all_factors)
                if len(matching_factors) >= 2:
                    warnings.append(InteractionWarning(
                        warning_id=f"CASCADE_PARTIAL_{'_'.join(list(matching_factors)[:2])}",
                        interaction_type=InteractionType.CASCADE_RISK,
                        severity=InteractionSeverity.MODERATE,  # Downgrade for partial
                        involved_elements=list(matching_factors),
                        description=f"Partial cascade risk for: {cascade.combined_risk}",
                        mechanism=cascade.mechanism,
                        recommendation=f"Monitor for: {cascade.recommendation}",
                    ))

        return warnings

    def extract_drugs_from_content(self, content: str) -> List[str]:
        """
        Extract drug names mentioned in content.

        This is a helper to identify drugs for interaction checking.
        """
        # Common neurosurgical drugs to search for
        drug_patterns = [
            # Osmotherapy
            r"\bmannitol\b",
            r"\bhypertonic\s+saline\b",
            # AEDs
            r"\blevetiracetam\b|\bkeppra\b",
            r"\bphenytoin\b|\bdilantin\b",
            r"\bvalproic\s+acid\b|\bdepakote\b|\bvalproate\b",
            r"\bcarbamazepine\b|\btegretol\b",
            r"\blacosamide\b|\bvimpat\b",
            r"\bgabapentin\b|\bneurontin\b",
            r"\bpregabalin\b|\blyrica\b",
            r"\blamotrigine\b|\blamictal\b",
            # Steroids
            r"\bdexamethasone\b|\bdecadron\b",
            r"\bmethylprednisolone\b|\bsolu-?medrol\b",
            # Anticoagulants
            r"\bwarfarin\b|\bcoumadin\b",
            r"\bheparin\b",
            r"\benoxaparin\b|\blovenox\b",
            r"\baspirin\b|\basa\b",
            r"\bclopidogrel\b|\bplavix\b",
            # Analgesics
            r"\bfentanyl\b",
            r"\bmorphine\b",
            r"\bhydromorphone\b|\bdilaudid\b",
            r"\btramadol\b",
            # Sedatives
            r"\bpropofol\b|\bdiprivan\b",
            r"\bmidazolam\b|\bversed\b",
            # Antibiotics
            r"\bvancomycin\b",
            r"\bceftriaxone\b|\brocephin\b",
            r"\blinezolid\b|\bzyvox\b",
            # SSRIs (for serotonin syndrome checking)
            r"\bssri\b",
            r"\bsertraline\b|\bzoloft\b",
            r"\bfluoxetine\b|\bprozac\b",
            r"\bescitalopram\b|\blexapro\b",
            # Contrast
            r"\bcontrast\b|\bgadolinium\b",
            # Other
            r"\bmetformin\b",
            r"\blithium\b",
            r"\bfurosemide\b|\blasix\b",
            r"\btpa\b|\balteplase\b|\bactivase\b",
        ]

        content_lower = content.lower()
        found_drugs = set()

        for pattern in drug_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                # Normalize drug name
                drug = match.replace("-", "").replace(" ", "_")
                # Map brand names to generic
                brand_to_generic = {
                    "keppra": "levetiracetam",
                    "dilantin": "phenytoin",
                    "depakote": "valproic_acid",
                    "tegretol": "carbamazepine",
                    "vimpat": "lacosamide",
                    "neurontin": "gabapentin",
                    "lyrica": "pregabalin",
                    "lamictal": "lamotrigine",
                    "decadron": "dexamethasone",
                    "solumedrol": "methylprednisolone",
                    "coumadin": "warfarin",
                    "lovenox": "enoxaparin",
                    "plavix": "clopidogrel",
                    "dilaudid": "hydromorphone",
                    "diprivan": "propofol",
                    "versed": "midazolam",
                    "rocephin": "ceftriaxone",
                    "zyvox": "linezolid",
                    "zoloft": "sertraline",
                    "prozac": "fluoxetine",
                    "lexapro": "escitalopram",
                    "lasix": "furosemide",
                    "alteplase": "tpa",
                    "activase": "tpa",
                }
                drug = brand_to_generic.get(drug, drug)
                found_drugs.add(drug)

        return list(found_drugs)

    def get_interaction_summary(
        self, warnings: List[InteractionWarning]
    ) -> Dict[str, Any]:
        """Generate summary of interaction warnings."""
        critical_count = sum(1 for w in warnings if w.severity == InteractionSeverity.CRITICAL)
        high_count = sum(1 for w in warnings if w.severity == InteractionSeverity.HIGH)

        return {
            "total_warnings": len(warnings),
            "critical_count": critical_count,
            "high_count": high_count,
            "requires_intervention": critical_count > 0,
            "by_type": {
                t.value: sum(1 for w in warnings if w.interaction_type == t)
                for t in InteractionType
            },
            "warnings": [
                {
                    "id": w.warning_id,
                    "severity": w.severity.value,
                    "type": w.interaction_type.value,
                    "description": w.description,
                    "recommendation": w.recommendation,
                }
                for w in warnings
            ],
        }
