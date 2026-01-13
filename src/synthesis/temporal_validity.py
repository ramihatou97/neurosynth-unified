"""
Temporal Validity Checker for Superseded Practices
===================================================

P0 Enhancement: Temporal Validity Checking

This module detects outdated or superseded medical practices in content,
preventing dangerous recommendations based on deprecated guidelines.

Clinical Safety Rationale:
- NASCIS protocol (1990s) for SCI: No longer recommended, potential harm
- BTF 3rd Edition ICP threshold (20mmHg): Superseded by 4th Edition (22mmHg)
- Routine hyperventilation for ICP: Now recognized as harmful
- High-dose steroids in TBI: CRASH trial showed increased mortality

Detection Method:
1. Pattern matching for superseded practice mentions
2. Guideline version checking
3. Citation date analysis
4. Explicit supersession warnings
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SupersessionSeverity(Enum):
    """Severity of superseded practice detection."""

    CRITICAL = "critical"      # Actively harmful (NASCIS, steroids in TBI)
    HIGH = "high"              # Outdated guideline, significant change
    MEDIUM = "medium"          # Minor guideline update
    INFORMATIONAL = "info"     # Newer evidence available but old still valid


class SupersessionType(Enum):
    """Type of supersession."""

    CONTRAINDICATED = "contraindicated"   # Now known to be harmful
    REPLACED = "replaced"                 # Superseded by newer guideline
    REFINED = "refined"                   # Updated with modifications
    DEPRECATED = "deprecated"             # No longer recommended
    WITHDRAWN = "withdrawn"               # Withdrawn from market/use


@dataclass
class SupersededPractice:
    """
    Represents a superseded medical practice or guideline.

    Attributes:
        practice_id: Unique identifier
        pattern: Regex pattern to detect the practice
        keywords: Keywords that suggest this practice
        original_practice: Description of the old practice
        superseded_by: What replaced it
        reason: Why it was superseded
        evidence: Key evidence (trial name, year)
        severity: How dangerous is following old practice
        supersession_type: Type of change
        supersession_year: When it was superseded
        current_recommendation: Current best practice
    """

    practice_id: str
    pattern: str
    keywords: List[str]
    original_practice: str
    superseded_by: str
    reason: str
    evidence: str
    severity: SupersessionSeverity
    supersession_type: SupersessionType
    supersession_year: int
    current_recommendation: str
    specialty_tags: List[str] = field(default_factory=list)
    additional_context: str = ""


# =============================================================================
# SUPERSEDED PRACTICES DATABASE
# =============================================================================

SUPERSEDED_PRACTICES: List[SupersededPractice] = [
    # =========================================================================
    # CRITICAL: Actively Harmful Practices
    # =========================================================================
    SupersededPractice(
        practice_id="NASCIS_SCI",
        pattern=r"(?i)methylprednisolone.*spinal\s+cord|nascis\s+protocol|30\s*mg/kg.*spinal|5\.4\s*mg/kg/hr.*spinal",
        keywords=["nascis", "methylprednisolone spinal cord", "high dose steroids sci"],
        original_practice="NASCIS protocol: Methylprednisolone 30mg/kg bolus + 5.4mg/kg/hr for 23-48h in acute SCI",
        superseded_by="No high-dose steroids for spinal cord injury",
        reason="Post-hoc analysis only, marginal benefit, significant complications (infection, GI bleed, hyperglycemia)",
        evidence="NASCIS I (1984), II (1990), III (1997) - heavily criticized methodology; Congress of Neurological Surgeons 2013 guidelines recommend against",
        severity=SupersessionSeverity.CRITICAL,
        supersession_type=SupersessionType.CONTRAINDICATED,
        supersession_year=2013,
        current_recommendation="No role for high-dose methylprednisolone in acute SCI per current guidelines. Consider neuroprotective strategies under investigation.",
        specialty_tags=["spine", "trauma"],
        additional_context="Some centers still offer as option with informed consent, but not standard of care.",
    ),
    SupersededPractice(
        practice_id="STEROIDS_TBI",
        pattern=r"(?i)steroids?\s+(for|in)\s+(?:traumatic\s+)?brain\s+injury|(?:dexa|methyl).*tbi|crash\s+trial",
        keywords=["steroids tbi", "dexamethasone brain injury", "methylprednisolone head injury"],
        original_practice="High-dose corticosteroids for traumatic brain injury",
        superseded_by="Corticosteroids contraindicated in TBI",
        reason="CRASH trial showed 21.1% vs 17.9% mortality - steroids INCREASE death",
        evidence="CRASH trial (Lancet 2004): 10,008 patients, terminated early for harm",
        severity=SupersessionSeverity.CRITICAL,
        supersession_type=SupersessionType.CONTRAINDICATED,
        supersession_year=2004,
        current_recommendation="Corticosteroids are CONTRAINDICATED in traumatic brain injury. Use other ICP management strategies.",
        specialty_tags=["trauma", "neurocritical_care"],
    ),
    SupersededPractice(
        practice_id="ROUTINE_HYPERVENTILATION",
        pattern=r"(?i)prophylactic\s+hyperventilation|routine\s+hyperventilation|hyperventilat.*icp.*first\s+24|paco2.*25.*30\s*mmhg",
        keywords=["prophylactic hyperventilation", "routine hyperventilation icp"],
        original_practice="Prophylactic or routine hyperventilation (PaCO2 25-30 mmHg) for ICP control",
        superseded_by="Hyperventilation only for acute herniation, brief duration",
        reason="Hyperventilation causes cerebral vasoconstriction, reducing CBF to already injured brain. Causes secondary ischemia.",
        evidence="BTF Guidelines 4th Edition 2016; Multiple studies showing worse outcomes with prolonged hyperventilation",
        severity=SupersessionSeverity.CRITICAL,
        supersession_type=SupersessionType.CONTRAINDICATED,
        supersession_year=2007,
        current_recommendation="Avoid prophylactic hyperventilation. Reserve for acute herniation crisis only, brief duration (<30 min). Target normocapnia (PaCO2 35-40 mmHg).",
        specialty_tags=["trauma", "neurocritical_care"],
    ),

    # =========================================================================
    # HIGH: Significant Guideline Changes
    # =========================================================================
    SupersededPractice(
        practice_id="BTF_ICP_20",
        pattern=r"(?i)icp\s*[>≥]\s*20\s*mmhg|treat\s+icp\s+above\s+20|20\s*mmhg\s+threshold",
        keywords=["icp threshold 20", "treat icp 20"],
        original_practice="ICP treatment threshold of 20 mmHg (BTF 3rd Edition)",
        superseded_by="ICP treatment threshold of 22 mmHg (BTF 4th Edition 2016)",
        reason="Refined analysis showed 22 mmHg as more evidence-based threshold",
        evidence="BTF 4th Edition 2016; Carney et al., Neurosurgery 2017",
        severity=SupersessionSeverity.HIGH,
        supersession_type=SupersessionType.REFINED,
        supersession_year=2016,
        current_recommendation="Treat ICP > 22 mmHg per BTF 4th Edition (2016). Consider patient-specific factors.",
        specialty_tags=["trauma", "neurocritical_care"],
        additional_context="The change from 20 to 22 mmHg is evidence-based but clinical judgment still applies.",
    ),
    SupersededPractice(
        practice_id="CPP_AGGRESSIVE",
        pattern=r"(?i)cpp\s*[>≥]\s*70|target\s+cpp\s+(?:above|over)\s+70|cpp.*80",
        keywords=["cpp above 70", "aggressive cpp", "cpp 80"],
        original_practice="Aggressive CPP targets > 70 mmHg",
        superseded_by="CPP target 60-70 mmHg, avoid > 70 mmHg",
        reason="CPP > 70 mmHg associated with ARDS and other respiratory complications without neurological benefit",
        evidence="BTF 4th Edition 2016; Robertson et al., Crit Care Med 1999",
        severity=SupersessionSeverity.HIGH,
        supersession_type=SupersessionType.REFINED,
        supersession_year=2016,
        current_recommendation="Target CPP 60-70 mmHg. Avoid > 70 mmHg due to ARDS risk. Minimum 60 mmHg.",
        specialty_tags=["trauma", "neurocritical_care"],
    ),
    SupersededPractice(
        practice_id="SURGICAL_ICH_STICH",
        pattern=r"(?i)early\s+surgery\s+(?:for\s+)?(?:all\s+)?ich|routine\s+evacuation.*intracerebral|operate.*supratentorial.*hemorrhage",
        keywords=["early surgery ich", "routine evacuation ich", "surgical ich supratentorial"],
        original_practice="Early surgical evacuation for supratentorial ICH",
        superseded_by="Surgery not beneficial for most supratentorial ICH",
        reason="STICH and STICH II trials showed no benefit for routine early surgery",
        evidence="STICH (2005): 26% vs 24% favorable; STICH II (2013): confirmed no benefit for lobar ICH",
        severity=SupersessionSeverity.HIGH,
        supersession_type=SupersessionType.REPLACED,
        supersession_year=2013,
        current_recommendation="Surgery for ICH should be individualized. Consider for: posterior fossa hemorrhage, deteriorating patient, large lobar hemorrhage with mass effect. Routine early surgery not recommended.",
        specialty_tags=["vascular", "neurocritical_care"],
    ),
    SupersededPractice(
        practice_id="ARUBA_INTERVENTION",
        pattern=r"(?i)treat\s+(?:all\s+)?unruptured\s+avm|intervention\s+for\s+unruptured\s+avm|prophylactic.*avm\s+treatment",
        keywords=["treat unruptured avm", "prophylactic avm", "intervene unruptured avm"],
        original_practice="Routine intervention for unruptured AVMs",
        superseded_by="Medical management often superior for unruptured AVMs",
        reason="ARUBA trial showed higher stroke/death with intervention (30.7%) vs medical (10.1%) at 33 months",
        evidence="ARUBA trial (Lancet 2014): Terminated early due to clear difference",
        severity=SupersessionSeverity.HIGH,
        supersession_type=SupersessionType.REPLACED,
        supersession_year=2014,
        current_recommendation="Shared decision-making for unruptured AVMs. Medical management is reasonable, especially for deep/eloquent AVMs. Consider treatment for: young patients, superficial AVMs, after AVM rupture.",
        specialty_tags=["vascular"],
        additional_context="ARUBA has limitations (short follow-up, heterogeneous treatment). Long-term data may modify recommendations.",
    ),

    # =========================================================================
    # MEDIUM: Minor Updates or Refinements
    # =========================================================================
    SupersededPractice(
        practice_id="DECRA_EARLY",
        pattern=r"(?i)early\s+(?:bifrontal\s+)?decompressive\s+craniectomy|decra\s+protocol|craniectomy.*icp.*20",
        keywords=["early decompressive craniectomy", "decra", "bifrontal dc"],
        original_practice="Early bifrontal DC for moderate ICP elevation (>20 mmHg for 15 min)",
        superseded_by="DC for refractory ICP after medical management",
        reason="DECRA showed surgery reduced ICP but worsened functional outcomes; too early intervention",
        evidence="DECRA trial (NEJM 2011): More unfavorable outcomes despite lower ICP",
        severity=SupersessionSeverity.MEDIUM,
        supersession_type=SupersessionType.REFINED,
        supersession_year=2011,
        current_recommendation="Reserve decompressive craniectomy for refractory ICP (> 22 mmHg) after maximal medical therapy. Consider RESCUEicp criteria. Discuss mortality/morbidity trade-offs with family.",
        specialty_tags=["trauma", "neurocritical_care"],
    ),
    SupersededPractice(
        practice_id="HEPARIN_STROKE",
        pattern=r"(?i)full\s+(?:dose\s+)?heparin.*(?:ischemic\s+)?stroke|anticoagulat.*acute\s+stroke|heparin.*prevent.*stroke\s+progression",
        keywords=["heparin ischemic stroke", "anticoagulation acute stroke"],
        original_practice="Full-dose heparin for acute ischemic stroke",
        superseded_by="No benefit from full anticoagulation in acute ischemic stroke",
        reason="IST showed no net benefit from heparin in acute stroke",
        evidence="International Stroke Trial (IST) 1997",
        severity=SupersessionSeverity.MEDIUM,
        supersession_type=SupersessionType.DEPRECATED,
        supersession_year=1997,
        current_recommendation="Antiplatelet therapy (aspirin 160-325 mg) for acute ischemic stroke. Reserve anticoagulation for specific indications (atrial fibrillation after 24-48h, cardiac thrombus).",
        specialty_tags=["vascular", "stroke"],
    ),
    SupersededPractice(
        practice_id="WHO_2016_GLIOMA",
        pattern=r"(?i)who\s+2016.*glioma|2016\s+(?:cns\s+)?tumor\s+classification|glioblastoma.*secondary",
        keywords=["who 2016 glioma", "secondary glioblastoma"],
        original_practice="WHO 2016 CNS Tumor Classification",
        superseded_by="WHO 2021 CNS Tumor Classification",
        reason="Refined molecular criteria, new integrated diagnoses, CDKN2A/B deletion for grade 4 astrocytoma",
        evidence="WHO 2021 CNS Tumor Classification (5th Edition)",
        severity=SupersessionSeverity.MEDIUM,
        supersession_type=SupersessionType.REPLACED,
        supersession_year=2021,
        current_recommendation="Use WHO 2021 classification. Key changes: IDH-mutant astrocytoma grade 4 with CDKN2A/B deletion; molecular-first diagnosis; elimination of 'secondary GBM' term.",
        specialty_tags=["tumor", "neuropathology"],
    ),

    # =========================================================================
    # INFORMATIONAL: Evolving Evidence
    # =========================================================================
    SupersededPractice(
        practice_id="ATACH2_INTENSIVE_BP",
        pattern=r"(?i)sbp\s*<\s*110|intensive\s+bp.*110.*139|atach.?2\s+protocol",
        keywords=["intensive bp ich", "sbp 110", "atach2"],
        original_practice="Very intensive BP lowering (SBP 110-139 mmHg) in ICH",
        superseded_by="Standard BP target (SBP < 140 mmHg) equally effective",
        reason="ATACH-2 showed no benefit of intensive vs standard BP lowering, more renal events with intensive",
        evidence="ATACH-2 trial (NEJM 2016)",
        severity=SupersessionSeverity.INFORMATIONAL,
        supersession_type=SupersessionType.REFINED,
        supersession_year=2016,
        current_recommendation="Target SBP < 140 mmHg within 1 hour for ICH (INTERACT2). Intensive lowering (< 110-120) not beneficial and may cause renal injury.",
        specialty_tags=["vascular", "stroke", "neurocritical_care"],
    ),
]


# =============================================================================
# DEPRECATED GUIDELINE VERSIONS
# =============================================================================

GUIDELINE_VERSIONS: Dict[str, Dict[str, Any]] = {
    "BTF": {
        "current_version": "4th Edition",
        "current_year": 2016,
        "previous_versions": [
            {"version": "3rd Edition", "year": 2007, "key_changes": ["ICP threshold was 20 mmHg"]},
            {"version": "2nd Edition", "year": 2000, "key_changes": ["Less evidence-based"]},
            {"version": "1st Edition", "year": 1995, "key_changes": ["Initial guidelines"]},
        ],
        "detection_patterns": [
            r"btf\s+(?:3rd|third)\s+edition",
            r"btf\s+2007",
            r"brain\s+trauma\s+foundation.*200[0-7]",
        ],
    },
    "WHO_CNS": {
        "current_version": "5th Edition",
        "current_year": 2021,
        "previous_versions": [
            {"version": "4th Edition (2016)", "year": 2016, "key_changes": ["Integrated diagnosis introduced"]},
            {"version": "3rd Edition", "year": 2007, "key_changes": ["Traditional histological"]},
        ],
        "detection_patterns": [
            r"who\s+2016",
            r"who\s+(?:4th|fourth)\s+edition",
            r"cns\s+classification.*2016",
        ],
    },
    "AHA_Stroke": {
        "current_version": "2019",
        "current_year": 2019,
        "previous_versions": [
            {"version": "2018", "year": 2018},
            {"version": "2015", "year": 2015},
        ],
        "detection_patterns": [
            r"aha.*stroke.*201[0-8]",
            r"stroke\s+guidelines.*201[0-8]",
        ],
    },
}


@dataclass
class TemporalValidityWarning:
    """Warning about temporal validity issue in content."""

    warning_id: str
    severity: SupersessionSeverity
    supersession_type: SupersessionType
    detected_text: str
    original_practice: str
    current_recommendation: str
    evidence: str
    supersession_year: int
    location_hint: str = ""  # Where in content this was found


class TemporalValidityChecker:
    """
    Checks content for superseded practices and outdated guidelines.

    Usage:
        checker = TemporalValidityChecker()
        warnings = checker.check_content(content)

        for warning in warnings:
            if warning.severity == SupersessionSeverity.CRITICAL:
                # Flag as AUTO-CRITICAL gap
                pass
    """

    def __init__(self, custom_practices: Optional[List[SupersededPractice]] = None):
        """
        Initialize the checker.

        Args:
            custom_practices: Additional superseded practices to check
        """
        self.practices = SUPERSEDED_PRACTICES.copy()
        if custom_practices:
            self.practices.extend(custom_practices)

        # Pre-compile patterns for efficiency
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        for practice in self.practices:
            try:
                self._compiled_patterns[practice.practice_id] = re.compile(
                    practice.pattern, re.IGNORECASE
                )
            except re.error as e:
                logger.warning(f"Invalid regex for {practice.practice_id}: {e}")

        self.logger = logging.getLogger(__name__)

    def check_content(
        self,
        content: str,
        specialty_filter: Optional[List[str]] = None,
    ) -> List[TemporalValidityWarning]:
        """
        Check content for superseded practices.

        Args:
            content: Text content to analyze
            specialty_filter: Only check practices relevant to these specialties

        Returns:
            List of temporal validity warnings
        """
        warnings: List[TemporalValidityWarning] = []
        content_lower = content.lower()

        for practice in self.practices:
            # Apply specialty filter if provided
            if specialty_filter:
                if not any(tag in specialty_filter for tag in practice.specialty_tags):
                    continue

            # First check keywords for quick filtering
            keyword_match = any(
                kw.lower() in content_lower
                for kw in practice.keywords
            )

            if not keyword_match:
                continue

            # Then apply regex pattern for precision
            pattern = self._compiled_patterns.get(practice.practice_id)
            if pattern:
                match = pattern.search(content)
                if match:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]

                    warnings.append(TemporalValidityWarning(
                        warning_id=f"TEMPORAL_{practice.practice_id}",
                        severity=practice.severity,
                        supersession_type=practice.supersession_type,
                        detected_text=match.group(),
                        original_practice=practice.original_practice,
                        current_recommendation=practice.current_recommendation,
                        evidence=practice.evidence,
                        supersession_year=practice.supersession_year,
                        location_hint=f"...{context}...",
                    ))

        # Also check for outdated guideline versions
        guideline_warnings = self._check_guideline_versions(content)
        warnings.extend(guideline_warnings)

        # Sort by severity (CRITICAL first)
        severity_order = {
            SupersessionSeverity.CRITICAL: 0,
            SupersessionSeverity.HIGH: 1,
            SupersessionSeverity.MEDIUM: 2,
            SupersessionSeverity.INFORMATIONAL: 3,
        }
        warnings.sort(key=lambda w: severity_order[w.severity])

        return warnings

    def _check_guideline_versions(self, content: str) -> List[TemporalValidityWarning]:
        """Check for references to outdated guideline versions."""
        warnings = []

        for guideline_name, info in GUIDELINE_VERSIONS.items():
            for pattern_str in info.get("detection_patterns", []):
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    match = pattern.search(content)
                    if match:
                        warnings.append(TemporalValidityWarning(
                            warning_id=f"OUTDATED_GUIDELINE_{guideline_name}",
                            severity=SupersessionSeverity.HIGH,
                            supersession_type=SupersessionType.REPLACED,
                            detected_text=match.group(),
                            original_practice=f"Reference to outdated {guideline_name} version",
                            current_recommendation=f"Use {info['current_version']} ({info['current_year']})",
                            evidence=f"Current version: {info['current_version']} ({info['current_year']})",
                            supersession_year=info["current_year"],
                        ))
                except re.error:
                    continue

        return warnings

    def check_citation_recency(
        self,
        citations: List[Dict[str, Any]],
        max_age_years: int = 10,
        critical_topics: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check citation recency for topics where evidence evolves quickly.

        Args:
            citations: List of citations with 'year' and optionally 'topic' fields
            max_age_years: Maximum acceptable age for citations
            critical_topics: Topics where recency is especially important

        Returns:
            List of recency warnings
        """
        current_year = date.today().year
        warnings = []

        critical_topics = critical_topics or {
            "trauma", "tbi", "stroke", "glioblastoma", "tumor classification",
            "icp management", "decompressive craniectomy"
        }

        for citation in citations:
            year = citation.get("year")
            if not year or not isinstance(year, int):
                continue

            age = current_year - year
            topic = citation.get("topic", "").lower()

            is_critical_topic = any(
                ct in topic for ct in critical_topics
            )

            if age > max_age_years:
                severity = "HIGH" if is_critical_topic else "MEDIUM"
                warnings.append({
                    "citation": citation,
                    "age_years": age,
                    "severity": severity,
                    "message": f"Citation from {year} ({age} years old) on rapidly evolving topic",
                    "recommendation": "Verify current guidelines and recent trials",
                })

        return warnings

    def get_superseded_practice_by_id(
        self,
        practice_id: str
    ) -> Optional[SupersededPractice]:
        """Get full details of a superseded practice by ID."""
        for practice in self.practices:
            if practice.practice_id == practice_id:
                return practice
        return None

    def list_critical_supersessions(self) -> List[SupersededPractice]:
        """List all CRITICAL severity superseded practices."""
        return [
            p for p in self.practices
            if p.severity == SupersessionSeverity.CRITICAL
        ]

    def get_recommendations_summary(
        self,
        warnings: List[TemporalValidityWarning]
    ) -> Dict[str, Any]:
        """
        Generate a summary of recommendations from warnings.

        Returns structured recommendations for gap filling.
        """
        critical_count = sum(1 for w in warnings if w.severity == SupersessionSeverity.CRITICAL)
        high_count = sum(1 for w in warnings if w.severity == SupersessionSeverity.HIGH)

        recommendations = []
        for warning in warnings:
            recommendations.append({
                "original": warning.original_practice,
                "current": warning.current_recommendation,
                "evidence": warning.evidence,
                "severity": warning.severity.value,
            })

        return {
            "total_warnings": len(warnings),
            "critical_count": critical_count,
            "high_count": high_count,
            "requires_immediate_correction": critical_count > 0,
            "recommendations": recommendations,
        }
