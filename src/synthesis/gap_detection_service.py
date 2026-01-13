"""
20-Stage Neurosurgical Gap Detection Service
=============================================

Proactively detects knowledge gaps in synthesized content using
comprehensive neurosurgical ontology and domain expertise.

20 Detection Stages:
1. Subspecialty Classification
2. Structural Analysis (canonical sections)
3. Danger Zone Coverage
4. Procedural Steps Analysis
5. Bailout/Complication Coverage
6. Instrument Coverage
7. Measurement Analysis (BTF guidelines)
8. Decision Point Analysis
9. Evidence Level Analysis
10. Source-Based Analysis
11. Visual/Imaging Analysis
12. DISORDER-Specific Analysis
13. ANATOMY-Specific Analysis
14. CONCEPT-Specific Analysis
15. Patient Context Analysis (P0 Enhancement)
16. Temporal Validity Analysis (P0 Enhancement)
17. Comorbidity Interaction Analysis (P1 Enhancement)
18. Trajectory-Aware Danger Zones (P1 Enhancement)
19. Negative Constraint Recognition (P2 Enhancement)
20. Disease Stage Context (P2 Enhancement)

Safety-Critical Gap Types (AUTO-CRITICAL):
- DANGER_ZONE: Missing safety-critical anatomy
- BAILOUT: Missing complication management
- MEASUREMENT: Missing quantitative thresholds
- SUPERSEDED_PRACTICE: Reference to contraindicated practice (P0)
- PATIENT_CONTEXT_VIOLATION: Contraindication for patient demographics (P0)
- COMORBIDITY_INTERACTION: Drug/condition interaction (P1)
- TRAJECTORY_DANGER: Approach-specific danger zone (P1)
- STAGE_MISMATCH: Stage-inappropriate recommendation (P2)
"""

import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set

from .gap_models import (
    Gap,
    GapPriority,
    GapReport,
    GapType,
    SAFETY_CRITICAL_TYPES,
    TemplateType,
)
from .patient_context import (
    PatientContext,
    PatientContextAnalyzer,
    AgeGroup,
)
from .temporal_validity import (
    TemporalValidityChecker,
    SupersessionSeverity,
    TemporalValidityWarning,
)
from .comorbidity_interactions import (
    ComorbidityInteractionAnalyzer,
    InteractionSeverity,
    InteractionWarning,
)
from .trajectory_danger_zones import (
    TrajectoryDangerZoneAnalyzer,
    SurgicalApproach,
)
from .negative_constraints import (
    NegativeConstraintExtractor,
    NegativeConstraint,
    ConstraintSeverity,
)
from .disease_stage_context import (
    DiseaseStageAnalyzer,
    DiseaseStage,
    StageContextWarning,
)
from .neurosurgical_ontology import (
    ARTERIAL_SEGMENTS,
    CRANIAL_FORAMINA,
    DANGER_ZONES,
    DISORDER_REQUIRED_ELEMENTS,
    ANATOMY_REQUIRED_ELEMENTS,
    CONCEPT_REQUIRED_ELEMENTS,
    ICP_PARAMETERS,
    LANDMARK_TRIALS,
    PROCEDURAL_TEMPLATES,
    NeurosurgicalOntology,
)
from .subspecialty_classifier import SubspecialtyClassifier, Subspecialty

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Protocol for embedding service interface."""

    async def embed_text(self, text: str) -> List[float]:
        ...


class QARepository(Protocol):
    """Protocol for Q&A repository interface."""

    async def get_related_questions(
        self, topic: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        ...


@dataclass
class SearchResult:
    """Represents a search result from the corpus."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    entity_names: List[str] = None
    cuis: List[str] = None
    source_title: str = ""
    embedding: List[float] = None

    def __post_init__(self):
        if self.entity_names is None:
            self.entity_names = []
        if self.cuis is None:
            self.cuis = []


class GapDetectionService:
    """
    14-stage gap detection service for neurosurgical content.

    Usage:
        service = GapDetectionService(embedding_service, qa_repository)
        await service.initialize()

        gap_report = await service.detect_gaps(
            topic="pterional craniotomy",
            template_type=TemplateType.PROCEDURAL,
            search_results=results,
        )
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qa_repository: Optional[QARepository] = None,
    ):
        self.embedding_service = embedding_service
        self.qa_repo = qa_repository
        self.subspecialty_classifier = SubspecialtyClassifier(embedding_service)
        self._is_initialized = False

        # P0 Enhancement: Patient Context and Temporal Validity
        self.patient_context_analyzer = PatientContextAnalyzer()
        self.temporal_validity_checker = TemporalValidityChecker()

        # P1 Enhancement: Comorbidity Interactions and Trajectory Danger Zones
        self.comorbidity_analyzer = ComorbidityInteractionAnalyzer()
        self.trajectory_analyzer = TrajectoryDangerZoneAnalyzer()

        # P2 Enhancement: Negative Constraints and Disease Stage Context
        self.constraint_extractor = NegativeConstraintExtractor()
        self.stage_analyzer = DiseaseStageAnalyzer()

    async def initialize(self) -> None:
        """Initialize the service and its components."""
        if self._is_initialized:
            return

        if self.embedding_service:
            await self.subspecialty_classifier.initialize()

        self._is_initialized = True
        logger.info("GapDetectionService initialized")

    async def detect_gaps(
        self,
        topic: str,
        template_type: TemplateType,
        search_results: List[SearchResult],
        images: Optional[List[Dict]] = None,
        patient_context: Optional[PatientContext] = None,
    ) -> GapReport:
        """
        Run all 16 detection stages and return comprehensive gap report.

        Args:
            topic: The synthesis topic
            template_type: Type of template (PROCEDURAL, DISORDER, etc.)
            search_results: Search results from corpus
            images: Available images for the topic
            patient_context: Optional patient demographics for context-aware detection

        Returns:
            GapReport with all detected gaps and metadata
        """
        start_time = time.time()

        if not self._is_initialized:
            await self.initialize()

        gaps: List[Gap] = []

        # Combine all content for analysis
        combined_content = "\n\n".join(r.content for r in search_results)

        # Extract all entities from search results
        all_entities: Set[str] = set()
        for result in search_results:
            all_entities.update(result.entity_names)

        # Stage 1: Subspecialty Classification
        classification = await self.subspecialty_classifier.classify(topic)
        subspecialty = classification.subspecialty.value
        logger.info(f"Stage 1: Classified '{topic}' as subspecialty: {subspecialty}")

        # Stage 2: Structural Analysis
        stage2_gaps = await self._structural_analysis(
            combined_content, template_type, topic
        )
        for g in stage2_gaps:
            g.detection_stage = 2
        gaps.extend(stage2_gaps)

        # Stage 3: Danger Zone Coverage
        stage3_gaps = self._danger_zone_analysis(subspecialty, combined_content)
        for g in stage3_gaps:
            g.detection_stage = 3
        gaps.extend(stage3_gaps)

        # Stage 4: Procedural Steps Analysis (for PROCEDURAL templates)
        if template_type == TemplateType.PROCEDURAL:
            stage4_gaps = self._procedural_step_analysis(topic, combined_content)
            for g in stage4_gaps:
                g.detection_stage = 4
            gaps.extend(stage4_gaps)

        # Stage 5: Bailout/Complication Coverage
        if template_type == TemplateType.PROCEDURAL:
            stage5_gaps = self._bailout_analysis(topic, combined_content)
            for g in stage5_gaps:
                g.detection_stage = 5
            gaps.extend(stage5_gaps)

        # Stage 6: Instrument Coverage
        if template_type == TemplateType.PROCEDURAL:
            stage6_gaps = self._instrument_analysis(topic, combined_content)
            for g in stage6_gaps:
                g.detection_stage = 6
            gaps.extend(stage6_gaps)

        # Stage 7: Measurement Analysis (BTF guidelines)
        stage7_gaps = self._measurement_analysis(subspecialty, combined_content)
        for g in stage7_gaps:
            g.detection_stage = 7
        gaps.extend(stage7_gaps)

        # Stage 8: Decision Point Analysis
        stage8_gaps = self._decision_point_analysis(combined_content, template_type)
        for g in stage8_gaps:
            g.detection_stage = 8
        gaps.extend(stage8_gaps)

        # Stage 9: Evidence Level Analysis
        stage9_gaps = self._evidence_analysis(subspecialty, topic, combined_content)
        for g in stage9_gaps:
            g.detection_stage = 9
        gaps.extend(stage9_gaps)

        # Stage 10: Source-Based Analysis
        stage10_gaps = self._source_analysis(search_results, all_entities)
        for g in stage10_gaps:
            g.detection_stage = 10
        gaps.extend(stage10_gaps)

        # Stage 11: Visual/Imaging Analysis
        stage11_gaps = self._visual_analysis(combined_content, images or [])
        for g in stage11_gaps:
            g.detection_stage = 11
        gaps.extend(stage11_gaps)

        # Stage 12: DISORDER-Specific Analysis
        if template_type == TemplateType.DISORDER:
            stage12_gaps = self._disorder_specific_analysis(combined_content)
            for g in stage12_gaps:
                g.detection_stage = 12
            gaps.extend(stage12_gaps)

        # Stage 13: ANATOMY-Specific Analysis
        if template_type == TemplateType.ANATOMY:
            stage13_gaps = self._anatomy_specific_analysis(combined_content)
            for g in stage13_gaps:
                g.detection_stage = 13
            gaps.extend(stage13_gaps)

        # Stage 14: CONCEPT-Specific Analysis
        if template_type in (TemplateType.CONCEPT, TemplateType.ENCYCLOPEDIA):
            stage14_gaps = self._concept_specific_analysis(combined_content)
            for g in stage14_gaps:
                g.detection_stage = 14
            gaps.extend(stage14_gaps)

        # Stage 15: Patient Context Analysis (P0 Enhancement)
        stage15_gaps = self._patient_context_analysis(
            combined_content, patient_context, subspecialty
        )
        for g in stage15_gaps:
            g.detection_stage = 15
        gaps.extend(stage15_gaps)

        # Stage 16: Temporal Validity Analysis (P0 Enhancement)
        stage16_gaps = self._temporal_validity_analysis(combined_content, subspecialty)
        for g in stage16_gaps:
            g.detection_stage = 16
        gaps.extend(stage16_gaps)

        # Stage 17: Comorbidity Interaction Analysis (P1 Enhancement)
        stage17_gaps = self._comorbidity_interaction_analysis(
            combined_content, patient_context
        )
        for g in stage17_gaps:
            g.detection_stage = 17
        gaps.extend(stage17_gaps)

        # Stage 18: Trajectory-Aware Danger Zones (P1 Enhancement)
        if template_type == TemplateType.PROCEDURAL:
            stage18_gaps = self._trajectory_danger_zone_analysis(combined_content)
            for g in stage18_gaps:
                g.detection_stage = 18
            gaps.extend(stage18_gaps)

        # Stage 19: Negative Constraint Recognition (P2 Enhancement)
        stage19_gaps = self._negative_constraint_analysis(combined_content)
        for g in stage19_gaps:
            g.detection_stage = 19
        gaps.extend(stage19_gaps)

        # Stage 20: Disease Stage Context (P2 Enhancement)
        stage20_gaps = self._disease_stage_analysis(combined_content)
        for g in stage20_gaps:
            g.detection_stage = 20
        gaps.extend(stage20_gaps)

        # User Demand Analysis (from Q&A history)
        if self.qa_repo:
            user_demand_gaps = await self._user_demand_analysis(
                topic, combined_content
            )
            gaps.extend(user_demand_gaps)

        # Calculate priority scores and deduplicate
        scored_gaps = self._prioritize_and_deduplicate(gaps)

        # Create report
        duration_ms = int((time.time() - start_time) * 1000)

        return GapReport(
            topic=topic,
            template_type=template_type,
            subspecialty=subspecialty,
            gaps=scored_gaps,
            analysis_duration_ms=duration_ms,
            source_document_ids=[r.document_id for r in search_results],
        )

    async def _structural_analysis(
        self,
        content: str,
        template_type: TemplateType,
        topic: str,
    ) -> List[Gap]:
        """Stage 2: Check for missing canonical sections."""
        gaps = []

        # Define expected sections by template type
        expected_sections = {
            TemplateType.PROCEDURAL: [
                ("indication", ["indication", "patient selection", "candidat"]),
                ("anatomy", ["anatomy", "anatomical", "neural", "vascular"]),
                ("equipment", ["equipment", "instrument", "microscope", "monitoring"]),
                ("approach", ["approach", "incision", "craniotomy", "exposure"]),
                ("technique", ["technique", "dissection", "microsurgical", "resection"]),
                ("closure", ["closure", "hemostasis", "dural", "bone flap"]),
                ("complications", ["complication", "risk", "hazard", "avoid"]),
                ("outcomes", ["outcome", "prognosis", "follow-up", "result"]),
            ],
            TemplateType.DISORDER: [
                ("epidemiology", ["incidence", "prevalence", "demographic"]),
                ("pathophysiology", ["pathophysiology", "mechanism", "etiology"]),
                ("presentation", ["presentation", "symptom", "sign", "clinical"]),
                ("diagnosis", ["diagnosis", "imaging", "workup", "differential"]),
                ("treatment", ["treatment", "management", "therapy", "surgical"]),
                ("prognosis", ["prognosis", "outcome", "survival", "recurrence"]),
            ],
            TemplateType.ANATOMY: [
                ("boundaries", ["boundary", "border", "limit", "extent"]),
                ("contents", ["content", "structure", "contain"]),
                ("relations", ["relation", "adjacent", "proximity"]),
                ("blood_supply", ["artery", "vein", "blood supply", "drainage"]),
                ("innervation", ["nerve", "innervation", "supply"]),
                ("clinical", ["clinical", "surgical", "significance"]),
            ],
        }

        sections = expected_sections.get(template_type, [])
        content_lower = content.lower()

        for section_name, keywords in sections:
            # Check if any keyword is present
            found = any(kw in content_lower for kw in keywords)

            if not found:
                gaps.append(Gap(
                    gap_type=GapType.STRUCTURAL,
                    topic=section_name.upper(),
                    priority_score=55.0,  # Medium-high priority
                    current_coverage="Section not found in content",
                    recommended_coverage=f"Include {section_name} section",
                    justification={
                        "reason": f"Missing canonical section: {section_name}",
                        "keywords_searched": keywords,
                    },
                    target_section=section_name.upper(),
                    external_query=f"{topic} {section_name}",
                ))

        return gaps

    def _danger_zone_analysis(
        self, subspecialty: str, content: str
    ) -> List[Gap]:
        """Stage 3: Check if danger zones are adequately covered."""
        gaps = []
        content_lower = content.lower()

        danger_zones = DANGER_ZONES.get(subspecialty, [])

        for zone_info in danger_zones:
            zone = zone_info["structure"]
            consequence = zone_info["consequence"]

            # Check if danger zone is mentioned
            zone_lower = zone.lower()
            # Handle variations (e.g., "petrous ICA" might appear as "petrous internal carotid")
            zone_terms = zone_lower.split()

            found = zone_lower in content_lower
            if not found and len(zone_terms) > 1:
                # Check if key terms are present
                found = all(term in content_lower for term in zone_terms if len(term) > 3)

            if not found:
                gaps.append(Gap(
                    gap_type=GapType.DANGER_ZONE,
                    topic=zone,
                    priority_score=100.0,  # Auto-critical
                    current_coverage="",
                    recommended_coverage=f"Discuss danger zone: {zone}. Risk: {consequence}",
                    justification={
                        "reason": f"Safety-critical anatomy '{zone}' not mentioned",
                        "consequence": consequence,
                        "subspecialty": subspecialty,
                    },
                    target_section="COMPLICATIONS",
                    external_query=f"{zone} surgical danger {subspecialty}",
                    safety_critical=True,
                ))

        return gaps

    def _procedural_step_analysis(
        self, procedure: str, content: str
    ) -> List[Gap]:
        """Stage 4: Check for missing key operative steps."""
        gaps = []
        content_lower = content.lower()

        # Find matching procedure template
        procedure_key = self._match_procedure(procedure)
        if not procedure_key or procedure_key not in PROCEDURAL_TEMPLATES:
            return gaps

        template = PROCEDURAL_TEMPLATES[procedure_key]
        key_steps = template.get("key_steps", [])

        for step in key_steps:
            step_lower = step.lower()
            # Check for step or key terms
            step_terms = [t for t in step_lower.split() if len(t) > 4]

            found = step_lower in content_lower
            if not found and step_terms:
                # Check if main terms present
                found = any(term in content_lower for term in step_terms)

            if not found:
                gaps.append(Gap(
                    gap_type=GapType.PROCEDURAL_STEP,
                    topic=step,
                    priority_score=65.0,
                    current_coverage="",
                    recommended_coverage=f"Include operative step: {step}",
                    justification={
                        "reason": f"Key procedural step not found: {step}",
                        "procedure": procedure_key,
                    },
                    target_section="TECHNIQUE",
                    external_query=f"{procedure} {step} technique",
                ))

        return gaps

    def _bailout_analysis(self, procedure: str, content: str) -> List[Gap]:
        """Stage 5: Check for complication management coverage."""
        gaps = []
        content_lower = content.lower()

        procedure_key = self._match_procedure(procedure)
        if not procedure_key or procedure_key not in PROCEDURAL_TEMPLATES:
            return gaps

        template = PROCEDURAL_TEMPLATES[procedure_key]
        bailouts = template.get("bailout", [])

        for bailout in bailouts:
            bailout_lower = bailout.lower()
            bailout_terms = [t for t in bailout_lower.split() if len(t) > 3]

            found = bailout_lower in content_lower
            if not found and bailout_terms:
                found = all(term in content_lower for term in bailout_terms[:2])

            if not found:
                gaps.append(Gap(
                    gap_type=GapType.BAILOUT,
                    topic=bailout,
                    priority_score=100.0,  # Auto-critical
                    current_coverage="",
                    recommended_coverage=f"Include bailout procedure: {bailout}",
                    justification={
                        "reason": f"Missing bailout/complication management: {bailout}",
                        "procedure": procedure_key,
                    },
                    target_section="COMPLICATIONS",
                    external_query=f"{procedure} {bailout} management",
                    safety_critical=True,
                ))

        return gaps

    def _instrument_analysis(self, procedure: str, content: str) -> List[Gap]:
        """Stage 6: Check for required instruments."""
        gaps = []
        content_lower = content.lower()

        procedure_key = self._match_procedure(procedure)
        if not procedure_key or procedure_key not in PROCEDURAL_TEMPLATES:
            return gaps

        template = PROCEDURAL_TEMPLATES[procedure_key]
        instruments = template.get("instruments", [])

        for instrument in instruments:
            if instrument.lower() not in content_lower:
                gaps.append(Gap(
                    gap_type=GapType.INSTRUMENT,
                    topic=instrument,
                    priority_score=40.0,
                    current_coverage="",
                    recommended_coverage=f"Mention required instrument: {instrument}",
                    justification={
                        "reason": f"Required instrument not mentioned: {instrument}",
                        "procedure": procedure_key,
                    },
                    target_section="EQUIPMENT",
                    external_query=f"{procedure} {instrument}",
                ))

        return gaps

    def _measurement_analysis(
        self, subspecialty: str, content: str
    ) -> List[Gap]:
        """Stage 7: Check for required quantitative thresholds (BTF guidelines)."""
        gaps = []
        content_lower = content.lower()

        # ICP management parameters for trauma
        if subspecialty == "trauma" or "icp" in content_lower:
            # Check ICP treatment threshold
            icp_threshold = ICP_PARAMETERS["treatment_threshold"]["value"]
            if "icp" in content_lower or "intracranial pressure" in content_lower:
                if "22" not in content and "22mmhg" not in content_lower.replace(" ", ""):
                    gaps.append(Gap(
                        gap_type=GapType.MEASUREMENT,
                        topic="ICP treatment threshold",
                        priority_score=100.0,  # Auto-critical
                        current_coverage="ICP discussed without threshold",
                        recommended_coverage=f"ICP treatment threshold: {icp_threshold} (BTF 4th Edition 2016)",
                        justification={
                            "reason": "Missing ICP treatment threshold per BTF guidelines",
                            "source": "BTF 4th Edition 2016",
                        },
                        target_section="TECHNIQUE",
                        external_query="ICP treatment threshold BTF guidelines",
                        safety_critical=True,
                    ))

            # Check CPP target
            if "cpp" in content_lower or "cerebral perfusion" in content_lower:
                cpp_pattern = r"60.*70|70.*60"
                if not re.search(cpp_pattern, content):
                    gaps.append(Gap(
                        gap_type=GapType.MEASUREMENT,
                        topic="CPP target range",
                        priority_score=100.0,  # Auto-critical
                        current_coverage="CPP discussed without target",
                        recommended_coverage="CPP target: 60-70 mmHg (BTF 4th Edition)",
                        justification={
                            "reason": "Missing CPP target range per BTF guidelines",
                            "source": "BTF 4th Edition",
                        },
                        target_section="TECHNIQUE",
                        external_query="CPP target BTF guidelines",
                        safety_critical=True,
                    ))

        return gaps

    def _decision_point_analysis(
        self, content: str, template_type: TemplateType
    ) -> List[Gap]:
        """Stage 8: Check for clinical decision logic."""
        gaps = []
        content_lower = content.lower()

        # Decision markers to look for
        decision_markers = [
            "indication",
            "contraindication",
            "criteria",
            "if ",
            "when ",
            "should",
            "recommend",
        ]

        # Check if content has decision logic
        has_decisions = any(marker in content_lower for marker in decision_markers)

        if not has_decisions and template_type in (TemplateType.PROCEDURAL, TemplateType.DISORDER):
            gaps.append(Gap(
                gap_type=GapType.DECISION_POINT,
                topic="Clinical decision criteria",
                priority_score=50.0,
                current_coverage="Limited decision guidance",
                recommended_coverage="Include indications, contraindications, and decision criteria",
                justification={
                    "reason": "Missing clinical decision logic",
                    "template_type": template_type.value,
                },
                target_section="INDICATION",
            ))

        return gaps

    def _evidence_analysis(
        self, subspecialty: str, topic: str, content: str
    ) -> List[Gap]:
        """Stage 9: Check for landmark trial coverage."""
        gaps = []
        content_lower = content.lower()
        topic_lower = topic.lower()

        # Map subspecialties to relevant trials
        relevant_trials_map = {
            "vascular": ["ISAT", "BRAT", "ARUBA"],
            "trauma": ["CRASH", "DECRA", "RESCUEicp", "STICH", "STICH_II"],
            "spine": [],  # Add spine trials
            "tumor": [],  # Add tumor trials
        }

        relevant_trials = relevant_trials_map.get(subspecialty, [])

        # Additional topic-based trial matching
        if "aneurysm" in topic_lower:
            relevant_trials.extend(["ISAT", "BRAT"])
        if "avm" in topic_lower:
            relevant_trials.append("ARUBA")
        if "ich" in topic_lower or "hemorrhage" in topic_lower:
            relevant_trials.extend(["STICH", "STICH_II", "INTERACT2", "ATACH2"])
        if "tbi" in topic_lower or "trauma" in topic_lower:
            relevant_trials.extend(["CRASH", "DECRA", "RESCUEicp"])

        # Deduplicate
        relevant_trials = list(set(relevant_trials))

        for trial_name in relevant_trials:
            if trial_name.lower() not in content_lower:
                trial_info = LANDMARK_TRIALS.get(trial_name, {})
                if trial_info:
                    gaps.append(Gap(
                        gap_type=GapType.EVIDENCE_LEVEL,
                        topic=f"{trial_name} trial",
                        priority_score=45.0,
                        current_coverage="",
                        recommended_coverage=(
                            f"{trial_name}: {trial_info.get('finding', '')} "
                            f"({trial_info.get('year', '')})"
                        ),
                        justification={
                            "reason": f"Landmark trial not cited: {trial_name}",
                            "trial_year": trial_info.get("year"),
                        },
                        target_section="OUTCOMES",
                        external_query=f"{trial_name} trial neurosurgery",
                    ))

        return gaps

    def _source_analysis(
        self,
        search_results: List[SearchResult],
        all_entities: Set[str],
    ) -> List[Gap]:
        """Stage 10: Analyze coverage across multiple sources."""
        gaps = []

        # Track which entities appear in how many sources
        entity_sources: Dict[str, Set[str]] = defaultdict(set)

        for result in search_results:
            for entity in result.entity_names:
                entity_sources[entity].add(result.document_id)

        # Entities in only 1 source might be under-covered
        for entity, sources in entity_sources.items():
            if len(sources) == 1 and len(entity) > 5:  # Skip short entities
                gaps.append(Gap(
                    gap_type=GapType.THIN_COVERAGE,
                    topic=entity,
                    priority_score=30.0,
                    current_coverage=f"Found in 1 source only",
                    recommended_coverage=f"Verify coverage of: {entity}",
                    justification={
                        "reason": "Concept appears in only one source",
                        "source_count": len(sources),
                    },
                ))

        return gaps[:10]  # Limit to top 10

    def _visual_analysis(
        self, content: str, images: List[Dict]
    ) -> List[Gap]:
        """Stage 11: Check for required imaging modalities and figures."""
        gaps = []
        content_lower = content.lower()

        # Check imaging modalities mentioned
        imaging_terms = {
            "MRI": ["mri", "magnetic resonance"],
            "CT": ["ct ", "ct,", "computed tomography"],
            "angiography": ["angiography", "angiogram", "dsa"],
            "fluoroscopy": ["fluoroscopy", "c-arm", "x-ray"],
        }

        mentioned_modalities = []
        for modality, terms in imaging_terms.items():
            if any(term in content_lower for term in terms):
                mentioned_modalities.append(modality)

        # Check if images are available for mentioned modalities
        if mentioned_modalities and not images:
            gaps.append(Gap(
                gap_type=GapType.VISUAL,
                topic="Supporting images",
                priority_score=35.0,
                current_coverage=f"Modalities mentioned: {', '.join(mentioned_modalities)}",
                recommended_coverage="Include supporting images/figures",
                justification={
                    "reason": "Imaging discussed but no figures available",
                    "modalities_mentioned": mentioned_modalities,
                },
                target_section="DIAGNOSIS",
            ))

        return gaps

    def _disorder_specific_analysis(self, content: str) -> List[Gap]:
        """Stage 12: DISORDER-specific required elements."""
        gaps = []
        content_lower = content.lower()

        for section, keywords in DISORDER_REQUIRED_ELEMENTS.items():
            found = any(kw in content_lower for kw in keywords)
            if not found:
                gaps.append(Gap(
                    gap_type=GapType.STRUCTURAL,
                    topic=f"DISORDER: {section}",
                    priority_score=50.0,
                    current_coverage="",
                    recommended_coverage=f"Include {section} content for disorder template",
                    justification={
                        "reason": f"Missing required DISORDER element: {section}",
                        "keywords_expected": keywords,
                    },
                    target_section=section.upper(),
                ))

        return gaps

    def _anatomy_specific_analysis(self, content: str) -> List[Gap]:
        """Stage 13: ANATOMY-specific required elements."""
        gaps = []
        content_lower = content.lower()

        for section, keywords in ANATOMY_REQUIRED_ELEMENTS.items():
            found = any(kw in content_lower for kw in keywords)
            if not found:
                gaps.append(Gap(
                    gap_type=GapType.STRUCTURAL,
                    topic=f"ANATOMY: {section}",
                    priority_score=50.0,
                    current_coverage="",
                    recommended_coverage=f"Include {section} content for anatomy template",
                    justification={
                        "reason": f"Missing required ANATOMY element: {section}",
                        "keywords_expected": keywords,
                    },
                    target_section=section.upper(),
                ))

        return gaps

    def _concept_specific_analysis(self, content: str) -> List[Gap]:
        """Stage 14: CONCEPT-specific required elements."""
        gaps = []
        content_lower = content.lower()

        for section, keywords in CONCEPT_REQUIRED_ELEMENTS.items():
            found = any(kw in content_lower for kw in keywords)
            if not found:
                gaps.append(Gap(
                    gap_type=GapType.STRUCTURAL,
                    topic=f"CONCEPT: {section}",
                    priority_score=45.0,
                    current_coverage="",
                    recommended_coverage=f"Include {section} content",
                    justification={
                        "reason": f"Missing required CONCEPT element: {section}",
                        "keywords_expected": keywords,
                    },
                    target_section=section.upper(),
                ))

        return gaps

    def _patient_context_analysis(
        self,
        content: str,
        patient_context: Optional[PatientContext],
        subspecialty: str,
    ) -> List[Gap]:
        """
        Stage 15: Patient Context Analysis (P0 Enhancement)

        Detects gaps related to patient demographics:
        - Pediatric thresholds not mentioned when applicable
        - Drug contraindications for patient conditions
        - Weight-based dosing requirements for pediatrics
        - Pregnancy/renal considerations
        """
        gaps = []

        if patient_context is None:
            return gaps

        content_lower = content.lower()

        # Get measurement profile for patient's age group
        measurement_profile = self.patient_context_analyzer.get_measurement_profile(
            patient_context
        )

        # Check for pediatric-specific threshold gaps
        if patient_context.is_pediatric:
            age_group = patient_context.age_group

            # Check if pediatric ICP threshold is mentioned
            if "icp" in content_lower or "intracranial pressure" in content_lower:
                pediatric_threshold = measurement_profile.get("icp", {}).get(
                    "treatment_threshold", {}
                ).get("value")

                if pediatric_threshold:
                    # Check if the correct pediatric threshold is mentioned
                    threshold_str = str(pediatric_threshold)
                    if threshold_str not in content:
                        gaps.append(Gap(
                            gap_type=GapType.MEASUREMENT,
                            topic=f"Pediatric ICP threshold ({age_group.value})",
                            priority_score=100.0,  # Auto-critical
                            current_coverage="Adult ICP guidelines may be referenced",
                            recommended_coverage=(
                                f"For {age_group.value} patients, ICP treatment threshold is "
                                f"{pediatric_threshold} mmHg (vs adult 22 mmHg). "
                                f"Source: {measurement_profile.get('source', 'Pediatric TBI Guidelines')}"
                            ),
                            justification={
                                "reason": "Pediatric patient requires age-appropriate ICP thresholds",
                                "patient_age_group": age_group.value,
                                "correct_threshold": pediatric_threshold,
                                "adult_threshold": 22,
                            },
                            target_section="MANAGEMENT",
                            safety_critical=True,
                        ))

            # Check for weight-based dosing mentions
            drug_keywords = ["mannitol", "levetiracetam", "phenytoin", "dexamethasone"]
            drugs_mentioned = [d for d in drug_keywords if d in content_lower]

            if drugs_mentioned and patient_context.weight_kg:
                # Check if weight-based dosing is mentioned
                weight_pattern = r"\d+\.?\d*\s*mg/kg|\d+\.?\d*\s*g/kg"
                has_weight_dosing = re.search(weight_pattern, content_lower)

                if not has_weight_dosing:
                    gaps.append(Gap(
                        gap_type=GapType.MEASUREMENT,
                        topic="Weight-based pediatric dosing",
                        priority_score=85.0,
                        current_coverage=f"Drugs mentioned: {', '.join(drugs_mentioned)}",
                        recommended_coverage=(
                            f"Pediatric patient (weight: {patient_context.weight_kg}kg) requires "
                            f"weight-based dosing. Include mg/kg calculations for: {', '.join(drugs_mentioned)}"
                        ),
                        justification={
                            "reason": "Pediatric patients require weight-based drug dosing",
                            "patient_weight_kg": patient_context.weight_kg,
                            "drugs_needing_adjustment": drugs_mentioned,
                        },
                        target_section="PHARMACOLOGY",
                        safety_critical=True,
                    ))

        # Check for pregnancy-related contraindications
        if patient_context.is_pregnant:
            pregnancy_contraindicated = ["valproic acid", "carbamazepine", "phenytoin", "warfarin"]
            for drug in pregnancy_contraindicated:
                if drug in content_lower:
                    # Check if contraindication is mentioned
                    contraindication_mentioned = any(
                        phrase in content_lower
                        for phrase in ["contraindicated in pregnancy", "avoid in pregnancy",
                                       "pregnancy category x", "teratogenic"]
                    )
                    if not contraindication_mentioned:
                        gaps.append(Gap(
                            gap_type=GapType.DANGER_ZONE,
                            topic=f"{drug.title()} pregnancy contraindication",
                            priority_score=100.0,  # Auto-critical
                            current_coverage=f"{drug.title()} mentioned without pregnancy warning",
                            recommended_coverage=(
                                f"WARNING: {drug.title()} is contraindicated in pregnancy. "
                                f"Patient is pregnant. Consider alternatives such as levetiracetam."
                            ),
                            justification={
                                "reason": f"{drug.title()} is teratogenic and contraindicated in pregnancy",
                                "patient_pregnancy_status": patient_context.pregnancy_status.value,
                            },
                            target_section="PHARMACOLOGY",
                            safety_critical=True,
                        ))

        # Check for renal impairment contraindications
        if patient_context.has_renal_impairment:
            if "mannitol" in content_lower:
                # Check if renal contraindication is mentioned
                renal_warning_mentioned = any(
                    phrase in content_lower
                    for phrase in ["renal failure", "renal impairment", "contraindicated",
                                   "avoid in renal", "gfr"]
                )
                if not renal_warning_mentioned:
                    gaps.append(Gap(
                        gap_type=GapType.DANGER_ZONE,
                        topic="Mannitol renal contraindication",
                        priority_score=100.0,  # Auto-critical
                        current_coverage="Mannitol mentioned without renal warning",
                        recommended_coverage=(
                            f"WARNING: Mannitol is contraindicated with GFR < 30. "
                            f"Patient GFR: {patient_context.gfr or 'unknown'}. "
                            f"Consider hypertonic saline as alternative."
                        ),
                        justification={
                            "reason": "Mannitol contraindicated in renal impairment",
                            "patient_gfr": patient_context.gfr,
                            "patient_renal_status": patient_context.renal_function.value,
                        },
                        target_section="PHARMACOLOGY",
                        safety_critical=True,
                    ))

        return gaps

    def _temporal_validity_analysis(
        self,
        content: str,
        subspecialty: str,
    ) -> List[Gap]:
        """
        Stage 16: Temporal Validity Analysis (P0 Enhancement)

        Detects references to superseded medical practices:
        - NASCIS protocol (now contraindicated)
        - Steroids in TBI (CRASH trial)
        - Routine hyperventilation (causes ischemia)
        - Outdated BTF guidelines
        """
        gaps = []

        # Check content for superseded practices
        specialty_filter = [subspecialty] if subspecialty else None
        warnings = self.temporal_validity_checker.check_content(
            content, specialty_filter=specialty_filter
        )

        for warning in warnings:
            # Map supersession severity to gap priority
            if warning.severity == SupersessionSeverity.CRITICAL:
                priority_score = 100.0
                safety_critical = True
                gap_type = GapType.DANGER_ZONE  # Treat as danger zone
            elif warning.severity == SupersessionSeverity.HIGH:
                priority_score = 80.0
                safety_critical = False
                gap_type = GapType.TEMPORAL
            elif warning.severity == SupersessionSeverity.MEDIUM:
                priority_score = 55.0
                safety_critical = False
                gap_type = GapType.TEMPORAL
            else:
                priority_score = 35.0
                safety_critical = False
                gap_type = GapType.TEMPORAL

            gaps.append(Gap(
                gap_type=gap_type,
                topic=f"Superseded: {warning.detected_text[:50]}",
                priority_score=priority_score,
                current_coverage=warning.original_practice,
                recommended_coverage=warning.current_recommendation,
                justification={
                    "reason": f"Reference to superseded practice (superseded {warning.supersession_year})",
                    "supersession_type": warning.supersession_type.value,
                    "evidence": warning.evidence,
                    "detected_text": warning.detected_text,
                    "context": warning.location_hint,
                },
                target_section="EVIDENCE",
                external_query=f"current guidelines {warning.detected_text[:30]}",
                safety_critical=safety_critical,
            ))

        return gaps

    async def _user_demand_analysis(
        self, topic: str, content: str
    ) -> List[Gap]:
        """Stage 5 (original): User demand analysis from Q&A history."""
        gaps = []

        if not self.qa_repo:
            return gaps

        try:
            # Get frequently asked but unanswered questions
            related_questions = await self.qa_repo.get_related_questions(topic, limit=20)

            for qa in related_questions:
                if not qa.get("was_answered", True) or qa.get("answer_quality_score", 1.0) < 0.5:
                    question = qa.get("question", "")
                    if question and question.lower() not in content.lower():
                        gaps.append(Gap(
                            gap_type=GapType.USER_DEMAND,
                            topic=question[:100],
                            priority_score=55.0,
                            current_coverage="",
                            recommended_coverage=f"Address user question: {question[:150]}",
                            justification={
                                "reason": "Frequently asked but poorly answered",
                                "question_count": qa.get("count", 1),
                            },
                            external_query=question,
                        ))

        except Exception as e:
            logger.warning(f"User demand analysis failed: {e}")

        return gaps[:5]  # Limit to top 5

    def _prioritize_and_deduplicate(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize gaps and remove duplicates."""
        # Deduplicate by topic (keep highest priority)
        seen_topics: Dict[str, Gap] = {}

        for gap in gaps:
            topic_key = gap.topic.lower().strip()

            if topic_key not in seen_topics:
                seen_topics[topic_key] = gap
            elif gap.priority_score > seen_topics[topic_key].priority_score:
                seen_topics[topic_key] = gap

        # Sort by priority score (highest first)
        deduped_gaps = list(seen_topics.values())
        deduped_gaps.sort(key=lambda g: g.priority_score, reverse=True)

        return deduped_gaps

    def _match_procedure(self, topic: str) -> Optional[str]:
        """Match topic to a known procedure template."""
        topic_lower = topic.lower()

        # Direct match
        for proc_key in PROCEDURAL_TEMPLATES.keys():
            if proc_key.replace("_", " ") in topic_lower:
                return proc_key

        # Keyword-based matching
        procedure_keywords = {
            "pterional_craniotomy": ["pterional", "frontotemporal"],
            "retrosigmoid_craniotomy": ["retrosigmoid", "cpa", "cerebellopontine"],
            "lumbar_microdiscectomy": ["microdiscectomy", "lumbar disc", "discectomy"],
            "ACDF": ["acdf", "anterior cervical", "cervical fusion"],
            "EVD_placement": ["evd", "ventriculostomy", "external ventricular"],
            "decompressive_craniectomy": ["decompressive craniectomy", "dc ", "hemicraniectomy"],
        }

        for proc_key, keywords in procedure_keywords.items():
            if any(kw in topic_lower for kw in keywords):
                return proc_key

        return None

    def _comorbidity_interaction_analysis(
        self,
        content: str,
        patient_context: Optional[PatientContext],
    ) -> List[Gap]:
        """
        Stage 17: Comorbidity Interaction Analysis (P1 Enhancement)

        Detects dangerous drug-drug, drug-condition, and cascade risk interactions.
        """
        gaps = []

        # Extract drugs from content
        drugs_in_content = self.comorbidity_analyzer.extract_drugs_from_content(content)

        if not drugs_in_content:
            return gaps

        # Get patient conditions if available
        patient_conditions = set()
        additional_factors = set()

        if patient_context:
            patient_conditions = patient_context.known_conditions.copy()

            # Add derived conditions
            if patient_context.has_renal_impairment:
                patient_conditions.add("renal_failure")
                if patient_context.gfr and patient_context.gfr < 30:
                    patient_conditions.add("gfr_below_30")
            if patient_context.is_pregnant:
                patient_conditions.add("pregnancy")
            if patient_context.age_group == AgeGroup.ELDERLY:
                additional_factors.add("elderly")

        # Analyze interactions
        warnings = self.comorbidity_analyzer.analyze(
            patient_conditions=patient_conditions,
            drugs_in_content=drugs_in_content,
            additional_factors=additional_factors,
        )

        # Convert warnings to gaps
        for warning in warnings:
            if warning.severity == InteractionSeverity.CRITICAL:
                priority = 100.0
                safety_critical = True
            elif warning.severity == InteractionSeverity.HIGH:
                priority = 80.0
                safety_critical = True
            elif warning.severity == InteractionSeverity.MODERATE:
                priority = 60.0
                safety_critical = False
            else:
                priority = 40.0
                safety_critical = False

            gaps.append(Gap(
                gap_type=GapType.DANGER_ZONE,
                topic=f"Interaction: {warning.description[:50]}",
                priority_score=priority,
                current_coverage=f"Drugs/conditions in content: {', '.join(warning.involved_elements)}",
                recommended_coverage=warning.recommendation,
                justification={
                    "reason": warning.description,
                    "mechanism": warning.mechanism,
                    "interaction_type": warning.interaction_type.value,
                    "involved_elements": warning.involved_elements,
                },
                target_section="PHARMACOLOGY",
                safety_critical=safety_critical,
            ))

        return gaps

    def _trajectory_danger_zone_analysis(
        self,
        content: str,
    ) -> List[Gap]:
        """
        Stage 18: Trajectory-Aware Danger Zones (P1 Enhancement)

        Detects missing approach-specific danger zones based on surgical trajectory.
        """
        gaps = []

        # Detect surgical approach from content
        approaches = self.trajectory_analyzer.detect_approach(content)

        if SurgicalApproach.UNKNOWN in approaches and len(approaches) == 1:
            return gaps

        # Analyze danger zone coverage
        missing_zones = self.trajectory_analyzer.analyze_danger_zone_coverage(
            content, approaches
        )

        for zone_info in missing_zones:
            priority = 100.0 if zone_info["frequency"] == "common" else 85.0

            gaps.append(Gap(
                gap_type=GapType.DANGER_ZONE,
                topic=f"Trajectory: {zone_info['structure'][:40]}",
                priority_score=priority,
                current_coverage=f"Approach: {zone_info['approach']}",
                recommended_coverage=(
                    f"Include danger zone: {zone_info['structure']}. "
                    f"Risk: {zone_info['consequence']}. "
                    f"Prevention: {zone_info['prevention_note']}"
                ),
                justification={
                    "reason": f"Approach-specific danger zone not covered",
                    "approach": zone_info["approach"],
                    "consequence": zone_info["consequence"],
                    "frequency": zone_info["frequency"],
                    "bailout": zone_info.get("bailout_strategy", ""),
                },
                target_section="COMPLICATIONS",
                safety_critical=True,
            ))

        return gaps

    def _negative_constraint_analysis(
        self,
        content: str,
    ) -> List[Gap]:
        """
        Stage 19: Negative Constraint Recognition (P2 Enhancement)

        Extracts and surfaces negative constraints (contraindications, prohibitions).
        """
        gaps = []

        # Extract constraints from content
        constraints = self.constraint_extractor.extract_constraints(content)

        # Check against known critical constraints
        known_applicable = self.constraint_extractor.check_known_constraints(content)

        # Surface absolute and strong constraints as informational gaps
        # to ensure they are emphasized in output
        for constraint in constraints:
            if constraint.severity in (ConstraintSeverity.ABSOLUTE, ConstraintSeverity.STRONG):
                gaps.append(Gap(
                    gap_type=GapType.DECISION_POINT,
                    topic=f"Constraint: {constraint.action[:40]}",
                    priority_score=70.0 if constraint.severity == ConstraintSeverity.ABSOLUTE else 55.0,
                    current_coverage=constraint.source_text[:200] if constraint.source_text else "",
                    recommended_coverage=(
                        f"CONSTRAINT DETECTED: {constraint.action}. "
                        f"Reason: {constraint.reason}. Ensure this is prominently featured."
                    ),
                    justification={
                        "reason": "Negative constraint should be emphasized",
                        "constraint_type": constraint.constraint_type.value,
                        "severity": constraint.severity.value,
                        "conditions": constraint.conditions,
                    },
                    target_section="CONTRAINDICATIONS",
                    safety_critical=constraint.severity == ConstraintSeverity.ABSOLUTE,
                ))

        # Check for known critical constraints that should be mentioned
        content_lower = content.lower()
        for known in known_applicable:
            # Check if the constraint is already mentioned appropriately
            constraint_mentioned = any(
                term in content_lower
                for term in ["contraindicated", "avoid", "do not", "never"]
                if term in content_lower
            )

            if not constraint_mentioned:
                gaps.append(Gap(
                    gap_type=GapType.DANGER_ZONE,
                    topic=f"Missing Constraint: {known.action[:40]}",
                    priority_score=95.0,
                    current_coverage="Known critical constraint not mentioned",
                    recommended_coverage=(
                        f"CRITICAL: {known.action} - {known.reason}. "
                        f"Source: {known.source_text}"
                    ),
                    justification={
                        "reason": "Known critical constraint must be included",
                        "constraint_id": known.constraint_id,
                        "conditions": known.conditions,
                    },
                    target_section="CONTRAINDICATIONS",
                    safety_critical=True,
                ))

        return gaps

    def _disease_stage_analysis(
        self,
        content: str,
    ) -> List[Gap]:
        """
        Stage 20: Disease Stage Context (P2 Enhancement)

        Validates that recommendations are appropriate for the detected disease stage.
        """
        gaps = []

        # Detect diseases in content
        diseases = self.stage_analyzer.detect_disease(content)
        if not diseases:
            return gaps

        # Detect stage
        stages = self.stage_analyzer.detect_stage(content)
        if stages[0][0] == DiseaseStage.UNKNOWN:
            return gaps

        detected_stage = stages[0][0]

        # Check for stage-inappropriate recommendations
        for disease in diseases:
            warnings = self.stage_analyzer.check_stage_appropriateness(content, disease)

            for warning in warnings:
                severity_map = {"CRITICAL": 95.0, "HIGH": 75.0, "MEDIUM": 55.0, "LOW": 35.0}
                priority = severity_map.get(warning.severity, 55.0)

                gaps.append(Gap(
                    gap_type=GapType.TEMPORAL,
                    topic=f"Stage Mismatch: {warning.intervention[:40]}",
                    priority_score=priority,
                    current_coverage=f"Detected stage: {warning.detected_stage.value}",
                    recommended_coverage=(
                        f"'{warning.intervention}' may be inappropriate for {warning.detected_stage.value} phase. "
                        f"{warning.recommendation}"
                    ),
                    justification={
                        "reason": warning.issue,
                        "detected_stage": warning.detected_stage.value,
                        "disease": disease,
                    },
                    target_section="MANAGEMENT",
                    safety_critical=warning.severity == "CRITICAL",
                ))

        # Also generate informational gaps about stage-specific management
        for disease in diseases:
            profile = self.stage_analyzer.get_stage_profile(disease, detected_stage)
            if profile:
                # Check if key interventions are mentioned
                content_lower = content.lower()
                for intervention in profile.key_interventions[:3]:  # Top 3
                    intervention_terms = intervention.lower().split()
                    mentioned = any(
                        term in content_lower
                        for term in intervention_terms
                        if len(term) > 3
                    )

                    if not mentioned:
                        gaps.append(Gap(
                            gap_type=GapType.STRUCTURAL,
                            topic=f"Stage-specific: {intervention[:40]}",
                            priority_score=50.0,
                            current_coverage=f"Stage: {detected_stage.value} {disease}",
                            recommended_coverage=(
                                f"Consider including stage-specific intervention: {intervention} "
                                f"(appropriate for {detected_stage.value} {disease})"
                            ),
                            justification={
                                "reason": "Stage-specific key intervention not mentioned",
                                "stage": detected_stage.value,
                                "paradigm": profile.typical_paradigm.value,
                            },
                            target_section="MANAGEMENT",
                        ))

        return gaps
