"""
Trajectory-Aware Danger Zone Detection
=======================================

P1 Enhancement: Approach-Specific Danger Zone Filtering

This module provides surgical approach-specific danger zone detection,
recognizing that different trajectories expose different critical structures.

Clinical Safety Rationale:
- Transsphenoidal approach: ICA at risk, optic nerves, cavernous sinus
- Pterional approach: MCA, sylvian veins, frontal branch facial nerve
- Retrosigmoid approach: AICA, facial nerve, lower cranial nerves
- Far lateral approach: Vertebral artery, CN IX-XII

Detection Approach:
1. Identify surgical approach from content
2. Map approach to relevant danger zones
3. Filter danger zone analysis by trajectory
4. Add approach-specific safety warnings
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SurgicalApproach(Enum):
    """Standard neurosurgical approaches."""

    # Skull Base - Anterior
    TRANSSPHENOIDAL = "transsphenoidal"
    ENDONASAL_EXPANDED = "expanded_endonasal"
    TRANSCRIBRIFORM = "transcribriform"
    TRANSORBITAL = "transorbital"

    # Skull Base - Lateral
    PTERIONAL = "pterional"
    FRONTOTEMPORAL = "frontotemporal"
    ORBITOZYGOMATIC = "orbitozygomatic"
    SUBTEMPORAL = "subtemporal"
    MIDDLE_FOSSA = "middle_fossa"
    TRANSLABYRINTHINE = "translabyrinthine"
    TRANSCOCHLEAR = "transcochlear"

    # Skull Base - Posterior
    RETROSIGMOID = "retrosigmoid"
    FAR_LATERAL = "far_lateral"
    EXTREME_LATERAL = "extreme_lateral"
    MIDLINE_SUBOCCIPITAL = "suboccipital"
    TELOVELAR = "telovelar"

    # Supratentorial - Convexity
    FRONTAL_CRANIOTOMY = "frontal"
    PARIETAL_CRANIOTOMY = "parietal"
    TEMPORAL_CRANIOTOMY = "temporal"
    OCCIPITAL_CRANIOTOMY = "occipital"

    # Deep/Functional
    INTERHEMISPHERIC = "interhemispheric"
    TRANSCALLOSAL = "transcallosal"
    TRANSCORTICAL = "transcortical"
    SUPRACEREBELLAR_INFRATENTORIAL = "supracerebellar"

    # Spine
    POSTERIOR_CERVICAL = "posterior_cervical"
    ANTERIOR_CERVICAL = "anterior_cervical"
    POSTERIOR_THORACIC = "posterior_thoracic"
    POSTERIOR_LUMBAR = "posterior_lumbar"
    LATERAL_EXTRACAVITARY = "lateral_extracavitary"
    ANTEROLATERAL_THORACIC = "anterolateral_thoracic"

    # Minimally Invasive
    ENDOSCOPIC_VENTRICULAR = "endoscopic_ventricular"
    STEREOTACTIC = "stereotactic"
    TUBULAR_RETRACTOR = "tubular"

    # Generic/Unknown
    UNKNOWN = "unknown"


@dataclass
class DangerZone:
    """Represents a structure at risk during surgery."""

    structure: str
    consequence: str
    frequency: str = "common"  # common, uncommon, rare
    prevention_note: str = ""
    bailout_strategy: str = ""
    anatomical_notes: str = ""


@dataclass
class ApproachProfile:
    """Complete profile for a surgical approach."""

    approach: SurgicalApproach
    aliases: List[str]
    description: str
    danger_zones: List[DangerZone]
    key_landmarks: List[str]
    positioning: str
    critical_steps: List[str]
    subspecialty: str


# =============================================================================
# APPROACH-SPECIFIC DANGER ZONE MAPPINGS
# =============================================================================

APPROACH_PROFILES: Dict[SurgicalApproach, ApproachProfile] = {
    # =========================================================================
    # TRANSSPHENOIDAL APPROACHES
    # =========================================================================
    SurgicalApproach.TRANSSPHENOIDAL: ApproachProfile(
        approach=SurgicalApproach.TRANSSPHENOIDAL,
        aliases=["transsphenoidal", "transphenoidal", "TSA", "endonasal", "pituitary approach"],
        description="Approach through sphenoid sinus to sella and parasellar region",
        danger_zones=[
            DangerZone(
                structure="Internal Carotid Artery (parasellar)",
                consequence="Catastrophic hemorrhage, potential fatality",
                frequency="uncommon",
                prevention_note="Use neuronavigation, identify carotid prominence on CT, stay midline",
                bailout_strategy="Packing with muscle/fat, possible endovascular rescue",
            ),
            DangerZone(
                structure="Optic Nerve/Chiasm",
                consequence="Visual loss (partial or complete)",
                frequency="common",
                prevention_note="Identify tuberculum sellae, avoid traction, maintain blood supply",
                bailout_strategy="Immediate decompression if noted intraoperatively",
            ),
            DangerZone(
                structure="Cavernous Sinus",
                consequence="Hemorrhage, cranial nerve palsy (III, IV, V1, V2, VI)",
                frequency="uncommon",
                prevention_note="Respect medial wall of cavernous sinus, know invasion patterns",
                bailout_strategy="Packing, bipolar at low settings",
            ),
            DangerZone(
                structure="Pituitary Stalk",
                consequence="Diabetes insipidus, panhypopituitarism",
                frequency="common",
                prevention_note="Identify stalk early, preserve if not tumor-infiltrated",
                bailout_strategy="Hormone replacement therapy",
            ),
            DangerZone(
                structure="Suprasellar Arachnoid",
                consequence="CSF leak",
                frequency="common",
                prevention_note="Intentional opening with repair (fascia, fat, Valsalva test)",
                bailout_strategy="Abdominal fat graft, nasoseptal flap, lumbar drain",
            ),
            DangerZone(
                structure="Hypothalamus",
                consequence="Endocrine dysfunction, cognitive changes, death",
                frequency="rare",
                prevention_note="Know superior limits, avoid traction on third ventricle floor",
                bailout_strategy="Supportive care",
            ),
        ],
        key_landmarks=["Sphenoid ostium", "Sella floor", "Carotid prominence", "Optic protuberance", "Clival recess"],
        positioning="Supine, head neutral, neuronavigation registered",
        critical_steps=["Wide sphenoidotomy", "Sella floor removal", "Dural opening", "Tumor removal", "Closure with graft"],
        subspecialty="skull_base",
    ),

    SurgicalApproach.ENDONASAL_EXPANDED: ApproachProfile(
        approach=SurgicalApproach.ENDONASAL_EXPANDED,
        aliases=["expanded endonasal", "EEA", "endoscopic skull base", "endonasal endoscopic"],
        description="Extended endonasal approach for clival, anterior fossa, or parasellar pathology",
        danger_zones=[
            DangerZone(
                structure="Internal Carotid Artery",
                consequence="Catastrophic hemorrhage",
                frequency="uncommon",
                prevention_note="Neuronavigation, intraoperative Doppler, know ICA course",
                bailout_strategy="Muscle packing, endovascular backup on standby",
            ),
            DangerZone(
                structure="Optic Nerves",
                consequence="Blindness",
                frequency="common",
                prevention_note="Identify optic canal, opticocarotid recess",
                bailout_strategy="Optic nerve decompression",
            ),
            DangerZone(
                structure="Basilar Artery/Perforators",
                consequence="Stroke, coma, death",
                frequency="rare",
                prevention_note="Identify basilar trunk, preserve perforators",
                bailout_strategy="None effective if perforator stroke occurs",
            ),
            DangerZone(
                structure="Cranial Nerve VI (Abducens)",
                consequence="Lateral rectus palsy, diplopia",
                frequency="common",
                prevention_note="Know Dorello's canal location, gentle dissection",
                bailout_strategy="Usually recovers with conservative management",
            ),
            DangerZone(
                structure="Anterior Cerebral Arteries (A1/A2)",
                consequence="Frontal lobe stroke",
                frequency="rare",
                prevention_note="Know ACA course, preserve branches",
                bailout_strategy="Vascular repair if injury noted",
            ),
        ],
        key_landmarks=["Vidian canal", "Opticocarotid recess", "Clival recess", "Foramen lacerum"],
        positioning="Supine, head extended, neuronavigation",
        critical_steps=["Nasoseptal flap harvest", "Wide sphenoidotomy", "Bony removal", "Dural opening", "Tumor removal", "Multilayer closure"],
        subspecialty="skull_base",
    ),

    # =========================================================================
    # PTERIONAL/LATERAL SKULL BASE
    # =========================================================================
    SurgicalApproach.PTERIONAL: ApproachProfile(
        approach=SurgicalApproach.PTERIONAL,
        aliases=["pterional", "frontotemporal", "sylvian", "FT craniotomy"],
        description="Workhorse approach for anterior circulation aneurysms, sellar/parasellar lesions",
        danger_zones=[
            DangerZone(
                structure="Middle Cerebral Artery (M1/M2)",
                consequence="Major hemispheric stroke, hemiplegia, aphasia",
                frequency="common",
                prevention_note="Meticulous sylvian dissection, sharp technique, identify M1 early",
                bailout_strategy="Temporary clip, bypass if needed",
            ),
            DangerZone(
                structure="Frontal Branch of Facial Nerve",
                consequence="Forehead paralysis, brow ptosis",
                frequency="common",
                prevention_note="Interfascial or subfascial dissection, protect in fat pad",
                bailout_strategy="Usually permanent if transected",
            ),
            DangerZone(
                structure="Superficial Temporal Artery",
                consequence="Loss of bypass conduit, scalp necrosis (rare)",
                frequency="uncommon",
                prevention_note="Identify in preauricular area, preserve during exposure",
                bailout_strategy="Alternative bypass conduit (radial artery graft)",
            ),
            DangerZone(
                structure="Sylvian Veins",
                consequence="Venous infarction, brain swelling",
                frequency="uncommon",
                prevention_note="Preserve superficial sylvian vein, open arachnoid widely",
                bailout_strategy="Maximize venous drainage, decompressive craniectomy if severe",
            ),
            DangerZone(
                structure="Internal Carotid Artery (supraclinoid)",
                consequence="Massive stroke, death",
                frequency="uncommon",
                prevention_note="Early proximal control, meticulous dissection",
                bailout_strategy="Temporary clip, direct repair or bypass",
            ),
            DangerZone(
                structure="Optic Nerve (intracranial)",
                consequence="Monocular blindness",
                frequency="uncommon",
                prevention_note="Open optic cistern, release falciform ligament",
                bailout_strategy="Usually irreversible if devascularized",
            ),
            DangerZone(
                structure="Olfactory Nerve",
                consequence="Anosmia",
                frequency="common",
                prevention_note="Minimize frontal lobe retraction, preserve olfactory sulcus",
                bailout_strategy="None - often accepted as trade-off",
            ),
            DangerZone(
                structure="Anterior Choroidal Artery",
                consequence="Internal capsule infarct, hemiplegia",
                frequency="rare",
                prevention_note="Identify at ICA, preserve during aneurysm clipping",
                bailout_strategy="ICG angiography to confirm patency",
            ),
        ],
        key_landmarks=["Sylvian fissure", "Sphenoid ridge", "Superior orbital fissure", "Optic nerve", "ICA bifurcation"],
        positioning="Supine, head rotated 30-45°, vertex down, pin fixation",
        critical_steps=["Curvilinear incision", "Interfascial dissection", "Craniotomy", "Sphenoid wing drill", "Sylvian fissure split", "Target exposure"],
        subspecialty="vascular",
    ),

    SurgicalApproach.ORBITOZYGOMATIC: ApproachProfile(
        approach=SurgicalApproach.ORBITOZYGOMATIC,
        aliases=["orbitozygomatic", "OZ", "orbito-zygomatic", "cranio-orbital"],
        description="Extended pterional with orbital and zygomatic osteotomies for basilar apex access",
        danger_zones=[
            DangerZone(
                structure="Basilar Artery Apex",
                consequence="Brainstem stroke, death",
                frequency="common",
                prevention_note="Temporary clipping, perforator preservation, ICG",
                bailout_strategy="Immediate repair, consider bypass",
            ),
            DangerZone(
                structure="P1 Perforators",
                consequence="Thalamic/midbrain stroke",
                frequency="common",
                prevention_note="Visualize all perforators, ensure flow after clipping",
                bailout_strategy="ICG angiography, clip repositioning",
            ),
            DangerZone(
                structure="Oculomotor Nerve (CN III)",
                consequence="Ptosis, diplopia, fixed dilated pupil",
                frequency="common",
                prevention_note="Identify CN III in ambient cistern, avoid traction",
                bailout_strategy="Usually recovers over months if not transected",
            ),
            DangerZone(
                structure="Posterior Cerebral Artery",
                consequence="Occipital stroke, visual field cut",
                frequency="uncommon",
                prevention_note="Identify P1 and P2 segments, protect during dissection",
                bailout_strategy="Direct repair if injured",
            ),
            DangerZone(
                structure="Superior Orbital Fissure Contents",
                consequence="Ophthalmoplegia, numbness (V1)",
                frequency="uncommon",
                prevention_note="Careful osteotomy, avoid SOF violation",
                bailout_strategy="Usually temporary if contusion only",
            ),
        ],
        key_landmarks=["Basilar apex", "P1 segment", "SCA", "CN III", "Posterior clinoid"],
        positioning="Supine, head rotated 30°, Mayfield",
        critical_steps=["Pterional craniotomy", "Orbital osteotomy", "Zygomatic osteotomy", "Anterior clinoidectomy", "Membrane opening"],
        subspecialty="vascular",
    ),

    # =========================================================================
    # POSTERIOR FOSSA APPROACHES
    # =========================================================================
    SurgicalApproach.RETROSIGMOID: ApproachProfile(
        approach=SurgicalApproach.RETROSIGMOID,
        aliases=["retrosigmoid", "retromastoid", "CPA approach", "suboccipital lateral"],
        description="Approach to cerebellopontine angle and lateral posterior fossa",
        danger_zones=[
            DangerZone(
                structure="Facial Nerve (CN VII)",
                consequence="Facial paralysis (House-Brackmann grade)",
                frequency="common",
                prevention_note="Continuous EMG monitoring, identify nerve early, preserve integrity",
                bailout_strategy="Primary repair, cable graft, facial-hypoglossal anastomosis",
            ),
            DangerZone(
                structure="Cochlear Nerve (CN VIII)",
                consequence="Hearing loss",
                frequency="common",
                prevention_note="Hearing preservation protocol, ABR monitoring, gentle dissection",
                bailout_strategy="Often not salvageable; document preop hearing status",
            ),
            DangerZone(
                structure="AICA (Anterior Inferior Cerebellar Artery)",
                consequence="Cerebellar infarct, labyrinthine artery loss",
                frequency="common",
                prevention_note="Identify AICA loop in IAC, preserve labyrinthine branch",
                bailout_strategy="Direct repair if main trunk injured",
            ),
            DangerZone(
                structure="Trigeminal Nerve (CN V)",
                consequence="Facial numbness, corneal anesthesia",
                frequency="common",
                prevention_note="Identify in upper CPA, protect during tumor manipulation",
                bailout_strategy="Corneal protection protocol if V1 affected",
            ),
            DangerZone(
                structure="Lower Cranial Nerves (IX, X, XI)",
                consequence="Dysphagia, aspiration, voice change",
                frequency="uncommon",
                prevention_note="Identify in lower CPA, avoid traction",
                bailout_strategy="Speech/swallow evaluation, possible tracheostomy/PEG",
            ),
            DangerZone(
                structure="Sigmoid/Transverse Sinus",
                consequence="Air embolism, hemorrhage",
                frequency="uncommon",
                prevention_note="Identify sinuses on imaging, avoid direct injury",
                bailout_strategy="Packing with Surgicel, direct repair",
            ),
            DangerZone(
                structure="Cerebellar Retraction Injury",
                consequence="Cerebellar edema, herniation",
                frequency="uncommon",
                prevention_note="Release CSF early, minimize retraction",
                bailout_strategy="Decompressive surgery if severe",
            ),
        ],
        key_landmarks=["Sigmoid sinus", "Transverse sinus", "Asterion", "Internal acoustic meatus", "Porus acusticus"],
        positioning="Lateral decubitus or park bench, head flexed",
        critical_steps=["C-shaped incision", "Craniotomy behind sigmoid", "Dural opening", "CSF release", "Tumor/target exposure"],
        subspecialty="skull_base",
    ),

    SurgicalApproach.FAR_LATERAL: ApproachProfile(
        approach=SurgicalApproach.FAR_LATERAL,
        aliases=["far lateral", "extreme lateral", "transcondylar", "FLATA"],
        description="Approach to foramen magnum, lower clivus, and anterior craniocervical junction",
        danger_zones=[
            DangerZone(
                structure="Vertebral Artery (V3 segment)",
                consequence="Brainstem stroke, death",
                frequency="common",
                prevention_note="Identify VA in suboccipital triangle, protect during condyle drilling",
                bailout_strategy="Direct repair, possible clip occlusion with bypass if needed",
            ),
            DangerZone(
                structure="Hypoglossal Nerve (CN XII)",
                consequence="Tongue deviation, dysphagia, dysarthria",
                frequency="common",
                prevention_note="Identify hypoglossal canal, protect during condyle removal",
                bailout_strategy="Usually recovers; speech therapy",
            ),
            DangerZone(
                structure="Spinal Accessory Nerve (CN XI)",
                consequence="Shoulder weakness, scapular winging",
                frequency="uncommon",
                prevention_note="Identify in posterior triangle during exposure",
                bailout_strategy="Physical therapy; usually recovers",
            ),
            DangerZone(
                structure="Lower Cranial Nerves (IX, X)",
                consequence="Dysphagia, aspiration",
                frequency="common",
                prevention_note="Identify in cerebellomedullary fissure",
                bailout_strategy="NPO, PEG if prolonged; swallow evaluation",
            ),
            DangerZone(
                structure="PICA (Posterior Inferior Cerebellar Artery)",
                consequence="Lateral medullary syndrome, cerebellar infarct",
                frequency="common",
                prevention_note="Identify PICA origin, preserve perforators to medulla",
                bailout_strategy="Direct repair if possible",
            ),
            DangerZone(
                structure="Craniocervical Junction Instability",
                consequence="Instability, quadriplegia",
                frequency="uncommon",
                prevention_note="Limit condyle resection to 50%, preserve occipitocervical ligaments",
                bailout_strategy="Occipitocervical fusion if destabilized",
            ),
        ],
        key_landmarks=["Vertebral artery V3", "C1 lateral mass", "Hypoglossal canal", "Jugular tubercle", "Occipital condyle"],
        positioning="Lateral decubitus or park bench, head flexed and rotated",
        critical_steps=["Hockey stick incision", "Suboccipital muscles", "Lateral rim foramen magnum", "C1 lateral mass", "Condyle drilling"],
        subspecialty="skull_base",
    ),

    # =========================================================================
    # SPINE APPROACHES
    # =========================================================================
    SurgicalApproach.ANTERIOR_CERVICAL: ApproachProfile(
        approach=SurgicalApproach.ANTERIOR_CERVICAL,
        aliases=["anterior cervical", "ACDF", "anterior discectomy", "Smith-Robinson"],
        description="Anterior approach to cervical spine for discectomy and fusion",
        danger_zones=[
            DangerZone(
                structure="Recurrent Laryngeal Nerve",
                consequence="Hoarseness, vocal cord paralysis",
                frequency="common",
                prevention_note="Retract gently, avoid excessive traction on esophagus",
                bailout_strategy="Usually recovers; ENT evaluation if persistent",
            ),
            DangerZone(
                structure="Carotid Artery",
                consequence="Stroke, hemorrhage",
                frequency="rare",
                prevention_note="Identify in sheath, retract gently medially",
                bailout_strategy="Direct repair, vascular surgery consult",
            ),
            DangerZone(
                structure="Vertebral Artery",
                consequence="Stroke, hemorrhage",
                frequency="rare",
                prevention_note="Stay midline, avoid lateral decompression >14mm from midline",
                bailout_strategy="Packing, possible endovascular treatment",
            ),
            DangerZone(
                structure="Esophagus",
                consequence="Perforation, mediastinitis",
                frequency="rare",
                prevention_note="Gentle retraction, avoid sharp instruments laterally",
                bailout_strategy="Primary repair, drainage, antibiotics",
            ),
            DangerZone(
                structure="Spinal Cord",
                consequence="Quadriplegia",
                frequency="rare",
                prevention_note="Meticulous technique, neuromonitoring, avoid over-distraction",
                bailout_strategy="Decompress if SSEP/MEP changes",
            ),
            DangerZone(
                structure="Sympathetic Chain",
                consequence="Horner syndrome",
                frequency="uncommon",
                prevention_note="Avoid lateral dissection at longus colli level",
                bailout_strategy="Usually permanent if injured",
            ),
        ],
        key_landmarks=["Carotid sheath", "Longus colli", "Uncovertebral joint", "Disc space"],
        positioning="Supine, neck slightly extended, shoulder roll",
        critical_steps=["Transverse incision", "Platysma division", "Deep dissection", "Longus colli elevation", "Discectomy", "Fusion"],
        subspecialty="spine",
    ),

    SurgicalApproach.POSTERIOR_LUMBAR: ApproachProfile(
        approach=SurgicalApproach.POSTERIOR_LUMBAR,
        aliases=["posterior lumbar", "laminectomy", "discectomy", "microdiscectomy", "lumbar fusion"],
        description="Posterior approach to lumbar spine",
        danger_zones=[
            DangerZone(
                structure="Nerve Root (exiting and traversing)",
                consequence="Radiculopathy, foot drop, bowel/bladder dysfunction",
                frequency="common",
                prevention_note="Identify nerve root, protect with cottonoid, avoid excessive retraction",
                bailout_strategy="Decompress if compressed; explore if deficit postop",
            ),
            DangerZone(
                structure="Cauda Equina",
                consequence="Cauda equina syndrome, paralysis, incontinence",
                frequency="rare",
                prevention_note="Midline approach, identify thecal sac early",
                bailout_strategy="Immediate decompression if suspected",
            ),
            DangerZone(
                structure="Dura Mater",
                consequence="CSF leak, pseudomeningocele",
                frequency="common",
                prevention_note="Careful dissection, especially with scar tissue",
                bailout_strategy="Primary repair with graft, fibrin sealant, lumbar drain",
            ),
            DangerZone(
                structure="Great Vessels (iliac, aorta)",
                consequence="Catastrophic hemorrhage",
                frequency="rare",
                prevention_note="Do not plunge instruments anteriorly, know disc space depth",
                bailout_strategy="Immediate vascular surgery, pack and transfuse",
            ),
            DangerZone(
                structure="Facet Joints",
                consequence="Iatrogenic instability",
                frequency="common",
                prevention_note="Preserve >50% of facet, consider fusion if excessive",
                bailout_strategy="Fusion procedure",
            ),
        ],
        key_landmarks=["Pars interarticularis", "Facet joint", "Disc space", "Pedicle", "Nerve root"],
        positioning="Prone on Wilson frame or Jackson table",
        critical_steps=["Midline incision", "Muscle dissection", "Laminotomy/laminectomy", "Discectomy/decompression", "Fusion if indicated"],
        subspecialty="spine",
    ),
}

# Add more approach profiles as needed...


class TrajectoryDangerZoneAnalyzer:
    """
    Analyzes content for surgical approach and provides trajectory-specific danger zones.

    Usage:
        analyzer = TrajectoryDangerZoneAnalyzer()
        approach = analyzer.detect_approach(content)
        danger_zones = analyzer.get_danger_zones(approach)
        gaps = analyzer.analyze_danger_zone_coverage(content, approach)
    """

    def __init__(self):
        self.approach_profiles = APPROACH_PROFILES
        self._build_alias_index()
        self.logger = logging.getLogger(__name__)

    def _build_alias_index(self):
        """Build index for fast approach detection."""
        self._alias_to_approach: Dict[str, SurgicalApproach] = {}
        for approach, profile in self.approach_profiles.items():
            for alias in profile.aliases:
                self._alias_to_approach[alias.lower()] = approach

    def detect_approach(self, content: str) -> List[SurgicalApproach]:
        """
        Detect surgical approach(es) mentioned in content.

        Returns list of detected approaches (may be multiple if content
        discusses multiple techniques).
        """
        content_lower = content.lower()
        detected = set()

        # Check each alias
        for alias, approach in self._alias_to_approach.items():
            if alias in content_lower:
                detected.add(approach)

        # Additional pattern-based detection
        approach_patterns = {
            SurgicalApproach.TRANSSPHENOIDAL: [
                r"\btrans-?sphenoidal\b",
                r"\bendonasal\s+(?:approach|surgery)\b",
                r"\bpituitary\s+(?:surgery|adenoma)\s+(?:approach|removal)\b",
            ],
            SurgicalApproach.PTERIONAL: [
                r"\bpterional\b",
                r"\bfronto-?temporal\s+craniotomy\b",
                r"\bsylvian\s+(?:fissure|approach)\b",
            ],
            SurgicalApproach.RETROSIGMOID: [
                r"\bretro-?sigmoid\b",
                r"\bCPA\s+(?:approach|surgery)\b",
                r"\bacoustic\s+neuroma\s+(?:surgery|resection)\b",
            ],
            SurgicalApproach.FAR_LATERAL: [
                r"\bfar\s+lateral\b",
                r"\bextreme\s+lateral\b",
                r"\btrans-?condylar\b",
                r"\bforamen\s+magnum\s+(?:approach|tumor)\b",
            ],
            SurgicalApproach.ANTERIOR_CERVICAL: [
                r"\bACDF\b",
                r"\banterior\s+cervical\s+discectomy\b",
                r"\bSmith-?Robinson\b",
            ],
            SurgicalApproach.POSTERIOR_LUMBAR: [
                r"\blaminectomy\b",
                r"\bmicro-?discectomy\b",
                r"\bposterior\s+lumbar\s+(?:fusion|decompression)\b",
            ],
            SurgicalApproach.ORBITOZYGOMATIC: [
                r"\borbito-?zygomatic\b",
                r"\bOZ\s+(?:approach|craniotomy)\b",
            ],
            SurgicalApproach.SUPRACEREBELLAR_INFRATENTORIAL: [
                r"\bsupra-?cerebellar\s+infra-?tentorial\b",
                r"\bSCIT\b",
                r"\bpineal\s+(?:region\s+)?approach\b",
            ],
        }

        for approach, patterns in approach_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    detected.add(approach)
                    break

        return list(detected) if detected else [SurgicalApproach.UNKNOWN]

    def get_danger_zones(
        self,
        approach: SurgicalApproach
    ) -> List[DangerZone]:
        """Get danger zones specific to an approach."""
        profile = self.approach_profiles.get(approach)
        if profile:
            return profile.danger_zones
        return []

    def get_all_danger_zones_for_approaches(
        self,
        approaches: List[SurgicalApproach]
    ) -> List[DangerZone]:
        """Get combined danger zones for multiple approaches."""
        seen = set()
        zones = []

        for approach in approaches:
            for zone in self.get_danger_zones(approach):
                if zone.structure not in seen:
                    seen.add(zone.structure)
                    zones.append(zone)

        return zones

    def analyze_danger_zone_coverage(
        self,
        content: str,
        approaches: Optional[List[SurgicalApproach]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze content for coverage of approach-specific danger zones.

        Returns list of gap entries for missing danger zone coverage.
        """
        if approaches is None:
            approaches = self.detect_approach(content)

        if SurgicalApproach.UNKNOWN in approaches and len(approaches) == 1:
            # Cannot analyze without knowing approach
            return []

        gaps = []
        content_lower = content.lower()

        for approach in approaches:
            if approach == SurgicalApproach.UNKNOWN:
                continue

            danger_zones = self.get_danger_zones(approach)
            profile = self.approach_profiles.get(approach)

            for zone in danger_zones:
                # Check if danger zone is mentioned
                zone_terms = zone.structure.lower().split()
                structure_mentioned = any(
                    term in content_lower
                    for term in zone_terms
                    if len(term) > 3
                )

                # Also check for abbreviated forms
                abbreviated_check = zone.structure.lower() in content_lower

                if not (structure_mentioned or abbreviated_check):
                    gaps.append({
                        "approach": approach.value,
                        "structure": zone.structure,
                        "consequence": zone.consequence,
                        "frequency": zone.frequency,
                        "prevention_note": zone.prevention_note,
                        "bailout_strategy": zone.bailout_strategy,
                        "priority": "CRITICAL" if zone.frequency == "common" else "HIGH",
                    })

        return gaps

    def get_approach_profile(
        self,
        approach: SurgicalApproach
    ) -> Optional[ApproachProfile]:
        """Get full profile for an approach."""
        return self.approach_profiles.get(approach)

    def get_key_landmarks(
        self,
        approaches: List[SurgicalApproach]
    ) -> List[str]:
        """Get combined key landmarks for approaches."""
        landmarks = set()
        for approach in approaches:
            profile = self.approach_profiles.get(approach)
            if profile:
                landmarks.update(profile.key_landmarks)
        return list(landmarks)

    def generate_safety_checklist(
        self,
        approach: SurgicalApproach
    ) -> Dict[str, Any]:
        """Generate a safety checklist for a specific approach."""
        profile = self.approach_profiles.get(approach)
        if not profile:
            return {}

        return {
            "approach": approach.value,
            "positioning": profile.positioning,
            "critical_steps": profile.critical_steps,
            "danger_zones": [
                {
                    "structure": z.structure,
                    "consequence": z.consequence,
                    "prevention": z.prevention_note,
                    "bailout": z.bailout_strategy,
                }
                for z in profile.danger_zones
            ],
            "key_landmarks": profile.key_landmarks,
        }
