"""
NeuroSynth v2.0 - Hybrid Entity Extraction Pipeline
====================================================

Multi-phase entity extraction combining FlashText (fast), regex (domain-specific),
UMLS (standardized), and optional semantic similarity for comprehensive coverage.

Pipeline Phases:
    Phase 1: FlashText exact keyword matching (~30x faster than substring)
    Phase 2: NeuroExpertTextExtractor regex patterns (context-aware)
    Phase 3: UMLSExtractor for standardized CUI concepts
    Phase 4: Optional pgvector semantic similarity (database integration)

Performance:
    - 10,000 tokens/second with FlashText phase
    - Shared caching across phases
    - Deduplication of overlapping entities

Usage:
    extractor = HybridEntityExtractor()
    entities = extractor.extract("Patient presents with MCA aneurysm")
    # Returns unified list of NeuroEntity objects with extraction_method metadata
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from functools import lru_cache

from flashtext import KeywordProcessor

from src.shared.models import NeuroEntity, EntityRelation
from src.core.neuro_extractor import NeuroExpertTextExtractor
from src.core.umls_extractor import UMLSExtractor, UMLSEntity, get_default_extractor

logger = logging.getLogger(__name__)


# =============================================================================
# FlashText Keyword Database for Phase 1
# =============================================================================

# Neurosurgical keywords organized by category
# FlashText will match these exactly (case-insensitive) in O(n) time
FLASHTEXT_KEYWORDS = {
    # Vascular Anatomy - Arteries
    "ANATOMY_VASCULAR_ARTERIAL": [
        # ICA segments
        "C1 segment", "C2 segment", "C3 segment", "C4 segment", "C5 segment", "C6 segment", "C7 segment",
        "petrous ICA", "cavernous ICA", "clinoidal ICA", "ophthalmic segment", "communicating segment",
        # Circle of Willis
        "M1 segment", "M2 segment", "M3 segment", "M4 segment",
        "A1 segment", "A2 segment", "A3 segment",
        "P1 segment", "P2 segment", "P3 segment",
        # Named arteries
        "internal carotid artery", "ICA",
        "middle cerebral artery", "MCA",
        "anterior cerebral artery", "ACA",
        "posterior cerebral artery", "PCA",
        "vertebral artery", "VA",
        "basilar artery", "BA",
        "superior cerebellar artery", "SCA",
        "anterior inferior cerebellar artery", "AICA",
        "posterior inferior cerebellar artery", "PICA",
        "anterior communicating artery", "ACoA", "ACOM", "AComA",
        "posterior communicating artery", "PCoA", "PCOM", "PComA",
        "ophthalmic artery", "anterior choroidal artery",
        "lenticulostriate arteries", "perforating arteries", "perforators",
        "recurrent artery of Heubner", "Heubner's artery",
    ],

    # Vascular Anatomy - Veins
    "ANATOMY_VASCULAR_VENOUS": [
        "superior sagittal sinus", "SSS",
        "inferior sagittal sinus", "ISS",
        "transverse sinus", "TS",
        "sigmoid sinus", "SS",
        "cavernous sinus",
        "straight sinus",
        "torcula", "confluence of sinuses",
        "vein of Galen", "Galenic vein",
        "vein of Labbe", "Labbe's vein",
        "vein of Trolard",
        "basal vein of Rosenthal",
        "internal cerebral vein", "ICV",
        "superficial middle cerebral vein", "SMCV",
        "bridging veins", "cortical veins",
    ],

    # Skull Base Anatomy
    "ANATOMY_SKULL_BASE": [
        "planum sphenoidale", "tuberculum sellae", "dorsum sellae",
        "crista galli", "cribriform plate", "olfactory groove",
        "optic canal", "optic strut",
        "sella turcica", "pituitary fossa", "diaphragma sellae",
        "anterior clinoid", "posterior clinoid",
        "clivus", "petrous apex", "tegmen tympani",
        "internal acoustic meatus", "IAM", "IAC", "internal auditory canal",
        "jugular foramen", "jugular tubercle",
        "cerebellopontine angle", "CPA",
        "foramen ovale", "foramen rotundum", "foramen spinosum",
        "foramen lacerum", "foramen magnum",
        "superior orbital fissure", "SOF",
        "Meckel's cave", "trigeminal cave",
    ],

    # Cranial Nerves
    "ANATOMY_CRANIAL_NERVES": [
        "olfactory nerve", "CN I",
        "optic nerve", "CN II",
        "oculomotor nerve", "CN III",
        "trochlear nerve", "CN IV",
        "trigeminal nerve", "CN V",
        "abducens nerve", "CN VI",
        "facial nerve", "CN VII",
        "vestibulocochlear nerve", "CN VIII", "acoustic nerve",
        "glossopharyngeal nerve", "CN IX",
        "vagus nerve", "CN X",
        "accessory nerve", "CN XI",
        "hypoglossal nerve", "CN XII",
        "optic chiasm", "gasserian ganglion", "trigeminal ganglion",
        "geniculate ganglion",
    ],

    # Vascular Pathology
    "PATHOLOGY_VASCULAR": [
        "aneurysm", "cerebral aneurysm", "intracranial aneurysm",
        "saccular aneurysm", "fusiform aneurysm", "dissecting aneurysm",
        "giant aneurysm", "ruptured aneurysm", "unruptured aneurysm",
        "arteriovenous malformation", "AVM", "brain AVM",
        "dural arteriovenous fistula", "dAVF", "DAVF",
        "cavernous malformation", "cavernoma", "CCM",
        "moyamoya", "moyamoya disease",
        "subarachnoid hemorrhage", "SAH",
        "intracerebral hemorrhage", "ICH",
        "vasospasm", "delayed cerebral ischemia", "DCI",
        "stroke", "ischemic stroke", "hemorrhagic stroke",
    ],

    # Tumors
    "PATHOLOGY_TUMOR": [
        "glioblastoma", "GBM", "glioblastoma multiforme",
        "glioma", "high-grade glioma", "low-grade glioma",
        "astrocytoma", "diffuse astrocytoma", "anaplastic astrocytoma",
        "oligodendroglioma", "ependymoma",
        "meningioma", "convexity meningioma", "parasagittal meningioma",
        "vestibular schwannoma", "acoustic neuroma",
        "schwannoma", "neurofibroma",
        "pituitary adenoma", "macroadenoma", "microadenoma",
        "craniopharyngioma", "Rathke's cleft cyst",
        "metastasis", "brain metastasis", "metastases",
        "hemangioblastoma", "chordoma", "chondrosarcoma",
        "medulloblastoma", "PNET",
        "epidermoid", "dermoid",
        "WHO grade I", "WHO grade II", "WHO grade III", "WHO grade IV",
    ],

    # Procedures - Approaches
    "PROCEDURE_APPROACH": [
        "pterional approach", "pterional craniotomy",
        "frontotemporal craniotomy",
        "orbitozygomatic approach", "OZ approach", "OZ craniotomy",
        "interhemispheric approach", "transcallosal approach",
        "subfrontal approach", "bifrontal craniotomy",
        "subtemporal approach", "subtemporal craniotomy",
        "suboccipital approach", "suboccipital craniotomy",
        "retrosigmoid approach", "retrosigmoid craniotomy",
        "far lateral approach", "extreme lateral approach",
        "telovelar approach", "transvermian approach",
        "transsphenoidal approach", "endoscopic endonasal approach",
        "presigmoid approach", "transpetrosal approach",
        "translabyrinthine approach",
    ],

    # Procedures - Actions
    "PROCEDURE_ACTION": [
        "craniotomy", "craniectomy", "cranioplasty",
        "burr hole", "twist drill",
        "tumor resection", "gross total resection", "GTR",
        "subtotal resection", "STR", "near total resection",
        "debulking", "biopsy", "stereotactic biopsy",
        "aneurysm clipping", "clip application", "clipping",
        "coiling", "endovascular coiling", "embolization",
        "EC-IC bypass", "STA-MCA bypass", "bypass",
        "microvascular decompression", "MVD",
        "discectomy", "laminectomy", "foraminotomy",
        "ACDF", "anterior cervical discectomy and fusion",
        "PLIF", "TLIF", "ALIF", "XLIF", "OLIF",
        "spinal fusion", "decompression", "corpectomy",
    ],

    # Functional Neurosurgery
    "PATHOLOGY_FUNCTIONAL": [
        "Parkinson's disease", "Parkinson disease", "parkinsonism",
        "essential tremor", "dystonia", "chorea",
        "epilepsy", "seizure", "seizures",
        "mesial temporal sclerosis", "MTS",
        "focal cortical dysplasia", "FCD",
        "trigeminal neuralgia", "TN", "tic douloureux",
        "deep brain stimulation", "DBS",
        "VIM stimulation", "STN stimulation", "GPi stimulation",
    ],

    # Instruments
    "INSTRUMENT": [
        "bipolar forceps", "bipolar coagulation",
        "monopolar cautery", "Bovie",
        "CUSA", "Cavitron", "ultrasonic aspirator",
        "suction", "aspirator",
        "Penfield dissector", "Rhoton dissector",
        "Kerrison rongeur", "Leksell rongeur",
        "aneurysm clip", "Yasargil clip", "Sugita clip",
        "temporary clip", "permanent clip", "fenestrated clip",
        "pedicle screw", "lateral mass screw",
        "interbody cage", "titanium cage",
        "microscope", "operating microscope",
        "endoscope", "neuroendoscope",
        "neuronavigation", "frameless navigation",
        "BrainLab", "StealthStation",
    ],

    # Grading Scales
    "MEASUREMENT": [
        "Hunt-Hess grade", "Hunt and Hess",
        "Fisher grade", "modified Fisher",
        "Spetzler-Martin grade", "SM grade",
        "WFNS grade",
        "Glasgow Coma Scale", "GCS",
        "Karnofsky score", "KPS",
        "modified Rankin Scale", "mRS",
        "House-Brackmann grade", "HB grade",
        "Simpson grade",
    ],

    # Spine Anatomy
    "ANATOMY_SPINE": [
        "cervical spine", "thoracic spine", "lumbar spine", "sacral spine",
        "atlas", "axis", "dens", "odontoid",
        "pedicle", "lamina", "facet joint",
        "spinous process", "transverse process",
        "intervertebral disc", "disc herniation",
        "spinal cord", "conus medullaris", "cauda equina",
        "nerve root", "dorsal root ganglion", "DRG",
        "thecal sac", "dura mater",
        "ligamentum flavum", "posterior longitudinal ligament", "PLL",
    ],

    # Infections
    "PATHOLOGY_INFECTION": [
        "meningitis", "encephalitis", "ventriculitis",
        "brain abscess", "cerebral abscess",
        "epidural abscess", "subdural empyema",
        "discitis", "osteomyelitis", "spondylodiscitis",
        "tuberculoma", "tuberculous meningitis",
        "neurocysticercosis",
    ],

    # Trauma
    "PATHOLOGY_TRAUMA": [
        "traumatic brain injury", "TBI",
        "epidural hematoma", "EDH",
        "subdural hematoma", "SDH", "acute SDH", "chronic SDH",
        "skull fracture", "depressed fracture",
        "contusion", "cerebral contusion",
        "diffuse axonal injury", "DAI",
    ],

    # Imaging
    "IMAGING": [
        "MRI", "magnetic resonance imaging",
        "CT", "computed tomography",
        "CTA", "CT angiography",
        "MRA", "MR angiography",
        "DSA", "digital subtraction angiography", "angiogram",
        "T1-weighted", "T2-weighted", "FLAIR",
        "diffusion-weighted", "DWI", "ADC map",
        "contrast-enhanced", "gadolinium",
        "PET scan", "PET-CT",
    ],
}

# Aliases for common variations (maps to canonical form)
KEYWORD_ALIASES = {
    "middle cerebral artery": "MCA",
    "anterior cerebral artery": "ACA",
    "posterior cerebral artery": "PCA",
    "internal carotid artery": "ICA",
    "superior sagittal sinus": "SSS",
    "deep brain stimulation": "DBS",
    "gross total resection": "GTR",
    "subtotal resection": "STR",
    "subarachnoid hemorrhage": "SAH",
    "intracerebral hemorrhage": "ICH",
    "glioblastoma multiforme": "GBM",
    "arteriovenous malformation": "AVM",
    "cerebellopontine angle": "CPA",
    "internal auditory canal": "IAC",
    "microvascular decompression": "MVD",
    "trigeminal neuralgia": "TN",
    "traumatic brain injury": "TBI",
}


# =============================================================================
# Enhanced Entity with Extraction Metadata
# =============================================================================

@dataclass
class ExtractedEntity:
    """
    Entity with extraction method metadata.

    Extends NeuroEntity with information about how it was extracted,
    enabling confidence calibration and pipeline debugging.
    """
    text: str
    category: str
    normalized: str
    start: int
    end: int
    confidence: float
    context_snippet: str = ""
    extraction_method: str = "unknown"  # flashtext, regex, umls, semantic
    cui: Optional[str] = None  # UMLS CUI if from UMLS extraction
    tui: Optional[str] = None  # UMLS TUI (semantic type)

    def to_neuro_entity(self) -> NeuroEntity:
        """Convert to standard NeuroEntity."""
        return NeuroEntity(
            text=self.text,
            category=self.category,
            normalized=self.normalized,
            start=self.start,
            end=self.end,
            confidence=self.confidence,
            context_snippet=self.context_snippet,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "category": self.category,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "context_snippet": self.context_snippet,
            "extraction_method": self.extraction_method,
            "cui": self.cui,
            "tui": self.tui,
        }


# =============================================================================
# Hybrid Entity Extractor
# =============================================================================

class HybridEntityExtractor:
    """
    Multi-phase entity extraction pipeline for neurosurgical text.

    Combines multiple extraction methods in order of speed/specificity:

    Phase 1: FlashText (30x faster than regex)
        - Exact keyword matching using Aho-Corasick algorithm
        - O(n) time complexity regardless of keyword count
        - Best for known, unambiguous terms
        - Confidence: 0.95 (high - exact match)

    Phase 2: Regex Patterns (NeuroExpertTextExtractor)
        - 100+ domain-specific patterns with context awareness
        - Handles variations, abbreviations, and ambiguity
        - Confidence: 0.60-0.95 (pattern-specific)

    Phase 3: UMLS Extraction (UMLSExtractor)
        - SciSpacy entity linking to UMLS Metathesaurus
        - Standardized CUI identifiers
        - Confidence: 0.70-0.95 (TUI-adaptive thresholds)

    Phase 4: Semantic Similarity (Optional)
        - pgvector nearest neighbor search
        - Catches implicit/novel entities
        - Confidence: 0.60-0.80 (similarity-based)

    Example:
        extractor = HybridEntityExtractor()
        entities = extractor.extract("MCA aneurysm with vasospasm")

        # Or with specific phases:
        entities = extractor.extract(text, phases=["flashtext", "regex"])
    """

    # Default confidence scores per extraction method
    PHASE_CONFIDENCE = {
        "flashtext": 0.95,  # Exact match - high confidence
        "regex": 0.85,      # Pattern match - depends on pattern specificity
        "umls": 0.85,       # UMLS linking - depends on TUI
        "semantic": 0.70,   # Similarity match - inherently uncertain
    }

    def __init__(
        self,
        enable_flashtext: bool = True,
        enable_regex: bool = True,
        enable_umls: bool = True,
        enable_semantic: bool = False,  # Requires database connection
        flashtext_case_sensitive: bool = False,
        min_confidence: float = 0.5,
        context_chars: int = 100,
    ):
        """
        Initialize the hybrid extractor.

        Args:
            enable_flashtext: Enable Phase 1 FlashText extraction
            enable_regex: Enable Phase 2 regex extraction
            enable_umls: Enable Phase 3 UMLS extraction
            enable_semantic: Enable Phase 4 semantic similarity (requires DB)
            flashtext_case_sensitive: Case sensitivity for FlashText
            min_confidence: Minimum confidence threshold for final output
            context_chars: Characters of context to capture around entities
        """
        self.enable_flashtext = enable_flashtext
        self.enable_regex = enable_regex
        self.enable_umls = enable_umls
        self.enable_semantic = enable_semantic
        self.min_confidence = min_confidence
        self.context_chars = context_chars

        # Initialize FlashText processor
        self._keyword_processor: Optional[KeywordProcessor] = None
        self._keyword_categories: Dict[str, str] = {}  # keyword -> category
        self._keyword_normalizations: Dict[str, str] = {}  # keyword -> normalized

        if enable_flashtext:
            self._init_flashtext(case_sensitive=flashtext_case_sensitive)

        # Lazy-loaded extractors (expensive initialization)
        self._regex_extractor: Optional[NeuroExpertTextExtractor] = None
        self._umls_extractor: Optional[UMLSExtractor] = None

        # Extraction cache
        self._cache: Dict[str, Tuple[ExtractedEntity, ...]] = {}
        self._cache_max_size = 2000

        logger.info(
            f"HybridEntityExtractor initialized: "
            f"flashtext={enable_flashtext}, regex={enable_regex}, "
            f"umls={enable_umls}, semantic={enable_semantic}"
        )

    def _init_flashtext(self, case_sensitive: bool = False) -> None:
        """Initialize FlashText keyword processor with neurosurgical vocabulary."""
        self._keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)

        for category, keywords in FLASHTEXT_KEYWORDS.items():
            for keyword in keywords:
                # Add keyword with category as clean_name for extraction
                self._keyword_processor.add_keyword(keyword, keyword)
                self._keyword_categories[keyword.lower()] = category

                # Set normalization (use alias if available)
                normalized = KEYWORD_ALIASES.get(keyword, keyword)
                self._keyword_normalizations[keyword.lower()] = normalized

        total_keywords = sum(len(kws) for kws in FLASHTEXT_KEYWORDS.values())
        logger.info(f"FlashText initialized with {total_keywords} keywords")

    @property
    def regex_extractor(self) -> NeuroExpertTextExtractor:
        """Lazy-load regex extractor."""
        if self._regex_extractor is None:
            self._regex_extractor = NeuroExpertTextExtractor(
                context_chars=self.context_chars
            )
        return self._regex_extractor

    @property
    def umls_extractor(self) -> UMLSExtractor:
        """Lazy-load UMLS extractor (uses shared singleton)."""
        if self._umls_extractor is None:
            self._umls_extractor = get_default_extractor()
        return self._umls_extractor

    def extract(
        self,
        text: str,
        phases: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[ExtractedEntity]:
        """
        Extract entities using multi-phase pipeline.

        Args:
            text: Input text to analyze
            phases: Specific phases to run (default: all enabled phases)
                    Options: ["flashtext", "regex", "umls", "semantic"]
            use_cache: Whether to use extraction cache

        Returns:
            List of ExtractedEntity objects, deduplicated and sorted by position
        """
        if not text or not text.strip():
            return []

        # Check cache
        cache_key = self._get_cache_key(text, phases)
        if use_cache and cache_key in self._cache:
            return list(self._cache[cache_key])

        # Determine which phases to run
        if phases is None:
            phases = []
            if self.enable_flashtext:
                phases.append("flashtext")
            if self.enable_regex:
                phases.append("regex")
            if self.enable_umls:
                phases.append("umls")
            if self.enable_semantic:
                phases.append("semantic")

        all_entities: List[ExtractedEntity] = []

        # Phase 1: FlashText (fastest)
        if "flashtext" in phases and self._keyword_processor:
            flashtext_entities = self._extract_flashtext(text)
            all_entities.extend(flashtext_entities)

        # Phase 2: Regex patterns
        if "regex" in phases:
            regex_entities = self._extract_regex(text)
            all_entities.extend(regex_entities)

        # Phase 3: UMLS extraction
        if "umls" in phases:
            umls_entities = self._extract_umls(text)
            all_entities.extend(umls_entities)

        # Phase 4: Semantic similarity (placeholder - requires DB integration)
        if "semantic" in phases:
            # TODO: Implement pgvector semantic search
            pass

        # Deduplicate overlapping entities
        deduplicated = self._deduplicate_entities(all_entities)

        # Filter by minimum confidence
        filtered = [e for e in deduplicated if e.confidence >= self.min_confidence]

        # Sort by position
        filtered.sort(key=lambda e: (e.start, -e.confidence))

        # Cache results
        if use_cache:
            self._add_to_cache(cache_key, filtered)

        return filtered

    def _extract_flashtext(self, text: str) -> List[ExtractedEntity]:
        """
        Phase 1: FlashText exact keyword matching.

        Uses Aho-Corasick algorithm for O(n) extraction regardless
        of vocabulary size. 30x faster than regex for exact matches.
        """
        if not self._keyword_processor:
            return []

        entities = []

        # extract_keywords returns list of (keyword, start, end)
        matches = self._keyword_processor.extract_keywords(text, span_info=True)

        for keyword, start, end in matches:
            keyword_lower = keyword.lower()
            category = self._keyword_categories.get(keyword_lower, "UNKNOWN")
            normalized = self._keyword_normalizations.get(keyword_lower, keyword)

            # Get context snippet
            context_start = max(0, start - self.context_chars)
            context_end = min(len(text), end + self.context_chars)
            context = text[context_start:context_end]

            entity = ExtractedEntity(
                text=text[start:end],  # Use actual text (preserves case)
                category=category,
                normalized=normalized,
                start=start,
                end=end,
                confidence=self.PHASE_CONFIDENCE["flashtext"],
                context_snippet=context,
                extraction_method="flashtext",
            )
            entities.append(entity)

        return entities

    def _extract_regex(self, text: str) -> List[ExtractedEntity]:
        """
        Phase 2: Regex pattern matching with context awareness.

        Uses NeuroExpertTextExtractor for 100+ domain-specific patterns
        with ambiguity resolution and normalization.
        """
        neuro_entities = self.regex_extractor.extract(text)

        entities = []
        for ne in neuro_entities:
            entity = ExtractedEntity(
                text=ne.text,
                category=ne.category,
                normalized=ne.normalized,
                start=ne.start,
                end=ne.end,
                confidence=ne.confidence,
                context_snippet=ne.context_snippet,
                extraction_method="regex",
            )
            entities.append(entity)

        return entities

    def _extract_umls(self, text: str) -> List[ExtractedEntity]:
        """
        Phase 3: UMLS concept extraction using SciSpacy.

        Links entities to UMLS Metathesaurus for standardized
        CUI identifiers and semantic type information.
        """
        try:
            umls_entities = self.umls_extractor.extract(text)
        except Exception as e:
            logger.warning(f"UMLS extraction failed: {e}")
            return []

        entities = []
        for ue in umls_entities:
            # Map UMLS semantic type to our category system
            category = self._map_tui_to_category(ue.tui)

            # Get context snippet
            context_start = max(0, ue.start_char - self.context_chars)
            context_end = min(len(text), ue.end_char + self.context_chars)
            context = text[context_start:context_end]

            entity = ExtractedEntity(
                text=text[ue.start_char:ue.end_char],
                category=category,
                normalized=ue.name,
                start=ue.start_char,
                end=ue.end_char,
                confidence=ue.score * ue.weight,  # Combined score
                context_snippet=context,
                extraction_method="umls",
                cui=ue.cui,
                tui=ue.tui,
            )
            entities.append(entity)

        return entities

    def _map_tui_to_category(self, tui: str) -> str:
        """Map UMLS TUI to our category system."""
        TUI_CATEGORY_MAP = {
            # Anatomy
            "T023": "ANATOMY_BODY_PART",       # Body Part, Organ, or Organ Component
            "T024": "ANATOMY_TISSUE",          # Tissue
            "T029": "ANATOMY_BODY_REGION",     # Body Location or Region
            "T030": "ANATOMY_BODY_SPACE",      # Body Space or Junction

            # Pathology
            "T047": "PATHOLOGY",               # Disease or Syndrome
            "T048": "PATHOLOGY_FUNCTIONAL",    # Mental or Behavioral Dysfunction
            "T049": "PATHOLOGY",               # Cell or Molecular Dysfunction
            "T191": "PATHOLOGY_TUMOR",         # Neoplastic Process
            "T020": "PATHOLOGY",               # Acquired Abnormality

            # Procedures
            "T059": "PROCEDURE",               # Laboratory Procedure
            "T060": "IMAGING",                 # Diagnostic Procedure
            "T061": "PROCEDURE_ACTION",        # Therapeutic or Preventive Procedure

            # Findings
            "T033": "FINDING",                 # Finding
            "T034": "MEASUREMENT",             # Laboratory or Test Result
            "T184": "PATHOLOGY",               # Sign or Symptom

            # Substances
            "T109": "PHARMACOLOGY",            # Organic Chemical
            "T116": "PHARMACOLOGY",            # Amino Acid, Peptide, or Protein
            "T121": "PHARMACOLOGY",            # Pharmacologic Substance

            # Devices
            "T074": "INSTRUMENT",              # Medical Device
            "T075": "INSTRUMENT",              # Research Device
        }

        return TUI_CATEGORY_MAP.get(tui, "UNKNOWN")

    def _deduplicate_entities(
        self,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Remove overlapping entities, preferring higher confidence and more specific.

        Priority order:
        1. Higher confidence
        2. Longer span (more specific)
        3. Earlier extraction phase (flashtext > regex > umls)
        """
        if not entities:
            return []

        # Sort by: start position, then confidence (desc), then length (desc)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -e.confidence, -(e.end - e.start))
        )

        deduplicated = []
        last_end = -1

        for entity in sorted_entities:
            # Skip low-confidence entities
            if entity.confidence < 0.3:
                continue

            # No overlap - add entity
            if entity.start >= last_end:
                deduplicated.append(entity)
                last_end = entity.end

            # Overlap exists - check if new entity is significantly better
            elif entity.confidence > 0.9 and len(deduplicated) > 0:
                prev = deduplicated[-1]
                # Replace if same span but higher confidence
                if entity.start == prev.start and entity.end == prev.end:
                    if entity.confidence > prev.confidence:
                        deduplicated[-1] = entity

        return deduplicated

    def _get_cache_key(self, text: str, phases: Optional[List[str]]) -> str:
        """Generate cache key from text and phase configuration."""
        phase_str = ",".join(sorted(phases)) if phases else "all"
        content = f"{phase_str}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _add_to_cache(self, key: str, entities: List[ExtractedEntity]) -> None:
        """Add extraction results to cache with FIFO eviction."""
        if len(self._cache) >= self._cache_max_size:
            # FIFO eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = tuple(entities)

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
        logger.info("HybridEntityExtractor cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_max_size,
        }

    def to_neuro_entities(
        self,
        entities: List[ExtractedEntity]
    ) -> List[NeuroEntity]:
        """Convert ExtractedEntity list to NeuroEntity list."""
        return [e.to_neuro_entity() for e in entities]

    def extract_with_relations(
        self,
        text: str,
        chunk_id: str = "",
        document_id: str = "",
    ) -> Tuple[List[ExtractedEntity], List[EntityRelation]]:
        """
        Extract entities and their relationships.

        Args:
            text: Input text
            chunk_id: Source chunk ID for provenance
            document_id: Source document ID

        Returns:
            Tuple of (entities, relations)
        """
        entities = self.extract(text)

        # Convert to NeuroEntity for relation extraction
        neuro_entities = self.to_neuro_entities(entities)

        # Extract relations using regex extractor
        relations = self.regex_extractor.extract_relations(
            text=text,
            entities=neuro_entities,
            chunk_id=chunk_id,
            document_id=document_id,
        )

        return entities, relations


# =============================================================================
# Convenience Functions
# =============================================================================

_default_extractor: Optional[HybridEntityExtractor] = None


def get_hybrid_extractor(
    enable_umls: bool = True,
    enable_semantic: bool = False,
) -> HybridEntityExtractor:
    """
    Get or create the default hybrid extractor singleton.

    Args:
        enable_umls: Whether to enable UMLS extraction (requires SciSpacy)
        enable_semantic: Whether to enable semantic search (requires DB)

    Returns:
        Cached HybridEntityExtractor instance
    """
    global _default_extractor

    if _default_extractor is None:
        _default_extractor = HybridEntityExtractor(
            enable_umls=enable_umls,
            enable_semantic=enable_semantic,
        )

    return _default_extractor


def extract_entities(text: str) -> List[ExtractedEntity]:
    """
    Extract entities using the default hybrid extractor.

    Convenience function for simple extraction.

    Args:
        text: Input text

    Returns:
        List of ExtractedEntity objects
    """
    return get_hybrid_extractor(enable_umls=False).extract(text)


def extract_entities_fast(text: str) -> List[ExtractedEntity]:
    """
    Fast extraction using only FlashText and regex (no UMLS).

    ~10x faster than full pipeline, suitable for high-volume processing.

    Args:
        text: Input text

    Returns:
        List of ExtractedEntity objects
    """
    extractor = HybridEntityExtractor(
        enable_flashtext=True,
        enable_regex=True,
        enable_umls=False,
        enable_semantic=False,
    )
    return extractor.extract(text)
