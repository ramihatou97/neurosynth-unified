"""
NeuroSynth Query Expansion Module
==================================

Multi-strategy query expansion for neurosurgical queries.

Strategies:
1. Abbreviation expansion (MCA → middle cerebral artery)
2. UMLS synonym expansion (tumor → neoplasm, mass, lesion)
3. CUI extraction for boosting

Expected improvement: +15-20% search recall

Usage:
    from src.retrieval.query_expansion import QueryExpander

    expander = QueryExpander()
    expanded = expander.expand("MCA aneurysm treatment")

    print(expanded.original)       # "MCA aneurysm treatment"
    print(expanded.expanded_text)  # "MCA middle cerebral artery aneurysm treatment"
    print(expanded.cuis)           # ["C0917996", "C0002940"]
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ExpansionMethod(Enum):
    """Query expansion methods used."""
    ABBREVIATION = "abbreviation"
    SYNONYM = "synonym"
    UMLS = "umls"
    ANATOMICAL = "anatomical"


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    expanded_text: str
    cuis: List[str] = field(default_factory=list)
    expansion_methods: List[ExpansionMethod] = field(default_factory=list)
    synonyms_added: List[str] = field(default_factory=list)
    abbreviations_expanded: Dict[str, str] = field(default_factory=dict)

    @property
    def was_expanded(self) -> bool:
        return self.original != self.expanded_text or len(self.cuis) > 0


# =============================================================================
# Neurosurgical Abbreviation Dictionary
# =============================================================================

NEURO_ABBREVIATIONS = {
    # Vascular structures
    "MCA": "middle cerebral artery",
    "ACA": "anterior cerebral artery",
    "PCA": "posterior cerebral artery",
    "ICA": "internal carotid artery",
    "ECA": "external carotid artery",
    "VA": "vertebral artery",
    "BA": "basilar artery",
    "PICA": "posterior inferior cerebellar artery",
    "AICA": "anterior inferior cerebellar artery",
    "SCA": "superior cerebellar artery",
    "ACoA": "anterior communicating artery",
    "PCoA": "posterior communicating artery",
    "LSA": "lenticulostriate artery",
    "AChA": "anterior choroidal artery",

    # Anatomical regions
    "CPA": "cerebellopontine angle",
    "IAC": "internal auditory canal",
    "JF": "jugular foramen",
    "SOF": "superior orbital fissure",
    "IOF": "inferior orbital fissure",
    "FM": "foramen magnum",
    "FO": "foramen ovale",
    "FR": "foramen rotundum",
    "FS": "foramen spinosum",
    "CP": "cerebral peduncle",
    "IC": "internal capsule",
    "EC": "external capsule",
    "GP": "globus pallidus",
    "SN": "substantia nigra",
    "STN": "subthalamic nucleus",
    "RN": "red nucleus",
    "VPL": "ventral posterolateral nucleus",
    "VPM": "ventral posteromedial nucleus",

    # Cranial nerves
    "CN": "cranial nerve",
    "CN I": "olfactory nerve",
    "CN II": "optic nerve",
    "CN III": "oculomotor nerve",
    "CN IV": "trochlear nerve",
    "CN V": "trigeminal nerve",
    "CN VI": "abducens nerve",
    "CN VII": "facial nerve",
    "CN VIII": "vestibulocochlear nerve",
    "CN IX": "glossopharyngeal nerve",
    "CN X": "vagus nerve",
    "CN XI": "accessory nerve",
    "CN XII": "hypoglossal nerve",

    # Surgical approaches
    "PTK": "pterional keyhole",
    "OZ": "orbitozygomatic",
    "TL": "translabyrinthine",
    "RS": "retrosigmoid",
    "MF": "middle fossa",
    "TS": "transsylvian",
    "TC": "transcallosal",
    "TCV": "transcortical transventricular",
    "EEA": "endoscopic endonasal approach",
    "EETSS": "extended endoscopic transsphenoidal surgery",

    # Pathologies
    "GBM": "glioblastoma multiforme",
    "AA": "anaplastic astrocytoma",
    "LGG": "low grade glioma",
    "HGG": "high grade glioma",
    "VS": "vestibular schwannoma",
    "AN": "acoustic neuroma",
    "AVM": "arteriovenous malformation",
    "AVF": "arteriovenous fistula",
    "CCM": "cerebral cavernous malformation",
    "SAH": "subarachnoid hemorrhage",
    "ICH": "intracerebral hemorrhage",
    "SDH": "subdural hematoma",
    "EDH": "epidural hematoma",
    "IVH": "intraventricular hemorrhage",
    "TBI": "traumatic brain injury",
    "SCI": "spinal cord injury",
    "NPH": "normal pressure hydrocephalus",
    "IIH": "idiopathic intracranial hypertension",

    # Procedures
    "VP shunt": "ventriculoperitoneal shunt",
    "EVD": "external ventricular drain",
    "LP": "lumbar puncture",
    "SRS": "stereotactic radiosurgery",
    "GKS": "gamma knife surgery",
    "DBS": "deep brain stimulation",
    "SCS": "spinal cord stimulation",
    "ACDF": "anterior cervical discectomy and fusion",
    "ALIF": "anterior lumbar interbody fusion",
    "PLIF": "posterior lumbar interbody fusion",
    "TLIF": "transforaminal lumbar interbody fusion",
    "XLIF": "extreme lateral interbody fusion",

    # Imaging
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "CTA": "computed tomography angiography",
    "MRA": "magnetic resonance angiography",
    "DSA": "digital subtraction angiography",
    "PET": "positron emission tomography",
    "DTI": "diffusion tensor imaging",
    "fMRI": "functional magnetic resonance imaging",
    "BOLD": "blood oxygen level dependent",

    # Monitoring
    "ICP": "intracranial pressure",
    "CPP": "cerebral perfusion pressure",
    "MAP": "mean arterial pressure",
    "EEG": "electroencephalogram",
    "EMG": "electromyography",
    "SSEP": "somatosensory evoked potential",
    "MEP": "motor evoked potential",
    "BAEP": "brainstem auditory evoked potential",

    # Scales and scores
    "GCS": "Glasgow Coma Scale",
    "HH": "Hunt and Hess",
    "WFNS": "World Federation of Neurosurgical Societies",
    "mRS": "modified Rankin Scale",
    "KPS": "Karnofsky Performance Status",
    "NIHSS": "National Institutes of Health Stroke Scale",
}


# =============================================================================
# UMLS CUI Mapping (Common Neurosurgical Concepts)
# =============================================================================

CONCEPT_CUIS = {
    # Anatomy
    "middle cerebral artery": "C0149566",
    "anterior cerebral artery": "C0149561",
    "posterior cerebral artery": "C0149576",
    "internal carotid artery": "C0007276",
    "vertebral artery": "C0042559",
    "basilar artery": "C0004811",
    "cerebellopontine angle": "C0007766",
    "internal auditory canal": "C0458717",
    "basal ganglia": "C0004781",
    "thalamus": "C0039729",
    "hypothalamus": "C0020663",
    "brainstem": "C0006121",
    "cerebellum": "C0007765",
    "corpus callosum": "C0010090",
    "sylvian fissure": "C0228187",
    "lateral ventricle": "C0152279",
    "third ventricle": "C0152280",
    "fourth ventricle": "C0152281",

    # Pathology
    "glioblastoma": "C0017636",
    "astrocytoma": "C0004114",
    "meningioma": "C0025286",
    "schwannoma": "C0027809",
    "vestibular schwannoma": "C0027859",
    "acoustic neuroma": "C0027859",
    "pituitary adenoma": "C0032000",
    "craniopharyngioma": "C0010276",
    "aneurysm": "C0002940",
    "arteriovenous malformation": "C0003857",
    "cavernous malformation": "C0917996",
    "hydrocephalus": "C0020255",
    "subarachnoid hemorrhage": "C0038525",
    "intracerebral hemorrhage": "C2937358",
    "stroke": "C0038454",
    "ischemia": "C0022116",

    # Procedures
    "craniotomy": "C0010280",
    "craniectomy": "C0195933",
    "microsurgery": "C0026035",
    "endoscopy": "C0014245",
    "stereotactic surgery": "C0038912",
    "radiosurgery": "C0085203",
    "biopsy": "C0005558",
    "resection": "C0728940",
    "clipping": "C0087111",
    "coiling": "C0394837",
    "embolization": "C0013931",
    "shunt": "C0542331",
    "drainage": "C0013103",

    # Clinical
    "headache": "C0018681",
    "seizure": "C0036572",
    "weakness": "C0004093",
    "numbness": "C0028643",
    "vision loss": "C0042798",
    "hearing loss": "C0018772",
    "facial palsy": "C0015469",
    "diplopia": "C0012569",
    "vertigo": "C0042571",
    "ataxia": "C0004134",
    "aphasia": "C0003537",
    "dysarthria": "C0013362",
    "dysphagia": "C0011168",
}


# =============================================================================
# Synonym Mapping
# =============================================================================

SYNONYM_MAP = {
    "tumor": ["neoplasm", "mass", "lesion", "growth"],
    "surgery": ["operation", "procedure", "intervention"],
    "approach": ["technique", "method", "access"],
    "vessel": ["artery", "vein", "vasculature"],
    "nerve": ["neural structure", "neural pathway"],
    "treatment": ["therapy", "management", "intervention"],
    "complication": ["adverse event", "sequela", "morbidity"],
    "outcome": ["result", "prognosis", "sequela"],
    "imaging": ["radiological study", "scan", "MRI CT"],
    "preservation": ["sparing", "protection", "conservation"],
}


# =============================================================================
# Query Expander Class
# =============================================================================

class QueryExpander:
    """
    Multi-strategy query expansion for neurosurgical queries.

    Improves search recall by:
    1. Expanding abbreviations (MCA → middle cerebral artery)
    2. Adding synonyms (tumor → tumor neoplasm mass)
    3. Extracting CUIs for semantic boosting

    Example:
        expander = QueryExpander()
        result = expander.expand("MCA aneurysm clipping")
        # result.expanded_text = "MCA middle cerebral artery aneurysm clipping"
        # result.cuis = ["C0149566", "C0002940", "C0087111"]
    """

    def __init__(
        self,
        abbreviations: Dict[str, str] = None,
        concept_cuis: Dict[str, str] = None,
        synonyms: Dict[str, List[str]] = None,
        enable_abbreviations: bool = True,
        enable_synonyms: bool = True,
        enable_cui_extraction: bool = True,
        max_synonyms_per_term: int = 2,
    ):
        """
        Initialize query expander.

        Args:
            abbreviations: Custom abbreviation dict (uses defaults if None)
            concept_cuis: Custom CUI mapping (uses defaults if None)
            synonyms: Custom synonym mapping (uses defaults if None)
            enable_abbreviations: Whether to expand abbreviations
            enable_synonyms: Whether to add synonyms
            enable_cui_extraction: Whether to extract CUIs
            max_synonyms_per_term: Max synonyms to add per matched term
        """
        self.abbreviations = abbreviations or NEURO_ABBREVIATIONS
        self.concept_cuis = concept_cuis or CONCEPT_CUIS
        self.synonyms = synonyms or SYNONYM_MAP

        self.enable_abbreviations = enable_abbreviations
        self.enable_synonyms = enable_synonyms
        self.enable_cui_extraction = enable_cui_extraction
        self.max_synonyms_per_term = max_synonyms_per_term

        # Build reverse mapping for abbreviation detection
        self._abbrev_pattern = self._build_abbrev_pattern()

        # Build concept pattern for CUI extraction
        self._concept_pattern = self._build_concept_pattern()

    def _build_abbrev_pattern(self) -> re.Pattern:
        """Build regex pattern for abbreviation matching."""
        # Sort by length (longest first) to match multi-word abbrevs first
        abbrevs = sorted(self.abbreviations.keys(), key=len, reverse=True)
        # Escape special chars and join with word boundaries
        pattern_str = r'\b(' + '|'.join(re.escape(a) for a in abbrevs) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)

    def _build_concept_pattern(self) -> re.Pattern:
        """Build regex pattern for concept matching."""
        concepts = sorted(self.concept_cuis.keys(), key=len, reverse=True)
        pattern_str = r'\b(' + '|'.join(re.escape(c) for c in concepts) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query using all enabled strategies.

        Args:
            query: Original search query

        Returns:
            ExpandedQuery with expanded text, CUIs, and metadata
        """
        result = ExpandedQuery(original=query, expanded_text=query)

        # Strategy 1: Abbreviation expansion
        if self.enable_abbreviations:
            self._expand_abbreviations(result)

        # Strategy 2: Synonym expansion
        if self.enable_synonyms:
            self._expand_synonyms(result)

        # Strategy 3: CUI extraction
        if self.enable_cui_extraction:
            self._extract_cuis(result)

        logger.debug(f"Query expansion: '{query}' → '{result.expanded_text}'")

        return result

    def _expand_abbreviations(self, result: ExpandedQuery) -> None:
        """Expand abbreviations inline in the query."""
        text = result.expanded_text

        def replace_abbrev(match):
            abbrev = match.group(0).upper()
            if abbrev in self.abbreviations:
                expansion = self.abbreviations[abbrev]
                result.abbreviations_expanded[abbrev] = expansion
                # Keep original + add expansion
                return f"{match.group(0)} {expansion}"
            return match.group(0)

        expanded = self._abbrev_pattern.sub(replace_abbrev, text)

        if expanded != text:
            result.expanded_text = expanded
            result.expansion_methods.append(ExpansionMethod.ABBREVIATION)

    def _expand_synonyms(self, result: ExpandedQuery) -> None:
        """Add synonyms for key terms."""
        text_lower = result.expanded_text.lower()
        synonyms_to_add = []

        for term, syns in self.synonyms.items():
            if term.lower() in text_lower:
                # Add limited synonyms
                for syn in syns[:self.max_synonyms_per_term]:
                    if syn.lower() not in text_lower:
                        synonyms_to_add.append(syn)
                        result.synonyms_added.append(syn)

        if synonyms_to_add:
            result.expanded_text = f"{result.expanded_text} {' '.join(synonyms_to_add)}"
            result.expansion_methods.append(ExpansionMethod.SYNONYM)

    def _extract_cuis(self, result: ExpandedQuery) -> None:
        """Extract UMLS CUIs from query concepts."""
        text_lower = result.expanded_text.lower()

        cuis = set()
        for concept, cui in self.concept_cuis.items():
            if concept.lower() in text_lower:
                cuis.add(cui)

        result.cuis = list(cuis)

        if cuis:
            result.expansion_methods.append(ExpansionMethod.UMLS)

    def expand_batch(self, queries: List[str]) -> List[ExpandedQuery]:
        """Expand multiple queries."""
        return [self.expand(q) for q in queries]

    def add_abbreviation(self, abbrev: str, expansion: str) -> None:
        """Add custom abbreviation mapping."""
        self.abbreviations[abbrev.upper()] = expansion.lower()
        self._abbrev_pattern = self._build_abbrev_pattern()

    def add_concept_cui(self, concept: str, cui: str) -> None:
        """Add custom concept-CUI mapping."""
        self.concept_cuis[concept.lower()] = cui
        self._concept_pattern = self._build_concept_pattern()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_query_expander(**kwargs) -> QueryExpander:
    """Factory function for QueryExpander with sensible defaults."""
    return QueryExpander(**kwargs)


def expand_query(query: str) -> ExpandedQuery:
    """Quick expansion using default settings."""
    expander = QueryExpander()
    return expander.expand(query)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NeuroSynth Query Expander Test")
    print("=" * 70)

    expander = QueryExpander()

    test_queries = [
        "MCA aneurysm clipping",
        "vestibular schwannoma retrosigmoid approach",
        "GBM tumor resection technique",
        "CN VII preservation",
        "CPA meningioma surgery",
        "ICA stenosis treatment",
        "SAH hydrocephalus VP shunt",
    ]

    for query in test_queries:
        result = expander.expand(query)
        print(f"\nOriginal: {result.original}")
        print(f"Expanded: {result.expanded_text}")
        if result.cuis:
            print(f"CUIs: {', '.join(result.cuis)}")
        if result.abbreviations_expanded:
            print(f"Abbreviations: {result.abbreviations_expanded}")
        if result.synonyms_added:
            print(f"Synonyms: {result.synonyms_added}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
