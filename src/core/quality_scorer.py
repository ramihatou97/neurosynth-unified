"""
NeuroSynth v2.2 - Chunk Quality Scoring Module
==============================================

Computes readability, coherence, completeness, and type-specific scores
for semantic chunks to improve synthesis output quality.

v2.2 Enhancements:
- 4-dimension scoring (readability, coherence, completeness, type-specific)
- Orphan detection with penalty scoring
- Type-specific requirement validation
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.shared.models import SemanticChunk, ChunkType

logger = logging.getLogger(__name__)


# =============================================================================
# ORPHAN DETECTION PATTERNS (v2.2)
# =============================================================================

ORPHAN_PATTERNS = [
    re.compile(r"^Step\s+[2-9]", re.I),
    re.compile(r"^[2-9]\.\s+(?=[A-Z])", re.I),
    re.compile(r"^(Then|Next|Subsequently|After\s+this)\b", re.I),
    re.compile(r"^(Second|Third|Fourth|Fifth|Sixth)\b", re.I),
]


# =============================================================================
# CONTENT ELEMENT PATTERNS (v2.2)
# =============================================================================

ELEMENT_PATTERNS = {
    'action': [
        re.compile(r'\b(incise|dissect|retract|expose|coagulate|clip|remove|resect)\b', re.I),
        re.compile(r'\b(open|close|cut|suture|place|insert|advance)\b', re.I),
    ],
    'anatomy': [
        re.compile(r'\b(nerve|artery|vein|muscle|bone|dura|cortex|cistern)\b', re.I),
        re.compile(r'\b(lateral|medial|superior|inferior|anterior|posterior)\b', re.I),
    ],
    'rationale': [
        re.compile(r'\b(to avoid|to prevent|in order to|this allows|this prevents)\b', re.I),
        re.compile(r'\b(because|since|therefore|thus)\b', re.I),
    ],
    'instrument': [
        re.compile(r'\b(microscope|bipolar|suction|dissector|retractor|forceps|drill)\b', re.I),
        re.compile(r'\b(Penfield|Kerrison|Bovie|Malis)\b', re.I),
    ],
    'spatial': [
        re.compile(r'\b(lateral to|medial to|above|below|anterior to|posterior to)\b', re.I),
        re.compile(r'\b(adjacent to|superficial to|deep to|parallel to)\b', re.I),
    ],
    'structure': [
        re.compile(r'\b(origin|insertion|course|termination|branches)\b', re.I),
        re.compile(r'\b(forms|gives rise to|divides into|anastomoses)\b', re.I),
    ],
    'function': [
        re.compile(r'\b(innervates|supplies|drains|controls|modulates)\b', re.I),
        re.compile(r'\b(function|role|purpose)\b', re.I),
    ],
    'pathology': [
        re.compile(r'\b(tumor|lesion|mass|cyst|hemorrhage|infarct)\b', re.I),
        re.compile(r'\b(grade|WHO|malignant|benign|aggressive)\b', re.I),
    ],
    'clinical': [
        re.compile(r'\b(symptom|sign|deficit|presentation|outcome)\b', re.I),
        re.compile(r'\b(patient|prognosis|survival|recurrence)\b', re.I),
    ],
    'grading': [
        re.compile(r'\b(Grade\s+[IVX\d]+|WHO\s+[IVX\d]+)\b', re.I),
        re.compile(r'\b(Spetzler|Hunt.Hess|Fisher|WFNS)\b', re.I),
    ],
}


# =============================================================================
# TYPE REQUIREMENTS (v2.2)
# =============================================================================

TYPE_REQUIREMENTS: Dict[str, List[str]] = {
    'procedure': ['action', 'anatomy', 'rationale', 'instrument'],
    'anatomy': ['anatomy', 'spatial', 'structure', 'function'],
    'pathology': ['pathology', 'clinical', 'grading'],
    'clinical': ['clinical', 'pathology'],
    'case': ['clinical', 'pathology'],
    'general': [],
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class QualityConfig:
    """Configuration for chunk quality scoring (v2.2)."""
    # Dimension weights (now 4 dimensions)
    readability_weight: float = 0.20
    coherence_weight: float = 0.25
    completeness_weight: float = 0.35
    type_specific_weight: float = 0.20

    # Orphan penalty (v2.2)
    orphan_penalty: float = -0.20

    # Readability parameters
    optimal_sentence_length_min: int = 15
    optimal_sentence_length_max: int = 25
    max_entity_density: float = 15.0  # Percent

    # Coherence parameters
    min_entity_continuity: float = 0.3
    max_specialty_tags: int = 2

    # Completeness parameters
    min_quality_threshold: float = 0.5


class ChunkQualityScorer:
    """
    Computes quality scores for semantic chunks.

    Quality scores:
    - readability_score: How clear and readable the chunk is (0.0-1.0)
    - coherence_score: How well sentences connect logically (0.0-1.0)
    - completeness_score: Whether chunk is self-contained (0.0-1.0)
    """

    # Sentence splitting pattern (preserves medical abbreviations)
    SENTENCE_PATTERN = re.compile(
        r'(?<!\bDr)(?<!\bFig)(?<!\bNo)(?<!\bet\sal)(?<!\bvs)'
        r'(?<!\bVol)(?<!\bpp)(?<!\bCh)(?<!\bSec)'
        r'(?<=[.!?])\s+(?=[A-Z])'
    )

    # Medical abbreviations (common in neurosurgery)
    ABBREVIATION_PATTERN = re.compile(r'\b([A-Z]{2,6})\b')

    # Discourse markers indicating logical flow
    FLOW_MARKERS = frozenset([
        "however", "therefore", "thus", "furthermore", "additionally",
        "consequently", "in contrast", "similarly", "as a result",
        "moreover", "nevertheless", "specifically", "importantly"
    ])

    # Explanatory phrases indicating clarity
    EXPLANATORY_MARKERS = frozenset([
        "meaning", "known as", "defined as", "refers to", "i.e.",
        "specifically", "that is", "in other words", "namely"
    ])

    # Dangling reference patterns (incomplete context)
    DANGLING_PATTERNS = [
        re.compile(r"^(This|These|That|Those|It|They)\s+(?!is\s+a|are\s+the|includes?|refers?)", re.IGNORECASE),
        re.compile(r"^(As\s+mentioned|Continuing|Following\s+this)", re.IGNORECASE),
        re.compile(r"^(The\s+above|The\s+following|The\s+latter)", re.IGNORECASE),
    ]

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving medical abbreviations."""
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def compute_readability(self, chunk: "SemanticChunk") -> float:
        """
        Compute readability score (0.0 - 1.0).

        Factors:
        - Sentence length distribution (optimal: 15-25 words for medical text)
        - Abbreviation clarity (defined vs undefined)
        - Technical term density
        - Presence of explanatory phrases
        """
        content = chunk.content
        if not content:
            return 0.0

        sentences = self._split_sentences(content)
        if not sentences:
            return 0.5

        # Factor 1: Sentence length (optimal: 15-25 words for medical)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        min_opt = self.config.optimal_sentence_length_min
        max_opt = self.config.optimal_sentence_length_max

        if min_opt <= avg_length <= max_opt:
            length_score = 1.0
        elif avg_length < min_opt:
            length_score = max(0.3, avg_length / min_opt)
        else:
            length_score = max(0.3, 1.0 - (avg_length - max_opt) / 30)

        # Factor 2: Abbreviation clarity (defined vs undefined)
        abbreviations = self.ABBREVIATION_PATTERN.findall(content)
        if abbreviations:
            # Check if abbreviations are defined in text
            entity_names = set(getattr(chunk, 'entity_names', []) or [])
            defined_count = sum(
                1 for a in abbreviations
                if a in entity_names or re.search(rf'\({a}\)', content)
            )
            abbrev_score = defined_count / len(abbreviations)
        else:
            abbrev_score = 1.0

        # Factor 3: Technical term density (too high = hard to read)
        entities = getattr(chunk, 'entities', []) or []
        word_count = len(content.split())
        if word_count > 0:
            entity_density = (len(entities) / word_count) * 100
            max_density = self.config.max_entity_density
            if entity_density <= max_density:
                density_score = 1.0
            else:
                density_score = max(0.3, 1.0 - (entity_density - max_density) / 30)
        else:
            density_score = 0.5

        # Factor 4: Presence of explanatory phrases (bonus)
        content_lower = content.lower()
        explanation_count = sum(
            1 for marker in self.EXPLANATORY_MARKERS
            if marker in content_lower
        )
        explanation_bonus = min(explanation_count * 0.05, 0.15)

        # Weighted combination
        score = (
            length_score * 0.35 +
            abbrev_score * 0.30 +
            density_score * 0.25 +
            explanation_bonus
        )

        return min(1.0, max(0.0, score))

    def compute_coherence(self, chunk: "SemanticChunk") -> float:
        """
        Compute coherence score (0.0 - 1.0).

        Factors:
        - Entity continuity across sentences
        - Discourse markers presence (logical flow)
        - Specialty consistency
        - Topic drift detection
        """
        content = chunk.content
        if not content:
            return 0.0

        sentences = self._split_sentences(content)

        # Single sentence chunks are coherent by default
        if len(sentences) < 2:
            return 0.85

        entities = getattr(chunk, 'entities', []) or []
        entity_texts = set(
            e.text.lower() if hasattr(e, 'text') else str(e).lower()
            for e in entities
        )

        # Factor 1: Entity continuity across sentences
        sentence_entities: List[Set[str]] = []
        for sent in sentences:
            sent_lower = sent.lower()
            sent_ents = {e for e in entity_texts if e in sent_lower}
            sentence_entities.append(sent_ents)

        continuity_scores = []
        for i in range(1, len(sentence_entities)):
            prev_ents = sentence_entities[i - 1]
            curr_ents = sentence_entities[i]
            if prev_ents:
                overlap = len(prev_ents & curr_ents)
                continuity_scores.append(min(overlap / len(prev_ents), 1.0))
            else:
                continuity_scores.append(0.5)  # No entities to compare

        continuity_score = (
            sum(continuity_scores) / len(continuity_scores)
            if continuity_scores else 0.5
        )

        # Factor 2: Discourse marker presence (logical flow)
        content_lower = content.lower()
        flow_count = sum(
            1 for marker in self.FLOW_MARKERS
            if marker in content_lower
        )
        flow_score = min(flow_count * 0.15, 0.30)

        # Factor 3: Specialty consistency
        specialty_tags = getattr(chunk, 'specialty_tags', []) or []
        if hasattr(chunk, 'specialty') and chunk.specialty:
            if isinstance(chunk.specialty, dict):
                specialty_tags = list(chunk.specialty.keys())
            elif isinstance(chunk.specialty, (list, set)):
                specialty_tags = list(chunk.specialty)

        specialty_count = len(set(specialty_tags))
        max_specialties = self.config.max_specialty_tags
        if specialty_count <= max_specialties:
            specialty_score = 1.0
        else:
            specialty_score = max(0.5, 1.0 - (specialty_count - max_specialties) * 0.15)

        # Factor 4: Topic drift detection (first vs last sentence overlap)
        if len(sentence_entities) >= 3:
            first_ents = sentence_entities[0]
            last_ents = sentence_entities[-1]
            if first_ents:
                drift_overlap = len(first_ents & last_ents)
                drift_score = min(drift_overlap / len(first_ents), 1.0)
            else:
                drift_score = 0.7
        else:
            drift_score = 1.0

        # Weighted combination
        score = (
            continuity_score * 0.35 +
            specialty_score * 0.25 +
            drift_score * 0.25 +
            flow_score
        )

        return min(1.0, max(0.0, score))

    def compute_completeness(self, chunk: "SemanticChunk") -> float:
        """
        Compute completeness score (0.0 - 1.0).

        Factors:
        - Context independence (doesn't start with dangling references)
        - Type-specific content requirements
        - Sentence completeness (proper ending)
        """
        content = chunk.content
        if not content:
            return 0.0

        # Factor 1: Context independence (no dangling references)
        independence_score = 1.0
        for pattern in self.DANGLING_PATTERNS:
            if pattern.search(content):
                independence_score -= 0.2
        independence_score = max(0.0, independence_score)

        # Factor 2: Type-specific completeness
        chunk_type = getattr(chunk, 'chunk_type', None)
        type_name = chunk_type.value if hasattr(chunk_type, 'value') else str(chunk_type)
        entities = getattr(chunk, 'entities', []) or []
        content_lower = content.lower()

        type_score = self._compute_type_score(type_name, entities, content_lower)

        # Factor 3: Sentence completeness (proper ending)
        stripped = content.rstrip()
        ends_complete = stripped.endswith(('.', '!', '?', ')', '"', "'"))
        complete_score = 1.0 if ends_complete else 0.7

        # Factor 4: Minimum content length
        word_count = len(content.split())
        if word_count < 20:
            length_penalty = word_count / 20
        elif word_count > 300:
            length_penalty = max(0.8, 1.0 - (word_count - 300) / 500)
        else:
            length_penalty = 1.0

        # Weighted combination
        score = (
            independence_score * 0.30 +
            type_score * 0.40 +
            complete_score * 0.15 +
            length_penalty * 0.15
        )

        return min(1.0, max(0.0, score))

    def _compute_type_score(
        self,
        type_name: str,
        entities: list,
        content_lower: str
    ) -> float:
        """Compute type-specific completeness score."""

        # Get entity categories
        entity_categories = set()
        for e in entities:
            if hasattr(e, 'category'):
                entity_categories.add(e.category)
            elif isinstance(e, dict) and 'category' in e:
                entity_categories.add(e['category'])

        if type_name == "procedure":
            # Procedures should have: action + anatomy + rationale
            has_action = any("PROCEDURE" in c for c in entity_categories)
            has_anatomy = any("ANATOMY" in c for c in entity_categories)
            has_rationale = any(
                w in content_lower
                for w in ["to avoid", "to prevent", "allows", "ensures", "in order to"]
            )
            return (has_action * 0.4 + has_anatomy * 0.3 + has_rationale * 0.3)

        elif type_name == "anatomy":
            # Anatomy should have: structure + relationships
            has_structure = any("ANATOMY" in c for c in entity_categories)
            has_relationships = any(
                w in content_lower
                for w in ["adjacent", "lateral", "medial", "superior", "inferior",
                         "anterior", "posterior", "proximal", "distal"]
            )
            has_function = any(
                w in content_lower
                for w in ["supplies", "innervates", "drains", "connects", "receives"]
            )
            return (has_structure * 0.40 + has_relationships * 0.35 + has_function * 0.25)

        elif type_name == "pathology":
            # Pathology should have: condition + manifestation
            has_pathology = any("PATHOLOGY" in c or "DISEASE" in c for c in entity_categories)
            has_clinical = any(
                w in content_lower
                for w in ["present", "symptom", "sign", "deficit", "pain",
                         "weakness", "numbness", "dysfunction"]
            )
            return (has_pathology * 0.50 + has_clinical * 0.50)

        elif type_name == "clinical":
            # Clinical should have: presentation + context
            has_presentation = any(
                w in content_lower
                for w in ["patient", "present", "complaint", "history", "examination"]
            )
            has_context = any(
                w in content_lower
                for w in ["management", "treatment", "diagnosis", "prognosis"]
            )
            return (has_presentation * 0.50 + has_context * 0.50)

        else:
            # General/case/unknown - moderate baseline
            return 0.7

    def compute_type_specific(self, chunk: "SemanticChunk") -> float:
        """
        Compute type-specific score based on required elements (v2.2).

        Args:
            chunk: SemanticChunk to evaluate

        Returns:
            Score from 0.0 to 1.0
        """
        content = chunk.content if hasattr(chunk, 'content') else ''
        chunk_type = getattr(chunk, 'chunk_type', None)
        type_name = chunk_type.value if hasattr(chunk_type, 'value') else str(chunk_type)

        requirements = TYPE_REQUIREMENTS.get(type_name.lower(), [])

        if not requirements:
            return 0.8  # No specific requirements

        elements_found = self._detect_elements(content)

        met_count = sum(1 for req in requirements if req in elements_found)

        if len(requirements) > 0:
            ratio = met_count / len(requirements)
            return 0.4 + 0.6 * ratio

        return 0.8

    def _detect_elements(self, content: str) -> List[str]:
        """Detect which content elements are present (v2.2)."""
        found = []
        for element_type, patterns in ELEMENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(content):
                    found.append(element_type)
                    break
        return found

    def _is_orphan(self, content: str) -> bool:
        """Check if chunk starts with orphan indicators (v2.2)."""
        content_start = content[:100].strip()
        return any(p.search(content_start) for p in ORPHAN_PATTERNS)

    def score_chunk(self, chunk: "SemanticChunk") -> "SemanticChunk":
        """
        Apply all quality scores to a chunk (v2.2 - 4 dimensions + orphan penalty).

        Modifies chunk in place and returns it.
        """
        chunk.readability_score = self.compute_readability(chunk)
        chunk.coherence_score = self.compute_coherence(chunk)
        chunk.completeness_score = self.compute_completeness(chunk)

        # v2.2: Compute and store type-specific score
        type_specific_score = self.compute_type_specific(chunk)
        if hasattr(chunk, 'type_specific_score'):
            chunk.type_specific_score = type_specific_score

        # v2.2: Check for orphan and store status
        content = chunk.content if hasattr(chunk, 'content') else ''
        is_orphan = self._is_orphan(content)
        if hasattr(chunk, 'is_orphan'):
            chunk.is_orphan = is_orphan

        # Note: quality_score is now a computed property on SemanticChunk
        # that applies orphan penalty automatically

        return chunk

    def get_aggregate_score(
        self,
        chunk: "SemanticChunk",
        type_specific_score: Optional[float] = None
    ) -> float:
        """
        Get weighted aggregate quality score (v2.2 - 4 dimensions).
        """
        if type_specific_score is None:
            type_specific_score = self.compute_type_specific(chunk)

        return (
            chunk.readability_score * self.config.readability_weight +
            chunk.coherence_score * self.config.coherence_weight +
            chunk.completeness_score * self.config.completeness_weight +
            type_specific_score * self.config.type_specific_weight
        )

    def is_quality_acceptable(self, chunk: "SemanticChunk") -> bool:
        """
        Check if chunk meets minimum quality threshold (v2.2).
        """
        content = chunk.content if hasattr(chunk, 'content') else ''
        aggregate = self.get_aggregate_score(chunk)

        # Apply orphan penalty if applicable
        if self._is_orphan(content):
            aggregate += self.config.orphan_penalty

        return aggregate >= self.config.min_quality_threshold


# Singleton instance for convenience
_default_scorer: Optional[ChunkQualityScorer] = None


def get_quality_scorer(config: Optional[QualityConfig] = None) -> ChunkQualityScorer:
    """Get or create the default quality scorer instance."""
    global _default_scorer
    if _default_scorer is None or config is not None:
        _default_scorer = ChunkQualityScorer(config)
    return _default_scorer
