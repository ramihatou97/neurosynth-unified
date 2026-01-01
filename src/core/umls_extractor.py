"""
UMLS Entity Extraction using SciSpacy

Extracts UMLS Concept Unique Identifiers (CUIs) from medical text
using SciSpacy's entity linking pipeline. Complements the regex-based
NeuroExpertTextExtractor with standardized medical ontology concepts.

Requirements:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
"""

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Set, Optional, Iterator, Any, Tuple

from .tui_weights import get_tui_weight, get_tui_name

logger = logging.getLogger(__name__)

# Module-level extraction cache (LRU with max 2000 entries)
# Key: MD5 hash of text, Value: tuple of UMLSEntity objects
_extraction_cache: dict = {}
_CACHE_MAX_SIZE = 2000


def _get_cache_key(text: str) -> str:
    """Generate cache key from text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@dataclass
class UMLSEntity:
    """
    Represents a UMLS concept extracted from text.

    Attributes:
        cui: Concept Unique Identifier (e.g., "C0027051")
        name: Surface form / preferred name
        score: Linking confidence score (0.0-1.0)
        tui: Semantic Type ID (e.g., "T047")
        semantic_type: Human-readable semantic type name
        weight: TUI-based importance weight for neurosurgical domain
        start_char: Start character offset in source text
        end_char: End character offset in source text
    """
    cui: str
    name: str
    score: float
    tui: str
    semantic_type: str
    weight: float
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cui": self.cui,
            "name": self.name,
            "score": self.score,
            "tui": self.tui,
            "semantic_type": self.semantic_type,
            "weight": self.weight,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UMLSEntity":
        """Create from dictionary."""
        return cls(**data)


class UMLSExtractor:
    """
    SciSpacy-based UMLS CUI extraction.

    Uses en_core_sci_lg model with scispacy_linker for entity linking
    to the UMLS Metathesaurus. Provides both single-text and batch
    extraction methods.

    Example:
        extractor = UMLSExtractor()
        entities = extractor.extract("Patient has glioblastoma multiforme")
        # Returns list of UMLSEntity with CUIs like C0017636 (Glioblastoma)

        cuis = extractor.get_cui_set("frontal lobe tumor")
        # Returns {"C0016733", "C0027651", ...}
    """

    def __init__(
        self,
        model: str = "en_core_sci_lg",
        linker_name: str = "umls",
        threshold: float = 0.80,
        resolve_abbreviations: bool = True,
        max_entities_per_text: int = 100,
    ):
        """
        Initialize the UMLS extractor.

        Args:
            model: SciSpacy model name (en_core_sci_sm, en_core_sci_md, en_core_sci_lg)
            linker_name: Entity linker to use ("umls", "mesh", "rxnorm", "go", "hpo")
            threshold: Minimum confidence score for entity linking (0.0-1.0)
            resolve_abbreviations: Whether to resolve abbreviations before linking
            max_entities_per_text: Maximum entities to extract per text
        """
        self.model_name = model
        self.linker_name = linker_name
        self.threshold = threshold
        self.resolve_abbreviations = resolve_abbreviations
        self.max_entities_per_text = max_entities_per_text

        self._nlp = None
        self._linker = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize spaCy model and linker (expensive operation)."""
        if self._initialized:
            return

        try:
            import spacy
            from scispacy.linking import EntityLinker  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SciSpacy is required for UMLS extraction. Install with:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz"
            ) from e

        logger.info(f"Loading SciSpacy model: {self.model_name}")
        try:
            self._nlp = spacy.load(self.model_name)
        except OSError:
            raise OSError(
                f"SciSpacy model '{self.model_name}' not found. Install with:\n"
                f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{self.model_name}-0.5.1.tar.gz"
            )

        # Add abbreviation detector if enabled
        if self.resolve_abbreviations:
            if "abbreviation_detector" not in self._nlp.pipe_names:
                try:
                    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
                    self._nlp.add_pipe("abbreviation_detector")
                    logger.debug("Added abbreviation detector")
                except Exception as e:
                    logger.warning(f"Could not add abbreviation detector: {e}")

        # Add entity linker
        if "scispacy_linker" not in self._nlp.pipe_names:
            logger.info(f"Adding entity linker: {self.linker_name}")
            self._nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": self.resolve_abbreviations,
                    "linker_name": self.linker_name,
                    "threshold": self.threshold,
                }
            )

        self._linker = self._nlp.get_pipe("scispacy_linker")
        self._initialized = True
        logger.info(f"UMLS extractor initialized with {self.model_name}")

    @property
    def is_initialized(self) -> bool:
        """Check if the model is loaded."""
        return self._initialized

    def extract(self, text: str, use_cache: bool = True) -> List[UMLSEntity]:
        """
        Extract UMLS entities from text with optional caching.

        Args:
            text: Input text to analyze
            use_cache: Whether to use extraction cache (default True)

        Returns:
            List of UMLSEntity objects with CUIs, scores, and TUI weights
        """
        global _extraction_cache

        if not text or not text.strip():
            return []

        # Check cache first (2x speedup for repeated sections)
        cache_key = _get_cache_key(text) if use_cache else None
        if use_cache and cache_key in _extraction_cache:
            return list(_extraction_cache[cache_key])

        self._lazy_init()

        try:
            doc = self._nlp(text)
            entities = self._extract_from_doc(doc)

            # Store in cache (with LRU eviction)
            if use_cache and cache_key:
                if len(_extraction_cache) >= _CACHE_MAX_SIZE:
                    # Simple eviction: remove oldest entry
                    oldest_key = next(iter(_extraction_cache))
                    del _extraction_cache[oldest_key]
                _extraction_cache[cache_key] = tuple(entities)

            return entities
        except Exception as e:
            logger.error(f"UMLS extraction failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3.2: ADAPTIVE TUI-BASED CONFIDENCE THRESHOLDS
    # Different semantic types require different confidence levels
    # Higher thresholds for types prone to false positives (drugs, chemicals)
    # Lower thresholds for high-value types (anatomy, diseases, procedures)
    # ─────────────────────────────────────────────────────────────────────────

    TUI_THRESHOLDS = {
        # Anatomy - lower threshold, high value for neurosurgery
        "T023": 0.70,  # Body Part, Organ, or Organ Component
        "T024": 0.70,  # Tissue
        "T025": 0.70,  # Cell
        "T029": 0.70,  # Body Location or Region
        "T030": 0.70,  # Body Space or Junction

        # Diseases/Pathology - medium-low threshold
        "T047": 0.72,  # Disease or Syndrome
        "T048": 0.72,  # Mental or Behavioral Dysfunction
        "T049": 0.72,  # Cell or Molecular Dysfunction
        "T191": 0.72,  # Neoplastic Process

        # Procedures - standard threshold
        "T059": 0.78,  # Laboratory Procedure
        "T060": 0.78,  # Diagnostic Procedure
        "T061": 0.78,  # Therapeutic or Preventive Procedure

        # Findings/Signs - standard threshold
        "T033": 0.78,  # Finding
        "T034": 0.78,  # Laboratory or Test Result
        "T184": 0.78,  # Sign or Symptom

        # Drugs/Chemicals - higher threshold (prone to false positives)
        "T109": 0.85,  # Organic Chemical
        "T116": 0.85,  # Amino Acid, Peptide, or Protein
        "T121": 0.85,  # Pharmacologic Substance
        "T126": 0.85,  # Enzyme
        "T127": 0.85,  # Vitamin
        "T129": 0.85,  # Immunologic Factor
        "T131": 0.85,  # Hazardous or Poisonous Substance

        # Medical devices/instruments - standard
        "T074": 0.80,  # Medical Device
        "T075": 0.80,  # Research Device
    }

    def _get_tui_threshold(self, tui: str) -> float:
        """Get confidence threshold for a specific TUI."""
        return self.TUI_THRESHOLDS.get(tui, self.threshold)

    def _extract_from_doc(self, doc: Any) -> List[UMLSEntity]:
        """Extract entities from a processed spaCy Doc with adaptive thresholds."""
        entities = []
        seen_cuis: Set[str] = set()

        for ent in doc.ents:
            # Get linked UMLS concepts
            if not hasattr(ent, "_") or not hasattr(ent._, "kb_ents"):
                continue

            kb_ents = ent._.kb_ents
            if not kb_ents:
                continue

            # Process top linked concepts (usually just take the best one)
            for cui, score in kb_ents[:3]:  # Top 3 candidates
                # Quick filter: if below minimum possible threshold, skip
                if score < 0.70:
                    continue

                if cui in seen_cuis:
                    continue

                # Get concept info from knowledge base
                kb = self._linker.kb
                if cui not in kb.cui_to_entity:
                    continue

                concept = kb.cui_to_entity[cui]

                # Get TUI (semantic type) first for adaptive threshold
                tuis = concept.types if hasattr(concept, "types") else []
                tui = tuis[0] if tuis else "T000"

                # Phase 3.2: Apply TUI-specific threshold
                tui_threshold = self._get_tui_threshold(tui)
                if score < tui_threshold:
                    continue

                seen_cuis.add(cui)

                # Get canonical name
                name = concept.canonical_name if hasattr(concept, "canonical_name") else ent.text

                # Calculate weight based on TUI
                weight = get_tui_weight(tui)
                semantic_type = get_tui_name(tui)

                entity = UMLSEntity(
                    cui=cui,
                    name=name,
                    score=float(score),
                    tui=tui,
                    semantic_type=semantic_type,
                    weight=weight,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
                entities.append(entity)

                if len(entities) >= self.max_entities_per_text:
                    break

            if len(entities) >= self.max_entities_per_text:
                break

        # Sort by weight (importance) then by score
        entities.sort(key=lambda e: (e.weight, e.score), reverse=True)

        return entities

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        n_process: int = 1,
    ) -> List[List[UMLSEntity]]:
        """
        Batch extraction using spaCy's nlp.pipe() for efficiency.

        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            n_process: Number of processes (1 = no multiprocessing)

        Returns:
            List of entity lists, one per input text
        """
        if not texts:
            return []

        self._lazy_init()

        results: List[List[UMLSEntity]] = []

        try:
            # Use nlp.pipe for efficient batch processing
            docs = self._nlp.pipe(
                texts,
                batch_size=batch_size,
                n_process=n_process,
            )

            for doc in docs:
                entities = self._extract_from_doc(doc)
                results.append(entities)

        except Exception as e:
            logger.error(f"Batch UMLS extraction failed: {e}")
            # Return empty lists for all texts on failure
            return [[] for _ in texts]

        return results

    def get_cui_set(self, text: str) -> Set[str]:
        """
        Get just the CUI strings from text.

        Convenience method for linking comparison where only
        CUI identifiers are needed.

        Args:
            text: Input text

        Returns:
            Set of CUI strings (e.g., {"C0027051", "C0016733"})
        """
        entities = self.extract(text)
        return {e.cui for e in entities}

    def get_weighted_cuis(self, text: str) -> List[tuple]:
        """
        Get CUIs with their weights for scoring.

        Args:
            text: Input text

        Returns:
            List of (cui, weight) tuples sorted by weight descending
        """
        entities = self.extract(text)
        return [(e.cui, e.weight) for e in entities]

    def cui_jaccard_similarity(
        self,
        cuis1: Set[str],
        cuis2: Set[str],
    ) -> float:
        """
        Calculate Jaccard similarity between two CUI sets.

        Used in tri-pass linker for UMLS-based matching.

        Args:
            cuis1: First set of CUIs
            cuis2: Second set of CUIs

        Returns:
            Jaccard coefficient (0.0-1.0)
        """
        if not cuis1 or not cuis2:
            return 0.0

        intersection = len(cuis1 & cuis2)
        union = len(cuis1 | cuis2)

        return intersection / union if union > 0 else 0.0

    def weighted_cui_overlap(
        self,
        entities1: List[UMLSEntity],
        entities2: List[UMLSEntity],
    ) -> float:
        """
        Calculate weighted overlap score between entity lists.

        Weights shared CUIs by their TUI importance for
        neurosurgical domain relevance.

        Args:
            entities1: First entity list
            entities2: Second entity list

        Returns:
            Weighted overlap score (0.0-1.0)
        """
        if not entities1 or not entities2:
            return 0.0

        # Build CUI -> weight maps
        weights1 = {e.cui: e.weight for e in entities1}
        weights2 = {e.cui: e.weight for e in entities2}

        # Find shared CUIs
        shared_cuis = set(weights1.keys()) & set(weights2.keys())

        if not shared_cuis:
            return 0.0

        # Calculate weighted overlap
        shared_weight = sum(
            max(weights1[cui], weights2[cui])
            for cui in shared_cuis
        )

        total_weight = sum(weights1.values()) + sum(weights2.values())

        return (2 * shared_weight) / total_weight if total_weight > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

_extractor_cache: dict = {}


def get_default_extractor(
    model: str = "en_core_sci_lg",
    threshold: float = 0.80
) -> UMLSExtractor:
    """
    Get or create a cached UMLS extractor singleton.

    Caches extractors by (model, threshold) to reuse expensive spaCy models
    while supporting different configurations. Saves ~1.2GB RAM per reused model.

    Args:
        model: SciSpacy model name
        threshold: Confidence threshold for entity linking

    Returns:
        Cached UMLSExtractor instance
    """
    global _extractor_cache
    cache_key = (model, threshold)

    if cache_key not in _extractor_cache:
        logger.info(f"Creating new UMLS extractor: model={model}, threshold={threshold}")
        _extractor_cache[cache_key] = UMLSExtractor(model=model, threshold=threshold)
    else:
        logger.debug(f"Reusing cached UMLS extractor: model={model}, threshold={threshold}")

    return _extractor_cache[cache_key]


def extract_umls_entities(text: str) -> List[UMLSEntity]:
    """
    Extract UMLS entities using the default extractor.

    Convenience function for simple extraction without
    managing extractor instances.

    Args:
        text: Input text

    Returns:
        List of UMLSEntity objects
    """
    return get_default_extractor().extract(text)


def get_cuis(text: str) -> Set[str]:
    """
    Get CUI set from text using the default extractor.

    Args:
        text: Input text

    Returns:
        Set of CUI strings
    """
    return get_default_extractor().get_cui_set(text)
