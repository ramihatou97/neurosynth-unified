"""
NeuroSynth Enhanced Context Adapter
====================================

Upgraded ContextAdapter aligned with:
- Extraction/Indexing pipeline (quality scores, type-specific chunking)
- Search/Retrieval layer (authority boost, type boost, CUI boost)

Fixes:
1. Composite quality score from 3 separate scores
2. Type-based section routing (chunk_type primary, keywords secondary)
3. Caption embedding passthrough for semantic figure resolution
4. CUI preservation for content validation
5. Improved authority weighting in combined score

Drop-in replacement for src/synthesis/engine.py ContextAdapter class.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# AUTHORITY SOURCE DEFINITIONS (matches existing)
# =============================================================================

class AuthoritySource(Enum):
    RHOTON = "rhoton"
    LAWTON = "lawton"
    SAMII = "samii"
    AL_MEFTY = "al-mefty"
    SEKHAR = "sekhar"
    SPETZLER = "spetzler"
    YASARGIL = "yasargil"
    YOUMANS = "youmans"
    SCHMIDEK = "schmidek"
    GREENBERG = "greenberg"
    WINN = "winn"
    QUINONES = "quinones"
    CONNOLLY = "connolly"
    BENZEL = "benzel"
    JOURNAL = "journal"
    GENERAL = "general"


AUTHORITY_SCORES = {
    AuthoritySource.RHOTON: 1.0,
    AuthoritySource.LAWTON: 1.0,
    AuthoritySource.SAMII: 1.0,
    AuthoritySource.AL_MEFTY: 1.0,
    AuthoritySource.SEKHAR: 0.95,
    AuthoritySource.SPETZLER: 0.95,
    AuthoritySource.YASARGIL: 0.95,
    AuthoritySource.YOUMANS: 0.9,
    AuthoritySource.SCHMIDEK: 0.9,
    AuthoritySource.GREENBERG: 0.85,
    AuthoritySource.WINN: 0.85,
    AuthoritySource.QUINONES: 0.85,
    AuthoritySource.CONNOLLY: 0.85,
    AuthoritySource.BENZEL: 0.8,
    AuthoritySource.JOURNAL: 0.75,
    AuthoritySource.GENERAL: 0.7,
}


# =============================================================================
# TEMPLATE DEFINITIONS (import from main engine or define here)
# =============================================================================

class TemplateType(str, Enum):
    """Must match src.synthesis.engine.TemplateType values."""
    PROCEDURAL = "PROCEDURAL"
    DISORDER = "DISORDER"
    ANATOMY = "ANATOMY"
    ENCYCLOPEDIA = "ENCYCLOPEDIA"


TEMPLATE_SECTIONS = {
    TemplateType.PROCEDURAL: [
        ("Indications", 2),
        ("Preoperative Considerations", 2),
        ("Patient Positioning", 2),
        ("Surgical Approach", 2),
        ("Step-by-Step Technique", 2),
        ("Closure", 2),
        ("Complications and Avoidance", 2),
        ("Outcomes", 2),
    ],
    TemplateType.DISORDER: [
        ("Overview", 2),
        ("Epidemiology", 2),
        ("Pathophysiology", 2),
        ("Clinical Presentation", 2),
        ("Diagnostic Workup", 2),
        ("Imaging Findings", 2),
        ("Differential Diagnosis", 2),
        ("Management", 2),
        ("Prognosis", 2),
    ],
    TemplateType.ANATOMY: [
        ("Boundaries and Relationships", 2),
        ("Surface Anatomy", 2),
        ("Osseous Anatomy", 2),
        ("Dural Relationships", 2),
        ("Arterial Supply", 2),
        ("Venous Drainage", 2),
        ("Neural Structures", 2),
        ("Surgical Corridors", 2),
        ("Key Measurements", 2),
    ],
    TemplateType.ENCYCLOPEDIA: [
        ("Definition and Overview", 2),
        ("Historical Perspective", 2),
        ("Anatomy", 2),
        ("Pathology", 2),
        ("Clinical Features", 2),
        ("Diagnostic Approach", 2),
        ("Treatment Options", 2),
        ("Surgical Technique", 2),
        ("Outcomes and Prognosis", 2),
        ("Future Directions", 2),
    ],
}

TEMPLATE_REQUIREMENTS = {
    TemplateType.PROCEDURAL: {"min_words": 3000, "min_figures": 8},
    TemplateType.DISORDER: {"min_words": 5000, "min_figures": 10},
    TemplateType.ANATOMY: {"min_words": 4000, "min_figures": 15},
    TemplateType.ENCYCLOPEDIA: {"min_words": 15000, "min_figures": 20},
}


# =============================================================================
# CHUNK TYPE TO SECTION MAPPING (NEW - FIX #2)
# =============================================================================

# Direct mapping from chunk_type to most relevant template sections
# Used as primary routing before keyword fallback
CHUNK_TYPE_SECTION_MAP = {
    TemplateType.PROCEDURAL: {
        "PROCEDURE": "Step-by-Step Technique",
        "ANATOMY": "Surgical Approach",
        "CLINICAL": "Indications",
        "PATHOLOGY": "Complications and Avoidance",
        "GENERAL": "Preoperative Considerations",
    },
    TemplateType.DISORDER: {
        "PROCEDURE": "Management",
        "ANATOMY": "Pathophysiology",
        "CLINICAL": "Clinical Presentation",
        "PATHOLOGY": "Diagnostic Workup",
        "GENERAL": "Overview",
    },
    TemplateType.ANATOMY: {
        "PROCEDURE": "Surgical Corridors",
        "ANATOMY": "Boundaries and Relationships",
        "CLINICAL": "Neural Structures",
        "PATHOLOGY": "Dural Relationships",
        "GENERAL": "Surface Anatomy",
    },
    TemplateType.ENCYCLOPEDIA: {
        "PROCEDURE": "Surgical Technique",
        "ANATOMY": "Anatomy",
        "CLINICAL": "Clinical Features",
        "PATHOLOGY": "Pathology",
        "GENERAL": "Definition and Overview",
    },
}


# =============================================================================
# ENHANCED CONTEXT ADAPTER
# =============================================================================

class EnhancedContextAdapter:
    """
    Enhanced ContextAdapter aligned with upgraded pipeline.

    Key improvements over original:
    1. Composite quality score from readability, coherence, completeness
    2. Type-based section routing with chunk_type
    3. Caption embedding passthrough for semantic figure matching
    4. CUI preservation for hallucination detection
    5. Improved authority weighting
    """

    # Keyword mappings for fallback classification
    SECTION_KEYWORDS = {
        "indications": ["indication", "candidate", "criteria", "select", "recommend"],
        "positioning": ["position", "lateral", "supine", "prone", "park bench", "mayfield"],
        "approach": ["approach", "exposure", "craniotomy", "incision", "corridor"],
        "technique": ["technique", "step", "dissect", "retract", "identify", "expose", "clip", "resect"],
        "closure": ["closure", "close", "dural", "bone flap", "scalp", "drain"],
        "complications": ["complication", "risk", "avoid", "injury", "deficit", "hemorrhage"],
        "outcomes": ["outcome", "result", "survival", "mortality", "morbidity", "follow"],
        "anatomy": ["anatomy", "structure", "relation", "boundary", "course", "origin"],
        "pathology": ["pathology", "tumor", "lesion", "malignant", "benign", "grade"],
        "clinical": ["presentation", "symptom", "sign", "history", "examination"],
        "imaging": ["imaging", "mri", "ct", "angiography", "enhancement", "signal"],
        "management": ["management", "treatment", "therapy", "surgery", "conservative"],
    }

    def __init__(self, min_quality_score: float = 0.3):
        """
        Initialize enhanced adapter.

        Args:
            min_quality_score: Minimum composite quality to include chunk (0.0-1.0)
        """
        self.min_quality_score = min_quality_score

    def compute_quality_score(self, result: Any) -> float:
        """
        Compute composite quality score from 3 separate scores.

        FIX #1: Handle both old (single quality_score) and new
        (readability, coherence, completeness) formats.
        """
        # Try new format first (3 separate scores)
        readability = getattr(result, 'readability_score', None)
        coherence = getattr(result, 'coherence_score', None)
        completeness = getattr(result, 'completeness_score', None)

        # Only use 3-score format if scores are actually computed (non-zero)
        if all(s is not None and s > 0 for s in [readability, coherence, completeness]):
            # New format: compute weighted average
            # Weight completeness slightly higher as it indicates more useful content
            composite = (
                readability * 0.3 +
                coherence * 0.3 +
                completeness * 0.4
            )
            return max(0.0, min(1.0, composite))

        # Fall back to single quality_score if available and non-zero
        quality = getattr(result, 'quality_score', None)
        if quality is not None and quality > 0:
            return float(quality)

        # Default if no quality metrics available or all zeros (uncomputed)
        return 0.7

    def classify_section_by_type(
        self,
        chunk_type: str,
        template_type: TemplateType
    ) -> Optional[str]:
        """
        FIX #2: Map chunk_type directly to section.

        Returns section name if direct mapping exists, None otherwise.
        """
        type_map = CHUNK_TYPE_SECTION_MAP.get(template_type, {})

        # Normalize chunk_type
        if hasattr(chunk_type, 'value'):
            chunk_type = chunk_type.value
        chunk_type = str(chunk_type).upper()

        return type_map.get(chunk_type)

    def classify_section_by_keywords(
        self,
        content: str,
        template_type: TemplateType
    ) -> str:
        """
        Keyword-based section classification (fallback).

        Same as original _classify_to_section but used as secondary method.
        """
        content_lower = content.lower()
        sections = TEMPLATE_SECTIONS.get(template_type, [])

        best_section = sections[0][0] if sections else "General"
        best_score = 0

        for section_name, _ in sections:
            keywords = []
            for key, kw_list in self.SECTION_KEYWORDS.items():
                if key.lower() in section_name.lower():
                    keywords.extend(kw_list)

            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_section = section_name

        return best_section

    def classify_section(
        self,
        result: Any,
        template_type: TemplateType
    ) -> str:
        """
        FIX #2: Classify chunk to section using type-first, keywords-second approach.
        """
        # Primary: Use chunk_type for direct mapping
        chunk_type = getattr(result, 'chunk_type', None)
        if chunk_type:
            type_section = self.classify_section_by_type(chunk_type, template_type)
            if type_section:
                return type_section

        # Secondary: Fall back to keyword matching
        content = getattr(result, 'content', '')
        return self.classify_section_by_keywords(content, template_type)

    def detect_authority_from_title(self, title: Optional[str]) -> Tuple[AuthoritySource, float]:
        """Detect authority source and score from document title."""
        if not title:
            return AuthoritySource.GENERAL, 0.7

        title_lower = title.lower()

        # Check for known authority sources
        authority_patterns = {
            AuthoritySource.RHOTON: ["rhoton", "cranial anatomy"],
            AuthoritySource.LAWTON: ["lawton", "seven aneurysms"],
            AuthoritySource.SAMII: ["samii"],
            AuthoritySource.AL_MEFTY: ["al-mefty", "almefty", "meningiomas"],
            AuthoritySource.SEKHAR: ["sekhar"],
            AuthoritySource.SPETZLER: ["spetzler"],
            AuthoritySource.YASARGIL: ["yasargil"],
            AuthoritySource.YOUMANS: ["youmans"],
            AuthoritySource.SCHMIDEK: ["schmidek"],
            AuthoritySource.GREENBERG: ["greenberg", "handbook"],
            AuthoritySource.WINN: ["winn", "youmans and winn"],
            AuthoritySource.QUINONES: ["quinones"],
            AuthoritySource.CONNOLLY: ["connolly"],
            AuthoritySource.BENZEL: ["benzel", "spine surgery"],
        }

        for authority, patterns in authority_patterns.items():
            if any(p in title_lower for p in patterns):
                return authority, AUTHORITY_SCORES[authority]

        # Check for journal patterns
        journal_patterns = ["journal", "neurosurgery", "j neurosurg", "spine"]
        if any(p in title_lower for p in journal_patterns):
            return AuthoritySource.JOURNAL, AUTHORITY_SCORES[AuthoritySource.JOURNAL]

        return AuthoritySource.GENERAL, AUTHORITY_SCORES[AuthoritySource.GENERAL]

    def build_image_catalog_entry(self, img: Any, result: Any) -> Dict[str, Any]:
        """
        FIX #3: Build image catalog entry with caption embedding for semantic matching.
        """
        entry = {
            "id": img.id if hasattr(img, 'id') else str(img),
            "caption": getattr(img, 'caption', '') or getattr(img, 'vlm_caption', ''),
            "vlm_caption": getattr(img, 'vlm_caption', ''),
            "path": str(getattr(img, 'file_path', '') or getattr(img, 'image_path', '')),
            "page": getattr(img, 'page_number', 0),
            "document_title": getattr(result, 'document_title', ''),
            "image_type": getattr(img, 'image_type', 'unknown'),
            "quality_score": getattr(img, 'quality_score', 0.0),
        }

        # FIX #3: Include caption embedding for semantic figure resolution
        caption_embedding = getattr(img, 'caption_embedding', None)
        if caption_embedding is not None:
            entry["caption_embedding"] = caption_embedding

        # Include CLIP embedding as fallback
        clip_embedding = getattr(img, 'embedding', None)
        if clip_embedding is not None:
            entry["clip_embedding"] = clip_embedding

        # Include image CUIs for medical concept matching
        img_cuis = getattr(img, 'cuis', [])
        if img_cuis:
            entry["cuis"] = img_cuis

        return entry

    def adapt(
        self,
        topic: str,
        search_results: List[Any],  # List[SearchResult]
        template_type: TemplateType,
    ) -> Dict[str, Any]:
        """
        Adapt SearchResult list into template-ready context.

        Enhanced with:
        - FIX #1: Composite quality score filtering
        - FIX #2: Type-based section routing
        - FIX #3: Caption embedding passthrough
        - FIX #4: CUI preservation
        - FIX #5: Improved authority weighting
        """
        # Initialize sections
        sections_content: Dict[str, List[Dict]] = {
            section[0]: [] for section in TEMPLATE_SECTIONS.get(template_type, [])
        }

        all_sources = []
        image_catalog = []
        all_cuis = set()  # FIX #4: Track all CUIs for validation
        filtered_count = 0

        for result in search_results:
            # FIX #1: Compute composite quality score
            quality_score = self.compute_quality_score(result)

            # Quality filtering
            if quality_score < self.min_quality_score:
                filtered_count += 1
                continue

            # Detect authority
            authority, detected_score = self.detect_authority_from_title(
                getattr(result, 'document_title', None)
            )

            # Use detected score unless SearchResult has explicit authority_score
            authority_score = getattr(result, 'authority_score', detected_score)
            if authority_score == 1.0 and detected_score < 1.0:
                # Default value - use detected
                authority_score = detected_score

            # FIX #2: Classify section (type-first, keywords-second)
            section = self.classify_section(result, template_type)

            # FIX #4: Collect CUIs
            chunk_cuis = getattr(result, 'cuis', []) or []
            all_cuis.update(chunk_cuis)

            # Get individual quality components (for transparency)
            readability = getattr(result, 'readability_score', None)
            coherence = getattr(result, 'coherence_score', None)
            completeness = getattr(result, 'completeness_score', None)

            # FIX #5: Improved combined score with quality weighting
            final_score = getattr(result, 'final_score', 0.5)
            combined_score = final_score * authority_score * quality_score

            # Build chunk data
            chunk_data = {
                "id": getattr(result, 'chunk_id', str(id(result))),
                "content": getattr(result, 'content', ''),
                "document_id": getattr(result, 'document_id', ''),
                "document_title": getattr(result, 'document_title', '') or getattr(result, 'title', ''),
                "page": getattr(result, 'page_start', 0),
                "chunk_type": self._normalize_chunk_type(getattr(result, 'chunk_type', 'GENERAL')),
                "authority": authority.value,
                "authority_score": authority_score,
                "semantic_score": getattr(result, 'semantic_score', 0.0),
                "keyword_score": getattr(result, 'keyword_score', 0.0),
                "final_score": final_score,
                "quality_score": quality_score,
                "combined_score": combined_score,
                "entity_names": getattr(result, 'entity_names', []),
                # FIX #4: Preserve CUIs
                "cuis": chunk_cuis,
                # Quality score components (for debugging)
                "quality_components": {
                    "readability": readability,
                    "coherence": coherence,
                    "completeness": completeness,
                } if readability is not None else None,
            }

            # Add to section
            if section in sections_content:
                sections_content[section].append(chunk_data)
            else:
                # If section doesn't exist, add to first section
                first_section = list(sections_content.keys())[0]
                sections_content[first_section].append(chunk_data)

            # Build source reference
            doc_title = chunk_data["document_title"] or "Unknown"
            source_ref = {
                "source": f"{doc_title}, p.{chunk_data['page']}",
                "document_id": chunk_data["document_id"],
                "authority": authority.value,
                "authority_score": authority_score,
                "chunks_used": 1,
            }

            # Deduplicate sources
            existing = next(
                (s for s in all_sources if s["document_id"] == chunk_data["document_id"]),
                None
            )
            if existing:
                existing["chunks_used"] += 1
            else:
                all_sources.append(source_ref)

            # FIX #3: Collect images with embeddings
            images = getattr(result, 'images', []) or []
            for img in images:
                catalog_entry = self.build_image_catalog_entry(img, result)
                image_catalog.append(catalog_entry)

        # Sort chunks within each section by combined score (highest first)
        for section in sections_content:
            sections_content[section].sort(
                key=lambda x: x["combined_score"],
                reverse=True
            )

        # Sort sources by authority score
        all_sources.sort(
            key=lambda x: x.get("authority_score", 0.7),
            reverse=True
        )

        # Log adaptation stats
        total_adapted = sum(len(chunks) for chunks in sections_content.values())
        logger.info(
            f"Adapted {total_adapted}/{len(search_results)} results "
            f"({filtered_count} filtered by quality), "
            f"{len(image_catalog)} images, "
            f"{len(all_cuis)} unique CUIs"
        )

        return {
            "topic": topic,
            "template_type": template_type,
            "sections": sections_content,
            "sources": all_sources,
            "image_catalog": image_catalog,
            "requirements": TEMPLATE_REQUIREMENTS.get(template_type, {}),
            "total_chunks": len(search_results),
            "filtered_chunks": filtered_count,
            # FIX #4: Include all CUIs for content validation
            "all_cuis": list(all_cuis),
        }

    def _normalize_chunk_type(self, chunk_type: Any) -> str:
        """Normalize chunk_type to string."""
        if hasattr(chunk_type, 'value'):
            return chunk_type.value
        return str(chunk_type).upper()


# =============================================================================
# ENHANCED FIGURE RESOLVER (FIX #3)
# =============================================================================

class EnhancedFigureResolver:
    """
    Enhanced FigureResolver with semantic matching via caption embeddings.

    FIX #3: Uses caption_embedding for semantic similarity matching
    instead of keyword-only matching.
    """

    def __init__(
        self,
        min_match_score: float = 0.3,
        prefer_semantic: bool = True
    ):
        """
        Initialize resolver.

        Args:
            min_match_score: Minimum score to consider a match
            prefer_semantic: If True, prefer semantic matches over keyword
        """
        self.min_match_score = min_match_score
        self.prefer_semantic = prefer_semantic

    def compute_keyword_score(
        self,
        request_desc: str,
        image_caption: str
    ) -> float:
        """Compute keyword overlap score."""
        if not request_desc or not image_caption:
            return 0.0

        # Tokenize and normalize
        request_words = set(request_desc.lower().split())
        caption_words = set(image_caption.lower().split())

        # Remove common words
        stop_words = {'the', 'a', 'an', 'of', 'in', 'for', 'to', 'with', 'and', 'or'}
        request_words -= stop_words
        caption_words -= stop_words

        if not request_words:
            return 0.0

        # Jaccard-like overlap
        overlap = len(request_words & caption_words)
        score = overlap / len(request_words)

        return score

    def compute_semantic_score(
        self,
        request_embedding: Any,
        caption_embedding: Any
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if request_embedding is None or caption_embedding is None:
            return 0.0

        try:
            import numpy as np

            # Ensure numpy arrays
            req_vec = np.array(request_embedding).flatten()
            cap_vec = np.array(caption_embedding).flatten()

            # Check dimensions match
            if req_vec.shape != cap_vec.shape:
                logger.warning(f"Embedding dimension mismatch: {req_vec.shape} vs {cap_vec.shape}")
                return 0.0

            # Cosine similarity
            dot = np.dot(req_vec, cap_vec)
            norm_req = np.linalg.norm(req_vec)
            norm_cap = np.linalg.norm(cap_vec)

            if norm_req == 0 or norm_cap == 0:
                return 0.0

            similarity = dot / (norm_req * norm_cap)

            return float(similarity)

        except Exception as e:
            logger.warning(f"Semantic score computation failed: {e}")
            return 0.0

    def compute_cui_score(
        self,
        request_cuis: List[str],
        image_cuis: List[str]
    ) -> float:
        """Compute CUI overlap score."""
        if not request_cuis or not image_cuis:
            return 0.0

        request_set = set(request_cuis)
        image_set = set(image_cuis)

        overlap = len(request_set & image_set)

        if not request_set:
            return 0.0

        return overlap / len(request_set)

    def resolve(
        self,
        figure_requests: List[Any],  # List[FigureRequest]
        image_catalog: List[Dict],
        request_embeddings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], List[Dict]]:
        """
        Match figure requests to images using semantic + keyword matching.

        Args:
            figure_requests: List of FigureRequest objects
            image_catalog: List of image dicts from ContextAdapter
            request_embeddings: Optional pre-computed embeddings for requests

        Returns:
            Tuple of (updated_requests, resolved_figures)
        """
        resolved = []
        updated_requests = []

        for request in figure_requests:
            req_desc = getattr(request, 'description', '') or getattr(request, 'topic', '') or str(request)
            req_section = getattr(request, 'section', '') or getattr(request, 'context', '')
            req_cuis = getattr(request, 'cuis', [])

            # Get request embedding if available
            req_embedding = None
            if request_embeddings:
                req_id = getattr(request, 'id', None) or getattr(request, 'placeholder_id', None)
                if req_id:
                    req_embedding = request_embeddings.get(req_id)

            best_match = None
            best_score = 0.0
            best_breakdown = {}

            for image in image_catalog:
                # Compute component scores
                keyword_score = self.compute_keyword_score(
                    req_desc,
                    image.get('caption', '') or image.get('vlm_caption', '')
                )

                semantic_score = 0.0
                if self.prefer_semantic:
                    semantic_score = self.compute_semantic_score(
                        req_embedding,
                        image.get('caption_embedding')
                    )

                cui_score = self.compute_cui_score(
                    req_cuis,
                    image.get('cuis', [])
                )

                # Combined score
                if self.prefer_semantic and semantic_score > 0:
                    # Semantic-weighted
                    combined = (
                        semantic_score * 0.5 +
                        keyword_score * 0.3 +
                        cui_score * 0.2
                    )
                else:
                    # Keyword-weighted
                    combined = (
                        keyword_score * 0.6 +
                        cui_score * 0.4
                    )

                if combined > best_score:
                    best_score = combined
                    best_match = image
                    best_breakdown = {
                        'keyword': keyword_score,
                        'semantic': semantic_score,
                        'cui': cui_score,
                        'combined': combined
                    }

            # Accept match if above threshold
            if best_match and best_score >= self.min_match_score:
                resolved.append({
                    'request': request,
                    'request_description': req_desc,
                    'image_id': best_match.get('id'),
                    'image_path': best_match.get('path'),
                    'image_caption': best_match.get('caption') or best_match.get('vlm_caption'),
                    'confidence': best_score,
                    'score_breakdown': best_breakdown,
                })

                # Update request with match
                if hasattr(request, 'resolved_id'):
                    request.resolved_id = best_match.get('id')
                if hasattr(request, 'resolved_path'):
                    request.resolved_path = best_match.get('path')
                if hasattr(request, 'confidence'):
                    request.confidence = best_score

            updated_requests.append(request)

        logger.info(f"Resolved {len(resolved)}/{len(figure_requests)} figure requests")

        return updated_requests, resolved


# =============================================================================
# CONTENT VALIDATOR (FIX #4 - NEW)
# =============================================================================

class ContentValidator:
    """
    Validates synthesized content against source CUIs.

    FIX #4: Uses preserved CUIs to detect potential hallucinations.
    """

    def __init__(self, umls_extractor=None, log_hallucinations: bool = True):
        """
        Initialize validator.

        Args:
            umls_extractor: Optional UMLSExtractor instance for entity extraction
            log_hallucinations: If True, log detected hallucinations
        """
        self.extractor = umls_extractor
        self.log_hallucinations = log_hallucinations
        self._hallucination_log: List[Dict] = []

    @property
    def hallucination_log(self) -> List[Dict]:
        """Get the hallucination log."""
        return self._hallucination_log

    def clear_log(self):
        """Clear the hallucination log."""
        self._hallucination_log = []

    def validate_against_sources(
        self,
        generated_text: str,
        source_cuis: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate generated content against source CUIs.

        Args:
            generated_text: Synthesized text to validate
            source_cuis: CUIs from source chunks
            threshold: Confidence threshold for extracted entities

        Returns:
            Validation report with potential issues
        """
        if not self.extractor:
            return {
                'validated': False,
                'reason': 'No UMLS extractor available',
                'issues': []
            }

        if not source_cuis:
            return {
                'validated': True,
                'reason': 'No source CUIs to validate against',
                'issues': []
            }

        try:
            # Extract entities from generated text
            entities = self.extractor.extract(generated_text)

            # Filter by confidence
            generated_cuis = set(
                e.cui for e in entities
                if e.score >= threshold and e.cui
            )

            source_cui_set = set(source_cuis)

            # Find CUIs in generated text not in sources
            unsupported = generated_cuis - source_cui_set

            issues = []
            for cui in unsupported:
                # Find the entity name for this CUI
                entity = next((e for e in entities if e.cui == cui), None)
                if entity:
                    issue = {
                        'type': 'POTENTIAL_HALLUCINATION',
                        'cui': cui,
                        'name': entity.name,
                        'confidence': entity.score,
                        'message': f"Medical concept '{entity.name}' ({cui}) not found in source materials"
                    }
                    issues.append(issue)

                    # Log if enabled
                    if self.log_hallucinations:
                        logger.warning(f"Potential hallucination: {issue['message']}")
                        self._hallucination_log.append(issue)

            return {
                'validated': True,
                'generated_cuis': len(generated_cuis),
                'source_cuis': len(source_cui_set),
                'unsupported_cuis': len(unsupported),
                'issues': issues,
                'hallucination_risk': len(issues) > 0
            }

        except Exception as e:
            logger.warning(f"Content validation failed: {e}")
            return {
                'validated': False,
                'reason': str(e),
                'issues': []
            }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Enhanced Context Adapter for NeuroSynth")
    print("=" * 50)
    print("\nFixes implemented:")
    print("1. Composite quality score from 3 separate scores")
    print("2. Type-based section routing (chunk_type primary)")
    print("3. Caption embedding passthrough for semantic figure matching")
    print("4. CUI preservation for content validation")
    print("5. Improved authority weighting in combined score")
    print("\nTo use, replace ContextAdapter in src/synthesis/engine.py:")
    print("  from synthesis_fixes import EnhancedContextAdapter")
    print("  self.adapter = EnhancedContextAdapter()")
