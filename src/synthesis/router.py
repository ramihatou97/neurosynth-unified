"""
NeuroSynth - Semantic Section Router
====================================

Vector-based section classification replacing naive keyword matching.
Routes chunks to appropriate synthesis sections based on semantic similarity.

Fixes Issue #1: Section classification uses naive keyword matching.
A chunk about "anatomy of the approach" could be misclassified as ANATOMY
when it belongs in SURGICAL APPROACH.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Protocol, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Protocol for embedding service interface."""
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        ...


@dataclass
class RouteResult:
    """Result of routing a chunk to a section."""
    section_name: str
    confidence_score: float
    is_confident: bool  # True if score > threshold
    all_scores: Optional[Dict[str, float]] = None  # Scores for all sections


class SectionPrototypes:
    """
    Gold-standard descriptions for each synthesis section.

    The router compares chunks against these definitions rather than keywords.
    Each description is designed to capture the semantic essence of what
    content belongs in that section.
    """

    PROCEDURAL = {
        "INDICATION": (
            "Clinical reasoning for surgery, patient selection criteria, "
            "diagnostic imaging analysis, decision-making factors, "
            "preoperative assessment, surgical candidacy evaluation, "
            "risk-benefit analysis for operative intervention."
        ),
        "ANATOMY": (
            "Detailed neuroanatomical structures, vascular relationships, "
            "neural pathways, cisterns, sulci, gyri, arterial territories, "
            "venous drainage patterns, skull base foramina, and "
            "physiological context relevant to the surgical region. "
            "Descriptive anatomy without surgical manipulation."
        ),
        "EQUIPMENT": (
            "Surgical instruments, microscope settings and positioning, "
            "intraoperative neuromonitoring setup including MEP SSEP BAEP, "
            "patient positioning on operating table, head fixation, "
            "retractor systems, irrigation, and operating room configuration."
        ),
        "APPROACH": (
            "Step-by-step surgical approach, skin incision planning, "
            "craniotomy technique, bone removal and landmarks, "
            "dural opening strategy, and exposure of the surgical target. "
            "How to get to the lesion, not what to do with it."
        ),
        "TECHNIQUE": (
            "Microsurgical dissection techniques, tumor resection strategies, "
            "aneurysm clipping methodology, handling of pathology, "
            "specific manipulative maneuvers, arachnoid dissection, "
            "vessel handling, hemostasis techniques during resection."
        ),
        "CLOSURE": (
            "Hemostasis verification and achievement, dural reconstruction "
            "and watertight closure, bone flap replacement and fixation, "
            "wound closure in layers, drain placement, wound dressing, "
            "postoperative head positioning."
        ),
        "COMPLICATIONS": (
            "Intraoperative hazards and how to avoid them, postoperative risks, "
            "adverse events and their management, error recognition and recovery, "
            "surgical pearls and pitfalls, danger zones, structures at risk, "
            "what to avoid, caution warnings."
        ),
        "OUTCOMES": (
            "Long-term prognosis and survival data, follow-up protocols, "
            "success rates and definitions, recurrence rates and patterns, "
            "quality of life metrics, mortality and morbidity statistics, "
            "functional outcomes and scales."
        ),
    }

    DISORDER = {
        "EPIDEMIOLOGY": (
            "Incidence and prevalence data, demographic distributions, "
            "risk factors and associations, geographic variations, "
            "temporal trends in disease occurrence."
        ),
        "PATHOPHYSIOLOGY": (
            "Disease mechanisms at cellular and molecular level, "
            "pathological changes, natural history progression, "
            "genetic and environmental factors in disease development."
        ),
        "CLINICAL_PRESENTATION": (
            "Symptoms and signs, clinical syndromes and variants, "
            "disease staging, neurological examination findings, "
            "presenting complaints and their patterns."
        ),
        "DIAGNOSIS": (
            "Imaging findings and interpretation, laboratory studies, "
            "differential diagnosis considerations, diagnostic criteria, "
            "staging systems and classification schemes."
        ),
        "MANAGEMENT": (
            "Treatment options including surgical and medical approaches, "
            "decision algorithms, evidence-based recommendations, "
            "timing of intervention, multimodal therapy integration."
        ),
        "PROGNOSIS": (
            "Survival statistics, prognostic factors and scoring systems, "
            "quality of life outcomes, long-term follow-up data, "
            "recurrence patterns and their management."
        ),
    }

    ANATOMY = {
        "SURFACE_ANATOMY": "External landmarks, surface projections, palpable structures.",
        "OSTEOLOGY": "Bone anatomy, foramina, sutures, articulations.",
        "VASCULAR": "Arterial supply, venous drainage, anastomoses, perforators.",
        "NEURAL": "Cranial nerves, nerve roots, neural pathways, nuclei.",
        "MENINGEAL": "Dural folds, arachnoid cisterns, subarachnoid spaces.",
        "VENTRICULAR": "Ventricles, CSF pathways, choroid plexus.",
        "CLINICAL_CORRELATES": "Anatomical basis of clinical findings and surgical implications.",
    }

    ENCYCLOPEDIA = {
        "OVERVIEW": "General introduction, definition, historical context.",
        "CLASSIFICATION": "Types, subtypes, classification systems.",
        "ETIOLOGY": "Causes, risk factors, pathogenesis.",
        "CLINICAL_FEATURES": "Presentation, symptoms, signs.",
        "WORKUP": "Diagnostic approach, investigations.",
        "TREATMENT": "Management options, surgical and medical.",
        "PROGNOSIS": "Outcomes, follow-up, recurrence.",
        "SPECIAL_CONSIDERATIONS": "Pediatric, elderly, pregnancy, special populations.",
    }


class SemanticRouter:
    """
    Routes chunks to sections based on semantic similarity.

    Replaces keyword-based classification with vector comparison against
    gold-standard section descriptions.

    Usage:
        router = SemanticRouter(embedding_service, template_type="PROCEDURAL")
        await router.initialize()

        result = await router.route_chunk(chunk_text)
        print(f"Section: {result.section_name}, Confidence: {result.confidence_score}")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        template_type: str = "PROCEDURAL",
        confidence_threshold: float = 0.72
    ):
        """
        Initialize the semantic router.

        Args:
            embedding_service: Service for generating embeddings
            template_type: Type of template (PROCEDURAL, DISORDER, ANATOMY, ENCYCLOPEDIA)
            confidence_threshold: Minimum confidence to assign to a section
        """
        self.embedder = embedding_service
        self.threshold = confidence_threshold
        self.template_type = template_type

        # Get prototypes for this template type
        self.prototypes = getattr(
            SectionPrototypes,
            template_type,
            SectionPrototypes.PROCEDURAL
        )

        # Cache for prototype embeddings
        self._route_map: Dict[str, np.ndarray] = {}
        self._is_initialized = False

    async def initialize(self):
        """
        Pre-compute embeddings for all section prototypes.

        Call this once at startup or lazy-load on first request.
        """
        if self._is_initialized:
            return

        logger.info(f"Initializing SemanticRouter for {self.template_type}")

        for section, description in self.prototypes.items():
            try:
                vector = await self.embedder.embed_text(description)
                self._route_map[section] = np.array(vector)
                logger.debug(f"Embedded prototype for section: {section}")
            except Exception as e:
                logger.error(f"Failed to embed prototype for {section}: {e}")
                raise

        self._is_initialized = True
        logger.info(f"SemanticRouter initialized with {len(self._route_map)} sections")

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    async def route_chunk(
        self,
        chunk_text: str,
        chunk_embedding: Optional[List[float]] = None
    ) -> RouteResult:
        """
        Determine which section a chunk belongs to.

        Args:
            chunk_text: The chunk content (used if embedding not provided)
            chunk_embedding: Pre-computed embedding (optional, saves API call)

        Returns:
            RouteResult with section name, confidence, and all scores
        """
        if not self._is_initialized:
            await self.initialize()

        # Get embedding for chunk - use head-tail strategy for long chunks
        if chunk_embedding is not None:
            target_vec = np.array(chunk_embedding)
        else:
            # Head-tail strategy: capture intro + conclusion for long chunks
            if len(chunk_text) > 2000:
                truncated = chunk_text[:1000] + "\n...\n" + chunk_text[-1000:]
            else:
                truncated = chunk_text
            target_vec = np.array(await self.embedder.embed_text(truncated))

        # Compare against all section prototypes
        scores = {}
        best_section = "UNCLASSIFIED"
        best_score = -1.0

        for section_name, proto_vec in self._route_map.items():
            score = self._cosine_similarity(target_vec, proto_vec)
            scores[section_name] = score

            if score > best_score:
                best_score = score
                best_section = section_name

        return RouteResult(
            section_name=best_section,
            confidence_score=best_score,
            is_confident=(best_score >= self.threshold),
            all_scores=scores
        )

    async def route_chunks_batch(
        self,
        chunks: List[Any],
        content_key: str = "content",
        embedding_key: str = "embedding"
    ) -> Dict[str, List[Any]]:
        """
        Route multiple chunks to sections.

        Args:
            chunks: List of chunk objects/dicts
            content_key: Key to access content in chunk
            embedding_key: Key to access embedding in chunk

        Returns:
            Dict mapping section names to lists of chunks
        """
        if not self._is_initialized:
            await self.initialize()

        # Initialize buckets for all sections plus UNCLASSIFIED
        organized = {section: [] for section in self._route_map.keys()}
        organized["UNCLASSIFIED"] = []

        for chunk in chunks:
            # Get content and embedding
            if isinstance(chunk, dict):
                content = chunk.get(content_key, "")
                embedding = chunk.get(embedding_key)
            else:
                content = getattr(chunk, content_key, "")
                embedding = getattr(chunk, embedding_key, None)

            # Route the chunk
            result = await self.route_chunk(content, embedding)

            # Assign to appropriate bucket
            if result.is_confident:
                organized[result.section_name].append(chunk)
            else:
                organized["UNCLASSIFIED"].append(chunk)
                logger.debug(
                    f"Low confidence routing ({result.confidence_score:.3f}): "
                    f"best={result.section_name}"
                )

        # Log routing summary
        for section, section_chunks in organized.items():
            if section_chunks:
                logger.info(f"Routed {len(section_chunks)} chunks to {section}")

        return organized

    def get_section_description(self, section_name: str) -> str:
        """Get the prototype description for a section."""
        return self.prototypes.get(section_name, "")

    def list_sections(self) -> List[str]:
        """List all available sections for current template."""
        return list(self.prototypes.keys())


class KeywordFallbackRouter:
    """
    Fallback keyword-based router for when embeddings are unavailable.

    Uses pattern matching as a degraded fallback.
    """

    KEYWORD_MAP = {
        "INDICATION": ["indication", "candidate", "selection", "diagnostic", "preoperative"],
        "ANATOMY": ["anatomy", "artery", "vein", "nerve", "cistern", "sulcus", "gyrus"],
        "EQUIPMENT": ["instrument", "microscope", "monitoring", "position", "retractor"],
        "APPROACH": ["approach", "incision", "craniotomy", "exposure", "dural opening"],
        "TECHNIQUE": ["technique", "dissection", "resection", "clipping", "manipulation"],
        "CLOSURE": ["closure", "hemostasis", "dural repair", "bone flap", "suture"],
        "COMPLICATIONS": ["complication", "hazard", "risk", "avoid", "caution", "pitfall"],
        "OUTCOMES": ["outcome", "prognosis", "survival", "recurrence", "follow-up"],
    }

    def route_chunk(self, chunk_text: str) -> RouteResult:
        """Route using keyword matching (fallback method)."""
        text_lower = chunk_text.lower()

        scores = {}
        for section, keywords in self.KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
            scores[section] = score

        best_section = max(scores, key=scores.get)
        best_score = scores[best_section]

        return RouteResult(
            section_name=best_section if best_score > 0.1 else "UNCLASSIFIED",
            confidence_score=best_score,
            is_confident=best_score > 0.2,
            all_scores=scores
        )
