"""
NeuroSynth Synthesis Module

Generates textbook-quality chapters from SearchResult retrieval.
Uses existing Phase 1 models - no adapters needed.

V3 Enhancements (2025):
- Semantic Router: Vector-based section classification
- Semantic Figure Resolver: Embedding-based figure matching
- Conflict-Aware Merger: Internal vs external fact resolution
- Adversarial Reviewer: Safety validation against Tier 1 sources
- Gap Analysis: LLM-based knowledge gap detection
"""

from .engine import (
    SynthesisEngine,
    TemplateType,
    SynthesisResult,
    SynthesisSection,
    FigureRequest,
    RateLimiter,
    ContextAdapter,
    FigureResolver,
    AuthoritySource,
    AuthorityConfig,
    AuthorityRegistry,
    get_authority_registry,
    set_authority_registry,
    DEFAULT_AUTHORITY_SCORES,
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
)

from .conflicts import (
    ConflictHandler,
    ConflictReport,
    Conflict,
    ConflictType,
)

# V3 Enhanced Synthesis
from .enhanced_engine import (
    EnhancedSynthesisEngine,
    EnhancedSynthesisResult,
    add_enrichment_to_engine,
    create_enhanced_engine_from_env,
)

from .research_enricher import (
    ResearchEnricher,
    EnrichmentConfig,
    EnrichmentResult,
    KnowledgeGap,
    GapType,
    GapPriority,
    ExternalSource,
    ExternalSearchResult,
    GapAnalyzer,
    PerplexityProvider,
    GeminiGroundingProvider,
    create_enricher_from_env,
)

# V3 Semantic Router
from .router import (
    SemanticRouter,
    SectionPrototypes,
    RouteResult,
    KeywordFallbackRouter,
)

# V3 Semantic Figure Resolver
from .figure_resolver_semantic import (
    SemanticFigureResolver,
    FigureMatch,
    LegacyFigureResolver,
)

# V3 Conflict-Aware Merger
from .conflict_merger import (
    ConflictAwareMerger,
    MergeResult,
    DetectedConflict,
    ExtractedFact,
    ConflictCategory,
    ResolutionStrategy,
    merge_internal_external,
)

# V3 Adversarial Reviewer Agent
from .agents.reviewer import (
    AdversarialReviewer,
    ReviewResult,
    ControversyWarning,
    HeuristicReviewer,
)

# V3 Enrichment Models (Pydantic schemas)
from .models_enrichment import (
    GapItem,
    GapAnalysisResult,
    V3SynthesisMetadata,
)

# 14-Stage Neurosurgical Gap Detection System
from .gap_models import (
    GapType as NeuroGapType,
    GapPriority as NeuroGapPriority,
    GapFillStrategy,
    TemplateType as GapTemplateType,
    Gap,
    GapFillResult,
    GapReport,
    SAFETY_CRITICAL_TYPES,
    PRIORITY_THRESHOLDS,
)

from .gap_detection_service import (
    GapDetectionService,
    SearchResult as GapSearchResult,
)

from .gap_filling_service import (
    GapFillingService,
    GapFillConfig,
)

from .subspecialty_classifier import (
    SubspecialtyClassifier,
    Subspecialty,
    ClassificationResult,
    classify_subspecialty,
)

from .neurosurgical_ontology import (
    NeurosurgicalOntology,
    CRANIAL_FORAMINA,
    SKULL_BASE_TRIANGLES,
    ARTERIAL_SEGMENTS,
    ICP_PARAMETERS,
    CEREBRAL_HEMODYNAMICS,
    WHO_2021_CNS_TUMORS,
    LANDMARK_TRIALS,
    DANGER_ZONES,
    PROCEDURAL_TEMPLATES,
)

__all__ = [
    # Synthesis Engine
    "SynthesisEngine",
    "TemplateType",
    "SynthesisResult",
    "SynthesisSection",
    "FigureRequest",
    "RateLimiter",
    "ContextAdapter",
    "FigureResolver",
    # Authority System
    "AuthoritySource",
    "AuthorityConfig",
    "AuthorityRegistry",
    "get_authority_registry",
    "set_authority_registry",
    "DEFAULT_AUTHORITY_SCORES",
    # Conflict Detection
    "ConflictHandler",
    "ConflictReport",
    "Conflict",
    "ConflictType",
    # Template Constants
    "TEMPLATE_SECTIONS",
    "TEMPLATE_REQUIREMENTS",
    # V3 Enhanced Synthesis
    "EnhancedSynthesisEngine",
    "EnhancedSynthesisResult",
    "add_enrichment_to_engine",
    "create_enhanced_engine_from_env",
    # V3 Research Enricher
    "ResearchEnricher",
    "EnrichmentConfig",
    "EnrichmentResult",
    "KnowledgeGap",
    "GapType",
    "GapPriority",
    "ExternalSource",
    "ExternalSearchResult",
    "GapAnalyzer",
    "PerplexityProvider",
    "GeminiGroundingProvider",
    "create_enricher_from_env",
    # V3 Semantic Router
    "SemanticRouter",
    "SectionPrototypes",
    "RouteResult",
    "KeywordFallbackRouter",
    # V3 Semantic Figure Resolver
    "SemanticFigureResolver",
    "FigureMatch",
    "LegacyFigureResolver",
    # V3 Conflict-Aware Merger
    "ConflictAwareMerger",
    "MergeResult",
    "DetectedConflict",
    "ExtractedFact",
    "ConflictCategory",
    "ResolutionStrategy",
    "merge_internal_external",
    # V3 Adversarial Reviewer
    "AdversarialReviewer",
    "ReviewResult",
    "ControversyWarning",
    "HeuristicReviewer",
    # V3 Enrichment Models
    "GapItem",
    "GapAnalysisResult",
    "V3SynthesisMetadata",
    # 14-Stage Neurosurgical Gap Detection
    "NeuroGapType",
    "NeuroGapPriority",
    "GapFillStrategy",
    "GapTemplateType",
    "Gap",
    "GapFillResult",
    "GapReport",
    "SAFETY_CRITICAL_TYPES",
    "PRIORITY_THRESHOLDS",
    "GapDetectionService",
    "GapSearchResult",
    "GapFillingService",
    "GapFillConfig",
    "SubspecialtyClassifier",
    "Subspecialty",
    "ClassificationResult",
    "classify_subspecialty",
    # Neurosurgical Ontology
    "NeurosurgicalOntology",
    "CRANIAL_FORAMINA",
    "SKULL_BASE_TRIANGLES",
    "ARTERIAL_SEGMENTS",
    "ICP_PARAMETERS",
    "CEREBRAL_HEMODYNAMICS",
    "WHO_2021_CNS_TUMORS",
    "LANDMARK_TRIALS",
    "DANGER_ZONES",
    "PROCEDURAL_TEMPLATES",
]
