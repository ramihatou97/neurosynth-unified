"""
NeuroSynth Synthesis Module

Generates textbook-quality chapters from SearchResult retrieval.
Uses existing Phase 1 models - no adapters needed.
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
]
