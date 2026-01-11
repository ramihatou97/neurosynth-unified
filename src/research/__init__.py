"""
NeuroSynth Unified - Research Package
======================================

External research integration for enhanced RAG capabilities.

Components:
- models.py: Data models for external search results
- external_search.py: Perplexity and Gemini search clients
- research_enricher.py: Gap analysis and context merging

Architecture:
    Query → Internal Search + External Search → Gap Analysis → Merged Context

Quick Start:
    from src.research import ResearchEnricher, SearchMode

    enricher = ResearchEnricher(
        search_service=search_service,
        perplexity_client=perplexity,
        gemini_client=gemini
    )

    context = await enricher.enrich(
        query="Latest DBS infection protocols",
        mode=SearchMode.HYBRID
    )

    # context.internal_chunks - from your database
    # context.external_results - from web search
    # context.gap_report - conflicts and gaps identified

Search Modes:
    - STANDARD: Internal database only (fastest, $0)
    - HYBRID: Internal + Perplexity web search (recommended)
    - DEEP_RESEARCH: Internal + Gemini reasoning (complex queries)
    - EXTERNAL_ONLY: Web search only (non-corpus questions)

Environment Variables:
    PERPLEXITY_API_KEY - Required for HYBRID mode
    GEMINI_API_KEY - Required for DEEP_RESEARCH mode
    ENABLE_EXTERNAL_SEARCH - Enable/disable external search (default: true)
    DEFAULT_SEARCH_MODE - Default mode (default: hybrid)
"""

# Models
from src.research.models import (
    SearchMode,
    ExternalSearchResult,
    ExternalCitation,
    GapReport,
    Conflict,
    EnrichedContext,
    EnricherConfig,
    ExternalSearchConfig,
    AlertLevel,
    ConflictSeverity,
)

# External Search Clients
from src.research.external_search import (
    PerplexitySearchClient,
    GeminiDeepResearchClient,
    ExternalSearchError,
    RateLimitError,
    APIKeyError,
)

# Research Enricher
from src.research.research_enricher import (
    ResearchEnricher,
    create_enricher_from_env,
)

__all__ = [
    # Enums
    'SearchMode',

    # Models
    'ExternalSearchResult',
    'ExternalCitation',
    'GapReport',
    'Conflict',
    'EnrichedContext',
    'EnricherConfig',
    'ExternalSearchConfig',

    # Clients
    'PerplexitySearchClient',
    'GeminiDeepResearchClient',

    # Errors
    'ExternalSearchError',
    'RateLimitError',
    'APIKeyError',

    # Main Interface
    'ResearchEnricher',
    'create_enricher_from_env',
]
