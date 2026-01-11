"""
NeuroSynth Unified - Research Models
=====================================

Data models for external search results, gap analysis, and enriched context.

These models define the contract between:
- External search clients (Perplexity, Gemini)
- Research enricher (gap analysis)
- Unified RAG engine (context assembly)
- API layer (response serialization)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from uuid import UUID


# =============================================================================
# Enums
# =============================================================================

class SearchMode(str, Enum):
    """
    Available search modes for the unified RAG engine.

    Each mode represents a different balance of:
    - Speed (latency)
    - Cost (API calls)
    - Comprehensiveness (source coverage)
    - Accuracy (grounding in external facts)
    """

    STANDARD = "standard"
    """
    Internal database only.
    - Fastest (~2s)
    - Zero external API cost
    - Uses only curated content
    - Best for: Known topics in your corpus
    """

    HYBRID = "hybrid"
    """
    Internal + Perplexity web search.
    - Moderate latency (~4s)
    - Small cost (~$0.01/query)
    - Cross-validates internal with live web
    - Best for: Clinical questions needing current guidelines
    """

    DEEP_RESEARCH = "deep_research"
    """
    Internal + Gemini 2.5 Pro with Google Search grounding.
    - Higher latency (~10s)
    - Higher cost (~$0.05/query)
    - Multi-hop reasoning capability
    - Best for: Complex synthesis, protocol drafting
    """

    EXTERNAL_ONLY = "external"
    """
    Web search only, no internal database.
    - Moderate latency (~3s)
    - Small cost (~$0.01/query)
    - Ignores curated content
    - Best for: Questions outside your corpus
    """

    AUTO = "auto"
    """
    Automatically select mode based on query analysis.
    - Analyzes query intent and complexity
    - Routes to most appropriate mode
    - Best for: General use when unsure
    """


class SourceType(str, Enum):
    """Source type for citation attribution."""
    INTERNAL = "internal"      # From curated database
    WEB = "web"                # From web search
    GROUNDED = "grounded"      # From Gemini with Google grounding
    SYNTHESIZED = "synthesized"  # AI-generated synthesis


class ConflictSeverity(str, Enum):
    """Severity level for detected conflicts."""
    LOW = "low"            # Minor discrepancy, likely terminology
    MEDIUM = "medium"      # Notable difference, may affect decision
    HIGH = "high"          # Significant conflict, requires attention
    CRITICAL = "critical"  # Safety-relevant conflict


class AlertLevel(str, Enum):
    """
    Overall alert level for gap analysis results.

    Determines UI presentation urgency.
    """
    NONE = "none"          # No conflicts, all validated
    CAUTION = "caution"    # Minor discrepancies, informational
    WARNING = "warning"    # Significant conflicts need review
    CRITICAL = "critical"  # Safety-critical conflicts, immediate attention


# =============================================================================
# External Search Results
# =============================================================================

@dataclass
class ExternalSearchResult:
    """
    Single result from external search (Perplexity/Gemini).

    Attributes:
        content: Extracted text content
        source_url: URL for citation
        source_title: Page/article title
        relevance_score: 0.0-1.0 relevance to query
        provider: Which service provided this result
        retrieved_at: Timestamp of retrieval
        snippet: Short preview for UI display
        metadata: Additional provider-specific data
    """
    content: str
    source_url: str
    source_title: str
    relevance_score: float
    provider: str  # "perplexity", "gemini", "google"
    retrieved_at: datetime
    snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields."""
        # Ensure relevance score is in valid range
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))

        # Ensure snippet is not too long
        if len(self.snippet) > 300:
            self.snippet = self.snippet[:297] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content': self.content,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'relevance_score': self.relevance_score,
            'provider': self.provider,
            'retrieved_at': self.retrieved_at.isoformat(),
            'snippet': self.snippet,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExternalSearchResult':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('retrieved_at'), str):
            data['retrieved_at'] = datetime.fromisoformat(data['retrieved_at'])
        return cls(**data)


@dataclass
class ExternalCitation:
    """
    Citation reference for external source.

    Distinguished from internal citations by:
    - URL-based reference instead of chunk_id
    - Provider attribution
    - Web-specific metadata (publication date, domain)
    """
    index: str                      # "W1", "W2", etc.
    source_url: str
    source_title: str
    snippet: str
    provider: str
    publication_date: Optional[datetime] = None
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'snippet': self.snippet,
            'provider': self.provider,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'domain': self.domain,
        }


# =============================================================================
# Gap Analysis Models
# =============================================================================

@dataclass
class Conflict:
    """
    Represents a conflict between internal and external sources.

    Used to alert users when their curated content differs from
    current external information.
    """
    internal_claim: str           # What internal source says
    external_claim: str           # What external source says
    internal_source: str          # Citation for internal
    external_source: str          # Citation for external (URL)
    severity: ConflictSeverity
    topic: str                    # What the conflict is about
    recommendation: Optional[str] = None  # Suggested resolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'internal_claim': self.internal_claim,
            'external_claim': self.external_claim,
            'internal_source': self.internal_source,
            'external_source': self.external_source,
            'severity': self.severity.value,
            'topic': self.topic,
            'recommendation': self.recommendation,
        }


@dataclass
class GapReport:
    """
    Result of the Gap Analyzer "Judge" comparing internal vs external sources.

    The Gap Analyzer acts as an impartial judge that compares:
    - GROUND TRUTH: User's internal database (what they currently believe/practice)
    - WORLD TRUTH: Current external sources (current standard of care)

    This comparison identifies:
    - Agreements: Where Ground Truth is validated by World Truth ✓
    - Conflicts: Where Ground Truth contradicts World Truth ⚠️
    - Gaps: Information in World Truth missing from Ground Truth 📋
    - Unique Internal: Specialized Ground Truth not in World Truth 💎

    The alert_level provides an overall urgency indicator for UI presentation.
    """
    agreements: List[str] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    unique_internal: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Overall confidence in synthesis
    analysis_time_ms: int = 0
    alert_level: str = "none"  # Overall alert level: none, caution, warning, critical

    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    @property
    def has_critical_conflicts(self) -> bool:
        """Check if any critical conflicts exist."""
        return any(
            c.severity == ConflictSeverity.CRITICAL
            for c in self.conflicts
        )

    @property
    def has_high_severity_conflicts(self) -> bool:
        """Check if HIGH or CRITICAL conflicts exist."""
        return any(
            c.severity in (ConflictSeverity.HIGH, ConflictSeverity.CRITICAL)
            for c in self.conflicts
        )

    @property
    def conflict_count(self) -> int:
        """Total number of conflicts."""
        return len(self.conflicts)

    @property
    def gap_count(self) -> int:
        """Total number of gaps identified."""
        return len(self.gaps)

    @property
    def highest_severity(self) -> Optional[ConflictSeverity]:
        """Get the highest severity level among conflicts."""
        if not self.conflicts:
            return None

        severity_order = [
            ConflictSeverity.LOW,
            ConflictSeverity.MEDIUM,
            ConflictSeverity.HIGH,
            ConflictSeverity.CRITICAL
        ]

        max_severity = ConflictSeverity.LOW
        for c in self.conflicts:
            if severity_order.index(c.severity) > severity_order.index(max_severity):
                max_severity = c.severity

        return max_severity

    @property
    def computed_alert_level(self) -> str:
        """Compute alert level based on conflicts."""
        if not self.conflicts:
            return "none"

        highest = self.highest_severity
        if highest == ConflictSeverity.CRITICAL:
            return "critical"
        elif highest == ConflictSeverity.HIGH:
            return "warning"
        elif highest == ConflictSeverity.MEDIUM:
            return "caution"
        else:
            return "caution"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agreements': self.agreements,
            'conflicts': [c.to_dict() for c in self.conflicts],
            'gaps': self.gaps,
            'unique_internal': self.unique_internal,
            'confidence': self.confidence,
            'analysis_time_ms': self.analysis_time_ms,
            'alert_level': self.alert_level or self.computed_alert_level,
            'has_conflicts': self.has_conflicts,
            'has_critical_conflicts': self.has_critical_conflicts,
            'has_high_severity_conflicts': self.has_high_severity_conflicts,
            'conflict_count': self.conflict_count,
            'gap_count': self.gap_count,
            'highest_severity': self.highest_severity.value if self.highest_severity else None,
        }

    @classmethod
    def empty(cls) -> 'GapReport':
        """Create empty gap report (no analysis performed)."""
        return cls()


# =============================================================================
# Enriched Context
# =============================================================================

@dataclass
class EnrichedContext:
    """
    Combined internal + external context for unified RAG.

    This is the primary output of the ResearchEnricher, containing
    everything needed to generate a comprehensive response.

    Attributes:
        internal_chunks: Search results from internal database
        external_results: Results from external search
        gap_report: Analysis of agreements/conflicts/gaps
        synthesis_prompt: Pre-built prompt for Claude
        total_tokens: Estimated token count
        internal_ratio: Percentage from internal sources (0.0-1.0)
        mode_used: Which search mode was applied
        search_time_ms: Total search time
    """
    internal_chunks: List[Any]  # List[SearchResult] from retrieval
    external_results: List[ExternalSearchResult]
    gap_report: Optional[GapReport]
    synthesis_prompt: str
    total_tokens: int
    internal_ratio: float
    mode_used: SearchMode
    search_time_ms: int = 0

    @property
    def has_internal(self) -> bool:
        """Check if internal results are present."""
        return len(self.internal_chunks) > 0

    @property
    def has_external(self) -> bool:
        """Check if external results are present."""
        return len(self.external_results) > 0

    @property
    def is_hybrid(self) -> bool:
        """Check if both internal and external results present."""
        return self.has_internal and self.has_external

    @property
    def internal_count(self) -> int:
        """Number of internal chunks."""
        return len(self.internal_chunks)

    @property
    def external_count(self) -> int:
        """Number of external results."""
        return len(self.external_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'internal_count': self.internal_count,
            'external_count': self.external_count,
            'gap_report': self.gap_report.to_dict() if self.gap_report else None,
            'total_tokens': self.total_tokens,
            'internal_ratio': self.internal_ratio,
            'mode_used': self.mode_used.value,
            'search_time_ms': self.search_time_ms,
            'has_internal': self.has_internal,
            'has_external': self.has_external,
            'is_hybrid': self.is_hybrid,
        }


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class ExternalSearchConfig:
    """Configuration for external search clients."""

    # Perplexity settings
    perplexity_model: str = "sonar-pro"
    perplexity_max_results: int = 5
    perplexity_recency_filter: Optional[str] = "month"  # week, month, year, None
    perplexity_timeout: int = 30

    # Gemini settings
    gemini_model: str = "gemini-2.5-pro"
    gemini_max_tokens: int = 4096
    gemini_temperature: float = 0.3
    gemini_timeout: int = 60
    gemini_enable_grounding: bool = True

    # Rate limiting
    max_requests_per_minute: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class EnricherConfig:
    """Configuration for the research enricher."""

    # Search behavior
    default_mode: SearchMode = SearchMode.HYBRID
    enable_gap_analysis: bool = True
    gap_analysis_model: str = "claude-sonnet-4-20250514"

    # Token budgets
    max_internal_tokens: int = 6000
    max_external_tokens: int = 2000
    max_total_tokens: int = 8000

    # Ranking
    internal_weight: float = 0.7  # Weight for internal sources in ranking
    external_weight: float = 0.3  # Weight for external sources
    authority_threshold: float = 0.5  # Min authority score for internal
    relevance_threshold: float = 0.3  # Min relevance for external

    # Conflict detection
    conflict_detection_enabled: bool = True
    conflict_sensitivity: float = 0.7  # Higher = more sensitive

    # Caching
    cache_external_results: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'default_mode': self.default_mode.value,
            'enable_gap_analysis': self.enable_gap_analysis,
            'max_internal_tokens': self.max_internal_tokens,
            'max_external_tokens': self.max_external_tokens,
            'max_total_tokens': self.max_total_tokens,
            'internal_weight': self.internal_weight,
            'external_weight': self.external_weight,
        }


# =============================================================================
# API Response Models (for Pydantic compatibility)
# =============================================================================

@dataclass
class ModeInfo:
    """Information about a search mode for API response."""
    id: str
    name: str
    description: str
    icon: str
    cost: str
    latency: str
    requires_api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'cost': self.cost,
            'latency': self.latency,
            'requires_api_key': self.requires_api_key,
        }


# Predefined mode information
SEARCH_MODES_INFO = [
    ModeInfo(
        id="standard",
        name="Internal Only",
        description="Fast search of your curated neurosurgical database",
        icon="database",
        cost="$0",
        latency="~2s",
        requires_api_key=None
    ),
    ModeInfo(
        id="hybrid",
        name="Smart Hybrid",
        description="Your database + live web validation for current guidelines",
        icon="globe",
        cost="~$0.01",
        latency="~4s",
        requires_api_key="PERPLEXITY_API_KEY"
    ),
    ModeInfo(
        id="deep_research",
        name="Deep Research",
        description="Complex reasoning with Gemini 2.5 Pro and Google Search",
        icon="brain",
        cost="~$0.05",
        latency="~10s",
        requires_api_key="GEMINI_API_KEY"
    ),
    ModeInfo(
        id="external",
        name="External Only",
        description="Web search only, ignores internal database",
        icon="cloud",
        cost="~$0.01",
        latency="~3s",
        requires_api_key="PERPLEXITY_API_KEY"
    ),
]


def get_available_modes(
    has_perplexity: bool = False,
    has_gemini: bool = False
) -> List[ModeInfo]:
    """
    Get list of available modes based on configured API keys.

    Args:
        has_perplexity: Whether Perplexity API is configured
        has_gemini: Whether Gemini API is configured

    Returns:
        List of available ModeInfo objects
    """
    available = [SEARCH_MODES_INFO[0]]  # Standard always available

    if has_perplexity:
        available.append(SEARCH_MODES_INFO[1])  # Hybrid
        available.append(SEARCH_MODES_INFO[3])  # External only

    if has_gemini:
        available.append(SEARCH_MODES_INFO[2])  # Deep research

    return available
