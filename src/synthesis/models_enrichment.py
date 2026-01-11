"""
NeuroSynth - Enrichment Data Models
===================================

Strict Pydantic schemas for gap analysis, conflict detection, and
research enrichment results.

These models enforce structure on LLM outputs, enabling robust parsing
and type-safe data handling throughout the V3 pipeline.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


# =============================================================================
# GAP ANALYSIS MODELS
# =============================================================================

class GapType(str, Enum):
    """Types of knowledge gaps that can be identified."""
    MISSING_DATA = "missing_data"           # External has info Internal lacks
    OUTDATED = "outdated"                   # Internal data is stale
    INCOMPLETE = "incomplete"               # Internal partial, External complete
    RECENT_DEVELOPMENTS = "recent_developments"  # New research/techniques
    CLINICAL_TRIALS = "clinical_trials"     # Active trials
    GUIDELINES = "guidelines"               # Updated protocols/guidelines


class GapPriority(str, Enum):
    """Priority levels for filling gaps."""
    CRITICAL = "critical"   # Must fill - safety implications
    HIGH = "high"           # Should fill - significant knowledge gap
    MEDIUM = "medium"       # Could fill - improves completeness
    LOW = "low"             # Optional - nice to have


class GapItem(BaseModel):
    """A specific gap identified between internal and external sources."""

    gap_type: Literal[
        "missing_data",
        "outdated",
        "incomplete",
        "recent_developments",
        "clinical_trials",
        "guidelines"
    ] = Field(..., description="Category of missing information")

    priority: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Urgency of filling this gap"
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Certainty that this is a genuine gap (0.0-1.0)"
    )

    description: str = Field(
        ..., min_length=10, max_length=500,
        description="Specific query to search for missing info"
    )

    section: Optional[str] = Field(
        default=None,
        description="Which synthesis section this gap affects"
    )

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """Ensure description is specific enough to be actionable."""
        if len(v.split()) < 3:
            raise ValueError("Description must be at least 3 words")
        return v


class GapAnalysisResult(BaseModel):
    """Complete result of gap analysis between internal and external sources."""

    summary: str = Field(
        ..., min_length=20,
        description="Brief overview of internal coverage quality"
    )

    gaps: List[GapItem] = Field(
        default_factory=list,
        description="List of identified gaps"
    )

    internal_coverage_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How well internal sources cover the topic (0.0-1.0)"
    )

    analysis_timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="When analysis was performed"
    )

    def has_critical_gaps(self) -> bool:
        """Check if any critical/high priority gaps exist with high confidence."""
        return any(
            g.priority in ["critical", "high"] and g.confidence > 0.7
            for g in self.gaps
        )

    def get_gaps_by_priority(self, priority: str) -> List[GapItem]:
        """Get gaps of a specific priority."""
        return [g for g in self.gaps if g.priority == priority]

    def get_fillable_gaps(self, min_confidence: float = 0.7) -> List[GapItem]:
        """Get gaps worth filling based on confidence threshold."""
        return [
            g for g in self.gaps
            if g.confidence >= min_confidence and g.priority in ["critical", "high"]
        ]


# =============================================================================
# CONFLICT DETECTION MODELS
# =============================================================================

class ConflictCategory(str, Enum):
    """Categories of conflicts between sources."""
    TEMPORAL = "temporal"               # Time-sensitive (guidelines, trials)
    ESTABLISHED_FACT = "established"    # Core anatomy/physiology facts
    QUANTITATIVE = "quantitative"       # Numerical disagreements
    APPROACH = "approach"               # Different valid techniques
    TERMINOLOGY = "terminology"         # Different names for same thing
    RECOMMENDATION = "recommendation"   # Different clinical recommendations


class ResolutionStrategy(str, Enum):
    """How to resolve different conflict types."""
    PREFER_INTERNAL = "prefer_internal"     # Internal corpus wins
    PREFER_EXTERNAL = "prefer_external"     # External (recent) wins
    NOTE_BOTH = "note_both"                 # Present both with comparison
    SYNTHESIZE = "synthesize"               # LLM synthesizes best answer
    FLAG_FOR_REVIEW = "flag_for_review"     # Highlight for human review


class ExtractedFact(BaseModel):
    """A factual claim extracted from source material."""

    claim: str = Field(..., description="The factual claim")
    value: Optional[str] = Field(default=None, description="Numeric/specific value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    source: str = Field(default="", description="Source identifier")
    source_type: Literal["internal", "external"] = Field(default="internal")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    context: str = Field(default="", description="Surrounding context")
    is_temporal: bool = Field(default=False, description="Time-sensitive info?")
    is_recommendation: bool = Field(default=False, description="Clinical recommendation?")


class DetectedConflict(BaseModel):
    """A conflict detected between internal and external sources."""

    category: Literal[
        "temporal", "established", "quantitative",
        "approach", "terminology", "recommendation"
    ] = Field(..., description="Type of conflict")

    description: str = Field(..., description="Brief description of the conflict")

    internal_claim: str = Field(..., description="What internal source says")
    external_claim: str = Field(..., description="What external source says")

    internal_source: Optional[str] = Field(default=None)
    external_source: Optional[str] = Field(default=None)

    resolution_strategy: Literal[
        "prefer_internal", "prefer_external",
        "note_both", "synthesize", "flag_for_review"
    ] = Field(default="note_both")

    resolved_content: str = Field(default="", description="Final resolved text")
    resolution_note: str = Field(default="", description="Explanation of resolution")

    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class MergeResult(BaseModel):
    """Result of conflict-aware merge operation."""

    topic: str
    resolved_content: str = Field(..., description="Final merged content")
    merge_strategy_used: str = Field(..., description="Description of merge approach")

    conflicts: List[DetectedConflict] = Field(default_factory=list)
    conflict_count: int = Field(default=0)
    high_severity_conflicts: int = Field(default=0)

    internal_facts_count: int = Field(default=0)
    external_facts_count: int = Field(default=0)
    facts_from_internal: int = Field(default=0)
    facts_from_external: int = Field(default=0)

    merge_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    requires_review: bool = Field(default=False)
    review_notes: List[str] = Field(default_factory=list)

    merge_time_ms: int = Field(default=0)


# =============================================================================
# ADVERSARIAL REVIEW MODELS
# =============================================================================

class Severity(str, Enum):
    """Severity levels for controversy warnings."""
    HIGH = "HIGH"       # Draft recommends action source forbids
    MEDIUM = "MEDIUM"   # Nuance error, different emphasis
    LOW = "LOW"         # Minor discrepancy, stylistic


class ControversyWarning(BaseModel):
    """Warning from adversarial review about contradictions with authoritative sources."""

    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Clinical risk level"
    )

    topic: str = Field(
        ..., min_length=3,
        description="The specific medical topic in question"
    )

    draft_claim: str = Field(
        ..., min_length=5,
        description="What the generated text currently says"
    )

    contradicting_source: str = Field(
        ..., description="Citation of the authoritative source"
    )

    source_quote: str = Field(
        ..., min_length=10,
        description="Exact quote proving the contradiction"
    )

    recommendation: str = Field(
        ..., min_length=10,
        description="Suggested correction"
    )

    section: Optional[str] = Field(
        default=None,
        description="Which section this warning applies to"
    )


class ReviewResult(BaseModel):
    """Result of adversarial section review."""

    has_issues: bool = Field(..., description="Whether any issues were found")
    warnings: List[ControversyWarning] = Field(default_factory=list)
    section_reviewed: Optional[str] = Field(default=None)
    review_time_ms: Optional[int] = Field(default=None)

    def get_high_severity(self) -> List[ControversyWarning]:
        """Get only HIGH severity warnings."""
        return [w for w in self.warnings if w.severity == "HIGH"]

    def get_critical_count(self) -> int:
        """Count HIGH severity warnings."""
        return len(self.get_high_severity())


# =============================================================================
# ENRICHMENT RESULT MODELS
# =============================================================================

class ExternalSource(BaseModel):
    """Metadata about an external source used in enrichment."""

    title: str
    url: Optional[str] = None
    date: Optional[str] = None
    source_type: Literal["web", "pubmed", "guidelines", "trial", "news"] = "web"
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class EnrichmentResult(BaseModel):
    """Result of enriching a section with external content."""

    section_name: str
    was_enriched: bool = Field(default=False)

    # Gap analysis
    gaps_found: int = Field(default=0)
    gaps_filled: int = Field(default=0)
    gap_details: Optional[GapAnalysisResult] = None

    # External content
    external_content_added: str = Field(default="")
    external_sources: List[ExternalSource] = Field(default_factory=list)

    # Timing
    enrichment_time_ms: int = Field(default=0)

    # Quality
    enrichment_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    required_manual_review: bool = Field(default=False)


class V3SynthesisMetadata(BaseModel):
    """
    Metadata for V3 enhanced synthesis results.

    Captures all enrichment, conflict, and validation information.
    """

    # Mode
    mode: Literal["standard", "hybrid", "deep_research"] = Field(default="standard")

    # Enrichment summary
    enrichment_used: bool = Field(default=False)
    total_external_sources: int = Field(default=0)
    external_citations: List[str] = Field(default_factory=list)

    # Gap analysis
    gaps_summary: Dict[str, Any] = Field(default_factory=dict)

    # Conflict detection
    conflict_count: int = Field(default=0)
    high_severity_conflicts: int = Field(default=0)
    conflicts: List[DetectedConflict] = Field(default_factory=list)

    # Adversarial review
    controversy_warnings: List[ControversyWarning] = Field(default_factory=list)

    # Validation
    validation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    requires_review: bool = Field(default=False)
    review_reasons: List[str] = Field(default_factory=list)

    # Timing
    total_enrichment_time_ms: int = Field(default=0)

    def to_frontend_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for frontend consumption."""
        return {
            "mode": self.mode,
            "enrichmentUsed": self.enrichment_used,
            "externalSourceCount": self.total_external_sources,
            "conflictCount": self.conflict_count,
            "highSeverityConflicts": self.high_severity_conflicts,
            "controversyCount": len(self.controversy_warnings),
            "validationScore": self.validation_score,
            "requiresReview": self.requires_review,
        }
