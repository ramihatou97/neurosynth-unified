"""
Gap Detection Models
====================

Data models for the 14-stage neurosurgical gap detection system.

Includes:
- Extended GapType enum (14 types)
- GapPriority with safety-critical override
- Gap dataclass with full metadata
- GapReport aggregation
- GapFillStrategy enum
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class GapType(Enum):
    """
    Extended gap types for neurosurgical content analysis.

    Original 5 types:
    - MISSING: Concept entirely absent from content
    - THIN_COVERAGE: Concept mentioned but insufficient depth
    - TEMPORAL: Evidence is outdated (>3 years for guidelines)
    - STRUCTURAL: Missing expected section for template type
    - USER_DEMAND: Frequently asked but poorly answered

    Neurosurgically-Critical Extensions (9 new types):
    - DANGER_ZONE: Missing safety-critical anatomy (AUTO-CRITICAL)
    - PROCEDURAL_STEP: Missing key operative step
    - MEASUREMENT: Missing quantitative thresholds (AUTO-CRITICAL)
    - DECISION_POINT: Missing clinical decision logic
    - BAILOUT: Missing complication management (AUTO-CRITICAL)
    - INSTRUMENT: Missing required surgical equipment
    - IMAGING: Missing required imaging modalities
    - VISUAL: Missing figures/illustrations
    - EVIDENCE_LEVEL: Missing landmark trial evidence
    """

    # Original 5 types
    MISSING = "missing"
    THIN_COVERAGE = "thin"
    TEMPORAL = "outdated"
    STRUCTURAL = "structural"
    USER_DEMAND = "user_demand"

    # Neurosurgically-Critical Extensions
    DANGER_ZONE = "danger_zone"
    PROCEDURAL_STEP = "procedural"
    MEASUREMENT = "measurement"
    DECISION_POINT = "decision"
    BAILOUT = "bailout"
    INSTRUMENT = "instrument"
    IMAGING = "imaging"
    VISUAL = "visual"
    EVIDENCE_LEVEL = "evidence"


class GapPriority(Enum):
    """
    Gap priority levels with thresholds.

    CRITICAL (score >= 80 OR safety-critical type):
        - Must be addressed before synthesis is considered complete
        - Includes all DANGER_ZONE, BAILOUT, and MEASUREMENT gaps

    HIGH (score 60-79):
        - Should be addressed for comprehensive coverage
        - May trigger external research fetch

    MEDIUM (score 35-59):
        - Nice to have but not blocking
        - External fetch only if strategy allows

    LOW (score < 35):
        - Minor gaps, informational only
        - Never triggers external fetch
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapFillStrategy(Enum):
    """
    User-selectable strategy for gap filling.

    NONE: Don't fetch external content, report gaps only
    HIGH_PRIORITY_ONLY: Fetch only for HIGH and CRITICAL gaps
    ALL_WITH_FALLBACK: Try internal first, then external for all gaps
    ALWAYS_EXTERNAL: Always fetch external for comparison/enrichment
    """

    NONE = "none"
    HIGH_PRIORITY_ONLY = "high"
    ALL_WITH_FALLBACK = "fallback"
    ALWAYS_EXTERNAL = "always"


class TemplateType(Enum):
    """Template types for synthesis."""

    PROCEDURAL = "PROCEDURAL"
    DISORDER = "DISORDER"
    ANATOMY = "ANATOMY"
    CONCEPT = "CONCEPT"
    ENCYCLOPEDIA = "ENCYCLOPEDIA"


# Safety-critical gap types that are ALWAYS escalated to CRITICAL priority
SAFETY_CRITICAL_TYPES: Set[GapType] = {
    GapType.DANGER_ZONE,   # Missing danger zone = potential patient harm
    GapType.BAILOUT,       # Missing bailout = no rescue plan for complications
    GapType.MEASUREMENT,   # Wrong measurement = potentially fatal (e.g., ICP threshold)
}


# Priority score thresholds
PRIORITY_THRESHOLDS = {
    GapPriority.CRITICAL: 80,
    GapPriority.HIGH: 60,
    GapPriority.MEDIUM: 35,
    GapPriority.LOW: 0,
}


@dataclass
class Gap:
    """
    Represents a detected knowledge gap.

    Attributes:
        gap_id: Unique identifier for this gap
        gap_type: Type of gap (from GapType enum)
        topic: The concept/topic that has a gap
        priority: Computed priority level
        priority_score: Raw numeric score (0-100)
        current_coverage: What the content currently says (if anything)
        recommended_coverage: What should be included
        justification: Dict with scoring breakdown and detection metadata
        target_section: Which synthesis section this gap belongs to
        external_query: Suggested query for external research
        auto_fill_available: Whether this gap can be auto-filled
        detection_stage: Which of the 14 stages detected this gap
        safety_critical: Whether this is a safety-critical gap
    """

    gap_type: GapType
    topic: str
    priority_score: float = 0.0
    current_coverage: str = ""
    recommended_coverage: str = ""
    justification: Dict[str, Any] = field(default_factory=dict)
    target_section: Optional[str] = None
    external_query: Optional[str] = None
    auto_fill_available: bool = True
    detection_stage: int = 0
    safety_critical: bool = False
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        """Set safety_critical flag and adjust priority for safety-critical types."""
        if self.gap_type in SAFETY_CRITICAL_TYPES:
            self.safety_critical = True
            self.priority_score = max(self.priority_score, 100.0)

    @property
    def priority(self) -> GapPriority:
        """Compute priority from score with safety-critical override."""
        if self.safety_critical:
            return GapPriority.CRITICAL

        for priority, threshold in sorted(
            PRIORITY_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if self.priority_score >= threshold:
                return priority

        return GapPriority.LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "topic": self.topic,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "current_coverage": self.current_coverage,
            "recommended_coverage": self.recommended_coverage,
            "justification": self.justification,
            "target_section": self.target_section,
            "external_query": self.external_query,
            "auto_fill_available": self.auto_fill_available,
            "detection_stage": self.detection_stage,
            "safety_critical": self.safety_critical,
        }


@dataclass
class GapFillResult:
    """Result of attempting to fill a gap."""

    gap_id: str
    gap_type: GapType
    topic: str
    fill_successful: bool
    fill_source: str  # 'internal', 'external', 'both', 'failed'
    filled_content: str = ""
    external_sources: List[Dict[str, str]] = field(default_factory=list)
    fill_duration_ms: int = 0
    error_message: Optional[str] = None


@dataclass
class GapReport:
    """
    Aggregated gap analysis report.

    Contains all detected gaps with summary statistics and flags.
    """

    topic: str
    template_type: TemplateType
    subspecialty: str = ""
    gaps: List[Gap] = field(default_factory=list)
    analysis_duration_ms: int = 0
    source_document_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_gaps(self) -> int:
        """Total number of gaps detected."""
        return len(self.gaps)

    @property
    def critical_gaps(self) -> List[Gap]:
        """Gaps with CRITICAL priority."""
        return [g for g in self.gaps if g.priority == GapPriority.CRITICAL]

    @property
    def high_priority_gaps(self) -> List[Gap]:
        """Gaps with HIGH or CRITICAL priority."""
        return [g for g in self.gaps if g.priority in (GapPriority.CRITICAL, GapPriority.HIGH)]

    @property
    def safety_flags(self) -> List[str]:
        """Human-readable safety flags."""
        flags = []
        for gap in self.gaps:
            if gap.safety_critical:
                flags.append(f"{gap.gap_type.value.upper()}: {gap.topic}")
        return flags

    @property
    def gaps_by_type(self) -> Dict[str, int]:
        """Count of gaps by type."""
        counts: Dict[str, int] = {}
        for gap in self.gaps:
            key = gap.gap_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def gaps_by_priority(self) -> Dict[str, int]:
        """Count of gaps by priority."""
        counts: Dict[str, int] = {}
        for gap in self.gaps:
            key = gap.priority.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def requires_expert_review(self) -> bool:
        """True if any CRITICAL safety gaps exist."""
        return any(g.safety_critical for g in self.gaps)

    @property
    def source_hash(self) -> str:
        """Hash of source document IDs for cache invalidation."""
        sorted_ids = sorted(self.source_document_ids)
        return hashlib.md5("".join(sorted_ids).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "template_type": self.template_type.value,
            "subspecialty": self.subspecialty,
            "total_gaps": self.total_gaps,
            "critical_gaps": len(self.critical_gaps),
            "safety_flags": self.safety_flags,
            "gaps_by_type": self.gaps_by_type,
            "gaps_by_priority": self.gaps_by_priority,
            "requires_expert_review": self.requires_expert_review,
            "gaps": [g.to_dict() for g in self.gaps],
            "analysis_duration_ms": self.analysis_duration_ms,
            "source_hash": self.source_hash,
            "created_at": self.created_at.isoformat(),
        }

    def get_top_gaps(self, n: int = 10) -> List[Gap]:
        """Get top N gaps by priority score."""
        return sorted(self.gaps, key=lambda g: g.priority_score, reverse=True)[:n]

    def filter_by_type(self, gap_type: GapType) -> List[Gap]:
        """Filter gaps by type."""
        return [g for g in self.gaps if g.gap_type == gap_type]

    def filter_by_section(self, section: str) -> List[Gap]:
        """Filter gaps by target section."""
        return [g for g in self.gaps if g.target_section == section]


@dataclass
class GapSummary:
    """Lightweight gap summary for API responses."""

    gap_id: str
    gap_type: str
    topic: str
    priority: str
    priority_score: float
    safety_critical: bool
    recommended_coverage: str

    @classmethod
    def from_gap(cls, gap: Gap) -> "GapSummary":
        """Create summary from full Gap object."""
        return cls(
            gap_id=gap.gap_id,
            gap_type=gap.gap_type.value,
            topic=gap.topic,
            priority=gap.priority.value,
            priority_score=gap.priority_score,
            safety_critical=gap.safety_critical,
            recommended_coverage=gap.recommended_coverage[:200] if gap.recommended_coverage else "",
        )
