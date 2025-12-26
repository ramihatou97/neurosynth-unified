"""Shared enumerations for Phase 1 and Phase 2."""

from enum import Enum


class SearchMode(Enum):
    """Search mode for hybrid retrieval."""
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class RerankerMode(Enum):
    """Reranking strategy."""
    NONE = "none"
    SIMPLE = "simple"
    CROSS_ENCODER = "cross_encoder"
    COLBERT = "colbert"


class TriageLevel(Enum):
    """Visual triage decision level."""
    SKIP = 1
    LOW_PRIORITY = 2
    NORMAL = 3
    HIGH_PRIORITY = 4
    CRITICAL = 5
