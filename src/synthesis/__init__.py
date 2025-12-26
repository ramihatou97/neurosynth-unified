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
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
)

__all__ = [
    "SynthesisEngine",
    "TemplateType",
    "SynthesisResult",
    "SynthesisSection",
    "FigureRequest",
    "RateLimiter",
    "ContextAdapter",
    "FigureResolver",
    "AuthoritySource",
    "TEMPLATE_SECTIONS",
    "TEMPLATE_REQUIREMENTS",
]
