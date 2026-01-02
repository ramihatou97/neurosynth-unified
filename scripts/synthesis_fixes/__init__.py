"""
NeuroSynth Synthesis Fixes Module
=================================

Drop-in replacements for synthesis engine classes with alignment fixes:
1. EnhancedContextAdapter - Type-based routing, CUI preservation, caption embeddings
2. EnhancedFigureResolver - Semantic matching via caption embeddings
3. ContentValidator - CUI-based hallucination detection

Usage:
    from synthesis_fixes import EnhancedContextAdapter, EnhancedFigureResolver, ContentValidator
"""

from .enhanced_context_adapter import (
    EnhancedContextAdapter,
    EnhancedFigureResolver,
    ContentValidator,
    TemplateType,
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
    CHUNK_TYPE_SECTION_MAP,
    AuthoritySource,
    AUTHORITY_SCORES,
)

__all__ = [
    "EnhancedContextAdapter",
    "EnhancedFigureResolver",
    "ContentValidator",
    "TemplateType",
    "TEMPLATE_SECTIONS",
    "TEMPLATE_REQUIREMENTS",
    "CHUNK_TYPE_SECTION_MAP",
    "AuthoritySource",
    "AUTHORITY_SCORES",
]
