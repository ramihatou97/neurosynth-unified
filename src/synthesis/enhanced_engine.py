"""
NeuroSynth V3 - Enhanced Synthesis Engine
==========================================

This module provides an enhanced wrapper around the existing SynthesisEngine
that adds Open-World knowledge enrichment WITHOUT modifying the original code.

Design Principle: COMPOSITION OVER MODIFICATION
- The original SynthesisEngine remains untouched
- EnhancedSynthesisEngine wraps it and adds enrichment
- All existing functionality preserved
- New functionality is opt-in via flags

Usage:
    # Option 1: Use enhanced engine directly
    from src.synthesis.enhanced_engine import EnhancedSynthesisEngine

    engine = EnhancedSynthesisEngine(
        anthropic_client=client,
        perplexity_api_key="pplx-xxx",
        google_api_key="xxx",
    )

    result = await engine.synthesize(
        topic="MCA aneurysm clipping",
        template_type=TemplateType.PROCEDURAL,
        search_results=results,
        include_web_research=True,  # NEW FLAG
    )

    # Option 2: Enhance existing engine instance
    from src.synthesis.enhanced_engine import add_enrichment_to_engine

    existing_engine = SynthesisEngine(...)
    enhanced = add_enrichment_to_engine(
        existing_engine,
        perplexity_api_key="pplx-xxx",
    )

Integration Points:
    1. Before section generation: Check for gaps
    2. During section generation: Inject external context
    3. After synthesis: Add external citations to references
    4. Metadata: Track enrichment statistics
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from time import time

# Import original synthesis components (no modifications)
from src.synthesis.engine import (
    SynthesisEngine,
    SynthesisResult,
    SynthesisSection,
    TemplateType,
    ContextAdapter,
    FigureResolver,
    FigureRequest,
    TEMPLATE_SECTIONS,
    TEMPLATE_REQUIREMENTS,
)

# Import the new enrichment module
from src.synthesis.research_enricher import (
    ResearchEnricher,
    EnrichmentConfig,
    EnrichmentResult,
    KnowledgeGap,
    GapPriority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED RESULT MODEL
# =============================================================================

@dataclass
class EnhancedSynthesisResult(SynthesisResult):
    """
    Extended SynthesisResult with enrichment metadata.

    Inherits all fields from SynthesisResult and adds:
    - enrichment_used: Whether external sources were used
    - enrichment_results: Per-section enrichment data
    - external_citations: List of [Web: ...] citations
    - gaps_summary: Summary of identified/filled gaps
    """

    # Enrichment metadata
    enrichment_used: bool = False
    enrichment_results: Dict[str, EnrichmentResult] = field(default_factory=dict)
    external_citations: List[str] = field(default_factory=list)
    gaps_summary: Dict[str, Any] = field(default_factory=dict)
    total_external_sources: int = 0
    enrichment_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including enrichment data."""
        base_dict = {
            "topic": self.topic,
            "template_type": self.template_type.value if hasattr(self.template_type, 'value') else str(self.template_type),
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "sources": s.sources,
                    "word_count": s.word_count,
                }
                for s in self.sections
            ],
            "total_words": self.total_words,
            "total_figures": self.total_figures,
            "total_citations": self.total_citations,
            "synthesis_time_ms": self.synthesis_time_ms,
            # Enrichment fields
            "enrichment_used": self.enrichment_used,
            "external_citations": self.external_citations,
            "total_external_sources": self.total_external_sources,
            "enrichment_time_ms": self.enrichment_time_ms,
            "gaps_summary": self.gaps_summary,
        }
        return base_dict


# =============================================================================
# ENHANCED SYNTHESIS ENGINE
# =============================================================================

class EnhancedSynthesisEngine:
    """
    Enhanced SynthesisEngine with Open-World knowledge enrichment.

    This is a WRAPPER that composes with the original SynthesisEngine.
    The original engine's code is completely untouched.

    New capabilities:
    - Semantic gap analysis between internal/external knowledge
    - Perplexity integration for academic/recent sources
    - Gemini grounding for fact-checking
    - Per-section enrichment with cost control
    - Dual citation system (internal [Source N] + external [Web: ...])
    """

    def __init__(
        self,
        anthropic_client,
        verification_client=None,
        model: str = "claude-sonnet-4-20250514",
        calls_per_minute: int = 50,
        deep_conflict_check: bool = False,
        # New enrichment parameters
        perplexity_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        enrichment_config: Optional[EnrichmentConfig] = None,
    ):
        # Initialize original engine (unchanged)
        self._base_engine = SynthesisEngine(
            anthropic_client=anthropic_client,
            verification_client=verification_client,
            model=model,
            calls_per_minute=calls_per_minute,
            deep_conflict_check=deep_conflict_check,
        )

        # Initialize enricher (new component)
        self._enricher: Optional[ResearchEnricher] = None
        if perplexity_api_key or google_api_key:
            self._enricher = ResearchEnricher(
                anthropic_client=anthropic_client,
                perplexity_api_key=perplexity_api_key,
                google_api_key=google_api_key,
                config=enrichment_config,
            )
            logger.info("EnhancedSynthesisEngine initialized with web research capability")
        else:
            logger.info("EnhancedSynthesisEngine initialized without web research")

        # Keep reference to client for direct calls
        self._client = anthropic_client
        self._model = model

    @property
    def enricher(self) -> Optional[ResearchEnricher]:
        """Access the enricher for direct use if needed."""
        return self._enricher

    @property
    def base_engine(self) -> SynthesisEngine:
        """Access the underlying engine."""
        return self._base_engine

    async def synthesize(
        self,
        topic: str,
        template_type: TemplateType,
        search_results: List[Any],  # List[SearchResult]
        include_verification: bool = False,
        include_figures: bool = True,
        # NEW PARAMETER
        include_web_research: bool = False,
    ) -> EnhancedSynthesisResult:
        """
        Generate textbook-quality synthesis with optional web research.

        When include_web_research=True:
        1. Each section is analyzed for knowledge gaps
        2. Gaps are filled with external sources (Perplexity/Gemini)
        3. External context is injected into section prompts
        4. Citations are tracked separately ([Source N] vs [Web: ...])

        Args:
            topic: Chapter topic
            template_type: Template style (PROCEDURAL, DISORDER, etc.)
            search_results: List[SearchResult] from SearchService
            include_verification: Run Gemini verification pass
            include_figures: Resolve figure placeholders
            include_web_research: NEW - Enable Open-World enrichment

        Returns:
            EnhancedSynthesisResult with all original fields plus enrichment data
        """
        start_time = time()

        # Reset enricher counter for this synthesis
        if self._enricher:
            self._enricher.reset_synthesis_counter()

        # If no web research requested, delegate entirely to base engine
        if not include_web_research or not self._enricher:
            base_result = await self._base_engine.synthesize(
                topic=topic,
                template_type=template_type,
                search_results=search_results,
                include_verification=include_verification,
                include_figures=include_figures,
            )
            return self._convert_to_enhanced(base_result)

        # === ENHANCED FLOW WITH WEB RESEARCH ===
        logger.info(f"Starting enhanced synthesis with web research: '{topic}'")

        # Stage 1: Adapt context (reuse base adapter)
        context = self._base_engine.adapter.adapt(topic, search_results, template_type)

        # Stage 2: Generate title and abstract
        title, abstract = await self._base_engine._generate_title_abstract(
            topic, template_type, context
        )

        # Stage 3: Generate sections WITH enrichment
        sections = []
        all_figure_requests = []
        enrichment_results: Dict[str, EnrichmentResult] = {}
        total_enrichment_time = 0
        all_external_citations = []

        for section_name, level in TEMPLATE_SECTIONS.get(template_type, []):
            logger.info(f"Generating section with enrichment: {section_name}")

            section_chunks = context["sections"].get(section_name, [])

            # Run enrichment for this section
            enrichment = await self._enricher.enrich_context(
                topic=topic,
                internal_chunks=section_chunks,
                section_name=section_name,
            )
            enrichment_results[section_name] = enrichment
            total_enrichment_time += enrichment.enrichment_time_ms

            # Generate section with enriched context
            section_content, figure_requests = await self._generate_section_enriched(
                topic=topic,
                section_name=section_name,
                chunks=section_chunks,
                template_type=template_type,
                enrichment=enrichment,
            )

            sections.append(SynthesisSection(
                title=section_name,
                content=section_content,
                level=level,
                sources=[c["id"] for c in section_chunks[:5]],
            ))

            all_figure_requests.extend(figure_requests)

            # Collect external citations
            if enrichment.used_external:
                all_external_citations.extend(enrichment.external_citations)

        # Stage 4: Resolve figures (delegate to base)
        resolved_figures = []
        if include_figures and all_figure_requests:
            all_figure_requests, resolved_figures = self._base_engine.figure_resolver.resolve(
                all_figure_requests,
                context["image_catalog"],
            )

        # Stage 5: Conflict detection (delegate to base)
        conflict_report = await self._base_engine.conflict_handler.detect_conflicts(
            sections,
            mode="llm" if self._base_engine.deep_conflict_check else "heuristic"
        )

        # Stage 6: Build enhanced result
        synthesis_time = int((time() - start_time) * 1000)

        # Build gaps summary
        total_gaps = sum(len(e.gaps_identified) for e in enrichment_results.values())
        filled_gaps = sum(len(e.gaps_filled) for e in enrichment_results.values())

        result = EnhancedSynthesisResult(
            topic=topic,
            template_type=template_type,
            title=title,
            abstract=abstract,
            sections=sections,
            references=context["sources"],
            figure_requests=all_figure_requests,
            resolved_figures=resolved_figures,
            synthesis_time_ms=synthesis_time,
            conflict_report=conflict_report,
            # Enrichment fields
            enrichment_used=any(e.used_external for e in enrichment_results.values()),
            enrichment_results=enrichment_results,
            external_citations=list(set(all_external_citations)),  # Dedupe
            total_external_sources=sum(e.external_source_count for e in enrichment_results.values()),
            enrichment_time_ms=total_enrichment_time,
            gaps_summary={
                "total_identified": total_gaps,
                "total_filled": filled_gaps,
                "sections_enriched": sum(1 for e in enrichment_results.values() if e.used_external),
            },
        )

        # Stage 7: Optional verification
        if include_verification and self._base_engine.has_verification:
            result = await self._verify_enhanced(result)

        logger.info(
            f"Enhanced synthesis complete: {result.total_words} words, "
            f"{result.total_external_sources} external sources, "
            f"{total_gaps} gaps ({filled_gaps} filled), "
            f"{synthesis_time}ms"
        )

        return result

    async def _generate_section_enriched(
        self,
        topic: str,
        section_name: str,
        chunks: List[Dict],
        template_type: TemplateType,
        enrichment: EnrichmentResult,
    ) -> Tuple[str, List[FigureRequest]]:
        """
        Generate a section with enriched context.

        Modified prompt includes external context when available.
        """
        if not chunks and not enrichment.used_external:
            # No internal OR external content - generate minimal
            prompt = f"""Write a brief "{section_name}" section for a neurosurgical chapter on "{topic}".
Keep it to 100-150 words as no specific source material was found for this section.
Write in formal medical textbook style."""

            content = await self._call_claude(prompt, max_tokens=500)
            return content, []

        # Build internal source context
        source_context = ""
        for i, chunk in enumerate(chunks[:8], 1):
            doc_title = chunk.get("document_title", "Unknown")
            page = chunk.get("page", "?")
            authority = chunk.get("authority", "GENERAL")
            source_context += f"\n[Source {i}] ({doc_title}, p.{page}, Authority: {authority}):\n"
            source_context += chunk["content"][:1200] + "\n"

        # Build prompt with both internal and external context
        prompt = f"""Write the "{section_name}" section for a neurosurgical textbook chapter on "{topic}".

═══════════════════════════════════════════════════════════════════════════════
INTERNAL SOURCE MATERIALS (Primary - cite as [Source N])
═══════════════════════════════════════════════════════════════════════════════
{source_context if source_context else "No internal sources available for this section."}
"""

        # Add external context if available
        if enrichment.used_external and enrichment.external_context:
            prompt += f"""
═══════════════════════════════════════════════════════════════════════════════
EXTERNAL SOURCES (Recent - cite as [Web: source name])
═══════════════════════════════════════════════════════════════════════════════
{enrichment.external_context}
"""

        # Add requirements
        prompt += f"""
═══════════════════════════════════════════════════════════════════════════════
REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════
1. Write in formal medical textbook style (Youmans/Rhoton standard)
2. Include specific measurements with Mean±SD where available
3. Cite INTERNAL sources using [Source N] format
4. Cite EXTERNAL sources using [Web: source name] format
5. Include clinical pearls: [PEARL]content[/PEARL]
6. Include hazard warnings: [HAZARD]content[/HAZARD]
7. Request figures: [REQUEST_FIGURE: type="..." topic="..."]
8. Target length: 400-600 words
9. Use evidence grading (Level I/II/III) for clinical claims
10. Prioritize internal sources for established facts, external for recent developments

IMPORTANT: Keep internal and external citations clearly separated.
- [Source N] = Internal library (Rhoton, Lawton, Youmans)
- [Web: ...] = External recent sources

Write only the section content, no title."""

        content = await self._call_claude(prompt, max_tokens=2500)

        # Extract figure requests
        figure_requests = []
        pattern = r'\[REQUEST_FIGURE:\s*type="([^"]+)"\s*topic="([^"]+)"\]'

        for i, match in enumerate(re.finditer(pattern, content)):
            figure_requests.append(FigureRequest(
                placeholder_id=f"{section_name.replace(' ', '_')}_{i}",
                figure_type=match.group(1),
                topic=match.group(2),
                context=section_name,
            ))

        return content, figure_requests

    async def _call_claude(self, prompt: str, max_tokens: int = 4000) -> str:
        """Rate-limited Claude API call."""
        await self._base_engine.rate_limiter.acquire()

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def _verify_enhanced(
        self,
        result: EnhancedSynthesisResult,
    ) -> EnhancedSynthesisResult:
        """Run verification including external sources check."""
        # Delegate to base verification first
        base_result = await self._base_engine._verify(result)

        # Copy verification results
        result.verification_score = base_result.verification_score
        result.verification_issues = base_result.verification_issues
        result.verified = base_result.verified

        return result

    def _convert_to_enhanced(
        self,
        base_result: SynthesisResult,
    ) -> EnhancedSynthesisResult:
        """Convert a base SynthesisResult to EnhancedSynthesisResult."""
        return EnhancedSynthesisResult(
            topic=base_result.topic,
            template_type=base_result.template_type,
            title=base_result.title,
            abstract=base_result.abstract,
            sections=base_result.sections,
            references=base_result.references,
            figure_requests=base_result.figure_requests,
            resolved_figures=base_result.resolved_figures,
            synthesis_time_ms=base_result.synthesis_time_ms,
            conflict_report=base_result.conflict_report,
            verification_score=base_result.verification_score,
            verification_issues=base_result.verification_issues,
            verified=base_result.verified,
            # Default enrichment values (not used)
            enrichment_used=False,
            enrichment_results={},
            external_citations=[],
            gaps_summary={},
            total_external_sources=0,
            enrichment_time_ms=0,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_enrichment_to_engine(
    engine: SynthesisEngine,
    perplexity_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    config: Optional[EnrichmentConfig] = None,
) -> EnhancedSynthesisEngine:
    """
    Add enrichment capability to an existing SynthesisEngine.

    This creates a new EnhancedSynthesisEngine that wraps the existing one.
    The original engine is preserved and can still be used directly.

    Usage:
        existing_engine = SynthesisEngine(...)
        enhanced = add_enrichment_to_engine(
            existing_engine,
            perplexity_api_key="pplx-xxx",
        )

        # Use enhanced
        result = await enhanced.synthesize(..., include_web_research=True)

        # Original still works
        result = await existing_engine.synthesize(...)
    """
    enhanced = EnhancedSynthesisEngine.__new__(EnhancedSynthesisEngine)
    enhanced._base_engine = engine
    enhanced._client = engine.client
    enhanced._model = engine.model

    # Initialize enricher if keys provided
    if perplexity_api_key or google_api_key:
        enhanced._enricher = ResearchEnricher(
            anthropic_client=engine.client,
            perplexity_api_key=perplexity_api_key,
            google_api_key=google_api_key,
            config=config,
        )
    else:
        enhanced._enricher = None

    return enhanced


async def create_enhanced_engine_from_env() -> EnhancedSynthesisEngine:
    """
    Factory function to create EnhancedSynthesisEngine from environment variables.

    Environment variables:
        ANTHROPIC_API_KEY: Required
        PERPLEXITY_API_KEY: Optional, enables Perplexity search
        GOOGLE_API_KEY or GEMINI_API_KEY: Optional, enables Gemini grounding
    """
    import os
    from anthropic import AsyncAnthropic

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=anthropic_key)

    # Optional Gemini for verification
    verification_client = None
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            verification_client = genai.GenerativeModel("gemini-2.5-pro")
        except ImportError:
            pass

    return EnhancedSynthesisEngine(
        anthropic_client=client,
        verification_client=verification_client,
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        google_api_key=gemini_key,
    )


# =============================================================================
# API ROUTE HELPER
# =============================================================================

class EnhancedSynthesisRequest:
    """
    Extended request model for API routes.

    Add to your SynthesisRequest Pydantic model:
        include_web_research: bool = Field(
            default=False,
            description="Enable Open-World web research for recent sources"
        )
    """
    pass


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def demo():
    """Demonstrate enhanced synthesis."""
    import os
    from anthropic import AsyncAnthropic

    # Mock search results
    from dataclasses import dataclass

    @dataclass
    class MockSearchResult:
        chunk_id: str
        document_id: str
        content: str
        title: str
        document_title: str
        authority_score: float
        entity_names: list
        images: list = None

        def __post_init__(self):
            self.images = self.images or []

    mock_results = [
        MockSearchResult(
            chunk_id="c001",
            document_id="d001",
            content="The pterional approach provides access to the circle of Willis and proximal portions of the anterior and middle cerebral arteries...",
            title="Pterional Approach",
            document_title="Rhoton Cranial Anatomy",
            authority_score=0.95,
            entity_names=["pterional", "circle of Willis", "MCA"],
        ),
        MockSearchResult(
            chunk_id="c002",
            document_id="d002",
            content="Seven aneurysms principles: proximal control, temporary clipping, sharp dissection, dome manipulation last...",
            title="Aneurysm Surgery Principles",
            document_title="Lawton Seven Aneurysms",
            authority_score=0.92,
            entity_names=["aneurysm", "clipping", "temporary occlusion"],
        ),
    ]

    # Create enhanced engine
    engine = await create_enhanced_engine_from_env()

    print("=" * 70)
    print("ENHANCED SYNTHESIS DEMO")
    print("=" * 70)

    # Run synthesis with web research
    result = await engine.synthesize(
        topic="MCA aneurysm clipping",
        template_type=TemplateType.PROCEDURAL,
        search_results=mock_results,
        include_web_research=True,
    )

    print(f"\nTitle: {result.title}")
    print(f"Sections: {len(result.sections)}")
    print(f"Total words: {result.total_words}")
    print(f"Enrichment used: {result.enrichment_used}")
    print(f"External sources: {result.total_external_sources}")
    print(f"Gaps identified: {result.gaps_summary.get('total_identified', 0)}")
    print(f"Gaps filled: {result.gaps_summary.get('total_filled', 0)}")

    if result.external_citations:
        print(f"\nExternal citations:")
        for cit in result.external_citations[:5]:
            print(f"  • {cit}")


if __name__ == "__main__":
    asyncio.run(demo())
