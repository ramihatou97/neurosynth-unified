"""
NeuroSynth V3 - Research Enricher Module
=========================================

Open-World Knowledge Integration for the Synthesis Engine.

This module adds "Hybrid Brain" capability to NeuroSynth:
- Internal: Curated PDF corpus (Rhoton, Lawton, Youmans, etc.)
- External: Real-time web/academic search (Perplexity, Gemini)

The system performs SEMANTIC GAP ANALYSIS - not simple threshold counting.
An LLM compares internal vs external knowledge to identify specific gaps.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │ Internal Search │     │ External Search │
    │ (SearchService) │     │ (Perplexity/    │
    │                 │     │  Gemini)        │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Gap Analyzer     │
              │  (LLM Comparison)   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Targeted Enrichment │
              │  (Fill Gaps Only)   │
              └─────────────────────┘

Usage:
    from src.synthesis.research_enricher import ResearchEnricher, EnrichmentConfig

    enricher = ResearchEnricher(
        anthropic_client=client,
        perplexity_api_key="pplx-xxx",
        google_api_key="xxx",
    )

    # In SynthesisEngine.synthesize():
    enriched_context = await enricher.enrich_context(
        topic="MCA aneurysm clipping",
        internal_chunks=search_results,
        section_name="Recent Advances",
    )

Integration Notes:
    - Does NOT modify existing SynthesisEngine, SearchService, or SearchResult
    - Operates as an optional enhancement layer
    - Controlled via `include_web_research` flag
    - External citations clearly marked as [Web: source]

Author: NeuroSynth Team
Version: 3.0.0
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class GapType(Enum):
    """Types of knowledge gaps that can be identified."""
    MISSING_DATA = "missing_data"           # Topic not covered at all
    OUTDATED = "outdated"                   # Internal data is old
    INCOMPLETE_COVERAGE = "incomplete"      # Partial coverage only
    RECENT_DEVELOPMENTS = "recent"          # New research since corpus
    CLINICAL_TRIALS = "trials"              # Missing trial data
    GUIDELINES_UPDATE = "guidelines"        # Updated clinical guidelines


class GapPriority(Enum):
    """Priority levels for addressing gaps."""
    CRITICAL = "critical"   # Must fill - core to the topic
    HIGH = "high"           # Should fill - significantly improves quality
    MEDIUM = "medium"       # Nice to have - adds depth
    LOW = "low"             # Optional - minor enhancement


@dataclass
class EnrichmentConfig:
    """Configuration for the research enrichment system."""

    # Feature toggles
    enabled: bool = True
    enable_perplexity: bool = True
    enable_gemini_grounding: bool = True
    enable_gap_analysis: bool = True

    # Gap analysis settings
    min_internal_chunks_before_external: int = 3  # Fallback if LLM gap analysis fails
    gap_analysis_model: str = "claude-sonnet-4-20250514"

    # Perplexity settings (Sonar Pro for academic)
    perplexity_model: str = "sonar-pro"
    perplexity_max_tokens: int = 2000
    perplexity_search_recency: str = "month"  # day, week, month, year

    # Gemini settings (Grounding with Google Search)
    gemini_model: str = "gemini-2.5-pro"
    gemini_grounding_enabled: bool = True

    # Cost control
    max_external_queries_per_synthesis: int = 5
    max_gap_fill_attempts: int = 3
    cache_external_results: bool = True
    cache_ttl_hours: int = 24

    # Sections that ALWAYS trigger external search
    always_enrich_sections: List[str] = field(default_factory=lambda: [
        "Recent Advances",
        "Future Directions",
        "Emerging Techniques",
        "Current Evidence",
        "Clinical Trials",
        "Guidelines",
    ])

    # Rate limiting
    perplexity_rpm: int = 20
    gemini_rpm: int = 60


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class KnowledgeGap:
    """
    A specific gap identified between internal and external knowledge.

    The LLM identifies these by comparing what the internal corpus has
    against what external sources report on the same topic.
    """
    topic: str                          # Specific subtopic with gap
    description: str                    # What information is missing
    gap_type: GapType                   # Category of gap
    priority: GapPriority               # How important to fill
    suggested_query: str                # Query to fill this gap
    internal_coverage: str              # What internal corpus HAS
    external_insight: str               # What external sources HAVE
    confidence: float = 0.0             # LLM confidence in this gap (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "description": self.description,
            "gap_type": self.gap_type.value,
            "priority": self.priority.value,
            "suggested_query": self.suggested_query,
            "internal_coverage": self.internal_coverage,
            "external_insight": self.external_insight,
            "confidence": self.confidence,
        }


@dataclass
class ExternalSource:
    """A source retrieved from external search."""
    title: str
    url: str
    snippet: str
    source_type: Literal["academic", "clinical", "news", "guideline", "other"]
    publication_date: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0

    def to_citation(self) -> str:
        """Format as citation for synthesis."""
        if self.doi:
            return f"[Web: {self.title} (DOI: {self.doi})]"
        elif self.publication_date:
            return f"[Web: {self.title} ({self.publication_date})]"
        else:
            return f"[Web: {self.title}]"


@dataclass
class ExternalSearchResult:
    """Result from an external search provider."""
    query: str
    provider: Literal["perplexity", "gemini", "combined"]
    sources: List[ExternalSource]
    summary: str                        # Provider's synthesized answer
    raw_response: Optional[Dict] = None
    search_time_ms: int = 0
    tokens_used: int = 0


@dataclass
class EnrichmentResult:
    """Complete result of the enrichment process for a topic/section."""
    topic: str
    section_name: Optional[str]

    # Gap analysis results
    gaps_identified: List[KnowledgeGap]
    gaps_filled: List[KnowledgeGap]
    gaps_unfilled: List[KnowledgeGap]

    # External content retrieved
    external_results: List[ExternalSearchResult]
    external_context: str               # Formatted for synthesis prompt
    external_citations: List[str]       # Citations to include

    # Metrics
    internal_chunk_count: int
    external_source_count: int
    enrichment_time_ms: int
    total_tokens_used: int

    # Control flags
    used_external: bool
    gap_analysis_succeeded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "section_name": self.section_name,
            "gaps_identified": [g.to_dict() for g in self.gaps_identified],
            "gaps_filled": [g.to_dict() for g in self.gaps_filled],
            "gaps_unfilled": [g.to_dict() for g in self.gaps_unfilled],
            "internal_chunk_count": self.internal_chunk_count,
            "external_source_count": self.external_source_count,
            "enrichment_time_ms": self.enrichment_time_ms,
            "used_external": self.used_external,
        }


# =============================================================================
# EXTERNAL SEARCH PROVIDERS
# =============================================================================

class ExternalSearchProvider(ABC):
    """Abstract base class for external search providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        context: Optional[str] = None
    ) -> ExternalSearchResult:
        """Execute a search query."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        pass


class PerplexityProvider(ExternalSearchProvider):
    """
    Perplexity AI search provider using Sonar Pro.

    Best for:
    - Academic/research queries
    - Recent publications and citations
    - State-of-the-art summaries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar-pro",
        search_recency: str = "month",
    ):
        self.api_key = api_key
        self.model = model
        self.search_recency = search_recency
        self._client = None

        if api_key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
                )
                logger.info("Perplexity provider initialized")
            except ImportError:
                logger.warning("openai package not installed for Perplexity")

    def is_available(self) -> bool:
        return self._client is not None

    async def search(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> ExternalSearchResult:
        """
        Search using Perplexity Sonar Pro.

        Returns academic-focused results with citations.
        """
        if not self.is_available():
            return ExternalSearchResult(
                query=query,
                provider="perplexity",
                sources=[],
                summary="Perplexity not available",
            )

        import time
        start = time.time()

        # Build system prompt for medical/academic focus
        system_prompt = """You are a medical research assistant specializing in neurosurgery.
Focus on:
1. Recent peer-reviewed publications (2023-2025)
2. Clinical trial results
3. Updated clinical guidelines
4. Systematic reviews and meta-analyses

Always cite sources with authors, publication year, and journal when available.
Prioritize high-impact journals: Neurosurgery, JNS, World Neurosurgery, Lancet Neurology."""

        # Add context if provided
        user_content = query
        if context:
            user_content = f"""CONTEXT (from internal medical library):
{context[:1500]}

QUESTION: {query}

Find recent information that complements or updates the above context."""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=2000,
            )

            elapsed_ms = int((time.time() - start) * 1000)

            # Parse response
            content = response.choices[0].message.content

            # Extract sources from Perplexity's citation format
            sources = self._extract_sources(content)

            return ExternalSearchResult(
                query=query,
                provider="perplexity",
                sources=sources,
                summary=content,
                search_time_ms=elapsed_ms,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return ExternalSearchResult(
                query=query,
                provider="perplexity",
                sources=[],
                summary=f"Search failed: {str(e)}",
            )

    def _extract_sources(self, content: str) -> List[ExternalSource]:
        """Extract source citations from Perplexity response."""
        sources = []

        # Perplexity often includes [N] citations
        # Extract URLs and titles from the response
        url_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        matches = re.findall(url_pattern, content)

        for title, url in matches[:10]:  # Limit to 10 sources
            sources.append(ExternalSource(
                title=title,
                url=url,
                snippet="",
                source_type=self._classify_source(url),
            ))

        return sources

    def _classify_source(self, url: str) -> str:
        """Classify source type based on URL."""
        url_lower = url.lower()
        if any(d in url_lower for d in ["pubmed", "ncbi", "doi.org", "scholar"]):
            return "academic"
        elif any(d in url_lower for d in ["clinicaltrials", "cochrane"]):
            return "clinical"
        elif any(d in url_lower for d in ["who.int", "cdc.gov", "aans", "cns.org"]):
            return "guideline"
        elif any(d in url_lower for d in ["news", "medscape", "healio"]):
            return "news"
        return "other"


class GeminiGroundingProvider(ExternalSearchProvider):
    """
    Google Gemini with Grounding (Google Search integration).

    Best for:
    - Fact-checking and verification
    - Broad knowledge queries
    - Recent news and developments
    - Cross-referencing information
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        enable_grounding: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.enable_grounding = enable_grounding
        self._client = None

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(model)
                logger.info("Gemini Grounding provider initialized")
            except ImportError:
                logger.warning("google-generativeai package not installed")

    def is_available(self) -> bool:
        return self._client is not None

    async def search(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> ExternalSearchResult:
        """
        Search using Gemini with Google Search grounding.

        Returns fact-checked, grounded results.
        """
        if not self.is_available():
            return ExternalSearchResult(
                query=query,
                provider="gemini",
                sources=[],
                summary="Gemini not available",
            )

        import time
        start = time.time()

        # Build prompt for medical grounding
        prompt = f"""You are a medical fact-checker specializing in neurosurgery.

TASK: Search for accurate, current information about:
{query}

{"CONTEXT from internal library:" + chr(10) + context[:1000] if context else ""}

REQUIREMENTS:
1. Verify information against authoritative medical sources
2. Note any discrepancies or updates to standard knowledge
3. Include publication dates for any cited information
4. Flag if guidelines or recommendations have changed recently
5. Cite specific sources (journals, guidelines, organizations)

Format your response with clear source attributions."""

        try:
            # Use grounding if enabled
            if self.enable_grounding:
                from google.generativeai import GenerationConfig
                response = await asyncio.to_thread(
                    self._client.generate_content,
                    prompt,
                    generation_config=GenerationConfig(
                        temperature=0.1,  # Low temp for factual accuracy
                        max_output_tokens=2000,
                    ),
                    # Note: Grounding tools require specific API setup
                    # tools=[{"google_search": {}}],  # Enable when available
                )
            else:
                response = await asyncio.to_thread(
                    self._client.generate_content,
                    prompt,
                )

            elapsed_ms = int((time.time() - start) * 1000)

            content = response.text
            sources = self._extract_sources(content)

            return ExternalSearchResult(
                query=query,
                provider="gemini",
                sources=sources,
                summary=content,
                search_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Gemini search failed: {e}")
            return ExternalSearchResult(
                query=query,
                provider="gemini",
                sources=[],
                summary=f"Search failed: {str(e)}",
            )

    def _extract_sources(self, content: str) -> List[ExternalSource]:
        """Extract sources from Gemini grounding response."""
        sources = []

        # Look for common citation patterns
        # Pattern 1: (Author et al., Year)
        author_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*(\d{4})\)'
        for match in re.finditer(author_pattern, content):
            sources.append(ExternalSource(
                title=f"{match.group(1)} ({match.group(2)})",
                url="",
                snippet="",
                source_type="academic",
                publication_date=match.group(2),
            ))

        # Pattern 2: URLs
        url_pattern = r'https?://[^\s\)\]<>]+'
        for url in re.findall(url_pattern, content)[:5]:
            if not any(s.url == url for s in sources):
                sources.append(ExternalSource(
                    title=url.split('/')[2],  # Domain as title
                    url=url,
                    snippet="",
                    source_type="other",
                ))

        return sources


# =============================================================================
# GAP ANALYZER - THE BRAIN OF THE SYSTEM
# =============================================================================

class GapAnalyzer:
    """
    Semantic Gap Analysis using LLM comparison.

    This is the core intelligence of the enrichment system.
    It doesn't just count chunks - it UNDERSTANDS what's missing.

    Process:
    1. Summarize internal corpus coverage
    2. Get external knowledge overview
    3. LLM compares both to identify specific gaps
    4. Return prioritized list of gaps to fill
    """

    def __init__(
        self,
        anthropic_client,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic_client
        self.model = model

    async def analyze_gaps(
        self,
        topic: str,
        section_name: Optional[str],
        internal_chunks: List[Dict[str, Any]],
        external_overview: str,
    ) -> List[KnowledgeGap]:
        """
        Compare internal corpus against external knowledge to find gaps.

        Args:
            topic: The main topic being synthesized
            section_name: Specific section (e.g., "Recent Advances")
            internal_chunks: Chunks from SearchService (as dicts)
            external_overview: Summary from Perplexity/Gemini

        Returns:
            List of identified gaps, prioritized
        """
        if not internal_chunks and not external_overview:
            return []

        # Build internal summary
        internal_summary = self._summarize_internal(internal_chunks)

        # Build comparison prompt
        prompt = self._build_gap_analysis_prompt(
            topic=topic,
            section_name=section_name,
            internal_summary=internal_summary,
            external_overview=external_overview,
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            gaps = self._parse_gaps(content)

            # Sort by priority
            priority_order = {
                GapPriority.CRITICAL: 0,
                GapPriority.HIGH: 1,
                GapPriority.MEDIUM: 2,
                GapPriority.LOW: 3,
            }
            gaps.sort(key=lambda g: priority_order.get(g.priority, 99))

            logger.info(f"Gap analysis found {len(gaps)} gaps for '{topic}'")
            return gaps

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return []

    def _summarize_internal(self, chunks: List[Dict[str, Any]]) -> str:
        """Create a summary of what the internal corpus covers."""
        if not chunks:
            return "NO INTERNAL COVERAGE - corpus has no relevant content."

        lines = []
        lines.append(f"INTERNAL CORPUS COVERAGE ({len(chunks)} chunks):")
        lines.append("")

        # Group by source
        by_source: Dict[str, List[Dict]] = {}
        for chunk in chunks:
            source = chunk.get("document_title", "Unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)

        for source, source_chunks in list(by_source.items())[:5]:  # Top 5 sources
            lines.append(f"• {source} ({len(source_chunks)} chunks):")
            # Sample content
            for chunk in source_chunks[:2]:
                content = chunk.get("content", "")[:300]
                lines.append(f"  - {content}...")
            lines.append("")

        # Entity coverage
        all_entities = set()
        for chunk in chunks:
            entities = chunk.get("entity_names", [])
            all_entities.update(entities[:5])

        if all_entities:
            lines.append(f"ENTITIES COVERED: {', '.join(list(all_entities)[:20])}")

        return "\n".join(lines)

    def _build_gap_analysis_prompt(
        self,
        topic: str,
        section_name: Optional[str],
        internal_summary: str,
        external_overview: str,
    ) -> str:
        """Build the prompt for gap analysis."""
        section_context = f' (Section: "{section_name}")' if section_name else ""

        return f"""You are a medical knowledge analyst comparing two sources of information about:
TOPIC: {topic}{section_context}

═══════════════════════════════════════════════════════════════════════════════
INTERNAL CORPUS (Curated Medical Library - Rhoton, Youmans, Lawton, etc.)
═══════════════════════════════════════════════════════════════════════════════
{internal_summary}

═══════════════════════════════════════════════════════════════════════════════
EXTERNAL KNOWLEDGE (Recent Web/Academic Sources - 2023-2025)
═══════════════════════════════════════════════════════════════════════════════
{external_overview[:3000]}

═══════════════════════════════════════════════════════════════════════════════
TASK: IDENTIFY KNOWLEDGE GAPS
═══════════════════════════════════════════════════════════════════════════════

Compare the internal corpus against external knowledge. Identify SPECIFIC gaps
where external sources have information that the internal corpus LACKS.

For each gap, determine:
1. TOPIC: The specific subtopic or concept that's missing
2. DESCRIPTION: What information is absent from internal corpus
3. GAP_TYPE: One of [missing_data, outdated, incomplete, recent, trials, guidelines]
4. PRIORITY: One of [critical, high, medium, low]
   - critical: Core to the topic, synthesis would be wrong without it
   - high: Significantly improves quality
   - medium: Adds depth
   - low: Minor enhancement
5. INTERNAL_HAS: Brief note on what internal DOES cover
6. EXTERNAL_HAS: Brief note on what external sources HAVE
7. SUGGESTED_QUERY: Search query to fill this gap
8. CONFIDENCE: 0.0-1.0 (how confident you are this is a real gap)

IMPORTANT RULES:
- Only report GENUINE gaps, not minor differences
- If internal coverage is comprehensive, return empty array
- Focus on clinically relevant information
- Prioritize safety-critical gaps (e.g., updated contraindications)
- Don't report gaps for historical/foundational content (internal is authoritative)

Return as JSON array:
```json
[
  {{
    "topic": "SACE trial outcomes",
    "description": "No data on 2024 SACE trial results for MCA aneurysm clipping vs coiling",
    "gap_type": "trials",
    "priority": "high",
    "internal_has": "General clipping techniques from Lawton",
    "external_has": "SACE trial 5-year follow-up showing 94% clip durability",
    "suggested_query": "SACE trial MCA aneurysm clipping outcomes 2024",
    "confidence": 0.9
  }}
]
```

Return empty array [] if internal coverage is sufficient."""

    def _parse_gaps(self, response: str) -> List[KnowledgeGap]:
        """Parse the LLM response into KnowledgeGap objects."""
        gaps = []

        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON array found in gap analysis response")
            return gaps

        try:
            data = json.loads(json_match.group())

            for item in data:
                try:
                    gap = KnowledgeGap(
                        topic=item.get("topic", ""),
                        description=item.get("description", ""),
                        gap_type=GapType(item.get("gap_type", "missing_data")),
                        priority=GapPriority(item.get("priority", "medium")),
                        suggested_query=item.get("suggested_query", ""),
                        internal_coverage=item.get("internal_has", ""),
                        external_insight=item.get("external_has", ""),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                    gaps.append(gap)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse gap: {e}")
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gap analysis JSON: {e}")

        return gaps


# =============================================================================
# RESEARCH ENRICHER - THE MAIN INTERFACE
# =============================================================================

class ResearchEnricher:
    """
    Main interface for Open-World knowledge enrichment.

    Integrates with SynthesisEngine without modifying existing code.

    Usage in SynthesisEngine:
        # During section generation
        if self.enricher and include_web_research:
            enrichment = await self.enricher.enrich_context(
                topic=topic,
                internal_chunks=section_chunks,
                section_name=section_name,
            )
            if enrichment.used_external:
                # Add external context to prompt
                prompt += f"\\n\\nADDITIONAL CONTEXT (Recent Sources):\\n{enrichment.external_context}"
    """

    def __init__(
        self,
        anthropic_client,
        perplexity_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[EnrichmentConfig] = None,
    ):
        self.config = config or EnrichmentConfig()
        self.anthropic_client = anthropic_client

        # Initialize providers
        self.perplexity = PerplexityProvider(
            api_key=perplexity_api_key,
            model=self.config.perplexity_model,
        ) if perplexity_api_key and self.config.enable_perplexity else None

        self.gemini = GeminiGroundingProvider(
            api_key=google_api_key,
            model=self.config.gemini_model,
            enable_grounding=self.config.gemini_grounding_enabled,
        ) if google_api_key and self.config.enable_gemini_grounding else None

        # Initialize gap analyzer
        self.gap_analyzer = GapAnalyzer(
            anthropic_client=anthropic_client,
            model=self.config.gap_analysis_model,
        ) if self.config.enable_gap_analysis else None

        # Cache for external results
        self._cache: Dict[str, ExternalSearchResult] = {}

        # Track usage for cost control
        self._queries_this_synthesis = 0

        logger.info(
            f"ResearchEnricher initialized: "
            f"perplexity={self.perplexity is not None}, "
            f"gemini={self.gemini is not None}, "
            f"gap_analysis={self.gap_analyzer is not None}"
        )

    def reset_synthesis_counter(self):
        """Reset query counter for new synthesis. Call at start of synthesize()."""
        self._queries_this_synthesis = 0

    def is_available(self) -> bool:
        """Check if any external provider is available."""
        return (
            (self.perplexity and self.perplexity.is_available()) or
            (self.gemini and self.gemini.is_available())
        )

    async def enrich_context(
        self,
        topic: str,
        internal_chunks: List[Dict[str, Any]],
        section_name: Optional[str] = None,
        force_external: bool = False,
    ) -> EnrichmentResult:
        """
        Main entry point for context enrichment.

        Process:
        1. Check if enrichment is needed/allowed
        2. Get external overview (Perplexity + Gemini)
        3. Run gap analysis (LLM comparison)
        4. Fill high-priority gaps
        5. Return enriched context

        Args:
            topic: Main topic being synthesized
            internal_chunks: Chunks from SearchService (as dicts from ContextAdapter)
            section_name: Section being generated (e.g., "Recent Advances")
            force_external: Skip gap analysis, always fetch external

        Returns:
            EnrichmentResult with gaps and external context
        """
        import time
        start = time.time()

        # Initialize result
        result = EnrichmentResult(
            topic=topic,
            section_name=section_name,
            gaps_identified=[],
            gaps_filled=[],
            gaps_unfilled=[],
            external_results=[],
            external_context="",
            external_citations=[],
            internal_chunk_count=len(internal_chunks),
            external_source_count=0,
            enrichment_time_ms=0,
            total_tokens_used=0,
            used_external=False,
            gap_analysis_succeeded=False,
        )

        # Check if enrichment is enabled and available
        if not self.config.enabled or not self.is_available():
            logger.debug("Enrichment disabled or no providers available")
            result.enrichment_time_ms = int((time.time() - start) * 1000)
            return result

        # Check query limit
        if self._queries_this_synthesis >= self.config.max_external_queries_per_synthesis:
            logger.warning("External query limit reached for this synthesis")
            result.enrichment_time_ms = int((time.time() - start) * 1000)
            return result

        # Determine if we should use external search
        should_search = force_external or self._should_search_external(
            internal_chunks=internal_chunks,
            section_name=section_name,
        )

        if not should_search:
            logger.debug(f"Skipping external search for '{topic}' - internal coverage sufficient")
            result.enrichment_time_ms = int((time.time() - start) * 1000)
            return result

        # Step 1: Get external overview
        external_overview = await self._get_external_overview(topic, section_name)
        if not external_overview.summary:
            result.enrichment_time_ms = int((time.time() - start) * 1000)
            return result

        result.external_results.append(external_overview)
        result.total_tokens_used += external_overview.tokens_used

        # Step 2: Run gap analysis
        if self.gap_analyzer:
            gaps = await self.gap_analyzer.analyze_gaps(
                topic=topic,
                section_name=section_name,
                internal_chunks=internal_chunks,
                external_overview=external_overview.summary,
            )
            result.gaps_identified = gaps
            result.gap_analysis_succeeded = True

        # Step 3: Fill high-priority gaps
        if result.gaps_identified:
            filled, unfilled = await self._fill_gaps(
                gaps=result.gaps_identified,
                max_attempts=self.config.max_gap_fill_attempts,
            )
            result.gaps_filled = filled
            result.gaps_unfilled = unfilled

            # Add gap-filling results
            for gap_result in filled:
                if hasattr(gap_result, '_search_result'):
                    result.external_results.append(gap_result._search_result)

        # Step 4: Build external context for synthesis
        result.external_context = self._build_external_context(
            overview=external_overview,
            gaps_filled=result.gaps_filled,
            external_results=result.external_results,
        )

        # Step 5: Collect citations
        result.external_citations = self._collect_citations(result.external_results)

        # Finalize
        result.external_source_count = sum(
            len(r.sources) for r in result.external_results
        )
        result.used_external = bool(result.external_context)
        result.enrichment_time_ms = int((time.time() - start) * 1000)

        logger.info(
            f"Enrichment complete for '{topic}': "
            f"gaps={len(result.gaps_identified)}, "
            f"filled={len(result.gaps_filled)}, "
            f"sources={result.external_source_count}, "
            f"time={result.enrichment_time_ms}ms"
        )

        return result

    def _should_search_external(
        self,
        internal_chunks: List[Dict],
        section_name: Optional[str],
    ) -> bool:
        """
        Determine if external search is warranted.

        This is a heuristic fallback if gap analysis is disabled.
        """
        # Always search for certain sections
        if section_name and section_name in self.config.always_enrich_sections:
            return True

        # Search if internal coverage is thin
        if len(internal_chunks) < self.config.min_internal_chunks_before_external:
            return True

        return False

    async def _get_external_overview(
        self,
        topic: str,
        section_name: Optional[str],
    ) -> ExternalSearchResult:
        """Get initial external overview from best available provider."""

        # Build search query
        query = topic
        if section_name:
            query = f"{topic} {section_name} neurosurgery"

        # Check cache
        cache_key = f"{query}_{section_name}"
        if self.config.cache_external_results and cache_key in self._cache:
            logger.debug(f"Using cached result for '{query}'")
            return self._cache[cache_key]

        # Prefer Perplexity for academic content
        if self.perplexity and self.perplexity.is_available():
            self._queries_this_synthesis += 1
            result = await self.perplexity.search(query)
            if result.summary and "failed" not in result.summary.lower():
                if self.config.cache_external_results:
                    self._cache[cache_key] = result
                return result

        # Fallback to Gemini
        if self.gemini and self.gemini.is_available():
            self._queries_this_synthesis += 1
            result = await self.gemini.search(query)
            if self.config.cache_external_results:
                self._cache[cache_key] = result
            return result

        return ExternalSearchResult(
            query=query,
            provider="combined",
            sources=[],
            summary="",
        )

    async def _fill_gaps(
        self,
        gaps: List[KnowledgeGap],
        max_attempts: int,
    ) -> Tuple[List[KnowledgeGap], List[KnowledgeGap]]:
        """
        Attempt to fill identified gaps with targeted searches.

        Returns:
            Tuple of (filled_gaps, unfilled_gaps)
        """
        filled = []
        unfilled = []

        # Only attempt high-priority gaps
        high_priority = [
            g for g in gaps
            if g.priority in (GapPriority.CRITICAL, GapPriority.HIGH)
            and g.confidence >= 0.7
        ]

        for gap in high_priority[:max_attempts]:
            if self._queries_this_synthesis >= self.config.max_external_queries_per_synthesis:
                unfilled.append(gap)
                continue

            # Search for this specific gap
            result = await self._get_external_overview(
                topic=gap.suggested_query,
                section_name=None,
            )

            if result.summary and "failed" not in result.summary.lower():
                gap._search_result = result  # Attach for later use
                filled.append(gap)
            else:
                unfilled.append(gap)

        # Mark remaining gaps as unfilled
        for gap in gaps:
            if gap not in filled and gap not in unfilled:
                unfilled.append(gap)

        return filled, unfilled

    def _build_external_context(
        self,
        overview: ExternalSearchResult,
        gaps_filled: List[KnowledgeGap],
        external_results: List[ExternalSearchResult],
    ) -> str:
        """Build formatted external context for synthesis prompt."""
        lines = []

        lines.append("=" * 70)
        lines.append("EXTERNAL SOURCES (2023-2025 - Use [Web: source] citations)")
        lines.append("=" * 70)
        lines.append("")

        # Add overview
        if overview.summary:
            lines.append("RECENT DEVELOPMENTS:")
            # Truncate to reasonable length
            summary = overview.summary[:2000]
            lines.append(summary)
            lines.append("")

        # Add gap-specific findings
        for gap in gaps_filled:
            if hasattr(gap, '_search_result'):
                lines.append(f"ON '{gap.topic.upper()}':")
                lines.append(gap._search_result.summary[:800])
                lines.append("")

        # Add source list
        all_sources = []
        for result in external_results:
            all_sources.extend(result.sources)

        if all_sources:
            lines.append("AVAILABLE CITATIONS:")
            for i, source in enumerate(all_sources[:10], 1):
                lines.append(f"  [{i}] {source.to_citation()}")

        return "\n".join(lines)

    def _collect_citations(
        self,
        results: List[ExternalSearchResult],
    ) -> List[str]:
        """Collect all citations from external results."""
        citations = []
        seen = set()

        for result in results:
            for source in result.sources:
                citation = source.to_citation()
                if citation not in seen:
                    citations.append(citation)
                    seen.add(citation)

        return citations


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_enricher_from_env() -> Optional[ResearchEnricher]:
    """
    Factory function to create ResearchEnricher from environment variables.

    Environment variables:
        ANTHROPIC_API_KEY: Required for gap analysis
        PERPLEXITY_API_KEY: Optional, enables Perplexity search
        GOOGLE_API_KEY: Optional, enables Gemini grounding
        ENABLE_WEB_RESEARCH: Optional, set to "false" to disable
    """
    import os

    # Check if disabled
    if os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "false":
        logger.info("Web research disabled via environment")
        return None

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set, cannot create enricher")
        return None

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=anthropic_key)
    except ImportError:
        logger.error("anthropic package not installed")
        return None

    return ResearchEnricher(
        anthropic_client=client,
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

async def demo():
    """Demonstrate the research enricher."""
    import os
    from anthropic import AsyncAnthropic

    # Initialize
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    enricher = ResearchEnricher(
        anthropic_client=client,
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Mock internal chunks
    internal_chunks = [
        {
            "id": "chunk_001",
            "content": "The pterional approach provides access to the circle of Willis...",
            "document_title": "Rhoton Cranial Anatomy",
            "authority_score": 0.95,
            "entity_names": ["pterional", "circle of Willis", "MCA"],
        },
        {
            "id": "chunk_002",
            "content": "Temporary clipping of the parent vessel...",
            "document_title": "Lawton Seven Aneurysms",
            "authority_score": 0.90,
            "entity_names": ["temporary clipping", "aneurysm"],
        },
    ]

    # Run enrichment
    result = await enricher.enrich_context(
        topic="MCA aneurysm clipping techniques",
        internal_chunks=internal_chunks,
        section_name="Recent Advances",
    )

    print("=" * 70)
    print("ENRICHMENT RESULT")
    print("=" * 70)
    print(f"Topic: {result.topic}")
    print(f"Used external: {result.used_external}")
    print(f"Gaps identified: {len(result.gaps_identified)}")
    print(f"Gaps filled: {len(result.gaps_filled)}")
    print(f"External sources: {result.external_source_count}")
    print(f"Time: {result.enrichment_time_ms}ms")
    print()

    if result.gaps_identified:
        print("IDENTIFIED GAPS:")
        for gap in result.gaps_identified[:3]:
            print(f"  • {gap.topic} ({gap.priority.value})")
            print(f"    {gap.description}")
        print()

    if result.external_context:
        print("EXTERNAL CONTEXT (first 500 chars):")
        print(result.external_context[:500])


if __name__ == "__main__":
    asyncio.run(demo())
