"""
NeuroSynth Unified - Research Enricher
=======================================

Core orchestration for dual-path retrieval (internal + external) with
gap analysis to synthesize comprehensive, validated answers.

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
              │  Context Assembler  │
              │  (Merged + Ranked)  │
              └─────────────────────┘

Usage:
    from src.research import ResearchEnricher, SearchMode

    enricher = ResearchEnricher(
        search_service=search_service,
        perplexity_client=perplexity,
        gemini_client=gemini
    )

    # Hybrid search with gap analysis
    context = await enricher.enrich(
        query="Latest DBS infection management protocols",
        mode=SearchMode.HYBRID
    )

    # Check for conflicts
    if context.gap_report and context.gap_report.has_conflicts:
        for conflict in context.gap_report.conflicts:
            print(f"⚠️ {conflict.topic}: {conflict.recommendation}")
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from src.research.models import (
    SearchMode,
    ExternalSearchResult,
    ExternalCitation,
    GapReport,
    Conflict,
    ConflictSeverity,
    EnrichedContext,
    EnricherConfig,
)
from src.research.external_search import (
    PerplexitySearchClient,
    GeminiDeepResearchClient,
    ExternalSearchError,
)
from src.synthesis.conflict_merger import (
    ConflictAwareMerger,
    DetectedConflict,
    ConflictCategory,
    ResolutionStrategy,
    MergeResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Gap Analysis Prompts - The "Judge" Architecture
# =============================================================================

GAP_ANALYSIS_SYSTEM_PROMPT = """You are a CLINICAL KNOWLEDGE AUDITOR acting as an impartial JUDGE.

Your role is NOT to answer the user's question yet. Your role is to COMPARE two knowledge sources
and identify discrepancies BEFORE any synthesis occurs.

## The Two Knowledge Sources

1. **GROUND TRUTH (Internal)**: The user's curated database - their personal notes, textbooks,
   institutional protocols. This is what they currently believe/practice.

2. **WORLD TRUTH (External)**: Current information from authoritative web sources - latest
   guidelines, recent publications, consensus statements. This is the current standard of care.

## Your Judge's Verdict Must Identify:

### AGREEMENTS (Validation)
Where Ground Truth aligns with World Truth. This confirms the internal data is current.
Example: "Both sources confirm that DBS infection rates range from 1-15%"

### CONFLICTS (⚠️ Critical for Patient Safety)
Where Ground Truth CONTRADICTS World Truth. These are potential CLINICAL RISKS.
- Drug dosages that have changed
- Protocols that have been updated
- Techniques that are now contraindicated
- Timing recommendations that have shifted

Example: "CONFLICT: Internal notes recommend Vancomycin monotherapy, but 2024 IDSA guidelines
now recommend adding Rifampin for retained DBS hardware."

### KNOWLEDGE GAPS (Missing Information)
Information in World Truth that is ABSENT from Ground Truth.
Example: "GAP: External sources describe a new MRI-conditional DBS system approved in 2024
that is not mentioned in internal notes."

### UNIQUE INTERNAL VALUE
Specialized information in Ground Truth not found in World Truth.
Example: "UNIQUE: Internal notes contain surgeon-specific anatomical landmarks and
institutional complication data not published externally."

## Severity Classification for Conflicts

- **LOW**: Terminology differences, citation updates, minor clarifications
- **MEDIUM**: Dosage variations, timing differences, technique modifications
- **HIGH**: Contradictory clinical recommendations that could affect outcomes
- **CRITICAL**: Safety-relevant conflicts - drug interactions, contraindications,
  complications that could cause patient harm

## Evidence Hierarchy (for conflict resolution recommendations)
1. Randomized Controlled Trials / Meta-analyses
2. Prospective cohort studies
3. Retrospective studies / Case series
4. Expert opinion / Textbook content
5. Personal notes / Institutional protocols

Newer high-quality evidence generally supersedes older lower-quality evidence."""

GAP_ANALYSIS_USER_PROMPT = """## JUDGE'S TASK
Compare the following two knowledge sources for the clinical query: "{query}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SOURCE A: GROUND TRUTH (User's Internal Database)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{internal_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SOURCE B: WORLD TRUTH (Current External Sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{external_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RENDER YOUR VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide your analysis in this exact JSON format:
{{
    "agreements": [
        "Specific point where internal and external sources AGREE (validates internal data)"
    ],
    "conflicts": [
        {{
            "topic": "Specific clinical topic (e.g., 'Antibiotic prophylaxis duration')",
            "internal_claim": "What YOUR DATABASE says (be specific with dosages, timings)",
            "external_claim": "What CURRENT GUIDELINES say (cite the guideline year if known)",
            "severity": "LOW|MEDIUM|HIGH|CRITICAL",
            "recommendation": "Clinical action: e.g., 'Review 2024 IDSA DBS infection guidelines'"
        }}
    ],
    "gaps": [
        "Specific information found in external sources MISSING from internal database"
    ],
    "unique_internal": [
        "Valuable specialized information in internal notes NOT found externally"
    ],
    "confidence": 0.85,
    "alert_level": "NONE|CAUTION|WARNING|CRITICAL"
}}

IMPORTANT:
- Be SPECIFIC with clinical details (dosages, timings, drug names)
- For conflicts, quote the EXACT discrepancy, not vague descriptions
- Set alert_level based on highest severity conflict found
- If no conflicts, set alert_level to "NONE"

Respond ONLY with valid JSON, no other text."""


# =============================================================================
# Research Enricher
# =============================================================================

class ResearchEnricher:
    """
    Orchestrates dual-path retrieval and gap analysis.

    The enricher combines:
    1. Internal search (existing SearchService)
    2. External search (Perplexity/Gemini)
    3. Gap analysis (Claude comparison)
    4. Context merging (prioritized assembly)

    Attributes:
        internal: SearchService for internal database search
        perplexity: Optional Perplexity client
        gemini: Optional Gemini client
        config: Enricher configuration
    """

    def __init__(
        self,
        search_service,  # Type hint avoided to prevent circular import
        perplexity_client: Optional[PerplexitySearchClient] = None,
        gemini_client: Optional[GeminiDeepResearchClient] = None,
        config: Optional[EnricherConfig] = None,
        anthropic_client = None,  # For gap analysis
    ):
        """
        Initialize research enricher.

        Args:
            search_service: SearchService instance for internal search
            perplexity_client: Optional Perplexity client for web search
            gemini_client: Optional Gemini client for deep research
            config: Configuration options
            anthropic_client: Claude client for gap analysis
        """
        self.internal = search_service
        self.perplexity = perplexity_client
        self.gemini = gemini_client
        self.config = config or EnricherConfig()
        self._anthropic = anthropic_client

        # Conflict-aware merger for enhanced fact-level conflict detection
        self._merger = ConflictAwareMerger(anthropic_client)

        # Validate at least some capability
        logger.info(
            f"ResearchEnricher initialized: "
            f"internal={search_service is not None}, "
            f"perplexity={perplexity_client is not None}, "
            f"gemini={gemini_client is not None}"
        )

    @property
    def has_perplexity(self) -> bool:
        """Check if Perplexity is available."""
        return self.perplexity is not None

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini is available."""
        return self.gemini is not None

    @property
    def has_external(self) -> bool:
        """Check if any external search is available."""
        return self.has_perplexity or self.has_gemini

    def get_available_modes(self) -> List[SearchMode]:
        """Get list of available search modes."""
        modes = [SearchMode.STANDARD]

        if self.has_perplexity:
            modes.append(SearchMode.HYBRID)
            modes.append(SearchMode.EXTERNAL_ONLY)

        if self.has_gemini:
            modes.append(SearchMode.DEEP_RESEARCH)

        return modes

    async def enrich(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        filters = None,  # SearchFilters
        include_gap_analysis: bool = True,
    ) -> EnrichedContext:
        """
        Main entry point for enriched search.

        Args:
            query: User's question
            mode: Search mode (defaults to config.default_mode)
            filters: Optional filters for internal search
            include_gap_analysis: Whether to perform gap analysis

        Returns:
            EnrichedContext with merged results and gap report
        """
        start_time = time.time()
        mode = mode or self.config.default_mode

        # Validate mode availability
        mode = self._validate_mode(mode)

        logger.info(f"Enriching query with mode={mode.value}: {query[:50]}...")

        try:
            # Route based on mode
            if mode == SearchMode.STANDARD:
                return await self._standard_search(query, filters, start_time)

            elif mode == SearchMode.HYBRID:
                return await self._hybrid_search(
                    query, filters, include_gap_analysis, start_time
                )

            elif mode == SearchMode.DEEP_RESEARCH:
                return await self._deep_research(
                    query, filters, include_gap_analysis, start_time
                )

            elif mode == SearchMode.EXTERNAL_ONLY:
                return await self._external_only_search(query, start_time)

            elif mode == SearchMode.AUTO:
                return await self._auto_route(
                    query, filters, include_gap_analysis, start_time
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            # Fallback to standard mode on error
            logger.info("Falling back to standard mode")
            return await self._standard_search(query, filters, start_time)

    def _validate_mode(self, mode: SearchMode) -> SearchMode:
        """Validate and potentially downgrade mode if required services unavailable."""
        if mode == SearchMode.HYBRID and not self.has_perplexity:
            logger.warning("HYBRID mode requested but Perplexity unavailable, using STANDARD")
            return SearchMode.STANDARD

        if mode == SearchMode.DEEP_RESEARCH and not self.has_gemini:
            if self.has_perplexity:
                logger.warning("DEEP_RESEARCH mode requested but Gemini unavailable, using HYBRID")
                return SearchMode.HYBRID
            logger.warning("DEEP_RESEARCH mode requested but no external available, using STANDARD")
            return SearchMode.STANDARD

        if mode == SearchMode.EXTERNAL_ONLY and not self.has_external:
            logger.warning("EXTERNAL mode requested but no external available, using STANDARD")
            return SearchMode.STANDARD

        return mode

    # =========================================================================
    # Search Modes
    # =========================================================================

    async def _standard_search(
        self,
        query: str,
        filters,
        start_time: float
    ) -> EnrichedContext:
        """Standard internal-only search (existing behavior)."""
        internal_results = await self._search_internal(query, filters)

        return EnrichedContext(
            internal_chunks=internal_results,
            external_results=[],
            gap_report=None,
            synthesis_prompt=self._build_standard_prompt(query, internal_results),
            total_tokens=self._estimate_tokens(internal_results, []),
            internal_ratio=1.0,
            mode_used=SearchMode.STANDARD,
            search_time_ms=int((time.time() - start_time) * 1000)
        )

    async def _hybrid_search(
        self,
        query: str,
        filters,
        include_gap_analysis: bool,
        start_time: float
    ) -> EnrichedContext:
        """Hybrid search: internal + Perplexity web search."""
        # Execute both searches in parallel
        internal_task = self._search_internal(query, filters)
        external_task = self._search_perplexity(query)

        internal_results, external_results = await asyncio.gather(
            internal_task,
            external_task,
            return_exceptions=True
        )

        # Handle any exceptions
        if isinstance(internal_results, Exception):
            logger.error(f"Internal search failed: {internal_results}")
            internal_results = []

        if isinstance(external_results, Exception):
            logger.warning(f"External search failed: {external_results}")
            external_results = []

        # Perform gap analysis if enabled and we have both result types
        gap_report = None
        if include_gap_analysis and internal_results and external_results:
            gap_report = await self._analyze_gaps(
                query, internal_results, external_results
            )

        # Build enriched context
        return EnrichedContext(
            internal_chunks=internal_results,
            external_results=external_results,
            gap_report=gap_report,
            synthesis_prompt=self._build_hybrid_prompt(
                query, internal_results, external_results, gap_report
            ),
            total_tokens=self._estimate_tokens(internal_results, external_results),
            internal_ratio=self._calculate_internal_ratio(internal_results, external_results),
            mode_used=SearchMode.HYBRID,
            search_time_ms=int((time.time() - start_time) * 1000)
        )

    async def _deep_research(
        self,
        query: str,
        filters,
        include_gap_analysis: bool,
        start_time: float
    ) -> EnrichedContext:
        """Deep research: internal + Gemini with Google grounding."""
        # First get internal results
        internal_results = await self._search_internal(query, filters)

        # Summarize internal for Gemini context
        internal_summary = self._summarize_internal_for_context(internal_results)

        # Execute Gemini deep research with internal context
        external_results = await self._search_gemini(query, internal_summary)

        # Gap analysis
        gap_report = None
        if include_gap_analysis and internal_results and external_results:
            gap_report = await self._analyze_gaps(
                query, internal_results, external_results
            )

        return EnrichedContext(
            internal_chunks=internal_results,
            external_results=external_results,
            gap_report=gap_report,
            synthesis_prompt=self._build_deep_research_prompt(
                query, internal_results, external_results, gap_report
            ),
            total_tokens=self._estimate_tokens(internal_results, external_results),
            internal_ratio=self._calculate_internal_ratio(internal_results, external_results),
            mode_used=SearchMode.DEEP_RESEARCH,
            search_time_ms=int((time.time() - start_time) * 1000)
        )

    async def _external_only_search(
        self,
        query: str,
        start_time: float
    ) -> EnrichedContext:
        """External-only search (no internal database)."""
        external_results = await self._search_perplexity(query)

        return EnrichedContext(
            internal_chunks=[],
            external_results=external_results,
            gap_report=None,
            synthesis_prompt=self._build_external_only_prompt(query, external_results),
            total_tokens=self._estimate_tokens([], external_results),
            internal_ratio=0.0,
            mode_used=SearchMode.EXTERNAL_ONLY,
            search_time_ms=int((time.time() - start_time) * 1000)
        )

    async def _auto_route(
        self,
        query: str,
        filters,
        include_gap_analysis: bool,
        start_time: float
    ) -> EnrichedContext:
        """Automatically select best mode based on query analysis."""
        # Simple heuristics for now
        # TODO: Could use LLM for more sophisticated routing

        query_lower = query.lower()

        # Deep research indicators
        deep_indicators = [
            "compare", "protocol", "guideline", "latest", "recent",
            "draft", "write", "create", "synthesize"
        ]

        # External indicators
        external_indicators = [
            "weather", "news", "stock", "current price", "today"
        ]

        if any(ind in query_lower for ind in deep_indicators) and self.has_gemini:
            return await self._deep_research(query, filters, include_gap_analysis, start_time)

        if any(ind in query_lower for ind in external_indicators):
            if self.has_perplexity:
                return await self._external_only_search(query, start_time)

        # Default to hybrid if available, else standard
        if self.has_perplexity:
            return await self._hybrid_search(query, filters, include_gap_analysis, start_time)

        return await self._standard_search(query, filters, start_time)

    # =========================================================================
    # Search Helpers
    # =========================================================================

    async def _search_internal(
        self,
        query: str,
        filters
    ) -> List[Any]:
        """Execute internal database search."""
        if not self.internal:
            return []

        try:
            response = await self.internal.search(
                query=query,
                mode="hybrid",
                top_k=self.config.max_internal_tokens // 400,  # Rough chunk estimate
                filters=filters,
                rerank=True
            )
            return response.results if hasattr(response, 'results') else []
        except Exception as e:
            logger.error(f"Internal search error: {e}")
            return []

    async def _search_perplexity(
        self,
        query: str,
        max_results: int = 5
    ) -> List[ExternalSearchResult]:
        """Execute Perplexity web search."""
        if not self.perplexity:
            return []

        try:
            return await self.perplexity.search(
                query=query,
                max_results=max_results
            )
        except ExternalSearchError as e:
            logger.warning(f"Perplexity search failed: {e}")
            return []

    async def _search_gemini(
        self,
        query: str,
        internal_context: str = ""
    ) -> List[ExternalSearchResult]:
        """Execute Gemini deep research."""
        if not self.gemini:
            return []

        try:
            return await self.gemini.deep_research(
                query=query,
                internal_context=internal_context
            )
        except ExternalSearchError as e:
            logger.warning(f"Gemini search failed: {e}")
            return []

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    async def _analyze_gaps(
        self,
        query: str,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult]
    ) -> Optional[GapReport]:
        """
        Analyze gaps between internal and external sources.

        Uses Claude to compare sources and identify:
        - Agreements
        - Conflicts
        - Gaps
        - Unique internal content
        """
        if not self.config.enable_gap_analysis:
            return None

        if not self._anthropic:
            logger.debug("Gap analysis skipped: no Anthropic client")
            return GapReport.empty()

        start_time = time.time()

        try:
            # Prepare summaries
            internal_summary = self._summarize_internal_for_analysis(internal_results)
            external_summary = self._summarize_external_for_analysis(external_results)

            # Build prompt
            prompt = GAP_ANALYSIS_USER_PROMPT.format(
                query=query,
                internal_summary=internal_summary,
                external_summary=external_summary
            )

            # Call Claude for analysis
            response = await self._anthropic.messages.create(
                model=self.config.gap_analysis_model,
                max_tokens=1500,
                system=GAP_ANALYSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            content = response.content[0].text
            report = self._parse_gap_analysis(content)

            # Enhance with deterministic conflict detection (quantitative, recommendations)
            # This catches conflicts the LLM might miss, especially numeric discrepancies
            report = self._enhance_with_deterministic_conflicts(
                report,
                internal_summary,
                external_summary
            )

            report.analysis_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Gap analysis complete: {report.conflict_count} conflicts, "
                f"{report.gap_count} gaps (incl. deterministic)"
            )

            return report

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return GapReport.empty()

    def _parse_gap_analysis(self, content: str) -> GapReport:
        """Parse Claude's gap analysis response."""
        import json

        try:
            # Try to extract JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Parse conflicts
            conflicts = []
            for c in data.get("conflicts", []):
                severity_str = c.get("severity", "MEDIUM").upper()
                try:
                    severity = ConflictSeverity[severity_str]
                except KeyError:
                    severity = ConflictSeverity.MEDIUM

                conflict = Conflict(
                    internal_claim=c.get("internal_claim", ""),
                    external_claim=c.get("external_claim", ""),
                    internal_source="Internal database",
                    external_source="Web search",
                    severity=severity,
                    topic=c.get("topic", "Unknown"),
                    recommendation=c.get("recommendation")
                )
                conflicts.append(conflict)

            return GapReport(
                agreements=data.get("agreements", []),
                conflicts=conflicts,
                gaps=data.get("gaps", []),
                unique_internal=data.get("unique_internal", []),
                confidence=data.get("confidence", 0.8),
                alert_level=data.get("alert_level", "none")
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse gap analysis: {e}")
            return GapReport.empty()

    def _enhance_with_deterministic_conflicts(
        self,
        report: GapReport,
        internal_content: str,
        external_content: str
    ) -> GapReport:
        """
        Enhance LLM-based gap report with deterministic conflict detection.

        The merger extracts quantitative facts and recommendations, then
        detects conflicts based on value comparisons. This catches conflicts
        the LLM might miss, especially numeric discrepancies.

        Args:
            report: LLM-generated gap report
            internal_content: Combined internal source content
            external_content: Combined external source content

        Returns:
            Enhanced GapReport with additional deterministic conflicts
        """
        try:
            # Use merger to extract facts and detect conflicts
            internal_facts = self._merger._extract_facts(internal_content, "internal")
            external_facts = self._merger._extract_facts(external_content, "external")
            detected = self._merger._detect_conflicts(internal_facts, external_facts)

            if not detected:
                return report

            # Convert DetectedConflict to Conflict and add to report
            existing_topics = {c.topic.lower() for c in report.conflicts}
            new_conflicts = []

            for dc in detected:
                # Skip if LLM already found a similar conflict
                topic = dc.description.lower()
                if any(topic in t or t in topic for t in existing_topics):
                    continue

                # Map ConflictCategory to ConflictSeverity
                severity_map = {
                    ConflictCategory.TEMPORAL: ConflictSeverity.MEDIUM,
                    ConflictCategory.ESTABLISHED_FACT: ConflictSeverity.HIGH,
                    ConflictCategory.QUANTITATIVE: ConflictSeverity.MEDIUM,
                    ConflictCategory.APPROACH: ConflictSeverity.LOW,
                    ConflictCategory.RECOMMENDATION: ConflictSeverity.HIGH,
                }
                if dc.severity == "high":
                    severity = ConflictSeverity.HIGH
                elif dc.severity == "critical":
                    severity = ConflictSeverity.CRITICAL
                else:
                    severity = severity_map.get(dc.category, ConflictSeverity.MEDIUM)

                # Build recommendation based on resolution strategy
                rec_map = {
                    ResolutionStrategy.PREFER_INTERNAL: "Defer to established internal source",
                    ResolutionStrategy.PREFER_EXTERNAL: "Consider updating to current guidelines",
                    ResolutionStrategy.NOTE_BOTH: "Present both values for clinical judgment",
                    ResolutionStrategy.FLAG_FOR_REVIEW: "Requires expert review before use",
                }
                recommendation = rec_map.get(dc.resolution_strategy, "Verify current guidelines")

                conflict = Conflict(
                    internal_claim=dc.internal_claim[:200],
                    external_claim=dc.external_claim[:200],
                    internal_source="Internal database (deterministic)",
                    external_source="External search (deterministic)",
                    severity=severity,
                    topic=f"[AUTO] {dc.category.value}: {dc.description}",
                    recommendation=recommendation
                )
                new_conflicts.append(conflict)

            if new_conflicts:
                logger.info(f"Deterministic analysis found {len(new_conflicts)} additional conflicts")
                report.conflicts.extend(new_conflicts)
                # Recalculate alert level
                report.alert_level = report.computed_alert_level

            return report

        except Exception as e:
            logger.warning(f"Deterministic conflict detection failed: {e}")
            return report

    def _summarize_internal_for_analysis(self, results: List[Any]) -> str:
        """Create summary of internal results for gap analysis."""
        if not results:
            return "(No internal results)"

        summaries = []
        for i, r in enumerate(results[:5], 1):
            content = getattr(r, 'content', str(r))[:300]
            source = getattr(r, 'source_document', 'Unknown')
            summaries.append(f"[{i}] {source}: {content}")

        return "\n\n".join(summaries)

    def _summarize_external_for_analysis(
        self,
        results: List[ExternalSearchResult]
    ) -> str:
        """Create summary of external results for gap analysis."""
        if not results:
            return "(No external results)"

        summaries = []
        for i, r in enumerate(results[:5], 1):
            content = r.content[:300] if r.content else r.snippet
            summaries.append(f"[W{i}] {r.source_title}: {content}")

        return "\n\n".join(summaries)

    def _summarize_internal_for_context(self, results: List[Any]) -> str:
        """Create concise summary for Gemini context injection."""
        if not results:
            return ""

        summaries = []
        for r in results[:3]:
            content = getattr(r, 'content', str(r))[:200]
            summaries.append(content)

        return " | ".join(summaries)

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_standard_prompt(
        self,
        query: str,
        internal_results: List[Any]
    ) -> str:
        """Build prompt for standard (internal-only) mode."""
        context_parts = []

        for i, r in enumerate(internal_results[:10], 1):
            content = getattr(r, 'content', str(r))
            source = getattr(r, 'source_document', 'Unknown')
            page = getattr(r, 'page_number', '')
            authority = getattr(r, 'authority_score', 0.8)

            context_parts.append(
                f"[{i}] {source}" + (f" (p.{page})" if page else "") +
                f" [Authority: {authority:.2f}]\n{content}"
            )

        context = "\n\n".join(context_parts)

        return f"""## Context (Internal Database)
{context}

## Question
{query}

Please provide a comprehensive answer based on the context above.
Use citations [1], [2], etc. to reference your sources."""

    def _build_hybrid_prompt(
        self,
        query: str,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult],
        gap_report: Optional[GapReport]
    ) -> str:
        """Build prompt for hybrid mode with source separation and explicit alerts."""
        # Internal sources - "Your Data"
        internal_parts = []
        for i, r in enumerate(internal_results[:8], 1):
            content = getattr(r, 'content', str(r))
            source = getattr(r, 'source_document', 'Unknown')
            authority = getattr(r, 'authority_score', 0.8)
            internal_parts.append(
                f"[{i}] {source} (Authority: {authority:.2f})\n{content}"
            )

        # External sources - "Current Research"
        external_parts = []
        for i, r in enumerate(external_results[:5], 1):
            external_parts.append(
                f"[W{i}] {r.source_title}\n{r.content[:400]}\nURL: {r.source_url}"
            )

        # Build protocol alert section if conflicts exist
        alert_section = ""
        if gap_report and gap_report.has_conflicts:
            alert_lines = [""]
            alert_lines.append("═══════════════════════════════════════════════════════════")
            alert_lines.append("⚠️  PROTOCOL ALERT: CONFLICTS DETECTED BETWEEN YOUR DATA AND CURRENT GUIDELINES")
            alert_lines.append("═══════════════════════════════════════════════════════════")

            for c in gap_report.conflicts[:3]:
                severity_icon = {
                    'LOW': 'ℹ️',
                    'MEDIUM': '⚠️',
                    'HIGH': '🔶',
                    'CRITICAL': '🚨'
                }.get(c.severity.value if hasattr(c.severity, 'value') else c.severity, '⚠️')

                alert_lines.append(f"\n{severity_icon} [{c.severity.value if hasattr(c.severity, 'value') else c.severity}] {c.topic}")
                alert_lines.append(f"   📚 YOUR DATABASE: {c.internal_claim[:150]}")
                alert_lines.append(f"   🌐 CURRENT STANDARD: {c.external_claim[:150]}")
                if c.recommendation:
                    alert_lines.append(f"   ✅ RECOMMENDED ACTION: {c.recommendation}")

            alert_lines.append("\n═══════════════════════════════════════════════════════════")
            alert_section = "\n".join(alert_lines)

        # Build knowledge gaps section
        gaps_section = ""
        if gap_report and gap_report.gaps:
            gaps_lines = ["\n### 📋 Knowledge Gaps Identified:"]
            gaps_lines.append("(Information in current sources NOT found in your database)")
            for g in gap_report.gaps[:3]:
                gaps_lines.append(f"• {g}")
            gaps_section = "\n".join(gaps_lines)

        # Build unique internal value section
        unique_section = ""
        if gap_report and gap_report.unique_internal:
            unique_lines = ["\n### 💎 Unique Value in Your Database:"]
            unique_lines.append("(Specialized information NOT found in external sources)")
            for u in gap_report.unique_internal[:3]:
                unique_lines.append(f"• {u}")
            unique_section = "\n".join(unique_lines)

        prompt = f"""
══════════════════════════════════════════════════════════════════════════════
## 📚 YOUR DATA (Internal Database - Ground Truth)
══════════════════════════════════════════════════════════════════════════════
{chr(10).join(internal_parts)}

══════════════════════════════════════════════════════════════════════════════
## 🌐 CURRENT RESEARCH (Live Web Search - World Truth)
══════════════════════════════════════════════════════════════════════════════
{chr(10).join(external_parts)}
{alert_section}{gaps_section}{unique_section}

══════════════════════════════════════════════════════════════════════════════
## ❓ QUESTION
══════════════════════════════════════════════════════════════════════════════
{query}

══════════════════════════════════════════════════════════════════════════════
## 📝 RESPONSE INSTRUCTIONS
══════════════════════════════════════════════════════════════════════════════

**CRITICAL: If conflicts were detected above, you MUST:**
1. Start your response with a ⚠️ PROTOCOL ALERT block summarizing the conflict
2. Clearly state what YOUR DATABASE says vs what CURRENT GUIDELINES say
3. Present BOTH perspectives - do not hide the discrepancy

**Citation Format:**
- Use [N] for YOUR DATABASE citations (e.g., [1], [2])
- Use [WN] for CURRENT RESEARCH citations (e.g., [W1], [W2])

**Response Structure:**
1. If conflicts exist: Begin with Protocol Alert block
2. Synthesize information from BOTH sources
3. Clearly attribute which recommendation comes from which source
4. For conflicting information: "Your notes indicate X [1], however current guidelines recommend Y [W1]"
5. End with clinical implications if relevant

**Priority Order:**
- For established surgical ANATOMY/TECHNIQUE: Prioritize your curated database
- For PROTOCOLS/DOSAGES/GUIDELINES: Give weight to more recent external sources
- For COMPLICATIONS/OUTCOMES: Cross-reference both sources"""

        return prompt

    def _build_deep_research_prompt(
        self,
        query: str,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult],
        gap_report: Optional[GapReport]
    ) -> str:
        """Build prompt for deep research mode."""
        # Similar to hybrid but with more emphasis on synthesis
        base_prompt = self._build_hybrid_prompt(
            query, internal_results, external_results, gap_report
        )

        # Add deep research specific instructions
        return base_prompt + """

## Deep Research Notes
This query was processed with deep reasoning capabilities.
The external sources include Google Search grounded information.
Provide a thorough, nuanced response suitable for clinical decision-making."""

    def _build_external_only_prompt(
        self,
        query: str,
        external_results: List[ExternalSearchResult]
    ) -> str:
        """Build prompt for external-only mode."""
        external_parts = []
        for i, r in enumerate(external_results[:10], 1):
            external_parts.append(
                f"[W{i}] {r.source_title}\n{r.content[:500]}\nURL: {r.source_url}"
            )

        return f"""## External Sources (Web Search)
{chr(10).join(external_parts)}

## Question
{query}

Please provide an answer based on the web search results above.
Use [WN] citations to reference sources.
Note: This response is based entirely on external web search, not internal database."""

    async def merge_with_conflict_resolution(
        self,
        topic: str,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult]
    ) -> MergeResult:
        """
        Perform conflict-aware merge of internal and external content.

        This uses the ConflictAwareMerger for deep research scenarios where
        we want actual merged content (not just conflict detection).

        Args:
            topic: Query topic
            internal_results: Internal search results
            external_results: External search results

        Returns:
            MergeResult with resolved_content and conflict details
        """
        # Extract content strings
        internal_content = "\n\n".join(
            getattr(r, 'content', str(r))[:500]
            for r in internal_results[:5]
        )
        external_content = "\n\n".join(
            r.content[:500] for r in external_results[:5]
        )

        # Use merger to produce resolved content
        result = await self._merger.merge(
            topic=topic,
            internal_content=internal_content,
            external_content=external_content,
        )

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def _estimate_tokens(
        self,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult]
    ) -> int:
        """Rough token estimation for context."""
        internal_chars = sum(
            len(getattr(r, 'content', str(r)))
            for r in internal_results
        )
        external_chars = sum(
            len(r.content) for r in external_results
        )

        # Rough estimate: 4 chars per token
        return (internal_chars + external_chars) // 4

    def _calculate_internal_ratio(
        self,
        internal_results: List[Any],
        external_results: List[ExternalSearchResult]
    ) -> float:
        """Calculate ratio of internal to total sources."""
        internal_count = len(internal_results)
        external_count = len(external_results)
        total = internal_count + external_count

        if total == 0:
            return 0.0

        return internal_count / total


# =============================================================================
# Factory Function
# =============================================================================

def create_enricher_from_env(
    search_service,
    anthropic_client = None
) -> ResearchEnricher:
    """
    Create ResearchEnricher with clients configured from environment.

    Reads:
    - PERPLEXITY_API_KEY
    - GEMINI_API_KEY / GOOGLE_API_KEY
    - ENABLE_EXTERNAL_SEARCH
    - DEFAULT_SEARCH_MODE

    Args:
        search_service: SearchService instance
        anthropic_client: Optional Claude client for gap analysis

    Returns:
        Configured ResearchEnricher
    """
    import os
    from src.research.external_search import (
        create_perplexity_client,
        create_gemini_client
    )

    # Check if external search is enabled
    if not os.getenv("ENABLE_EXTERNAL_SEARCH", "true").lower() == "true":
        logger.info("External search disabled via ENABLE_EXTERNAL_SEARCH=false")
        return ResearchEnricher(
            search_service=search_service,
            perplexity_client=None,
            gemini_client=None,
            anthropic_client=anthropic_client
        )

    # Create clients
    perplexity = create_perplexity_client()
    gemini = create_gemini_client()

    # Create config
    default_mode_str = os.getenv("DEFAULT_SEARCH_MODE", "hybrid")
    try:
        default_mode = SearchMode(default_mode_str)
    except ValueError:
        default_mode = SearchMode.HYBRID

    config = EnricherConfig(default_mode=default_mode)

    return ResearchEnricher(
        search_service=search_service,
        perplexity_client=perplexity,
        gemini_client=gemini,
        config=config,
        anthropic_client=anthropic_client
    )
