"""
Gap Filling Service
===================

Fills detected gaps using external research or additional internal search.
Integrates with ResearchEnricher for external content retrieval.

Strategies:
- NONE: Don't fill, just report gaps
- HIGH_PRIORITY_ONLY: Only fill HIGH and CRITICAL gaps
- ALL_WITH_FALLBACK: Try internal first, then external
- ALWAYS_EXTERNAL: Always fetch external for enrichment
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .gap_models import (
    Gap,
    GapFillResult,
    GapFillStrategy,
    GapPriority,
    GapType,
)

logger = logging.getLogger(__name__)


class SearchService(Protocol):
    """Protocol for search service interface."""

    async def search(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search internal corpus."""
        ...


class ResearchEnricher(Protocol):
    """Protocol for external research enrichment."""

    async def fetch_external(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Fetch external research content."""
        ...


@dataclass
class GapFillConfig:
    """Configuration for gap filling."""

    strategy: GapFillStrategy = GapFillStrategy.HIGH_PRIORITY_ONLY
    max_external_sources: int = 3
    internal_search_limit: int = 5
    timeout_seconds: int = 30
    min_content_length: int = 100


class GapFillingService:
    """
    Fills detected gaps with content from internal or external sources.

    Usage:
        service = GapFillingService(search_service, research_enricher)

        results = await service.fill_gaps(
            gaps=gap_report.high_priority_gaps,
            strategy=GapFillStrategy.HIGH_PRIORITY_ONLY,
        )
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        research_enricher: Optional[ResearchEnricher] = None,
        anthropic_client: Optional[Any] = None,
    ):
        self.search_service = search_service
        self.research_enricher = research_enricher
        self.anthropic_client = anthropic_client

    async def fill_gaps(
        self,
        gaps: List[Gap],
        strategy: GapFillStrategy = GapFillStrategy.HIGH_PRIORITY_ONLY,
        config: Optional[GapFillConfig] = None,
    ) -> List[GapFillResult]:
        """
        Fill gaps based on the selected strategy.

        Args:
            gaps: List of gaps to fill
            strategy: Filling strategy
            config: Optional configuration

        Returns:
            List of fill results
        """
        if config is None:
            config = GapFillConfig(strategy=strategy)

        results = []

        for gap in gaps:
            if self._should_fill(gap, strategy):
                start_time = time.time()
                result = await self._fill_single_gap(gap, strategy, config)
                result.fill_duration_ms = int((time.time() - start_time) * 1000)
                results.append(result)

        return results

    def _should_fill(self, gap: Gap, strategy: GapFillStrategy) -> bool:
        """Determine if a gap should be filled based on strategy."""
        if strategy == GapFillStrategy.NONE:
            return False

        if strategy == GapFillStrategy.HIGH_PRIORITY_ONLY:
            return gap.priority in (GapPriority.CRITICAL, GapPriority.HIGH)

        if strategy == GapFillStrategy.ALL_WITH_FALLBACK:
            return gap.auto_fill_available

        if strategy == GapFillStrategy.ALWAYS_EXTERNAL:
            return True

        return False

    async def _fill_single_gap(
        self,
        gap: Gap,
        strategy: GapFillStrategy,
        config: GapFillConfig,
    ) -> GapFillResult:
        """Fill a single gap."""
        query = gap.external_query or gap.topic

        # Try internal search first (except for ALWAYS_EXTERNAL)
        if strategy != GapFillStrategy.ALWAYS_EXTERNAL:
            internal_result = await self._try_internal_fill(gap, query, config)
            if internal_result and internal_result.fill_successful:
                return internal_result

        # Try external research
        if strategy in (
            GapFillStrategy.HIGH_PRIORITY_ONLY,
            GapFillStrategy.ALL_WITH_FALLBACK,
            GapFillStrategy.ALWAYS_EXTERNAL,
        ):
            external_result = await self._try_external_fill(gap, query, config)
            if external_result:
                return external_result

        # Return failed result
        return GapFillResult(
            gap_id=gap.gap_id,
            gap_type=gap.gap_type,
            topic=gap.topic,
            fill_successful=False,
            fill_source="failed",
            error_message="No content found for gap",
        )

    async def _try_internal_fill(
        self,
        gap: Gap,
        query: str,
        config: GapFillConfig,
    ) -> Optional[GapFillResult]:
        """Attempt to fill gap from internal corpus."""
        if not self.search_service:
            return None

        try:
            results = await self.search_service.search(
                query=query,
                limit=config.internal_search_limit,
            )

            if not results:
                return None

            # Combine content from search results
            content_parts = []
            for result in results:
                content = result.get("content", result.get("chunk_content", ""))
                if content and len(content) >= config.min_content_length:
                    content_parts.append(content)

            if not content_parts:
                return None

            combined_content = "\n\n".join(content_parts[:3])

            return GapFillResult(
                gap_id=gap.gap_id,
                gap_type=gap.gap_type,
                topic=gap.topic,
                fill_successful=True,
                fill_source="internal",
                filled_content=combined_content,
            )

        except Exception as e:
            logger.warning(f"Internal fill failed for gap '{gap.topic}': {e}")
            return None

    async def _try_external_fill(
        self,
        gap: Gap,
        query: str,
        config: GapFillConfig,
    ) -> Optional[GapFillResult]:
        """Attempt to fill gap from external research."""
        if not self.research_enricher:
            return None

        try:
            # Add neurosurgery context to query
            enriched_query = f"{query} neurosurgery"

            result = await self.research_enricher.fetch_external(
                query=enriched_query,
                max_results=config.max_external_sources,
            )

            if not result or not result.get("content"):
                return None

            content = result.get("content", "")
            sources = result.get("sources", [])

            if len(content) < config.min_content_length:
                return None

            return GapFillResult(
                gap_id=gap.gap_id,
                gap_type=gap.gap_type,
                topic=gap.topic,
                fill_successful=True,
                fill_source="external",
                filled_content=content,
                external_sources=[
                    {"url": s.get("url", ""), "title": s.get("title", "")}
                    for s in sources[:config.max_external_sources]
                ],
            )

        except Exception as e:
            logger.warning(f"External fill failed for gap '{gap.topic}': {e}")
            return None

    async def fill_gaps_with_synthesis(
        self,
        gaps: List[Gap],
        strategy: GapFillStrategy,
        original_content: str,
    ) -> str:
        """
        Fill gaps and synthesize into original content.

        Args:
            gaps: Gaps to fill
            strategy: Filling strategy
            original_content: Original synthesized content

        Returns:
            Enhanced content with filled gaps integrated
        """
        if strategy == GapFillStrategy.NONE:
            return original_content

        fill_results = await self.fill_gaps(gaps, strategy)

        # Filter successful fills
        successful_fills = [r for r in fill_results if r.fill_successful]

        if not successful_fills:
            return original_content

        # If no LLM available, append as addendum
        if not self.anthropic_client:
            return self._append_fill_results(original_content, successful_fills)

        # Use LLM to integrate fills
        return await self._integrate_with_llm(
            original_content, successful_fills
        )

    def _append_fill_results(
        self,
        original_content: str,
        fill_results: List[GapFillResult],
    ) -> str:
        """Append fill results as addendum."""
        addendum_parts = ["\n\n---\n\n**Additional Information**\n"]

        for result in fill_results:
            source_tag = "[Internal]" if result.fill_source == "internal" else "[External]"
            addendum_parts.append(f"\n**{result.topic}** {source_tag}\n")
            addendum_parts.append(result.filled_content[:1500])

            if result.external_sources:
                addendum_parts.append("\n*Sources:*\n")
                for src in result.external_sources[:3]:
                    addendum_parts.append(f"- {src.get('title', src.get('url', 'Source'))}\n")

        return original_content + "".join(addendum_parts)

    async def _integrate_with_llm(
        self,
        original_content: str,
        fill_results: List[GapFillResult],
    ) -> str:
        """Use LLM to integrate fill results into content."""
        try:
            # Prepare fill content
            fills_text = "\n\n".join(
                f"**Gap: {r.topic}**\n{r.filled_content[:1000]}"
                for r in fill_results
            )

            prompt = f"""Integrate the following gap-fill content into the original document.
Preserve the original structure and flow. Add the new information in appropriate sections.
Mark new content with [Added] tags.

ORIGINAL CONTENT:
{original_content[:6000]}

GAP FILL CONTENT:
{fills_text[:3000]}

INTEGRATED CONTENT:"""

            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.warning(f"LLM integration failed: {e}")
            return self._append_fill_results(original_content, fill_results)


# Convenience function
async def fill_gaps(
    gaps: List[Gap],
    strategy: GapFillStrategy,
    search_service: Optional[SearchService] = None,
    research_enricher: Optional[ResearchEnricher] = None,
) -> List[GapFillResult]:
    """Convenience function to fill gaps."""
    service = GapFillingService(search_service, research_enricher)
    return await service.fill_gaps(gaps, strategy)
