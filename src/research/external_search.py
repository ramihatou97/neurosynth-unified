"""
NeuroSynth Unified - External Search Clients
=============================================

Clients for external search providers:
- PerplexitySearchClient: Live web search via Perplexity Sonar API
- GeminiDeepResearchClient: Deep reasoning with Google Search grounding

Both clients implement:
- Async operation
- Retry with exponential backoff
- Rate limiting
- Structured result parsing
- Medical query optimization
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from urllib.parse import urlparse

import aiohttp

from src.research.models import (
    ExternalSearchResult,
    ExternalSearchConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class ExternalSearchError(Exception):
    """Base exception for external search errors."""
    pass


class RateLimitError(ExternalSearchError):
    """Rate limit exceeded."""
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"{provider} rate limit exceeded" +
            (f", retry after {retry_after}s" if retry_after else "")
        )


class APIKeyError(ExternalSearchError):
    """API key missing or invalid."""
    def __init__(self, provider: str, message: str = "API key invalid"):
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ProviderError(ExternalSearchError):
    """Provider-specific error."""
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} error ({status_code}): {message}")


# =============================================================================
# Base Client
# =============================================================================

class ExternalSearchClient(ABC):
    """Abstract base class for external search clients."""

    def __init__(
        self,
        api_key: str,
        config: Optional[ExternalSearchConfig] = None
    ):
        self.api_key = api_key
        self.config = config or ExternalSearchConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_request_time: Optional[datetime] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.perplexity_timeout)
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _retry_with_backoff(
        self,
        operation: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except RateLimitError as e:
                wait_time = e.retry_after or (2 ** attempt)
                logger.warning(
                    f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                last_error = e
            except aiohttp.ClientError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Request failed: {e}, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                last_error = ExternalSearchError(str(e))
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                last_error = e
                break

        raise last_error or ExternalSearchError("Max retries exceeded")

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[ExternalSearchResult]:
        """Execute search and return structured results."""
        pass


# =============================================================================
# Perplexity Client
# =============================================================================

class PerplexitySearchClient(ExternalSearchClient):
    """
    Perplexity Sonar API client for live web search.

    Perplexity's Sonar models are optimized for search and provide:
    - Real-time web search with citations
    - Structured responses with source attribution
    - Medical/scientific query understanding

    Usage:
        client = PerplexitySearchClient(api_key="pplx-xxx")
        results = await client.search("latest DBS infection guidelines 2024")

        for r in results:
            print(f"[{r.source_title}] {r.snippet}")
            print(f"  URL: {r.source_url}")
    """

    BASE_URL = "https://api.perplexity.ai"

    # Medical search system prompt
    MEDICAL_SYSTEM_PROMPT = """You are a medical research assistant helping a neurosurgeon find current, evidence-based information.

When searching for medical topics:
1. Prioritize peer-reviewed sources (PubMed, medical journals)
2. Include clinical guidelines from medical societies when relevant
3. Note publication dates - recent is usually better for guidelines
4. Cite specific sources with URLs

Format your response as a structured research summary with clear citations."""

    def __init__(
        self,
        api_key: str,
        config: Optional[ExternalSearchConfig] = None
    ):
        super().__init__(api_key, config)
        self.model = config.perplexity_model if config else "sonar-pro"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_filter: Optional[str] = None
    ) -> List[ExternalSearchResult]:
        """
        Execute web search via Perplexity API.

        Args:
            query: Search query (will be optimized for medical search)
            max_results: Maximum number of results to return
            recency_filter: Time filter ("week", "month", "year", None)

        Returns:
            List of ExternalSearchResult with citations
        """
        return await self._retry_with_backoff(
            self._execute_search,
            query,
            max_results,
            recency_filter or self.config.perplexity_recency_filter
        )

    async def _execute_search(
        self,
        query: str,
        max_results: int,
        recency_filter: Optional[str]
    ) -> List[ExternalSearchResult]:
        """Internal search execution."""
        session = await self._get_session()

        # Build request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.MEDICAL_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Search and summarize current information about: {query}"
                }
            ],
            "return_citations": True,
            "return_related_questions": False,
        }

        # Add search recency filter if specified
        if recency_filter:
            payload["search_recency_filter"] = recency_filter

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.debug(f"Perplexity search: {query[:50]}...")

        async with session.post(
            f"{self.BASE_URL}/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            # Handle errors
            if response.status == 401:
                raise APIKeyError("Perplexity", "Invalid or expired API key")
            elif response.status == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError("Perplexity", int(retry_after) if retry_after else None)
            elif response.status != 200:
                text = await response.text()
                raise ProviderError("Perplexity", response.status, text)

            data = await response.json()

        # Parse response
        return self._parse_response(data, max_results)

    def _parse_response(
        self,
        data: Dict[str, Any],
        max_results: int
    ) -> List[ExternalSearchResult]:
        """Parse Perplexity API response into structured results."""
        results = []

        # Extract the assistant message
        choices = data.get("choices", [])
        if not choices:
            logger.warning("No choices in Perplexity response")
            return results

        message = choices[0].get("message", {})
        content = message.get("content", "")
        citations = data.get("citations", [])

        # If we have structured citations, use those
        if citations:
            for i, citation in enumerate(citations[:max_results]):
                # Extract domain from URL
                url = citation if isinstance(citation, str) else citation.get("url", "")
                domain = urlparse(url).netloc if url else "unknown"

                # Create structured result
                result = ExternalSearchResult(
                    content=self._extract_citation_context(content, i + 1),
                    source_url=url,
                    source_title=citation.get("title", domain) if isinstance(citation, dict) else domain,
                    relevance_score=1.0 - (i * 0.1),  # Rank-based score
                    provider="perplexity",
                    retrieved_at=datetime.utcnow(),
                    snippet=self._extract_snippet(content, i + 1),
                    metadata={"citation_index": i + 1}
                )
                results.append(result)

        # If no structured citations, parse from content
        if not results and content:
            results = self._parse_unstructured_content(content, max_results)

        logger.info(f"Perplexity returned {len(results)} results")
        return results

    def _extract_citation_context(self, content: str, index: int) -> str:
        """Extract context around a citation reference."""
        # Look for [index] in the content
        pattern = rf'\[{index}\][^[]*'
        matches = re.findall(pattern, content)
        if matches:
            return matches[0].strip()
        return ""

    def _extract_snippet(self, content: str, index: int) -> str:
        """Extract a short snippet for the citation."""
        context = self._extract_citation_context(content, index)
        if context:
            # Clean and truncate
            clean = re.sub(r'\[\d+\]', '', context).strip()
            return clean[:200] + "..." if len(clean) > 200 else clean
        return ""

    def _parse_unstructured_content(
        self,
        content: str,
        max_results: int
    ) -> List[ExternalSearchResult]:
        """Parse content without structured citations."""
        results = []

        # Try to find URLs in content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)

        for i, url in enumerate(urls[:max_results]):
            domain = urlparse(url).netloc

            result = ExternalSearchResult(
                content=content,
                source_url=url,
                source_title=domain,
                relevance_score=0.8 - (i * 0.05),
                provider="perplexity",
                retrieved_at=datetime.utcnow(),
                snippet=f"Source from {domain}",
                metadata={"parsed_from_content": True}
            )
            results.append(result)

        # If no URLs found, create a single result from content
        if not results and content:
            result = ExternalSearchResult(
                content=content,
                source_url="",
                source_title="Perplexity Search",
                relevance_score=0.7,
                provider="perplexity",
                retrieved_at=datetime.utcnow(),
                snippet=content[:200] + "..." if len(content) > 200 else content,
                metadata={"no_citations": True}
            )
            results.append(result)

        return results


# =============================================================================
# Gemini Deep Research Client
# =============================================================================

class GeminiDeepResearchClient(ExternalSearchClient):
    """
    Gemini 2.5 Pro client with Google Search grounding.

    Used for complex queries requiring:
    - Multi-hop reasoning
    - Deep synthesis across multiple sources
    - Comparison with internal context

    The client uses Gemini's Google Search grounding feature to
    ensure responses are factually accurate and well-cited.

    Usage:
        client = GeminiDeepResearchClient(api_key="AIza-xxx")
        results = await client.deep_research(
            query="Compare DBS programming protocols",
            internal_context="Our protocol uses..."
        )
    """

    DEEP_RESEARCH_PROMPT = """You are a senior neurosurgical researcher with access to Google Search.

Your task is to:
1. Research the given query using current web sources
2. Compare findings with the provided internal context (if any)
3. Identify any discrepancies or updates
4. Provide a comprehensive, well-cited response

For medical topics:
- Prioritize peer-reviewed sources and clinical guidelines
- Note publication dates for time-sensitive recommendations
- Flag any conflicts between sources
- Be precise about evidence levels

Internal Context (User's Notes):
{internal_context}

Research Query:
{query}

Provide a structured response with:
1. Key Findings (with citations)
2. Comparison with Internal Context (if provided)
3. Recent Updates or Changes
4. Evidence Quality Assessment"""

    def __init__(
        self,
        api_key: str,
        config: Optional[ExternalSearchConfig] = None
    ):
        super().__init__(api_key, config)
        self._model = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy initialization of Gemini client."""
        if not self._initialized:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)

                # Configure model with grounding
                generation_config = {
                    "temperature": self.config.gemini_temperature,
                    "max_output_tokens": self.config.gemini_max_tokens,
                }

                # Enable Google Search grounding if configured
                tools = None
                if self.config.gemini_enable_grounding:
                    tools = [{"google_search_retrieval": {}}]

                self._model = genai.GenerativeModel(
                    self.config.gemini_model,
                    generation_config=generation_config,
                    tools=tools
                )
                self._initialized = True
                logger.info(f"Gemini client initialized: {self.config.gemini_model}")

            except ImportError:
                raise ExternalSearchError(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
                )
            except Exception as e:
                raise APIKeyError("Gemini", str(e))

    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[ExternalSearchResult]:
        """
        Basic search (without internal context comparison).

        For deep research with comparison, use deep_research() instead.
        """
        return await self.deep_research(query, internal_context="")

    async def deep_research(
        self,
        query: str,
        internal_context: str = "",
        max_tokens: Optional[int] = None
    ) -> List[ExternalSearchResult]:
        """
        Execute deep research with optional internal context comparison.

        Args:
            query: Research query
            internal_context: Summary of internal sources for comparison
            max_tokens: Override max output tokens

        Returns:
            List of ExternalSearchResult with grounded citations
        """
        return await self._retry_with_backoff(
            self._execute_deep_research,
            query,
            internal_context,
            max_tokens
        )

    async def _execute_deep_research(
        self,
        query: str,
        internal_context: str,
        max_tokens: Optional[int]
    ) -> List[ExternalSearchResult]:
        """Internal deep research execution."""
        await self._ensure_initialized()

        # Build prompt
        prompt = self.DEEP_RESEARCH_PROMPT.format(
            internal_context=internal_context or "(No internal context provided)",
            query=query
        )

        logger.debug(f"Gemini deep research: {query[:50]}...")

        try:
            # Generate response
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt
            )

            # Check for safety blocks
            if not response.candidates:
                logger.warning("Gemini returned no candidates")
                return []

            candidate = response.candidates[0]

            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason.name == 'SAFETY':
                    logger.warning("Gemini blocked response for safety")
                    return []

            # Extract content
            content = candidate.content.parts[0].text if candidate.content.parts else ""

            # Extract grounding metadata if available
            grounding_metadata = getattr(candidate, 'grounding_metadata', None)

            return self._parse_response(content, grounding_metadata)

        except Exception as e:
            if "RATE_LIMIT" in str(e).upper():
                raise RateLimitError("Gemini")
            elif "API_KEY" in str(e).upper() or "PERMISSION" in str(e).upper():
                raise APIKeyError("Gemini", str(e))
            else:
                raise ExternalSearchError(f"Gemini error: {e}")

    def _parse_response(
        self,
        content: str,
        grounding_metadata: Optional[Any]
    ) -> List[ExternalSearchResult]:
        """Parse Gemini response into structured results."""
        results = []

        # If we have grounding metadata, extract structured citations
        if grounding_metadata:
            sources = getattr(grounding_metadata, 'search_entry_point', None)
            web_sources = getattr(grounding_metadata, 'web_search_queries', [])

            # Try to extract grounding chunks
            grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', [])

            for i, chunk in enumerate(grounding_chunks[:10]):
                url = getattr(chunk, 'web', {}).get('uri', '')
                title = getattr(chunk, 'web', {}).get('title', '')

                result = ExternalSearchResult(
                    content=content,
                    source_url=url,
                    source_title=title or urlparse(url).netloc,
                    relevance_score=1.0 - (i * 0.05),
                    provider="gemini",
                    retrieved_at=datetime.utcnow(),
                    snippet=self._extract_grounding_snippet(content, i),
                    metadata={
                        "grounded": True,
                        "chunk_index": i
                    }
                )
                results.append(result)

        # If no grounding metadata, parse content for references
        if not results:
            results = self._parse_content_references(content)

        # Always include the main content as a result if nothing else
        if not results and content:
            result = ExternalSearchResult(
                content=content,
                source_url="",
                source_title="Gemini Deep Research",
                relevance_score=0.9,
                provider="gemini",
                retrieved_at=datetime.utcnow(),
                snippet=content[:300] + "..." if len(content) > 300 else content,
                metadata={
                    "deep_research": True,
                    "grounded": False
                }
            )
            results.append(result)

        logger.info(f"Gemini returned {len(results)} results")
        return results

    def _extract_grounding_snippet(self, content: str, index: int) -> str:
        """Extract a relevant snippet for a grounding chunk."""
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)

        # Return a relevant sentence based on index
        if index < len(sentences):
            sentence = sentences[index].strip()
            return sentence[:200] + "..." if len(sentence) > 200 else sentence

        return content[:200] + "..."

    def _parse_content_references(self, content: str) -> List[ExternalSearchResult]:
        """Parse references from content without grounding metadata."""
        results = []

        # Look for URLs in the content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)

        for i, url in enumerate(urls[:5]):
            domain = urlparse(url).netloc

            result = ExternalSearchResult(
                content=content,
                source_url=url,
                source_title=domain,
                relevance_score=0.8 - (i * 0.05),
                provider="gemini",
                retrieved_at=datetime.utcnow(),
                snippet=f"Reference from {domain}",
                metadata={"parsed_from_content": True}
            )
            results.append(result)

        return results


# =============================================================================
# Factory Functions
# =============================================================================

def create_perplexity_client(
    api_key: Optional[str] = None,
    config: Optional[ExternalSearchConfig] = None
) -> Optional[PerplexitySearchClient]:
    """
    Create Perplexity client if API key is available.

    Args:
        api_key: Perplexity API key (or read from PERPLEXITY_API_KEY env)
        config: Optional configuration

    Returns:
        PerplexitySearchClient or None if no API key
    """
    import os

    key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not key:
        logger.info("Perplexity API key not configured")
        return None

    return PerplexitySearchClient(key, config)


def create_gemini_client(
    api_key: Optional[str] = None,
    config: Optional[ExternalSearchConfig] = None
) -> Optional[GeminiDeepResearchClient]:
    """
    Create Gemini client if API key is available.

    Args:
        api_key: Google API key (or read from GEMINI_API_KEY/GOOGLE_API_KEY env)
        config: Optional configuration

    Returns:
        GeminiDeepResearchClient or None if no API key
    """
    import os

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.info("Gemini API key not configured")
        return None

    return GeminiDeepResearchClient(key, config)
