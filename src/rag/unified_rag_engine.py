"""
NeuroSynth Unified - Unified RAG Engine
========================================

Unified RAG engine supporting multiple search modes:
- STANDARD: Internal database only (existing behavior)
- HYBRID: Internal + Perplexity web search
- DEEP_RESEARCH: Internal + Gemini reasoning
- EXTERNAL: Web search only

This engine wraps the existing RAGEngine and adds:
- Mode selection
- Research enrichment integration
- Dual-source citation tracking
- Streaming with source attribution

Architecture:
    UnifiedRAGEngine
    ├── RAGEngine (internal)
    ├── ResearchEnricher (external orchestration)
    └── Citation Merger (unified output)

Usage:
    from src.rag import UnifiedRAGEngine, RAGEngine
    from src.research import ResearchEnricher, SearchMode

    unified = UnifiedRAGEngine(
        rag_engine=rag_engine,
        enricher=enricher
    )

    # Ask with mode selection
    response = await unified.ask(
        question="Latest DBS infection protocols",
        mode=SearchMode.HYBRID
    )

    # Access dual citations
    print(response.internal_citations)  # [1], [2]...
    print(response.external_citations)  # [W1], [W2]...
    print(response.gap_report)          # Conflicts and gaps
"""

import asyncio
import logging
import os
import time
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Tuple

import anthropic

from src.research.models import (
    SearchMode,
    ExternalCitation,
    GapReport,
    EnrichedContext,
    get_available_modes,
    ModeInfo,
)
from src.research.research_enricher import ResearchEnricher
from src.rag.context import Citation

logger = logging.getLogger(__name__)


# =============================================================================
# Query Complexity Analysis (merged from V3)
# =============================================================================

class QueryComplexity(Enum):
    """Complexity classification for queries."""
    SIMPLE = "simple"               # Single fact lookup
    MODERATE = "moderate"           # Multi-fact synthesis
    COMPLEX = "complex"             # Cross-document analysis
    RESEARCH = "research"           # Chapter-level synthesis


# Complexity indicators
COMPLEXITY_KEYWORDS = {
    QueryComplexity.SIMPLE: [
        "what is", "define", "dose", "name", "list", "which",
        "how many", "when was", "where is", "who",
    ],
    QueryComplexity.MODERATE: [
        "explain", "describe", "how does", "why does",
        "steps", "technique", "approach", "procedure",
    ],
    QueryComplexity.COMPLEX: [
        "compare", "contrast", "difference between", "versus",
        "advantages", "disadvantages", "pros and cons",
        "across", "between these", "all the",
    ],
    QueryComplexity.RESEARCH: [
        "write a chapter", "comprehensive review", "summarize all",
        "compare outcomes across", "longitudinal", "evolution of",
        "systematic", "meta-analysis", "compare these papers",
    ],
}

# Temporal indicators for external search
TEMPORAL_KEYWORDS = [
    "latest", "recent", "2024", "2025", "2026", "new", "current",
    "updated", "guidelines", "trial", "now", "today",
]


@dataclass
class QueryAnalysis:
    """Result of query analysis/triage."""
    original_query: str
    complexity: QueryComplexity
    recommended_mode: SearchMode

    # Analysis details
    entity_count: int = 0
    requires_temporal: bool = False
    requires_comparison: bool = False
    document_scope: Optional[List[str]] = None  # Specific docs to search

    # Decomposed sub-queries (for complex queries)
    sub_queries: List[str] = field(default_factory=list)

    # Confidence in routing decision
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "recommended_mode": self.recommended_mode.value,
            "entity_count": self.entity_count,
            "requires_temporal": self.requires_temporal,
            "requires_comparison": self.requires_comparison,
            "sub_queries": self.sub_queries,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class QueryRouter:
    """
    Intelligent query router that analyzes queries and suggests optimal mode.

    The router performs:
    1. Keyword-based complexity detection
    2. Entity extraction for scope analysis
    3. Temporal requirement detection
    4. Query decomposition for complex queries
    """

    def analyze(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        force_mode: Optional[SearchMode] = None,
    ) -> QueryAnalysis:
        """
        Analyze query and determine optimal processing mode.

        Args:
            query: User's question
            document_ids: Specific documents to search (scope limiter)
            force_mode: Override automatic routing

        Returns:
            QueryAnalysis with routing decision
        """
        query_lower = query.lower()

        # Step 1: Detect complexity from keywords
        complexity = self._detect_complexity(query_lower)

        # Step 2: Check temporal requirements
        requires_temporal = any(kw in query_lower for kw in TEMPORAL_KEYWORDS)

        # Step 3: Check comparison requirements
        requires_comparison = any(
            kw in query_lower
            for kw in COMPLEXITY_KEYWORDS[QueryComplexity.COMPLEX]
        )

        # Step 4: Estimate entity count (rough heuristic)
        entity_count = self._estimate_entity_count(query)

        # Step 5: Determine mode
        if force_mode and force_mode != SearchMode.AUTO:
            mode = force_mode
            reasoning = f"Mode forced to {mode.value}"
        else:
            mode, reasoning = self._determine_mode(
                complexity=complexity,
                requires_temporal=requires_temporal,
                requires_comparison=requires_comparison,
                entity_count=entity_count,
                document_count=len(document_ids) if document_ids else 0,
            )

        # Step 6: Decompose if complex
        sub_queries = []
        if complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH):
            sub_queries = self._decompose_query(query)

        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            recommended_mode=mode,
            entity_count=entity_count,
            requires_temporal=requires_temporal,
            requires_comparison=requires_comparison,
            document_scope=document_ids,
            sub_queries=sub_queries,
            confidence=0.8,  # Heuristic confidence
            reasoning=reasoning,
        )

    def _detect_complexity(self, query_lower: str) -> QueryComplexity:
        """Detect query complexity from keywords."""
        # Check from most complex to simplest
        for complexity in [
            QueryComplexity.RESEARCH,
            QueryComplexity.COMPLEX,
            QueryComplexity.MODERATE,
            QueryComplexity.SIMPLE,
        ]:
            if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS[complexity]):
                return complexity

        # Default based on length
        word_count = len(query_lower.split())
        if word_count > 30:
            return QueryComplexity.COMPLEX
        elif word_count > 15:
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE

    def _estimate_entity_count(self, query: str) -> int:
        """Estimate number of entities in query."""
        # Simple heuristic: count capitalized words and medical terms
        words = query.split()
        capitalized = sum(1 for w in words if len(w) > 2 and w[0].isupper())

        # Medical term indicators
        medical_suffixes = ['oma', 'itis', 'ectomy', 'otomy', 'plasty', 'graphy']
        medical_terms = sum(
            1 for w in words
            if any(w.lower().endswith(s) for s in medical_suffixes)
        )

        return capitalized + medical_terms

    def _determine_mode(
        self,
        complexity: QueryComplexity,
        requires_temporal: bool,
        requires_comparison: bool,
        entity_count: int,
        document_count: int,
    ) -> Tuple[SearchMode, str]:
        """Determine optimal mode based on analysis."""

        # Research-level complexity -> Deep Research
        if complexity == QueryComplexity.RESEARCH:
            return SearchMode.DEEP_RESEARCH, "Research-level query requires full-text analysis"

        # Temporal requirements -> External or Hybrid
        if requires_temporal:
            if complexity == QueryComplexity.SIMPLE:
                return SearchMode.EXTERNAL_ONLY, "Temporal query with simple complexity"
            return SearchMode.HYBRID, "Temporal query requires external + internal"

        # Multi-document comparison -> Deep Research
        if requires_comparison and document_count >= 2:
            return SearchMode.DEEP_RESEARCH, f"Comparing {document_count} documents"

        # Complex with many entities -> Deep Research
        if complexity == QueryComplexity.COMPLEX and entity_count >= 3:
            return SearchMode.DEEP_RESEARCH, f"Complex query with {entity_count} entities"

        # Default to standard for most queries
        return SearchMode.STANDARD, "Standard chunk-based retrieval sufficient"

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        sub_queries = []

        # Handle "compare X and Y" patterns
        compare_match = re.search(
            r'compare\s+(.+?)\s+(?:and|vs|versus|with)\s+(.+?)(?:\s+for|\s+in|\.|$)',
            query,
            re.IGNORECASE
        )
        if compare_match:
            topic_a, topic_b = compare_match.groups()
            sub_queries.append(f"What are the key features of {topic_a}?")
            sub_queries.append(f"What are the key features of {topic_b}?")
            sub_queries.append(f"What are the differences between {topic_a} and {topic_b}?")

        # Handle "across these papers" patterns
        if "across" in query.lower() or "papers" in query.lower():
            sub_queries.append("What is the main finding of each paper?")
            sub_queries.append("What methodology was used in each study?")
            sub_queries.append("What are the key differences in outcomes?")

        return sub_queries


# =============================================================================
# Response Models
# =============================================================================

@dataclass
class UnifiedRAGResponse:
    """
    Response from unified RAG engine with dual-source citations.

    Extends standard RAGResponse with:
    - Separate internal vs external citations
    - Gap report with conflicts/agreements
    - Mode attribution
    - Source composition metrics
    """
    answer: str
    question: str

    # Dual citations
    internal_citations: List[Citation]
    external_citations: List[ExternalCitation]

    # Combined for backwards compatibility
    used_citations: List[Citation]

    # Gap analysis
    gap_report: Optional[GapReport]

    # Mode information
    mode_used: SearchMode
    internal_ratio: float  # 0.0-1.0, % from internal

    # Images (from internal only)
    images: List[Any] = field(default_factory=list)

    # Timing
    search_time_ms: int = 0
    context_time_ms: int = 0
    generation_time_ms: int = 0
    total_time_ms: int = 0

    # Model info
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            'answer': self.answer,
            'question': self.question,
            'internal_citations': [c.to_dict() for c in self.internal_citations],
            'external_citations': [c.to_dict() for c in self.external_citations],
            'used_citations': [c.to_dict() for c in self.used_citations],
            'gap_report': self.gap_report.to_dict() if self.gap_report else None,
            'mode_used': self.mode_used.value,
            'internal_ratio': self.internal_ratio,
            'images': [img.to_dict() if hasattr(img, 'to_dict') else img for img in self.images],
            'search_time_ms': self.search_time_ms,
            'context_time_ms': self.context_time_ms,
            'generation_time_ms': self.generation_time_ms,
            'total_time_ms': self.total_time_ms,
            'model': self.model,
        }


@dataclass
class UnifiedRAGConfig:
    """Configuration for unified RAG engine."""

    # Claude settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.3

    # Context settings
    max_total_tokens: int = 8000

    # Behavior
    default_mode: SearchMode = SearchMode.HYBRID
    fallback_to_standard: bool = True  # Fallback on external failure
    include_images: bool = True

    # Citation format
    internal_citation_prefix: str = ""      # [1], [2]...
    external_citation_prefix: str = "W"     # [W1], [W2]...


# =============================================================================
# System Prompts
# =============================================================================

UNIFIED_SYSTEM_PROMPT = """You are an expert neurosurgical clinical decision support assistant.

You have access to TWO knowledge sources:
1. 📚 USER'S DATABASE (Ground Truth): Their curated textbooks, notes, institutional protocols
2. 🌐 CURRENT RESEARCH (World Truth): Live web search with latest guidelines and publications

## CRITICAL: Protocol Alert Requirement

If the GAP ANALYSIS in the context shows CONFLICTS between these sources, you MUST:

1. **START your response with a Protocol Alert block** (do not skip this):

   ⚠️ PROTOCOL ALERT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Your database indicates [X], however current guidelines recommend [Y].
   Recommended action: [specific guidance]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. **Clearly distinguish sources in your synthesis**:
   - "According to your notes [1]..."
   - "However, the 2024 guidelines [W1] recommend..."
   - "This represents a change from your documented protocol..."

3. **Do NOT hide or minimize conflicts** - they are patient safety relevant

## Citation Format
- Internal (Your Database): [1], [2], [3]...
- External (Current Research): [W1], [W2], [W3]...

## Response Priority
- For ANATOMY/TECHNIQUE questions: Your curated database is authoritative
- For PROTOCOLS/DOSAGES/GUIDELINES: Weight recent external sources appropriately
- For CONFLICTS: Present both perspectives clearly, recommend verification

## Clinical Communication Style
- Be direct and specific (dosages, timings, technique details)
- Avoid hedging when sources are clear
- Flag uncertainty explicitly when present
- End with actionable guidance when appropriate"""

STANDARD_SYSTEM_PROMPT = """You are an expert neurosurgical assistant with access to a curated database of authoritative textbooks and clinical notes.

Your task is to provide comprehensive, accurate answers based on the provided context.

## Citation Format
Use citations [1], [2], [3]... to reference your sources.

## Response Guidelines
- Be precise with anatomical and procedural details
- Provide specific dosages, timings, and technique descriptions when available
- If the context doesn't contain sufficient information, acknowledge the limitation
- Structure complex answers with clear organization

## Clinical Communication Style
- Direct and specific (avoid unnecessary hedging)
- Include relevant technical details
- Flag any uncertainty explicitly
- End with actionable guidance when appropriate"""


# =============================================================================
# Unified RAG Engine
# =============================================================================

class UnifiedRAGEngine:
    """
    Unified RAG engine supporting multiple search modes.

    This engine wraps the existing RAGEngine and adds external search
    capabilities via the ResearchEnricher. It maintains full backward
    compatibility - STANDARD mode delegates entirely to the existing engine.

    Attributes:
        internal_engine: Existing RAGEngine for internal-only search
        enricher: ResearchEnricher for external search integration
        config: UnifiedRAGConfig
    """

    def __init__(
        self,
        rag_engine,  # RAGEngine
        enricher: Optional[ResearchEnricher] = None,
        config: Optional[UnifiedRAGConfig] = None,
        anthropic_client = None,  # For unified generation
    ):
        """
        Initialize unified RAG engine.

        Args:
            rag_engine: Existing RAGEngine instance
            enricher: Optional ResearchEnricher for external search
            config: Configuration options
            anthropic_client: Claude client for unified generation
        """
        self.internal_engine = rag_engine
        self.enricher = enricher
        self.config = config or UnifiedRAGConfig()

        # Initialize query router for complexity analysis
        self._router = QueryRouter()

        # Use provided client or try to get from internal engine or create new
        self._anthropic = anthropic_client
        if not self._anthropic and rag_engine and hasattr(rag_engine, '_async_client'):
            self._anthropic = rag_engine._async_client
        if not self._anthropic:
            # Create Anthropic client if API key is available
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._anthropic = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("Created new AsyncAnthropic client for UnifiedRAGEngine")

        logger.info(
            f"UnifiedRAGEngine initialized: "
            f"internal={rag_engine is not None}, "
            f"enricher={enricher is not None}"
        )

    @property
    def has_external(self) -> bool:
        """Check if external search is available."""
        return self.enricher is not None and self.enricher.has_external

    def get_available_modes(self) -> List[ModeInfo]:
        """Get list of available search modes with descriptions."""
        has_perplexity = self.enricher.has_perplexity if self.enricher else False
        has_gemini = self.enricher.has_gemini if self.enricher else False
        return get_available_modes(has_perplexity, has_gemini)

    async def ask(
        self,
        question: str,
        mode: Optional[SearchMode] = None,
        filters = None,
        include_citations: bool = True,
        include_images: bool = True,
        stream: bool = False,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Union[UnifiedRAGResponse, AsyncGenerator]:
        """
        Ask a question with mode selection.

        Args:
            question: User's question
            mode: Search mode (defaults to config.default_mode)
            filters: Optional SearchFilters for internal search
            include_citations: Include citation tracking
            include_images: Include related images
            stream: Return async generator for streaming
            conversation_history: Previous conversation turns

        Returns:
            UnifiedRAGResponse or AsyncGenerator if streaming
        """
        start_time = time.time()
        mode = mode or self.config.default_mode

        logger.info(f"Unified RAG ask: mode={mode.value}, question={question[:50]}...")

        # STANDARD mode: Delegate entirely to existing engine
        if mode == SearchMode.STANDARD:
            return await self._standard_ask(
                question, filters, include_citations, include_images,
                stream, conversation_history, start_time
            )

        # Other modes require enricher
        if not self.enricher:
            logger.warning("No enricher available, falling back to STANDARD")
            return await self._standard_ask(
                question, filters, include_citations, include_images,
                stream, conversation_history, start_time
            )

        # Execute enriched search
        try:
            enriched = await self.enricher.enrich(
                query=question,
                mode=mode,
                filters=filters,
                include_gap_analysis=True
            )

            # Generate unified response
            if stream:
                return self._stream_unified_response(
                    question, enriched, conversation_history, start_time
                )

            return await self._generate_unified_response(
                question, enriched, conversation_history, start_time
            )

        except Exception as e:
            logger.error(f"Enriched search failed: {e}")
            if self.config.fallback_to_standard:
                logger.info("Falling back to STANDARD mode")
                return await self._standard_ask(
                    question, filters, include_citations, include_images,
                    stream, conversation_history, start_time
                )
            raise

    async def _standard_ask(
        self,
        question: str,
        filters,
        include_citations: bool,
        include_images: bool,
        stream: bool,
        conversation_history: Optional[List[Dict]],
        start_time: float
    ) -> UnifiedRAGResponse:
        """Delegate to standard RAGEngine."""
        # Call existing engine
        response = await self.internal_engine.ask(
            question=question,
            filters=filters,
            include_citations=include_citations,
            include_images=include_images,
            stream=stream,
            conversation_history=conversation_history
        )

        # Handle streaming
        if stream:
            return response  # Return generator as-is

        # Wrap in UnifiedRAGResponse
        total_time = int((time.time() - start_time) * 1000)

        return UnifiedRAGResponse(
            answer=response.answer,
            question=question,
            internal_citations=response.citations,
            external_citations=[],
            used_citations=response.used_citations,
            gap_report=None,
            mode_used=SearchMode.STANDARD,
            internal_ratio=1.0,
            images=response.images,
            search_time_ms=response.search_time_ms,
            context_time_ms=response.context_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=total_time,
            model=response.model
        )

    async def _generate_unified_response(
        self,
        question: str,
        enriched: EnrichedContext,
        conversation_history: Optional[List[Dict]],
        start_time: float
    ) -> UnifiedRAGResponse:
        """Generate response from enriched context."""
        gen_start = time.time()

        # Build messages
        messages = []

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add enriched prompt as user message
        messages.append({
            "role": "user",
            "content": enriched.synthesis_prompt
        })

        # Select system prompt based on mode
        system_prompt = (
            UNIFIED_SYSTEM_PROMPT
            if enriched.mode_used != SearchMode.STANDARD
            else STANDARD_SYSTEM_PROMPT
        )

        # Generate with Claude
        try:
            if self._anthropic:
                response = await self._anthropic.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=messages
                )
                answer = response.content[0].text
            else:
                # Fallback to internal engine's generate method
                answer = await self.internal_engine._generate(
                    enriched.synthesis_prompt,
                    conversation_history
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"I apologize, but I encountered an error generating a response: {e}"

        gen_time = int((time.time() - gen_start) * 1000)
        total_time = int((time.time() - start_time) * 1000)

        # Extract citations
        internal_citations, external_citations = self._extract_dual_citations(
            answer, enriched
        )

        # Get images from internal chunks
        images = self._extract_images(enriched.internal_chunks)

        return UnifiedRAGResponse(
            answer=answer,
            question=question,
            internal_citations=internal_citations,
            external_citations=external_citations,
            used_citations=internal_citations,  # For backwards compatibility
            gap_report=enriched.gap_report,
            mode_used=enriched.mode_used,
            internal_ratio=enriched.internal_ratio,
            images=images,
            search_time_ms=enriched.search_time_ms,
            context_time_ms=0,
            generation_time_ms=gen_time,
            total_time_ms=total_time,
            model=self.config.model
        )

    async def _stream_unified_response(
        self,
        question: str,
        enriched: EnrichedContext,
        conversation_history: Optional[List[Dict]],
        start_time: float
    ) -> AsyncGenerator:
        """Stream unified response with SSE."""
        # Build messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": enriched.synthesis_prompt})

        system_prompt = (
            UNIFIED_SYSTEM_PROMPT
            if enriched.mode_used != SearchMode.STANDARD
            else STANDARD_SYSTEM_PROMPT
        )

        accumulated_answer = ""

        # Check if Anthropic client is available
        if not self._anthropic:
            # Fallback to non-streaming response
            try:
                answer = await self.internal_engine._generate(
                    enriched.synthesis_prompt,
                    conversation_history
                )
                yield {"type": "token", "content": answer}

                internal_citations, external_citations = self._extract_dual_citations(
                    answer, enriched
                )
                total_time = int((time.time() - start_time) * 1000)
                yield {
                    "type": "done",
                    "answer": answer,
                    "internal_citations": [c.to_dict() for c in internal_citations],
                    "external_citations": [c.to_dict() for c in external_citations],
                    "gap_report": enriched.gap_report.to_dict() if enriched.gap_report else None,
                    "mode_used": enriched.mode_used.value,
                    "internal_ratio": enriched.internal_ratio,
                    "total_time_ms": total_time
                }
                return
            except Exception as e:
                logger.error(f"Fallback generation error: {e}")
                yield {"type": "error", "message": str(e)}
                return

        # Guard check for Anthropic client
        if not self._anthropic:
            logger.error("Anthropic client not available for streaming")
            yield {"type": "error", "message": "AI service unavailable. Please check ANTHROPIC_API_KEY configuration."}
            return

        try:
            # Stream from Claude
            async with self._anthropic.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    accumulated_answer += text
                    yield {
                        "type": "token",
                        "content": text
                    }

            # Extract citations from final answer
            internal_citations, external_citations = self._extract_dual_citations(
                accumulated_answer, enriched
            )

            # Yield final metadata
            total_time = int((time.time() - start_time) * 1000)

            yield {
                "type": "done",
                "answer": accumulated_answer,
                "internal_citations": [c.to_dict() for c in internal_citations],
                "external_citations": [c.to_dict() for c in external_citations],
                "gap_report": enriched.gap_report.to_dict() if enriched.gap_report else None,
                "mode_used": enriched.mode_used.value,
                "internal_ratio": enriched.internal_ratio,
                "total_time_ms": total_time
            }

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }

    def _extract_dual_citations(
        self,
        answer: str,
        enriched: EnrichedContext
    ) -> tuple:
        """Extract both internal and external citations from answer."""
        internal_citations = []
        external_citations = []

        # Find internal citations [1], [2], etc.
        internal_pattern = r'\[(\d+)\]'
        internal_refs = set(re.findall(internal_pattern, answer))

        for ref in internal_refs:
            idx = int(ref)
            if idx <= len(enriched.internal_chunks):
                chunk = enriched.internal_chunks[idx - 1]
                chunk_type_val = getattr(chunk, 'chunk_type', None)
                if chunk_type_val and hasattr(chunk_type_val, 'value'):
                    chunk_type_val = chunk_type_val.value
                citation = Citation(
                    index=idx,
                    chunk_id=getattr(chunk, 'id', str(idx)),
                    content=getattr(chunk, 'content', ''),
                    snippet=getattr(chunk, 'content', '')[:100],
                    document_id=getattr(chunk, 'document_id', None),
                    page_number=getattr(chunk, 'page_number', None),
                    chunk_type=chunk_type_val
                )
                internal_citations.append(citation)

        # Find external citations [W1], [W2], etc.
        external_pattern = r'\[W(\d+)\]'
        external_refs = set(re.findall(external_pattern, answer))

        for ref in external_refs:
            idx = int(ref)
            if idx <= len(enriched.external_results):
                result = enriched.external_results[idx - 1]
                citation = ExternalCitation(
                    index=f"W{idx}",
                    source_url=result.source_url,
                    source_title=result.source_title,
                    snippet=result.snippet,
                    provider=result.provider
                )
                external_citations.append(citation)

        return internal_citations, external_citations

    def _extract_images(self, internal_chunks: List[Any]) -> List[Any]:
        """Extract linked images from internal chunks."""
        images = []
        seen_ids = set()

        for chunk in internal_chunks:
            chunk_images = getattr(chunk, 'images', [])
            for img in chunk_images:
                img_id = getattr(img, 'image_id', id(img))
                if img_id not in seen_ids:
                    seen_ids.add(img_id)
                    images.append(img)

        return images[:8]  # Limit to 8 images

    def analyze_complexity(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
    ) -> QueryAnalysis:
        """
        Analyze query complexity and get recommended mode.

        This is useful for:
        - Auto-selecting the best mode
        - Showing users why a mode was selected
        - Query decomposition for complex questions

        Args:
            query: User's question
            document_ids: Specific documents to search

        Returns:
            QueryAnalysis with complexity, recommended mode, and reasoning
        """
        return self._router.analyze(query, document_ids)


# =============================================================================
# Conversation Manager
# =============================================================================

class UnifiedRAGConversation:
    """
    Manages multi-turn conversations with unified RAG.

    Extends RAGConversation to support mode persistence and
    dual-source citation tracking across turns.
    """

    def __init__(
        self,
        engine: UnifiedRAGEngine,
        mode: SearchMode = SearchMode.HYBRID,
        max_history: int = 10
    ):
        self.engine = engine
        self.mode = mode
        self.max_history = max_history
        self.history: List[Dict] = []
        self.all_internal_citations: List[Citation] = []
        self.all_external_citations: List[ExternalCitation] = []

    async def ask(
        self,
        question: str,
        **kwargs
    ) -> UnifiedRAGResponse:
        """Ask a question in conversation context."""
        # Build conversation history for Claude
        messages = []
        for turn in self.history[-self.max_history:]:
            messages.append({"role": "user", "content": turn['question']})
            messages.append({"role": "assistant", "content": turn['answer']})

        # Allow mode override
        mode = kwargs.pop('mode', self.mode)

        # Get response
        response = await self.engine.ask(
            question=question,
            mode=mode,
            conversation_history=messages if messages else None,
            **kwargs
        )

        # Store in history
        self.history.append({
            'question': question,
            'answer': response.answer,
            'mode': response.mode_used.value
        })

        # Track citations
        self.all_internal_citations.extend(response.internal_citations)
        self.all_external_citations.extend(response.external_citations)

        return response

    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.all_internal_citations = []
        self.all_external_citations = []

    def set_mode(self, mode: SearchMode):
        """Update default mode for future questions."""
        self.mode = mode

    def get_all_citations(self) -> tuple:
        """Get all citations from conversation."""
        # Deduplicate
        seen_internal = set()
        seen_external = set()

        unique_internal = []
        unique_external = []

        for c in self.all_internal_citations:
            if c.chunk_id not in seen_internal:
                seen_internal.add(c.chunk_id)
                unique_internal.append(c)

        for c in self.all_external_citations:
            if c.source_url not in seen_external:
                seen_external.add(c.source_url)
                unique_external.append(c)

        return unique_internal, unique_external
