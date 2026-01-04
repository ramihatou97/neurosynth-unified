"""
NeuroSynth V3 - Unified RAG QA Engine
======================================

The "Hybrid Brain" - Three-Lobe Architecture for Medical QA

This module implements the optimal RAG system that intelligently routes queries
to the appropriate processing mode based on complexity analysis:

┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUERY ROUTER (Triage)                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Analyzes: complexity, scope, temporal requirements, entity count            │
│  Decides:  STANDARD | DEEP_RESEARCH | EXTERNAL | HYBRID                     │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────────────────┐
        │           │                       │
        ▼           ▼                       ▼
┌─────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  STANDARD   │ │  DEEP RESEARCH  │ │  EXTERNAL ENRICHER  │
│  (80% Q's)  │ │  (Complex)      │ │  (Gap Filling)      │
│             │ │                 │ │                     │
│  Vector DB  │ │  Full-Text      │ │  Perplexity/Gemini  │
│  Chunks     │ │  Pages Table    │ │  Web Search         │
│  Claude     │ │  Gemini 1.5 Pro │ │  Academic APIs      │
│             │ │  2M Context     │ │                     │
│  Fast/Cheap │ │  Deep/Accurate  │ │  Fresh/Broad        │
└─────────────┘ └─────────────────┘ └─────────────────────┘
        │           │                       │
        └───────────┼───────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED RESPONSE ASSEMBLY                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Dual Citations: [Source N] (Internal) + [Web: X] (External)                │
│  Conflict Detection: Flags disagreements between sources                     │
│  Provenance Tracking: Full chain of custody for every claim                 │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    from src.rag.unified_engine import UnifiedRAGEngine, QueryMode

    engine = UnifiedRAGEngine(
        search_service=search,
        anthropic_client=client,
        gemini_client=gemini,
        perplexity_api_key="pplx-xxx",
    )

    # Automatic mode selection
    response = await engine.ask("What is the dose of dexamethasone?")

    # Force specific mode
    response = await engine.ask(
        "Compare outcomes across these 5 papers on MCA aneurysms",
        mode=QueryMode.DEEP_RESEARCH,
        document_ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )

Author: NeuroSynth Team
Version: 3.0.0
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class QueryMode(Enum):
    """Processing mode for queries."""
    STANDARD = "standard"           # Chunk-based RAG (fast, cheap)
    DEEP_RESEARCH = "deep_research" # Full-text analysis (thorough, expensive)
    EXTERNAL = "external"           # Web search only
    HYBRID = "hybrid"               # Internal + External combined
    AUTO = "auto"                   # Let router decide


class QueryComplexity(Enum):
    """Complexity classification for queries."""
    SIMPLE = "simple"               # Single fact lookup
    MODERATE = "moderate"           # Multi-fact synthesis
    COMPLEX = "complex"             # Cross-document analysis
    RESEARCH = "research"           # Chapter-level synthesis


class SourceType(Enum):
    """Source classification for citations."""
    INTERNAL_CHUNK = "internal_chunk"
    INTERNAL_PAGE = "internal_page"
    EXTERNAL_WEB = "external_web"
    EXTERNAL_ACADEMIC = "external_academic"


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
    "latest", "recent", "2024", "2025", "new", "current",
    "updated", "guidelines", "trial", "now", "today",
]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Citation:
    """Unified citation model."""
    index: int
    source_type: SourceType
    content_snippet: str

    # Internal source fields
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None

    # External source fields
    url: Optional[str] = None
    publication_date: Optional[str] = None
    authors: Optional[List[str]] = None

    def format_citation(self) -> str:
        """Format for display in answer."""
        if self.source_type in (SourceType.INTERNAL_CHUNK, SourceType.INTERNAL_PAGE):
            return f"[Source {self.index}]"
        else:
            title = self.document_title or self.url or "Web"
            return f"[Web: {title}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "source_type": self.source_type.value,
            "content_snippet": self.content_snippet[:200],
            "document_title": self.document_title,
            "page_number": self.page_number,
            "url": self.url,
        }


@dataclass
class QueryAnalysis:
    """Result of query analysis/triage."""
    original_query: str
    complexity: QueryComplexity
    recommended_mode: QueryMode

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


@dataclass
class UnifiedResponse:
    """Unified response from any processing mode."""
    answer: str
    citations: List[Citation]
    used_citations: List[Citation]  # Only those referenced in answer

    # Query metadata
    query: str
    mode_used: QueryMode
    query_analysis: QueryAnalysis

    # Performance metrics
    total_time_ms: int
    search_time_ms: int = 0
    generation_time_ms: int = 0

    # Source breakdown
    internal_sources: int = 0
    external_sources: int = 0
    pages_analyzed: int = 0  # For deep research

    # Confidence and quality
    confidence_score: float = 0.0
    has_conflicts: bool = False
    conflict_details: Optional[str] = None

    # Images (if requested)
    images: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "used_citations": [c.to_dict() for c in self.used_citations],
            "query": self.query,
            "mode_used": self.mode_used.value,
            "query_analysis": self.query_analysis.to_dict(),
            "total_time_ms": self.total_time_ms,
            "internal_sources": self.internal_sources,
            "external_sources": self.external_sources,
            "pages_analyzed": self.pages_analyzed,
            "has_conflicts": self.has_conflicts,
        }


@dataclass
class UnifiedRAGConfig:
    """Configuration for the unified RAG engine."""

    # Model selection
    standard_model: str = "claude-sonnet-4-20250514"
    deep_research_model: str = "gemini-2.5-pro"  # Large context window

    # Standard mode settings
    standard_top_k: int = 10
    standard_max_tokens: int = 4000

    # Deep research settings
    deep_max_pages: int = 100  # Max pages to load for deep research
    deep_max_tokens: int = 8000

    # External enrichment settings
    enable_external: bool = True
    max_external_queries: int = 3
    external_cache_ttl: int = 3600  # 1 hour

    # Routing thresholds
    complexity_threshold_for_deep: int = 3  # Entities or comparison scope
    auto_deep_for_multi_doc: bool = True

    # Quality settings
    enable_conflict_detection: bool = True
    min_confidence_threshold: float = 0.3

    # Cost control
    max_cost_per_query: float = 0.50  # USD
    prefer_cheaper_mode: bool = True


# =============================================================================
# QUERY ROUTER
# =============================================================================

class QueryRouter:
    """
    Intelligent query router that analyzes queries and selects optimal mode.

    The router performs:
    1. Keyword-based complexity detection
    2. Entity extraction for scope analysis
    3. Temporal requirement detection
    4. Query decomposition for complex queries
    """

    def __init__(
        self,
        anthropic_client=None,
        use_llm_routing: bool = False,
    ):
        self.client = anthropic_client
        self.use_llm_routing = use_llm_routing and anthropic_client is not None

    async def analyze(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        force_mode: Optional[QueryMode] = None,
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
        if force_mode and force_mode != QueryMode.AUTO:
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
        capitalized = sum(1 for w in words if w[0].isupper() and len(w) > 2)

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
    ) -> Tuple[QueryMode, str]:
        """Determine optimal mode based on analysis."""

        # Research-level complexity -> Deep Research
        if complexity == QueryComplexity.RESEARCH:
            return QueryMode.DEEP_RESEARCH, "Research-level query requires full-text analysis"

        # Temporal requirements -> External or Hybrid
        if requires_temporal:
            if complexity == QueryComplexity.SIMPLE:
                return QueryMode.EXTERNAL, "Temporal query with simple complexity"
            return QueryMode.HYBRID, "Temporal query requires external + internal"

        # Multi-document comparison -> Deep Research
        if requires_comparison and document_count >= 2:
            return QueryMode.DEEP_RESEARCH, f"Comparing {document_count} documents"

        # Complex with many entities -> Deep Research
        if complexity == QueryComplexity.COMPLEX and entity_count >= 3:
            return QueryMode.DEEP_RESEARCH, f"Complex query with {entity_count} entities"

        # Default to standard for most queries
        return QueryMode.STANDARD, "Standard chunk-based retrieval sufficient"

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        # Simple rule-based decomposition
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
# PROCESSING ENGINES
# =============================================================================

class ProcessingEngine(ABC):
    """Abstract base class for processing engines."""

    @abstractmethod
    async def process(
        self,
        query: str,
        analysis: QueryAnalysis,
        **kwargs
    ) -> UnifiedResponse:
        """Process query and return response."""
        pass


class StandardEngine(ProcessingEngine):
    """
    Standard chunk-based RAG engine.

    Uses existing SearchService + Claude for fast, efficient Q&A.
    Best for: Simple facts, specific procedures, single-document queries.
    """

    def __init__(
        self,
        search_service,
        anthropic_client,
        config: UnifiedRAGConfig,
    ):
        self.search = search_service
        self.client = anthropic_client
        self.config = config

        # Import existing RAG components
        from src.rag.context import ContextAssembler, CitationExtractor
        self.assembler = ContextAssembler()
        self.citation_extractor = CitationExtractor

    async def process(
        self,
        query: str,
        analysis: QueryAnalysis,
        filters=None,
        include_images: bool = False,
        **kwargs
    ) -> UnifiedResponse:
        """Process with standard chunk retrieval."""
        start_time = time.time()

        # Step 1: Search
        search_start = time.time()
        search_response = await self.search.search(
            query=query,
            mode="hybrid",
            top_k=self.config.standard_top_k,
            filters=filters,
            include_images=include_images,
            rerank=True,
        )
        search_time = int((time.time() - search_start) * 1000)

        # Step 2: Assemble context
        context = self.assembler.assemble(
            search_results=search_response.results,
            query=query
        )

        # Step 3: Build prompt
        prompt = self._build_prompt(query, context)

        # Step 4: Generate
        gen_start = time.time()
        answer = await self._generate(prompt)
        gen_time = int((time.time() - gen_start) * 1000)

        # Step 5: Extract used citations
        citations = self._convert_citations(context.citations)
        used_citations = self._extract_used_citations(answer, citations)

        total_time = int((time.time() - start_time) * 1000)

        return UnifiedResponse(
            answer=answer,
            citations=citations,
            used_citations=used_citations,
            query=query,
            mode_used=QueryMode.STANDARD,
            query_analysis=analysis,
            total_time_ms=total_time,
            search_time_ms=search_time,
            generation_time_ms=gen_time,
            internal_sources=len(citations),
            external_sources=0,
            confidence_score=0.8,
        )

    def _build_prompt(self, query: str, context) -> str:
        """Build prompt for Claude."""
        # AssembledContext uses 'text' attribute for the formatted context string
        context_text = getattr(context, 'text', str(context))
        return f"""You are a neurosurgical expert assistant. Answer the following question using ONLY the provided context.

CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer accurately based on the context
2. Cite sources using [N] format where N is the source number
3. If the context doesn't contain enough information, say so
4. Be concise but complete

ANSWER:"""

    async def _generate(self, prompt: str) -> str:
        """Generate response with Claude."""
        response = await self.client.messages.create(
            model=self.config.standard_model,
            max_tokens=self.config.standard_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _convert_citations(self, context_citations) -> List[Citation]:
        """Convert context citations to unified format."""
        return [
            Citation(
                index=c.index,
                source_type=SourceType.INTERNAL_CHUNK,
                content_snippet=c.snippet,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=getattr(c, 'document_title', None),
                page_number=c.page_number,
            )
            for c in context_citations
        ]

    def _extract_used_citations(
        self,
        answer: str,
        citations: List[Citation]
    ) -> List[Citation]:
        """Extract citations actually used in answer."""
        used = []
        for citation in citations:
            if f"[{citation.index}]" in answer:
                used.append(citation)
        return used


class DeepResearchEngine(ProcessingEngine):
    """
    Deep research engine using full-text analysis.

    Uses pages table + Gemini 1.5 Pro's 2M context window.
    Best for: Cross-document analysis, chapter writing, longitudinal studies.
    """

    def __init__(
        self,
        database,
        gemini_client,
        config: UnifiedRAGConfig,
    ):
        self.database = database
        self.gemini = gemini_client
        self.config = config

    async def process(
        self,
        query: str,
        analysis: QueryAnalysis,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> UnifiedResponse:
        """Process with full-text deep analysis."""
        start_time = time.time()

        # Step 1: Load full text from pages table
        search_start = time.time()
        full_texts = await self._load_documents(document_ids)
        search_time = int((time.time() - search_start) * 1000)

        if not full_texts:
            return self._empty_response(query, analysis, "No documents found for deep research")

        # Step 2: Build comprehensive prompt
        prompt = self._build_deep_prompt(query, full_texts, analysis.sub_queries)

        # Step 3: Generate with Gemini's large context
        gen_start = time.time()
        answer, citations = await self._generate_deep(prompt, full_texts)
        gen_time = int((time.time() - gen_start) * 1000)

        total_time = int((time.time() - start_time) * 1000)

        return UnifiedResponse(
            answer=answer,
            citations=citations,
            used_citations=citations,  # All used in deep research
            query=query,
            mode_used=QueryMode.DEEP_RESEARCH,
            query_analysis=analysis,
            total_time_ms=total_time,
            search_time_ms=search_time,
            generation_time_ms=gen_time,
            internal_sources=len(full_texts),
            external_sources=0,
            pages_analyzed=sum(len(t["pages"]) for t in full_texts),
            confidence_score=0.9,  # Higher confidence for full-text
        )

    async def _load_documents(
        self,
        document_ids: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Load full text from pages table."""
        if not document_ids:
            # Get recent/relevant documents
            document_ids = await self._get_relevant_documents()

        documents = []
        for doc_id in document_ids[:5]:  # Limit to 5 documents
            try:
                # Query pages table for full text
                pages = await self.database.fetch("""
                    SELECT page_number, content
                    FROM pages
                    WHERE document_id = $1
                    ORDER BY page_number
                    LIMIT $2
                """, UUID(doc_id), self.config.deep_max_pages)

                # Get document metadata
                doc = await self.database.fetchrow("""
                    SELECT id, title, file_path, authority_score
                    FROM documents WHERE id = $1
                """, UUID(doc_id))

                if pages and doc:
                    documents.append({
                        "id": str(doc["id"]),
                        "title": doc["title"] or "Unknown",
                        "authority_score": doc["authority_score"] or 1.0,
                        "pages": [
                            {"number": p["page_number"], "content": p["content"]}
                            for p in pages
                        ],
                        "full_text": "\n\n".join(p["content"] for p in pages),
                    })
            except Exception as e:
                logger.warning(f"Failed to load document {doc_id}: {e}")

        return documents

    async def _get_relevant_documents(self) -> List[str]:
        """Get relevant document IDs when none specified."""
        rows = await self.database.fetch("""
            SELECT id FROM documents
            WHERE deleted_at IS NULL
            ORDER BY authority_score DESC, created_at DESC
            LIMIT 5
        """)
        return [str(row["id"]) for row in rows]

    def _build_deep_prompt(
        self,
        query: str,
        documents: List[Dict],
        sub_queries: List[str],
    ) -> str:
        """Build prompt for deep research."""
        # Build document sections
        doc_sections = []
        for i, doc in enumerate(documents, 1):
            doc_sections.append(f"""
═══════════════════════════════════════════════════════════════════════════════
DOCUMENT {i}: {doc['title']}
Authority Score: {doc['authority_score']}
Pages: {len(doc['pages'])}
═══════════════════════════════════════════════════════════════════════════════

{doc['full_text'][:50000]}  # Limit per doc for safety
""")

        # Build sub-queries section
        sub_query_section = ""
        if sub_queries:
            sub_query_section = f"""
ANALYSIS SUB-QUESTIONS:
{chr(10).join(f'- {sq}' for sq in sub_queries)}
"""

        return f"""You are conducting deep research on neurosurgical literature.

TASK: {query}
{sub_query_section}

AVAILABLE DOCUMENTS:
{chr(10).join(doc_sections)}

INSTRUCTIONS:
1. Analyze the COMPLETE text of all documents
2. Identify connections between documents that chunking would miss
3. Synthesize findings across all sources
4. Use [Doc N, p.X] citations (N=document number, X=page number)
5. Note any contradictions or evolution of understanding
6. Provide comprehensive, textbook-quality response

RESPONSE:"""

    async def _generate_deep(
        self,
        prompt: str,
        documents: List[Dict],
    ) -> Tuple[str, List[Citation]]:
        """Generate with Gemini's large context window."""
        try:
            response = await asyncio.to_thread(
                self.gemini.generate_content,
                prompt,
            )
            answer = response.text

            # Extract citations from [Doc N, p.X] format
            citations = self._extract_deep_citations(answer, documents)

            return answer, citations

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Deep research failed: {str(e)}", []

    def _extract_deep_citations(
        self,
        answer: str,
        documents: List[Dict],
    ) -> List[Citation]:
        """Extract citations from deep research answer."""
        citations = []
        pattern = r'\[Doc\s*(\d+),?\s*p\.?\s*(\d+)\]'

        for i, match in enumerate(re.finditer(pattern, answer), 1):
            doc_num = int(match.group(1))
            page_num = int(match.group(2))

            if 0 < doc_num <= len(documents):
                doc = documents[doc_num - 1]
                citations.append(Citation(
                    index=i,
                    source_type=SourceType.INTERNAL_PAGE,
                    content_snippet=f"Document: {doc['title']}, Page {page_num}",
                    document_id=doc["id"],
                    document_title=doc["title"],
                    page_number=page_num,
                ))

        return citations

    def _empty_response(
        self,
        query: str,
        analysis: QueryAnalysis,
        message: str,
    ) -> UnifiedResponse:
        """Return empty response when no documents available."""
        return UnifiedResponse(
            answer=message,
            citations=[],
            used_citations=[],
            query=query,
            mode_used=QueryMode.DEEP_RESEARCH,
            query_analysis=analysis,
            total_time_ms=0,
            confidence_score=0.0,
        )


class ExternalEngine(ProcessingEngine):
    """
    External enrichment engine using Perplexity/Gemini.

    Fetches information from the live web.
    Best for: Recent news, updated guidelines, clinical trials.
    """

    def __init__(
        self,
        perplexity_api_key: Optional[str],
        google_api_key: Optional[str],
        config: UnifiedRAGConfig,
    ):
        self.config = config

        # Initialize providers (reuse from research_enricher)
        self.perplexity = None
        self.gemini = None

        if perplexity_api_key:
            try:
                from openai import AsyncOpenAI
                self.perplexity = AsyncOpenAI(
                    api_key=perplexity_api_key,
                    base_url="https://api.perplexity.ai"
                )
            except ImportError:
                logger.warning("OpenAI package not installed for Perplexity")

        if google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                self.gemini = genai.GenerativeModel("gemini-2.5-pro")
            except ImportError:
                logger.warning("Google AI package not installed")

    async def process(
        self,
        query: str,
        analysis: QueryAnalysis,
        **kwargs
    ) -> UnifiedResponse:
        """Process with external web search."""
        start_time = time.time()

        # Prefer Perplexity for academic queries
        if self.perplexity:
            answer, citations = await self._search_perplexity(query)
        elif self.gemini:
            answer, citations = await self._search_gemini(query)
        else:
            return self._unavailable_response(query, analysis)

        total_time = int((time.time() - start_time) * 1000)

        return UnifiedResponse(
            answer=answer,
            citations=citations,
            used_citations=citations,
            query=query,
            mode_used=QueryMode.EXTERNAL,
            query_analysis=analysis,
            total_time_ms=total_time,
            generation_time_ms=total_time,
            internal_sources=0,
            external_sources=len(citations),
            confidence_score=0.7,  # Lower confidence for external
        )

    async def _search_perplexity(
        self,
        query: str,
    ) -> Tuple[str, List[Citation]]:
        """Search with Perplexity."""
        system = """You are a medical research assistant specializing in neurosurgery.
Focus on recent peer-reviewed publications, clinical trials, and guidelines.
Always cite your sources."""

        try:
            response = await self.perplexity.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                max_tokens=2000,
            )

            answer = response.choices[0].message.content
            citations = self._extract_web_citations(answer)

            return answer, citations

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            return f"External search failed: {str(e)}", []

    async def _search_gemini(
        self,
        query: str,
    ) -> Tuple[str, List[Citation]]:
        """Search with Gemini grounding."""
        prompt = f"""Search for recent medical information about: {query}

Focus on:
- Peer-reviewed sources
- Recent guidelines
- Clinical trial results

Provide cited response."""

        try:
            response = await asyncio.to_thread(
                self.gemini.generate_content,
                prompt,
            )

            answer = response.text
            citations = self._extract_web_citations(answer)

            return answer, citations

        except Exception as e:
            logger.error(f"Gemini search failed: {e}")
            return f"External search failed: {str(e)}", []

    def _extract_web_citations(self, answer: str) -> List[Citation]:
        """Extract citations from web search answer."""
        citations = []

        # Extract URLs
        url_pattern = r'https?://[^\s\)\]<>]+'
        urls = re.findall(url_pattern, answer)

        for i, url in enumerate(urls[:10], 1):
            citations.append(Citation(
                index=i,
                source_type=SourceType.EXTERNAL_WEB,
                content_snippet="",
                url=url,
                document_title=url.split('/')[2],  # Domain as title
            ))

        return citations

    def _unavailable_response(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> UnifiedResponse:
        """Return response when external search unavailable."""
        return UnifiedResponse(
            answer="External search is not configured. Please set PERPLEXITY_API_KEY or GOOGLE_API_KEY.",
            citations=[],
            used_citations=[],
            query=query,
            mode_used=QueryMode.EXTERNAL,
            query_analysis=analysis,
            total_time_ms=0,
            confidence_score=0.0,
        )


# =============================================================================
# UNIFIED RAG ENGINE - THE ORCHESTRATOR
# =============================================================================

class UnifiedRAGEngine:
    """
    The Unified RAG Engine - orchestrates all processing modes.

    This is the main entry point that:
    1. Analyzes queries with the Router
    2. Dispatches to appropriate Engine
    3. Optionally combines results (Hybrid mode)
    4. Handles conflict detection
    5. Returns unified response
    """

    def __init__(
        self,
        search_service,
        anthropic_client,
        database=None,
        gemini_client=None,
        perplexity_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[UnifiedRAGConfig] = None,
    ):
        self.config = config or UnifiedRAGConfig()

        # Initialize router
        self.router = QueryRouter(
            anthropic_client=anthropic_client,
            use_llm_routing=False,  # Heuristic routing is faster
        )

        # Initialize engines
        self.standard_engine = StandardEngine(
            search_service=search_service,
            anthropic_client=anthropic_client,
            config=self.config,
        )

        self.deep_engine = None
        if database and gemini_client:
            self.deep_engine = DeepResearchEngine(
                database=database,
                gemini_client=gemini_client,
                config=self.config,
            )

        self.external_engine = None
        if perplexity_api_key or google_api_key:
            self.external_engine = ExternalEngine(
                perplexity_api_key=perplexity_api_key,
                google_api_key=google_api_key,
                config=self.config,
            )

        logger.info(
            f"UnifiedRAGEngine initialized: "
            f"standard=True, deep={self.deep_engine is not None}, "
            f"external={self.external_engine is not None}"
        )

    async def ask(
        self,
        query: str,
        mode: QueryMode = QueryMode.AUTO,
        document_ids: Optional[List[str]] = None,
        filters=None,
        include_images: bool = False,
        include_external: bool = True,
    ) -> UnifiedResponse:
        """
        Ask a question with automatic mode selection.

        Args:
            query: User's question
            mode: Force specific mode or AUTO for intelligent routing
            document_ids: Specific documents to search
            filters: Search filters for standard mode
            include_images: Include related images
            include_external: Allow external enrichment

        Returns:
            UnifiedResponse with answer and metadata
        """
        start_time = time.time()

        # Step 1: Analyze query
        analysis = await self.router.analyze(
            query=query,
            document_ids=document_ids,
            force_mode=mode if mode != QueryMode.AUTO else None,
        )

        logger.info(
            f"Query routed: complexity={analysis.complexity.value}, "
            f"mode={analysis.recommended_mode.value}, "
            f"reason='{analysis.reasoning}'"
        )

        # Step 2: Dispatch to appropriate engine
        mode_to_use = analysis.recommended_mode

        if mode_to_use == QueryMode.STANDARD:
            response = await self.standard_engine.process(
                query=query,
                analysis=analysis,
                filters=filters,
                include_images=include_images,
            )

        elif mode_to_use == QueryMode.DEEP_RESEARCH:
            if self.deep_engine:
                response = await self.deep_engine.process(
                    query=query,
                    analysis=analysis,
                    document_ids=document_ids,
                )
            else:
                # Fallback to standard if deep not available
                logger.warning("Deep research requested but not configured, falling back to standard")
                response = await self.standard_engine.process(
                    query=query,
                    analysis=analysis,
                    filters=filters,
                )

        elif mode_to_use == QueryMode.EXTERNAL:
            # External mode always uses external engine if available
            # (include_external flag only affects hybrid mode)
            if self.external_engine:
                response = await self.external_engine.process(
                    query=query,
                    analysis=analysis,
                )
            else:
                # Fallback to standard if external not configured
                logger.warning("External mode requested but not configured, falling back to standard")
                response = await self.standard_engine.process(
                    query=query,
                    analysis=analysis,
                    filters=filters,
                )

        elif mode_to_use == QueryMode.HYBRID:
            response = await self._process_hybrid(
                query=query,
                analysis=analysis,
                document_ids=document_ids,
                filters=filters,
                include_images=include_images,
            )

        else:
            response = await self.standard_engine.process(
                query=query,
                analysis=analysis,
                filters=filters,
            )

        # Step 3: Conflict detection (if enabled)
        if self.config.enable_conflict_detection and response.internal_sources > 0:
            response = await self._detect_conflicts(response)

        # Update total time
        response.total_time_ms = int((time.time() - start_time) * 1000)

        return response

    async def _process_hybrid(
        self,
        query: str,
        analysis: QueryAnalysis,
        document_ids: Optional[List[str]],
        filters,
        include_images: bool,
    ) -> UnifiedResponse:
        """Process with both internal and external sources."""

        # Run internal and external in parallel
        internal_task = self.standard_engine.process(
            query=query,
            analysis=analysis,
            filters=filters,
            include_images=include_images,
        )

        external_task = None
        if self.external_engine:
            external_task = self.external_engine.process(
                query=query,
                analysis=analysis,
            )

        # Await results
        internal_response = await internal_task
        external_response = None
        if external_task:
            external_response = await external_task

        # Merge responses
        return self._merge_responses(
            query=query,
            analysis=analysis,
            internal=internal_response,
            external=external_response,
        )

    def _merge_responses(
        self,
        query: str,
        analysis: QueryAnalysis,
        internal: UnifiedResponse,
        external: Optional[UnifiedResponse],
    ) -> UnifiedResponse:
        """Merge internal and external responses."""
        if not external:
            return internal

        # Combine citations
        all_citations = internal.citations.copy()

        # Renumber external citations
        offset = len(all_citations)
        for citation in external.citations:
            citation.index += offset
            all_citations.append(citation)

        # Combine answers
        combined_answer = f"""**Based on Internal Library:**
{internal.answer}

**Based on Recent External Sources:**
{external.answer}"""

        return UnifiedResponse(
            answer=combined_answer,
            citations=all_citations,
            used_citations=internal.used_citations + external.used_citations,
            query=query,
            mode_used=QueryMode.HYBRID,
            query_analysis=analysis,
            total_time_ms=max(internal.total_time_ms, external.total_time_ms),
            search_time_ms=internal.search_time_ms,
            generation_time_ms=internal.generation_time_ms + external.generation_time_ms,
            internal_sources=internal.internal_sources,
            external_sources=external.external_sources,
            confidence_score=(internal.confidence_score + external.confidence_score) / 2,
        )

    async def _detect_conflicts(
        self,
        response: UnifiedResponse,
    ) -> UnifiedResponse:
        """Detect conflicts between sources."""
        # Simple heuristic: check for contradictory keywords
        contradictory_pairs = [
            ("recommended", "contraindicated"),
            ("safe", "dangerous"),
            ("effective", "ineffective"),
            ("increase", "decrease"),
        ]

        answer_lower = response.answer.lower()

        for word1, word2 in contradictory_pairs:
            if word1 in answer_lower and word2 in answer_lower:
                response.has_conflicts = True
                response.conflict_details = f"Potential conflict detected: '{word1}' vs '{word2}'"
                break

        return response

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def quick_answer(self, query: str) -> str:
        """Get just the answer text."""
        response = await self.ask(query, mode=QueryMode.STANDARD)
        return response.answer

    async def research(
        self,
        topic: str,
        document_ids: List[str],
    ) -> UnifiedResponse:
        """Force deep research mode."""
        return await self.ask(
            query=f"Provide comprehensive analysis of: {topic}",
            mode=QueryMode.DEEP_RESEARCH,
            document_ids=document_ids,
        )

    async def compare(
        self,
        item_a: str,
        item_b: str,
        aspects: Optional[List[str]] = None,
    ) -> UnifiedResponse:
        """Compare two items."""
        aspects_str = ", ".join(aspects) if aspects else "all relevant aspects"
        query = f"Compare {item_a} and {item_b} in terms of {aspects_str}"
        return await self.ask(query, mode=QueryMode.AUTO)

    async def get_latest(self, topic: str) -> UnifiedResponse:
        """Get latest information from external sources."""
        query = f"What are the latest developments in {topic} (2024-2025)?"
        return await self.ask(query, mode=QueryMode.EXTERNAL)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

async def create_unified_engine_from_env():
    """Create UnifiedRAGEngine from environment variables."""
    import os
    from anthropic import AsyncAnthropic

    # Required
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY required")

    anthropic_client = AsyncAnthropic(api_key=anthropic_key)

    # Search service (required)
    # This would come from your dependency injection
    search_service = None  # Placeholder - inject from container

    # Optional: Gemini for deep research
    gemini_client = None
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            gemini_client = genai.GenerativeModel("gemini-2.5-pro")
        except ImportError:
            pass

    # Optional: Database for deep research
    database = None  # Placeholder - inject from container

    return UnifiedRAGEngine(
        search_service=search_service,
        anthropic_client=anthropic_client,
        database=database,
        gemini_client=gemini_client,
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
        google_api_key=google_key,
    )
