"""
NeuroSynth Unified - RAG Engine
================================

Retrieval-Augmented Generation engine with Claude integration.

Features:
- Context-aware question answering
- Medical domain prompts
- Citation tracking
- Streaming responses
- Multi-turn conversations

Usage:
    from src.rag.engine import RAGEngine
    
    engine = RAGEngine(
        search_service=search_service,
        api_key="..."
    )
    
    response = await engine.ask(
        question="What is the retrosigmoid approach?",
        include_citations=True
    )
    
    print(response.answer)
    for citation in response.citations:
        print(f"[{citation.index}] {citation.snippet}")
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json

# Graph-RAG integration
try:
    from src.retrieval.graph_rag import GraphRAGContext
    from src.core.relation_extractor import NeuroRelationExtractor
    HAS_GRAPH_RAG = True
except ImportError:
    HAS_GRAPH_RAG = False

from src.rag.context import (
    ContextAssembler,
    AssembledContext,
    Citation,
    ContextImage,
    CitationExtractor,
    ContextFormat
)
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    retry_with_backoff
)

logger = logging.getLogger(__name__)

# Global circuit breaker for Claude API
claude_breaker = CircuitBreaker(
    name="claude",
    failure_threshold=3,
    success_threshold=2,
    reset_timeout=60.0,
    timeout=30.0
)


# =============================================================================
# Models
# =============================================================================

@dataclass
class RAGResponse:
    """Response from RAG engine."""
    answer: str
    citations: List[Citation]
    used_citations: List[Citation]  # Only citations referenced in answer
    images: List[ContextImage]
    
    # Metadata
    question: str
    context_chunks_used: int
    total_tokens_used: int
    generation_time_ms: int
    model: str
    
    # Debug info
    search_time_ms: int = 0
    context_time_ms: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'citations': [c.to_dict() for c in self.citations],
            'used_citations': [c.to_dict() for c in self.used_citations],
            'images': [i.to_dict() for i in self.images],
            'question': self.question,
            'context_chunks_used': self.context_chunks_used,
            'total_tokens_used': self.total_tokens_used,
            'generation_time_ms': self.generation_time_ms,
            'model': self.model
        }


@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.3
    
    # Context settings
    max_context_tokens: int = 8000
    max_context_chunks: int = 10
    max_images: int = 5
    context_format: ContextFormat = ContextFormat.NUMBERED
    
    # Search settings
    search_top_k: int = 20
    search_mode: str = "hybrid"
    enable_rerank: bool = True
    
    # Citation settings
    include_citations: bool = True
    citation_style: str = "inline"  # inline, footnote
    
    # Behavior
    stream: bool = False
    include_sources: bool = True

    # Graph-RAG settings
    use_graph_rag: bool = True
    graph_hop_limit: int = 2
    graph_max_edges: int = 10
    graph_use_mmr: bool = True


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT_MEDICAL = """You are a neurosurgical knowledge assistant with expertise in neuroanatomy, surgical procedures, and clinical decision-making. Your role is to provide accurate, evidence-based answers using the provided context.

Guidelines:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so clearly.
2. Use inline citations [1], [2], etc. to reference your sources.
3. Be precise with medical terminology.
4. For surgical procedures, emphasize key steps and safety considerations.
5. If asked about multiple topics, organize your response clearly.
6. When uncertain, express appropriate epistemic humility.

Citation format: Use [N] immediately after the information from source N.
Example: The retrosigmoid approach provides excellent exposure [1] and allows for facial nerve preservation [2]."""

SYSTEM_PROMPT_GENERAL = """You are a knowledgeable assistant. Answer questions based on the provided context.

Guidelines:
1. Use only the information from the provided context.
2. Cite your sources using [1], [2], etc.
3. If the context doesn't contain relevant information, say so.
4. Be clear and organized in your responses."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above, using citations [1], [2], etc. to reference your sources."""

USER_PROMPT_WITH_IMAGES = """Context:
{context}

Related Images:
{images}

Question: {question}

Please provide a comprehensive answer based on the context above, using citations [1], [2], etc. to reference your sources. You may reference the images if relevant."""


# =============================================================================
# RAG Engine
# =============================================================================

class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    
    Combines:
    - Search service for context retrieval
    - Context assembler for prompt building
    - Claude API for generation
    - Citation tracking and extraction
    """
    
    def __init__(
        self,
        search_service,
        api_key: str = None,
        config: RAGConfig = None,
        system_prompt: str = None,
        db_pool=None,
        embed_fn: Callable[[str], Awaitable[list[float]]] = None,
    ):
        """
        Initialize RAG engine.

        Args:
            search_service: SearchService instance
            api_key: Anthropic API key
            config: RAG configuration
            system_prompt: Custom system prompt (or use medical default)
            db_pool: Database connection pool for Graph-RAG
            embed_fn: Async embedding function for Graph-RAG
        """
        self.search = search_service
        self.api_key = api_key
        self.config = config or RAGConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT_MEDICAL
        self.db_pool = db_pool
        self.embed_fn = embed_fn

        # Initialize context assembler
        self.assembler = ContextAssembler(
            max_context_tokens=self.config.max_context_tokens,
            max_chunks=self.config.max_context_chunks,
            max_images=self.config.max_images,
            format=self.config.context_format
        )

        # Claude client (lazy initialized)
        self._client = None
        self._async_client = None

        # Graph-RAG components (lazy initialized)
        self._graph_ctx: Optional["GraphRAGContext"] = None
        self._entity_extractor: Optional["NeuroRelationExtractor"] = None
        self._graph_initialized = False

        if self.config.use_graph_rag and HAS_GRAPH_RAG and db_pool and embed_fn:
            try:
                self._entity_extractor = NeuroRelationExtractor()
                self._graph_ctx = GraphRAGContext(
                    db_pool=db_pool,
                    embed_fn=embed_fn,
                )
                logger.info("Graph-RAG context initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Graph-RAG: {e}")

    async def initialize(self):
        """Initialize async components (Graph-RAG)."""
        if self._graph_ctx and not self._graph_initialized:
            try:
                await self._graph_ctx.initialize()
                self._graph_initialized = True
                logger.info("Graph-RAG async components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Graph-RAG async: {e}")
    
    def _get_client(self):
        """Get synchronous Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    def _get_async_client(self):
        """Get async Anthropic client."""
        if self._async_client is None:
            import anthropic
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client
    
    # =========================================================================
    # Main Methods
    # =========================================================================
    
    async def ask(
        self,
        question: str,
        filters: Any = None,
        include_citations: bool = None,
        include_images: bool = True,
        stream: bool = None,
        conversation_history: List[Dict] = None,
        use_graph: bool = None,
    ) -> RAGResponse:
        """
        Ask a question and get an answer with citations.

        Args:
            question: User question
            filters: SearchFilters for context retrieval
            include_citations: Include citations in response
            include_images: Include linked images
            stream: Stream response (returns AsyncGenerator if True)
            conversation_history: Previous messages for multi-turn
            use_graph: Use Graph-RAG for context expansion (default: config.use_graph_rag)

        Returns:
            RAGResponse with answer, citations, and metadata
        """
        import time
        start_time = time.time()

        include_citations = include_citations if include_citations is not None else self.config.include_citations
        stream = stream if stream is not None else self.config.stream
        use_graph = use_graph if use_graph is not None else self.config.use_graph_rag

        # Step 0: Extract entities and get graph context (if Graph-RAG enabled)
        graph_context = None
        entities = []
        if use_graph and self._graph_ctx and self._entity_extractor:
            try:
                entities = self._extract_entities_from_question(question)
                if entities:
                    graph_context = await self._graph_ctx.get_context(
                        question=question,
                        entities=entities,
                        hop_limit=self.config.graph_hop_limit,
                        use_mmr=self.config.graph_use_mmr,
                    )
                    logger.debug(f"Graph-RAG: {len(entities)} entities, {len(graph_context.get('edges', []))} edges")
            except Exception as e:
                logger.warning(f"Graph-RAG context failed: {e}")

        # Step 1: Search for relevant context
        search_start = time.time()
        search_response = await self.search.search(
            query=question,
            mode=self.config.search_mode,
            top_k=self.config.search_top_k,
            filters=filters,
            include_images=include_images,
            rerank=self.config.enable_rerank
        )
        search_time = int((time.time() - search_start) * 1000)

        # Step 2: Assemble context
        context_start = time.time()
        context = self.assembler.assemble(
            search_results=search_response.results,
            query=question
        )
        context_time = int((time.time() - context_start) * 1000)

        # Step 3: Build prompt (with graph context if available)
        prompt = self._build_prompt(question, context, include_images, graph_context)
        
        # Step 4: Generate answer
        if stream:
            return self._stream_response(
                question=question,
                prompt=prompt,
                context=context,
                search_time=search_time,
                context_time=context_time,
                conversation_history=conversation_history
            )
        
        gen_start = time.time()
        answer = await self._generate(prompt, conversation_history)
        gen_time = int((time.time() - gen_start) * 1000)
        
        # Step 5: Extract used citations
        used_citations = CitationExtractor.get_used_citations(
            answer, context.citations
        ) if include_citations else []
        
        total_time = int((time.time() - start_time) * 1000)
        
        return RAGResponse(
            answer=answer,
            citations=context.citations,
            used_citations=used_citations,
            images=context.images,
            question=question,
            context_chunks_used=context.chunks_used,
            total_tokens_used=context.total_tokens,
            generation_time_ms=gen_time,
            model=self.config.model,
            search_time_ms=search_time,
            context_time_ms=context_time
        )
    
    async def ask_with_context(
        self,
        question: str,
        context_text: str,
        citations: List[Citation] = None
    ) -> RAGResponse:
        """
        Ask a question with pre-assembled context.
        
        Useful when context is already prepared or comes from
        a different source.
        """
        import time
        start_time = time.time()
        
        prompt = USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question
        )
        
        answer = await self._generate(prompt)
        gen_time = int((time.time() - start_time) * 1000)
        
        used_citations = []
        if citations:
            used_citations = CitationExtractor.get_used_citations(answer, citations)
        
        return RAGResponse(
            answer=answer,
            citations=citations or [],
            used_citations=used_citations,
            images=[],
            question=question,
            context_chunks_used=len(citations) if citations else 0,
            total_tokens_used=0,
            generation_time_ms=gen_time,
            model=self.config.model
        )
    
    # =========================================================================
    # Entity Extraction (for Graph-RAG)
    # =========================================================================

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extract medical entities from question for Graph-RAG.

        Uses spaCy noun chunks + abbreviation expansion.
        """
        if not self._entity_extractor:
            return []

        try:
            doc = self._entity_extractor.nlp(question)
            entities = []

            # Extract noun chunks
            for chunk in doc.noun_chunks:
                normalized = self._entity_extractor.normalize_entity(chunk.text)
                if len(normalized) > 2:  # Skip tiny fragments
                    entities.append(normalized)

            # Also check for known medical abbreviations
            known_abbrevs = ["MCA", "ACA", "PCA", "GBM", "SAH", "ICH", "AVM", "DBS", "EVD"]
            for token in doc:
                if token.text.upper() in known_abbrevs:
                    normalized = self._entity_extractor.normalize_entity(token.text)
                    entities.append(normalized)

            return list(set(entities))
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_prompt(
        self,
        question: str,
        context: AssembledContext,
        include_images: bool = True,
        graph_context: Optional[Dict] = None,
    ) -> str:
        """Build user prompt with context and optional graph relationships."""
        # Build graph context section
        graph_section = ""
        if graph_context and graph_context.get("prompt_context"):
            graph_section = f"\n\n{graph_context['prompt_context']}"

        if include_images and context.images:
            # Format images
            image_lines = []
            for i, img in enumerate(context.images, 1):
                caption = img.caption or "No caption"
                image_lines.append(f"Image {i}: {caption}")

            images_text = "\n".join(image_lines)

            return USER_PROMPT_WITH_IMAGES.format(
                context=context.text + graph_section,
                images=images_text,
                question=question
            )
        else:
            return USER_PROMPT_TEMPLATE.format(
                context=context.text + graph_section,
                question=question
            )
    
    # =========================================================================
    # Generation
    # =========================================================================
    
    async def _generate(
        self,
        prompt: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate response using Claude with circuit breaker protection."""
        client = self._get_async_client()

        # Build messages
        messages = []

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": prompt})

        try:
            async with claude_breaker:
                response = await client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.system_prompt,
                    messages=messages
                )
                return response.content[0].text

        except CircuitOpenError as e:
            logger.warning(f"Claude circuit open: {e}")
            return self._fallback_response(prompt)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _fallback_response(self, prompt: str) -> str:
        """Generate fallback response when Claude is unavailable."""
        return (
            "I'm currently unable to generate a response. "
            "The AI service is temporarily unavailable. "
            "Please try again in a few moments.\n\n"
            "In the meantime, you can review the context documents directly."
        )
    
    async def _stream_response(
        self,
        question: str,
        prompt: str,
        context: AssembledContext,
        search_time: int,
        context_time: int,
        conversation_history: List[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens with circuit breaker protection."""
        client = self._get_async_client()

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        full_response = ""

        try:
            async with claude_breaker:
                async with client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.system_prompt,
                    messages=messages
                ) as stream:
                    async for text in stream.text_stream:
                        full_response += text
                        yield text

        except asyncio.CancelledError:
            logger.info("Streaming cancelled by client disconnect")
            raise  # Re-raise to stop execution properly
        except CircuitOpenError as e:
            logger.warning(f"Claude circuit open during streaming: {e}")
            yield self._fallback_response(prompt)
            full_response = self._fallback_response(prompt)
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield f"Error: {str(e)}"
            return

        # After streaming, yield final metadata as JSON
        used_citations = CitationExtractor.get_used_citations(
            full_response, context.citations
        )

        metadata = {
            "type": "metadata",
            "citations": [c.to_dict() for c in context.citations],
            "used_citations": [c.to_dict() for c in used_citations],
            "context_chunks_used": context.chunks_used,
            "search_time_ms": search_time,
            "context_time_ms": context_time
        }

        yield f"\n\n<!-- RAG_METADATA: {json.dumps(metadata)} -->"
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def summarize_document(
        self,
        document_id: str,
        max_chunks: int = 20
    ) -> RAGResponse:
        """Summarize a document by ID."""
        from src.retrieval.search_service import SearchFilters
        
        question = "Please provide a comprehensive summary of this document, including main topics, key procedures, and important findings."
        
        filters = SearchFilters(document_ids=[document_id])
        
        # Override config for summary
        original_top_k = self.config.search_top_k
        self.config.search_top_k = max_chunks
        
        try:
            response = await self.ask(
                question=question,
                filters=filters,
                include_citations=True
            )
            return response
        finally:
            self.config.search_top_k = original_top_k
    
    async def explain_procedure(
        self,
        procedure_name: str
    ) -> RAGResponse:
        """Get detailed explanation of a surgical procedure."""
        from src.retrieval.search_service import SearchFilters
        
        question = f"""Explain the {procedure_name} procedure in detail, including:
1. Indications and patient selection
2. Key anatomical considerations
3. Step-by-step surgical technique
4. Potential complications and how to avoid them
5. Expected outcomes"""
        
        filters = SearchFilters(chunk_types=["PROCEDURE", "ANATOMY"])
        
        return await self.ask(
            question=question,
            filters=filters,
            include_citations=True,
            include_images=True
        )
    
    async def compare_approaches(
        self,
        approach1: str,
        approach2: str
    ) -> RAGResponse:
        """Compare two surgical approaches."""
        question = f"""Compare the {approach1} and {approach2} approaches:
1. Indications for each approach
2. Anatomical exposure provided
3. Advantages and disadvantages
4. Complication profiles
5. When to choose one over the other"""
        
        return await self.ask(
            question=question,
            include_citations=True
        )


# =============================================================================
# Conversation Manager
# =============================================================================

class RAGConversation:
    """
    Manages multi-turn RAG conversations.
    
    Maintains conversation history and context across turns.
    """
    
    def __init__(
        self,
        engine: RAGEngine,
        max_history: int = 10
    ):
        self.engine = engine
        self.max_history = max_history
        self.history: List[Dict] = []
        self.all_citations: List[Citation] = []
    
    async def ask(
        self,
        question: str,
        **kwargs
    ) -> RAGResponse:
        """Ask a question in the conversation context."""
        # Build conversation history for Claude
        messages = []
        for turn in self.history[-self.max_history:]:
            messages.append({"role": "user", "content": turn['question']})
            messages.append({"role": "assistant", "content": turn['answer']})
        
        # Get response
        response = await self.engine.ask(
            question=question,
            conversation_history=messages if messages else None,
            **kwargs
        )
        
        # Store in history
        self.history.append({
            'question': question,
            'answer': response.answer,
            'citations': response.citations
        })
        
        # Track all citations
        self.all_citations.extend(response.citations)
        
        return response
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.all_citations = []
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations from the conversation."""
        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for c in self.all_citations:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique.append(c)
        return unique


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("RAG Engine - requires search service and API key")
    print()
    print("Usage:")
    print("  engine = RAGEngine(search_service, api_key='...')")
    print("  response = await engine.ask('What is the retrosigmoid approach?')")
    print("  print(response.answer)")
