"""
NeuroSynth Enhanced RAG Engine
===============================

Hardened RAG implementation with:
1. Synthesis context linking - Chat about generated chapters
2. Enhanced citation formatting - Rich citation objects
3. Improved multi-turn support - Context accumulation
4. Streaming responses - Real-time generation

Drop-in enhancement for src/rag/engine.py
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED DATA MODELS
# =============================================================================

@dataclass
class EnhancedCitation:
    """Rich citation with full context."""
    index: int
    chunk_id: str
    content: str
    snippet: str

    # Source info
    document_id: str
    document_title: str
    page_number: int

    # Type and quality
    chunk_type: str
    authority_score: float

    # Medical concepts
    cuis: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'chunk_id': self.chunk_id,
            'snippet': self.snippet,
            'document_id': self.document_id,
            'document_title': self.document_title,
            'page_number': self.page_number,
            'chunk_type': self.chunk_type,
            'authority_score': self.authority_score,
            'cuis': self.cuis,
            'entity_names': self.entity_names
        }

    def format_reference(self) -> str:
        """Format as readable reference."""
        return f"{self.document_title}, p.{self.page_number}"


@dataclass
class SynthesisContext:
    """Context from a synthesis result for follow-up questions."""
    synthesis_id: str
    topic: str
    template_type: str

    # Extracted from synthesis
    sections: Dict[str, str]  # section_name -> content
    all_cuis: List[str]
    sources: List[Dict] = field(default_factory=list)
    image_catalog: List[Dict] = field(default_factory=list)

    # For efficient retrieval
    chunk_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'synthesis_id': self.synthesis_id,
            'topic': self.topic,
            'template_type': self.template_type,
            'sections': self.sections,
            'all_cuis': self.all_cuis,
            'sources': self.sources,
            'image_catalog': self.image_catalog,
            'chunk_ids': self.chunk_ids,
            'document_ids': self.document_ids
        }

    @classmethod
    def from_synthesis_result(cls, result: Any, synthesis_id: str = None) -> "SynthesisContext":
        """Create context from SynthesisResult."""
        synthesis_id = synthesis_id or str(uuid4())

        sections = {}
        chunk_ids = set()

        for section in getattr(result, 'sections', []):
            sections[section.title] = section.content
            chunk_ids.update(section.sources or [])

        document_ids = list(set(
            ref.get('document_id', '')
            for ref in getattr(result, 'references', [])
            if isinstance(ref, dict)
        ))

        return cls(
            synthesis_id=synthesis_id,
            topic=result.topic,
            template_type=str(result.template_type),
            sections=sections,
            all_cuis=getattr(result, 'all_cuis', []) or [],
            sources=result.references if isinstance(result.references, list) else [],
            image_catalog=getattr(result, 'image_catalog', []) or [],
            chunk_ids=list(chunk_ids),
            document_ids=document_ids
        )


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    turn_id: str
    role: str  # user, assistant
    content: str
    timestamp: float

    # For assistant turns
    citations: List[EnhancedCitation] = field(default_factory=list)
    cuis_mentioned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'turn_id': self.turn_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'citations': [c.to_dict() for c in self.citations]
        }


@dataclass
class EnhancedRAGResponse:
    """Enhanced RAG response with rich metadata."""
    answer: str

    # Citations
    citations: List[EnhancedCitation]
    used_citation_indices: List[int]

    # Images
    images: List[Dict]

    # Metadata
    question: str
    context_chunks_used: int
    total_tokens_used: int

    # Timing
    search_time_ms: int
    context_time_ms: int
    generation_time_ms: int
    total_time_ms: int

    # Model info
    model: str

    # Enhanced fields
    synthesis_context_id: Optional[str] = None
    cuis_in_answer: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    follow_up_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'citations': [c.to_dict() for c in self.citations],
            'used_citation_indices': self.used_citation_indices,
            'images': self.images,
            'question': self.question,
            'context_chunks_used': self.context_chunks_used,
            'total_tokens_used': self.total_tokens_used,
            'search_time_ms': self.search_time_ms,
            'context_time_ms': self.context_time_ms,
            'generation_time_ms': self.generation_time_ms,
            'total_time_ms': self.total_time_ms,
            'model': self.model,
            'synthesis_context_id': self.synthesis_context_id,
            'cuis_in_answer': self.cuis_in_answer,
            'confidence_score': self.confidence_score,
            'follow_up_questions': self.follow_up_questions
        }


# =============================================================================
# CONVERSATION MANAGER (Enhanced)
# =============================================================================

class EnhancedConversationManager:
    """
    Manages multi-turn conversations with context accumulation.

    Features:
    - Synthesis context linking
    - Citation tracking across turns
    - Context window management
    - Follow-up question generation
    """

    def __init__(
        self,
        max_history: int = 10,
        max_context_tokens: int = 4000
    ):
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens

        # Storage
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.synthesis_contexts: Dict[str, SynthesisContext] = {}
        self.all_citations: Dict[str, List[EnhancedCitation]] = {}

    def create_conversation(
        self,
        synthesis_context: Optional[SynthesisContext] = None
    ) -> str:
        """Create new conversation, optionally linked to synthesis."""
        conv_id = str(uuid4())

        self.conversations[conv_id] = []
        self.all_citations[conv_id] = []

        if synthesis_context:
            self.synthesis_contexts[conv_id] = synthesis_context

        return conv_id

    def link_synthesis(
        self,
        conv_id: str,
        synthesis_context: SynthesisContext
    ):
        """Link existing conversation to synthesis context."""
        if conv_id not in self.conversations:
            raise ValueError(f"Conversation {conv_id} not found")
        self.synthesis_contexts[conv_id] = synthesis_context

    def add_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        citations: List[EnhancedCitation] = None
    ):
        """Add turn to conversation."""
        if conv_id not in self.conversations:
            raise ValueError(f"Conversation {conv_id} not found")

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            citations=citations or []
        )

        self.conversations[conv_id].append(turn)

        # Track citations
        if citations:
            self.all_citations[conv_id].extend(citations)

        # Trim history if needed
        if len(self.conversations[conv_id]) > self.max_history * 2:
            self.conversations[conv_id] = self.conversations[conv_id][-self.max_history * 2:]

    def get_history_for_claude(self, conv_id: str) -> List[Dict]:
        """Get conversation history formatted for Claude API."""
        if conv_id not in self.conversations:
            return []

        messages = []
        for turn in self.conversations[conv_id][-self.max_history * 2:]:
            messages.append({
                'role': turn.role,
                'content': turn.content
            })

        return messages

    def get_synthesis_context(self, conv_id: str) -> Optional[SynthesisContext]:
        """Get linked synthesis context."""
        return self.synthesis_contexts.get(conv_id)

    def get_all_citations(self, conv_id: str) -> List[EnhancedCitation]:
        """Get all unique citations from conversation."""
        if conv_id not in self.all_citations:
            return []

        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for c in self.all_citations[conv_id]:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique.append(c)

        return unique

    def clear_conversation(self, conv_id: str):
        """Clear conversation history."""
        if conv_id in self.conversations:
            self.conversations[conv_id] = []
            self.all_citations[conv_id] = []

    def delete_conversation(self, conv_id: str):
        """Delete conversation entirely."""
        self.conversations.pop(conv_id, None)
        self.synthesis_contexts.pop(conv_id, None)
        self.all_citations.pop(conv_id, None)


# =============================================================================
# ENHANCED RAG ENGINE
# =============================================================================

class EnhancedRAGEngine:
    """
    Enhanced RAG engine with synthesis context linking.

    Improvements over base RAGEngine:
    1. Synthesis context linking for follow-up questions
    2. Enhanced citations with medical concepts
    3. Improved conversation management
    4. Confidence scoring
    5. Follow-up question generation
    """

    # Prompts
    SYSTEM_PROMPT = """You are a neurosurgical knowledge assistant with expertise in neuroanatomy, surgical procedures, and clinical decision-making.

Guidelines:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so clearly.
2. Use inline citations [1], [2], etc. to reference your sources.
3. Be precise with medical terminology.
4. For surgical procedures, emphasize key steps and safety considerations.
5. When uncertain, express appropriate epistemic humility.

Citation format: Use [N] immediately after the information from source N."""

    SYNTHESIS_CONTEXT_PROMPT = """You are answering follow-up questions about a generated textbook chapter.

Chapter Topic: {topic}
Template: {template_type}

The user has access to the following sections:
{section_list}

Use the provided context to answer questions about this chapter. Reference specific sections when relevant."""

    FOLLOW_UP_PROMPT = """Based on the conversation so far, suggest 2-3 relevant follow-up questions the user might want to ask. Format as a JSON array of strings."""

    def __init__(
        self,
        search_service=None,
        anthropic_client=None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        enable_follow_ups: bool = True
    ):
        self.search = search_service
        self.client = anthropic_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_follow_ups = enable_follow_ups

        # Conversation manager
        self.conversations = EnhancedConversationManager()

        # Context assembler (lazy loaded)
        self._assembler = None

    @property
    def assembler(self):
        """Lazy-load context assembler."""
        if self._assembler is None:
            from src.rag.context import ContextAssembler
            self._assembler = ContextAssembler(
                max_context_tokens=8000,
                max_chunks=10,
                include_metadata=True
            )
        return self._assembler

    async def _get_client(self):
        """Get or create Anthropic client."""
        if self.client is None:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        return self.client

    async def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        filters: Optional[Any] = None,
        include_citations: bool = True,
        include_images: bool = True,
        stream: bool = False
    ) -> EnhancedRAGResponse:
        """
        Ask a question with optional conversation context.

        Args:
            question: The question to answer
            conversation_id: Optional conversation ID for context
            filters: Optional search filters
            include_citations: Include citation extraction
            include_images: Include related images
            stream: Stream response (returns generator)
        """
        start_time = time.time()

        # Get synthesis context if conversation is linked
        synthesis_context = None
        if conversation_id:
            synthesis_context = self.conversations.get_synthesis_context(conversation_id)

            # Add user turn
            self.conversations.add_turn(conversation_id, 'user', question)

        # Build search filters based on synthesis context
        if synthesis_context and not filters:
            from src.retrieval.search_service import SearchFilters
            filters = SearchFilters(
                document_ids=synthesis_context.document_ids or None,
                cuis=synthesis_context.all_cuis[:10] if synthesis_context.all_cuis else None
            )

        # Step 1: Search
        search_start = time.time()
        search_response = await self.search.search(
            query=question,
            top_k=20,
            include_images=include_images,
            filters=filters
        )
        search_time = int((time.time() - search_start) * 1000)

        # Step 2: Assemble context
        context_start = time.time()
        context = self.assembler.assemble(
            search_results=search_response.results,
            query=question
        )
        context_time = int((time.time() - context_start) * 1000)

        # Step 3: Build enhanced citations
        enhanced_citations = self._build_enhanced_citations(
            search_response.results[:context.chunks_used]
        )

        # Step 4: Build prompt
        prompt = self._build_prompt(
            question=question,
            context=context,
            synthesis_context=synthesis_context,
            include_images=include_images
        )

        # Step 5: Get conversation history
        conversation_history = None
        if conversation_id:
            conversation_history = self.conversations.get_history_for_claude(conversation_id)

        # Step 6: Generate
        if stream:
            return self._stream_response(
                question=question,
                prompt=prompt,
                enhanced_citations=enhanced_citations,
                context=context,
                search_time=search_time,
                context_time=context_time,
                conversation_id=conversation_id,
                synthesis_context=synthesis_context,
                conversation_history=conversation_history
            )

        gen_start = time.time()
        client = await self._get_client()
        answer = await self._generate(prompt, conversation_history, client)
        gen_time = int((time.time() - gen_start) * 1000)

        # Step 7: Extract used citations
        used_indices = self._extract_citation_indices(answer)

        # Step 8: Generate follow-up questions
        follow_ups = []
        if self.enable_follow_ups:
            follow_ups = await self._generate_follow_ups(question, answer, client)

        total_time = int((time.time() - start_time) * 1000)

        # Build response
        response = EnhancedRAGResponse(
            answer=answer,
            citations=enhanced_citations,
            used_citation_indices=used_indices,
            images=[img.to_dict() if hasattr(img, 'to_dict') else img for img in context.images],
            question=question,
            context_chunks_used=context.chunks_used,
            total_tokens_used=context.total_tokens,
            search_time_ms=search_time,
            context_time_ms=context_time,
            generation_time_ms=gen_time,
            total_time_ms=total_time,
            model=self.model,
            synthesis_context_id=synthesis_context.synthesis_id if synthesis_context else None,
            follow_up_questions=follow_ups
        )

        # Add assistant turn to conversation
        if conversation_id:
            used_citations = [c for c in enhanced_citations if c.index in used_indices]
            self.conversations.add_turn(
                conversation_id,
                'assistant',
                answer,
                citations=used_citations
            )

        return response

    async def ask_about_synthesis(
        self,
        question: str,
        synthesis_result: Any,
        synthesis_id: Optional[str] = None
    ) -> EnhancedRAGResponse:
        """
        Ask a question about a specific synthesis result.

        Creates a new conversation linked to the synthesis.
        """
        # Create synthesis context
        synthesis_context = SynthesisContext.from_synthesis_result(
            synthesis_result,
            synthesis_id
        )

        # Create conversation
        conv_id = self.conversations.create_conversation(synthesis_context)

        # Ask question
        return await self.ask(
            question=question,
            conversation_id=conv_id,
            include_citations=True,
            include_images=True
        )

    def _build_enhanced_citations(
        self,
        results: List[Any]
    ) -> List[EnhancedCitation]:
        """Build enhanced citations from search results."""
        citations = []

        for i, result in enumerate(results, 1):
            content = getattr(result, 'content', '')

            # Create snippet
            snippet = content[:150]
            if len(content) > 150:
                last_period = snippet.rfind('.')
                if last_period > 75:
                    snippet = snippet[:last_period + 1]
                else:
                    snippet = snippet.rstrip() + "..."

            citations.append(EnhancedCitation(
                index=i,
                chunk_id=getattr(result, 'chunk_id', str(i)),
                content=content,
                snippet=snippet,
                document_id=getattr(result, 'document_id', ''),
                document_title=getattr(result, 'document_title', 'Unknown'),
                page_number=getattr(result, 'page_start', 0) or 0,
                chunk_type=str(getattr(result, 'chunk_type', 'GENERAL')),
                authority_score=getattr(result, 'authority_score', 0.7),
                cuis=getattr(result, 'cuis', []) or [],
                entity_names=getattr(result, 'entity_names', []) or []
            ))

        return citations

    def _build_prompt(
        self,
        question: str,
        context: Any,
        synthesis_context: Optional[SynthesisContext],
        include_images: bool
    ) -> str:
        """Build prompt with optional synthesis context."""
        parts = []

        # Add synthesis context if available
        if synthesis_context:
            section_list = "\n".join(f"- {name}" for name in synthesis_context.sections.keys())
            parts.append(self.SYNTHESIS_CONTEXT_PROMPT.format(
                topic=synthesis_context.topic,
                template_type=synthesis_context.template_type,
                section_list=section_list
            ))
            parts.append("")

        # Add retrieved context
        parts.append("Context:")
        parts.append(context.text)
        parts.append("")

        # Add images if present
        if include_images and context.images:
            parts.append("Related Images:")
            for img in context.images[:5]:
                caption = img.caption if hasattr(img, 'caption') else img.get('caption', '')
                parts.append(f"- {caption}")
            parts.append("")

        # Add question
        parts.append(f"Question: {question}")
        parts.append("")
        parts.append("Please provide a comprehensive answer based on the context above, using citations [1], [2], etc.")

        return "\n".join(parts)

    async def _generate(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        client=None
    ) -> str:
        """Generate answer using Claude."""
        messages = []

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current prompt
        messages.append({'role': 'user', 'content': prompt})

        if client is None:
            client = await self._get_client()

        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.SYSTEM_PROMPT,
            messages=messages
        )

        return response.content[0].text

    async def _stream_response(
        self,
        question: str,
        prompt: str,
        enhanced_citations: List[EnhancedCitation],
        context: Any,
        search_time: int,
        context_time: int,
        conversation_id: Optional[str],
        synthesis_context: Optional[SynthesisContext],
        conversation_history: Optional[List[Dict]]
    ) -> AsyncGenerator[str, None]:
        """Stream response with metadata at end."""
        messages = []

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({'role': 'user', 'content': prompt})

        client = await self._get_client()
        gen_start = time.time()
        full_answer = []

        async with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.SYSTEM_PROMPT,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                full_answer.append(text)
                yield text

        gen_time = int((time.time() - gen_start) * 1000)
        answer = ''.join(full_answer)

        # Extract used citations
        used_indices = self._extract_citation_indices(answer)

        # Yield metadata as final chunk
        metadata = {
            'type': 'RAG_METADATA',
            'citations': [c.to_dict() for c in enhanced_citations],
            'used_citation_indices': used_indices,
            'search_time_ms': search_time,
            'context_time_ms': context_time,
            'generation_time_ms': gen_time,
            'synthesis_context_id': synthesis_context.synthesis_id if synthesis_context else None
        }

        yield f"\n\n<!-- {json.dumps(metadata)} -->"

        # Add to conversation
        if conversation_id:
            used_citations = [c for c in enhanced_citations if c.index in used_indices]
            self.conversations.add_turn(
                conversation_id,
                'assistant',
                answer,
                citations=used_citations
            )

    def _extract_citation_indices(self, text: str) -> List[int]:
        """Extract citation indices from text."""
        import re

        indices = set()

        # Match [1], [2], [1,2], [1-3], etc.
        patterns = [
            r'\[(\d+)\]',           # [1]
            r'\[(\d+),\s*(\d+)\]',  # [1, 2]
            r'\[(\d+)-(\d+)\]',     # [1-3]
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m:
                            indices.add(int(m))
                else:
                    indices.add(int(match))

        return sorted(indices)

    async def _generate_follow_ups(
        self,
        question: str,
        answer: str,
        client=None
    ) -> List[str]:
        """Generate follow-up questions."""
        try:
            prompt = f"""Original question: {question}

Answer provided: {answer[:500]}...

{self.FOLLOW_UP_PROMPT}"""

            if client is None:
                client = await self._get_client()

            response = await client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.7,
                messages=[{'role': 'user', 'content': prompt}]
            )

            text = response.content[0].text

            # Parse JSON array
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())

            return []

        except Exception as e:
            logger.warning(f"Failed to generate follow-ups: {e}")
            return []


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Enhanced RAG Engine for NeuroSynth")
    print("=" * 50)
    print("\nFeatures:")
    print("1. Synthesis context linking")
    print("2. Enhanced citations with medical concepts")
    print("3. Multi-turn conversation management")
    print("4. Follow-up question generation")
    print("5. Streaming responses")
    print("\nUsage:")
    print("  from src.chat.engine import EnhancedRAGEngine")
    print("  engine = EnhancedRAGEngine(search_service=search)")
    print("  response = await engine.ask('What is the retrosigmoid approach?')")
