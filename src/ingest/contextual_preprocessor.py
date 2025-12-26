"""
Contextual Chunk Pre-processor for NeuroSynth v2.0
===================================================

Implements Anthropic's Contextual Retrieval approach:
- Enriches each chunk with document/section context BEFORE embedding
- Uses LLM to generate situating context (optional, higher quality)
- Falls back to template-based context (faster, no API cost)

Expected improvement: 49% reduction in retrieval failures (Anthropic research)

Usage:
    from contextual_preprocessor import ContextualPreprocessor, ContextConfig, ContextMode

    # For LLM-powered context (highest quality)
    config = ContextConfig(
        mode=ContextMode.LLM_FULL,
        llm_model="claude-sonnet-4-20250514"
    )
    preprocessor = ContextualPreprocessor(config)

    # Enrich chunks before embedding
    enriched_texts = await preprocessor.process_chunks(chunks, document)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ContextMode(Enum):
    """Context generation mode."""
    TEMPLATE = "template"      # Fast, no LLM calls, ~20% improvement
    LLM_LIGHT = "llm_light"    # haiku-class, ~35% improvement
    LLM_FULL = "llm_full"      # sonnet-class, ~49% improvement


@dataclass
class ContextConfig:
    """Configuration for contextual preprocessing."""
    mode: ContextMode = ContextMode.LLM_FULL
    include_entities: bool = True
    include_chunk_type: bool = True
    max_context_tokens: int = 100
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: Optional[str] = None
    # Batch processing settings
    batch_size: int = 10  # Process N chunks concurrently
    max_retries: int = 3
    retry_delay: float = 1.0


# Template for neurosurgical context (no LLM needed)
NEURO_CONTEXT_TEMPLATE = """[SOURCE: {document_title}]
[CHAPTER: {chapter}]
[SECTION: {section}]
[CONTENT TYPE: {chunk_type}]
[KEY CONCEPTS: {entities}]

{content}"""


# Prompt for LLM-based context generation - optimized for neurosurgery
LLM_CONTEXT_PROMPT = """<document_context>
Title: {doc_title}
Chapter/Section: {section_path}
Content Type: {chunk_type}
Key Medical Entities: {entities}
</document_context>

<chunk>
{chunk_content}
</chunk>

Generate a brief 1-2 sentence context that situates this chunk within the neurosurgical document. Focus on:
- Anatomical structures and their relationships
- Surgical procedure context (if applicable)
- Clinical significance

Respond with ONLY the situating context, no explanations."""


@dataclass
class ContextResult:
    """Result of context generation for a single chunk."""
    chunk_id: str
    original_content: str
    enriched_content: str
    context_mode: ContextMode
    generated_context: str = ""
    success: bool = True
    error: Optional[str] = None


class ContextualPreprocessor:
    """
    Pre-processes chunks to add contextual information before embedding.

    This addresses the "No Context" gap where chunks like "The M1 segment
    courses through the sylvian fissure" lose meaning without knowing
    the document is about MCA anatomy.

    Attributes:
        config: Configuration for context generation
        stats: Processing statistics
    """

    def __init__(self, config: ContextConfig = None):
        """
        Initialize the contextual preprocessor.

        Args:
            config: Configuration settings. Defaults to LLM_FULL mode.
        """
        self.config = config or ContextConfig()
        self._llm_client = None
        self._async_client = None
        self.stats = {
            "total_processed": 0,
            "llm_calls": 0,
            "template_fallbacks": 0,
            "errors": 0
        }

        if self.config.mode in (ContextMode.LLM_LIGHT, ContextMode.LLM_FULL):
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client for context generation."""
        try:
            import anthropic
            api_key = self.config.llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning(
                    "No ANTHROPIC_API_KEY found, falling back to template mode. "
                    "Set ANTHROPIC_API_KEY environment variable or pass llm_api_key in config."
                )
                self.config.mode = ContextMode.TEMPLATE
                return

            self._llm_client = anthropic.Anthropic(api_key=api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info(f"Initialized LLM client with model: {self.config.llm_model}")
        except ImportError:
            logger.warning("anthropic package not found, falling back to template mode")
            self.config.mode = ContextMode.TEMPLATE

    def _extract_entity_text(self, entities: List[Any]) -> str:
        """Extract and format entities for context."""
        if not entities:
            return ""

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities[:15]:  # Top 15 entities
            normalized = getattr(e, 'normalized', str(e))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        return ", ".join(unique[:10])  # Return top 10

    def _build_section_path(self, section_path: List[str]) -> str:
        """Build readable section path."""
        if not section_path:
            return "Unknown Section"
        return " > ".join(section_path)

    def add_context_template(
        self,
        chunk: Any,
        document: Any
    ) -> ContextResult:
        """
        Add template-based context to chunk (fast, no LLM).

        Args:
            chunk: The semantic chunk to enrich (SemanticChunk or similar)
            document: The source document (Document or similar)

        Returns:
            ContextResult with enriched text
        """
        chunk_id = getattr(chunk, 'id', 'unknown')
        content = getattr(chunk, 'content', str(chunk))

        # Extract entities
        entities = ""
        if self.config.include_entities:
            chunk_entities = getattr(chunk, 'entities', [])
            entities = self._extract_entity_text(chunk_entities)

        # Build section path
        section_path = getattr(chunk, 'section_path', [])
        chapter = section_path[0] if section_path else "Unknown"
        section = " > ".join(section_path[1:]) if len(section_path) > 1 else ""

        # Format chunk type
        chunk_type = ""
        if self.config.include_chunk_type:
            ct = getattr(chunk, 'chunk_type', None)
            if ct:
                chunk_type = ct.value.upper() if hasattr(ct, 'value') else str(ct).upper()

        # Get document title
        doc_title = getattr(document, 'title', 'Unknown Document')

        enriched = NEURO_CONTEXT_TEMPLATE.format(
            document_title=doc_title,
            chapter=chapter,
            section=section,
            chunk_type=chunk_type,
            entities=entities,
            content=content
        )

        return ContextResult(
            chunk_id=chunk_id,
            original_content=content,
            enriched_content=enriched,
            context_mode=ContextMode.TEMPLATE,
            success=True
        )

    async def add_context_llm(
        self,
        chunk: Any,
        document: Any,
        document_summary: str = ""
    ) -> ContextResult:
        """
        Add LLM-generated context to chunk (higher quality).

        Args:
            chunk: The semantic chunk to enrich
            document: The source document
            document_summary: Optional pre-computed document summary

        Returns:
            ContextResult with LLM-enriched text
        """
        chunk_id = getattr(chunk, 'id', 'unknown')
        content = getattr(chunk, 'content', str(chunk))

        if not self._async_client:
            result = self.add_context_template(chunk, document)
            self.stats["template_fallbacks"] += 1
            return result

        # Prepare context for LLM
        doc_title = getattr(document, 'title', 'Unknown Document')
        section_path = getattr(chunk, 'section_path', [])
        chunk_type = getattr(chunk, 'chunk_type', None)
        chunk_type_str = chunk_type.value if hasattr(chunk_type, 'value') else str(chunk_type) if chunk_type else "general"

        entities = self._extract_entity_text(getattr(chunk, 'entities', []))

        prompt = LLM_CONTEXT_PROMPT.format(
            doc_title=doc_title,
            section_path=self._build_section_path(section_path),
            chunk_type=chunk_type_str,
            entities=entities,
            chunk_content=content[:2000]  # Limit content length
        )

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = await self._async_client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=self.config.max_context_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                situating_context = response.content[0].text.strip()
                self.stats["llm_calls"] += 1

                # Prepend generated context to chunk
                enriched = f"{situating_context}\n\n{content}"

                return ContextResult(
                    chunk_id=chunk_id,
                    original_content=content,
                    enriched_content=enriched,
                    context_mode=self.config.mode,
                    generated_context=situating_context,
                    success=True
                )

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue

                logger.warning(f"LLM context generation failed after {self.config.max_retries} attempts: {e}")
                self.stats["errors"] += 1

                # Fallback to template
                result = self.add_context_template(chunk, document)
                result.error = str(e)
                self.stats["template_fallbacks"] += 1
                return result

        # Should not reach here, but fallback just in case
        return self.add_context_template(chunk, document)

    async def process_chunks(
        self,
        chunks: List[Any],
        document: Any,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Process all chunks to add contextual information.

        Args:
            chunks: List of chunks to process
            document: Source document
            on_progress: Progress callback (processed, total)

        Returns:
            List of contextually enriched texts for embedding
        """
        total = len(chunks)
        enriched_texts = []
        results: List[ContextResult] = []

        if self.config.mode == ContextMode.TEMPLATE:
            # Fast path: no async needed
            for i, chunk in enumerate(chunks):
                result = self.add_context_template(chunk, document)
                results.append(result)
                enriched_texts.append(result.enriched_content)
                self.stats["total_processed"] += 1

                if on_progress and (i + 1) % 10 == 0:
                    on_progress(i + 1, total)
        else:
            # LLM path: batch processing with concurrency
            batch_size = self.config.batch_size

            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = chunks[batch_start:batch_end]

                # Process batch concurrently
                tasks = [
                    self.add_context_llm(chunk, document)
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        # Handle unexpected exceptions
                        logger.error(f"Unexpected error in batch processing: {result}")
                        self.stats["errors"] += 1
                        continue

                    results.append(result)
                    enriched_texts.append(result.enriched_content)
                    self.stats["total_processed"] += 1

                if on_progress:
                    on_progress(batch_end, total)

        if on_progress:
            on_progress(total, total)

        return enriched_texts

    async def process_chunks_with_results(
        self,
        chunks: List[Any],
        document: Any,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[ContextResult]:
        """
        Process chunks and return full ContextResult objects.

        Useful for debugging or when you need access to the generated context.
        """
        total = len(chunks)
        results: List[ContextResult] = []

        if self.config.mode == ContextMode.TEMPLATE:
            for i, chunk in enumerate(chunks):
                result = self.add_context_template(chunk, document)
                results.append(result)
                self.stats["total_processed"] += 1

                if on_progress and (i + 1) % 10 == 0:
                    on_progress(i + 1, total)
        else:
            batch_size = self.config.batch_size

            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = chunks[batch_start:batch_end]

                tasks = [self.add_context_llm(chunk, document) for chunk in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Unexpected error: {result}")
                        self.stats["errors"] += 1
                        continue
                    results.append(result)
                    self.stats["total_processed"] += 1

                if on_progress:
                    on_progress(batch_end, total)

        if on_progress:
            on_progress(total, total)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        return {
            **self.stats,
            "mode": self.config.mode.value,
            "model": self.config.llm_model if self.config.mode != ContextMode.TEMPLATE else None
        }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_contextual_embedder(
    base_embedder,
    context_config: ContextConfig = None
):
    """
    Wraps an existing embedder with contextual preprocessing.

    Usage:
        from embeddings import LocalTextEmbedder
        from contextual_preprocessor import create_contextual_embedder, ContextConfig, ContextMode

        base = LocalTextEmbedder(model="all-MiniLM-L6-v2")
        contextual = create_contextual_embedder(
            base,
            ContextConfig(mode=ContextMode.LLM_FULL)
        )

        # Now embeddings include context
        embeddings = await contextual.embed_chunks(chunks, document)
    """

    class ContextualTextEmbedder:
        def __init__(self, base, config):
            self.base = base
            self.preprocessor = ContextualPreprocessor(config)

        @property
        def dimension(self):
            return self.base.dimension

        @property
        def model_name(self):
            return f"contextual-{self.base.model_name}"

        async def embed(self, text: str):
            """Embed single text (no context added for raw text)."""
            return await self.base.embed(text)

        async def embed_batch(self, texts: List[str], on_progress=None):
            """Embed batch of texts (no context added for raw texts)."""
            return await self.base.embed_batch(texts, on_progress)

        async def embed_chunks(
            self,
            chunks: List[Any],
            document: Any,
            on_progress: Optional[Callable[[int, int], None]] = None
        ) -> List:
            """Embed chunks with contextual preprocessing."""

            # Step 1: Enrich chunks with context
            enriched_texts = await self.preprocessor.process_chunks(
                chunks, document, on_progress
            )

            # Step 2: Embed enriched texts
            return await self.base.embed_batch(enriched_texts)

        def get_preprocessor_stats(self) -> Dict[str, Any]:
            """Get preprocessing statistics."""
            return self.preprocessor.get_stats()

    return ContextualTextEmbedder(base_embedder, context_config or ContextConfig())


# =============================================================================
# STANDALONE TEST
# =============================================================================

async def _test():
    """Quick test of contextual preprocessing."""
    from dataclasses import dataclass
    from enum import Enum

    class ChunkType(Enum):
        ANATOMY = "anatomy"
        PROCEDURE = "procedure"

    @dataclass
    class MockEntity:
        normalized: str

    @dataclass
    class MockChunk:
        id: str
        content: str
        section_path: List[str]
        chunk_type: ChunkType
        entities: List[MockEntity]

    @dataclass
    class MockDocument:
        title: str

    # Create test data
    chunk = MockChunk(
        id="test-1",
        content="The M1 segment courses through the sylvian fissure, giving off lenticulostriate arteries.",
        section_path=["Vascular Anatomy", "Middle Cerebral Artery", "M1 Segment"],
        chunk_type=ChunkType.ANATOMY,
        entities=[
            MockEntity("M1 segment"),
            MockEntity("sylvian fissure"),
            MockEntity("lenticulostriate arteries"),
            MockEntity("MCA")
        ]
    )

    doc = MockDocument(title="Surgical Anatomy of Cerebral Vasculature")

    # Test template mode
    print("=== TEMPLATE MODE ===")
    preprocessor = ContextualPreprocessor(ContextConfig(mode=ContextMode.TEMPLATE))
    result = preprocessor.add_context_template(chunk, doc)
    print(f"Enriched content:\n{result.enriched_content}\n")

    # Test LLM mode (if API key available)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("=== LLM_FULL MODE ===")
        preprocessor_llm = ContextualPreprocessor(ContextConfig(mode=ContextMode.LLM_FULL))
        result_llm = await preprocessor_llm.add_context_llm(chunk, doc)
        print(f"Generated context: {result_llm.generated_context}")
        print(f"\nEnriched content:\n{result_llm.enriched_content}\n")
        print(f"Stats: {preprocessor_llm.get_stats()}")
    else:
        print("Skipping LLM test - no ANTHROPIC_API_KEY set")


if __name__ == "__main__":
    asyncio.run(_test())
