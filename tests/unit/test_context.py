"""
NeuroSynth - Context Assembly Unit Tests
=========================================

Tests for RAG context assembly and citation tracking.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# Mock SearchResult for testing
# =============================================================================

@dataclass
class MockSearchResult:
    """Mock search result for context tests."""
    id: str
    content: str
    score: float = 0.9
    result_type: str = "chunk"
    document_id: str = "doc-1"
    page_number: int = 1
    chunk_type: str = "PROCEDURE"
    specialty: str = "skull_base"
    cuis: List[str] = None
    linked_images: List[dict] = None
    
    def __post_init__(self):
        self.cuis = self.cuis or []
        self.linked_images = self.linked_images or []


# =============================================================================
# Token Estimation Tests
# =============================================================================

class TestTokenEstimation:
    """Tests for token estimation utilities."""
    
    def test_estimate_tokens_empty(self):
        """Empty string returns 0."""
        from src.rag.context import estimate_tokens
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0
    
    def test_estimate_tokens_short(self):
        """Short text estimation."""
        from src.rag.context import estimate_tokens
        # "Hello world" = 2 words * 1.3 ≈ 2-3 tokens
        result = estimate_tokens("Hello world")
        assert 2 <= result <= 4
    
    def test_estimate_tokens_paragraph(self):
        """Paragraph estimation."""
        from src.rag.context import estimate_tokens
        text = "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle. Patient positioning is lateral with the head turned."
        # ~22 words * 1.3 ≈ 28 tokens
        result = estimate_tokens(text)
        assert 20 <= result <= 40
    
    def test_truncate_to_tokens_no_truncation(self):
        """Short text not truncated."""
        from src.rag.context import truncate_to_tokens
        text = "Short text"
        result = truncate_to_tokens(text, 100)
        assert result == text
    
    def test_truncate_to_tokens_with_truncation(self):
        """Long text is truncated."""
        from src.rag.context import truncate_to_tokens
        text = "This is a longer sentence that should be truncated. " * 10
        result = truncate_to_tokens(text, 20)
        assert len(result) < len(text)
        assert result.endswith("...") or result.endswith(".")


# =============================================================================
# Citation Tests
# =============================================================================

class TestCitation:
    """Tests for Citation dataclass."""
    
    def test_citation_creation(self):
        """Create citation with required fields."""
        from src.rag.context import Citation
        
        citation = Citation(
            index=1,
            chunk_id="chunk-123",
            content="Full content here",
            snippet="Short snippet..."
        )
        
        assert citation.index == 1
        assert citation.chunk_id == "chunk-123"
        assert citation.snippet == "Short snippet..."
    
    def test_citation_to_dict(self):
        """Citation converts to dict."""
        from src.rag.context import Citation
        
        citation = Citation(
            index=1,
            chunk_id="chunk-123",
            content="Content",
            snippet="Snippet",
            page_number=45,
            chunk_type="PROCEDURE"
        )
        
        d = citation.to_dict()
        assert d["index"] == 1
        assert d["chunk_id"] == "chunk-123"
        assert d["page_number"] == 45
        assert "content" not in d  # Full content not in dict


# =============================================================================
# Context Assembler Tests
# =============================================================================

class TestContextAssembler:
    """Tests for ContextAssembler."""
    
    def test_assemble_empty(self):
        """Empty results return no context message."""
        from src.rag.context import ContextAssembler
        
        assembler = ContextAssembler()
        context = assembler.assemble([])
        
        assert "No relevant" in context.text
        assert context.chunks_used == 0
        assert len(context.citations) == 0
    
    def test_assemble_single_chunk(self):
        """Single chunk assembled correctly."""
        from src.rag.context import ContextAssembler
        
        results = [
            MockSearchResult(
                id="1",
                content="Test content for chunk one.",
                chunk_type="PROCEDURE",
                page_number=10
            )
        ]
        
        assembler = ContextAssembler()
        context = assembler.assemble(results)
        
        assert context.chunks_used == 1
        assert len(context.citations) == 1
        assert "[1]" in context.text
        assert "Test content" in context.text
    
    def test_assemble_multiple_chunks(self):
        """Multiple chunks assembled with citations."""
        from src.rag.context import ContextAssembler
        
        results = [
            MockSearchResult(id="1", content="First chunk content."),
            MockSearchResult(id="2", content="Second chunk content."),
            MockSearchResult(id="3", content="Third chunk content.")
        ]
        
        assembler = ContextAssembler(max_chunks=10)
        context = assembler.assemble(results)
        
        assert context.chunks_used == 3
        assert len(context.citations) == 3
        assert "[1]" in context.text
        assert "[2]" in context.text
        assert "[3]" in context.text
    
    def test_assemble_respects_max_chunks(self):
        """Assembler respects max_chunks limit."""
        from src.rag.context import ContextAssembler
        
        results = [
            MockSearchResult(id=str(i), content=f"Chunk {i} content.")
            for i in range(10)
        ]
        
        assembler = ContextAssembler(max_chunks=3)
        context = assembler.assemble(results)
        
        assert context.chunks_used == 3
        assert context.chunks_available == 10
    
    def test_assemble_respects_token_budget(self):
        """Assembler respects token budget."""
        from src.rag.context import ContextAssembler
        
        long_content = "Word " * 500  # ~500 tokens
        results = [
            MockSearchResult(id=str(i), content=long_content)
            for i in range(5)
        ]
        
        assembler = ContextAssembler(max_context_tokens=200, max_chunks=10)
        context = assembler.assemble(results)
        
        # Should only include partial content due to token limit
        assert context.total_tokens <= 250  # Some buffer
    
    def test_assemble_deduplicates(self):
        """Assembler deduplicates identical content."""
        from src.rag.context import ContextAssembler
        
        same_content = "Identical content here."
        results = [
            MockSearchResult(id="1", content=same_content),
            MockSearchResult(id="2", content=same_content),  # Duplicate
            MockSearchResult(id="3", content="Different content.")
        ]
        
        assembler = ContextAssembler()
        context = assembler.assemble(results)
        
        # Should dedupe to 2 unique
        assert context.chunks_used == 2
    
    def test_assemble_includes_metadata(self):
        """Assembler includes metadata when enabled."""
        from src.rag.context import ContextAssembler
        
        results = [
            MockSearchResult(
                id="1",
                content="Content here.",
                chunk_type="PROCEDURE",
                page_number=45
            )
        ]
        
        assembler = ContextAssembler(include_metadata=True)
        context = assembler.assemble(results)
        
        assert "PROCEDURE" in context.text
        assert "p.45" in context.text
    
    def test_assemble_xml_format(self):
        """Assembler supports XML format."""
        from src.rag.context import ContextAssembler, ContextFormat
        
        results = [
            MockSearchResult(id="1", content="Content here.")
        ]
        
        assembler = ContextAssembler(format=ContextFormat.XML)
        context = assembler.assemble(results)
        
        assert "<source" in context.text
        assert "</source>" in context.text
    
    def test_assemble_collects_images(self):
        """Assembler collects linked images."""
        from src.rag.context import ContextAssembler
        
        results = [
            MockSearchResult(
                id="1",
                content="Content with image.",
                linked_images=[
                    {"image_id": "img1", "file_path": "/img/1.png", "caption": "Figure 1"}
                ]
            )
        ]
        
        assembler = ContextAssembler()
        context = assembler.assemble(results)
        
        assert len(context.images) == 1
        assert context.images[0].image_id == "img1"


# =============================================================================
# Citation Extractor Tests
# =============================================================================

class TestCitationExtractor:
    """Tests for CitationExtractor."""
    
    def test_extract_single(self):
        """Extract single citation [1]."""
        from src.rag.context import CitationExtractor
        
        text = "The approach provides exposure [1]."
        indices = CitationExtractor.extract(text)
        
        assert indices == [1]
    
    def test_extract_multiple(self):
        """Extract multiple citations."""
        from src.rag.context import CitationExtractor
        
        text = "First point [1]. Second point [2]. Third point [3]."
        indices = CitationExtractor.extract(text)
        
        assert indices == [1, 2, 3]
    
    def test_extract_comma_separated(self):
        """Extract comma-separated citations [1,2,3]."""
        from src.rag.context import CitationExtractor
        
        text = "Multiple sources support this [1, 2, 3]."
        indices = CitationExtractor.extract(text)
        
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices
    
    def test_extract_range(self):
        """Extract range citations [1-3]."""
        from src.rag.context import CitationExtractor
        
        text = "See sources [1-3] for details."
        indices = CitationExtractor.extract(text)
        
        assert indices == [1, 2, 3]
    
    def test_extract_no_duplicates(self):
        """Extract without duplicates."""
        from src.rag.context import CitationExtractor
        
        text = "Point [1] and again [1] with [2]."
        indices = CitationExtractor.extract(text)
        
        assert indices == [1, 2]
    
    def test_extract_empty(self):
        """No citations returns empty list."""
        from src.rag.context import CitationExtractor
        
        text = "No citations here."
        indices = CitationExtractor.extract(text)
        
        assert indices == []
    
    def test_get_used_citations(self):
        """Get only used citations from list."""
        from src.rag.context import CitationExtractor, Citation
        
        all_citations = [
            Citation(index=1, chunk_id="1", content="", snippet="First"),
            Citation(index=2, chunk_id="2", content="", snippet="Second"),
            Citation(index=3, chunk_id="3", content="", snippet="Third")
        ]
        
        text = "Only uses first [1] and third [3]."
        used = CitationExtractor.get_used_citations(text, all_citations)
        
        assert len(used) == 2
        assert used[0].index == 1
        assert used[1].index == 3


# =============================================================================
# AssembledContext Tests
# =============================================================================

class TestAssembledContext:
    """Tests for AssembledContext."""
    
    def test_to_dict(self):
        """AssembledContext converts to dict."""
        from src.rag.context import AssembledContext, Citation, ContextImage
        
        context = AssembledContext(
            text="Context text",
            citations=[
                Citation(index=1, chunk_id="1", content="C", snippet="S")
            ],
            images=[
                ContextImage(image_id="i1", file_path="/p", caption="Cap")
            ],
            total_tokens=100,
            chunks_used=1,
            chunks_available=5
        )
        
        d = context.to_dict()
        
        assert d["context_text"] == "Context text"
        assert len(d["citations"]) == 1
        assert len(d["images"]) == 1
        assert d["total_tokens"] == 100
    
    def test_get_citation_by_index(self):
        """Get citation by index number."""
        from src.rag.context import AssembledContext, Citation
        
        context = AssembledContext(
            text="",
            citations=[
                Citation(index=1, chunk_id="1", content="", snippet="First"),
                Citation(index=2, chunk_id="2", content="", snippet="Second")
            ],
            images=[],
            total_tokens=0,
            chunks_used=2,
            chunks_available=2
        )
        
        c1 = context.get_citation_by_index(1)
        c2 = context.get_citation_by_index(2)
        c3 = context.get_citation_by_index(3)
        
        assert c1.snippet == "First"
        assert c2.snippet == "Second"
        assert c3 is None
