"""
NeuroSynth Unified - RAG Context Assembler
==========================================

Assembles context from search results for RAG generation.
Handles token budgets, chunk selection, and citation preparation.

Usage:
    from src.rag.context import ContextAssembler
    
    assembler = ContextAssembler(max_tokens=8000)
    context = assembler.assemble(search_results)
    
    # context.text - formatted context string
    # context.citations - list of citation objects
    # context.images - relevant images
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================

@dataclass
class Citation:
    """Citation reference for RAG output."""
    index: int                    # [1], [2], etc.
    chunk_id: str                 # Source chunk ID
    content: str                  # Chunk content (for reference)
    snippet: str                  # Short snippet for display
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'chunk_id': self.chunk_id,
            'snippet': self.snippet,
            'document_id': self.document_id,
            'page_number': self.page_number,
            'chunk_type': self.chunk_type
        }


@dataclass
class ContextImage:
    """Image included in RAG context."""
    image_id: str
    file_path: str
    caption: str
    image_type: Optional[str] = None
    link_score: float = 0.0
    linked_chunk_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'image_id': self.image_id,
            'file_path': self.file_path,
            'caption': self.caption,
            'image_type': self.image_type
        }


@dataclass
class AssembledContext:
    """Complete assembled context for RAG."""
    text: str                           # Formatted context string
    citations: List[Citation]           # Citation objects
    images: List[ContextImage]          # Related images
    total_tokens: int                   # Estimated token count
    chunks_used: int                    # Number of chunks included
    chunks_available: int               # Total chunks from search
    
    def get_citation_by_index(self, index: int) -> Optional[Citation]:
        """Get citation by index number."""
        for c in self.citations:
            if c.index == index:
                return c
        return None
    
    def to_dict(self) -> Dict:
        return {
            'context_text': self.text,
            'citations': [c.to_dict() for c in self.citations],
            'images': [i.to_dict() for i in self.images],
            'total_tokens': self.total_tokens,
            'chunks_used': self.chunks_used,
            'chunks_available': self.chunks_available
        }


class ContextFormat(Enum):
    """Context formatting styles."""
    NUMBERED = "numbered"       # [1] Content... [2] Content...
    XML = "xml"                 # <context id="1">Content</context>
    MARKDOWN = "markdown"       # ## Source 1\nContent...


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Rough approximation: ~4 characters per token for English.
    Claude's tokenizer is more nuanced but this is close enough
    for budget planning.
    """
    if not text:
        return 0
    # More accurate: count words and multiply by ~1.3
    words = len(text.split())
    return int(words * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return text
    
    # Approximate characters for max_tokens
    max_chars = int(max_tokens * 4)
    
    if len(text) <= max_chars:
        return text
    
    # Truncate at sentence boundary if possible
    truncated = text[:max_chars]
    
    # Find last sentence end
    for end in ['. ', '.\n', '? ', '! ']:
        last_end = truncated.rfind(end)
        if last_end > max_chars * 0.7:  # At least 70% of target
            return truncated[:last_end + 1]
    
    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


# =============================================================================
# Context Assembler
# =============================================================================

class ContextAssembler:
    """
    Assembles context from search results for RAG.
    
    Features:
    - Token budget management
    - Multiple formatting styles
    - Citation tracking
    - Image inclusion
    - Deduplication
    """
    
    def __init__(
        self,
        max_context_tokens: int = 8000,
        max_chunks: int = 10,
        max_images: int = 5,
        format: ContextFormat = ContextFormat.NUMBERED,
        include_metadata: bool = True,
        snippet_length: int = 150
    ):
        """
        Initialize context assembler.
        
        Args:
            max_context_tokens: Maximum tokens for context
            max_chunks: Maximum number of chunks to include
            max_images: Maximum number of images
            format: Context formatting style
            include_metadata: Include chunk type, page number
            snippet_length: Length of citation snippets
        """
        self.max_context_tokens = max_context_tokens
        self.max_chunks = max_chunks
        self.max_images = max_images
        self.format = format
        self.include_metadata = include_metadata
        self.snippet_length = snippet_length
    
    def assemble(
        self,
        search_results: List[Any],  # SearchResult objects
        query: str = None
    ) -> AssembledContext:
        """
        Assemble context from search results.
        
        Args:
            search_results: List of SearchResult objects
            query: Original query (for relevance hints)
        
        Returns:
            AssembledContext with formatted text and citations
        """
        if not search_results:
            return AssembledContext(
                text="No relevant information found.",
                citations=[],
                images=[],
                total_tokens=5,
                chunks_used=0,
                chunks_available=0
            )
        
        # Filter to chunks only (not images)
        chunks = [r for r in search_results if getattr(r, 'result_type', 'chunk') == 'chunk']
        chunks_available = len(chunks)
        
        # Select chunks within budget
        selected_chunks, total_tokens = self._select_chunks(chunks)
        
        # Build citations
        citations = self._build_citations(selected_chunks)
        
        # Format context
        context_text = self._format_context(selected_chunks, citations)
        
        # Collect images from linked images
        images = self._collect_images(selected_chunks)
        
        return AssembledContext(
            text=context_text,
            citations=citations,
            images=images,
            total_tokens=total_tokens,
            chunks_used=len(selected_chunks),
            chunks_available=chunks_available
        )
    
    def _select_chunks(
        self,
        chunks: List[Any]
    ) -> Tuple[List[Any], int]:
        """Select chunks within token budget."""
        selected = []
        total_tokens = 0
        seen_content = set()  # For deduplication
        
        for chunk in chunks[:self.max_chunks * 2]:  # Consider more for dedup
            if len(selected) >= self.max_chunks:
                break
            
            content = getattr(chunk, 'content', '')
            
            # Skip duplicates (by content hash)
            content_hash = hash(content[:200])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Estimate tokens for this chunk
            chunk_tokens = estimate_tokens(content)
            
            # Check if adding this chunk exceeds budget
            if total_tokens + chunk_tokens > self.max_context_tokens:
                # Try truncating
                remaining = self.max_context_tokens - total_tokens
                if remaining > 200:  # Worth including truncated
                    truncated = truncate_to_tokens(content, remaining)
                    chunk.content = truncated
                    selected.append(chunk)
                    total_tokens += estimate_tokens(truncated)
                break
            
            selected.append(chunk)
            total_tokens += chunk_tokens
        
        return selected, total_tokens
    
    def _build_citations(
        self,
        chunks: List[Any]
    ) -> List[Citation]:
        """Build citation objects from chunks."""
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            content = getattr(chunk, 'content', '')
            
            # Create snippet
            snippet = content[:self.snippet_length]
            if len(content) > self.snippet_length:
                # Try to end at sentence
                last_period = snippet.rfind('.')
                if last_period > self.snippet_length * 0.5:
                    snippet = snippet[:last_period + 1]
                else:
                    snippet = snippet.rstrip() + "..."
            
            citations.append(Citation(
                index=i,
                chunk_id=getattr(chunk, 'id', str(i)),
                content=content,
                snippet=snippet,
                document_id=getattr(chunk, 'document_id', None),
                page_number=getattr(chunk, 'page_number', None),
                chunk_type=getattr(chunk, 'chunk_type', None)
            ))
        
        return citations
    
    def _format_context(
        self,
        chunks: List[Any],
        citations: List[Citation]
    ) -> str:
        """Format context string based on format style."""
        if self.format == ContextFormat.NUMBERED:
            return self._format_numbered(chunks, citations)
        elif self.format == ContextFormat.XML:
            return self._format_xml(chunks, citations)
        elif self.format == ContextFormat.MARKDOWN:
            return self._format_markdown(chunks, citations)
        else:
            return self._format_numbered(chunks, citations)
    
    def _format_numbered(
        self,
        chunks: List[Any],
        citations: List[Citation]
    ) -> str:
        """Format as numbered paragraphs: [1] Content..."""
        parts = []
        
        for chunk, citation in zip(chunks, citations):
            content = getattr(chunk, 'content', '')
            
            if self.include_metadata:
                meta_parts = []
                if citation.chunk_type:
                    meta_parts.append(citation.chunk_type)
                if citation.page_number:
                    meta_parts.append(f"p.{citation.page_number}")
                
                if meta_parts:
                    meta_str = f" ({', '.join(meta_parts)})"
                else:
                    meta_str = ""
                
                parts.append(f"[{citation.index}]{meta_str} {content}")
            else:
                parts.append(f"[{citation.index}] {content}")
        
        return "\n\n".join(parts)
    
    def _format_xml(
        self,
        chunks: List[Any],
        citations: List[Citation]
    ) -> str:
        """Format as XML elements."""
        parts = []
        
        for chunk, citation in zip(chunks, citations):
            content = getattr(chunk, 'content', '')
            
            attrs = [f'id="{citation.index}"']
            if self.include_metadata:
                if citation.chunk_type:
                    attrs.append(f'type="{citation.chunk_type}"')
                if citation.page_number:
                    attrs.append(f'page="{citation.page_number}"')
            
            attr_str = " ".join(attrs)
            parts.append(f"<source {attr_str}>\n{content}\n</source>")
        
        return "\n\n".join(parts)
    
    def _format_markdown(
        self,
        chunks: List[Any],
        citations: List[Citation]
    ) -> str:
        """Format as markdown sections."""
        parts = []
        
        for chunk, citation in zip(chunks, citations):
            content = getattr(chunk, 'content', '')
            
            header = f"### Source {citation.index}"
            if self.include_metadata:
                meta_parts = []
                if citation.chunk_type:
                    meta_parts.append(citation.chunk_type)
                if citation.page_number:
                    meta_parts.append(f"Page {citation.page_number}")
                if meta_parts:
                    header += f" ({', '.join(meta_parts)})"
            
            parts.append(f"{header}\n\n{content}")
        
        return "\n\n---\n\n".join(parts)
    
    def _collect_images(
        self,
        chunks: List[Any]
    ) -> List[ContextImage]:
        """Collect unique images from chunk linked_images."""
        images = []
        seen_ids = set()
        
        for chunk in chunks:
            linked = getattr(chunk, 'linked_images', [])
            
            for img in linked:
                img_id = img.get('image_id', '')
                if img_id and img_id not in seen_ids:
                    seen_ids.add(img_id)
                    
                    images.append(ContextImage(
                        image_id=img_id,
                        file_path=img.get('file_path', ''),
                        caption=img.get('caption', ''),
                        image_type=img.get('image_type'),
                        link_score=img.get('link_score', 0),
                        linked_chunk_ids=[getattr(chunk, 'id', '')]
                    ))
                    
                    if len(images) >= self.max_images:
                        return images
        
        return images


# =============================================================================
# Citation Extractor
# =============================================================================

class CitationExtractor:
    """
    Extracts citation references from generated text.
    
    Matches patterns like [1], [2], [1,2], [1-3], etc.
    """
    
    # Patterns for citation references
    PATTERNS = [
        r'\[(\d+)\]',           # [1]
        r'\[(\d+),\s*(\d+)\]',  # [1, 2]
        r'\[(\d+)-(\d+)\]',     # [1-3]
        r'\[(\d+(?:,\s*\d+)+)\]'  # [1,2,3]
    ]
    
    @classmethod
    def extract(cls, text: str) -> List[int]:
        """
        Extract all citation indices from text.
        
        Returns:
            Sorted list of unique citation indices
        """
        indices = set()
        
        # Single citations [1]
        for match in re.finditer(r'\[(\d+)\]', text):
            indices.add(int(match.group(1)))
        
        # Comma-separated [1,2,3]
        for match in re.finditer(r'\[([\d,\s]+)\]', text):
            for num in re.findall(r'\d+', match.group(1)):
                indices.add(int(num))
        
        # Ranges [1-3]
        for match in re.finditer(r'\[(\d+)-(\d+)\]', text):
            start, end = int(match.group(1)), int(match.group(2))
            indices.update(range(start, end + 1))
        
        return sorted(indices)
    
    @classmethod
    def get_used_citations(
        cls,
        text: str,
        all_citations: List[Citation]
    ) -> List[Citation]:
        """Get only the citations that were actually used in text."""
        used_indices = cls.extract(text)
        return [c for c in all_citations if c.index in used_indices]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Test context assembler
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        id: str
        content: str
        result_type: str = "chunk"
        chunk_type: str = None
        page_number: int = None
        linked_images: list = None
    
    # Create test results
    results = [
        MockResult(
            id="1",
            content="The retrosigmoid approach provides excellent exposure of the cerebellopontine angle. Patient positioning is crucial for optimal surgical access.",
            chunk_type="PROCEDURE",
            page_number=45,
            linked_images=[{"image_id": "img1", "file_path": "/img/1.png", "caption": "Surgical corridor"}]
        ),
        MockResult(
            id="2", 
            content="Facial nerve preservation is a key goal in acoustic neuroma surgery. Intraoperative monitoring should be used throughout the procedure.",
            chunk_type="PROCEDURE",
            page_number=47
        ),
        MockResult(
            id="3",
            content="The tumor is typically found medial to the facial nerve at the internal auditory canal. Careful dissection is required.",
            chunk_type="ANATOMY",
            page_number=48
        )
    ]
    
    assembler = ContextAssembler(max_context_tokens=1000, max_chunks=5)
    context = assembler.assemble(results)
    
    print("=== Assembled Context ===")
    print(f"Chunks used: {context.chunks_used}/{context.chunks_available}")
    print(f"Tokens: ~{context.total_tokens}")
    print(f"Images: {len(context.images)}")
    print()
    print("=== Context Text ===")
    print(context.text)
    print()
    print("=== Citations ===")
    for c in context.citations:
        print(f"[{c.index}] {c.snippet}")
