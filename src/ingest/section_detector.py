"""
NeuroSynth v2.0 - Section Detector
===================================

Detect document structure using font analysis and pattern matching.

Features:
1. Font size analysis for header detection
2. Pattern matching for common header formats
3. Hierarchical section building
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import Counter

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from src.shared.models import Section


@dataclass
class FontStats:
    """Statistics about fonts used in the document."""
    body_size: float           # Most common font size (body text)
    min_size: float
    max_size: float
    sizes: Dict[float, int]    # Size -> count mapping


class SectionDetector:
    """
    Detect document structure using font analysis.
    
    Headers are identified by:
    1. Larger font size than body text
    2. Bold weight
    3. Pattern matching (numbered sections, ALL CAPS)
    4. Position (start of page/paragraph)
    """
    
    # Header patterns for pattern-based detection
    HEADER_PATTERNS = [
        # Numbered sections
        r"^(\d+\.?\d*\.?\d*)\s+([A-Z][^.!?\n]{3,80})$",
        # ALL CAPS headers
        r"^([A-Z][A-Z\s]{4,60})$",
        # Chapter markers
        r"^(Chapter\s+\d+)",
        r"^(CHAPTER\s+\d+)",
        # Common medical section headers
        r"^(Introduction|Background|Methods?|Results?|Discussion|Conclusion)s?$",
        r"^(Surgical\s+Technique|Clinical\s+Presentation|Anatomy|Pathology)$",
        r"^(Indications?|Contraindications?|Complications?)$",
        r"^(Preoperative|Intraoperative|Postoperative)\s+",
        r"^(Differential\s+Diagnosis|Treatment\s+Options?)$",
        r"^(Case\s+Report|Case\s+Presentation|Clinical\s+Case)$",
        r"^(Summary|Key\s+Points?|Pearls?|Pitfalls?)$",
        r"^(References?|Bibliography)$",
    ]
    
    def __init__(self, body_font_size: float = None):
        """
        Initialize the section detector.
        
        Args:
            body_font_size: Expected body text font size (auto-detected if None)
        """
        self.body_font_size = body_font_size
        self._font_stats: Optional[FontStats] = None
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HEADER_PATTERNS
        ]
    
    def detect_sections(self, doc: "fitz.Document") -> List[Section]:
        """
        Detect all sections in a document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of Section objects with hierarchical structure
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required for section detection")
        
        # First pass: analyze font statistics
        self._analyze_fonts(doc)
        
        # Second pass: detect headers and build sections
        sections = []
        current_section: Optional[Section] = None
        current_content: List[str] = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") != 0:  # Skip non-text blocks
                    continue
                
                for line in block.get("lines", []):
                    line_text, font_size, is_bold = self._extract_line_info(line)
                    line_text = line_text.strip()
                    
                    if not line_text:
                        continue
                    
                    # Check if this line is a header
                    is_header, level = self._is_header(line_text, font_size, is_bold)
                    
                    if is_header:
                        # Save previous section
                        if current_section:
                            current_section.content = "\n".join(current_content)
                            current_section.page_end = page_num
                            if current_section.content.strip():
                                sections.append(current_section)
                        
                        # Start new section
                        current_section = Section(
                            title=self._clean_title(line_text),
                            level=level,
                            page_start=page_num,
                            page_end=page_num,
                            content=""
                        )
                        current_content = []
                    else:
                        current_content.append(line_text)
        
        # Save last section
        if current_section:
            current_section.content = "\n".join(current_content)
            current_section.page_end = len(doc) - 1
            if current_section.content.strip():
                sections.append(current_section)
        
        # If no sections found, create a single section for the entire document
        if not sections:
            sections.append(Section(
                title="Content",
                level=1,
                page_start=0,
                page_end=len(doc) - 1,
                content=self._extract_full_text(doc)
            ))

        # Post-process: merge sections that are too short to be meaningful
        sections = self._merge_small_sections(sections)

        return sections

    def _merge_small_sections(
        self,
        sections: List[Section],
        min_words: int = 30
    ) -> List[Section]:
        """
        Merge sections that are too short to be meaningful.

        Prevents table rows and fragment headers from becoming standalone sections.
        Short sections are merged with the following section.

        Args:
            sections: List of detected sections
            min_words: Minimum word count for a section to stand alone

        Returns:
            Merged section list
        """
        if len(sections) <= 1:
            return sections

        merged = []
        pending_merge: Optional[Section] = None

        for section in sections:
            word_count = len(section.content.split())
            title_word_count = len(section.title.split())

            if pending_merge:
                # Merge pending short section with current section
                merged_title = f"{pending_merge.title} / {section.title}"
                merged_content = pending_merge.content
                if merged_content and section.content:
                    merged_content += "\n\n" + section.content
                elif section.content:
                    merged_content = section.content

                section = Section(
                    title=merged_title[:100] if len(merged_title) > 100 else merged_title,
                    level=min(pending_merge.level, section.level),
                    page_start=pending_merge.page_start,
                    page_end=section.page_end,
                    content=merged_content
                )
                pending_merge = None
                word_count = len(section.content.split())

            # Check if this section is too short
            # Consider both content and title (some table headers have title but no content)
            if word_count < min_words and title_word_count < 5:
                # Mark for merging with next section
                pending_merge = section
            else:
                merged.append(section)

        # Handle trailing pending section
        if pending_merge:
            if merged:
                # Append to last section
                last = merged[-1]
                last.content += f"\n\n{pending_merge.title}\n{pending_merge.content}"
                last.page_end = pending_merge.page_end
            else:
                # No sections to merge with, keep it
                merged.append(pending_merge)

        return merged
    
    def _analyze_fonts(self, doc: "fitz.Document"):
        """
        Analyze font usage statistics in the document.
        
        Determines the most common font size (body text) for comparison.
        """
        sizes: List[float] = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 10)
                        if size > 0:
                            sizes.append(round(size, 1))
        
        if sizes:
            # Most common size is likely body text
            size_counts = Counter(sizes)
            body_size = size_counts.most_common(1)[0][0]
            
            self._font_stats = FontStats(
                body_size=body_size,
                min_size=min(sizes),
                max_size=max(sizes),
                sizes=dict(size_counts)
            )
            
            # Override manual setting if auto-detected
            if self.body_font_size is None:
                self.body_font_size = body_size
        else:
            # Fallback defaults
            self.body_font_size = self.body_font_size or 10.0
            self._font_stats = FontStats(
                body_size=self.body_font_size,
                min_size=8.0,
                max_size=20.0,
                sizes={}
            )
    
    def _extract_line_info(self, line: dict) -> Tuple[str, float, bool]:
        """
        Extract text, font size, and bold status from a line.
        
        Args:
            line: PyMuPDF line dictionary
            
        Returns:
            Tuple of (text, max_font_size, is_bold)
        """
        text = ""
        max_size = 0.0
        is_bold = False
        
        for span in line.get("spans", []):
            text += span.get("text", "")
            size = span.get("size", 10)
            max_size = max(max_size, size)
            
            # Check bold flag (bit 2^4 = 16)
            flags = span.get("flags", 0)
            if flags & 16:
                is_bold = True
        
        return text, max_size, is_bold
    
    def _is_header(
        self,
        text: str,
        font_size: float,
        is_bold: bool
    ) -> Tuple[bool, int]:
        """
        Determine if text is a header and its level.
        
        Args:
            text: The text content
            font_size: Font size of the text
            is_bold: Whether the text is bold
            
        Returns:
            Tuple of (is_header, level) where level 1 is main heading
        """
        text = text.strip()
        
        # Skip very short or very long text
        if len(text) < 3 or len(text) > 120:
            return False, 0
        
        # Skip if it looks like a sentence (ends with period and has lowercase)
        if text.endswith('.') and any(c.islower() for c in text[:-20]):
            return False, 0
        
        body_size = self.body_font_size or 10.0
        
        # Font size analysis
        size_ratio = font_size / body_size
        
        # Large font = likely header
        if size_ratio >= 1.4:
            level = 1 if size_ratio >= 1.6 else 2
            return True, level
        
        # Bold with moderate size increase
        if is_bold and size_ratio >= 1.1:
            return True, 2
        
        # Bold same size but ALL CAPS
        if is_bold and text.isupper() and len(text) > 4:
            return True, 2
        
        # Pattern matching
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                # Determine level by pattern type
                if re.match(r"^Chapter\s+\d+", text, re.IGNORECASE):
                    return True, 1
                if re.match(r"^\d+\.\d+", text):  # Sub-section numbering
                    return True, 2
                if re.match(r"^\d+\.\d+\.\d+", text):  # Sub-sub-section
                    return True, 3
                return True, 1
        
        return False, 0
    
    def _clean_title(self, text: str) -> str:
        """
        Clean up header text for use as section title.
        """
        # Remove trailing punctuation
        text = text.strip().rstrip(':.-–')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Limit length
        if len(text) > 100:
            text = text[:100] + "..."
        
        return text
    
    def _extract_full_text(self, doc: "fitz.Document") -> str:
        """
        Extract all text from document.
        """
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n\n".join(texts)
    
    def get_font_stats(self) -> Optional[FontStats]:
        """
        Get font statistics from last analysis.
        """
        return self._font_stats


class OutlineBasedSectionDetector(SectionDetector):
    """
    Extended section detector that uses PDF outline (bookmarks) if available.
    
    Falls back to font-based detection if no outline exists.
    """
    
    def detect_sections(self, doc: "fitz.Document") -> List[Section]:
        """
        Detect sections using outline first, then fall back to font analysis.
        """
        # Try outline first
        outline = doc.get_toc()
        
        if outline and len(outline) > 3:
            # Use outline-based detection
            return self._sections_from_outline(doc, outline)
        
        # Fall back to font-based detection
        return super().detect_sections(doc)
    
    def _sections_from_outline(
        self,
        doc: "fitz.Document",
        outline: List[List]
    ) -> List[Section]:
        """
        Build sections from PDF outline/bookmarks.
        
        Args:
            doc: PyMuPDF document
            outline: List of [level, title, page] entries
            
        Returns:
            List of Section objects
        """
        sections = []
        
        for i, entry in enumerate(outline):
            level, title, page_num = entry[0], entry[1], entry[2]
            
            # Determine page range
            if i + 1 < len(outline):
                next_page = outline[i + 1][2]
                page_end = max(page_num, next_page - 1)
            else:
                page_end = len(doc) - 1
            
            # Extract content for this section
            content = self._extract_section_content(doc, page_num, page_end)
            
            sections.append(Section(
                title=title,
                level=level,
                page_start=page_num,
                page_end=page_end,
                content=content
            ))
        
        return sections
    
    def _extract_section_content(
        self,
        doc: "fitz.Document",
        start_page: int,
        end_page: int
    ) -> str:
        """
        Extract text content for a page range.
        """
        texts = []
        for page_num in range(start_page, min(end_page + 1, len(doc))):
            page = doc[page_num]
            texts.append(page.get_text("text"))
        return "\n".join(texts)
