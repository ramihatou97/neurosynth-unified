"""
NeuroSynth v2.0 - Table Extractor (Enhanced)
=============================================

Extract tables with structure preserved as both Markdown and HTML.

Critical for neurosurgical content:
- Grading scales (Spetzler-Martin, Hunt-Hess, Fisher)
- Outcome tables
- Comparison tables
- Dosage tables

Enhancements:
- HTML output preserves structural fidelity (rowspans/colspans)
- Improved table type classification
- Better title detection
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from uuid import uuid4
import html

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

logger = logging.getLogger(__name__)

from src.shared.models import ExtractedTable


@dataclass
class TableExtractionConfig:
    """Configuration for table extraction."""
    title_search_height: float = 60.0  # Pixels above table to search for title
    min_rows: int = 2                   # Minimum rows to be considered a table
    min_cols: int = 2                   # Minimum columns
    include_html: bool = True           # Generate HTML in addition to Markdown


class TableExtractor:
    """
    Extract tables with structure preserved as Markdown and HTML.
    
    Strategy:
    1. Use PyMuPDF's built-in table detection
    2. Convert to Markdown format (backward compatible)
    3. Convert to HTML format (preserves structural fidelity)
    4. Classify table type
    5. Extract title from surrounding text
    """
    
    # Table type classification patterns
    TYPE_PATTERNS = {
        "grading_scale": [
            r"grade|grading|scale|score|classification",
            r"spetzler|hunt.?hess|fisher|rankin|karnofsky|who|wfns",
            r"house.?brackmann|mccormick|nurick"
        ],
        "outcomes": [
            r"outcome|result|follow.?up|survival|mortality|morbidity",
            r"complication|adverse|event"
        ],
        "comparison": [
            r"versus|vs\.?|compared|comparison|difference",
            r"advantages?|disadvantages?"
        ],
        "dosage": [
            r"dose|dosage|mg|ml|concentration|medication",
            r"regimen|protocol|infusion"
        ],
        "anatomy": [
            r"nerve|artery|vein|muscle|origin|insertion|innervation",
            r"course|branch|supply|drain"
        ],
        "differential": [
            r"differential|diagnosis|distinguish|feature",
            r"criteria|characteristics"
        ],
        "surgical": [
            r"approach|technique|step|procedure|instrument",
            r"position|retractor|exposure"
        ],
        "imaging": [
            r"mri|ct|radiograph|imaging|finding",
            r"signal|enhancement|characteristic"
        ]
    }
    
    def __init__(self, config: TableExtractionConfig = None):
        """
        Initialize the table extractor.
        
        Args:
            config: Extraction configuration
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required for table extraction")
        
        self.config = config or TableExtractionConfig()
        
        self._compiled_patterns = {
            table_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for table_type, patterns in self.TYPE_PATTERNS.items()
        }
    
    def extract_tables(
        self,
        page: "fitz.Page",
        page_num: int,
        document_id: str
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            document_id: Parent document ID
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        try:
            # Use PyMuPDF's table finder (requires PyMuPDF 1.23+)
            tab_finder = page.find_tables()
            
            for i, tab in enumerate(tab_finder.tables):
                # Get table data
                try:
                    data = tab.extract()
                except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                    logger.debug(f"Table extraction failed on page {page_num}, table {i}: {e}")
                    continue
                
                if not data:
                    continue
                
                # Check minimum size
                if len(data) < self.config.min_rows:
                    continue
                if len(data[0]) < self.config.min_cols:
                    continue
                
                # Convert to Markdown
                markdown = self._to_markdown(data)
                
                # Convert to HTML (preserves structure)
                html_content = ""
                if self.config.include_html:
                    html_content = self._to_html(data)
                
                # Get raw text representation
                raw_text = self._to_raw_text(data)
                
                # Classify table type
                table_type = self._classify_table(markdown, raw_text)
                
                # Find title
                title = self._find_title(page, tab.bbox)
                
                # Create table object
                table = ExtractedTable(
                    id=str(uuid4()),
                    document_id=document_id,
                    page_number=page_num,
                    markdown_content=markdown,
                    html_content=html_content,
                    raw_text=raw_text,
                    table_type=table_type,
                    title=title
                )
                
                tables.append(table)
                
        except AttributeError:
            # find_tables() not available in older PyMuPDF versions
            print(f"Warning: Table extraction requires PyMuPDF 1.23+")
        except Exception as e:
            print(f"Warning: Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _to_markdown(self, data: List[List]) -> str:
        """
        Convert table data to Markdown format.
        
        Args:
            data: List of rows, each row is list of cell values
            
        Returns:
            Markdown table string
        """
        if not data:
            return ""
        
        lines = []
        
        # Normalize column count
        max_cols = max(len(row) for row in data)
        
        # Header row
        header = data[0]
        header_cells = [
            self._escape_markdown(str(cell or "").strip())
            for cell in header
        ]
        # Pad to max columns
        while len(header_cells) < max_cols:
            header_cells.append("")
        lines.append("| " + " | ".join(header_cells) + " |")
        
        # Separator row
        sep_cells = ["---" for _ in range(max_cols)]
        lines.append("| " + " | ".join(sep_cells) + " |")
        
        # Data rows
        for row in data[1:]:
            cells = [
                self._escape_markdown(str(cell or "").strip())
                for cell in row
            ]
            # Pad to max columns
            while len(cells) < max_cols:
                cells.append("")
            lines.append("| " + " | ".join(cells[:max_cols]) + " |")
        
        return "\n".join(lines)
    
    def _escape_markdown(self, text: str) -> str:
        """Escape special Markdown characters in table cells."""
        # Replace pipe character (breaks table)
        text = text.replace("|", "\\|")
        # Replace newlines (breaks table row)
        text = text.replace("\n", " ")
        return text
    
    def _to_html(self, data: List[List]) -> str:
        """
        Convert table data to HTML format.
        
        Preserves structural fidelity for LLM consumption.
        Uses semantic HTML (thead, tbody) for clarity.
        
        Args:
            data: List of rows, each row is list of cell values
            
        Returns:
            HTML table string
        """
        if not data:
            return ""
        
        # Normalize column count
        max_cols = max(len(row) for row in data)
        
        html_parts = ['<table>']
        
        # Header row (first row)
        html_parts.append('<thead><tr>')
        for cell in data[0]:
            clean_cell = html.escape(str(cell or "").strip())
            html_parts.append(f'<th>{clean_cell}</th>')
        # Pad empty headers
        for _ in range(max_cols - len(data[0])):
            html_parts.append('<th></th>')
        html_parts.append('</tr></thead>')
        
        # Body rows
        html_parts.append('<tbody>')
        for row in data[1:]:
            html_parts.append('<tr>')
            for cell in row:
                clean_cell = html.escape(str(cell or "").strip())
                html_parts.append(f'<td>{clean_cell}</td>')
            # Pad empty cells
            for _ in range(max_cols - len(row)):
                html_parts.append('<td></td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        
        return ''.join(html_parts)
    
    def _to_raw_text(self, data: List[List]) -> str:
        """
        Convert table data to plain text for search.
        """
        texts = []
        for row in data:
            row_text = " ".join(str(cell or "").strip() for cell in row if cell)
            if row_text:
                texts.append(row_text)
        return " | ".join(texts)
    
    def _classify_table(self, markdown: str, raw_text: str) -> str:
        """
        Classify table by content.
        
        Args:
            markdown: Markdown representation
            raw_text: Plain text representation
            
        Returns:
            Table type string
        """
        combined = f"{markdown} {raw_text}".lower()
        
        scores: Dict[str, int] = {}
        
        for table_type, patterns in self._compiled_patterns.items():
            score = sum(1 for p in patterns if p.search(combined))
            if score > 0:
                scores[table_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return "general"
    
    def _find_title(
        self,
        page: "fitz.Page",
        table_bbox: Tuple[float, float, float, float]
    ) -> Optional[str]:
        """
        Find title text above the table.
        """
        x0, y0, x1, y1 = table_bbox
        
        # Search area above table
        search_rect = fitz.Rect(
            max(0, x0 - 20),
            max(0, y0 - self.config.title_search_height),
            x1 + 20,
            y0
        )
        
        try:
            blocks = page.get_text("dict", clip=search_rect)["blocks"]
        except (TypeError, RuntimeError, IndexError, KeyError) as e:
            logger.debug(f"Title search failed in rect {search_rect}: {e}")
            return None
        
        # Check blocks from bottom to top
        candidates = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
            
            text = text.strip()
            if text:
                y_pos = block["bbox"][3]  # Bottom y
                candidates.append((y_pos, text))
        
        # Sort by y position (bottom first = closest to table)
        candidates.sort(key=lambda x: -x[0])
        
        for _, text in candidates:
            # Check for Table pattern
            if re.match(r"^(Table|Tbl\.?)\s+\d+", text, re.IGNORECASE):
                return text
            
            # Short title-like text (not ending with period)
            if 5 < len(text) < 150 and not text.rstrip().endswith('.'):
                return text
        
        return None
    
    def extract_from_document(
        self,
        doc: "fitz.Document",
        document_id: str
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a document.
        """
        all_tables = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_tables = self.extract_tables(page, page_num, document_id)
            all_tables.extend(page_tables)
        
        return all_tables
