# src/learning/nprss/generators/table_cards.py
"""
Table-Based MCQ Generator

Leverages the existing TableExtractor to generate MCQs from:
- Grading scales (Hunt-Hess, Fisher, etc.)
- Classification systems (WHO tumor grades, etc.)
- Comparison tables
- Anatomical measurements

Particularly valuable for clinical content.
"""

import logging
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from ..models import LearningCard, CardType

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE TYPE CONFIGURATION
# =============================================================================

# Table types suitable for MCQ generation
MCQ_SUITABLE_TABLE_TYPES = {
    'grading_scale',      # Hunt-Hess, Fisher, WFNS, etc.
    'classification',     # WHO grades, tumor classifications
    'comparison',         # Differential diagnosis tables
    'measurements',       # Normal values, anatomical measurements
    'staging',            # Cancer staging (TNM, etc.)
    'criteria',           # Diagnostic criteria tables
}

# Question templates by table type
TABLE_QUESTION_TEMPLATES = {
    'grading_scale': [
        "{table_title}: What are the criteria for Grade {grade}?",
        "In the {table_title}, what grade corresponds to: {criteria}?",
        "{table_title}: A patient with '{criteria}' is classified as?",
    ],
    'classification': [
        "According to {table_title}, what defines {category}?",
        "In {table_title}, what category has the following features: {features}?",
    ],
    'comparison': [
        "What distinguishes {item_a} from {item_b} regarding {aspect}?",
        "Which condition is associated with {feature}?",
    ],
    'measurements': [
        "What is the normal range for {parameter}?",
        "A {parameter} of {value} indicates?",
    ],
    'staging': [
        "In {table_title}, what defines Stage {stage}?",
        "What stage corresponds to: {criteria}?",
    ],
    'criteria': [
        "What are the diagnostic criteria for {condition}?",
        "How many criteria are needed to diagnose {condition}?",
    ],
}


# =============================================================================
# EXTRACTED TABLE MODEL
# =============================================================================

@dataclass
class ExtractedTable:
    """
    Represents an extracted table.
    Mirrors output of TableExtractor.
    """
    markdown_content: str
    table_type: str
    title: str = ""
    caption: str = ""
    headers: List[str] = None
    rows: List[List[str]] = None
    page_number: int = None
    confidence: float = 0.8

    def __post_init__(self):
        if self.headers is None:
            self.headers = []
        if self.rows is None:
            self.rows = []


# =============================================================================
# TABLE MCQ GENERATOR
# =============================================================================

class TableMCQGenerator:
    """
    Generate MCQs from extracted tables.

    Particularly effective for:
    - Grading scales (Hunt-Hess, Fisher, WFNS, Spetzler-Martin)
    - Classification systems
    - Diagnostic criteria

    Usage:
        from src.ingest.table_extractor import TableExtractor

        extractor = TableExtractor()
        generator = TableMCQGenerator()

        tables = extractor.extract_from_page(page_content)
        mcqs = generator.generate_from_tables(tables, chunk_metadata)
    """

    def __init__(
        self,
        table_extractor=None,
        min_rows: int = 2,
        min_distractors: int = 2,
        max_distractors: int = 4
    ):
        """
        Initialize generator.

        Args:
            table_extractor: Optional TableExtractor instance
            min_rows: Minimum table rows for MCQ generation
            min_distractors: Minimum distractors needed for valid MCQ
            max_distractors: Maximum distractors to include
        """
        self._extractor = table_extractor
        self.min_rows = min_rows
        self.min_distractors = min_distractors
        self.max_distractors = max_distractors

    @property
    def extractor(self):
        """Lazy load extractor"""
        if self._extractor is None:
            try:
                from src.ingest.table_extractor import TableExtractor
                self._extractor = TableExtractor()
            except ImportError:
                self._extractor = None
        return self._extractor

    def generate_from_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate MCQs from tables in a chunk.

        Args:
            chunk: Dict with 'content', 'tables', etc.

        Returns:
            List of LearningCard objects
        """
        # Check for pre-extracted tables
        tables_data = chunk.get('tables', [])

        if not tables_data and self.extractor:
            # Extract tables from content
            content = chunk.get('content', '')
            if content:
                try:
                    tables_data = self.extractor.extract_from_page(content)
                except Exception as e:
                    logger.warning(f"Table extraction failed: {e}")
                    return []

        if not tables_data:
            return []

        # Convert to ExtractedTable format
        tables = []
        for t in tables_data:
            if isinstance(t, dict):
                tables.append(ExtractedTable(
                    markdown_content=t.get('markdown_content', t.get('markdown', '')),
                    table_type=t.get('table_type', 'unknown'),
                    title=t.get('title', ''),
                    caption=t.get('caption', ''),
                    headers=t.get('headers', []),
                    rows=t.get('rows', []),
                    page_number=t.get('page_number'),
                    confidence=t.get('confidence', 0.8)
                ))
            elif hasattr(t, 'markdown_content'):
                tables.append(t)

        return self.generate_from_tables(tables, chunk)

    def generate_from_tables(
        self,
        tables: List[ExtractedTable],
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate MCQs from extracted tables.

        Args:
            tables: List of ExtractedTable objects
            chunk_metadata: Chunk metadata for attribution

        Returns:
            List of LearningCard objects
        """
        all_mcqs = []

        for table in tables:
            # Skip unsuitable table types
            if table.table_type not in MCQ_SUITABLE_TABLE_TYPES:
                continue

            # Parse table if needed
            if not table.rows and table.markdown_content:
                table.headers, table.rows = self._parse_markdown_table(
                    table.markdown_content
                )

            # Skip tables with too few rows
            if len(table.rows) < self.min_rows:
                continue

            # Generate MCQs based on table type
            if table.table_type == 'grading_scale':
                mcqs = self._generate_grading_scale_mcqs(table, chunk_metadata)
            elif table.table_type in {'classification', 'staging'}:
                mcqs = self._generate_classification_mcqs(table, chunk_metadata)
            elif table.table_type == 'comparison':
                mcqs = self._generate_comparison_mcqs(table, chunk_metadata)
            elif table.table_type == 'measurements':
                mcqs = self._generate_measurement_mcqs(table, chunk_metadata)
            elif table.table_type == 'criteria':
                mcqs = self._generate_criteria_mcqs(table, chunk_metadata)
            else:
                mcqs = self._generate_generic_mcqs(table, chunk_metadata)

            all_mcqs.extend(mcqs)

        return all_mcqs

    def _generate_grading_scale_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate MCQs from grading scale tables."""
        mcqs = []

        if len(table.headers) < 2 or len(table.rows) < 2:
            return mcqs

        # Identify grade and criteria columns
        grade_col = 0
        criteria_col = 1 if len(table.headers) > 1 else 0

        # Try to identify columns by header content
        for i, header in enumerate(table.headers):
            header_lower = header.lower()
            if any(term in header_lower for term in ['grade', 'score', 'level', 'class']):
                grade_col = i
            elif any(term in header_lower for term in ['criteria', 'description', 'findings', 'features']):
                criteria_col = i

        table_title = table.title or "Grading Scale"

        for row in table.rows:
            if len(row) <= max(grade_col, criteria_col):
                continue

            grade = row[grade_col].strip()
            criteria = row[criteria_col].strip()

            if not grade or not criteria:
                continue

            # Type 1: Grade → Criteria
            distractors = [
                r[criteria_col].strip()
                for r in table.rows
                if r != row and len(r) > criteria_col and r[criteria_col].strip()
            ][:self.max_distractors]

            if len(distractors) >= self.min_distractors:
                question = f"{table_title}: What are the criteria for Grade {grade}?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=criteria,
                    distractors=distractors,
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['grading_scale', 'grade_to_criteria']
                ))

            # Type 2: Criteria → Grade
            grade_distractors = [
                r[grade_col].strip()
                for r in table.rows
                if r != row and len(r) > grade_col and r[grade_col].strip()
            ][:self.max_distractors]

            if len(grade_distractors) >= self.min_distractors:
                question = f"{table_title}: A patient with '{criteria[:80]}...' is classified as?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=f"Grade {grade}",
                    distractors=[f"Grade {g}" for g in grade_distractors],
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['grading_scale', 'criteria_to_grade']
                ))

        return mcqs

    def _generate_classification_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate MCQs from classification/staging tables."""
        mcqs = []

        if len(table.headers) < 2 or len(table.rows) < 2:
            return mcqs

        table_title = table.title or "Classification"

        for row in table.rows:
            if len(row) < 2:
                continue

            category = row[0].strip()
            features = row[1].strip() if len(row) > 1 else ""

            if not category or not features:
                continue

            # Category → Features
            distractors = [
                r[1].strip() if len(r) > 1 else ""
                for r in table.rows
                if r != row and len(r) > 1 and r[1].strip()
            ][:self.max_distractors]

            if len(distractors) >= self.min_distractors:
                question = f"According to {table_title}, what defines {category}?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=features,
                    distractors=distractors,
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['classification', 'category_to_features']
                ))

        return mcqs

    def _generate_comparison_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate MCQs from comparison tables."""
        mcqs = []

        if len(table.headers) < 3 or len(table.rows) < 2:
            return mcqs

        # Comparison tables: Feature | Condition A | Condition B | ...
        feature_col = 0
        condition_cols = list(range(1, len(table.headers)))

        for row in table.rows:
            if len(row) < 3:
                continue

            feature = row[feature_col].strip()
            if not feature:
                continue

            # For each condition, ask which condition has this feature value
            for col_idx in condition_cols:
                condition = table.headers[col_idx]
                value = row[col_idx].strip() if col_idx < len(row) else ""

                if not value or value.lower() in {'-', 'n/a', 'none', ''}:
                    continue

                # Other conditions as distractors
                distractor_conditions = [
                    table.headers[c]
                    for c in condition_cols
                    if c != col_idx
                ][:self.max_distractors]

                if len(distractor_conditions) >= self.min_distractors:
                    question = f"Which condition is associated with {feature}: '{value}'?"
                    mcqs.append(self._create_mcq(
                        question=question,
                        correct_answer=condition,
                        distractors=distractor_conditions,
                        table=table,
                        chunk_metadata=chunk_metadata,
                        tags=['comparison', 'differential']
                    ))

        return mcqs

    def _generate_measurement_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate MCQs from measurement/normal values tables."""
        mcqs = []

        if len(table.headers) < 2 or len(table.rows) < 2:
            return mcqs

        for row in table.rows:
            if len(row) < 2:
                continue

            parameter = row[0].strip()
            value = row[1].strip()

            if not parameter or not value:
                continue

            # Simple factual question about normal values
            distractors = self._generate_numeric_distractors(value)

            if len(distractors) >= self.min_distractors:
                question = f"What is the normal range/value for {parameter}?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=value,
                    distractors=distractors,
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['measurements', 'normal_values']
                ))

        return mcqs

    def _generate_criteria_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate MCQs from diagnostic criteria tables."""
        mcqs = []

        # Often criteria tables have criteria as rows
        if len(table.rows) < 3:
            return mcqs

        table_title = table.title or "Diagnostic Criteria"

        # Ask about individual criteria
        all_criteria = [
            row[0].strip() if row else ""
            for row in table.rows
            if row and row[0].strip()
        ]

        for i, criterion in enumerate(all_criteria):
            if len(criterion) < 10:  # Skip very short entries
                continue

            distractors = [c for j, c in enumerate(all_criteria) if j != i][:self.max_distractors]

            if len(distractors) >= self.min_distractors:
                question = f"Which of the following is a criterion for {table_title}?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=criterion,
                    distractors=distractors,
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['criteria', 'diagnostic']
                ))

        return mcqs

    def _generate_generic_mcqs(
        self,
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """Generate generic MCQs from any table."""
        mcqs = []

        if len(table.headers) < 2 or len(table.rows) < 2:
            return mcqs

        for row in table.rows:
            if len(row) < 2:
                continue

            key = row[0].strip()
            value = row[1].strip()

            if not key or not value:
                continue

            distractors = [
                r[1].strip()
                for r in table.rows
                if r != row and len(r) > 1 and r[1].strip()
            ][:self.max_distractors]

            if len(distractors) >= self.min_distractors:
                question = f"What is associated with '{key}'?"
                mcqs.append(self._create_mcq(
                    question=question,
                    correct_answer=value,
                    distractors=distractors,
                    table=table,
                    chunk_metadata=chunk_metadata,
                    tags=['table_fact']
                ))

        return mcqs

    def _create_mcq(
        self,
        question: str,
        correct_answer: str,
        distractors: List[str],
        table: ExtractedTable,
        chunk_metadata: Dict[str, Any],
        tags: List[str] = None
    ) -> LearningCard:
        """Create an MCQ LearningCard."""
        all_tags = ['mcq', 'table_based']
        if table.table_type:
            all_tags.append(table.table_type)
        if tags:
            all_tags.extend(tags)

        return LearningCard(
            procedure_id=chunk_metadata.get('procedure_id'),
            element_id=chunk_metadata.get('element_id'),
            card_type=CardType.MCQ,
            prompt=question,
            answer=correct_answer,
            options=self._shuffle_options(correct_answer, distractors),
            explanation=f"From: {table.title}" if table.title else None,
            difficulty_preset=self._estimate_difficulty(table, distractors),
            tags=all_tags,
            source_chunk_id=chunk_metadata.get('id'),
            source_document_id=chunk_metadata.get('document_id'),
            source_page=chunk_metadata.get('page_number') or table.page_number,
            generation_method='table_based',
            quality_score=table.confidence
        )

    def _parse_markdown_table(self, markdown: str) -> Tuple[List[str], List[List[str]]]:
        """Parse markdown table into headers and rows."""
        lines = markdown.strip().split('\n')
        headers = []
        rows = []

        for i, line in enumerate(lines):
            if '---' in line or '===' in line:
                continue

            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # Remove empty

            if not cells:
                continue

            if i == 0 or not headers:
                headers = cells
            else:
                rows.append(cells)

        return headers, rows

    def _shuffle_options(
        self,
        correct: str,
        distractors: List[str]
    ) -> List[Dict[str, Any]]:
        """Shuffle correct answer with distractors."""
        options = [{'text': correct, 'is_correct': True}]
        options.extend([{'text': d, 'is_correct': False} for d in distractors])
        random.shuffle(options)
        return options

    def _generate_numeric_distractors(self, value: str) -> List[str]:
        """Generate plausible numeric distractors."""
        # Extract numeric part
        numbers = re.findall(r'[\d.]+', value)
        if not numbers:
            return []

        try:
            num = float(numbers[0])

            # Generate variations
            variations = [
                num * 0.5,
                num * 0.75,
                num * 1.25,
                num * 1.5,
                num * 2,
            ]

            # Format like original
            unit_match = re.search(r'[a-zA-Z%/]+', value)
            unit = unit_match.group() if unit_match else ''

            distractors = []
            for v in variations:
                if v != num:
                    if '.' in value:
                        distractors.append(f"{v:.1f}{unit}")
                    else:
                        distractors.append(f"{int(v)}{unit}")

            return distractors[:4]

        except (ValueError, IndexError):
            return []

    def _estimate_difficulty(
        self,
        table: ExtractedTable,
        distractors: List[str]
    ) -> float:
        """Estimate MCQ difficulty."""
        difficulty = 0.4  # Base

        # More rows = harder (more to remember)
        if len(table.rows) > 5:
            difficulty += 0.1
        if len(table.rows) > 10:
            difficulty += 0.1

        # More similar distractors = harder
        if len(distractors) >= 4:
            difficulty += 0.1

        # Longer answers = harder
        avg_len = sum(len(d) for d in distractors) / len(distractors) if distractors else 0
        if avg_len > 50:
            difficulty += 0.1

        return min(1.0, difficulty)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_table_mcqs(
    markdown_table: str,
    table_type: str = 'grading_scale',
    title: str = "",
    chunk_id: str = None,
    document_id: str = None
) -> List[LearningCard]:
    """
    Convenience function to generate MCQs from a markdown table.

    Args:
        markdown_table: Markdown-formatted table
        table_type: Type of table (grading_scale, classification, etc.)
        title: Table title
        chunk_id: Optional chunk ID
        document_id: Optional document ID

    Returns:
        List of LearningCard objects
    """
    generator = TableMCQGenerator()

    table = ExtractedTable(
        markdown_content=markdown_table,
        table_type=table_type,
        title=title
    )

    chunk = {
        'id': chunk_id or str(uuid4()),
        'document_id': document_id
    }

    return generator.generate_from_tables([table], chunk)
