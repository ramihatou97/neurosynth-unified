# src/learning/nprss/generators/umls_cards.py
"""
UMLS-Based Definition Flashcard Generator

Leverages the existing UMLSExtractor to generate definition flashcards
from medical terminology already being extracted.

Uses UMLS CUIs for concept-level mastery tracking.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from uuid import uuid4

from ..models import LearningCard, CardType

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC TYPE CONFIGURATION
# =============================================================================

# Semantic types worth generating flashcards for
VALUABLE_SEMANTIC_TYPES = {
    # Anatomy
    'T023': 'Body Part, Organ, or Organ Component',
    'T024': 'Tissue',
    'T029': 'Body Location or Region',
    'T030': 'Body Space or Junction',

    # Neuroanatomy specific
    'T061': 'Therapeutic or Preventive Procedure',
    'T060': 'Diagnostic Procedure',

    # Pathology
    'T047': 'Disease or Syndrome',
    'T191': 'Neoplastic Process',
    'T046': 'Pathologic Function',

    # Clinical
    'T184': 'Sign or Symptom',
    'T033': 'Finding',
    'T034': 'Laboratory or Test Result',

    # Substances
    'T121': 'Pharmacologic Substance',
    'T109': 'Organic Chemical',
    'T116': 'Amino Acid, Peptide, or Protein',
}

# Terms to skip (too common/generic)
SKIP_TERMS = {
    'patient', 'patients', 'surgery', 'surgical', 'treatment', 'procedure',
    'diagnosis', 'prognosis', 'therapy', 'clinical', 'medical', 'study',
    'result', 'results', 'finding', 'findings', 'disease', 'condition',
    'symptom', 'symptoms', 'test', 'tests', 'blood', 'tissue', 'cell',
    'cells', 'nerve', 'artery', 'vein', 'muscle', 'bone', 'brain',
}

# Question templates by semantic type category
QUESTION_TEMPLATES = {
    'anatomy': [
        "Define: {term}",
        "What is the {term}?",
        "Describe the anatomical significance of the {term}."
    ],
    'pathology': [
        "Define: {term}",
        "What is {term}?",
        "Describe the pathological process of {term}."
    ],
    'procedure': [
        "What is {term}?",
        "Describe the procedure: {term}",
    ],
    'clinical': [
        "Define: {term}",
        "What is the clinical significance of {term}?",
    ],
    'pharmacology': [
        "What is {term}?",
        "What is the mechanism/use of {term}?",
    ],
    'default': [
        "Define: {term}",
        "What is {term}?",
    ]
}


# =============================================================================
# UMLS ENTITY MODEL
# =============================================================================

@dataclass
class UMLSEntity:
    """
    Represents an extracted UMLS entity.
    Mirrors output of UMLSExtractor.
    """
    text: str
    cui: str
    preferred_name: str
    semantic_type: str
    semantic_type_id: str = ""
    confidence: float = 0.8
    context: str = ""
    definition: str = ""
    start_char: int = 0
    end_char: int = 0


# =============================================================================
# UMLS DEFINITION GENERATOR
# =============================================================================

class UMLSDefinitionGenerator:
    """
    Generate definition flashcards from UMLS entities.

    Leverages UMLSExtractor's output to create terminology flashcards
    with CUI-based tracking for concept mastery.

    Usage:
        # With existing extractor
        from src.core.umls_extractor import UMLSExtractor

        extractor = UMLSExtractor()
        generator = UMLSDefinitionGenerator()

        entities = extractor.extract(chunk_content)
        flashcards = generator.generate_from_entities(entities, chunk_metadata)

        # Or directly from chunk
        flashcards = generator.generate_from_chunk(chunk_dict)
    """

    def __init__(
        self,
        umls_extractor=None,
        min_confidence: float = 0.75,
        include_semantic_types: Set[str] = None,
        max_cards_per_chunk: int = 10
    ):
        """
        Initialize generator.

        Args:
            umls_extractor: Optional UMLSExtractor instance
            min_confidence: Minimum confidence threshold
            include_semantic_types: Semantic type IDs to include (None = all valuable)
            max_cards_per_chunk: Maximum cards to generate per chunk
        """
        self._extractor = umls_extractor
        self.min_confidence = min_confidence
        self.include_semantic_types = include_semantic_types or set(VALUABLE_SEMANTIC_TYPES.keys())
        self.max_cards_per_chunk = max_cards_per_chunk

    @property
    def extractor(self):
        """Lazy load extractor"""
        if self._extractor is None:
            try:
                from src.core.umls_extractor import UMLSExtractor
                self._extractor = UMLSExtractor()
            except ImportError:
                self._extractor = None
        return self._extractor

    def generate_from_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate flashcards from a chunk dict.

        Args:
            chunk: Dict with 'content', 'id', 'document_id', etc.

        Returns:
            List of LearningCard objects
        """
        if not self.extractor:
            return []

        content = chunk.get('content', '')
        if not content:
            return []

        try:
            raw_entities = self.extractor.extract(content)
        except Exception as e:
            logger.warning(f"UMLS extraction failed: {e}")
            return []

        # Convert to UMLSEntity format
        entities = []
        for ent in raw_entities:
            if hasattr(ent, 'cui'):
                entities.append(UMLSEntity(
                    text=ent.text if hasattr(ent, 'text') else str(ent),
                    cui=ent.cui,
                    preferred_name=getattr(ent, 'preferred_name', ent.text if hasattr(ent, 'text') else ''),
                    semantic_type=getattr(ent, 'semantic_type', ''),
                    semantic_type_id=getattr(ent, 'semantic_type_id', ''),
                    confidence=getattr(ent, 'confidence', 0.8),
                    context=getattr(ent, 'context', ''),
                    definition=getattr(ent, 'definition', '')
                ))
            elif isinstance(ent, dict):
                entities.append(UMLSEntity(**ent))

        return self.generate_from_entities(entities, chunk)

    def generate_from_entities(
        self,
        entities: List[UMLSEntity],
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate flashcards from UMLS entities.

        Args:
            entities: List of UMLSEntity objects
            chunk_metadata: Chunk metadata for attribution

        Returns:
            List of LearningCard objects
        """
        flashcards = []
        seen_cuis = set()  # Deduplicate by CUI

        for entity in entities:
            # Skip low confidence
            if entity.confidence < self.min_confidence:
                continue

            # Skip common terms
            if entity.text.lower() in SKIP_TERMS:
                continue

            # Skip short terms (likely noise)
            if len(entity.text) < 3:
                continue

            # Skip duplicates
            if entity.cui in seen_cuis:
                continue
            seen_cuis.add(entity.cui)

            # Filter by semantic type if specified
            if entity.semantic_type_id and entity.semantic_type_id not in self.include_semantic_types:
                continue

            # Create flashcard
            card = self._create_card(entity, chunk_metadata)
            flashcards.append(card)

            # Limit cards per chunk
            if len(flashcards) >= self.max_cards_per_chunk:
                break

        return flashcards

    def _create_card(
        self,
        entity: UMLSEntity,
        chunk_metadata: Dict[str, Any]
    ) -> LearningCard:
        """Create a definition flashcard from UMLS entity."""
        # Determine category for template selection
        category = self._get_category(entity.semantic_type_id)

        # Select question template
        templates = QUESTION_TEMPLATES.get(category, QUESTION_TEMPLATES['default'])
        question = templates[0].format(term=entity.text)

        # Build answer
        answer_parts = []

        # Preferred name if different from text
        if entity.preferred_name and entity.preferred_name.lower() != entity.text.lower():
            answer_parts.append(f"**{entity.preferred_name}**")
        else:
            answer_parts.append(f"**{entity.text}**")

        # CUI reference
        answer_parts.append(f"(CUI: {entity.cui})")

        # Semantic type
        if entity.semantic_type:
            answer_parts.append(f"\n\nSemantic Type: {entity.semantic_type}")

        # Definition if available
        if entity.definition:
            answer_parts.append(f"\n\nDefinition: {entity.definition}")

        answer = " ".join(answer_parts)

        # Build tags
        tags = ['terminology', 'umls', 'definition']
        if entity.semantic_type:
            # Normalize semantic type for tag
            st_tag = entity.semantic_type.lower().replace(' ', '_').replace(',', '')
            tags.append(f"semtype:{st_tag}")
        tags.append(category)

        # Estimate difficulty
        difficulty = self._estimate_difficulty(entity)

        return LearningCard(
            procedure_id=chunk_metadata.get('procedure_id'),
            element_id=chunk_metadata.get('element_id'),
            card_type=CardType.MCQ,
            prompt=question,
            answer=answer,
            explanation=f"Context: {entity.context}" if entity.context else None,
            difficulty_preset=difficulty,
            tags=tags,
            cuis=[entity.cui],
            source_chunk_id=chunk_metadata.get('id'),
            source_document_id=chunk_metadata.get('document_id'),
            source_page=chunk_metadata.get('page_number'),
            generation_method='umls_based',
            quality_score=entity.confidence
        )

    def _get_category(self, semantic_type_id: str) -> str:
        """Map semantic type to category."""
        if not semantic_type_id:
            return 'default'

        # Anatomy types
        if semantic_type_id in {'T023', 'T024', 'T029', 'T030'}:
            return 'anatomy'

        # Pathology types
        if semantic_type_id in {'T047', 'T191', 'T046'}:
            return 'pathology'

        # Procedure types
        if semantic_type_id in {'T061', 'T060'}:
            return 'procedure'

        # Clinical types
        if semantic_type_id in {'T184', 'T033', 'T034'}:
            return 'clinical'

        # Pharmacology types
        if semantic_type_id in {'T121', 'T109', 'T116'}:
            return 'pharmacology'

        return 'default'

    def _estimate_difficulty(self, entity: UMLSEntity) -> float:
        """
        Estimate difficulty based on term characteristics.

        Returns:
            Float 0-1 (0=easy, 1=hard)
        """
        difficulty = 0.4  # Base difficulty

        # Longer terms are harder
        if len(entity.text) > 20:
            difficulty += 0.15
        elif len(entity.text) > 30:
            difficulty += 0.25

        # Multi-word terms are harder
        word_count = len(entity.text.split())
        if word_count > 2:
            difficulty += 0.1
        if word_count > 4:
            difficulty += 0.1

        # Terms with numbers/special chars are harder
        if any(c.isdigit() for c in entity.text):
            difficulty += 0.1

        # Abbreviations (all caps) are harder
        if entity.text.isupper() and len(entity.text) <= 6:
            difficulty += 0.15

        # Lower confidence = harder (more ambiguous)
        confidence_penalty = (1 - entity.confidence) * 0.2
        difficulty += confidence_penalty

        return min(1.0, difficulty)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_umls_cards(
    content: str,
    chunk_id: str = None,
    document_id: str = None,
    page_number: int = None,
    max_cards: int = 10
) -> List[LearningCard]:
    """
    Convenience function to generate UMLS-based flashcards.

    Args:
        content: Text content to extract entities from
        chunk_id: Optional chunk ID
        document_id: Optional document ID
        page_number: Optional page number
        max_cards: Maximum cards to generate

    Returns:
        List of LearningCard objects
    """
    generator = UMLSDefinitionGenerator(max_cards_per_chunk=max_cards)

    chunk = {
        'content': content,
        'id': chunk_id or str(uuid4()),
        'document_id': document_id,
        'page_number': page_number
    }

    return generator.generate_from_chunk(chunk)
