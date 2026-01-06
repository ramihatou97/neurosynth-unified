# src/learning/nprss/generators/relation_cards.py
"""
Relation-Based Flashcard Generator

Leverages the existing NeuroRelationExtractor to generate flashcards
from anatomical relationships already being extracted.

This provides ~90% code reuse - we just format existing extractions
as learning materials.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import uuid4
from enum import Enum

from ..models import LearningCard, CardType

logger = logging.getLogger(__name__)


# =============================================================================
# RELATION TYPES (Mirrors NeuroRelationExtractor.RelationType)
# =============================================================================

class RelationType(str, Enum):
    """Anatomical relation types - matches existing extractor"""
    SUPPLIES = "supplies"
    DRAINS = "drains"
    INNERVATES = "innervates"
    PASSES_THROUGH = "passes_through"
    TRAVERSES = "traverses"
    ORIGINATES_FROM = "originates_from"
    INSERTS_INTO = "inserts_into"
    BRANCHES_INTO = "branches_into"
    ANASTOMOSES_WITH = "anastomoses_with"
    ADJACENT_TO = "adjacent_to"
    CONTAINS = "contains"
    PART_OF = "part_of"
    CONNECTS = "connects"
    DIVIDES_INTO = "divides_into"
    RECEIVES_FROM = "receives_from"


# =============================================================================
# FLASHCARD TEMPLATES BY RELATION TYPE
# =============================================================================

RELATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    RelationType.SUPPLIES: {
        'forward': "What structures does the {source} supply blood to?",
        'reverse': "What is the arterial supply to the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'vascular', 'arterial_supply'],
        'difficulty': 'medium'
    },
    RelationType.DRAINS: {
        'forward': "What does the {source} drain?",
        'reverse': "What is the venous drainage of the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'vascular', 'venous_drainage'],
        'difficulty': 'medium'
    },
    RelationType.INNERVATES: {
        'forward': "What does the {source} innervate?",
        'reverse': "What nerve innervates the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'neuroanatomy', 'innervation'],
        'difficulty': 'medium'
    },
    RelationType.PASSES_THROUGH: {
        'forward': "What foramen/structure does the {source} pass through?",
        'reverse': "What passes through the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'foramina', 'spatial'],
        'difficulty': 'hard'
    },
    RelationType.TRAVERSES: {
        'forward': "What structure does the {source} traverse?",
        'reverse': "What traverses the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'spatial', 'relationships'],
        'difficulty': 'medium'
    },
    RelationType.ORIGINATES_FROM: {
        'forward': "Where does the {source} originate?",
        'reverse': "What originates from the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'origin', 'muscles'],
        'difficulty': 'medium'
    },
    RelationType.INSERTS_INTO: {
        'forward': "Where does the {source} insert?",
        'reverse': "What inserts into the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'insertion', 'muscles'],
        'difficulty': 'medium'
    },
    RelationType.BRANCHES_INTO: {
        'forward': "What are the branches of the {source}?",
        'reverse': "What does the {target} branch from?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'vascular', 'branches'],
        'difficulty': 'hard'
    },
    RelationType.ANASTOMOSES_WITH: {
        'forward': "What does the {source} anastomose with?",
        'reverse': "What anastomoses with the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'vascular', 'anastomosis'],
        'difficulty': 'hard'
    },
    RelationType.ADJACENT_TO: {
        'forward': "What is adjacent to the {source}?",
        'reverse': "What is adjacent to the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'spatial', 'relationships'],
        'difficulty': 'easy'
    },
    RelationType.CONTAINS: {
        'forward': "What does the {source} contain?",
        'reverse': "What contains the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'spatial', 'contents'],
        'difficulty': 'medium'
    },
    RelationType.PART_OF: {
        'forward': "What is the {source} part of?",
        'reverse': "What are the parts of the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'hierarchy', 'structure'],
        'difficulty': 'easy'
    },
    RelationType.CONNECTS: {
        'forward': "What does the {source} connect to?",
        'reverse': "What connects to the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'connections', 'pathways'],
        'difficulty': 'medium'
    },
    RelationType.DIVIDES_INTO: {
        'forward': "What does the {source} divide into?",
        'reverse': "What forms by division of the {target}?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'divisions', 'branches'],
        'difficulty': 'hard'
    },
    RelationType.RECEIVES_FROM: {
        'forward': "What does the {source} receive from?",
        'reverse': "What does the {target} give to?",
        'answer_forward': "{target}",
        'answer_reverse': "{source}",
        'tags': ['anatomy', 'connections', 'tributaries'],
        'difficulty': 'medium'
    },
}


# =============================================================================
# RELATION FLASHCARD GENERATOR
# =============================================================================

@dataclass
class ExtractedRelation:
    """
    Represents an extracted anatomical relation.
    Mirrors the output of NeuroRelationExtractor.
    """
    source: str
    target: str
    relation: str
    confidence: float
    context_snippet: str = ""
    source_sentence: str = ""


class RelationFlashcardGenerator:
    """
    Generate flashcards from extracted anatomical relations.

    Leverages NeuroRelationExtractor's output to create bidirectional
    flashcards for each relationship.

    Usage:
        # With existing extractor
        from src.core.relation_extractor import NeuroRelationExtractor

        extractor = NeuroRelationExtractor()
        generator = RelationFlashcardGenerator()

        relations = extractor.extract_from_text(chunk_content)
        flashcards = generator.generate_from_relations(relations, chunk_metadata)

        # Or directly from chunk
        flashcards = generator.generate_from_chunk(chunk_dict)
    """

    def __init__(
        self,
        relation_extractor=None,
        min_confidence: float = 0.7,
        generate_reverse: bool = True
    ):
        """
        Initialize generator.

        Args:
            relation_extractor: Optional NeuroRelationExtractor instance.
                               If None, will import lazily.
            min_confidence: Minimum confidence for flashcard generation
            generate_reverse: Whether to generate reverse cards
        """
        self._extractor = relation_extractor
        self.min_confidence = min_confidence
        self.generate_reverse = generate_reverse

    @property
    def extractor(self):
        """Lazy load extractor to avoid circular imports"""
        if self._extractor is None:
            try:
                from src.core.relation_extractor import NeuroRelationExtractor
                self._extractor = NeuroRelationExtractor()
            except ImportError:
                # Fallback for testing without full NeuroSynth
                self._extractor = None
        return self._extractor

    def generate_from_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate flashcards from a chunk dict.

        Args:
            chunk: Dict with 'content', 'id', 'document_id', 'page_number', etc.

        Returns:
            List of LearningCard objects
        """
        if not self.extractor:
            return []

        # Extract relations from content
        content = chunk.get('content', '')
        if not content:
            return []

        try:
            raw_relations = self.extractor.extract_from_text(content)
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            return []

        # Convert to our ExtractedRelation format if needed
        relations = []
        for rel in raw_relations:
            if hasattr(rel, 'source'):
                relations.append(ExtractedRelation(
                    source=rel.source,
                    target=rel.target,
                    relation=rel.relation if hasattr(rel, 'relation') else str(rel.relation_type),
                    confidence=getattr(rel, 'confidence', 0.8),
                    context_snippet=getattr(rel, 'context_snippet', ''),
                    source_sentence=getattr(rel, 'source_sentence', '')
                ))
            elif isinstance(rel, dict):
                relations.append(ExtractedRelation(**rel))

        return self.generate_from_relations(relations, chunk)

    def generate_from_relations(
        self,
        relations: List[ExtractedRelation],
        chunk_metadata: Dict[str, Any]
    ) -> List[LearningCard]:
        """
        Generate flashcards from extracted relations.

        Args:
            relations: List of ExtractedRelation objects
            chunk_metadata: Chunk metadata for source attribution

        Returns:
            List of LearningCard objects
        """
        flashcards = []
        seen_pairs = set()  # Avoid duplicates

        for rel in relations:
            # Skip low confidence
            if rel.confidence < self.min_confidence:
                continue

            # Normalize relation type
            relation_type = rel.relation.lower().replace('-', '_').replace(' ', '_')

            # Skip if no template
            if relation_type not in RELATION_TEMPLATES and \
               not any(relation_type == rt.value for rt in RelationType):
                continue

            # Get template
            template = None
            for rt in RelationType:
                if rt.value == relation_type:
                    template = RELATION_TEMPLATES.get(rt)
                    break

            if not template:
                continue

            # Deduplicate
            pair_key = (rel.source.lower(), rel.target.lower(), relation_type)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Generate forward card
            forward_card = self._create_card(
                template=template,
                relation=rel,
                direction='forward',
                chunk_metadata=chunk_metadata
            )
            flashcards.append(forward_card)

            # Generate reverse card
            if self.generate_reverse:
                reverse_card = self._create_card(
                    template=template,
                    relation=rel,
                    direction='reverse',
                    chunk_metadata=chunk_metadata
                )
                flashcards.append(reverse_card)

        return flashcards

    def _create_card(
        self,
        template: Dict[str, Any],
        relation: ExtractedRelation,
        direction: str,
        chunk_metadata: Dict[str, Any]
    ) -> LearningCard:
        """Create a single flashcard from template and relation."""
        # Format question and answer
        prompt = template[direction].format(
            source=relation.source,
            target=relation.target
        )
        answer = template[f'answer_{direction}'].format(
            source=relation.source,
            target=relation.target
        )

        # Build tags
        tags = list(template.get('tags', []))
        if direction == 'reverse':
            tags.append('reverse')

        # Add relation type to tags
        tags.append(f"relation:{relation.relation}")

        # Determine difficulty
        difficulty = self._estimate_difficulty(relation, template)

        # Create card
        return LearningCard(
            procedure_id=chunk_metadata.get('procedure_id'),
            element_id=chunk_metadata.get('element_id'),
            card_type=CardType.MCQ,  # Relation cards are factual Q&A
            prompt=prompt,
            answer=answer,
            explanation=f"Context: {relation.context_snippet}" if relation.context_snippet else None,
            difficulty_preset=difficulty,
            tags=tags,
            source_chunk_id=chunk_metadata.get('id'),
            source_document_id=chunk_metadata.get('document_id'),
            source_page=chunk_metadata.get('page_number'),
            generation_method='relation_based',
            quality_score=relation.confidence
        )

    def _estimate_difficulty(
        self,
        relation: ExtractedRelation,
        template: Dict[str, Any]
    ) -> float:
        """
        Estimate difficulty based on entity complexity.

        Returns:
            Float 0-1 (0=easy, 1=hard)
        """
        base_difficulty = {
            'easy': 0.3,
            'medium': 0.5,
            'hard': 0.7
        }.get(template.get('difficulty', 'medium'), 0.5)

        # Adjust based on entity name length (longer = harder)
        avg_length = (len(relation.source) + len(relation.target)) / 2
        length_modifier = min(0.2, avg_length / 100)

        # Adjust based on confidence (lower confidence = harder)
        confidence_modifier = (1 - relation.confidence) * 0.1

        return min(1.0, base_difficulty + length_modifier + confidence_modifier)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_relation_cards(
    content: str,
    chunk_id: str = None,
    document_id: str = None,
    page_number: int = None
) -> List[LearningCard]:
    """
    Convenience function to generate relation-based flashcards.

    Args:
        content: Text content to extract relations from
        chunk_id: Optional chunk ID for attribution
        document_id: Optional document ID
        page_number: Optional page number

    Returns:
        List of LearningCard objects
    """
    generator = RelationFlashcardGenerator()

    chunk = {
        'content': content,
        'id': chunk_id or str(uuid4()),
        'document_id': document_id,
        'page_number': page_number
    }

    return generator.generate_from_chunk(chunk)
