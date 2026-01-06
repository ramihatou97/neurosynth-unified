# src/learning/nprss/generators/hybrid_generator.py
"""
Hybrid Flashcard Generator

Orchestrates all flashcard generation strategies:
1. NPRSS Procedural: CSP triggers, phase sequences, surgical cards
2. Content-Based: Relations, UMLS definitions, table MCQs

This is the main entry point for comprehensive flashcard generation.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from uuid import uuid4
from datetime import datetime

from ..models import LearningCard, CardType

# Import all generators
from .relation_cards import RelationFlashcardGenerator
from .umls_cards import UMLSDefinitionGenerator
from .table_cards import TableMCQGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HybridGeneratorSettings:
    """Configuration for hybrid generation"""

    # Content-based generation
    enable_relation_cards: bool = True
    enable_umls_cards: bool = True
    enable_table_mcqs: bool = True

    # Procedural generation (NPRSS)
    enable_csp_cards: bool = True
    enable_sequence_cards: bool = True
    enable_dictation_cards: bool = True

    # Limits
    max_cards_per_chunk: int = 20
    max_cards_per_document: int = 200

    # Quality thresholds
    min_quality_score: float = 0.6
    min_confidence: float = 0.7

    # Deduplication
    deduplicate_by_prompt: bool = True


# =============================================================================
# HYBRID GENERATOR
# =============================================================================

class HybridFlashcardGenerator:
    """
    Unified flashcard generator combining all strategies.

    Strategies:
    - **Relation-based**: Anatomical relationships (supplies, innervates, etc.)
    - **UMLS-based**: Medical terminology definitions
    - **Table-based**: MCQs from grading scales, classifications
    - **CSP-based**: Critical Safety Point trigger cards (NPRSS)
    - **Sequence-based**: Phase and step sequence cards (NPRSS)

    Usage:
        generator = HybridFlashcardGenerator()

        # Generate from a single chunk
        cards = await generator.generate_from_chunk(chunk_dict)

        # Generate from entire document
        cards = await generator.generate_from_document(
            document_id,
            chunk_repository
        )

        # Generate from synthesis result
        cards = await generator.generate_from_synthesis(
            synthesis_result,
            include_procedural=True
        )
    """

    def __init__(
        self,
        settings: Optional[HybridGeneratorSettings] = None,
        relation_extractor=None,
        umls_extractor=None,
        table_extractor=None
    ):
        """
        Initialize hybrid generator.

        Args:
            settings: Generator configuration
            relation_extractor: Optional NeuroRelationExtractor
            umls_extractor: Optional UMLSExtractor
            table_extractor: Optional TableExtractor
        """
        self.settings = settings or HybridGeneratorSettings()

        # Initialize content-based generators
        self.relation_gen = RelationFlashcardGenerator(
            relation_extractor=relation_extractor,
            min_confidence=self.settings.min_confidence
        ) if self.settings.enable_relation_cards else None

        self.umls_gen = UMLSDefinitionGenerator(
            umls_extractor=umls_extractor,
            min_confidence=self.settings.min_confidence
        ) if self.settings.enable_umls_cards else None

        self.table_gen = TableMCQGenerator(
            table_extractor=table_extractor
        ) if self.settings.enable_table_mcqs else None

        # Track generated prompts for deduplication
        self._seen_prompts: Set[str] = set()

    async def generate_from_chunk(
        self,
        chunk: Dict[str, Any],
        strategies: List[str] = None
    ) -> List[LearningCard]:
        """
        Generate flashcards from a single chunk.

        Args:
            chunk: Chunk dict with content, id, etc.
            strategies: List of strategies to use. Options:
                       ['relation', 'umls', 'table', 'csp', 'sequence']
                       None = use all enabled strategies

        Returns:
            List of LearningCard objects
        """
        if strategies is None:
            strategies = ['relation', 'umls', 'table']

        all_cards = []

        # Content-based generation
        if 'relation' in strategies and self.relation_gen:
            try:
                cards = self.relation_gen.generate_from_chunk(chunk)
                all_cards.extend(cards)
            except Exception as e:
                logger.warning(f"Relation generation failed: {e}")

        if 'umls' in strategies and self.umls_gen:
            try:
                cards = self.umls_gen.generate_from_chunk(chunk)
                all_cards.extend(cards)
            except Exception as e:
                logger.warning(f"UMLS generation failed: {e}")

        if 'table' in strategies and self.table_gen:
            try:
                cards = self.table_gen.generate_from_chunk(chunk)
                all_cards.extend(cards)
            except Exception as e:
                logger.warning(f"Table generation failed: {e}")

        # Deduplicate
        if self.settings.deduplicate_by_prompt:
            all_cards = self._deduplicate(all_cards)

        # Apply quality filter
        all_cards = [
            c for c in all_cards
            if getattr(c, 'quality_score', 1.0) >= self.settings.min_quality_score
        ]

        # Limit
        return all_cards[:self.settings.max_cards_per_chunk]

    async def generate_from_document(
        self,
        document_id: str,
        chunk_repository,
        strategies: List[str] = None,
        chunk_types: List[str] = None
    ) -> List[LearningCard]:
        """
        Generate flashcards from all chunks in a document.

        Args:
            document_id: Document UUID
            chunk_repository: Repository for fetching chunks
            strategies: Generation strategies to use
            chunk_types: Filter chunks by type (ANATOMY, PROCEDURE, etc.)

        Returns:
            List of LearningCard objects
        """
        # Fetch chunks
        filters = {'document_id': document_id}
        if chunk_types:
            filters['chunk_type'] = chunk_types

        chunks = await chunk_repository.find_by(
            filters,
            limit=self.settings.max_cards_per_document
        )

        all_cards = []

        for chunk in chunks:
            chunk_dict = chunk if isinstance(chunk, dict) else chunk.to_dict()
            cards = await self.generate_from_chunk(chunk_dict, strategies)
            all_cards.extend(cards)

        # Global deduplication across chunks
        if self.settings.deduplicate_by_prompt:
            all_cards = self._deduplicate(all_cards)

        return all_cards[:self.settings.max_cards_per_document]

    async def generate_from_synthesis(
        self,
        synthesis: Dict[str, Any],
        include_procedural: bool = True,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """
        Generate flashcards from synthesis result.

        Combines content-based and procedural generation.

        Args:
            synthesis: Synthesis result with 'sections', 'chunks', etc.
            include_procedural: Include NPRSS procedural cards
            include_content: Include content-based cards

        Returns:
            Dict with cards by category
        """
        result = {
            'content_cards': [],
            'procedural_cards': [],
            'surgical_card': None,
            'total': 0,
            'by_strategy': {}
        }

        # Content-based from synthesis chunks/sections
        if include_content:
            content_cards = []

            # Process sections
            for section in synthesis.get('sections', []):
                chunk = {
                    'content': section.get('content', ''),
                    'id': section.get('id', str(uuid4())),
                    'document_id': synthesis.get('document_id'),
                    'chunk_type': section.get('type', 'GENERAL')
                }

                cards = await self.generate_from_chunk(chunk)
                content_cards.extend(cards)

            # Process source chunks if available
            for chunk in synthesis.get('source_chunks', []):
                chunk_dict = chunk if isinstance(chunk, dict) else chunk.to_dict()
                cards = await self.generate_from_chunk(chunk_dict)
                content_cards.extend(cards)

            result['content_cards'] = self._deduplicate(content_cards)
            result['by_strategy']['content'] = len(result['content_cards'])

        result['total'] = len(result['content_cards']) + len(result['procedural_cards'])

        return result

    async def generate_interleaved_set(
        self,
        chunks: List[Dict[str, Any]],
        num_cards: int = 20,
        difficulty_distribution: Dict[str, float] = None
    ) -> List[LearningCard]:
        """
        Generate an interleaved set of cards for optimal learning.

        Mixes card types and chunk types according to research.

        Args:
            chunks: List of chunk dicts
            num_cards: Target number of cards
            difficulty_distribution: {easy: 0.3, medium: 0.5, hard: 0.2}

        Returns:
            Interleaved list of LearningCard objects
        """
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}

        all_cards = []

        # Generate from all chunks
        for chunk in chunks:
            cards = await self.generate_from_chunk(chunk)
            all_cards.extend(cards)

        # Deduplicate
        all_cards = self._deduplicate(all_cards)

        # Categorize by difficulty
        by_difficulty = {'easy': [], 'medium': [], 'hard': []}

        for card in all_cards:
            diff = getattr(card, 'difficulty_preset', 0.5)
            if diff < 0.4:
                by_difficulty['easy'].append(card)
            elif diff < 0.7:
                by_difficulty['medium'].append(card)
            else:
                by_difficulty['hard'].append(card)

        # Select according to distribution
        selected = []

        for difficulty, ratio in difficulty_distribution.items():
            count = int(num_cards * ratio)
            available = by_difficulty[difficulty]

            if len(available) >= count:
                import random
                selected.extend(random.sample(available, count))
            else:
                selected.extend(available)

        # Interleave by chunk type
        return self._interleave_by_chunk_type(selected)

    def _deduplicate(self, cards: List[LearningCard]) -> List[LearningCard]:
        """Remove duplicate cards by prompt."""
        unique = []

        for card in cards:
            prompt_key = card.prompt.lower().strip()

            if prompt_key not in self._seen_prompts:
                self._seen_prompts.add(prompt_key)
                unique.append(card)

        return unique

    def _interleave_by_chunk_type(
        self,
        cards: List[LearningCard]
    ) -> List[LearningCard]:
        """Interleave cards by chunk type for better learning."""
        from collections import defaultdict

        by_type = defaultdict(list)

        for card in cards:
            chunk_type = card.tags[0] if card.tags else 'general'
            by_type[chunk_type].append(card)

        # Round-robin interleave
        result = []
        types = list(by_type.keys())
        max_len = max(len(by_type[t]) for t in types) if types else 0

        for i in range(max_len):
            for t in types:
                if i < len(by_type[t]):
                    result.append(by_type[t][i])

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'total_unique_prompts': len(self._seen_prompts),
            'generators_enabled': {
                'relation': self.relation_gen is not None,
                'umls': self.umls_gen is not None,
                'table': self.table_gen is not None,
            },
            'settings': {
                'max_per_chunk': self.settings.max_cards_per_chunk,
                'max_per_document': self.settings.max_cards_per_document,
                'min_quality': self.settings.min_quality_score,
            }
        }

    def reset_deduplication(self):
        """Reset deduplication cache (for new generation session)."""
        self._seen_prompts.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_comprehensive_flashcards(
    content: str,
    chunk_type: str = 'GENERAL',
    document_id: str = None,
    include_all_strategies: bool = True
) -> List[LearningCard]:
    """
    Generate flashcards using all available strategies.

    Args:
        content: Text content
        chunk_type: Type of content
        document_id: Optional document reference
        include_all_strategies: Use all generation methods

    Returns:
        List of LearningCard objects
    """
    generator = HybridFlashcardGenerator()

    chunk = {
        'content': content,
        'id': str(uuid4()),
        'document_id': document_id,
        'chunk_type': chunk_type
    }

    strategies = ['relation', 'umls', 'table'] if include_all_strategies else ['relation']

    return await generator.generate_from_chunk(chunk, strategies)
