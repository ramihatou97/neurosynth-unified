# src/learning/nprss/generators/__init__.py
"""
NPRSS Learning Generators Package

Exports all 7 flashcard and learning content generators:

1. **NPRSS Procedural Generators** (from parent module):
   - CardGenerator: FSRS flashcards from procedures
   - SurgicalCardGenerator: One-page procedure summaries

2. **Content-Based Generators** (from this package):
   - RelationFlashcardGenerator: Uses NeuroRelationExtractor
   - UMLSDefinitionGenerator: Uses UMLSExtractor
   - TableMCQGenerator: Uses TableExtractor
   - PhaseCardGenerator: Individual surgical phase cards
   - HybridFlashcardGenerator: Orchestrates all strategies

3. **Socratic Mode** (from socratic subpackage):
   - SocraticEngine: Guided learning through questioning
"""

# Content-based generators
from .relation_cards import (
    RelationFlashcardGenerator,
    RelationType,
    RELATION_TEMPLATES,
    generate_relation_cards
)

from .umls_cards import (
    UMLSDefinitionGenerator,
    VALUABLE_SEMANTIC_TYPES,
    generate_umls_cards
)

from .table_cards import (
    TableMCQGenerator,
    MCQ_SUITABLE_TABLE_TYPES,
    generate_table_mcqs
)

from .phase_cards import (
    PhaseCardGenerator,
    PHASE_CARD_TEMPLATES,
    generate_phase_cards
)

from .hybrid_generator import (
    HybridFlashcardGenerator,
    HybridGeneratorSettings,
    generate_comprehensive_flashcards
)

__all__ = [
    # Relation-based
    'RelationFlashcardGenerator',
    'RelationType',
    'RELATION_TEMPLATES',
    'generate_relation_cards',

    # UMLS-based
    'UMLSDefinitionGenerator',
    'VALUABLE_SEMANTIC_TYPES',
    'generate_umls_cards',

    # Table-based
    'TableMCQGenerator',
    'MCQ_SUITABLE_TABLE_TYPES',
    'generate_table_mcqs',

    # Phase-based
    'PhaseCardGenerator',
    'PHASE_CARD_TEMPLATES',
    'generate_phase_cards',

    # Hybrid
    'HybridFlashcardGenerator',
    'HybridGeneratorSettings',
    'generate_comprehensive_flashcards',
]
