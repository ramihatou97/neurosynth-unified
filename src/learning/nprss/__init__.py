# src/learning/nprss/__init__.py
"""
NPRSS - Neurosurgical Procedural Retrieval-based Spaced System

A procedural learning system that transforms neurosurgical procedures
into optimized learning materials using cognitive science research.

Core Components:
- models: Data models for procedures, CSPs, cards
- fsrs: Free Spaced Repetition Scheduler algorithm
- transformers: Phase Mapper, CSP Extractor
- generators: Card Generator, Surgical Card Generator
- service: Learning Enrichment Service

Usage:
    from src.learning.nprss import (
        LearningEnrichmentService,
        ProcedureBridge,
        HybridScheduler,
        FSRS
    )

    # Enrich a procedure
    service = LearningEnrichmentService(db, llm_client)
    result = await service.enrich_procedure(
        procedure_id="...",
        generate_surgical_card=True,
        generate_fsrs_cards=True
    )

    # Create learning schedule
    scheduler = HybridScheduler()
    schedule = scheduler.create_hybrid_schedule(
        procedure_id="...",
        user_id="...",
        target_retention_days=180
    )
"""

from .models import (
    # Enums
    ACGMESubspecialty,
    Complexity,
    ElementType,
    PhaseType,
    MasteryLevel,
    EntrustmentLevel,
    MillerLevel,
    CardType,
    FSRSState,

    # Core Models
    Procedure,
    ProcedureElement,
    SafeEntryZone,
    DangerZone,
    DecisionBranch,
    VisualDescription,

    # Learning Models
    PhaseGate,
    CriticalSafetyPoint,
    VisuospatialAnchor,
    SurgicalCardRow,
    SurgicalCard,
    LearningCard,
    CardMemoryState,
    RetrievalSchedule,
    RetrievalSession,
    ProcedureMastery,

    # Assessment Models
    AssessmentItem,
    EntrustmentAssessment,

    # Factory
    RetrievalScheduleFactory,
)

from .fsrs import (
    FSRS,
    FSRSParameters,
    MemoryState,
    Rating,
    State,
    ReviewLog,
    HybridScheduler,
    create_fsrs_card,
    review_card,
)

from .transformers import (
    PhaseMapper,
    CSPExtractor,
)

from .card_generators import (
    CardGenerator,
    SurgicalCardGenerator,
)

from .service import (
    LearningEnrichmentService,
    RetrievalScheduleService,
    MasteryService,
)

from .bridge import (
    ProcedureBridge,
)

__all__ = [
    # Enums
    'ACGMESubspecialty',
    'Complexity',
    'ElementType',
    'PhaseType',
    'MasteryLevel',
    'EntrustmentLevel',
    'MillerLevel',
    'CardType',
    'FSRSState',

    # Core Models
    'Procedure',
    'ProcedureElement',
    'SafeEntryZone',
    'DangerZone',
    'DecisionBranch',
    'VisualDescription',

    # Learning Models
    'PhaseGate',
    'CriticalSafetyPoint',
    'VisuospatialAnchor',
    'SurgicalCardRow',
    'SurgicalCard',
    'LearningCard',
    'CardMemoryState',
    'RetrievalSchedule',
    'RetrievalSession',
    'ProcedureMastery',

    # Assessment
    'AssessmentItem',
    'EntrustmentAssessment',

    # Factory
    'RetrievalScheduleFactory',

    # FSRS
    'FSRS',
    'FSRSParameters',
    'MemoryState',
    'Rating',
    'State',
    'ReviewLog',
    'HybridScheduler',
    'create_fsrs_card',
    'review_card',

    # Transformers
    'PhaseMapper',
    'CSPExtractor',

    # Generators
    'CardGenerator',
    'SurgicalCardGenerator',

    # Services
    'LearningEnrichmentService',
    'RetrievalScheduleService',
    'MasteryService',

    # Bridge
    'ProcedureBridge',
]
