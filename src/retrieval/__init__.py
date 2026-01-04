"""
NeuroSynth Unified - Retrieval Layer
=====================================

Semantic search and retrieval components.

Components:
- search_service.py: Unified search interface with pgvector HNSW (primary)
- faiss_manager.py: FAISS index management (deprecated, retained for rollback)
- reranker.py: Result re-ranking

Architecture (pgvector HNSW - Default):
    Query → Embed → pgvector HNSW → Results (with filters)
                          ↓
                   Re-rank (optional) → Top K

Architecture (FAISS - Legacy, disabled by default):
    Query → Embed → FAISS (fast ANN) → Candidates
                          ↓
                   pgvector (filter) → Results
                          ↓
                   Re-rank (optional) → Top K

Quick Start:
    from src.retrieval import SearchService, VoyageEmbedder, PostgresVectorSearcher

    # Initialize with pgvector (recommended)
    embedder = VoyageEmbedder(api_key="...")
    service = SearchService(db, embedder=embedder, use_pgvector=True)

    # Search
    results = await service.search(
        query="retrosigmoid approach",
        mode="hybrid",
        top_k=10
    )

    for r in results.results:
        print(f"{r.final_score:.3f} - {r.content[:100]}...")
"""

# FAISS Manager
from src.retrieval.faiss_manager import (
    FAISSManager,
    FAISSIndex,
    FAISSIndexConfig,
    TEXT_CONFIG,
    IMAGE_CONFIG,
    CAPTION_CONFIG
)

# Search Service
from src.retrieval.search_service import (
    SearchService,
    SearchFilters,
    SearchResult,
    SearchResponse,
    SearchMode,
    VoyageEmbedder,
    PostgresVectorSearcher,
    EMBEDDING_DIMENSIONS
)

# Rerankers
from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    EnsembleReranker,
    MedicalReranker,
    DiversityReranker,
    QualityReranker,
    PipelineReranker,
    create_reranker,
    create_pipeline_reranker
)

# Query Expansion
from src.retrieval.query_expansion import (
    QueryExpander,
    ExpandedQuery,
    ExpansionMethod,
    expand_query,
    create_query_expander,
    NEURO_ABBREVIATIONS,
    CONCEPT_CUIS
)

__all__ = [
    # FAISS (deprecated, retained for rollback)
    'FAISSManager',
    'FAISSIndex',
    'FAISSIndexConfig',
    'TEXT_CONFIG',
    'IMAGE_CONFIG',
    'CAPTION_CONFIG',

    # Search (pgvector HNSW - primary)
    'SearchService',
    'SearchFilters',
    'SearchResult',
    'SearchResponse',
    'SearchMode',
    'VoyageEmbedder',
    'PostgresVectorSearcher',
    'EMBEDDING_DIMENSIONS',

    # Rerankers
    'BaseReranker',
    'CrossEncoderReranker',
    'LLMReranker',
    'EnsembleReranker',
    'MedicalReranker',
    'DiversityReranker',
    'QualityReranker',
    'PipelineReranker',
    'create_reranker',
    'create_pipeline_reranker',

    # Query Expansion
    'QueryExpander',
    'ExpandedQuery',
    'ExpansionMethod',
    'expand_query',
    'create_query_expander',
    'NEURO_ABBREVIATIONS',
    'CONCEPT_CUIS'
]
