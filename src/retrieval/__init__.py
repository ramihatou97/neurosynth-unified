"""
NeuroSynth Unified - Retrieval Layer
=====================================

Semantic search and retrieval components.

Components:
- faiss_manager.py: FAISS index management
- search_service.py: Unified search interface
- reranker.py: Result re-ranking

Architecture:
    Query → Embed → FAISS (fast ANN) → Candidates
                          ↓
                   pgvector (filter) → Results
                          ↓
                   Re-rank (optional) → Top K

Quick Start:
    from src.retrieval import SearchService, FAISSManager, VoyageEmbedder
    
    # Initialize components
    faiss = FAISSManager("./indexes")
    faiss.load()  # Load pre-built indexes
    
    embedder = VoyageEmbedder(api_key="...")
    service = SearchService(db, faiss, embedder)
    
    # Search
    results = await service.search(
        query="retrosigmoid approach",
        mode="hybrid",
        top_k=10
    )
    
    for r in results.results:
        print(f"{r.score:.3f} - {r.content[:100]}...")
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
    VoyageEmbedder
)

# Rerankers
from src.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    EnsembleReranker,
    MedicalReranker,
    create_reranker
)

__all__ = [
    # FAISS
    'FAISSManager',
    'FAISSIndex',
    'FAISSIndexConfig',
    'TEXT_CONFIG',
    'IMAGE_CONFIG', 
    'CAPTION_CONFIG',
    
    # Search
    'SearchService',
    'SearchFilters',
    'SearchResult',
    'SearchResponse',
    'SearchMode',
    'VoyageEmbedder',
    
    # Rerankers
    'BaseReranker',
    'CrossEncoderReranker',
    'LLMReranker',
    'EnsembleReranker',
    'MedicalReranker',
    'create_reranker'
]
