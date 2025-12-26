"""
NeuroSynth Unified - Search Service
====================================

Unified search interface combining FAISS (speed) and pgvector (filtering).

Architecture:
    Query → Embed → FAISS (fast ANN, no filters)
                        ↓
                 Candidate IDs (top 100-200)
                        ↓
                 pgvector (filter + enrich)
                        ↓
                 Re-rank (optional)
                        ↓
                 Final Results (top K)

Usage:
    from src.retrieval.search_service import SearchService
    
    service = SearchService(db, faiss_manager, embedder)
    
    results = await service.search(
        query="retrosigmoid approach for acoustic neuroma",
        mode="hybrid",
        top_k=10,
        filters={"chunk_types": ["PROCEDURE"]}
    )
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from uuid import UUID
from enum import Enum
import time

import numpy as np

from src.retrieval.faiss_manager import FAISSManager
from src.shared.models import SearchResult, ExtractedImage, ChunkType

logger = logging.getLogger(__name__)


# =============================================================================
# Search Models
# =============================================================================

class SearchMode(Enum):
    """Search mode options."""
    TEXT = "text"           # Text chunks only
    IMAGE = "image"         # Images only
    HYBRID = "hybrid"       # Combined text + image
    SIMILAR = "similar"     # Find similar to given ID


@dataclass
class SearchFilters:
    """Search filter options."""
    document_ids: List[str] = field(default_factory=list)
    chunk_types: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)
    image_types: List[str] = field(default_factory=list)
    cuis: List[str] = field(default_factory=list)
    page_range: Tuple[int, int] = None  # (min_page, max_page)
    
    @property
    def has_filters(self) -> bool:
        return bool(
            self.document_ids or 
            self.chunk_types or 
            self.specialties or 
            self.image_types or 
            self.cuis or
            self.page_range
        )


# SearchResult class imported from src.shared.models (lines 827-843)
# This provides synthesis-compatible structure with:
#   - chunk_id, document_title, authority_score, entity_names
#   - images: List[ExtractedImage] (not Dict)
#   - semantic_score, keyword_score, final_score


@dataclass
class SearchResponse:
    """Complete search response."""
    results: List[SearchResult]
    total_candidates: int
    query: str
    mode: str
    search_time_ms: int
    
    # Search metadata
    faiss_time_ms: int = 0
    filter_time_ms: int = 0
    rerank_time_ms: int = 0
    
    # Facets (optional)
    facets: Dict = field(default_factory=dict)


# =============================================================================
# Search Service
# =============================================================================

class SearchService:
    """
    Unified search service for NeuroSynth.
    
    Combines:
    - FAISS for fast approximate nearest neighbor search
    - pgvector for filtered search and enrichment
    - Optional re-ranking for quality improvement
    
    Search Pipeline:
    1. Query embedding (Voyage-3)
    2. FAISS search (fast, unfiltered) → candidate IDs
    3. Database fetch with filters → enriched results
    4. Optional: Re-rank with cross-encoder
    5. Return top K results with linked content
    """
    
    def __init__(
        self,
        database,
        faiss_manager: FAISSManager,
        embedder,
        reranker=None,
        config: Dict = None
    ):
        """
        Initialize search service.
        
        Args:
            database: Database connection or repositories
            faiss_manager: FAISS index manager
            embedder: Text embedding service
            reranker: Optional re-ranking model
            config: Search configuration
        """
        self.db = database
        self.faiss = faiss_manager
        self.embedder = embedder
        self.reranker = reranker
        
        # Configuration
        self.config = config or {}
        self.faiss_k_multiplier = self.config.get('faiss_k_multiplier', 10)
        self.text_weight = self.config.get('text_weight', 0.7)
        self.image_weight = self.config.get('image_weight', 0.3)
        self.cui_boost = self.config.get('cui_boost', 1.2)
        self.min_similarity = self.config.get('min_similarity', 0.3)
        self.max_linked_images = self.config.get('max_linked_images', 3)
    
    # =========================================================================
    # Main Search Method
    # =========================================================================
    
    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        filters: SearchFilters = None,
        include_images: bool = True,
        include_similar: bool = False,
        rerank: bool = True
    ) -> SearchResponse:
        """
        Execute search.
        
        Args:
            query: Search query text
            mode: Search mode (text, image, hybrid)
            top_k: Number of results
            filters: Search filters
            include_images: Include linked images in results
            include_similar: Include similar chunks
            rerank: Apply re-ranking
        
        Returns:
            SearchResponse with results
        """
        start_time = time.time()
        filters = filters or SearchFilters()
        
        # Step 1: Get query embedding
        query_embedding = await self._embed_query(query)
        
        # Step 2: FAISS search for candidates
        faiss_start = time.time()
        
        if mode == "text":
            candidates = await self._search_text_faiss(query_embedding, top_k, filters)
        elif mode == "image":
            candidates = await self._search_image_faiss(query_embedding, top_k, filters)
        else:  # hybrid
            candidates = await self._search_hybrid_faiss(query_embedding, top_k, filters)
        
        faiss_time = int((time.time() - faiss_start) * 1000)
        
        # Step 3: Enrich with database (apply filters)
        filter_start = time.time()
        results = await self._enrich_results(candidates, filters, mode)
        filter_time = int((time.time() - filter_start) * 1000)
        
        total_candidates = len(candidates)
        
        # Step 4: CUI boosting
        if filters.cuis:
            results = self._apply_cui_boost(results, filters.cuis)
        
        # Step 5: Re-ranking
        rerank_time = 0
        if rerank and self.reranker and len(results) > 1:
            rerank_start = time.time()
            results = await self._rerank_results(query, results)
            rerank_time = int((time.time() - rerank_start) * 1000)
        
        # Step 6: Attach linked images
        if include_images and mode != "image":
            results = await self._attach_linked_images(results)
        
        # Step 7: Trim to top_k
        results = results[:top_k]
        
        # Build response
        total_time = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            results=results,
            total_candidates=total_candidates,
            query=query,
            mode=mode,
            search_time_ms=total_time,
            faiss_time_ms=faiss_time,
            filter_time_ms=filter_time,
            rerank_time_ms=rerank_time
        )
    
    # =========================================================================
    # FAISS Search Methods
    # =========================================================================
    
    async def _search_text_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: SearchFilters
    ) -> List[Tuple[str, float]]:
        """Text search using FAISS."""
        # Over-fetch to account for filtering
        fetch_k = top_k * self.faiss_k_multiplier if filters.has_filters else top_k * 2
        
        results = self.faiss.search_text(query_embedding, k=fetch_k)
        return results
    
    async def _search_image_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: SearchFilters
    ) -> List[Tuple[str, float]]:
        """Image search using FAISS (via caption embeddings)."""
        fetch_k = top_k * self.faiss_k_multiplier if filters.has_filters else top_k * 2
        
        # Use caption embeddings for text-to-image search
        results = self.faiss.search_by_text_for_images(query_embedding, k=fetch_k)
        return results
    
    async def _search_hybrid_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: SearchFilters
    ) -> List[Tuple[str, float]]:
        """Hybrid search combining text and image."""
        fetch_k = top_k * self.faiss_k_multiplier if filters.has_filters else top_k * 3
        
        results = self.faiss.search_hybrid(
            text_embedding=query_embedding,
            image_embedding=None,  # Use text embedding for both
            k=fetch_k,
            text_weight=self.text_weight
        )
        return results
    
    # =========================================================================
    # Database Enrichment
    # =========================================================================
    
    async def _enrich_results(
        self,
        candidates: List[Tuple[str, float]],
        filters: SearchFilters,
        mode: str
    ) -> List[SearchResult]:
        """Fetch full results from database with filtering."""
        if not candidates:
            return []
        
        # Separate chunk and image IDs
        candidate_ids = [c[0] for c in candidates]
        scores = {c[0]: c[1] for c in candidates}
        
        results = []
        
        if mode in ("text", "hybrid"):
            # Fetch chunks
            chunk_results = await self._fetch_chunks(candidate_ids, filters)
            for chunk in chunk_results:
                # FIXED: Use chunk_id (not id) and update semantic/final scores
                faiss_score = scores.get(chunk.chunk_id, 0.0)
                chunk.semantic_score = faiss_score
                chunk.final_score = faiss_score  # Will be updated by reranking/CUI boost
                results.append(chunk)
        
        if mode in ("image", "hybrid"):
            # Fetch images
            image_results = await self._fetch_images(candidate_ids, filters)
            for img in image_results:
                # FIXED: Use chunk_id for consistency (images also use SearchResult)
                faiss_score = scores.get(img.chunk_id, 0.0)
                img.semantic_score = faiss_score
                img.final_score = faiss_score
                results.append(img)
        
        # Sort by final_score (synthesis-compatible)
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    async def _fetch_chunks(
        self,
        ids: List[str],
        filters: SearchFilters
    ) -> List[SearchResult]:
        """Fetch chunks from database with filters."""
        if not ids:
            return []
        
        # Build query with filters
        conditions = ["id = ANY($1)"]
        params = [ids]
        param_idx = 2
        
        if filters.document_ids:
            conditions.append(f"document_id = ANY(${param_idx})")
            params.append([UUID(d) for d in filters.document_ids])
            param_idx += 1
        
        if filters.chunk_types:
            conditions.append(f"chunk_type = ANY(${param_idx})")
            params.append(filters.chunk_types)
            param_idx += 1
        
        if filters.specialties:
            conditions.append(f"specialty = ANY(${param_idx})")
            params.append(filters.specialties)
            param_idx += 1
        
        if filters.cuis:
            conditions.append(f"cuis && ${param_idx}")
            params.append(filters.cuis)
            param_idx += 1
        
        if filters.page_range:
            conditions.append(f"page_number >= ${param_idx} AND page_number <= ${param_idx + 1}")
            params.extend(filters.page_range)
            param_idx += 2
        
        where_clause = " AND ".join(conditions)

        # FIXED: JOIN documents table for authority_score and document_title
        # SELECT metadata for entity_names extraction
        query = f"""
            SELECT
                c.id,
                c.document_id,
                c.content,
                c.page_number,
                c.chunk_type,
                c.specialty,
                c.cuis,
                c.metadata,
                d.title AS document_title,
                COALESCE(d.authority_score, 1.0) AS authority_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
        """

        rows = await self.db.fetch(query, *params)

        results = []
        for row in rows:
            row_id = str(row['id'])
            metadata = row.get('metadata', {}) or {}

            # Parse chunk_type string to ChunkType enum
            chunk_type = ChunkType.GENERAL
            if row.get('chunk_type'):
                try:
                    chunk_type = ChunkType[row['chunk_type'].upper()]
                except (KeyError, ValueError):
                    # If conversion fails, default to GENERAL
                    pass

            # FIXED: Populate synthesis-compatible SearchResult
            results.append(SearchResult(
                chunk_id=row_id,  # FIXED: chunk_id not id
                document_id=str(row['document_id']),
                content=row['content'],
                title=metadata.get('title', ''),  # Section title from metadata
                chunk_type=chunk_type,  # FIXED: Enum not string
                page_start=row.get('page_number', 0),  # FIXED: page_start not page_number
                entity_names=metadata.get('entity_names', []),  # FIXED: From metadata
                image_ids=metadata.get('image_ids', []),
                authority_score=float(row['authority_score']),  # FIXED: From JOIN
                keyword_score=0.0,  # TODO: Implement BM25 scoring
                semantic_score=0.0,  # Will be set from FAISS scores
                final_score=0.0,  # Will be computed from weighted scores
                document_title=row.get('document_title'),  # FIXED: From JOIN
                images=[]  # Will be populated by _attach_linked_images()
            ))

        return results
    
    async def _fetch_images(
        self,
        ids: List[str],
        filters: SearchFilters
    ) -> List[SearchResult]:
        """Fetch images from database with filters."""
        if not ids:
            return []
        
        conditions = ["id = ANY($1)", "NOT is_decorative"]
        params = [ids]
        param_idx = 2
        
        if filters.document_ids:
            conditions.append(f"document_id = ANY(${param_idx})")
            params.append([UUID(d) for d in filters.document_ids])
            param_idx += 1
        
        if filters.image_types:
            conditions.append(f"image_type = ANY(${param_idx})")
            params.append(filters.image_types)
            param_idx += 1
        
        if filters.page_range:
            conditions.append(f"page_number >= ${param_idx} AND page_number <= ${param_idx + 1}")
            params.extend(filters.page_range)
            param_idx += 2
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT id, document_id, file_path, vlm_caption, 
                   page_number, image_type, cuis
            FROM images
            WHERE {where_clause}
        """
        
        rows = await self.db.fetch(query, *params)
        
        results = []
        for row in rows:
            results.append(SearchResult(
                id=str(row['id']),
                content=row.get('vlm_caption', ''),
                score=0,
                result_type="image",
                document_id=str(row['document_id']),
                page_number=row.get('page_number'),
                image_type=row.get('image_type'),
                cuis=row.get('cuis', [])
            ))
        
        return results
    
    # =========================================================================
    # Scoring & Ranking
    # =========================================================================
    
    def _apply_cui_boost(
        self,
        results: List[SearchResult],
        query_cuis: List[str]
    ) -> List[SearchResult]:
        """Boost final_score for results with matching CUIs."""
        query_cuis_set = set(query_cuis)

        for result in results:
            if result.cuis:
                overlap = len(set(result.cuis) & query_cuis_set)
                if overlap > 0:
                    boost = 1 + (self.cui_boost - 1) * min(overlap / len(query_cuis), 1.0)
                    # FIXED: Update final_score (cui_score field doesn't exist in shared model)
                    result.final_score *= boost

        # Re-sort by final_score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Re-rank results using cross-encoder."""
        if not self.reranker:
            return results
        
        # Prepare pairs
        texts = [r.content for r in results]
        
        # Get re-ranked scores
        scores = await self.reranker.score(query, texts)
        
        # Update scores with reranking results
        for result, score in zip(results, scores):
            # FIXED: Update final_score to be weighted combination
            # Reranking score should override or combine with semantic score
            result.final_score = score  # Replace with rerank score

        # Re-sort by final_score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    # =========================================================================
    # Linked Content
    # =========================================================================
    
    async def _attach_linked_images(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Attach linked images to chunk results as ExtractedImage objects.

        FIXED: Returns List[ExtractedImage] instead of List[Dict] for synthesis compatibility.
        """
        if not results:
            return results

        # Get chunk_ids (all results are chunks in synthesis context)
        chunk_ids = [UUID(r.chunk_id) for r in results]

        if not chunk_ids:
            return results

        # FIXED: Fetch more image fields for ExtractedImage construction
        query = """
            SELECT
                l.chunk_id,
                l.score AS link_score,
                i.id,
                i.document_id,
                i.file_path,
                i.page_number,
                i.caption,
                i.vlm_caption,
                i.image_type,
                i.width,
                i.height,
                i.format,
                i.content_hash
            FROM chunk_image_links l
            JOIN images i ON l.image_id = i.id
            WHERE l.chunk_id = ANY($1) AND l.score >= 0.5
            ORDER BY l.chunk_id, l.score DESC
        """

        rows = await self.db.fetch(query, chunk_ids)

        # Group images by chunk_id
        images_by_chunk = {}
        for row in rows:
            chunk_id = str(row['chunk_id'])
            if chunk_id not in images_by_chunk:
                images_by_chunk[chunk_id] = []

            # FIXED: Create ExtractedImage objects (not Dict)
            img_obj = ExtractedImage(
                id=str(row['id']),
                document_id=str(row['document_id']),
                file_path=row['file_path'],
                page_number=row.get('page_number', 0),
                width=row.get('width', 0),
                height=row.get('height', 0),
                format=row.get('format', 'JPEG'),
                content_hash=row.get('content_hash', ''),
                caption=row.get('caption', ''),
                vlm_caption=row.get('vlm_caption', ''),
                image_type=row.get('image_type', 'unknown'),
                # Store link score in metadata for potential later use
                metadata={"link_score": float(row['link_score'])}
            )

            # Limit to max 3 images per chunk
            if len(images_by_chunk[chunk_id]) < self.max_linked_images:
                images_by_chunk[chunk_id].append(img_obj)

        # Attach to results
        for result in results:
            if result.chunk_id in images_by_chunk:
                result.images = images_by_chunk[result.chunk_id]  # FIXED: .images not .linked_images

        return results
    
    # =========================================================================
    # Embedding
    # =========================================================================
    
    async def _embed_query(self, query: str) -> np.ndarray:
        """Get embedding for query text."""
        if hasattr(self.embedder, 'embed'):
            # Async embedder
            embedding = await self.embedder.embed(query)
        elif hasattr(self.embedder, 'embed_text'):
            # Sync embedder
            embedding = self.embedder.embed_text(query)
        else:
            raise ValueError("Embedder must have embed() or embed_text() method")
        
        return np.array(embedding, dtype=np.float32)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def search_text(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResponse:
        """Search text chunks only."""
        return await self.search(query, mode="text", top_k=top_k, **kwargs)
    
    async def search_images(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResponse:
        """Search images only."""
        return await self.search(query, mode="image", top_k=top_k, **kwargs)
    
    async def find_similar(
        self,
        item_id: str,
        item_type: str = "chunk",
        top_k: int = 10
    ) -> List[SearchResult]:
        """Find items similar to a given item."""
        if item_type == "chunk":
            candidates = self.faiss.search_similar_chunks(item_id, k=top_k * 2)
        else:
            candidates = self.faiss.search_similar_images(item_id, k=top_k * 2)
        
        # Enrich from database
        results = await self._enrich_results(
            candidates,
            SearchFilters(),
            "text" if item_type == "chunk" else "image"
        )
        
        return results[:top_k]


# =============================================================================
# Simple Embedder (for standalone use)
# =============================================================================

class VoyageEmbedder:
    """Simple Voyage embedder for search service."""
    
    def __init__(self, api_key: str = None, model: str = "voyage-3"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    async def embed(self, text: str) -> List[float]:
        """Embed single text."""
        import voyageai
        
        if self._client is None:
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        
        response = await self._client.embed([text], model=self.model)
        return response.embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        import voyageai
        
        if self._client is None:
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        
        response = await self._client.embed(texts, model=self.model)
        return response.embeddings


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("SearchService - requires database and FAISS indexes")
    print()
    print("Usage:")
    print("  service = SearchService(db, faiss_manager, embedder)")
    print("  results = await service.search('acoustic neuroma', top_k=10)")
