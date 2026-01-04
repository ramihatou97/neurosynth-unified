"""
NeuroSynth Unified - Search Service
====================================

Unified search interface combining FAISS (speed) and pgvector (filtering).

SCHEMA ALIGNMENT NOTES (v4.0):
- Table: links (not chunk_image_links)
- Column: score (not relevance_score)
- Column: file_path (not storage_path)
- Column: cuis (not topic_tags)
- Column: page_number (not start_page)
- Column: specialty (not specialty_relevance)

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
from pathlib import Path
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
    min_quality: float = None  # Minimum quality score (0-1)

    @property
    def has_filters(self) -> bool:
        return bool(
            self.document_ids or
            self.chunk_types or
            self.specialties or
            self.image_types or
            self.cuis or
            self.page_range or
            self.min_quality
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
        faiss_manager: FAISSManager = None,
        embedder=None,
        reranker=None,
        config: Dict = None,
        use_pgvector: bool = True,
        cache=None
    ):
        """
        Initialize search service.

        Args:
            database: Database connection or repositories
            faiss_manager: FAISS index manager (optional, deprecated)
            embedder: Text embedding service
            reranker: Optional re-ranking model
            config: Search configuration
            use_pgvector: Use pgvector HNSW indexes (default: True)
            cache: Optional SearchCache instance for embedding/result caching
        """
        self.db = database
        self.faiss = faiss_manager
        self.embedder = embedder
        self.reranker = reranker
        self._use_pgvector = use_pgvector
        self.cache = cache  # SearchCache instance (optional)

        # Initialize pgvector searcher when FAISS is not available
        self._pgvector_searcher = None
        if self._use_pgvector or not self.faiss:
            self._pgvector_searcher = PostgresVectorSearcher(
                database=database,
                embedder=embedder,
                config=config
            )

        # Configuration
        self.config = config or {}
        self.faiss_k_multiplier = self.config.get('faiss_k_multiplier', 10)
        self.text_weight = self.config.get('text_weight', 0.7)
        self.image_weight = self.config.get('image_weight', 0.3)
        self.cui_boost = self.config.get('cui_boost', 1.2)
        self.min_similarity = self.config.get('min_similarity', 0.1)
        self.max_linked_images = self.config.get('max_linked_images', 3)

        # Determine active backend for logging/debugging
        self._backend_name = self._determine_backend_name()
        logger.info(f"SearchService initialized with backend: {self._backend_name}")

    def _determine_backend_name(self) -> str:
        """Determine which search backend is active."""
        if self._use_pgvector and self._pgvector_searcher:
            return "pgvector-hnsw"
        elif self.faiss:
            return "faiss"
        elif self._pgvector_searcher:
            return "pgvector-hnsw (fallback)"
        else:
            return "none"

    @property
    def backend_name(self) -> str:
        """Get the name of the active search backend."""
        return self._backend_name

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

        # Step 4b: Type-aware boosting (detect query intent)
        query_intent = self._detect_query_intent(query)
        if query_intent:
            results = self._apply_type_boost(results, query_intent)

        # Step 4c: Authority boosting (always applied)
        results = self._apply_authority_boost(results)

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
        """Text search using FAISS or pgvector."""
        # Use pgvector if FAISS is not available
        if not self.faiss and self._pgvector_searcher:
            results = await self._pgvector_searcher.search_chunks(
                query_embedding=list(query_embedding),
                top_k=top_k,
                filters=filters
            )
            # Return as (id, score) tuples for compatibility
            return [(r.chunk_id, r.semantic_score) for r in results]

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
        """Image search using FAISS (via caption embeddings) or pgvector."""
        # Use pgvector if FAISS is not available
        if not self.faiss and self._pgvector_searcher:
            results = await self._pgvector_searcher.search_images(
                query_embedding=list(query_embedding),
                top_k=top_k,
                use_caption_embedding=True,  # Text-to-image search
                filters=filters
            )
            # Return as (id, score) tuples for compatibility
            return [(r['id'], r['similarity_score']) for r in results]

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
        # Use pgvector if FAISS is not available
        if not self.faiss and self._pgvector_searcher:
            results = await self._pgvector_searcher.search_hybrid(
                query_embedding=list(query_embedding),
                top_k=top_k,
                text_weight=self.text_weight,
                filters=filters
            )
            # Return as (id, score) tuples for compatibility
            return [(r.chunk_id, r.semantic_score) for r in results]

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
        
        # NOTE: Image-only search temporarily disabled for synthesis compatibility
        # Images are attached to chunks via _attach_linked_images() instead
        # TODO: Implement separate ImageSearchResult type for image-only searches
        # if mode in ("image", "hybrid"):
        #     image_results = await self._fetch_images(candidate_ids, filters)
        #     results.extend(image_results)
        
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
        
        # Build query with filters (using table aliases for JOIN)
        conditions = ["c.id = ANY($1)"]
        params = [ids]
        param_idx = 2

        if filters.document_ids:
            conditions.append(f"c.document_id = ANY(${param_idx})")
            params.append([UUID(d) for d in filters.document_ids])
            param_idx += 1

        if filters.chunk_types:
            conditions.append(f"c.chunk_type = ANY(${param_idx})")
            params.append(filters.chunk_types)
            param_idx += 1

        if filters.specialties:
            # specialty is JSONB in schema
            conditions.append(f"c.specialty ? ${param_idx}")
            params.append(filters.specialties)
            param_idx += 1

        if filters.cuis:
            # cuis array column
            conditions.append(f"c.cuis && ${param_idx}")
            params.append(filters.cuis)
            param_idx += 1

        if filters.page_range:
            conditions.append(f"c.page_number >= ${param_idx} AND c.page_number <= ${param_idx + 1}")
            params.extend(filters.page_range)
            param_idx += 2

        if filters.min_quality is not None and filters.min_quality > 0:
            # Use type_specific_score as quality proxy (v2.2+)
            conditions.append(f"COALESCE(c.type_specific_score, 0) >= ${param_idx}")
            params.append(filters.min_quality)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Query using schema-aligned column names (v4.1)
        # Schema columns: page_number, cuis, entities, specialty, summary
        query = f"""
            SELECT
                c.id,
                c.document_id,
                c.content,
                c.summary,
                c.page_number,
                c.chunk_type,
                c.cuis,
                c.entities,
                c.specialty,
                c.embedding,
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
            # Extract entity names from entities JSONB (schema column name)
            entities_data = row.get('entities', []) or []
            entity_names = []
            if isinstance(entities_data, dict):
                # Extract entity text values from JSONB structure
                entity_names = list(entities_data.keys()) if entities_data else []
            elif isinstance(entities_data, list):
                # List of entity objects - extract names
                for e in entities_data:
                    if isinstance(e, dict) and 'name' in e:
                        entity_names.append(e['name'])
                    elif isinstance(e, str):
                        entity_names.append(e)

            # Parse chunk_type string to ChunkType enum
            chunk_type = ChunkType.GENERAL
            if row.get('chunk_type'):
                try:
                    chunk_type = ChunkType[row['chunk_type'].upper()]
                except (KeyError, ValueError):
                    # If conversion fails, default to GENERAL
                    pass

            # ADAPTED: Populate SearchResult using actual schema fields
            # Now includes quality scores for synthesis filtering
            # Parse embedding from pgvector format if present
            embedding = None
            raw_embedding = row.get('embedding')
            if raw_embedding is not None:
                if isinstance(raw_embedding, (list, tuple)):
                    embedding = np.array(raw_embedding, dtype=np.float32)
                elif hasattr(raw_embedding, 'tolist'):
                    embedding = np.array(raw_embedding.tolist(), dtype=np.float32)

            results.append(SearchResult(
                chunk_id=row_id,
                document_id=str(row['document_id']),
                content=row['content'],
                title='',  # No title in current schema
                chunk_type=chunk_type,
                page_start=row.get('page_number', 0),
                entity_names=entity_names,  # Extracted from entities JSONB
                image_ids=[],  # Not stored in chunks table
                cuis=row.get('cuis', []) or [],
                authority_score=float(row['authority_score']),
                keyword_score=0.0,
                semantic_score=0.0,  # Set from FAISS scores later
                final_score=0.0,  # Computed from weighted scores
                document_title=row.get('document_title'),
                summary=row.get('summary'),  # AI-generated summary
                images=[],  # Populated by _attach_linked_images()
                readability_score=0.0,  # Not in DB schema, use default
                coherence_score=0.0,    # Not in DB schema, use default
                completeness_score=0.0, # Not in DB schema, use default
                embedding=embedding,
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

    def _detect_query_intent(self, query: str) -> Optional[ChunkType]:
        """
        Detect query intent to enable type-aware boosting.

        Analyzes query text for keywords indicating procedure, anatomy, etc.
        Returns None if no clear intent detected.
        """
        query_lower = query.lower()

        # Procedure indicators
        procedure_keywords = [
            'approach', 'technique', 'procedure', 'craniotomy', 'resection',
            'dissection', 'how to', 'step', 'surgical', 'incision', 'exposure',
            'retraction', 'clipping', 'coiling', 'decompression'
        ]
        if any(kw in query_lower for kw in procedure_keywords):
            return ChunkType.PROCEDURE

        # Anatomy indicators
        anatomy_keywords = [
            'anatomy', 'structure', 'nerve', 'artery', 'vein', 'cistern',
            'fissure', 'sulcus', 'gyrus', 'nucleus', 'foramen', 'sinus',
            'where is', 'location of', 'relationship'
        ]
        if any(kw in query_lower for kw in anatomy_keywords):
            return ChunkType.ANATOMY

        # Pathology indicators
        pathology_keywords = [
            'tumor', 'meningioma', 'schwannoma', 'glioma', 'aneurysm',
            'malformation', 'disease', 'pathology', 'lesion', 'diagnosis'
        ]
        if any(kw in query_lower for kw in pathology_keywords):
            return ChunkType.PATHOLOGY

        return None

    def _apply_type_boost(
        self,
        results: List[SearchResult],
        target_type: ChunkType,
        boost_factor: float = 1.15
    ) -> List[SearchResult]:
        """
        Boost results that match the detected query intent type.

        Args:
            results: Search results with chunk_type set
            target_type: ChunkType to boost
            boost_factor: Multiplier for matching types (default 1.15 = 15% boost)

        Returns:
            Results with adjusted final_score, re-sorted
        """
        for result in results:
            if result.chunk_type == target_type:
                result.final_score *= boost_factor

        # Re-sort by final_score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _apply_authority_boost(
        self,
        results: List[SearchResult],
        boost_factor: float = 0.15
    ) -> List[SearchResult]:
        """
        Boost results from high-authority sources.

        Authority scores are tiered:
        - 1.0: Primary authoritative texts (Rhoton, Lawton, Samii)
        - 0.9: Major textbooks
        - 0.8: Peer-reviewed journals
        - 0.7: Default (baseline)

        The boost formula gives higher-authority sources an advantage:
        - Authority 1.0 → 1.045x boost
        - Authority 0.9 → 1.030x boost
        - Authority 0.8 → 1.015x boost
        - Authority 0.7 → 1.0x (no change)

        Args:
            results: Search results with authority_score set
            boost_factor: Multiplier for authority difference (default 0.15)

        Returns:
            Results with adjusted final_score, re-sorted
        """
        BASELINE_AUTHORITY = 0.7
        FRONT_MATTER_PENALTY = 0.1  # Severely deprioritize front matter chunks

        for result in results:
            # Check for FRONT_MATTER chunk type - apply severe penalty
            if result.chunk_type and result.chunk_type.value == "front_matter":
                result.final_score *= FRONT_MATTER_PENALTY
                continue

            authority = result.authority_score or BASELINE_AUTHORITY
            # Calculate boost: 1.0 + (authority - baseline) * factor
            # e.g., authority=1.0 → 1.0 + (1.0-0.7)*0.15 = 1.045
            boost = 1.0 + (authority - BASELINE_AUTHORITY) * boost_factor
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

        # Query using schema-aligned column names (v4.1)
        # Includes caption_embedding for semantic figure matching
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
                i.caption_summary,
                i.caption_embedding,
                i.image_type,
                i.width,
                i.height,
                i.format,
                i.quality_score
            FROM links l
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

            # Strip output/images/ prefix from storage_path for frontend compatibility
            file_path_str = row['file_path'] or ''
            if file_path_str.startswith('output/images/'):
                file_path_str = file_path_str[len('output/images/'):]

            # Parse caption_embedding from pgvector for semantic figure matching
            caption_embedding = None
            raw_caption_emb = row.get('caption_embedding')
            if raw_caption_emb is not None:
                if isinstance(raw_caption_emb, (list, tuple)):
                    caption_embedding = np.array(raw_caption_emb, dtype=np.float32)
                elif hasattr(raw_caption_emb, 'tolist'):
                    caption_embedding = np.array(raw_caption_emb.tolist(), dtype=np.float32)

            img_obj = ExtractedImage(
                id=str(row['id']),
                document_id=str(row['document_id']),
                file_path=Path(file_path_str),
                page_number=row.get('page_number') or 0,
                width=row.get('width') or 0,
                height=row.get('height') or 0,
                format=row.get('format') or 'JPEG',
                content_hash='',
                caption=row.get('caption') or '',
                vlm_caption=row.get('vlm_caption') or '',
                caption_summary=row.get('caption_summary'),
                caption_embedding=caption_embedding,
                image_type=row.get('image_type') or 'unknown',
                quality_score=row.get('quality_score') or 0.0
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
        """
        Get embedding for query text, using cache if available.

        When cache is enabled, this provides 20-30ms savings on repeated queries.
        """
        # Use cache if available
        if self.cache:
            embedding = await self.cache.get_or_compute_embedding(
                text=query,
                embed_func=self._embed_query_uncached,
                model=getattr(self.embedder, 'model', 'voyage-3'),
                dimension=1024
            )
            return np.array(embedding, dtype=np.float32)

        # Direct embedding (no cache)
        return await self._embed_query_uncached(query)

    async def _embed_query_uncached(self, query: str) -> np.ndarray:
        """Get embedding for query text (uncached, direct)."""
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
# Embedding Dimension Constants
# =============================================================================

EMBEDDING_DIMENSIONS = {
    "text": 1024,     # Voyage-3
    "image": 512,     # BiomedCLIP
    "caption": 1024,  # Voyage-3 (for image captions)
}


# =============================================================================
# PostgresVectorSearcher - pgvector-native search (FAISS alternative)
# =============================================================================

class PostgresVectorSearcher:
    """
    Pure pgvector search implementation.

    Uses PostgreSQL pgvector extension directly for vector similarity search,
    eliminating need for separate FAISS indexes. Recommended for datasets
    under 5M vectors with HNSW indexes.

    Benefits over FAISS:
    - Single source of truth (no index sync issues)
    - Transactional consistency
    - Native filtering in query
    - Simplified deployment

    Performance expectations with HNSW:
    - 1K vectors: <10ms
    - 10K vectors: <20ms
    - 100K vectors: <50ms
    - 1M vectors: <100ms
    """

    def __init__(self, database, embedder=None, config: Dict = None):
        """
        Initialize pgvector searcher.

        Args:
            database: Database connection with fetch/execute methods
            embedder: Optional embedder for query embedding
            config: Search configuration
        """
        self.db = database
        self.embedder = embedder
        self.config = config or {}
        self.min_similarity = self.config.get('min_similarity', 0.1)

    def _validate_embedding_dimension(
        self,
        embedding: List[float],
        expected_type: str
    ) -> None:
        """
        Validate embedding dimension matches expected type.

        Raises ValueError if dimension mismatch detected.
        """
        expected_dim = EMBEDDING_DIMENSIONS.get(expected_type)
        if expected_dim is None:
            raise ValueError(f"Unknown embedding type: {expected_type}")

        actual_dim = len(embedding)
        if actual_dim != expected_dim:
            raise ValueError(
                f"{expected_type.capitalize()} search requires {expected_dim}d embedding, "
                f"got {actual_dim}d. "
                f"(text=1024d Voyage-3, image=512d BiomedCLIP)"
            )

    def _format_embedding_for_pgvector(self, embedding: List[float]) -> str:
        """
        Format embedding list as pgvector string.

        asyncpg doesn't natively support pgvector type, so we convert
        the Python list to the string format pgvector expects: '[val1,val2,...]'
        """
        return '[' + ','.join(str(v) for v in embedding) + ']'

    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: SearchFilters = None,
        min_similarity: float = None
    ) -> List[SearchResult]:
        """
        Search text chunks using pgvector cosine similarity.

        Args:
            query_embedding: 1024d Voyage-3 embedding
            top_k: Number of results
            filters: Optional filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Validate dimension
        self._validate_embedding_dimension(query_embedding, "text")

        filters = filters or SearchFilters()
        min_sim = min_similarity or self.min_similarity

        # Format embedding for pgvector (asyncpg doesn't natively support vector type)
        embedding_str = self._format_embedding_for_pgvector(query_embedding)

        # Build filter conditions
        # Note: Using 'embedding' column (actual data), not 'text_embedding' (empty)
        conditions = ["1 - (c.embedding <=> $1::vector) >= $2"]
        params = [embedding_str, min_sim]
        param_idx = 3

        if filters.document_ids:
            conditions.append(f"c.document_id = ANY(${param_idx})")
            params.append([UUID(d) for d in filters.document_ids])
            param_idx += 1

        if filters.chunk_types:
            conditions.append(f"c.chunk_type = ANY(${param_idx})")
            params.append(filters.chunk_types)
            param_idx += 1

        if filters.cuis:
            conditions.append(f"c.cuis && ${param_idx}")
            params.append(filters.cuis)
            param_idx += 1

        if filters.page_range:
            conditions.append(f"c.page_number >= ${param_idx}")
            conditions.append(f"c.page_number <= ${param_idx + 1}")
            params.extend(filters.page_range)
            param_idx += 2

        if filters.min_quality is not None and filters.min_quality > 0:
            conditions.append(f"COALESCE(c.type_specific_score, 0) >= ${param_idx}")
            params.append(filters.min_quality)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # pgvector cosine distance: <=> returns distance, so similarity = 1 - distance
        # Schema-aligned columns (v4.1): page_number, cuis, entities
        query = f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                c.summary,
                c.page_number,
                c.chunk_type,
                c.cuis,
                c.entities,
                d.title AS document_title,
                COALESCE(d.authority_score, 1.0) AS authority_score,
                1 - (c.embedding <=> $1::vector) AS similarity_score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
                AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(top_k)

        rows = await self.db.fetch(query, *params)

        results = []
        for row in rows:
            entities_data = row.get('entities', []) or []
            entity_names = []
            if isinstance(entities_data, dict):
                entity_names = list(entities_data.keys())
            elif isinstance(entities_data, list):
                for e in entities_data:
                    if isinstance(e, dict) and 'name' in e:
                        entity_names.append(e['name'])
                    elif isinstance(e, str):
                        entity_names.append(e)

            chunk_type = ChunkType.GENERAL
            if row.get('chunk_type'):
                try:
                    chunk_type = ChunkType[row['chunk_type'].upper()]
                except (KeyError, ValueError):
                    pass

            results.append(SearchResult(
                chunk_id=str(row['chunk_id']),
                document_id=str(row['document_id']),
                content=row['content'],
                title='',
                chunk_type=chunk_type,
                page_start=row.get('page_number', 0),
                entity_names=entity_names,
                image_ids=[],
                cuis=row.get('cuis', []) or [],
                authority_score=float(row['authority_score']),
                keyword_score=0.0,
                semantic_score=float(row['similarity_score']),
                final_score=float(row['similarity_score']),
                document_title=row.get('document_title'),
                summary=row.get('summary'),  # AI-generated summary
                images=[],
            ))

        return results

    async def search_images(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        use_caption_embedding: bool = True,
        filters: SearchFilters = None,
        min_similarity: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search images using pgvector.

        Args:
            query_embedding: Embedding vector (1024d for text query, 512d for image query)
            top_k: Number of results
            use_caption_embedding: If True, search caption embeddings (text-to-image).
                                   If False, search image embeddings (image-to-image).
            filters: Optional filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of image result dicts
        """
        # Validate dimension based on search type
        # Note: Using 'clip_embedding' (actual data), not 'image_embedding' (empty)
        if use_caption_embedding:
            self._validate_embedding_dimension(query_embedding, "caption")
            embedding_col = "caption_embedding"
        else:
            self._validate_embedding_dimension(query_embedding, "image")
            embedding_col = "clip_embedding"  # BiomedCLIP embeddings

        filters = filters or SearchFilters()
        min_sim = min_similarity or self.min_similarity

        # Format embedding for pgvector (asyncpg doesn't natively support vector type)
        embedding_str = self._format_embedding_for_pgvector(query_embedding)

        conditions = [
            f"1 - (i.{embedding_col} <=> $1::vector) >= $2",
            "NOT i.is_decorative"
        ]
        params = [embedding_str, min_sim]
        param_idx = 3

        if filters.document_ids:
            conditions.append(f"i.document_id = ANY(${param_idx})")
            params.append([UUID(d) for d in filters.document_ids])
            param_idx += 1

        if filters.image_types:
            conditions.append(f"i.image_type = ANY(${param_idx})")
            params.append(filters.image_types)
            param_idx += 1

        if filters.page_range:
            conditions.append(f"i.page_number >= ${param_idx}")
            conditions.append(f"i.page_number <= ${param_idx + 1}")
            params.extend(filters.page_range)
            param_idx += 2

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                i.id,
                i.document_id,
                i.file_path,
                i.caption,
                i.image_type,
                i.width,
                i.height,
                i.format,
                1 - (i.{embedding_col} <=> $1::vector) AS similarity_score
            FROM images i
            WHERE {where_clause}
                AND i.{embedding_col} IS NOT NULL
            ORDER BY i.{embedding_col} <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(top_k)

        rows = await self.db.fetch(query, *params)

        results = []
        for row in rows:
            results.append({
                'id': str(row['id']),
                'document_id': str(row['document_id']),
                'file_path': row['file_path'],
                'caption': row.get('caption', ''),
                'image_type': row.get('image_type'),
                'width': row.get('width'),
                'height': row.get('height'),
                'format': row.get('format'),
                'similarity_score': float(row['similarity_score']),
            })

        return results

    async def search_hybrid(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        text_weight: float = 0.7,
        filters: SearchFilters = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining text chunks with linked images.

        Args:
            query_embedding: 1024d Voyage-3 embedding
            top_k: Number of results
            text_weight: Weight for text results (images get 1 - text_weight)
            filters: Optional filters

        Returns:
            List of SearchResult with attached images
        """
        # Search chunks first
        chunk_results = await self.search_chunks(
            query_embedding,
            top_k=top_k * 2,  # Over-fetch for filtering
            filters=filters
        )

        if not chunk_results:
            return []

        # Attach linked images
        chunk_ids = [UUID(r.chunk_id) for r in chunk_results]

        # Query with caption_embedding for semantic figure matching
        link_query = """
            SELECT
                l.chunk_id,
                l.score,
                i.id,
                i.document_id,
                i.file_path,
                i.caption,
                i.vlm_caption,
                i.caption_embedding,
                i.image_type,
                i.width,
                i.height,
                i.format,
                i.quality_score
            FROM links l
            JOIN images i ON l.image_id = i.id
            WHERE l.chunk_id = ANY($1)
                AND l.score >= 0.5
                AND NOT i.is_decorative
            ORDER BY l.chunk_id, l.score DESC
        """

        link_rows = await self.db.fetch(link_query, chunk_ids)

        # Group images by chunk
        images_by_chunk = {}
        for row in link_rows:
            chunk_id = str(row['chunk_id'])
            if chunk_id not in images_by_chunk:
                images_by_chunk[chunk_id] = []

            file_path_str = row['file_path'] or ''
            if file_path_str.startswith('output/images/'):
                file_path_str = file_path_str[len('output/images/'):]

            # Parse caption_embedding from pgvector for semantic figure matching
            caption_embedding = None
            raw_caption_emb = row.get('caption_embedding')
            if raw_caption_emb is not None:
                if isinstance(raw_caption_emb, (list, tuple)):
                    caption_embedding = np.array(raw_caption_emb, dtype=np.float32)
                elif hasattr(raw_caption_emb, 'tolist'):
                    caption_embedding = np.array(raw_caption_emb.tolist(), dtype=np.float32)

            if len(images_by_chunk[chunk_id]) < 3:  # Max 3 images per chunk
                images_by_chunk[chunk_id].append(ExtractedImage(
                    id=str(row['id']),
                    document_id=str(row['document_id']),
                    file_path=Path(file_path_str),
                    page_number=0,
                    width=row.get('width') or 0,
                    height=row.get('height') or 0,
                    format=row.get('format') or 'JPEG',
                    content_hash='',
                    caption=row.get('caption') or '',
                    vlm_caption=row.get('vlm_caption') or '',
                    caption_embedding=caption_embedding,
                    image_type=row.get('image_type') or 'unknown',
                    quality_score=float(row.get('quality_score') or 0.0)
                ))

        # Attach images to results
        for result in chunk_results:
            if result.chunk_id in images_by_chunk:
                result.images = images_by_chunk[result.chunk_id]

        return chunk_results[:top_k]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("SearchService - requires database and FAISS indexes")
    print("PostgresVectorSearcher - requires database with pgvector")
    print()
    print("Usage:")
    print("  # FAISS-based (existing):")
    print("  service = SearchService(db, faiss_manager, embedder)")
    print("  results = await service.search('acoustic neuroma', top_k=10)")
    print()
    print("  # pgvector-based (new, recommended):")
    print("  searcher = PostgresVectorSearcher(db, embedder)")
    print("  results = await searcher.search_chunks(query_embedding, top_k=10)")
