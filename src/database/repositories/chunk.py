"""
NeuroSynth Unified - Chunk Repository
======================================

Repository for text chunk CRUD and vector search operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime
import json

import numpy as np

from src.database.repositories.base import BaseRepository, VectorSearchMixin

logger = logging.getLogger(__name__)


class ChunkRepository(BaseRepository, VectorSearchMixin):
    """
    Repository for text chunk operations with vector search.
    
    Chunks are semantic segments of text extracted from documents,
    with 1024-dimensional Voyage-3 embeddings and UMLS CUIs.
    """
    
    @property
    def table_name(self) -> str:
        return "chunks"
    
    @property
    def embedding_column(self) -> str:
        return "embedding"
    
    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to chunk dict."""
        return {
            'id': row['id'],
            'document_id': row['document_id'],
            'content': row['content'],
            'content_hash': row.get('content_hash'),
            'page_number': row.get('page_number'),
            'chunk_index': row.get('chunk_index'),
            'start_char': row.get('start_char'),
            'end_char': row.get('end_char'),
            'chunk_type': row.get('chunk_type'),
            'specialty': row.get('specialty'),
            'embedding': row.get('embedding'),  # numpy array from pgvector
            'cuis': row.get('cuis', []),
            'entities': row.get('entities', []),
            'created_at': row.get('created_at'),
            'metadata': row.get('metadata', {}),
            # Search result fields
            'similarity': row.get('similarity'),
            'cui_overlap': row.get('cui_overlap'),
            'combined_score': row.get('combined_score')
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunk dict to database record."""
        from src.database.connection import DatabaseConnection
        
        # Handle embedding
        embedding = entity.get('embedding')
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                embedding = DatabaseConnection._encode_vector(embedding)
            elif isinstance(embedding, list):
                embedding = DatabaseConnection._encode_vector(embedding)
        
        # Handle JSON fields
        entities = entity.get('entities', [])
        if isinstance(entities, list):
            entities = json.dumps(entities)
        
        metadata = entity.get('metadata', {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        
        return {
            'id': entity.get('id'),
            'document_id': entity['document_id'],
            'content': entity['content'],
            'content_hash': entity.get('content_hash'),
            'page_number': entity.get('page_number'),
            'chunk_index': entity.get('chunk_index'),
            'start_char': entity.get('start_char'),
            'end_char': entity.get('end_char'),
            'chunk_type': entity.get('chunk_type'),
            'specialty': entity.get('specialty'),
            'embedding': embedding,
            'cuis': entity.get('cuis', []),
            'entities': entities,
            'metadata': metadata
        }
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def create_many_for_document(
        self,
        document_id: UUID,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert chunks for a document.
        
        Args:
            document_id: Parent document ID
            chunks: List of chunk dicts (from Phase 1 SemanticChunk objects)
        
        Returns:
            Number of chunks inserted
        """
        from src.database.connection import DatabaseConnection
        from uuid import uuid4
        
        if not chunks:
            return 0
        
        records = []
        for chunk in chunks:
            # Generate ID if not present
            if 'id' not in chunk:
                chunk['id'] = uuid4()
            
            # Set document ID
            chunk['document_id'] = document_id
            
            # Prepare embedding
            embedding = chunk.get('embedding') or chunk.get('text_embedding')
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = DatabaseConnection._encode_vector(embedding)
                elif isinstance(embedding, list):
                    embedding = DatabaseConnection._encode_vector(embedding)
            
            # Prepare JSON fields
            entities = chunk.get('entities', [])
            if isinstance(entities, list):
                entities = json.dumps(entities)
            
            metadata = chunk.get('metadata', {})
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            records.append((
                chunk['id'],
                document_id,
                chunk.get('content', '') or chunk.get('text_content', ''),
                chunk.get('content_hash'),
                chunk.get('page_number'),
                chunk.get('chunk_index') or chunk.get('index'),
                chunk.get('start_char'),
                chunk.get('end_char'),
                chunk.get('chunk_type'),
                chunk.get('specialty'),
                embedding,
                list(chunk.get('cuis', []) or []),
                entities,
                metadata
            ))
        
        async with self.db.transaction() as conn:
            await conn.executemany("""
                INSERT INTO chunks (
                    id, document_id, content, content_hash,
                    page_number, chunk_index, start_char, end_char,
                    chunk_type, specialty, embedding, cuis, entities, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::vector, $12, $13::jsonb, $14::jsonb
                )
            """, records)
        
        logger.info(f"Inserted {len(records)} chunks for document {document_id}")
        return len(records)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    async def get_by_document(
        self,
        document_id: UUID,
        page_number: int = None,
        include_embedding: bool = False
    ) -> List[Dict[str, Any]]:
        """Get chunks for a document, optionally filtered by page."""
        columns = """
            id, document_id, content, page_number, chunk_index,
            chunk_type, specialty, cuis, entities, metadata
        """
        if include_embedding:
            columns += ", embedding"
        
        if page_number is not None:
            query = f"""
                SELECT {columns}
                FROM chunks
                WHERE document_id = $1 AND page_number = $2
                ORDER BY chunk_index
            """
            rows = await self.db.fetch(query, document_id, page_number)
        else:
            query = f"""
                SELECT {columns}
                FROM chunks
                WHERE document_id = $1
                ORDER BY page_number, chunk_index
            """
            rows = await self.db.fetch(query, document_id)
        
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_type(
        self,
        chunk_type: str,
        document_id: UUID = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks by type (PROCEDURE, ANATOMY, etc.)."""
        if document_id:
            query = """
                SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
                FROM chunks
                WHERE chunk_type = $1 AND document_id = $2
                ORDER BY page_number, chunk_index
                LIMIT $3
            """
            rows = await self.db.fetch(query, chunk_type, document_id, limit)
        else:
            query = """
                SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
                FROM chunks
                WHERE chunk_type = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await self.db.fetch(query, chunk_type, limit)
        
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_specialty(
        self,
        specialty: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks by neurosurgery specialty."""
        query = """
            SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
            FROM chunks
            WHERE specialty = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, specialty, limit)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_cui(
        self,
        cui: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks containing a specific UMLS CUI."""
        query = """
            SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
            FROM chunks
            WHERE $1 = ANY(cuis)
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, cui, limit)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_cuis(
        self,
        cuis: List[str],
        match_all: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks containing specified CUIs."""
        if match_all:
            # All CUIs must be present
            query = """
                SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
                FROM chunks
                WHERE cuis @> $1
                ORDER BY created_at DESC
                LIMIT $2
            """
        else:
            # Any CUI matches
            query = """
                SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
                FROM chunks
                WHERE cuis && $1
                ORDER BY created_at DESC
                LIMIT $2
            """
        
        rows = await self.db.fetch(query, cuis, limit)
        return [self._to_entity(dict(row)) for row in rows]
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_ids: List[UUID] = None,
        chunk_types: List[str] = None,
        specialties: List[str] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over chunks using vector similarity.
        
        Args:
            query_embedding: 1024d query vector
            top_k: Number of results
            document_ids: Filter by documents
            chunk_types: Filter by chunk types
            specialties: Filter by specialties
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of chunks with similarity scores
        """
        filters = {}
        
        if document_ids:
            filters['document_id'] = document_ids
        
        if chunk_types:
            filters['chunk_type'] = chunk_types
        
        if specialties:
            filters['specialty'] = specialties
        
        results = await self.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filters
        )
        
        return [self._to_entity(row) for row in results]
    
    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_cuis: List[str] = None,
        top_k: int = 10,
        cui_boost: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and CUI matching.
        """
        results = await self.search_by_vector_hybrid(
            query_embedding=query_embedding,
            query_cuis=query_cuis,
            top_k=top_k,
            cui_boost=cui_boost
        )
        
        return [self._to_entity(row) for row in results]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_statistics(self, document_id: UUID = None) -> Dict[str, Any]:
        """Get chunk statistics."""
        if document_id:
            query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embedding,
                    COUNT(*) FILTER (WHERE array_length(cuis, 1) > 0) as with_cuis,
                    COUNT(DISTINCT chunk_type) as unique_types,
                    COUNT(DISTINCT specialty) as unique_specialties
                FROM chunks
                WHERE document_id = $1
            """
            row = await self.db.fetchrow(query, document_id)
        else:
            query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embedding,
                    COUNT(*) FILTER (WHERE array_length(cuis, 1) > 0) as with_cuis,
                    COUNT(DISTINCT chunk_type) as unique_types,
                    COUNT(DISTINCT specialty) as unique_specialties
                FROM chunks
            """
            row = await self.db.fetchrow(query)
        
        return dict(row) if row else {}
    
    async def get_type_distribution(
        self,
        document_id: UUID = None
    ) -> Dict[str, int]:
        """Get distribution of chunk types."""
        if document_id:
            query = """
                SELECT chunk_type, COUNT(*) as count
                FROM chunks
                WHERE document_id = $1 AND chunk_type IS NOT NULL
                GROUP BY chunk_type
                ORDER BY count DESC
            """
            rows = await self.db.fetch(query, document_id)
        else:
            query = """
                SELECT chunk_type, COUNT(*) as count
                FROM chunks
                WHERE chunk_type IS NOT NULL
                GROUP BY chunk_type
                ORDER BY count DESC
            """
            rows = await self.db.fetch(query)
        
        return {row['chunk_type']: row['count'] for row in rows}
