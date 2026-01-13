"""
NeuroSynth Unified - Chunk Repository
======================================

Repository for text chunk CRUD and vector search operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Set
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

    @property
    def updatable_columns(self) -> Set[str]:
        return {'content', 'content_hash', 'chunk_type',
                'specialty', 'entities',
                'embedding', 'metadata'}

    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to chunk dict."""
        return {
            'id': row['id'],
            'document_id': row['document_id'],
            'content': row['content'],
            'content_hash': row.get('content_hash'),
            'summary': row.get('summary'),  # AI-generated summary (Stage 4.5)
            'page_number': row.get('page_number'),
            'chunk_index': row.get('chunk_index'),
            'start_char': row.get('start_char'),
            'end_char': row.get('end_char'),
            'chunk_type': row.get('chunk_type'),
            'specialty': row.get('specialty', {}),
            'embedding': row.get('embedding'),
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

        # Handle JSON fields - DB column is entities (v4.1 schema)
        entities = entity.get('entities', [])
        if isinstance(entities, list):
            entities = json.dumps(entities)

        # DB column is specialty (v4.1 schema)
        specialty = entity.get('specialty') or entity.get('specialty_relevance', {})
        if isinstance(specialty, dict):
            specialty = json.dumps(specialty)

        return {
            'id': entity.get('id'),
            'document_id': entity['document_id'],
            'content': entity['content'],
            'content_hash': entity.get('content_hash'),
            'page_number': entity.get('page_number') or entity.get('start_page'),
            'chunk_index': entity.get('chunk_index') or entity.get('sequence_in_doc'),
            'start_char': entity.get('start_char') or entity.get('char_offset_start'),
            'end_char': entity.get('end_char') or entity.get('char_offset_end'),
            'chunk_type': entity.get('chunk_type'),
            'specialty': specialty,
            'embedding': embedding,
            'entities': entities,
            'cuis': entity.get('cuis', [])
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
            
            # Prepare cuis array
            cuis = chunk.get('cuis', [])
            if isinstance(cuis, set):
                cuis = list(cuis)

            # Prepare summary embedding if available
            summary_embedding = chunk.get('summary_embedding')
            if summary_embedding is not None:
                if isinstance(summary_embedding, np.ndarray):
                    summary_embedding = DatabaseConnection._encode_vector(summary_embedding)
                elif isinstance(summary_embedding, list):
                    summary_embedding = DatabaseConnection._encode_vector(summary_embedding)

            records.append((
                chunk['id'],
                document_id,
                chunk.get('content', '') or chunk.get('text_content', ''),
                chunk.get('page_number') or chunk.get('start_page'),
                chunk.get('chunk_index') or chunk.get('sequence_in_doc') or chunk.get('index'),
                chunk.get('start_char') or chunk.get('char_offset_start'),
                chunk.get('end_char') or chunk.get('char_offset_end'),
                chunk.get('chunk_type'),
                chunk.get('specialty') or chunk.get('specialty_relevance'),
                embedding,
                cuis or [],
                entities,
                # Quality scores (v2.2)
                chunk.get('readability_score', 0.0),
                chunk.get('coherence_score', 0.0),
                chunk.get('completeness_score', 0.0),
                chunk.get('type_specific_score', 0.0),
                chunk.get('summary'),
                summary_embedding,
            ))

        async with self.db.transaction() as conn:
            await conn.executemany("""
                INSERT INTO chunks (
                    id, document_id, content,
                    page_number, chunk_index, start_char, end_char,
                    chunk_type, specialty, embedding, cuis, entities,
                    readability_score, coherence_score, completeness_score,
                    type_specific_score, summary, summary_embedding
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11, $12::jsonb,
                    $13, $14, $15, $16, $17, $18::vector
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
        include_embedding: bool = False,
        limit: int = None,
        offset: int = None
    ) -> List[Dict[str, Any]]:
        """Get chunks for a document, optionally filtered by page.

        Args:
            document_id: Document UUID
            page_number: Filter by PDF page number
            include_embedding: Include embedding vector
            limit: Max results (for pagination)
            offset: Skip results (for pagination)
        """
        columns = """
            id, document_id, content, summary, page_number, chunk_index,
            chunk_type, specialty, entities, cuis, metadata
        """
        if include_embedding:
            columns += ", embedding"

        params = [document_id]
        pagination = ""

        if limit is not None:
            params.append(limit)
            pagination += f" LIMIT ${len(params)}"
            if offset is not None:
                params.append(offset)
                pagination += f" OFFSET ${len(params)}"

        if page_number is not None:
            params.insert(1, page_number)  # Insert after document_id
            query = f"""
                SELECT {columns}
                FROM chunks
                WHERE document_id = $1 AND page_number = $2
                ORDER BY chunk_index
                {pagination}
            """
        else:
            query = f"""
                SELECT {columns}
                FROM chunks
                WHERE document_id = $1
                ORDER BY page_number, chunk_index
                {pagination}
            """

        rows = await self.db.fetch(query, *params)
        return [self._to_entity(dict(row)) for row in rows]

    async def count_by_document(
        self,
        document_id: UUID,
        page_number: int = None,
        chunk_type: str = None
    ) -> int:
        """Count chunks for a document."""
        params = [document_id]
        where_parts = ["document_id = $1"]

        if page_number is not None:
            params.append(page_number)
            where_parts.append(f"page_number = ${len(params)}")

        if chunk_type:
            params.append(chunk_type)
            where_parts.append(f"chunk_type = ${len(params)}")

        query = f"""
            SELECT COUNT(*) FROM chunks
            WHERE {' AND '.join(where_parts)}
        """
        return await self.db.fetchval(query, *params) or 0
    
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
        specialty_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks by neurosurgery specialty."""
        # specialty is JSONB, query checks if specialty key exists
        query = """
            SELECT id, document_id, content, page_number, chunk_type, specialty, cuis
            FROM chunks
            WHERE specialty ? $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, specialty_name, limit)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_cui(
        self,
        cui: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get chunks containing a specific UMLS CUI."""
        query = """
            SELECT * FROM chunks
            WHERE $1 = ANY(cuis)
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
                SELECT * FROM chunks
                WHERE cuis @> $1::TEXT[]
                LIMIT $2
            """
        else:
            # Any CUI matches (overlap)
            query = """
                SELECT * FROM chunks
                WHERE cuis && $1::TEXT[]
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
                    COUNT(*) FILTER (WHERE cuis IS NOT NULL AND array_length(cuis, 1) > 0) as with_cuis,
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
                    COUNT(*) FILTER (WHERE cuis IS NOT NULL AND array_length(cuis, 1) > 0) as with_cuis,
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

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document."""
        query = "DELETE FROM chunks WHERE document_id = $1"
        result = await self.db.execute(query, document_id)
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted {count} chunks for document {document_id}")
        return count

    async def delete_many(self, ids: List[UUID]) -> int:
        """Delete multiple chunks by ID."""
        if not ids:
            return 0
        query = "DELETE FROM chunks WHERE id = ANY($1::uuid[])"
        result = await self.db.execute(query, [str(id) for id in ids])
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted {count} chunks")
        return count

    async def list_all_ids(self) -> List[UUID]:
        """Get all chunk IDs."""
        query = "SELECT id FROM chunks ORDER BY created_at DESC"
        rows = await self.db.fetch(query)
        return [row['id'] for row in rows]

    async def delete_all(self) -> int:
        """Delete all chunks."""
        result = await self.db.execute("DELETE FROM chunks")
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted all chunks: {count} total")
        return count
