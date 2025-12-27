"""
NeuroSynth Unified - Image Repository
======================================

Repository for image CRUD and vector search operations.
Supports dual embeddings: visual (BiomedCLIP 512d) and caption (Voyage 1024d).
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import json

import numpy as np

from src.database.repositories.base import BaseRepository, VectorSearchMixin

logger = logging.getLogger(__name__)


class ImageRepository(BaseRepository, VectorSearchMixin):
    """
    Repository for medical image operations with dual vector search.
    
    Images have two embedding types:
    - embedding: 512d BiomedCLIP visual embedding
    - caption_embedding: 1024d Voyage embedding of VLM caption
    """
    
    @property
    def table_name(self) -> str:
        return "images"
    
    @property
    def embedding_column(self) -> str:
        return "embedding"  # Default to visual embedding
    
    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to image dict."""
        return {
            'id': row['id'],
            'document_id': row['document_id'],
            'file_path': row['file_path'],
            'file_name': row.get('file_name'),
            'content_hash': row.get('content_hash'),
            'width': row.get('width'),
            'height': row.get('height'),
            'format': row.get('format'),
            'page_number': row.get('page_number'),
            'image_index': row.get('image_index'),
            'image_type': row.get('image_type'),
            'is_decorative': row.get('is_decorative', False),
            'vlm_caption': row.get('vlm_caption'),
            'vlm_confidence': row.get('vlm_confidence'),
            'embedding': row.get('embedding'),  # 512d visual
            'caption_embedding': row.get('caption_embedding'),  # 1024d caption
            'cuis': row.get('cuis', []),
            'entities': row.get('entities', []),
            'triage_skipped': row.get('triage_skipped', False),
            'triage_reason': row.get('triage_reason'),
            'created_at': row.get('created_at'),
            'metadata': row.get('metadata', {}),
            # Search result fields
            'similarity': row.get('similarity')
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image dict to database record."""
        from src.database.connection import DatabaseConnection
        
        # Handle embeddings
        embedding = entity.get('embedding')
        caption_embedding = entity.get('caption_embedding')
        
        if embedding is not None:
            if isinstance(embedding, (np.ndarray, list)):
                embedding = DatabaseConnection._encode_vector(embedding)
        
        if caption_embedding is not None:
            if isinstance(caption_embedding, (np.ndarray, list)):
                caption_embedding = DatabaseConnection._encode_vector(caption_embedding)
        
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
            'file_path': entity['file_path'],
            'file_name': entity.get('file_name'),
            'content_hash': entity.get('content_hash'),
            'width': entity.get('width'),
            'height': entity.get('height'),
            'format': entity.get('format'),
            'page_number': entity.get('page_number'),
            'image_index': entity.get('image_index'),
            'image_type': entity.get('image_type'),
            'is_decorative': entity.get('is_decorative', False),
            'vlm_caption': entity.get('vlm_caption'),
            'vlm_confidence': entity.get('vlm_confidence'),
            'embedding': embedding,
            'caption_embedding': caption_embedding,
            'cuis': entity.get('cuis', []),
            'entities': entities,
            'triage_skipped': entity.get('triage_skipped', False),
            'triage_reason': entity.get('triage_reason'),
            'metadata': metadata
        }
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def create_many_for_document(
        self,
        document_id: UUID,
        images: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert images for a document.
        
        Args:
            document_id: Parent document ID
            images: List of image dicts (from Phase 1 ExtractedImage objects)
        
        Returns:
            Number of images inserted
        """
        from src.database.connection import DatabaseConnection
        from uuid import uuid4
        
        if not images:
            return 0
        
        records = []
        for img in images:
            # Generate ID if not present
            if 'id' not in img:
                img['id'] = uuid4()
            
            # Prepare embeddings
            embedding = img.get('embedding')
            caption_embedding = img.get('caption_embedding')
            
            if embedding is not None:
                if isinstance(embedding, (np.ndarray, list)):
                    embedding = DatabaseConnection._encode_vector(embedding)
            
            if caption_embedding is not None:
                if isinstance(caption_embedding, (np.ndarray, list)):
                    caption_embedding = DatabaseConnection._encode_vector(caption_embedding)
            
            # Prepare JSON fields
            entities = img.get('entities', [])
            if isinstance(entities, list):
                entities = json.dumps(entities)
            
            metadata = img.get('metadata', {})
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            
            records.append((
                img['id'],
                document_id,
                img.get('file_path', ''),
                img.get('file_name'),
                img.get('content_hash'),
                img.get('width'),
                img.get('height'),
                img.get('format'),
                img.get('page_number'),
                img.get('image_index') or img.get('index'),
                img.get('image_type'),
                img.get('is_decorative', False),
                img.get('vlm_caption'),
                img.get('vlm_confidence'),
                embedding,
                caption_embedding,
                list(img.get('cuis', []) or []),
                entities,
                img.get('triage_skipped', False),
                img.get('triage_reason'),
                metadata
            ))
        
        async with self.db.transaction() as conn:
            await conn.executemany("""
                INSERT INTO images (
                    id, document_id, file_path, file_name, content_hash,
                    width, height, format, page_number, image_index,
                    image_type, is_decorative, vlm_caption, vlm_confidence,
                    embedding, caption_embedding, cuis, entities,
                    triage_skipped, triage_reason, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15::vector, $16::vector, $17, $18::jsonb,
                    $19, $20, $21::jsonb
                )
            """, records)
        
        logger.info(f"Inserted {len(records)} images for document {document_id}")
        return len(records)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    async def get_by_document(
        self,
        document_id: UUID,
        page_number: int = None,
        include_decorative: bool = False,
        include_embedding: bool = False
    ) -> List[Dict[str, Any]]:
        """Get images for a document."""
        columns = """
            i.id, i.document_id, i.storage_path as file_path, i.original_filename as file_name,
            i.image_type, i.is_decorative, i.caption as vlm_caption
        """
        if include_embedding:
            columns += ", i.clip_embedding as embedding"
        
        conditions = ["i.document_id = $1"]
        params = [document_id]
        param_idx = 2

        if page_number is not None:
            # Note: page_number filter not available in current schema (uses page_id UUID)
            pass

        if not include_decorative:
            conditions.append("NOT i.is_decorative")

        query = f"""
            SELECT {columns}
            FROM images i
            WHERE {' AND '.join(conditions)}
            ORDER BY i.created_at
        """
        
        rows = await self.db.fetch(query, *params)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_by_type(
        self,
        image_type: str,
        document_id: UUID = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get images by type (MRI_CT, SURGICAL_PHOTO, etc.)."""
        if document_id:
            query = """
                SELECT id, document_id, storage_path as file_path, page_number, image_type, caption as vlm_caption
                FROM images
                WHERE image_type = $1 AND document_id = $2 AND NOT is_decorative
                ORDER BY page_number
                LIMIT $3
            """
            rows = await self.db.fetch(query, image_type, document_id, limit)
        else:
            query = """
                SELECT id, document_id, storage_path as file_path, page_number, image_type, caption as vlm_caption
                FROM images
                WHERE image_type = $1 AND NOT is_decorative
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await self.db.fetch(query, image_type, limit)
        
        return [self._to_entity(dict(row)) for row in rows]
    
    async def get_with_captions(
        self,
        document_id: UUID = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get images that have VLM captions."""
        if document_id:
            query = """
                SELECT id, document_id, storage_path as file_path, page_number, image_type,
                       caption as vlm_caption, 0.0 as vlm_confidence
                FROM images
                WHERE document_id = $1 AND caption IS NOT NULL
                ORDER BY page_number
                LIMIT $2
            """
            rows = await self.db.fetch(query, document_id, limit)
        else:
            query = """
                SELECT id, document_id, storage_path as file_path, page_number, image_type,
                       caption as vlm_caption, 0.0 as vlm_confidence
                FROM images
                WHERE caption IS NOT NULL
                ORDER BY created_at DESC
                LIMIT $1
            """
            rows = await self.db.fetch(query, limit)
        
        return [self._to_entity(dict(row)) for row in rows]
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    async def search_by_visual_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_ids: List[UUID] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search images by visual (BiomedCLIP) embedding.
        
        Args:
            query_embedding: 512d visual query vector
            top_k: Number of results
            document_ids: Filter by documents
            min_similarity: Minimum similarity threshold
        """
        from src.database.connection import DatabaseConnection
        embedding_str = DatabaseConnection._encode_vector(query_embedding)
        
        conditions = ["embedding IS NOT NULL", "NOT is_decorative"]
        params = [embedding_str, top_k]
        
        if document_ids:
            conditions.append("document_id = ANY($3)")
            params.append(document_ids)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT *,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM images
            WHERE {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        
        rows = await self.db.fetch(query, *params)
        
        results = []
        for row in rows:
            if row['similarity'] >= min_similarity:
                results.append(self._to_entity(dict(row)))
        
        return results
    
    async def search_by_caption_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_ids: List[UUID] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search images by caption (text) embedding.
        
        This allows finding images based on text queries by searching
        the 1024d caption embeddings.
        
        Args:
            query_embedding: 1024d text query vector
            top_k: Number of results
            document_ids: Filter by documents
            min_similarity: Minimum similarity threshold
        """
        from src.database.connection import DatabaseConnection
        embedding_str = DatabaseConnection._encode_vector(query_embedding)
        
        conditions = ["caption_embedding IS NOT NULL", "NOT is_decorative"]
        params = [embedding_str, top_k]
        
        if document_ids:
            conditions.append("document_id = ANY($3)")
            params.append(document_ids)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT *,
                   1 - (caption_embedding <=> $1::vector) AS similarity
            FROM images
            WHERE {where_clause}
            ORDER BY caption_embedding <=> $1::vector
            LIMIT $2
        """
        
        rows = await self.db.fetch(query, *params)
        
        results = []
        for row in rows:
            if row['similarity'] >= min_similarity:
                results.append(self._to_entity(dict(row)))
        
        return results
    
    async def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: search images by text query using caption embeddings.
        """
        return await self.search_by_caption_embedding(query_embedding, top_k, **kwargs)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_statistics(self, document_id: UUID = None) -> Dict[str, Any]:
        """Get image statistics."""
        if document_id:
            query = """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE clip_embedding IS NOT NULL) as with_visual_embedding,
                    COUNT(*) FILTER (WHERE caption IS NOT NULL) as with_caption,
                    COUNT(*) FILTER (WHERE is_decorative) as decorative,
                    COUNT(DISTINCT image_type) as unique_types
                FROM images
                WHERE document_id = $1
            """
            row = await self.db.fetchrow(query, document_id)
        else:
            query = """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE clip_embedding IS NOT NULL) as with_visual_embedding,
                    COUNT(*) FILTER (WHERE caption IS NOT NULL) as with_caption,
                    COUNT(*) FILTER (WHERE is_decorative) as decorative,
                    COUNT(DISTINCT image_type) as unique_types
                FROM images
            """
            row = await self.db.fetchrow(query)

        return dict(row) if row else {}
    
    async def get_type_distribution(
        self,
        document_id: UUID = None
    ) -> Dict[str, int]:
        """Get distribution of image types."""
        if document_id:
            query = """
                SELECT image_type, COUNT(*) as count
                FROM images
                WHERE document_id = $1 AND image_type IS NOT NULL
                GROUP BY image_type
                ORDER BY count DESC
            """
            rows = await self.db.fetch(query, document_id)
        else:
            query = """
                SELECT image_type, COUNT(*) as count
                FROM images
                WHERE image_type IS NOT NULL
                GROUP BY image_type
                ORDER BY count DESC
            """
            rows = await self.db.fetch(query)
        
        return {row['image_type']: row['count'] for row in rows}
    
    async def get_triage_statistics(
        self,
        document_id: UUID = None
    ) -> Dict[str, Any]:
        """Get visual triage statistics."""
        # Note: triage_skipped column not in current schema - returning total count only
        if document_id:
            query = """
                SELECT COUNT(*) as total
                FROM images
                WHERE document_id = $1
            """
            row = await self.db.fetchrow(query, document_id)
        else:
            query = """
                SELECT COUNT(*) as total
                FROM images
            """
            row = await self.db.fetchrow(query)

        total = row['total'] if row else 0
        return {
            'total': total,
            'skipped': 0,
            'processed': total,
            'skip_rate': 0,
            'process_rate': 1.0 if total > 0 else 0
        }

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all images for a document."""
        query = "DELETE FROM images WHERE document_id = $1"
        result = await self.db.execute(query, document_id)
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted {count} images for document {document_id}")
        return count
