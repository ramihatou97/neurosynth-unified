"""
NeuroSynth Unified - Image Repository
======================================

Repository for image CRUD and vector search operations.
Supports dual embeddings: visual (BiomedCLIP 512d) and caption (Voyage 1024d).
"""

import logging
import os
from typing import List, Optional, Dict, Any, Set
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

    @property
    def updatable_columns(self) -> Set[str]:
        return {'caption', 'vlm_caption', 'caption_summary', 'alt_text',
                'image_type', 'image_subtype', 'is_decorative',
                'quality_score', 'clip_embedding', 'caption_embedding',
                'cuis', 'metadata'}

    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to image dict."""
        # DB columns: storage_path, caption, clip_embedding, original_filename
        # API expects: file_path, vlm_caption, embedding, file_name
        # DB stores: output/images/{job_id}/{filename}
        # API serves from: ./output/images/ so we need just {job_id}/{filename}
        storage_path = row.get('storage_path') or row.get('file_path') or ''
        # Strip 'output/images/' prefix if present
        if storage_path and storage_path.startswith('output/images/'):
            storage_path = storage_path[len('output/images/'):]
        return {
            'id': row['id'],
            'document_id': row['document_id'],
            'file_path': storage_path,
            'file_name': row.get('original_filename') or row.get('file_name'),
            'thumbnail_path': row.get('thumbnail_path'),
            'width': row.get('width'),
            'height': row.get('height'),
            'format': row.get('format'),
            'page_id': row.get('page_id'),
            'image_type': row.get('image_type'),
            'image_subtype': row.get('image_subtype'),
            'is_decorative': row.get('is_decorative', False),
            'vlm_caption': row.get('caption') or row.get('vlm_caption'),
            'caption_summary': row.get('caption_summary'),  # Pre-computed brief summary
            'alt_text': row.get('alt_text'),
            'surrounding_text': row.get('surrounding_text'),
            'figure_number': row.get('figure_number'),
            'embedding': row.get('clip_embedding') or row.get('embedding'),  # 512d visual
            'quality_score': row.get('quality_score'),
            'created_at': row.get('created_at'),
            # Search result fields
            'similarity': row.get('similarity')
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image dict to database record."""
        from src.database.connection import DatabaseConnection

        # Handle embedding - DB column is clip_embedding
        embedding = entity.get('embedding') or entity.get('clip_embedding')
        if embedding is not None:
            if isinstance(embedding, (np.ndarray, list)):
                embedding = DatabaseConnection._encode_vector(embedding)

        return {
            'id': entity.get('id'),
            'document_id': entity['document_id'],
            'storage_path': entity.get('file_path') or entity.get('storage_path', ''),
            'original_filename': entity.get('file_name') or entity.get('original_filename'),
            'thumbnail_path': entity.get('thumbnail_path'),
            'width': entity.get('width'),
            'height': entity.get('height'),
            'format': entity.get('format'),
            'page_id': entity.get('page_id'),
            'image_type': entity.get('image_type'),
            'image_subtype': entity.get('image_subtype'),
            'is_decorative': entity.get('is_decorative', False),
            'caption': entity.get('vlm_caption') or entity.get('caption'),
            'alt_text': entity.get('alt_text'),
            'surrounding_text': entity.get('surrounding_text'),
            'figure_number': entity.get('figure_number'),
            'quality_score': entity.get('quality_score'),
            'clip_embedding': embedding
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
                img.get('file_path') or img.get('storage_path', ''),
                img.get('original_filename') or img.get('file_name'),
                img.get('image_type'),
                img.get('image_subtype'),
                img.get('is_decorative', False),
                img.get('vlm_caption') or img.get('caption'),
                img.get('alt_text'),
                embedding,
                img.get('caption_summary')  # Pre-computed brief summary
            ))

        async with self.db.transaction() as conn:
            await conn.executemany("""
                INSERT INTO images (
                    id, document_id, storage_path, original_filename,
                    image_type, image_subtype, is_decorative, caption,
                    alt_text, clip_embedding, caption_summary
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11
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
        include_embedding: bool = False,
        limit: int = None,
        offset: int = None
    ) -> List[Dict[str, Any]]:
        """Get images for a document with optional pagination."""
        columns = """
            i.id, i.document_id, COALESCE(i.storage_path, i.file_path) as file_path, i.original_filename as file_name,
            i.image_type, i.is_decorative, i.vlm_caption, i.caption_summary
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

        # Add pagination if specified (database-level for efficiency)
        if limit is not None:
            params.append(limit)
            query += f" LIMIT ${len(params)}"
        if offset is not None:
            params.append(offset)
            query += f" OFFSET ${len(params)}"

        rows = await self.db.fetch(query, *params)
        return [self._to_entity(dict(row)) for row in rows]

    async def count_by_document(
        self,
        document_id: UUID,
        page_number: int = None,
        include_decorative: bool = False
    ) -> int:
        """Count images for a document (for pagination)."""
        conditions = ["document_id = $1"]
        params = [document_id]

        if page_number is not None:
            # Note: page_number filter not available in current schema
            pass

        if not include_decorative:
            conditions.append("NOT is_decorative")

        query = f"""
            SELECT COUNT(*) FROM images
            WHERE {' AND '.join(conditions)}
        """

        count = await self.db.fetchval(query, *params)
        return count or 0
    
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

        # DB column is clip_embedding, not embedding
        conditions = ["clip_embedding IS NOT NULL", "NOT is_decorative"]
        params = [embedding_str, top_k]

        if document_ids:
            conditions.append("document_id = ANY($3)")
            params.append(document_ids)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT *,
                   1 - (clip_embedding <=> $1::vector) AS similarity
            FROM images
            WHERE {where_clause}
            ORDER BY clip_embedding <=> $1::vector
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
        the image_embedding column (Voyage text embedding of caption).

        Args:
            query_embedding: 1024d text query vector
            top_k: Number of results
            document_ids: Filter by documents
            min_similarity: Minimum similarity threshold
        """
        from src.database.connection import DatabaseConnection
        embedding_str = DatabaseConnection._encode_vector(query_embedding)

        # DB column is image_embedding (Voyage embedding of caption text)
        conditions = ["image_embedding IS NOT NULL", "NOT is_decorative"]
        params = [embedding_str, top_k]

        if document_ids:
            conditions.append("document_id = ANY($3)")
            params.append(document_ids)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT *,
                   1 - (image_embedding <=> $1::vector) AS similarity
            FROM images
            WHERE {where_clause}
            ORDER BY image_embedding <=> $1::vector
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

    async def delete_many(self, ids: List[UUID]) -> int:
        """Delete multiple images by ID."""
        if not ids:
            return 0
        query = "DELETE FROM images WHERE id = ANY($1::uuid[])"
        result = await self.db.execute(query, [str(id) for id in ids])
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted {count} images")
        return count

    async def list_all_ids(self) -> List[UUID]:
        """Get all image IDs."""
        query = "SELECT id FROM images ORDER BY created_at DESC"
        rows = await self.db.fetch(query)
        return [row['id'] for row in rows]

    async def delete_all(self) -> int:
        """Delete all images."""
        result = await self.db.execute("DELETE FROM images")
        count = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted all images: {count} total")
        return count
