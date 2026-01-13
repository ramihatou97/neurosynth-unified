"""
NeuroSynth Unified - Document Repository
=========================================

Repository for document CRUD operations.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from uuid import UUID
from datetime import datetime
import json

from src.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """
    Repository for document operations.
    
    Documents are the top-level container for chunks and images
    extracted from PDF files.
    """
    
    @property
    def table_name(self) -> str:
        return "documents"

    @property
    def updatable_columns(self) -> Set[str]:
        return {'title', 'file_path', 'total_pages', 'total_chunks',
                'total_images', 'specialty', 'authority_score',
                'metadata', 'status', 'error_message'}

    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to document dict."""
        # Note: Schema uses file_path, not source_path
        # Handle metadata - can be dict (from JSONB) or None
        metadata = row.get('metadata')
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        elif metadata is None:
            metadata = {}

        return {
            'id': row['id'],
            'source_path': row.get('file_path', ''),  # Map DB file_path to API source_path
            'title': row.get('title'),
            'total_pages': row.get('total_pages', 0),
            'total_chunks': row.get('total_chunks', 0),
            'total_images': row.get('total_images', 0),
            'created_at': row.get('created_at'),
            'updated_at': row.get('updated_at'),
            'specialty': row.get('specialty'),
            'authority_score': row.get('authority_score'),
            'metadata': metadata
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert document dict to database record."""
        metadata = entity.get('metadata', {}) or {}
        
        # Ensure processing_time_seconds is in metadata
        if 'processing_time_seconds' in entity:
            metadata['processing_time_seconds'] = entity['processing_time_seconds']
            
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        
        return {
            'id': entity.get('id'),
            'source_path': entity['source_path'],
            'title': entity.get('title'),
            'total_pages': entity.get('total_pages', 0),
            # AVG: Removed processing_time_seconds as it's not in schema
            # 'processing_time_seconds': entity.get('processing_time_seconds'),
            'metadata': metadata
        }
    
    # =========================================================================
    # Custom Queries
    # =========================================================================
    
    async def create_from_path(
        self,
        source_path: str,
        title: str = None,
        total_pages: int = 0,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """Create a document from a file path."""
        from uuid import uuid4
        
        entity = {
            'id': uuid4(),
            'source_path': source_path,
            'title': title,
            'total_pages': total_pages,
            'metadata': metadata or {}
        }
        
        return await self.create(entity)
    
    async def get_by_source_path(self, source_path: str) -> Optional[Dict[str, Any]]:
        """Find document by source file path."""
        # DB column is file_path, not source_path
        return await self.find_one_by({'file_path': source_path})
    
    async def get_with_stats(self, id: UUID) -> Optional[Dict[str, Any]]:
        """Get document with chunk and image statistics."""
        # Database has both old and new column names; use correct ones:
        # - Images: 'embedding' column has 268 embeddings, 'vlm_caption' has 180 captions
        # - Chunks: 'embedding' column has 1346 embeddings
        query = """
            SELECT
                d.id,
                d.file_path,
                d.title,
                d.created_at,
                d.updated_at,
                d.specialty,
                d.authority_score,
                d.metadata,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as total_chunks,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id AND (text_embedding IS NOT NULL OR embedding IS NOT NULL)) as chunks_with_embedding,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id) as total_images,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id AND (embedding IS NOT NULL OR clip_embedding IS NOT NULL)) as images_with_embedding,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id AND vlm_caption IS NOT NULL) as images_with_caption
            FROM documents d
            WHERE d.id = $1
        """

        row = await self.db.fetchrow(query, id)
        if not row:
            return None

        doc = self._to_entity(dict(row))
        
        # Parse metadata if it's a string (asyncpg usu handles jsonb as dict, but just in case)
        if isinstance(doc.get('metadata'), str):
            try:
                doc['metadata'] = json.loads(doc['metadata'])
            except:
                pass

        doc['stats'] = {
            'chunks': {
                'total': row.get('total_chunks', 0),
                'with_embedding': row.get('chunks_with_embedding', 0),
                'with_cuis': 0
            },
            'images': {
                'total': row.get('total_images', 0),
                'with_embedding': row.get('images_with_embedding', 0),
                'with_caption': row.get('images_with_caption', 0)
            }
        }
        
        return doc
    
    async def list_with_counts(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents with chunk and image counts."""
        # Schema: file_path (not source_path), count pages from pages table
        query = """
            SELECT
                d.id,
                d.file_path,
                d.title,
                d.specialty,
                d.authority_score,
                (SELECT COUNT(*) FROM pages WHERE document_id = d.id) as total_pages,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as total_chunks,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id) as total_images,
                d.created_at,
                d.updated_at
            FROM documents d
            WHERE d.deleted_at IS NULL
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
        """

        rows = await self.db.fetch(query, limit, offset)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def update_stats(self, id: UUID) -> None:
        """Manually update document statistics."""
        # AVG: schema doesn't have cached stats columns, skipping update.
        # Stats are calculated via views or subqueries now.
        query = """
            UPDATE documents SET
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """
        await self.db.execute(query, id)
    
    async def delete_with_cascade(self, id: UUID) -> bool:
        """Delete document and all related chunks, images, links."""
        # Due to CASCADE constraints, this should work automatically
        return await self.delete(id)

    async def list_all_ids(self) -> List[UUID]:
        """Get all document IDs."""
        query = "SELECT id FROM documents ORDER BY created_at DESC"
        rows = await self.db.fetch(query)
        return [row['id'] for row in rows]
