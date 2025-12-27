"""
NeuroSynth Unified - Document Repository
=========================================

Repository for document CRUD operations.
"""

import logging
from typing import List, Optional, Dict, Any
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
    
    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to document dict."""
        return {
            'id': row['id'],
            'source_path': row['source_path'],
            'title': row.get('title'),
            'total_pages': row.get('total_pages', 0),
            'total_chunks': row.get('total_chunks', 0),
            'total_images': row.get('total_images', 0),
            'processing_time_seconds': row.get('processing_time_seconds'),
            'created_at': row.get('created_at'),
            'updated_at': row.get('updated_at'),
            'metadata': row.get('metadata', {})
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert document dict to database record."""
        metadata = entity.get('metadata', {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        
        return {
            'id': entity.get('id'),
            'source_path': entity['source_path'],
            'title': entity.get('title'),
            'total_pages': entity.get('total_pages', 0),
            'processing_time_seconds': entity.get('processing_time_seconds'),
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
        return await self.find_one_by({'file_path': source_path})
    
    async def get_with_stats(self, id: UUID) -> Optional[Dict[str, Any]]:
        """Get document with chunk and image statistics."""
        query = """
            SELECT
                d.id,
                d.file_path as source_path,
                d.title,
                d.created_at,
                d.updated_at,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as total_chunks,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id AND embedding IS NOT NULL) as chunks_with_embedding,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id) as total_images,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id AND image_embedding IS NOT NULL) as images_with_embedding,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id AND caption IS NOT NULL) as images_with_caption
            FROM documents d
            WHERE d.id = $1
        """

        row = await self.db.fetchrow(query, id)
        if not row:
            return None

        doc = self._to_entity(dict(row))
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
        query = """
            SELECT
                d.id,
                d.file_path as source_path,
                d.title,
                0 as total_pages,
                (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) as total_chunks,
                (SELECT COUNT(*) FROM images WHERE document_id = d.id) as total_images,
                d.created_at,
                d.updated_at
            FROM documents d
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
        """
        
        rows = await self.db.fetch(query, limit, offset)
        return [self._to_entity(dict(row)) for row in rows]
    
    async def update_stats(self, id: UUID) -> None:
        """Manually update document statistics."""
        query = """
            UPDATE documents SET
                total_chunks = (SELECT COUNT(*) FROM chunks WHERE document_id = $1),
                total_images = (SELECT COUNT(*) FROM images WHERE document_id = $1),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """
        await self.db.execute(query, id)
    
    async def delete_with_cascade(self, id: UUID) -> bool:
        """Delete document and all related chunks, images, links."""
        # Due to CASCADE constraints, this should work automatically
        return await self.delete(id)
