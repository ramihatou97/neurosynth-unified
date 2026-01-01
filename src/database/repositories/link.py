"""
NeuroSynth Unified - Link Repository
=====================================

Repository for chunk-image link operations.
Links represent relationships between text chunks and images
discovered through proximity, semantic similarity, and CUI matching.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import json

from src.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class LinkRepository(BaseRepository):
    """
    Repository for chunk-image link operations.
    
    Links are created by Phase 1's TriPassLinker using:
    - Proximity: Same page or adjacent pages
    - Semantic: Embedding similarity between chunk and caption
    - CUI Match: Shared UMLS concepts
    """
    
    @property
    def table_name(self) -> str:
        return "links"
    
    def _to_entity(self, row: dict) -> Dict[str, Any]:
        """Convert database row to link dict."""
        return {
            'id': row['id'],
            'chunk_id': row['chunk_id'],
            'image_id': row['image_id'],
            'link_type': row['link_type'],
            'score': row['score'],
            'proximity_score': row.get('proximity_score'),
            'semantic_score': row.get('semantic_score'),
            'cui_overlap_score': row.get('cui_overlap_score'),
            'created_at': row.get('created_at'),
            'metadata': row.get('metadata', {})
        }
    
    def _to_record(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert link dict to database record."""
        metadata = entity.get('metadata', {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        
        return {
            'id': entity.get('id'),
            'chunk_id': entity['chunk_id'],
            'image_id': entity['image_id'],
            'link_type': entity.get('link_type', 'unknown'),
            'score': entity.get('score', 0.0),
            'proximity_score': entity.get('proximity_score'),
            'semantic_score': entity.get('semantic_score'),
            'cui_overlap_score': entity.get('cui_overlap_score'),
            'metadata': metadata
        }
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def create_many(
        self,
        links: List[Dict[str, Any]],
        chunk_id_map: Dict[str, UUID] = None,
        image_id_map: Dict[str, UUID] = None
    ) -> int:
        """
        Batch insert links with optional ID remapping.
        
        Args:
            links: List of link dicts (from Phase 1 LinkResult objects)
            chunk_id_map: Map old chunk IDs to new UUIDs
            image_id_map: Map old image IDs to new UUIDs
        
        Returns:
            Number of links inserted
        """
        from uuid import uuid4
        
        if not links:
            return 0
        
        records = []
        for link in links:
            # Get chunk and image IDs
            chunk_id = link.get('chunk_id')
            image_id = link.get('image_id')
            
            # Remap IDs if maps provided
            if chunk_id_map and str(chunk_id) in chunk_id_map:
                chunk_id = chunk_id_map[str(chunk_id)]
            
            if image_id_map and str(image_id) in image_id_map:
                image_id = image_id_map[str(image_id)]
            
            # Skip if IDs couldn't be mapped
            if chunk_id is None or image_id is None:
                continue
            
            # Prepare metadata
            # Add scores to metadata
            if link.get('proximity_score') is not None:
                metadata['proximity_score'] = link.get('proximity_score')
            if link.get('semantic_score') is not None:
                metadata['semantic_score'] = link.get('semantic_score')
            if link.get('cui_overlap_score') is not None:
                metadata['cui_overlap_score'] = link.get('cui_overlap_score')

            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)

            records.append((
                uuid4(),  # New ID
                chunk_id,
                image_id,
                link.get('link_type') or link.get('match_type', 'unknown'),
                link.get('score') or link.get('strength', 0.0),
                metadata
            ))
        
        if not records:
            return 0
        
        async with self.db.transaction() as conn:
            await conn.executemany("""
                INSERT INTO links (
                    id, chunk_id, image_id, link_type, score, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (chunk_id, image_id) DO UPDATE SET
                    score = EXCLUDED.score,
                    link_type = EXCLUDED.link_type
            """, records)
        
        logger.info(f"Inserted {len(records)} links")
        return len(records)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    async def get_by_chunk(
        self,
        chunk_id: UUID,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get all links for a chunk."""
        query = """
            SELECT l.*, 
                   i.file_path as image_file_path,
                   i.vlm_caption as image_caption,
                   i.image_type
            FROM links l
            JOIN images i ON l.image_id = i.id
            WHERE l.chunk_id = $1 AND l.score >= $2
            ORDER BY l.score DESC
        """
        rows = await self.db.fetch(query, chunk_id, min_score)
        
        results = []
        for row in rows:
            link = self._to_entity(dict(row))
            link['image'] = {
                'file_path': row['image_file_path'],
                'caption': row['image_caption'],
                'image_type': row['image_type']
            }
            results.append(link)
        
        return results
    
    async def get_by_image(
        self,
        image_id: UUID,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get all links for an image."""
        query = """
            SELECT l.*,
                   c.content as chunk_content,
                   c.chunk_type,
                   c.page_number as chunk_page
            FROM links l
            JOIN chunks c ON l.chunk_id = c.id
            WHERE l.image_id = $1 AND l.score >= $2
            ORDER BY l.score DESC
        """
        rows = await self.db.fetch(query, image_id, min_score)
        
        results = []
        for row in rows:
            link = self._to_entity(dict(row))
            link['chunk'] = {
                'content': row['chunk_content'],
                'chunk_type': row['chunk_type'],
                'page_number': row['chunk_page']
            }
            results.append(link)
        
        return results
    
    async def get_images_for_chunks(
        self,
        chunk_ids: List[UUID],
        min_score: float = 0.5,
        max_images_per_chunk: int = 3
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """
        Get linked images for multiple chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            min_score: Minimum link score
            max_images_per_chunk: Maximum images per chunk
        
        Returns:
            Dict mapping chunk_id to list of linked images
        """
        if not chunk_ids:
            return {}
        
        query = """
            SELECT 
                l.chunk_id,
                l.image_id,
                l.link_type,
                l.score,
                i.file_path,
                i.vlm_caption,
                i.image_type,
                i.page_number
            FROM links l
            JOIN images i ON l.image_id = i.id
            WHERE l.chunk_id = ANY($1) AND l.score >= $2
            ORDER BY l.chunk_id, l.score DESC
        """
        
        rows = await self.db.fetch(query, chunk_ids, min_score)
        
        # Group by chunk with limit
        result = {cid: [] for cid in chunk_ids}
        counts = {cid: 0 for cid in chunk_ids}
        
        for row in rows:
            chunk_id = row['chunk_id']
            if counts[chunk_id] < max_images_per_chunk:
                result[chunk_id].append({
                    'image_id': row['image_id'],
                    'file_path': row['file_path'],
                    'caption': row['vlm_caption'],
                    'image_type': row['image_type'],
                    'page_number': row['page_number'],
                    'link_type': row['link_type'],
                    'link_score': row['score']
                })
                counts[chunk_id] += 1
        
        return result
    
    async def get_chunks_for_images(
        self,
        image_ids: List[UUID],
        min_score: float = 0.5,
        max_chunks_per_image: int = 3
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """Get linked chunks for multiple images."""
        if not image_ids:
            return {}
        
        query = """
            SELECT 
                l.image_id,
                l.chunk_id,
                l.link_type,
                l.score,
                c.content,
                c.chunk_type,
                c.page_number
            FROM links l
            JOIN chunks c ON l.chunk_id = c.id
            WHERE l.image_id = ANY($1) AND l.score >= $2
            ORDER BY l.image_id, l.score DESC
        """
        
        rows = await self.db.fetch(query, image_ids, min_score)
        
        result = {iid: [] for iid in image_ids}
        counts = {iid: 0 for iid in image_ids}
        
        for row in rows:
            image_id = row['image_id']
            if counts[image_id] < max_chunks_per_image:
                result[image_id].append({
                    'chunk_id': row['chunk_id'],
                    'content': row['content'],
                    'chunk_type': row['chunk_type'],
                    'page_number': row['page_number'],
                    'link_type': row['link_type'],
                    'link_score': row['score']
                })
                counts[image_id] += 1
        
        return result
    
    async def get_by_link_type(
        self,
        link_type: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get links by type (proximity, semantic, cui_match)."""
        query = """
            SELECT *
            FROM links
            WHERE link_type = $1
            ORDER BY score DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, link_type, limit)
        return [self._to_entity(dict(row)) for row in rows]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get link statistics."""
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT chunk_id) as unique_chunks,
                COUNT(DISTINCT image_id) as unique_images,
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score
            FROM links
        """
        row = await self.db.fetchrow(query)
        return dict(row) if row else {}
    
    async def get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of link types."""
        query = """
            SELECT link_type, COUNT(*) as count
            FROM links
            GROUP BY link_type
            ORDER BY count DESC
        """
        rows = await self.db.fetch(query)
        return {row['link_type']: row['count'] for row in rows}
    
    async def get_score_distribution(self, buckets: int = 10) -> List[Dict[str, Any]]:
        """Get distribution of link scores in buckets."""
        bucket_size = 1.0 / buckets
        
        query = f"""
            SELECT 
                FLOOR(score / {bucket_size}) * {bucket_size} as bucket_start,
                FLOOR(score / {bucket_size}) * {bucket_size} + {bucket_size} as bucket_end,
                COUNT(*) as count
            FROM links
            GROUP BY FLOOR(score / {bucket_size})
            ORDER BY bucket_start
        """
        rows = await self.db.fetch(query)
        
        return [
            {
                'range': f"{row['bucket_start']:.1f}-{row['bucket_end']:.1f}",
                'count': row['count']
            }
            for row in rows
        ]
