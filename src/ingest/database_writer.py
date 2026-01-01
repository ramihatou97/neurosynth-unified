"""
NeuroSynth Unified - Pipeline Database Writer
==============================================

Adapter to write Phase 1 pipeline output directly to PostgreSQL.
Integrates with the existing pipeline while maintaining backward compatibility.

Usage:
    from src.ingest.database_writer import PipelineDatabaseWriter
    
    # Create writer
    writer = PipelineDatabaseWriter(connection_string)
    await writer.connect()
    
    # Write pipeline result
    doc_id = await writer.write_pipeline_result(result, source_path)
    
    # Or use as pipeline callback
    pipeline = NeuroIngestPipeline(
        config=config,
        on_complete=writer.write_pipeline_result
    )
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4
from pathlib import Path
from datetime import datetime
import json
import pickle

import numpy as np

from src.database import (
    DatabaseConnection,
    init_database,
    close_database,
    get_repositories,
    Repositories
)
from src.database.repositories.entity import EntityRepository

logger = logging.getLogger(__name__)


class PipelineDatabaseWriter:
    """
    Writes Phase 1 pipeline output to PostgreSQL.
    
    Handles:
    - Document creation
    - Chunk batch insert with embeddings
    - Image batch insert with dual embeddings
    - Link creation with ID remapping
    - Optional file export for backup
    
    Features:
    - Transaction support (all-or-nothing)
    - ID mapping (Phase 1 IDs → PostgreSQL UUIDs)
    - Progress callbacks
    - Error recovery
    """
    
    def __init__(
        self,
        connection_string: str = None,
        db: DatabaseConnection = None,
        export_files: bool = False,
        export_dir: Path = None
    ):
        """
        Initialize database writer.
        
        Args:
            connection_string: PostgreSQL connection string
            db: Existing DatabaseConnection (alternative to connection_string)
            export_files: Also export to pkl/json files
            export_dir: Directory for file exports
        """
        self.connection_string = connection_string
        self._db = db
        self._repos: Optional[Repositories] = None
        self._connected = False
        self._owns_connection = False  # Track if we created the connection

        # File export options
        self.export_files = export_files
        self.export_dir = Path(export_dir) if export_dir else None

        # ID mappings (Phase 1 ID → PostgreSQL UUID)
        self._chunk_id_map: Dict[str, UUID] = {}
        self._image_id_map: Dict[str, UUID] = {}
    
    async def connect(self) -> None:
        """Connect to database."""
        if self._connected:
            return

        if self._db is None:
            if not self.connection_string:
                raise ValueError("Either connection_string or db must be provided")
            self._db = await init_database(self.connection_string)
            self._owns_connection = True  # We created it, we own it

        self._repos = get_repositories(self._db)
        self._connected = True
        logger.info("PipelineDatabaseWriter connected to database")
    
    async def close(self) -> None:
        """Release database reference without closing shared pool.

        The DatabaseConnection is a singleton managed by the API's lifespan handler.
        We only release our reference, never close the pool.
        """
        if self._connected:
            self._db = None
            self._repos = None
            self._connected = False
            logger.info("PipelineDatabaseWriter released database reference")
    
    @property
    def repos(self) -> Repositories:
        """Get repositories."""
        if not self._repos:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._repos
    
    # =========================================================================
    # Main Write Method
    # =========================================================================
    
    async def write_pipeline_result(
        self,
        result,  # PipelineResult from Phase 1
        source_path: str = None,
        title: str = None,
        on_progress: callable = None
    ) -> UUID:
        """
        Write complete pipeline result to database.
        
        Args:
            result: PipelineResult object from Phase 1 pipeline
            source_path: Override source path (uses result.source_path if not provided)
            title: Document title
            on_progress: Progress callback(stage, current, total)
        
        Returns:
            Document UUID
        
        Raises:
            Exception if write fails (transaction rolled back)
        """
        if not self._connected:
            await self.connect()
        
        # Reset ID mappings
        self._chunk_id_map = {}
        self._image_id_map = {}
        
        # Extract data from result
        source_path = source_path or getattr(result, 'source_path', 'unknown')
        chunks = getattr(result, 'chunks', []) or []
        images = getattr(result, 'images', []) or []
        links = getattr(result, 'links', []) or []
        
        # Get metadata
        processing_time = getattr(result, 'processing_time_seconds', None)
        total_pages = getattr(result, 'total_pages', 0)
        
        logger.info(f"Writing pipeline result: {len(chunks)} chunks, {len(images)} images, {len(links)} links")
        
        try:
            # Step 1: Create document
            if on_progress:
                on_progress("document", 0, 1)
            
            doc_id = await self._create_document(
                source_path=source_path,
                title=title,
                total_pages=total_pages,
                processing_time=processing_time,
                result=result
            )
            
            if on_progress:
                on_progress("document", 1, 1)
            
            # Step 2: Insert chunks
            if chunks:
                if on_progress:
                    on_progress("chunks", 0, len(chunks))

                await self._insert_chunks(doc_id, chunks, on_progress)

            # Step 2.5: Populate entities from chunk CUIs
            if chunks:
                if on_progress:
                    on_progress("entities", 0, 1)
                await self._populate_entities(chunks, on_progress)
                if on_progress:
                    on_progress("entities", 1, 1)

            # Step 3: Insert images
            if images:
                if on_progress:
                    on_progress("images", 0, len(images))
                
                await self._insert_images(doc_id, images, on_progress)
            
            # Step 4: Insert links (with ID remapping)
            if links:
                if on_progress:
                    on_progress("links", 0, len(links))
                
                await self._insert_links(links, on_progress)
            
            # Step 5: Update document stats
            await self.repos.documents.update_stats(doc_id)
            
            # Step 6: Optional file export
            if self.export_files and self.export_dir:
                await self._export_files(doc_id, result)
            
            logger.info(f"Successfully wrote document {doc_id} to database")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to write pipeline result: {e}")
            raise
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    async def _create_document(
        self,
        source_path: str,
        title: str,
        total_pages: int,
        processing_time: float,
        result
    ) -> UUID:
        """Create document record."""
        # Build metadata from result
        metadata = {
            'phase1_document_id': getattr(result, 'document_id', None),
            'text_embedding_dim': getattr(result, 'text_embedding_dim', 1024),
            'image_embedding_dim': getattr(result, 'image_embedding_dim', 512),
            'text_embedding_provider': getattr(result, 'text_embedding_provider', 'voyage'),
            'chunk_count': len(getattr(result, 'chunks', []) or []),
            'image_count': len(getattr(result, 'images', []) or []),
            'link_count': len(getattr(result, 'links', []) or [])
        }
        
        # Check if document already exists
        existing = await self.repos.documents.get_by_source_path(source_path)
        if existing:
            logger.warning(f"Document already exists for {source_path}, updating...")
            # Could implement update logic here
            # For now, delete and recreate
            await self.repos.documents.delete_with_cascade(existing['id'])
        
        doc = await self.repos.documents.create({
            'id': uuid4(),
            'source_path': source_path,
            'title': title or Path(source_path).stem,
            'total_pages': total_pages,
            'processing_time_seconds': processing_time,
            'metadata': metadata
        })
        
        return doc['id']
    
    async def _insert_chunks(
        self,
        doc_id: UUID,
        chunks: List,
        on_progress: callable = None
    ) -> int:
        """Insert chunks and build ID mapping."""
        chunk_records = []
        
        for i, chunk in enumerate(chunks):
            # Extract data from chunk object
            chunk_data = self._extract_chunk_data(chunk)
            
            # Generate new UUID
            new_id = uuid4()
            old_id = str(chunk_data.get('id', i))
            self._chunk_id_map[old_id] = new_id
            
            chunk_data['id'] = new_id
            chunk_records.append(chunk_data)
            
            if on_progress and (i + 1) % 50 == 0:
                on_progress("chunks", i + 1, len(chunks))
        
        # Batch insert
        count = await self.repos.chunks.create_many_for_document(doc_id, chunk_records)
        
        if on_progress:
            on_progress("chunks", len(chunks), len(chunks))
        
        logger.info(f"Inserted {count} chunks")
        return count
    
    async def _insert_images(
        self,
        doc_id: UUID,
        images: List,
        on_progress: callable = None
    ) -> int:
        """Insert images and build ID mapping."""
        image_records = []
        
        for i, image in enumerate(images):
            # Extract data from image object
            image_data = self._extract_image_data(image)
            
            # Generate new UUID
            new_id = uuid4()
            old_id = str(image_data.get('id', i))
            self._image_id_map[old_id] = new_id
            
            image_data['id'] = new_id
            image_records.append(image_data)
            
            if on_progress and (i + 1) % 20 == 0:
                on_progress("images", i + 1, len(images))
        
        # Batch insert
        count = await self.repos.images.create_many_for_document(doc_id, image_records)
        
        if on_progress:
            on_progress("images", len(images), len(images))
        
        logger.info(f"Inserted {count} images")
        return count
    
    async def _insert_links(
        self,
        links: List,
        on_progress: callable = None
    ) -> int:
        """Insert links with ID remapping."""
        link_records = []
        
        for i, link in enumerate(links):
            link_data = self._extract_link_data(link)
            
            # Remap IDs
            old_chunk_id = str(link_data.get('chunk_id', ''))
            old_image_id = str(link_data.get('image_id', ''))
            
            if old_chunk_id in self._chunk_id_map and old_image_id in self._image_id_map:
                link_data['chunk_id'] = self._chunk_id_map[old_chunk_id]
                link_data['image_id'] = self._image_id_map[old_image_id]
                link_records.append(link_data)
        
        if not link_records:
            logger.warning("No links could be mapped - check ID consistency")
            return 0

        # Batch insert
        count = await self.repos.links.create_many(link_records)

        if on_progress:
            on_progress("links", len(links), len(links))

        logger.info(f"Inserted {count} links (from {len(links)} original)")
        return count

    async def _populate_entities(
        self,
        chunks: List,
        on_progress: callable = None
    ) -> int:
        """
        Extract unique entities from chunks and upsert to entities table.

        This populates the entities table from:
        1. UMLS CUIs found in chunks (when SciSpacy works)
        2. Regex-extracted entities (as fallback when no CUIs)

        This enables the Entities tab to show extracted medical concepts.
        """
        import hashlib

        # Create entity repository
        entity_repo = EntityRepository(self._db)

        # Collect all entities from chunks
        entities_data = []
        seen_regex_entities = set()  # Track unique regex entities

        for chunk in chunks:
            chunk_data = self._extract_chunk_data(chunk) if not isinstance(chunk, dict) else chunk
            cuis = chunk_data.get('cuis', [])
            entities = chunk_data.get('entities', [])

            # Build entity lookup from entities list if available
            entity_lookup = {}
            if entities:
                for e in entities:
                    if isinstance(e, dict):
                        cui = e.get('cui')
                        if cui:
                            entity_lookup[cui] = {
                                'name': e.get('text', cui),
                                'semantic_type': e.get('type') or e.get('semantic_type'),
                                'tui': e.get('tui')
                            }

            # Add UMLS entities (with real CUIs)
            for cui in cuis:
                if cui:
                    entity_info = entity_lookup.get(cui, {})
                    entities_data.append({
                        'cui': cui,
                        'name': entity_info.get('name', cui),
                        'semantic_type': entity_info.get('semantic_type'),
                        'tui': entity_info.get('tui'),
                        'chunk_count_increment': 1
                    })

            # FALLBACK: Add regex-extracted entities when no CUIs available
            # This ensures entities are shown even when SciSpacy is unavailable
            if not cuis and entities:
                for e in entities:
                    if isinstance(e, dict):
                        text = e.get('text', '').strip()
                        category = e.get('category') or e.get('type') or 'EXTRACTED'

                        if text and len(text) >= 2:
                            # Generate synthetic CUI from text hash
                            text_normalized = text.lower()
                            text_hash = hashlib.md5(text_normalized.encode()).hexdigest()[:8]
                            synthetic_cui = f"REGEX_{text_hash}"

                            # Track to avoid duplicates within same chunk
                            entity_key = (synthetic_cui, text_normalized)
                            if entity_key not in seen_regex_entities:
                                seen_regex_entities.add(entity_key)
                                entities_data.append({
                                    'cui': synthetic_cui,
                                    'name': text,
                                    'semantic_type': category,
                                    'tui': None,
                                    'chunk_count_increment': 1
                                })

        if not entities_data:
            logger.info("No entities found in chunks")
            return 0

        # Batch upsert
        count = await entity_repo.upsert_many(entities_data)

        logger.info(f"Populated {count} unique entities from {len(entities_data)} occurrences")
        return count

    # =========================================================================
    # Data Extraction
    # =========================================================================
    
    def _extract_chunk_data(self, chunk) -> Dict[str, Any]:
        """Extract data from SemanticChunk object."""
        if isinstance(chunk, dict):
            return chunk
        
        data = {}
        
        # ID
        data['id'] = getattr(chunk, 'id', None)
        
        # Content
        data['content'] = (
            getattr(chunk, 'content', None) or 
            getattr(chunk, 'text_content', None) or 
            ''
        )
        data['content_hash'] = getattr(chunk, 'content_hash', None)
        
        # Position
        data['page_number'] = getattr(chunk, 'page_number', None)
        data['chunk_index'] = (
            getattr(chunk, 'chunk_index', None) or 
            getattr(chunk, 'index', None)
        )
        data['start_char'] = getattr(chunk, 'start_char', None)
        data['end_char'] = getattr(chunk, 'end_char', None)
        
        # Classification
        chunk_type = getattr(chunk, 'chunk_type', None)
        if hasattr(chunk_type, 'value'):
            chunk_type = chunk_type.value
        data['chunk_type'] = chunk_type
        
        data['specialty'] = getattr(chunk, 'specialty', None)
        
        # Embedding
        embedding = getattr(chunk, 'text_embedding', None)
        if embedding is None:
            embedding = getattr(chunk, 'embedding', None)
            
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                data['embedding'] = embedding.tolist()
            else:
                data['embedding'] = list(embedding)
        
        # UMLS
        cuis = getattr(chunk, 'cuis', [])
        if isinstance(cuis, set):
            cuis = list(cuis)
        data['cuis'] = cuis or []
        
        # Entities
        entities = getattr(chunk, 'entities', [])
        if entities:
            data['entities'] = [
                e if isinstance(e, dict) else {
                    'text': getattr(e, 'text', ''),
                    'category': getattr(e, 'category', ''),
                    'type': getattr(e, 'category', ''),  # Alias for compatibility
                    'normalized': getattr(e, 'normalized', ''),
                    'confidence': getattr(e, 'confidence', 0.0),
                    'cui': getattr(e, 'cui', None)
                }
                for e in entities
            ]
        else:
            data['entities'] = []
        
        # Metadata
        data['metadata'] = getattr(chunk, 'metadata', {}) or {}

        # Summary (generated by ContentSummarizer)
        data['summary'] = getattr(chunk, 'summary', None)

        return data
    
    def _extract_image_data(self, image) -> Dict[str, Any]:
        """Extract data from ExtractedImage object."""
        if isinstance(image, dict):
            return image
        
        data = {}
        
        # ID
        data['id'] = getattr(image, 'id', None)
        
        # File info
        data['file_path'] = str(
            getattr(image, 'file_path', '') or 
            getattr(image, 'path', '')
        )
        data['file_name'] = getattr(image, 'file_name', None)
        data['content_hash'] = getattr(image, 'content_hash', None)
        data['width'] = getattr(image, 'width', None)
        data['height'] = getattr(image, 'height', None)
        data['format'] = getattr(image, 'format', None)
        
        # Position
        data['page_number'] = getattr(image, 'page_number', None)
        data['image_index'] = (
            getattr(image, 'image_index', None) or 
            getattr(image, 'index', None)
        )
        
        # Classification
        image_type = (
            getattr(image, 'image_type', None) or 
            getattr(image, 'vlm_image_type', None)
        )
        if hasattr(image_type, 'value'):
            image_type = image_type.value
        data['image_type'] = image_type
        data['is_decorative'] = getattr(image, 'is_decorative', False)
        
        # VLM caption
        data['vlm_caption'] = getattr(image, 'vlm_caption', None)
        data['vlm_confidence'] = getattr(image, 'vlm_confidence', None)
        data['caption_summary'] = getattr(image, 'caption_summary', None)
        
        # Visual embedding (512d BiomedCLIP)
        embedding = getattr(image, 'embedding', None)
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                data['embedding'] = embedding.tolist()
            else:
                data['embedding'] = list(embedding)
        
        # Caption embedding (1024d Voyage)
        caption_embedding = getattr(image, 'caption_embedding', None)
        if caption_embedding is not None:
            if isinstance(caption_embedding, np.ndarray):
                data['caption_embedding'] = caption_embedding.tolist()
            else:
                data['caption_embedding'] = list(caption_embedding)
        
        # UMLS
        data['cuis'] = list(getattr(image, 'cuis', []) or [])
        data['entities'] = list(getattr(image, 'entities', []) or [])
        
        # Triage info
        data['triage_skipped'] = getattr(image, 'triage_skipped', False)
        data['triage_reason'] = getattr(image, 'triage_reason', None)
        
        # Metadata
        data['metadata'] = getattr(image, 'metadata', {}) or {}
        
        return data
    
    def _extract_link_data(self, link) -> Dict[str, Any]:
        """Extract data from LinkResult object."""
        if isinstance(link, dict):
            return {
                'chunk_id': link.get('chunk_id'),
                'image_id': link.get('image_id'),
                'link_type': link.get('link_type', link.get('match_type', 'unknown')),
                'score': link.get('score', link.get('strength', 0.0)),
                'proximity_score': link.get('proximity_score'),
                'semantic_score': link.get('semantic_score'),
                'cui_overlap_score': link.get('cui_overlap_score'),
                'metadata': link.get('metadata', {})
            }
        
        link_type = getattr(link, 'link_type', None) or getattr(link, 'match_type', 'unknown')
        if hasattr(link_type, 'value'):
            link_type = link_type.value
            
        return {
            'chunk_id': getattr(link, 'chunk_id', None),
            'image_id': getattr(link, 'image_id', None),
            'link_type': link_type,
            'score': getattr(link, 'score', 0.0) or getattr(link, 'strength', 0.0),
            'proximity_score': getattr(link, 'proximity_score', None),
            'semantic_score': getattr(link, 'semantic_score', None),
            'cui_overlap_score': getattr(link, 'cui_overlap_score', None),
            'metadata': getattr(link, 'metadata', {}) or {}
        }
    
    # =========================================================================
    # File Export (Backward Compatibility)
    # =========================================================================
    
    async def _export_files(self, doc_id: UUID, result) -> Path:
        """Export to pkl/json files for backward compatibility."""
        export_path = self.export_dir / str(doc_id)
        export_path.mkdir(parents=True, exist_ok=True)
        
        chunks = getattr(result, 'chunks', []) or []
        images = getattr(result, 'images', []) or []
        links = getattr(result, 'links', []) or []
        
        # Export chunks
        with open(export_path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        
        # Export images
        with open(export_path / "images.pkl", "wb") as f:
            pickle.dump(images, f)
        
        # Export links
        links_data = [self._extract_link_data(l) for l in links]
        # Convert UUIDs to strings for JSON
        for l in links_data:
            if isinstance(l.get('chunk_id'), UUID):
                l['chunk_id'] = str(l['chunk_id'])
            if isinstance(l.get('image_id'), UUID):
                l['image_id'] = str(l['image_id'])
        
        with open(export_path / "links.json", "w") as f:
            json.dump(links_data, f, indent=2, default=str)
        
        # Export manifest
        manifest = {
            'document_id': str(doc_id),
            'source_path': getattr(result, 'source_path', 'unknown'),
            'chunk_count': len(chunks),
            'image_count': len(images),
            'link_count': len(links),
            'text_embedding_dim': getattr(result, 'text_embedding_dim', 1024),
            'image_embedding_dim': getattr(result, 'image_embedding_dim', 512),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(export_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Exported files to {export_path}")
        return export_path
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_id_mappings(self) -> Tuple[Dict[str, UUID], Dict[str, UUID]]:
        """Get ID mappings from last write operation."""
        return self._chunk_id_map.copy(), self._image_id_map.copy()
    
    async def get_document_stats(self, doc_id: UUID) -> Dict[str, Any]:
        """Get statistics for a written document."""
        doc = await self.repos.documents.get_with_stats(doc_id)
        return doc


# =============================================================================
# Convenience Functions
# =============================================================================

async def write_phase1_result_to_database(
    result,
    connection_string: str,
    source_path: str = None,
    export_files: bool = False,
    export_dir: Path = None
) -> UUID:
    """
    Convenience function to write Phase 1 result to database.
    
    Args:
        result: PipelineResult from Phase 1
        connection_string: PostgreSQL connection string
        source_path: Override source path
        export_files: Also export to files
        export_dir: Directory for file exports
    
    Returns:
        Document UUID
    """
    writer = PipelineDatabaseWriter(
        connection_string=connection_string,
        export_files=export_files,
        export_dir=export_dir
    )
    
    try:
        await writer.connect()
        doc_id = await writer.write_pipeline_result(result, source_path)
        return doc_id
    finally:
        await writer.close()


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    async def test():
        print("PipelineDatabaseWriter Test")
        print("=" * 50)
        
        # This would need actual connection and result
        print("To test, provide connection string and Phase 1 result")
        print()
        print("Usage:")
        print("  writer = PipelineDatabaseWriter(connection_string)")
        print("  await writer.connect()")
        print("  doc_id = await writer.write_pipeline_result(result)")
    
    asyncio.run(test())
