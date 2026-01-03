"""
NeuroSynth v2.0 - Database Layer (Enhanced)
============================================

PostgreSQL + pgvector storage with:
- Vector similarity search
- Full-text search (tsvector)
- Structured filtering (JSONB, arrays)
- Atomic transactions
- Connection health checks
- Batch COPY operations
- Robust error handling
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from src.shared.models import (
    Document, Page, SemanticChunk, ExtractedImage, ExtractedTable,
    DocumentStatus, ChunkType, ImageType, NeuroEntity, SearchResult
)

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base database error."""
    pass


class ConnectionError(DatabaseError):
    """Connection-related errors."""
    pass


class ValidationError(DatabaseError):
    """Data validation errors."""
    pass


class DimensionMismatchError(DatabaseError):
    """Embedding dimension mismatch."""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration."""
    min_pool_size: int = 2
    max_pool_size: int = 10
    connection_timeout: float = 30.0
    command_timeout: float = 60.0
    validate_on_connect: bool = True
    expected_embedding_dim: Optional[int] = 1536


@dataclass
class DatabaseStats:
    """Database statistics."""
    total_documents: int = 0
    ready_documents: int = 0
    total_chunks: int = 0
    total_images: int = 0
    chunks_with_embeddings: int = 0


class NeuroDatabase:
    """
    PostgreSQL + pgvector storage layer.
    
    Supports:
    - Vector similarity search (pgvector)
    - Full-text search (tsvector)
    - Structured filtering (JSONB, arrays)
    - Atomic transactions
    - Batch operations via COPY
    - Connection health monitoring
    """
    
    def __init__(self, pool: "asyncpg.Pool", config: DatabaseConfig = None):
        """
        Initialize with connection pool.
        
        Args:
            pool: asyncpg connection pool
            config: Database configuration
        """
        self._pool = pool
        self._config = config or DatabaseConfig()
        self._embedding_dim: Optional[int] = None

    @property
    def pool(self) -> "asyncpg.Pool":
        """Get the connection pool."""
        return self._pool

    @classmethod
    async def connect(
        cls,
        dsn: str,
        config: DatabaseConfig = None
    ) -> "NeuroDatabase":
        """
        Create database connection with validation.
        
        Args:
            dsn: PostgreSQL connection string
            config: Database configuration
            
        Returns:
            NeuroDatabase instance
            
        Raises:
            ConnectionError: If connection fails
            ValidationError: If pgvector not available
        """
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg required: pip install asyncpg")
        
        config = config or DatabaseConfig()
        
        try:
            pool = await asyncpg.create_pool(
                dsn,
                min_size=config.min_pool_size,
                max_size=config.max_pool_size,
                timeout=config.connection_timeout,
                command_timeout=config.command_timeout,
                init=cls._init_connection
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create connection pool: {e}") from e
        
        db = cls(pool, config)
        
        if config.validate_on_connect:
            await db._validate_connection()
        
        return db
    
    @staticmethod
    async def _init_connection(conn: "asyncpg.Connection"):
        """Initialize connection with custom type codecs."""
        await conn.set_type_codec(
            'vector',
            encoder=NeuroDatabase._encode_vector,
            decoder=NeuroDatabase._decode_vector,
            schema='public',
            format='text'
        )
    
    @staticmethod
    def _encode_vector(v) -> str:
        """Encode numpy array to pgvector format."""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return f'[{",".join(str(x) for x in v)}]'
        return f'[{",".join(str(x) for x in v.tolist())}]'
    
    @staticmethod
    def _decode_vector(v: str) -> Optional[np.ndarray]:
        """Decode pgvector format to numpy array."""
        if not v or v in ('[]', '', 'NULL'):
            return None
        try:
            cleaned = v.strip('[]')
            if not cleaned:
                return None
            values = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
            return np.array(values, dtype=np.float32)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to decode vector: {v[:50]}... Error: {e}")
            return None
    
    async def _validate_connection(self):
        """Validate database connection and extensions."""
        async with self._pool.acquire() as conn:
            # Basic connectivity
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise ConnectionError("Database connectivity check failed")
            
            # pgvector extension
            ext_version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            if not ext_version:
                raise ValidationError(
                    "pgvector extension not installed. "
                    "Run: CREATE EXTENSION vector;"
                )
            logger.info(f"Connected to PostgreSQL with pgvector v{ext_version}")
            
            # Detect embedding dimension from schema
            dim = await conn.fetchval("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'chunks'::regclass 
                AND attname = 'text_embedding'
            """)
            if dim and dim > 0:
                self._embedding_dim = dim
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            async with self._pool.acquire() as conn:
                # Test query
                await conn.fetchval("SELECT 1")
                
                # Pool stats
                return {
                    "healthy": True,
                    "pool_size": self._pool.get_size(),
                    "pool_free": self._pool.get_idle_size(),
                    "pool_min": self._pool.get_min_size(),
                    "pool_max": self._pool.get_max_size(),
                    "embedding_dim": self._embedding_dim
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM documents) as total_documents,
                    (SELECT COUNT(*) FROM documents WHERE status = 'ready') as ready_documents,
                    (SELECT COUNT(*) FROM chunks) as total_chunks,
                    (SELECT COUNT(*) FROM images) as total_images,
                    (SELECT COUNT(*) FROM chunks WHERE text_embedding IS NOT NULL) as chunks_with_embeddings
            """)
            return DatabaseStats(
                total_documents=row['total_documents'],
                ready_documents=row['ready_documents'],
                total_chunks=row['total_chunks'],
                total_images=row['total_images'],
                chunks_with_embeddings=row['chunks_with_embeddings']
            )
    
    async def close(self):
        """Close connection pool."""
        await self._pool.close()
    
    @asynccontextmanager
    async def transaction(self):
        """Atomic transaction context manager."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    # =========================================================================
    # DOCUMENT OPERATIONS
    # =========================================================================
    
    async def insert_document(
        self,
        doc: Document,
        conn: "asyncpg.Connection" = None
    ):
        """Insert document record."""
        conn = conn or self._pool
        
        await conn.execute("""
            INSERT INTO documents (
                id, content_hash, title, file_path, series, volume, chapter,
                authors, year, specialty, authority_score, status, page_count,
                error_message, created_at, indexed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ON CONFLICT (content_hash) DO UPDATE SET
                status = EXCLUDED.status,
                indexed_at = EXCLUDED.indexed_at,
                error_message = EXCLUDED.error_message
        """,
            UUID(doc.id), doc.content_hash, doc.title, str(doc.file_path),
            doc.series, doc.volume, doc.chapter, doc.authors, doc.year,
            doc.specialty, doc.authority_score, doc.status.value,
            doc.page_count, doc.error_message, doc.created_at, doc.indexed_at
        )
    
    async def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash."""
        row = await self._pool.fetchrow(
            "SELECT * FROM documents WHERE content_hash = $1",
            content_hash
        )
        return self._row_to_document(row) if row else None
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM documents WHERE id = $1",
            UUID(doc_id)
        )
        return self._row_to_document(row) if row else None
    
    async def get_all_document_paths(self) -> set:
        """Get all tracked document paths."""
        rows = await self._pool.fetch("SELECT file_path FROM documents")
        return {row['file_path'] for row in rows}
    
    async def delete_document(self, doc_id: str):
        """Delete document and all related data (cascade)."""
        await self._pool.execute(
            "DELETE FROM documents WHERE id = $1",
            UUID(doc_id)
        )
    
    async def delete_document_by_path(self, file_path: str):
        """Delete document by file path."""
        await self._pool.execute(
            "DELETE FROM documents WHERE file_path = $1",
            file_path
        )
    
    async def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ):
        """Update document processing status."""
        await self._pool.execute("""
            UPDATE documents SET
                status = $2,
                error_message = $3,
                indexed_at = CASE WHEN $2 = 'ready' THEN NOW() ELSE indexed_at END
            WHERE id = $1
        """, UUID(doc_id), status.value, error_message)
    
    # =========================================================================
    # PAGE OPERATIONS
    # =========================================================================
    
    async def insert_pages(
        self,
        pages: List[Page],
        conn: "asyncpg.Connection" = None
    ):
        """Batch insert pages using COPY."""
        conn = conn or self._pool
        
        if not pages:
            return
        
        # Prepare records for COPY
        records = [
            (
                UUID(p.document_id),
                p.page_number,
                p.content,
                p.has_images,
                p.has_tables,
                p.word_count
            )
            for p in pages
        ]
        
        await conn.copy_records_to_table(
            'pages',
            records=records,
            columns=['document_id', 'page_number', 'content', 'has_images', 'has_tables', 'word_count']
        )
    
    async def get_document_pages(self, doc_id: str) -> List[Page]:
        """Get all pages for a document."""
        rows = await self._pool.fetch(
            "SELECT * FROM pages WHERE document_id = $1 ORDER BY page_number",
            UUID(doc_id)
        )
        return [self._row_to_page(row) for row in rows]
    
    # =========================================================================
    # CHUNK OPERATIONS
    # =========================================================================
    
    async def insert_chunks(
        self,
        chunks: List[SemanticChunk],
        conn: "asyncpg.Connection" = None
    ):
        """
        Batch insert chunks.
        
        Uses individual inserts for complex JSONB/array columns.
        For very large batches, consider insert_chunks_copy().
        """
        conn = conn or self._pool
        
        for chunk in chunks:
            entities_json = [e.to_dict() for e in chunk.entities] if chunk.entities else []
            
            text_emb = chunk.text_embedding.tolist() if chunk.text_embedding is not None else None
            fused_emb = chunk.fused_embedding.tolist() if chunk.fused_embedding is not None else None
            
            # Validate embedding dimensions
            if text_emb and self._embedding_dim and len(text_emb) != self._embedding_dim:
                raise DimensionMismatchError(
                    f"Expected {self._embedding_dim} dimensions, got {len(text_emb)}. "
                    f"Update schema or use matching embedding model."
                )
            
            await conn.execute("""
                INSERT INTO chunks (
                    id, document_id, content, title, chunk_type,
                    page_start, page_end, section_path,
                    entity_names, entities_json, specialty_tags,
                    figure_refs, table_refs, image_ids, keywords,
                    text_embedding, fused_embedding, summary
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
            """,
                UUID(chunk.id), UUID(chunk.document_id), chunk.content, chunk.title,
                chunk.chunk_type.value, chunk.page_start, chunk.page_end,
                chunk.section_path, chunk.entity_names, json.dumps(entities_json),
                chunk.specialty_tags, chunk.figure_refs, chunk.table_refs,
                chunk.image_ids, chunk.keywords, text_emb, fused_emb,
                chunk.summary
            )
    
    async def insert_chunks_batch(
        self,
        chunks: List[SemanticChunk],
        conn: "asyncpg.Connection" = None
    ):
        """
        High-performance batch insert using prepared statement.
        
        More efficient than insert_chunks for large batches (100+ chunks).
        """
        conn = conn or self._pool
        
        if not chunks:
            return
        
        # Use executemany with prepared statement
        stmt = await conn.prepare("""
            INSERT INTO chunks (
                id, document_id, content, title, chunk_type,
                page_start, page_end, section_path,
                entity_names, entities_json, specialty_tags,
                figure_refs, table_refs, image_ids, keywords,
                text_embedding, fused_embedding
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
        """)
        
        records = []
        for chunk in chunks:
            entities_json = [e.to_dict() for e in chunk.entities] if chunk.entities else []
            text_emb = chunk.text_embedding.tolist() if chunk.text_embedding is not None else None
            fused_emb = chunk.fused_embedding.tolist() if chunk.fused_embedding is not None else None
            
            records.append((
                UUID(chunk.id), UUID(chunk.document_id), chunk.content, chunk.title,
                chunk.chunk_type.value, chunk.page_start, chunk.page_end,
                chunk.section_path, chunk.entity_names, json.dumps(entities_json),
                chunk.specialty_tags, chunk.figure_refs, chunk.table_refs,
                chunk.image_ids, chunk.keywords, text_emb, fused_emb
            ))
        
        await stmt.executemany(records)
    
    async def get_document_chunks(self, doc_id: str) -> List[SemanticChunk]:
        """Get all chunks for a document."""
        rows = await self._pool.fetch(
            "SELECT * FROM chunks WHERE document_id = $1 ORDER BY page_start",
            UUID(doc_id)
        )
        return [self._row_to_chunk(row) for row in rows]
    
    async def update_chunk_embeddings(
        self,
        updates: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]
    ):
        """Batch update embeddings."""
        async with self.transaction() as conn:
            stmt = await conn.prepare("""
                UPDATE chunks SET
                    text_embedding = $2,
                    fused_embedding = $3
                WHERE id = $1
            """)
            
            for chunk_id, text_emb, fused_emb in updates:
                await stmt.fetchval(
                    UUID(chunk_id),
                    text_emb.tolist() if text_emb is not None else None,
                    fused_emb.tolist() if fused_emb is not None else None
                )
    
    # =========================================================================
    # IMAGE OPERATIONS
    # =========================================================================
    
    async def insert_images(
        self,
        images: List[ExtractedImage],
        conn: "asyncpg.Connection" = None
    ):
        """Batch insert images."""
        conn = conn or self._pool
        
        if not images:
            return
        
        stmt = await conn.prepare("""
            INSERT INTO images (
                id, document_id, page_number, file_path,
                width, height, format, content_hash,
                image_type, is_decorative, quality_score,
                caption, caption_confidence, figure_id, surrounding_text,
                sequence_id, sequence_position, chunk_ids, embedding,
                vlm_caption, caption_summary, caption_embedding
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )
        """)

        for img in images:
            embedding = img.embedding.tolist() if img.embedding is not None else None
            caption_embedding = img.caption_embedding.tolist() if img.caption_embedding is not None else None

            await stmt.fetchval(
                UUID(img.id), UUID(img.document_id), img.page_number, str(img.file_path),
                img.width, img.height, img.format, img.content_hash,
                img.image_type.value, img.is_decorative, img.quality_score,
                img.caption, img.caption_confidence, img.figure_id,
                img.surrounding_text[:1000] if img.surrounding_text else None,
                img.sequence_id, img.sequence_position, img.chunk_ids, embedding,
                getattr(img, 'vlm_caption', None), getattr(img, 'caption_summary', None),
                caption_embedding
            )
    
    async def get_document_images(self, doc_id: str) -> List[ExtractedImage]:
        """Get all images for a document."""
        rows = await self._pool.fetch(
            "SELECT * FROM images WHERE document_id = $1 ORDER BY page_number",
            UUID(doc_id)
        )
        return [self._row_to_image(row) for row in rows]
    
    async def get_images_by_ids(self, image_ids: List[str]) -> List[ExtractedImage]:
        """Get images by IDs."""
        if not image_ids:
            return []
        
        rows = await self._pool.fetch(
            "SELECT * FROM images WHERE id = ANY($1)",
            [UUID(id) for id in image_ids]
        )
        return [self._row_to_image(row) for row in rows]
    
    # =========================================================================
    # TABLE OPERATIONS
    # =========================================================================
    
    async def insert_tables(
        self,
        tables: List[ExtractedTable],
        conn: "asyncpg.Connection" = None
    ):
        """Batch insert tables."""
        conn = conn or self._pool
        
        if not tables:
            return
        
        for table in tables:
            await conn.execute("""
                INSERT INTO extracted_tables (
                    id, document_id, page_number,
                    markdown_content, raw_text, table_type, title, chunk_ids
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                UUID(table.id), UUID(table.document_id), table.page_number,
                table.markdown_content, table.raw_text,
                table.table_type, table.title, table.chunk_ids
            )
    
    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================
    
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        specialty_filter: Optional[str] = None,
        entity_filter: Optional[List[str]] = None,
        chunk_type_filter: Optional[str] = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity, keyword search, and filters.
        """
        rows = await self._pool.fetch("""
            SELECT * FROM hybrid_search($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            query_text,
            query_embedding.tolist(),
            specialty_filter,
            entity_filter,
            chunk_type_filter,
            keyword_weight,
            semantic_weight,
            limit
        )
        
        return [
            SearchResult(
                chunk_id=str(row['chunk_id']),
                document_id=str(row['document_id']),
                content=row['content'],
                title=row['title'],
                chunk_type=ChunkType(row['chunk_type']) if row['chunk_type'] else ChunkType.GENERAL,
                page_start=row['page_start'],
                entity_names=row['entity_names'] or [],
                image_ids=row['image_ids'] or [],
                authority_score=row['authority_score'],
                keyword_score=row['keyword_score'],
                semantic_score=row['semantic_score'],
                final_score=row['final_score']
            )
            for row in rows
        ]
    
    async def entity_search(
        self,
        entities: List[str],
        specialty_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[SemanticChunk]:
        """Search by entity overlap."""
        rows = await self._pool.fetch("""
            SELECT * FROM entity_search($1, $2, $3)
        """, entities, specialty_filter, limit)
        
        return [self._row_to_chunk(row) for row in rows]
    
    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        specialty_filter: Optional[str] = None
    ) -> List[SemanticChunk]:
        """Pure vector similarity search."""
        query = """
            SELECT c.*, 1 - (c.fused_embedding <=> $1) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.fused_embedding IS NOT NULL
        """
        params = [query_embedding.tolist()]
        
        if specialty_filter:
            query += " AND $2 = ANY(c.specialty_tags)"
            params.append(specialty_filter)
        
        query += f" ORDER BY c.fused_embedding <=> $1 LIMIT {limit}"
        
        rows = await self._pool.fetch(query, *params)
        return [self._row_to_chunk(row) for row in rows]
    
    async def keyword_search(
        self,
        query: str,
        limit: int = 20
    ) -> List[SemanticChunk]:
        """Full-text keyword search."""
        rows = await self._pool.fetch("""
            SELECT c.*, ts_rank(c.content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM chunks c
            WHERE c.content_tsv @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2
        """, query, limit)
        
        return [self._row_to_chunk(row) for row in rows]
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _row_to_document(self, row) -> Document:
        """Convert DB row to Document."""
        return Document(
            id=str(row['id']),
            file_path=Path(row['file_path']),
            content_hash=row['content_hash'],
            title=row['title'],
            series=row['series'],
            volume=row['volume'],
            chapter=row['chapter'],
            authors=row['authors'],
            year=row['year'],
            specialty=row['specialty'],
            authority_score=row['authority_score'],
            status=DocumentStatus(row['status']),
            page_count=row['page_count'],
            error_message=row['error_message'],
            created_at=row['created_at'],
            indexed_at=row['indexed_at']
        )
    
    def _row_to_page(self, row) -> Page:
        """Convert DB row to Page."""
        return Page(
            document_id=str(row['document_id']),
            page_number=row['page_number'],
            content=row['content'],
            has_images=row['has_images'],
            has_tables=row['has_tables'],
            word_count=row['word_count']
        )
    
    def _row_to_chunk(self, row) -> SemanticChunk:
        """Convert DB row to SemanticChunk."""
        entities = []
        if row.get('entities_json'):
            try:
                entities_data = json.loads(row['entities_json']) if isinstance(row['entities_json'], str) else row['entities_json']
                entities = [NeuroEntity.from_dict(e) for e in entities_data]
            except Exception as e:
                logger.warning(f"Failed to parse entities_json: {e}")
        
        return SemanticChunk(
            id=str(row['id']),
            document_id=str(row['document_id']),
            content=row['content'],
            title=row['title'],
            section_path=row['section_path'] or [],
            page_start=row['page_start'],
            page_end=row['page_end'],
            chunk_type=ChunkType(row['chunk_type']) if row['chunk_type'] else ChunkType.GENERAL,
            specialty_tags=row['specialty_tags'] or [],
            entities=entities,
            entity_names=row['entity_names'] or [],
            figure_refs=row['figure_refs'] or [],
            table_refs=row.get('table_refs') or [],
            image_ids=row['image_ids'] or [],
            keywords=row['keywords'] or [],
            text_embedding=row.get('text_embedding'),
            fused_embedding=row.get('fused_embedding')
        )
    
    def _row_to_image(self, row) -> ExtractedImage:
        """Convert DB row to ExtractedImage."""
        return ExtractedImage(
            id=str(row['id']),
            document_id=str(row['document_id']),
            page_number=row['page_number'],
            file_path=Path(row['file_path']),
            width=row['width'],
            height=row['height'],
            format=row['format'],
            content_hash=row['content_hash'],
            image_type=ImageType(row['image_type']) if row['image_type'] else ImageType.UNKNOWN,
            is_decorative=row['is_decorative'],
            quality_score=row['quality_score'],
            caption=row['caption'],
            caption_confidence=row['caption_confidence'],
            figure_id=row['figure_id'],
            surrounding_text=row['surrounding_text'],
            sequence_id=row['sequence_id'],
            sequence_position=row['sequence_position'],
            chunk_ids=row['chunk_ids'] or [],
            embedding=row.get('embedding')
        )
