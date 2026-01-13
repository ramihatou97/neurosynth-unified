"""
NeuroSynth Enhanced Relation Extraction Pipeline

Integrates the enhanced relation extractor into the ingestion pipeline.
Supports all enhancement features through configuration.

Usage:
    from src.ingest.relation_pipeline import RelationExtractionPipeline
    from src.core.relation_config import RelationExtractionConfig
    
    # Default config (safe enhancements enabled)
    pipeline = RelationExtractionPipeline(db_pool)
    
    # Full features
    config = RelationExtractionConfig(
        enable_tiered_llm=True,
        enable_coreference=True,
    )
    pipeline = RelationExtractionPipeline(
        db_pool,
        config=config,
        llm_client=anthropic_client,
    )
    
    # Process chunks
    await pipeline.process_chunk(chunk_id, chunk_text)
    await pipeline.flush()
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from uuid import UUID

# Import from enhanced extractor
from src.core.relation_extractor import (
    NeuroRelationExtractor,
    ExtractedRelation,
    RelationType,
    ExtractionMethod,
    TAXONOMY,
    build_graph_from_relations,
)
from src.core.relation_config import RelationExtractionConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkRelations:
    """Relations extracted from a single chunk."""
    chunk_id: UUID
    relations: List[ExtractedRelation]
    
    @property
    def count(self) -> int:
        return len(self.relations)
    
    @property
    def negated_count(self) -> int:
        return sum(1 for r in self.relations if r.is_negated)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    total_chunks: int = 0
    total_relations: int = 0
    total_entities: int = 0
    negated_relations: int = 0
    relations_by_method: Dict[str, int] = None
    relations_by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.relations_by_method is None:
            self.relations_by_method = {}
        if self.relations_by_type is None:
            self.relations_by_type = {}
    
    def to_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "total_relations": self.total_relations,
            "total_entities": self.total_entities,
            "negated_relations": self.negated_relations,
            "relations_by_method": self.relations_by_method,
            "relations_by_type": self.relations_by_type,
        }


class RelationExtractionPipeline:
    """
    Integrates relation extraction into the NeuroSynth ingestion pipeline.
    
    Features:
    - Configurable extraction enhancements
    - Batch database insertion for performance
    - Statistics tracking
    - Graceful error handling
    
    Usage in pipeline.py:
    
        from src.ingest.relation_pipeline import RelationExtractionPipeline
        from src.core.relation_config import RelationExtractionConfig
        
        # Initialize with config
        config = RelationExtractionConfig(
            enable_coordination=True,
            enable_negation=True,
        )
        relation_pipeline = RelationExtractionPipeline(db_pool, config=config)
        
        # During ingestion, after chunk creation:
        await relation_pipeline.process_chunk(chunk_id, chunk_text)
        
        # At end of document:
        await relation_pipeline.flush()
        
        # Get stats:
        stats = relation_pipeline.get_stats()
    """
    
    def __init__(
        self,
        db_pool,  # asyncpg connection pool
        extractor: Optional[NeuroRelationExtractor] = None,
        config: Optional[RelationExtractionConfig] = None,
        llm_client: Optional[Any] = None,
        umls_extractor: Optional[Any] = None,
        batch_size: int = 50,
        min_confidence: float = 0.0,
    ):
        """
        Initialize the pipeline.
        
        Args:
            db_pool: asyncpg database connection pool
            extractor: Pre-configured extractor (if None, creates from config)
            config: Extraction configuration (uses defaults if None)
            llm_client: LLM client for tiered verification
            umls_extractor: UMLS extractor for entity-first strategy
            batch_size: Number of relations to batch before inserting
            min_confidence: Minimum confidence threshold for storage
        """
        self.db_pool = db_pool
        self.config = config or RelationExtractionConfig()
        self.batch_size = batch_size
        self.min_confidence = min_confidence or self.config.min_confidence
        
        # Initialize extractor with all enhancements
        if extractor:
            self.extractor = extractor
        else:
            self.extractor = NeuroRelationExtractor(
                config=self.config,
                llm_client=llm_client if self.config.enable_tiered_llm else None,
                umls_extractor=umls_extractor if self.config.enable_entity_first_ner else None,
            )
        
        # Pending relations for batch insert
        self._pending_relations: List[ExtractedRelation] = []
        self._pending_chunk_ids: List[UUID] = []
        
        # Statistics
        self._stats = PipelineStats()
    
    def get_stats(self) -> dict:
        """Get extraction statistics."""
        return self._stats.to_dict()
    
    def get_config(self) -> dict:
        """Get current configuration."""
        return self.config.to_dict()
    
    async def process_chunk(
        self,
        chunk_id: UUID,
        chunk_text: str,
        use_llm: bool = False,
    ) -> List[ExtractedRelation]:
        """
        Extract relations from a chunk and queue for batch insert.
        
        Args:
            chunk_id: UUID of the chunk being processed
            chunk_text: Text content of the chunk
            use_llm: Force LLM verification even if not in config
            
        Returns:
            List of extracted relations
        """
        self._stats.total_chunks += 1
        
        try:
            # Extract relations
            if use_llm and self.config.enable_tiered_llm:
                relations = await self.extractor.extract_with_llm_verification(
                    chunk_text, str(chunk_id)
                )
            else:
                relations = self.extractor.extract_from_text(chunk_text)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_id}: {e}")
            return []
        
        # Filter by confidence and queue for insert
        for rel in relations:
            if rel.confidence >= self.min_confidence:
                self._pending_relations.append(rel)
                self._pending_chunk_ids.append(chunk_id)
                
                # Update stats
                self._stats.total_relations += 1
                
                if rel.is_negated:
                    self._stats.negated_relations += 1
                
                # Track by method
                method = rel.extraction_method
                self._stats.relations_by_method[method] = (
                    self._stats.relations_by_method.get(method, 0) + 1
                )
                
                # Track by type
                rel_type = rel.relation.value
                self._stats.relations_by_type[rel_type] = (
                    self._stats.relations_by_type.get(rel_type, 0) + 1
                )
        
        # Flush if batch is full
        if len(self._pending_relations) >= self.batch_size:
            await self.flush()
        
        return relations
    
    async def process_chunks_batch(
        self,
        chunks: List[tuple],
    ) -> List[ChunkRelations]:
        """
        Process multiple chunks in batch.
        
        Args:
            chunks: List of (chunk_id, chunk_text) tuples
            
        Returns:
            List of ChunkRelations for each chunk
        """
        results = []
        
        for chunk_id, chunk_text in chunks:
            relations = await self.process_chunk(chunk_id, chunk_text)
            results.append(ChunkRelations(
                chunk_id=chunk_id,
                relations=relations,
            ))
        
        return results
    
    async def flush(self):
        """Flush pending relations to database."""
        if not self._pending_relations:
            return
        
        if not self.db_pool:
            # No database - just clear pending and return
            logger.debug("No database pool, clearing pending relations")
            self._pending_relations = []
            self._pending_chunk_ids = []
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # First, ensure all entities exist
                entities = set()
                for rel in self._pending_relations:
                    entities.add(rel.source_normalized or rel.source)
                    entities.add(rel.target_normalized or rel.target)
                
                self._stats.total_entities += len(entities)
                entity_ids = await self._ensure_entities(conn, entities)
                
                # Insert relations
                await self._insert_relations(
                    conn,
                    self._pending_relations,
                    self._pending_chunk_ids,
                    entity_ids,
                )
                
                logger.info(
                    f"Flushed {len(self._pending_relations)} relations "
                    f"for {len(entities)} entities"
                )
        except Exception as e:
            logger.error(f"Failed to flush relations: {e}")
            raise
        finally:
            self._pending_relations = []
            self._pending_chunk_ids = []
    
    async def _ensure_entities(
        self,
        conn,
        entities: set,
    ) -> Dict[str, UUID]:
        """
        Ensure entities exist in DB and return their IDs.
        
        Args:
            conn: Database connection
            entities: Set of entity names to ensure
            
        Returns:
            Dict mapping entity name to UUID
        """
        entity_ids = {}
        
        for entity in entities:
            try:
                # First check if entity exists
                result = await conn.fetchrow(
                    "SELECT id FROM entities WHERE name = $1 LIMIT 1",
                    entity
                )
                
                if result:
                    entity_ids[entity] = result['id']
                else:
                    # Insert new entity
                    result = await conn.fetchrow(
                        """
                        INSERT INTO entities (name, category, source)
                        VALUES ($1, 'extracted', 'relation_extraction')
                        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                        RETURNING id
                        """,
                        entity
                    )
                    entity_ids[entity] = result['id']
            except Exception as e:
                logger.warning(f"Failed to ensure entity '{entity}': {e}")
        
        return entity_ids
    
    async def _insert_relations(
        self,
        conn,
        relations: List[ExtractedRelation],
        chunk_ids: List[UUID],
        entity_ids: Dict[str, UUID],
    ):
        """
        Batch insert relations into database.
        
        Args:
            conn: Database connection
            relations: Relations to insert
            chunk_ids: Corresponding chunk IDs
            entity_ids: Mapping of entity name to UUID
        """
        if not relations:
            return
        
        # Build values for batch insert
        values = []
        
        for rel, chunk_id in zip(relations, chunk_ids):
            source_name = rel.source_normalized or rel.source
            target_name = rel.target_normalized or rel.target
            
            source_id = entity_ids.get(source_name)
            target_id = entity_ids.get(target_name)
            
            if not source_id or not target_id:
                logger.warning(
                    f"Missing entity ID for relation: {source_name} -> {target_name}"
                )
                continue
            
            values.append((
                source_id,
                target_id,
                rel.relation.value,
                rel.confidence,
                [chunk_id],  # chunk_ids array
                rel.context_snippet[:500] if rel.context_snippet else None,
                rel.is_negated,
                rel.negation_cue,
                rel.extraction_method,
            ))
        
        if not values:
            return
        
        # Use executemany with ON CONFLICT to handle duplicates
        await conn.executemany(
            """
            INSERT INTO entity_relations (
                source_entity_id,
                target_entity_id,
                relation_type,
                confidence,
                chunk_ids,
                context_snippet,
                is_negated,
                negation_cue,
                extraction_method
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (source_entity_id, target_entity_id, relation_type)
            DO UPDATE SET
                confidence = GREATEST(entity_relations.confidence, EXCLUDED.confidence),
                chunk_ids = array_cat(entity_relations.chunk_ids, EXCLUDED.chunk_ids),
                context_snippet = COALESCE(EXCLUDED.context_snippet, entity_relations.context_snippet),
                is_negated = EXCLUDED.is_negated,
                negation_cue = EXCLUDED.negation_cue,
                extraction_method = EXCLUDED.extraction_method
            """,
            values
        )
    
    async def reprocess_document(
        self,
        document_id: UUID,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Reprocess all chunks in a document with current config.
        
        Args:
            document_id: Document to reprocess
            force: If True, delete existing relations first
            
        Returns:
            Dict with processing statistics
        """
        if not self.db_pool:
            raise ValueError("Database pool required for reprocessing")
        
        async with self.db_pool.acquire() as conn:
            # Get all chunks for document
            chunks = await conn.fetch(
                """
                SELECT id, content
                FROM chunks
                WHERE document_id = $1
                ORDER BY start_page, id
                """,
                document_id
            )
            
            if not chunks:
                return {"error": f"No chunks found for document {document_id}"}
            
            # Optionally clear existing relations
            if force:
                # Get entity IDs from this document's chunks
                chunk_ids = [row['id'] for row in chunks]
                await conn.execute(
                    """
                    DELETE FROM entity_relations
                    WHERE chunk_ids && $1::UUID[]
                    """,
                    chunk_ids
                )
                logger.info(f"Cleared existing relations for document {document_id}")
            
            # Process each chunk
            initial_stats = self._stats.total_relations
            
            for chunk in chunks:
                await self.process_chunk(chunk['id'], chunk['content'])
            
            # Flush remaining
            await self.flush()
            
            new_relations = self._stats.total_relations - initial_stats
            
            return {
                "document_id": str(document_id),
                "chunks_processed": len(chunks),
                "relations_extracted": new_relations,
                "config": self.config.to_dict(),
            }


# =============================================================================
# Graph Context Expansion
# =============================================================================

class GraphContextExpander:
    """
    Expands query context using the knowledge graph.
    
    Usage in RAG pipeline:
        expander = GraphContextExpander(db_pool, extractor)
        
        # Get related entities and chunks for a query
        context = await expander.expand(
            query="What are the risks of aneurysm rupture?",
            seed_entities=["aneurysm", "subarachnoid hemorrhage"],
        )
    """
    
    def __init__(
        self,
        db_pool,
        extractor: Optional[NeuroRelationExtractor] = None,
    ):
        self.db_pool = db_pool
        self.extractor = extractor or NeuroRelationExtractor()
    
    async def expand(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None,
        hop_limit: int = 2,
        max_related: int = 20,
    ) -> Dict[str, Any]:
        """
        Expand query context using knowledge graph.
        
        Args:
            query: User query
            seed_entities: Starting entities (extracted from query if None)
            hop_limit: Maximum hops from seed entities
            max_related: Maximum related entities to return
            
        Returns:
            Dict with expanded entities and chunk IDs
        """
        if not self.db_pool:
            return {"entities": [], "chunk_ids": []}
        
        # Extract entities from query if not provided
        if not seed_entities:
            relations = self.extractor.extract_from_text(query)
            seed_entities = set()
            for rel in relations:
                seed_entities.add(rel.source_normalized or rel.source)
                seed_entities.add(rel.target_normalized or rel.target)
            seed_entities = list(seed_entities)
        
        if not seed_entities:
            return {"entities": [], "chunk_ids": []}
        
        # Normalize seed entities
        normalized_seeds = [
            self.extractor.normalize_entity(e) for e in seed_entities
        ]
        
        # Traverse graph
        async with self.db_pool.acquire() as conn:
            related_entities = set(normalized_seeds)
            chunk_ids = set()
            
            current_layer = set(normalized_seeds)
            
            for hop in range(hop_limit):
                if not current_layer or len(related_entities) >= max_related:
                    break
                
                # Get relations from current layer
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT
                        e2.name as target_name,
                        er.chunk_ids,
                        er.is_negated
                    FROM entity_relations er
                    JOIN entities e1 ON er.source_entity_id = e1.id
                    JOIN entities e2 ON er.target_entity_id = e2.id
                    WHERE e1.name = ANY($1)
                    AND er.is_negated = FALSE
                    AND er.confidence >= 0.5
                    LIMIT $2
                    """,
                    list(current_layer),
                    max_related - len(related_entities),
                )
                
                next_layer = set()
                
                for row in rows:
                    target = row['target_name']
                    if target not in related_entities:
                        related_entities.add(target)
                        next_layer.add(target)
                    
                    if row['chunk_ids']:
                        chunk_ids.update(row['chunk_ids'])
                
                current_layer = next_layer
            
            # Also get taxonomy expansions
            for entity in list(related_entities)[:10]:
                if entity in TAXONOMY:
                    related_entities.update(TAXONOMY[entity])
            
            return {
                "entities": list(related_entities)[:max_related],
                "chunk_ids": list(chunk_ids),
                "seed_entities": normalized_seeds,
                "hops_traversed": hop_limit,
            }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RelationExtractionPipeline",
    "ChunkRelations",
    "PipelineStats",
    "GraphContextExpander",
]
