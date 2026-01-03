"""
NeuroSynth Pipeline Integration for Relation Extraction

Integrates the relation extractor into the ingestion pipeline and 
provides graph-augmented RAG context expansion.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

# Import from main extractor module
from src.core.relation_extractor import (
    NeuroRelationExtractor,
    ExtractedRelation,
    RelationType,
    TAXONOMY,
    build_graph_from_relations,
)


@dataclass
class ChunkRelations:
    """Relations extracted from a single chunk."""
    chunk_id: UUID
    relations: list[ExtractedRelation]
    
    
class RelationExtractionPipeline:
    """
    Integrates relation extraction into the NeuroSynth ingestion pipeline.
    
    Usage in pipeline.py:
    
        from relation_pipeline import RelationExtractionPipeline
        
        relation_pipeline = RelationExtractionPipeline(db_pool)
        
        # During ingestion, after chunk creation:
        await relation_pipeline.process_chunk(chunk_id, chunk_text)
    """
    
    def __init__(
        self,
        db_pool,  # asyncpg connection pool
        extractor: Optional[NeuroRelationExtractor] = None,
        batch_size: int = 50,
        min_confidence: float = 0.0,
    ):
        self.db_pool = db_pool
        self.extractor = extractor or NeuroRelationExtractor()
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self._pending_relations: list[ExtractedRelation] = []
        self._pending_chunk_ids: list[UUID] = []
        # Stats tracking
        self._total_relations = 0
        self._total_entities = 0
        self._total_chunks = 0

    def get_stats(self) -> dict:
        """Get extraction statistics."""
        return {
            "relations": self._total_relations,
            "entities": self._total_entities,
            "chunks": self._total_chunks,
        }
    
    async def process_chunk(
        self,
        chunk_id: UUID,
        chunk_text: str,
    ) -> list[ExtractedRelation]:
        """
        Extract relations from a chunk and queue for batch insert.

        Args:
            chunk_id: UUID of the chunk being processed
            chunk_text: Text content of the chunk

        Returns:
            List of extracted relations
        """
        self._total_chunks += 1
        relations = self.extractor.extract_from_text(chunk_text)

        # Filter by min_confidence and add chunk_id reference
        for rel in relations:
            if rel.confidence >= self.min_confidence:
                self._pending_relations.append(rel)
                self._pending_chunk_ids.append(chunk_id)
                self._total_relations += 1

        # Flush if batch is full
        if len(self._pending_relations) >= self.batch_size:
            await self.flush()

        return relations
    
    async def flush(self):
        """Flush pending relations to database."""
        if not self._pending_relations:
            return

        if not self.db_pool:
            # No database - just clear pending and return
            self._pending_relations = []
            self._pending_chunk_ids = []
            return

        async with self.db_pool.acquire() as conn:
            # First, ensure all entities exist
            entities = set()
            for rel in self._pending_relations:
                entities.add(rel.source_normalized)
                entities.add(rel.target_normalized)

            self._total_entities += len(entities)
            entity_ids = await self._ensure_entities(conn, entities)

            # Insert relations
            await self._insert_relations(
                conn,
                self._pending_relations,
                self._pending_chunk_ids,
                entity_ids,
            )

        self._pending_relations = []
        self._pending_chunk_ids = []
    
    async def _ensure_entities(
        self,
        conn,
        entities: set[str],
    ) -> dict[str, UUID]:
        """Ensure entities exist in DB and return their IDs."""
        entity_ids = {}

        for entity in entities:
            # First check if entity exists
            result = await conn.fetchrow("""
                SELECT id FROM entities WHERE name = $1 LIMIT 1
            """, entity)

            if result:
                entity_ids[entity] = result['id']
            else:
                # Insert new entity
                result = await conn.fetchrow("""
                    INSERT INTO entities (name, category, source)
                    VALUES ($1, 'extracted', 'relation_extraction')
                    RETURNING id
                """, entity)
                entity_ids[entity] = result['id']

        return entity_ids
    
    async def _insert_relations(
        self,
        conn,
        relations: list[ExtractedRelation],
        chunk_ids: list[UUID],
        entity_ids: dict[str, UUID],
    ):
        """Batch insert relations."""
        for rel, chunk_id in zip(relations, chunk_ids):
            source_id = entity_ids[rel.source_normalized]
            target_id = entity_ids[rel.target_normalized]
            
            # Upsert relation, appending chunk_id to array
            await conn.execute("""
                INSERT INTO entity_relations 
                    (source_entity_id, target_entity_id, relation_type, 
                     confidence, chunk_ids, context_snippet)
                VALUES ($1, $2, $3, $4, ARRAY[$5]::uuid[], $6)
                ON CONFLICT (source_entity_id, target_entity_id, relation_type) 
                DO UPDATE SET
                    confidence = GREATEST(entity_relations.confidence, EXCLUDED.confidence),
                    chunk_ids = array_cat(entity_relations.chunk_ids, EXCLUDED.chunk_ids),
                    context_snippet = CASE 
                        WHEN length(EXCLUDED.context_snippet) > length(entity_relations.context_snippet)
                        THEN EXCLUDED.context_snippet
                        ELSE entity_relations.context_snippet
                    END
            """, source_id, target_id, rel.relation.value, 
                rel.confidence, chunk_id, rel.context_snippet)
            
            # Insert reverse edge for bidirectional relations
            if rel.bidirectional:
                await conn.execute("""
                    INSERT INTO entity_relations 
                        (source_entity_id, target_entity_id, relation_type,
                         confidence, chunk_ids, context_snippet)
                    VALUES ($1, $2, $3, $4, ARRAY[$5]::uuid[], $6)
                    ON CONFLICT (source_entity_id, target_entity_id, relation_type) 
                    DO NOTHING
                """, target_id, source_id, rel.relation.value,
                    rel.confidence, chunk_id, rel.context_snippet)


class GraphAugmentedRAG:
    """
    Graph-augmented RAG context expansion.
    
    Integrates with the RAG engine to expand context using knowledge graph traversal.
    
    Usage in engine.py:
    
        from relation_pipeline import GraphAugmentedRAG
        
        graph_rag = GraphAugmentedRAG(db_pool)
        
        async def ask(question: str) -> str:
            # Extract entities from question
            entities = extract_entities(question)
            
            # Get graph context
            graph_context = await graph_rag.get_context(
                entities=entities,
                question_embedding=question_embedding,
                hop_limit=2,
            )
            
            # Merge with search results
            combined_chunks = merge_chunks(search_chunks, graph_context.chunks)
    """
    
    def __init__(
        self,
        db_pool,
        extractor: Optional[NeuroRelationExtractor] = None,
        max_neighbors: int = 10,
    ):
        self.db_pool = db_pool
        self.extractor = extractor or NeuroRelationExtractor()
        self.max_neighbors = max_neighbors
    
    async def get_context(
        self,
        entities: list[str],
        question_embedding: Optional[list[float]] = None,
        hop_limit: int = 2,
    ) -> dict:
        """
        Get graph-augmented context for a set of entities.
        
        Args:
            entities: List of entity names from the question
            question_embedding: Optional embedding for relevance ranking
            hop_limit: Maximum traversal depth
            
        Returns:
            Dict containing:
                - related_chunks: chunk IDs from graph neighbors
                - relationships: list of relevant relationships
                - expanded_entities: entities reachable within hop_limit
        """
        # Normalize entities
        normalized = [self.extractor.normalize_entity(e) for e in entities]
        
        # Expand with taxonomy
        expanded = set(normalized)
        for entity in normalized:
            taxonomy_chain = self._get_taxonomy_chain(entity)
            expanded.update(taxonomy_chain)
        
        # Traverse graph
        async with self.db_pool.acquire() as conn:
            neighbors, relationships = await self._traverse(
                conn, 
                list(expanded), 
                hop_limit,
            )
        
        # Rank neighbors by relevance if embedding provided
        if question_embedding:
            neighbors = await self._rank_by_relevance(
                neighbors, 
                question_embedding,
            )
        
        # Limit to top N
        neighbors = neighbors[:self.max_neighbors]
        
        # Collect chunk IDs from neighbors
        chunk_ids = await self._get_neighbor_chunks(neighbors)
        
        return {
            "related_chunks": chunk_ids,
            "relationships": relationships,
            "expanded_entities": list(expanded),
        }
    
    def _get_taxonomy_chain(self, entity: str) -> list[str]:
        """Get all parent entities from taxonomy."""
        parents = []
        visited = set()
        queue = [entity]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current in TAXONOMY:
                for parent in TAXONOMY[current]:
                    parents.append(parent)
                    queue.append(parent)
        
        return parents
    
    async def _traverse(
        self,
        conn,
        entities: list[str],
        hop_limit: int,
    ) -> tuple[list[str], list[dict]]:
        """
        Traverse graph from seed entities.
        
        Returns (neighbor_entities, relationships)
        """
        neighbors = set()
        relationships = []
        
        current_layer = set(entities)
        
        for hop in range(hop_limit):
            if not current_layer:
                break
            
            # Get all edges from current layer
            rows = await conn.fetch("""
                SELECT 
                    e1.name as source_name,
                    e2.name as target_name,
                    er.relation_type,
                    er.confidence,
                    er.context_snippet
                FROM entity_relations er
                JOIN entities e1 ON er.source_entity_id = e1.id
                JOIN entities e2 ON er.target_entity_id = e2.id
                WHERE e1.name = ANY($1)
            """, list(current_layer))
            
            next_layer = set()
            
            for row in rows:
                target = row['target_name']
                
                if target not in neighbors and target not in entities:
                    next_layer.add(target)
                    neighbors.add(target)
                    
                    relationships.append({
                        "source": row['source_name'],
                        "target": target,
                        "relation": row['relation_type'],
                        "confidence": row['confidence'],
                        "context": row['context_snippet'],
                        "hop": hop + 1,
                    })
            
            current_layer = next_layer
        
        return list(neighbors), relationships
    
    async def _rank_by_relevance(
        self,
        neighbors: list[str],
        question_embedding: list[float],
    ) -> list[str]:
        """
        Rank neighbors by semantic similarity to question.

        Uses precomputed entity embeddings from entity_embeddings table.
        Falls back to original order if embeddings not available.
        """
        import numpy as np

        if not neighbors:
            return neighbors

        if not question_embedding:
            logger.debug("No question embedding provided, skipping ranking")
            return neighbors

        try:
            async with self.db_pool.acquire() as conn:
                # Check if entity_embeddings table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'entity_embeddings'
                    )
                """)

                if not table_exists:
                    logger.warning(
                        "entity_embeddings table not found. "
                        "Run migration 003_entity_relations.sql and embed_entities.py"
                    )
                    return neighbors

                # Get embeddings for neighbor entities
                rows = await conn.fetch("""
                    SELECT e.name, ee.embedding
                    FROM entities e
                    JOIN entity_embeddings ee ON e.id = ee.entity_id
                    WHERE e.name = ANY($1)
                      AND ee.embedding IS NOT NULL
                """, neighbors)

            if not rows:
                logger.debug(
                    f"No embeddings found for {len(neighbors)} neighbors. "
                    "Run embed_entities.py to populate."
                )
                return neighbors

            coverage = len(rows) / len(neighbors) * 100
            logger.debug(f"Entity embedding coverage: {coverage:.1f}%")

            # Build embedding lookup
            entity_embeddings = {
                row['name']: np.array(row['embedding'])
                for row in rows
            }

            # Convert question embedding
            query_vec = np.array(question_embedding)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

            # Score each neighbor
            scored = []
            for neighbor in neighbors:
                if neighbor in entity_embeddings:
                    emb = entity_embeddings[neighbor]
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                    similarity = float(np.dot(query_norm, emb_norm))
                    scored.append((neighbor, similarity))
                else:
                    scored.append((neighbor, 0.0))

            # Sort by similarity descending
            scored.sort(key=lambda x: x[1], reverse=True)

            if logger.isEnabledFor(logging.DEBUG):
                top_3 = scored[:3]
                logger.debug(
                    f"Top ranked entities: "
                    f"{[(e, f'{s:.3f}') for e, s in top_3]}"
                )

            return [entity for entity, _ in scored]

        except Exception as e:
            logger.warning(f"Error ranking entities: {e}. Using original order.")
            return neighbors
    
    async def _get_neighbor_chunks(
        self,
        neighbors: list[str],
    ) -> list[UUID]:
        """Get chunk IDs associated with neighbor entities."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT unnest(er.chunk_ids) as chunk_id
                FROM entity_relations er
                JOIN entities e1 ON er.source_entity_id = e1.id
                JOIN entities e2 ON er.target_entity_id = e2.id
                WHERE e1.name = ANY($1) OR e2.name = ANY($1)
            """, neighbors)
            
            return [row['chunk_id'] for row in rows]


# ============================================================================
# SQL Schema Addition (add to schema.sql)
# ============================================================================

SCHEMA_SQL = """
-- Relation extraction schema addition for NeuroSynth

-- Entity relations table (add to existing schema)
CREATE TABLE IF NOT EXISTS entity_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    chunk_ids UUID[] DEFAULT '{}',
    context_snippet TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

-- Indexes for efficient traversal
CREATE INDEX IF NOT EXISTS idx_entity_relations_source 
    ON entity_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_target 
    ON entity_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_type 
    ON entity_relations(relation_type);

-- GIN index for chunk_ids array queries
CREATE INDEX IF NOT EXISTS idx_entity_relations_chunks 
    ON entity_relations USING GIN(chunk_ids);

-- Add source column to entities if not exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'entities' AND column_name = 'source'
    ) THEN
        ALTER TABLE entities ADD COLUMN source TEXT DEFAULT 'manual';
    END IF;
END $$;
"""


# ============================================================================
# Example Integration with engine.py
# ============================================================================

INTEGRATION_EXAMPLE = """
# In rag/engine.py, modify the ask() method:

from relation_pipeline import GraphAugmentedRAG

class RAGEngine:
    def __init__(self, ...):
        ...
        self.graph_rag = GraphAugmentedRAG(self.db_pool)
    
    async def ask(self, question: str, ...) -> str:
        # 1. Extract entities from question
        entities = await self._extract_entities(question)
        
        # 2. Get embedding for question
        question_embedding = await self._embed(question)
        
        # 3. Standard vector search
        search_results = await self._search(question_embedding, limit=10)
        
        # 4. Graph-augmented context (NEW)
        graph_context = await self.graph_rag.get_context(
            entities=entities,
            question_embedding=question_embedding,
            hop_limit=2,
        )
        
        # 5. Fetch graph-related chunks
        graph_chunks = await self._fetch_chunks(graph_context["related_chunks"])
        
        # 6. Merge and deduplicate
        all_chunks = self._merge_chunks(search_results, graph_chunks)
        
        # 7. Build prompt with relationship context
        prompt = self._build_prompt(
            question=question,
            chunks=all_chunks,
            relationships=graph_context["relationships"],  # NEW
        )
        
        # 8. Generate response
        response = await self._generate(prompt)
        
        return response
    
    def _build_prompt(self, question, chunks, relationships):
        # Include relationship context in prompt
        rel_context = ""
        if relationships:
            rel_lines = []
            for r in relationships[:5]:  # Limit to top 5
                rel_lines.append(
                    f"- {r['source']} {r['relation']} {r['target']}"
                )
            rel_context = "\\n\\nRelevant anatomical relationships:\\n" + "\\n".join(rel_lines)
        
        return f'''Answer the following question using the provided context.

Context:
{self._format_chunks(chunks)}
{rel_context}

Question: {question}

Answer:'''
"""


if __name__ == "__main__":
    print("=" * 70)
    print("NEUROSYNTH RELATION PIPELINE - INTEGRATION GUIDE")
    print("=" * 70)
    
    print("\n1. ADD SCHEMA")
    print("-" * 70)
    print(SCHEMA_SQL)
    
    print("\n2. PIPELINE INTEGRATION")
    print("-" * 70)
    print("""
In ingest/pipeline.py:

    from relation_pipeline import RelationExtractionPipeline
    
    async def process_document(doc_path: str):
        relation_pipeline = RelationExtractionPipeline(db_pool)
        
        chunks = extract_chunks(doc_path)
        
        for chunk in chunks:
            chunk_id = await insert_chunk(chunk)
            
            # Extract relations (NEW)
            relations = await relation_pipeline.process_chunk(
                chunk_id, 
                chunk.text
            )
            
            logger.info(f"Extracted {len(relations)} relations from chunk")
        
        # Flush remaining
        await relation_pipeline.flush()
""")
    
    print("\n3. RAG INTEGRATION")
    print("-" * 70)
    print(INTEGRATION_EXAMPLE)
