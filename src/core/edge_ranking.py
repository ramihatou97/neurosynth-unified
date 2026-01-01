"""
NeuroSynth Edge Relevance Ranking

Ranks graph neighbors by semantic similarity to the query before
injecting into RAG context. Prevents context pollution from irrelevant anatomy.

Problem: User asks "Aneurysm rupture risks"
- Graph neighbors: Coiling, Hemorrhage, Circle of Willis, Endovascular, SAH
- Without ranking: All 5 injected, wastes tokens on "Coiling" (treatment, not risk)
- With ranking: Hemorrhage, SAH ranked highest → prioritized in context
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from enum import Enum


@dataclass
class RankedEdge:
    """An edge with relevance score."""
    source: str
    target: str
    relation: str
    confidence: float
    context_snippet: str
    hop_distance: int
    relevance_score: float = 0.0
    combined_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "context": self.context_snippet,
            "hop": self.hop_distance,
            "relevance": round(self.relevance_score, 3),
            "combined": round(self.combined_score, 3),
        }


class ScoringStrategy(str, Enum):
    """How to combine relevance with other signals."""
    RELEVANCE_ONLY = "relevance_only"
    WEIGHTED_AVERAGE = "weighted_average"
    RELEVANCE_GATED = "relevance_gated"  # Filter then rank by confidence


@dataclass
class RankingConfig:
    """Configuration for edge ranking."""
    # Weights for combined scoring
    relevance_weight: float = 0.6
    confidence_weight: float = 0.2
    hop_penalty_weight: float = 0.2
    
    # Thresholds
    min_relevance: float = 0.3  # Edges below this are filtered out
    max_edges: int = 10  # Maximum edges to return
    
    # Strategy
    strategy: ScoringStrategy = ScoringStrategy.WEIGHTED_AVERAGE
    
    # Hop penalty (closer = better)
    hop_decay: float = 0.7  # Score multiplied by hop_decay^hop_distance


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: np.ndarray, 
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple candidates.
    
    Args:
        query: (D,) query embedding
        candidates: (N, D) candidate embeddings
        
    Returns:
        (N,) similarity scores
    """
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    candidate_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-9)
    return np.dot(candidate_norms, query_norm)


class EdgeRelevanceRanker:
    """
    Ranks graph edges by relevance to a query.
    
    Usage:
        ranker = EdgeRelevanceRanker(embed_fn=your_embed_function)
        
        ranked_edges = await ranker.rank(
            query="What are the risks of aneurysm rupture?",
            edges=[...],  # From graph traversal
        )
    """
    
    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        config: Optional[RankingConfig] = None,
        cache_embeddings: bool = True,
    ):
        """
        Args:
            embed_fn: Async function that returns embedding for text
            config: Ranking configuration
            cache_embeddings: Whether to cache entity embeddings
        """
        self.embed_fn = embed_fn
        self.config = config or RankingConfig()
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = await self.embed_fn(text)
        arr = np.array(embedding)
        
        if self.cache_embeddings:
            self._embedding_cache[text] = arr
        
        return arr
    
    async def _get_edge_embedding(self, edge: RankedEdge) -> np.ndarray:
        """
        Get embedding for an edge.
        
        Combines entity names + relation + context for rich representation.
        """
        # Build edge description
        edge_text = f"{edge.source} {edge.relation} {edge.target}"
        
        # Include context snippet if available
        if edge.context_snippet:
            edge_text += f". {edge.context_snippet[:100]}"
        
        return await self._get_embedding(edge_text)
    
    def _compute_combined_score(self, edge: RankedEdge) -> float:
        """Compute combined score based on strategy."""
        cfg = self.config
        
        if cfg.strategy == ScoringStrategy.RELEVANCE_ONLY:
            return edge.relevance_score
        
        # Hop penalty: closer edges score higher
        hop_score = cfg.hop_decay ** edge.hop_distance
        
        if cfg.strategy == ScoringStrategy.RELEVANCE_GATED:
            # If relevance passes threshold, rank by confidence
            if edge.relevance_score >= cfg.min_relevance:
                return edge.confidence * hop_score
            return 0.0
        
        # WEIGHTED_AVERAGE (default)
        return (
            cfg.relevance_weight * edge.relevance_score +
            cfg.confidence_weight * edge.confidence +
            cfg.hop_penalty_weight * hop_score
        )
    
    async def rank(
        self,
        query: str,
        edges: list[dict],
    ) -> list[RankedEdge]:
        """
        Rank edges by relevance to query.
        
        Args:
            query: The user's question
            edges: List of edge dicts from graph traversal
            
        Returns:
            Sorted list of RankedEdge objects
        """
        if not edges:
            return []
        
        # Convert to RankedEdge objects
        ranked_edges = [
            RankedEdge(
                source=e["source"],
                target=e["target"],
                relation=e["relation"],
                confidence=e.get("confidence", 0.5),
                context_snippet=e.get("context", ""),
                hop_distance=e.get("hop", 1),
            )
            for e in edges
        ]
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Get edge embeddings in parallel
        edge_embeddings = await asyncio.gather(*[
            self._get_edge_embedding(edge) for edge in ranked_edges
        ])
        
        # Stack into matrix for batch computation
        edge_matrix = np.stack(edge_embeddings)
        
        # Compute relevance scores
        relevance_scores = batch_cosine_similarity(query_embedding, edge_matrix)
        
        # Assign scores and compute combined
        for edge, rel_score in zip(ranked_edges, relevance_scores):
            edge.relevance_score = float(rel_score)
            edge.combined_score = self._compute_combined_score(edge)
        
        # Filter by minimum relevance
        ranked_edges = [
            e for e in ranked_edges 
            if e.relevance_score >= self.config.min_relevance
        ]
        
        # Sort by combined score
        ranked_edges.sort(key=lambda e: e.combined_score, reverse=True)
        
        # Limit to max edges
        return ranked_edges[:self.config.max_edges]
    
    async def rank_with_diversity(
        self,
        query: str,
        edges: list[dict],
        diversity_weight: float = 0.3,
    ) -> list[RankedEdge]:
        """
        Rank with Maximal Marginal Relevance (MMR) for diversity.
        
        Prevents returning many edges about the same sub-topic.
        
        Args:
            query: The user's question
            edges: List of edge dicts
            diversity_weight: Balance relevance vs diversity (0=pure relevance, 1=pure diversity)
        """
        if not edges:
            return []
        
        # First get all ranked edges
        all_ranked = await self.rank(query, edges)
        
        if len(all_ranked) <= 1:
            return all_ranked
        
        # MMR selection
        selected: list[RankedEdge] = []
        remaining = list(all_ranked)
        
        # Get embeddings for MMR
        edge_embeddings = {
            e.source + e.target: await self._get_edge_embedding(e)
            for e in remaining
        }
        
        while remaining and len(selected) < self.config.max_edges:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance to query
                relevance = candidate.combined_score
                
                # Diversity: max similarity to already selected
                if selected:
                    candidate_emb = edge_embeddings[candidate.source + candidate.target]
                    max_sim = max(
                        cosine_similarity(
                            candidate_emb,
                            edge_embeddings[s.source + s.target]
                        )
                        for s in selected
                    )
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected


class EntityEmbeddingStore:
    """
    Persistent storage for entity embeddings.
    
    Stores embeddings in PostgreSQL with pgvector for efficient similarity search.
    """
    
    def __init__(self, db_pool, embed_fn: Callable[[str], Awaitable[list[float]]]):
        self.db_pool = db_pool
        self.embed_fn = embed_fn
    
    async def ensure_schema(self):
        """Create embedding storage schema."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                -- Requires pgvector extension
                CREATE EXTENSION IF NOT EXISTS vector;
                
                -- Entity embeddings table
                CREATE TABLE IF NOT EXISTS entity_embeddings (
                    entity_id UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
                    embedding vector(1536),  -- Adjust dimension for your model
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for similarity search
                CREATE INDEX IF NOT EXISTS idx_entity_embeddings_vector
                    ON entity_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
            """)
    
    async def store_embedding(self, entity_id: str, entity_name: str):
        """Compute and store embedding for an entity."""
        embedding = await self.embed_fn(entity_name)
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO entity_embeddings (entity_id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (entity_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
            """, entity_id, embedding)
    
    async def get_embedding(self, entity_id: str) -> Optional[list[float]]:
        """Retrieve stored embedding."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT embedding FROM entity_embeddings
                WHERE entity_id = $1
            """, entity_id)
            
            if row:
                return list(row['embedding'])
            return None
    
    async def find_similar_entities(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """
        Find entities similar to a query embedding.
        
        Returns list of (entity_name, similarity_score) tuples.
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    e.name,
                    1 - (ee.embedding <=> $1) as similarity
                FROM entity_embeddings ee
                JOIN entities e ON ee.entity_id = e.id
                WHERE 1 - (ee.embedding <=> $1) >= $2
                ORDER BY ee.embedding <=> $1
                LIMIT $3
            """, query_embedding, min_similarity, limit)
            
            return [(row['name'], row['similarity']) for row in rows]


# ============================================================================
# Integration with GraphAugmentedRAG
# ============================================================================

class RankedGraphRAG:
    """
    Full Graph-RAG with relevance ranking.
    
    Replaces the basic GraphAugmentedRAG with ranked edge selection.
    """
    
    def __init__(
        self,
        db_pool,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        extractor=None,  # NeuroRelationExtractor
        ranking_config: Optional[RankingConfig] = None,
    ):
        self.db_pool = db_pool
        self.embed_fn = embed_fn
        self.extractor = extractor
        self.ranker = EdgeRelevanceRanker(
            embed_fn=embed_fn,
            config=ranking_config or RankingConfig(),
        )
        self.embedding_store = EntityEmbeddingStore(db_pool, embed_fn)
    
    async def get_context(
        self,
        question: str,
        entities: list[str],
        hop_limit: int = 2,
        use_mmr: bool = True,
    ) -> dict:
        """
        Get ranked graph context for RAG.
        
        Args:
            question: User's question
            entities: Entities extracted from question
            hop_limit: Max traversal depth
            use_mmr: Use MMR for diverse edge selection
            
        Returns:
            Dict with ranked edges and chunk IDs
        """
        # Normalize entities
        if self.extractor:
            entities = [self.extractor.normalize_entity(e) for e in entities]
        
        # Expand with taxonomy
        expanded = set(entities)
        if self.extractor:
            for entity in entities:
                chain = self._get_taxonomy_chain(entity)
                expanded.update(chain)
        
        # Also find semantically similar entities
        question_embedding = await self.embed_fn(question)
        similar_entities = await self.embedding_store.find_similar_entities(
            question_embedding,
            limit=5,
            min_similarity=0.6,
        )
        for name, _ in similar_entities:
            expanded.add(name)
        
        # Traverse graph
        raw_edges = await self._traverse(list(expanded), hop_limit)
        
        # Rank edges
        if use_mmr:
            ranked_edges = await self.ranker.rank_with_diversity(
                question, 
                raw_edges,
                diversity_weight=0.3,
            )
        else:
            ranked_edges = await self.ranker.rank(question, raw_edges)
        
        # Collect chunk IDs from top edges
        chunk_ids = await self._get_edge_chunks(ranked_edges)
        
        return {
            "edges": [e.to_dict() for e in ranked_edges],
            "chunk_ids": chunk_ids,
            "expanded_entities": list(expanded),
            "similar_entities": [name for name, _ in similar_entities],
        }
    
    def _get_taxonomy_chain(self, entity: str) -> list[str]:
        """Get taxonomy parents."""
        from relation_extractor import TAXONOMY
        
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
        entities: list[str],
        hop_limit: int,
    ) -> list[dict]:
        """Traverse graph and return raw edges."""
        edges = []
        current_layer = set(entities)
        visited_edges = set()
        
        async with self.db_pool.acquire() as conn:
            for hop in range(hop_limit):
                if not current_layer:
                    break
                
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
                    edge_key = (row['source_name'], row['target_name'], row['relation_type'])
                    
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)
                    
                    edges.append({
                        "source": row['source_name'],
                        "target": row['target_name'],
                        "relation": row['relation_type'],
                        "confidence": row['confidence'],
                        "context": row['context_snippet'],
                        "hop": hop + 1,
                    })
                    
                    if row['target_name'] not in entities:
                        next_layer.add(row['target_name'])
                
                current_layer = next_layer
        
        return edges
    
    async def _get_edge_chunks(self, edges: list[RankedEdge]) -> list[str]:
        """Get chunk IDs for ranked edges."""
        if not edges:
            return []
        
        sources = [e.source for e in edges]
        targets = [e.target for e in edges]
        all_entities = list(set(sources + targets))
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT unnest(er.chunk_ids) as chunk_id
                FROM entity_relations er
                JOIN entities e1 ON er.source_entity_id = e1.id
                JOIN entities e2 ON er.target_entity_id = e2.id
                WHERE e1.name = ANY($1) OR e2.name = ANY($1)
            """, all_entities)
            
            return [str(row['chunk_id']) for row in rows]


# ============================================================================
# Example Usage
# ============================================================================

async def demo():
    """Demonstrate edge ranking."""
    
    # Mock embedding function (replace with real one)
    async def mock_embed(text: str) -> list[float]:
        # In production: call OpenAI, Voyage, etc.
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        # Generate deterministic pseudo-embedding
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)] * 96  # 1536 dim
    
    # Sample edges from graph traversal
    sample_edges = [
        {
            "source": "aneurysm",
            "target": "hemorrhage",
            "relation": "causes",
            "confidence": 0.9,
            "context": "Aneurysm rupture causes subarachnoid hemorrhage",
            "hop": 1,
        },
        {
            "source": "aneurysm",
            "target": "coiling",
            "relation": "treats",
            "confidence": 0.85,
            "context": "Endovascular coiling treats intracranial aneurysms",
            "hop": 1,
        },
        {
            "source": "aneurysm",
            "target": "circle of willis",
            "relation": "adjacent_to",
            "confidence": 0.7,
            "context": "Aneurysms commonly occur at the circle of Willis",
            "hop": 1,
        },
        {
            "source": "hemorrhage",
            "target": "vasospasm",
            "relation": "causes",
            "confidence": 0.88,
            "context": "SAH leads to delayed cerebral vasospasm",
            "hop": 2,
        },
        {
            "source": "aneurysm",
            "target": "headache",
            "relation": "causes",
            "confidence": 0.75,
            "context": "Sentinel headache may precede aneurysm rupture",
            "hop": 1,
        },
        {
            "source": "coiling",
            "target": "recurrence",
            "relation": "causes",
            "confidence": 0.6,
            "context": "Coiled aneurysms may have recurrence requiring retreatment",
            "hop": 2,
        },
    ]
    
    # Create ranker
    config = RankingConfig(
        relevance_weight=0.6,
        confidence_weight=0.2,
        hop_penalty_weight=0.2,
        min_relevance=0.2,
        max_edges=5,
    )
    
    ranker = EdgeRelevanceRanker(
        embed_fn=mock_embed,
        config=config,
    )
    
    # Query about rupture risks
    query = "What are the risks and complications of aneurysm rupture?"
    
    print("=" * 70)
    print("EDGE RELEVANCE RANKING DEMO")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"\nInput edges: {len(sample_edges)}")
    
    # Rank edges
    ranked = await ranker.rank(query, sample_edges)
    
    print(f"\nRanked edges (top {len(ranked)}):")
    print("-" * 70)
    
    for i, edge in enumerate(ranked, 1):
        print(f"\n{i}. {edge.source} --[{edge.relation}]--> {edge.target}")
        print(f"   Relevance: {edge.relevance_score:.3f}")
        print(f"   Confidence: {edge.confidence:.2f}")
        print(f"   Hop: {edge.hop_distance}")
        print(f"   Combined: {edge.combined_score:.3f}")
        print(f"   Context: {edge.context_snippet[:50]}...")
    
    # Compare with treatment query
    print("\n" + "=" * 70)
    query2 = "How is an aneurysm treated?"
    print(f"\nQuery: {query2}")
    
    ranked2 = await ranker.rank(query2, sample_edges)
    
    print(f"\nRanked edges (top {len(ranked2)}):")
    print("-" * 70)
    
    for i, edge in enumerate(ranked2, 1):
        print(f"\n{i}. {edge.source} --[{edge.relation}]--> {edge.target}")
        print(f"   Relevance: {edge.relevance_score:.3f}")
        print(f"   Combined: {edge.combined_score:.3f}")


if __name__ == "__main__":
    asyncio.run(demo())
