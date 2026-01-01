"""
Graph-Augmented RAG with Relevance Ranking

Integrates relation extraction, graph traversal, and edge ranking
into the RAG pipeline.

REQUIRED DATABASE SCHEMA:
- entities table: id (UUID), name, aliases (text[])
- entity_relations table: source_entity_id, target_entity_id, relation_type, confidence, context_snippet, chunk_ids

If tables don't exist, traverse_graph returns empty list gracefully.
Run migration 003_entity_relations.sql to create required tables.
"""

from typing import Optional, Callable, Awaitable
from uuid import UUID
import logging

from src.core.relation_extractor import (
    NeuroRelationExtractor,
    ExtractedRelation,
    TAXONOMY,
)
from src.core.edge_ranking import (
    EdgeRelevanceRanker,
    RankedGraphRAG,
    RankingConfig,
    EntityEmbeddingStore,
)

logger = logging.getLogger(__name__)

# Schema verification cache (module-level, reset on restart)
_graph_schema_verified = False
_graph_tables_exist = False


async def _check_graph_tables(db_pool) -> bool:
    """
    Check if entities and entity_relations tables exist.

    Result is cached after first successful check.
    Returns False if tables don't exist or on error.
    """
    global _graph_schema_verified, _graph_tables_exist

    if _graph_schema_verified:
        return _graph_tables_exist

    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'entities'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'entity_relations'
                )
            """)
            _graph_tables_exist = bool(result)
            _graph_schema_verified = True

            if not _graph_tables_exist:
                logger.warning(
                    "Graph tables not found (entities, entity_relations). "
                    "Run migration 003_entity_relations.sql to enable graph features."
                )
            return _graph_tables_exist
    except Exception as e:
        logger.error(f"Error checking graph schema: {e}")
        return False


class GraphRAGContext:
    """
    Central Graph-RAG context manager.

    Usage:
        graph_ctx = GraphRAGContext(db_pool, embed_fn)
        await graph_ctx.initialize()

        context = await graph_ctx.get_context(
            question="What causes aneurysm rupture?",
            entities=["aneurysm"],
        )
    """

    def __init__(
        self,
        db_pool,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        spacy_model: str = "en_core_web_lg",
        ranking_config: Optional[RankingConfig] = None,
    ):
        self.db_pool = db_pool
        self.embed_fn = embed_fn

        # Initialize components
        self.extractor = NeuroRelationExtractor(model=spacy_model)
        self.ranker = EdgeRelevanceRanker(
            embed_fn=embed_fn,
            config=ranking_config or RankingConfig(
                relevance_weight=0.6,
                confidence_weight=0.2,
                hop_penalty_weight=0.2,
                min_relevance=0.3,
                max_edges=10,
            ),
        )
        self.embedding_store = EntityEmbeddingStore(db_pool, embed_fn)
        self._initialized = False

    async def initialize(self):
        """Initialize schema and indexes."""
        if self._initialized:
            return
        await self.embedding_store.ensure_schema()
        self._initialized = True

    def normalize_entity(self, entity: str) -> str:
        """Normalize entity name."""
        return self.extractor.normalize_entity(entity)

    def get_taxonomy_parents(self, entity: str) -> list[str]:
        """Get taxonomy chain for query expansion."""
        entity_norm = self.normalize_entity(entity)
        parents = []
        visited = set()
        queue = [entity_norm]

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

    async def expand_entities(
        self,
        entities: list[str],
        use_taxonomy: bool = True,
        use_semantic: bool = True,
        semantic_limit: int = 5,
        semantic_threshold: float = 0.6,
    ) -> set[str]:
        """
        Expand seed entities via taxonomy and semantic similarity.

        Args:
            entities: Initial entity list
            use_taxonomy: Include taxonomy parents
            use_semantic: Include semantically similar entities

        Returns:
            Expanded set of entity names
        """
        expanded = set()

        for entity in entities:
            norm = self.normalize_entity(entity)
            expanded.add(norm)

            # Taxonomy expansion
            if use_taxonomy:
                parents = self.get_taxonomy_parents(norm)
                expanded.update(parents)

        # Semantic expansion
        if use_semantic:
            try:
                # Get question embedding (use first entity as proxy)
                query_text = " ".join(entities)
                query_embedding = await self.embed_fn(query_text)

                similar = await self.embedding_store.find_similar_entities(
                    query_embedding,
                    limit=semantic_limit,
                    min_similarity=semantic_threshold,
                )

                for name, _ in similar:
                    expanded.add(name)
            except Exception as e:
                logger.warning(f"Semantic expansion failed: {e}")

        return expanded

    async def traverse_graph(
        self,
        entities: list[str],
        hop_limit: int = 2,
    ) -> list[dict]:
        """
        Traverse graph from seed entities.

        Returns list of edge dicts.
        Returns empty list if graph tables don't exist (graceful degradation).
        """
        # Check schema before querying
        if not await _check_graph_tables(self.db_pool):
            logger.debug("Graph tables not available, returning empty edges")
            return []

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
                        er.context_snippet,
                        er.chunk_ids
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
                        "context": row['context_snippet'] or "",
                        "chunk_ids": row['chunk_ids'] or [],
                        "hop": hop + 1,
                    })

                    if row['target_name'] not in entities:
                        next_layer.add(row['target_name'])

                current_layer = next_layer

        return edges

    async def get_context(
        self,
        question: str,
        entities: list[str],
        hop_limit: int = 2,
        use_mmr: bool = True,
        expand_taxonomy: bool = True,
        expand_semantic: bool = True,
    ) -> dict:
        """
        Get ranked graph context for RAG.

        Args:
            question: User's question
            entities: Entities extracted from question
            hop_limit: Max graph traversal depth
            use_mmr: Use MMR for diverse edge selection
            expand_taxonomy: Expand via taxonomy
            expand_semantic: Expand via semantic similarity

        Returns:
            {
                "edges": [...],        # Ranked edge list
                "chunk_ids": [...],    # Chunk IDs from relevant edges
                "entities": [...],     # Expanded entity set
                "prompt_context": "...", # Formatted for prompt injection
            }
        """
        # 1. Expand entities
        expanded = await self.expand_entities(
            entities,
            use_taxonomy=expand_taxonomy,
            use_semantic=expand_semantic,
        )

        # 2. Traverse graph
        raw_edges = await self.traverse_graph(list(expanded), hop_limit)

        if not raw_edges:
            return {
                "edges": [],
                "chunk_ids": [],
                "entities": list(expanded),
                "prompt_context": "",
            }

        # 3. Rank edges
        try:
            if use_mmr:
                ranked_edges = await self.ranker.rank_with_diversity(
                    question,
                    raw_edges,
                    diversity_weight=0.3,
                )
            else:
                ranked_edges = await self.ranker.rank(question, raw_edges)
        except Exception as e:
            logger.warning(f"Edge ranking failed, using unranked: {e}")
            ranked_edges = raw_edges

        # 4. Collect chunk IDs
        chunk_ids = []
        for edge in (ranked_edges if isinstance(ranked_edges, list) and len(ranked_edges) > 0 and isinstance(ranked_edges[0], dict) else []):
            chunk_ids.extend(edge.get("chunk_ids", []))

        # Handle RankedEdge objects
        if ranked_edges and hasattr(ranked_edges[0], 'source'):
            for edge in ranked_edges:
                # Find original edge with chunk_ids
                for raw in raw_edges:
                    if (raw["source"] == edge.source and
                        raw["target"] == edge.target and
                        raw["relation"] == edge.relation):
                        chunk_ids.extend(raw.get("chunk_ids", []))
                        break

        chunk_ids = list(set(chunk_ids))

        # 5. Format for prompt
        prompt_lines = []
        edges_for_prompt = ranked_edges[:5] if ranked_edges else []

        for e in edges_for_prompt:
            if hasattr(e, 'source'):
                # RankedEdge object
                prompt_lines.append(f"- {e.source} {e.relation.replace('_', ' ')} {e.target}")
            else:
                # Dict
                prompt_lines.append(f"- {e['source']} {e['relation'].replace('_', ' ')} {e['target']}")

        prompt_context = ""
        if prompt_lines:
            prompt_context = "Relevant anatomical relationships:\n" + "\n".join(prompt_lines)

        # Format edges for output
        output_edges = []
        for e in ranked_edges:
            if hasattr(e, 'to_dict'):
                output_edges.append(e.to_dict())
            else:
                output_edges.append(e)

        return {
            "edges": output_edges,
            "chunk_ids": chunk_ids,
            "entities": list(expanded),
            "prompt_context": prompt_context,
        }
