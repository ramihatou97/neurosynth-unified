"""
Knowledge Graph for NeuroSynth v2.0
====================================

NetworkX-based graph for entity relationships and GraphRAG traversal.

Key Features:
- Entity node storage with metadata (category, aliases, chunk refs)
- Relationship edges with types, confidence, and evidence
- GraphRAG queries for multi-hop reasoning
- Persistence via JSON save/load

Expected Improvement: +25% entity query recall
"""

import networkx as nx
import json
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, deque

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the knowledge graph representing an entity."""
    entity_name: str            # Normalized name (unique key)
    category: str               # ANATOMY_VASCULAR, ANATOMY_NEURAL, etc.
    aliases: Set[str] = field(default_factory=set)  # Alternative surface forms
    chunk_ids: Set[str] = field(default_factory=set)  # Chunks mentioning this entity
    document_ids: Set[str] = field(default_factory=set)  # Source documents
    mention_count: int = 0      # Total mentions across all chunks


@dataclass
class EntityRelation:
    """
    Relationship between two entities.

    This is duplicated here to avoid circular imports.
    The canonical definition should be in models.py.
    """
    source_entity: str          # Normalized entity name
    target_entity: str          # Normalized entity name
    relation_type: str          # supplies|innervates|traverses|causes|spatial
    source_category: str        # ANATOMY_VASCULAR, etc.
    target_category: str
    confidence: float           # From extraction
    chunk_id: str               # Source chunk
    document_id: str            # Source document
    context_snippet: str        # Surrounding text evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type,
            "source_cat": self.source_category,
            "target_cat": self.target_category,
            "confidence": self.confidence,
            "chunk_id": self.chunk_id,
            "doc_id": self.document_id,
            "context": self.context_snippet[:200] if self.context_snippet else ""
        }


class NeurosurgicalKnowledgeGraph:
    """
    Directed graph of neurosurgical entities and relationships.

    Supports:
    - Entity node storage with metadata
    - Relationship edges with types and confidence
    - GraphRAG traversal for relationship queries
    - Multi-hop reasoning (e.g., "What does X supply?")

    Example Usage:
    ```python
    kg = NeurosurgicalKnowledgeGraph()

    # Add entities
    kg.add_entity(NeuroEntity(text="MCA", ...), chunk_id="c1", document_id="d1")

    # Add relationships
    kg.add_relation(EntityRelation(
        source_entity="middle cerebral artery",
        target_entity="basal ganglia",
        relation_type="supplies",
        ...
    ))

    # Query: What does MCA supply?
    results = kg.graphrag_query("MCA", ["supplies"])
    ```
    """

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph = nx.DiGraph()
        self._entity_index: Dict[str, GraphNode] = {}  # normalized_name -> node
        self._alias_map: Dict[str, str] = {}  # alias -> normalized_name

        # Statistics
        self._stats = {
            "entities_added": 0,
            "relations_added": 0,
            "duplicate_entities": 0,
            "duplicate_relations": 0
        }

    def add_entity(
        self,
        entity: Any,  # NeuroEntity from models.py
        chunk_id: str,
        document_id: str
    ) -> None:
        """
        Add or update an entity node.

        Args:
            entity: NeuroEntity object with text, normalized, category fields
            chunk_id: ID of the chunk where this entity was found
            document_id: ID of the source document
        """
        # Get normalized name (lowercase for matching)
        normalized = entity.normalized.lower() if hasattr(entity, 'normalized') else entity.text.lower()
        category = entity.category if hasattr(entity, 'category') else "UNKNOWN"
        original_text = entity.text if hasattr(entity, 'text') else normalized

        if normalized not in self._entity_index:
            # Create new node
            node = GraphNode(
                entity_name=normalized,
                category=category
            )
            self._entity_index[normalized] = node
            self.graph.add_node(normalized, **{
                "category": category,
                "display_name": entity.normalized if hasattr(entity, 'normalized') else original_text
            })
            self._stats["entities_added"] += 1
        else:
            self._stats["duplicate_entities"] += 1

        # Update node metadata
        node = self._entity_index[normalized]
        node.chunk_ids.add(chunk_id)
        node.document_ids.add(document_id)
        node.mention_count += 1

        # Track aliases (surface forms that map to normalized name)
        if original_text.lower() != normalized:
            node.aliases.add(original_text.lower())
            self._alias_map[original_text.lower()] = normalized

    def add_relation(self, relation: EntityRelation) -> bool:
        """
        Add a relationship edge between two entities.

        Args:
            relation: EntityRelation object describing the relationship

        Returns:
            True if relation was added, False if entities not found
        """
        # Resolve entity names to their normalized forms (handles aliases like MCA -> middle cerebral artery)
        source = self.resolve_entity(relation.source_entity)
        target = self.resolve_entity(relation.target_entity)

        # Ensure both nodes exist
        if not source:
            logger.debug(f"Source entity not in graph: {relation.source_entity}")
            return False
        if not target:
            logger.debug(f"Target entity not in graph: {relation.target_entity}")
            return False

        if self.graph.has_edge(source, target):
            # Update existing edge - accumulate evidence
            edge_data = self.graph[source][target]
            edge_data["confidence"] = max(
                edge_data.get("confidence", 0),
                relation.confidence
            )
            edge_data["evidence_count"] = edge_data.get("evidence_count", 0) + 1

            # Track all chunk IDs where this relationship was found
            existing_chunks = edge_data.get("chunk_ids", set())
            if isinstance(existing_chunks, list):
                existing_chunks = set(existing_chunks)
            existing_chunks.add(relation.chunk_id)
            edge_data["chunk_ids"] = existing_chunks

            self._stats["duplicate_relations"] += 1
        else:
            # Add new edge
            self.graph.add_edge(
                source, target,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                evidence_count=1,
                chunk_ids={relation.chunk_id},
                context=relation.context_snippet[:200] if relation.context_snippet else ""
            )
            self._stats["relations_added"] += 1

        return True

    def resolve_entity(self, query: str) -> Optional[str]:
        """
        Resolve query text to normalized entity name.

        Tries:
        1. Direct match on normalized name
        2. Alias lookup
        3. Partial match (for abbreviations like MCA)

        Args:
            query: Entity name to resolve

        Returns:
            Normalized entity name or None if not found
        """
        query_lower = query.lower().strip()

        # Direct match
        if query_lower in self._entity_index:
            return query_lower

        # Alias match
        if query_lower in self._alias_map:
            return self._alias_map[query_lower]

        # Partial match with constraints (useful for abbreviations like MCA, ICA)
        # Only match if query is meaningful (3+ chars) and represents significant portion of target
        for normalized, node in self._entity_index.items():
            # Skip very short queries for substring matching (prevents "CA" matching "internal carotid artery")
            if len(query_lower) < 3:
                continue

            # Check word boundary match (e.g., "mca" at start/end of entity name)
            if normalized.startswith(query_lower + " ") or normalized.endswith(" " + query_lower):
                return normalized

            # Check significant overlap (query is >30% of target length)
            if query_lower in normalized:
                match_ratio = len(query_lower) / len(normalized)
                if match_ratio > 0.3:
                    return normalized

            # Check if normalized is fully contained in query (e.g., "mca" in "left mca aneurysm")
            if normalized in query_lower:
                return normalized

            # Check aliases with same constraints
            for alias in node.aliases:
                if query_lower == alias:  # Exact alias match always works
                    return normalized
                if len(query_lower) >= 3 and query_lower in alias:
                    alias_ratio = len(query_lower) / len(alias)
                    if alias_ratio > 0.3:
                        return normalized

        return None

    def get_relationships(
        self,
        entity: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"  # outgoing|incoming|both
    ) -> List[Tuple[str, str, Dict]]:
        """
        Get relationships for an entity.

        Args:
            entity: Entity name (will be resolved)
            relation_type: Optional filter by relation type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (source, target, edge_data) tuples
        """
        resolved = self.resolve_entity(entity)
        if not resolved:
            return []

        results = []

        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(resolved, data=True):
                if relation_type is None or data.get("relation_type") == relation_type:
                    results.append((resolved, target, dict(data)))

        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(resolved, data=True):
                if relation_type is None or data.get("relation_type") == relation_type:
                    results.append((source, resolved, dict(data)))

        return results

    def graphrag_query(
        self,
        start_entity: str,
        relation_chain: List[str],
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Multi-hop GraphRAG query following a chain of relations.

        Examples:
            graphrag_query("MCA", ["supplies"])
                → All structures supplied by MCA

            graphrag_query("MCA", ["supplies", "innervates"])
                → Structures innervated by structures that MCA supplies

        Args:
            start_entity: Starting entity for traversal
            relation_chain: List of relation types to follow in order
            max_depth: Maximum traversal depth

        Returns:
            List of path results with depth, source, relation, target, confidence
        """
        resolved = self.resolve_entity(start_entity)
        if not resolved:
            logger.debug(f"Could not resolve entity: {start_entity}")
            return []

        current_nodes = {resolved}
        path_results = []

        for i, relation_type in enumerate(relation_chain):
            if i >= max_depth:
                break

            next_nodes = set()
            for node in current_nodes:
                for source, target, data in self.graph.out_edges(node, data=True):
                    if data.get("relation_type") == relation_type:
                        next_nodes.add(target)

                        # Get chunk IDs, handling both set and list
                        chunk_ids = data.get("chunk_ids", [])
                        if isinstance(chunk_ids, set):
                            chunk_ids = list(chunk_ids)

                        path_results.append({
                            "depth": i + 1,
                            "source": source,
                            "relation": relation_type,
                            "target": target,
                            "confidence": data.get("confidence", 0),
                            "evidence_count": data.get("evidence_count", 1),
                            "chunk_ids": chunk_ids
                        })

            current_nodes = next_nodes
            if not current_nodes:
                break

        return path_results

    def get_entity_context(
        self,
        entity: str,
        hop_limit: int = 2
    ) -> Dict[str, Any]:
        """
        Get rich context for an entity (for RAG augmentation).

        Returns the entity plus all related entities within hop_limit,
        along with chunk IDs for evidence retrieval.

        Args:
            entity: Entity to get context for
            hop_limit: How many hops to traverse

        Returns:
            Dict with entity info, relationships, and related chunk IDs
        """
        resolved = self.resolve_entity(entity)
        if not resolved:
            return {}

        node = self._entity_index.get(resolved)
        if not node:
            return {}

        context = {
            "entity": resolved,
            "display_name": self.graph.nodes[resolved].get("display_name", resolved),
            "category": node.category,
            "mention_count": node.mention_count,
            "aliases": list(node.aliases),
            "direct_chunks": list(node.chunk_ids),
            "relationships": [],
            "related_chunks": set()
        }

        # BFS to find related entities within hop_limit
        visited = {resolved}
        frontier = deque([(resolved, 0)])  # Use deque for O(1) popleft instead of O(n) list.pop(0)

        while frontier:
            current, depth = frontier.popleft()  # O(1) operation
            if depth >= hop_limit:
                continue

            # Outgoing edges
            for source, target, data in self.graph.out_edges(current, data=True):
                chunk_ids = data.get("chunk_ids", set())
                if isinstance(chunk_ids, list):
                    chunk_ids = set(chunk_ids)

                context["relationships"].append({
                    "from": source,
                    "to": target,
                    "type": data.get("relation_type"),
                    "confidence": data.get("confidence", 0),
                    "depth": depth + 1
                })
                context["related_chunks"].update(chunk_ids)

                if target not in visited:
                    visited.add(target)
                    frontier.append((target, depth + 1))

            # Incoming edges
            for source, _, data in self.graph.in_edges(current, data=True):
                chunk_ids = data.get("chunk_ids", set())
                if isinstance(chunk_ids, list):
                    chunk_ids = set(chunk_ids)

                context["relationships"].append({
                    "from": source,
                    "to": current,
                    "type": data.get("relation_type"),
                    "confidence": data.get("confidence", 0),
                    "depth": depth + 1
                })
                context["related_chunks"].update(chunk_ids)

                if source not in visited:
                    visited.add(source)
                    frontier.append((source, depth + 1))

        context["related_chunks"] = list(context["related_chunks"])
        return context

    def get_neighbors(
        self,
        entity: str,
        relation_type: Optional[str] = None
    ) -> List[str]:
        """
        Get immediate neighbors of an entity.

        Args:
            entity: Entity to get neighbors for
            relation_type: Optional filter by relation type

        Returns:
            List of neighbor entity names
        """
        resolved = self.resolve_entity(entity)
        if not resolved:
            return []

        neighbors = set()

        for _, target, data in self.graph.out_edges(resolved, data=True):
            if relation_type is None or data.get("relation_type") == relation_type:
                neighbors.add(target)

        for source, _, data in self.graph.in_edges(resolved, data=True):
            if relation_type is None or data.get("relation_type") == relation_type:
                neighbors.add(source)

        return list(neighbors)

    def save(self, path: Path) -> None:
        """
        Save graph to JSON file.

        Args:
            path: Path to save JSON file
        """
        data = {
            "version": "1.0",
            "stats": self._stats,
            "nodes": [
                {
                    "name": name,
                    "category": node.category,
                    "aliases": list(node.aliases),
                    "chunk_ids": list(node.chunk_ids),
                    "document_ids": list(node.document_ids),
                    "mention_count": node.mention_count,
                    "display_name": self.graph.nodes[name].get("display_name", name)
                }
                for name, node in self._entity_index.items()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation_type": data.get("relation_type"),
                    "confidence": data.get("confidence"),
                    "evidence_count": data.get("evidence_count"),
                    "chunk_ids": list(data.get("chunk_ids", [])) if isinstance(data.get("chunk_ids"), set) else data.get("chunk_ids", []),
                    "context": data.get("context", "")
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

        path = Path(path)
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved knowledge graph: {len(data['nodes'])} nodes, {len(data['edges'])} edges to {path}")

    def load(self, path: Path) -> None:
        """
        Load graph from JSON file.

        Args:
            path: Path to JSON file
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Knowledge graph file not found: {path}")
            return

        data = json.loads(path.read_text())

        # Clear existing data
        self.graph.clear()
        self._entity_index.clear()
        self._alias_map.clear()

        # Load stats
        self._stats = data.get("stats", self._stats)

        # Reconstruct nodes
        for node_data in data.get("nodes", []):
            name = node_data["name"]
            node = GraphNode(
                entity_name=name,
                category=node_data.get("category", "UNKNOWN"),
                aliases=set(node_data.get("aliases", [])),
                chunk_ids=set(node_data.get("chunk_ids", [])),
                document_ids=set(node_data.get("document_ids", [])),
                mention_count=node_data.get("mention_count", 0)
            )
            self._entity_index[name] = node
            self.graph.add_node(name,
                category=node.category,
                display_name=node_data.get("display_name", name)
            )

            # Rebuild alias map
            for alias in node.aliases:
                self._alias_map[alias] = name

        # Reconstruct edges
        for edge_data in data.get("edges", []):
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                relation_type=edge_data.get("relation_type"),
                confidence=edge_data.get("confidence", 0),
                evidence_count=edge_data.get("evidence_count", 1),
                chunk_ids=set(edge_data.get("chunk_ids", [])),
                context=edge_data.get("context", "")
            )

        logger.info(f"Loaded knowledge graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges from {path}")

    def stats(self) -> Dict[str, Any]:
        """
        Return graph statistics.

        Returns:
            Dict with node count, edge count, category distribution, etc.
        """
        node_categories = Counter(
            node.category for node in self._entity_index.values()
        )

        edge_types = Counter(
            data.get("relation_type", "unknown")
            for _, _, data in self.graph.edges(data=True)
        )

        degrees = dict(self.graph.degree())
        avg_degree = sum(degrees.values()) / max(1, len(degrees))

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_categories": dict(node_categories),
            "edge_types": dict(edge_types),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max(degrees.values()) if degrees else 0,
            "entities_added": self._stats["entities_added"],
            "relations_added": self._stats["relations_added"],
            "aliases_count": len(self._alias_map)
        }

    def __repr__(self) -> str:
        return f"NeurosurgicalKnowledgeGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


# =============================================================================
# TESTING / DEMO
# =============================================================================

def demo():
    """Demonstrate knowledge graph functionality."""

    # Create mock entity class for demo
    class MockEntity:
        def __init__(self, text, normalized, category):
            self.text = text
            self.normalized = normalized
            self.category = category

    # Initialize graph
    kg = NeurosurgicalKnowledgeGraph()

    # Add some neurosurgical entities
    entities = [
        MockEntity("MCA", "middle cerebral artery", "ANATOMY_VASCULAR"),
        MockEntity("middle cerebral artery", "middle cerebral artery", "ANATOMY_VASCULAR"),
        MockEntity("basal ganglia", "basal ganglia", "ANATOMY_NEURAL"),
        MockEntity("internal capsule", "internal capsule", "ANATOMY_NEURAL"),
        MockEntity("sylvian fissure", "sylvian fissure", "ANATOMY_VASCULAR"),
        MockEntity("lenticulostriate arteries", "lenticulostriate arteries", "ANATOMY_VASCULAR"),
    ]

    for i, entity in enumerate(entities):
        kg.add_entity(entity, chunk_id=f"chunk-{i}", document_id="doc-1")

    # Add relationships
    relations = [
        EntityRelation(
            source_entity="middle cerebral artery",
            target_entity="basal ganglia",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.95,
            chunk_id="chunk-1",
            document_id="doc-1",
            context_snippet="The MCA supplies the basal ganglia via lenticulostriate branches"
        ),
        EntityRelation(
            source_entity="middle cerebral artery",
            target_entity="internal capsule",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.92,
            chunk_id="chunk-1",
            document_id="doc-1",
            context_snippet="The MCA also supplies the internal capsule"
        ),
        EntityRelation(
            source_entity="middle cerebral artery",
            target_entity="sylvian fissure",
            relation_type="traverses",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_VASCULAR",
            confidence=0.98,
            chunk_id="chunk-2",
            document_id="doc-1",
            context_snippet="The MCA courses through the sylvian fissure"
        ),
    ]

    for rel in relations:
        kg.add_relation(rel)

    # Demo queries
    print("=" * 60)
    print("Knowledge Graph Demo")
    print("=" * 60)

    print(f"\nGraph stats: {kg.stats()}")

    # Resolve entity
    print(f"\nResolve 'MCA': {kg.resolve_entity('MCA')}")
    print(f"Resolve 'middle cerebral artery': {kg.resolve_entity('middle cerebral artery')}")

    # GraphRAG query
    print(f"\nWhat does MCA supply?")
    supplies = kg.graphrag_query("MCA", ["supplies"])
    for r in supplies:
        print(f"  → {r['target']} (confidence: {r['confidence']:.2f})")

    # Get entity context
    print(f"\nContext for 'middle cerebral artery':")
    context = kg.get_entity_context("middle cerebral artery", hop_limit=2)
    print(f"  Category: {context.get('category')}")
    print(f"  Relationships: {len(context.get('relationships', []))}")
    print(f"  Related chunks: {context.get('related_chunks')}")

    # Save and reload
    test_path = Path("/tmp/test_knowledge_graph.json")
    kg.save(test_path)

    kg2 = NeurosurgicalKnowledgeGraph()
    kg2.load(test_path)
    print(f"\nReloaded graph stats: {kg2.stats()}")

    return kg


if __name__ == "__main__":
    demo()
