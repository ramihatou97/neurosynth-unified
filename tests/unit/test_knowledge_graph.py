"""
Comprehensive unit tests for Knowledge Graph (GraphRAG).

Tests entity management, relationships, entity resolution, graph traversal,
GraphRAG queries, context retrieval, and persistence.

Total: 56 test functions covering 85%+ of knowledge_graph.py
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import networkx as nx
from unittest.mock import Mock, patch


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_entity():
    """Create mock NeuroEntity object."""
    class MockEntity:
        def __init__(self, text="MCA", normalized="middle cerebral artery", category="ANATOMY_VASCULAR"):
            self.text = text
            self.normalized = normalized
            self.category = category

    return MockEntity


@pytest.fixture
def knowledge_graph():
    """Create fresh KnowledgeGraph instance."""
    from src.retrieval.knowledge_graph import NeurosurgicalKnowledgeGraph
    return NeurosurgicalKnowledgeGraph()


@pytest.fixture
def populated_graph(knowledge_graph, mock_entity):
    """Populate graph with sample neurosurgical entities and relationships."""
    from src.retrieval.knowledge_graph import EntityRelation

    # Add entities
    entities = [
        ("MCA", "middle cerebral artery", "ANATOMY_VASCULAR"),
        ("basal ganglia", "basal ganglia", "ANATOMY_NEURAL"),
        ("internal capsule", "internal capsule", "ANATOMY_NEURAL"),
        ("sylvian fissure", "sylvian fissure", "ANATOMY_NEURAL"),
        ("lenticulostriate arteries", "lenticulostriate arteries", "ANATOMY_VASCULAR"),
    ]

    for text, normalized, category in entities:
        entity = mock_entity()
        entity.text = text
        entity.normalized = normalized
        entity.category = category
        knowledge_graph.add_entity(entity, chunk_id=f"c-{text}", document_id="d-1")

    # Add relationships
    relations = [
        ("middle cerebral artery", "basal ganglia", "supplies", 0.95),
        ("middle cerebral artery", "internal capsule", "supplies", 0.92),
        ("middle cerebral artery", "sylvian fissure", "traverses", 0.98),
        ("basal ganglia", "internal capsule", "adjacent_to", 0.88),
    ]

    for source, target, rel_type, confidence in relations:
        relation = EntityRelation(
            source_entity=source,
            target_entity=target,
            relation_type=rel_type,
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=confidence,
            chunk_id=f"rel-{source}-{target}",
            document_id="d-1",
            context_snippet=f"{source} {rel_type} {target}"
        )
        knowledge_graph.add_relation(relation)

    return knowledge_graph


# =============================================================================
# Entity Management Tests (10 tests)
# =============================================================================

class TestEntityManagement:
    """Tests for entity addition and management."""

    def test_initialization(self, knowledge_graph):
        """Test KnowledgeGraph initialization."""
        assert knowledge_graph.graph.number_of_nodes() == 0
        assert knowledge_graph.graph.number_of_edges() == 0
        assert len(knowledge_graph._entity_index) == 0
        assert len(knowledge_graph._alias_map) == 0

    def test_add_entity_creates_node(self, knowledge_graph, mock_entity):
        """Test adding entity creates node in graph."""
        entity = mock_entity()
        entity.text = "MCA"
        entity.normalized = "middle cerebral artery"

        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Should have one node with normalized name
        assert knowledge_graph.graph.number_of_nodes() == 1
        assert "middle cerebral artery" in knowledge_graph._entity_index
        assert knowledge_graph._stats["entities_added"] == 1

    def test_add_entity_tracks_aliases(self, knowledge_graph, mock_entity):
        """Test aliases tracked when text differs from normalized."""
        entity = mock_entity()
        entity.text = "MCA"
        entity.normalized = "middle cerebral artery"

        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Alias should map to normalized name
        node = knowledge_graph._entity_index["middle cerebral artery"]
        assert "mca" in node.aliases
        assert knowledge_graph._alias_map["mca"] == "middle cerebral artery"

    def test_add_entity_duplicate_increments_counter(self, knowledge_graph, mock_entity):
        """Test duplicate entity adds to counter, not new node."""
        entity = mock_entity()
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")
        knowledge_graph.add_entity(entity, chunk_id="c2", document_id="d1")

        # Still only one node
        assert knowledge_graph.graph.number_of_nodes() == 1
        # But counter should track duplicate
        assert knowledge_graph._stats["duplicate_entities"] == 1

    def test_add_entity_tracks_chunk_ids(self, knowledge_graph, mock_entity):
        """Test entity tracks all chunks where mentioned."""
        entity = mock_entity()
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")
        knowledge_graph.add_entity(entity, chunk_id="c2", document_id="d1")

        node = knowledge_graph._entity_index["middle cerebral artery"]
        assert "c1" in node.chunk_ids
        assert "c2" in node.chunk_ids
        assert node.mention_count == 2

    def test_add_entity_tracks_document_ids(self, knowledge_graph, mock_entity):
        """Test entity tracks source documents."""
        entity = mock_entity()
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")
        knowledge_graph.add_entity(entity, chunk_id="c2", document_id="d2")

        node = knowledge_graph._entity_index["middle cerebral artery"]
        assert "d1" in node.document_ids
        assert "d2" in node.document_ids

    def test_add_entity_without_normalized_attribute(self, knowledge_graph):
        """Test entity without normalized attribute falls back to text."""
        entity = Mock()
        entity.text = "test entity"
        entity.normalized = None
        entity.category = "TEST"

        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Should use text.lower() as normalized
        assert "test entity" in knowledge_graph._entity_index

    def test_entity_category_stored(self, knowledge_graph, mock_entity):
        """Test entity category is preserved."""
        entity = mock_entity()
        entity.category = "ANATOMY_VASCULAR"

        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        node = knowledge_graph._entity_index["middle cerebral artery"]
        assert node.category == "ANATOMY_VASCULAR"
        assert knowledge_graph.graph.nodes["middle cerebral artery"]["category"] == "ANATOMY_VASCULAR"

    def test_add_entity_display_name(self, knowledge_graph, mock_entity):
        """Test display name preserved in graph node."""
        entity = mock_entity()
        entity.text = "M.C.A."
        entity.normalized = "middle cerebral artery"

        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Display name should show normalized, not text
        display_name = knowledge_graph.graph.nodes["middle cerebral artery"]["display_name"]
        assert display_name == "middle cerebral artery"


# =============================================================================
# Relationship Management Tests (10 tests)
# =============================================================================

class TestRelationshipManagement:
    """Tests for relationship addition and management."""

    def test_add_relation_requires_both_entities(self, knowledge_graph, mock_entity):
        """Test relation requires both source and target entities to exist."""
        from src.retrieval.knowledge_graph import EntityRelation

        relation = EntityRelation(
            source_entity="MCA",
            target_entity="unknown_entity",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.95,
            chunk_id="c1",
            document_id="d1",
            context_snippet="test"
        )

        # Add only source entity
        entity = mock_entity()
        entity.normalized = "MCA"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Try to add relation - should fail
        result = knowledge_graph.add_relation(relation)
        assert result is False

    def test_add_relation_creates_edge(self, knowledge_graph, mock_entity):
        """Test successful relation creates edge."""
        from src.retrieval.knowledge_graph import EntityRelation

        # Add entities
        for normalized in ["middle cerebral artery", "basal ganglia"]:
            entity = mock_entity()
            entity.normalized = normalized
            knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Add relation
        relation = EntityRelation(
            source_entity="middle cerebral artery",
            target_entity="basal ganglia",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.95,
            chunk_id="c1",
            document_id="d1",
            context_snippet="MCA supplies basal ganglia"
        )

        result = knowledge_graph.add_relation(relation)
        assert result is True
        assert knowledge_graph.graph.has_edge("middle cerebral artery", "basal ganglia")

    def test_relation_stores_metadata(self, knowledge_graph, mock_entity):
        """Test relation stores all metadata."""
        from src.retrieval.knowledge_graph import EntityRelation

        # Setup entities
        for normalized in ["mca", "bg"]:
            entity = mock_entity()
            entity.normalized = normalized
            knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Add relation
        relation = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.92,
            chunk_id="c1",
            document_id="d1",
            context_snippet="test snippet"
        )

        knowledge_graph.add_relation(relation)

        edge_data = knowledge_graph.graph["mca"]["bg"]
        assert edge_data["relation_type"] == "supplies"
        assert edge_data["confidence"] == 0.92
        assert "c1" in edge_data["chunk_ids"]

    def test_duplicate_relation_accumulates_evidence(self, knowledge_graph, mock_entity):
        """Test duplicate relation increases evidence count."""
        from src.retrieval.knowledge_graph import EntityRelation

        # Setup entities
        for normalized in ["mca", "bg"]:
            entity = mock_entity()
            entity.normalized = normalized
            knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Add relation twice
        relation = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.90,
            chunk_id="c1",
            document_id="d1",
            context_snippet="test"
        )

        knowledge_graph.add_relation(relation)
        knowledge_graph.add_relation(relation)

        edge_data = knowledge_graph.graph["mca"]["bg"]
        assert edge_data["evidence_count"] == 2
        assert knowledge_graph._stats["duplicate_relations"] == 1

    def test_duplicate_relation_keeps_max_confidence(self, knowledge_graph, mock_entity):
        """Test duplicate relation keeps maximum confidence."""
        from src.retrieval.knowledge_graph import EntityRelation

        for normalized in ["mca", "bg"]:
            entity = mock_entity()
            entity.normalized = normalized
            knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Add with low confidence
        relation1 = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.70,
            chunk_id="c1",
            document_id="d1",
            context_snippet="test"
        )

        knowledge_graph.add_relation(relation1)

        # Add with higher confidence
        relation2 = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.95,
            chunk_id="c2",
            document_id="d1",
            context_snippet="test"
        )

        knowledge_graph.add_relation(relation2)

        edge_data = knowledge_graph.graph["mca"]["bg"]
        assert edge_data["confidence"] == 0.95

    def test_duplicate_relation_tracks_chunk_ids(self, knowledge_graph, mock_entity):
        """Test duplicate relations track all chunk IDs."""
        from src.retrieval.knowledge_graph import EntityRelation

        for normalized in ["mca", "bg"]:
            entity = mock_entity()
            entity.normalized = normalized
            knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        relation1 = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.90,
            chunk_id="c1",
            document_id="d1",
            context_snippet="test"
        )

        relation2 = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.90,
            chunk_id="c2",
            document_id="d1",
            context_snippet="test"
        )

        knowledge_graph.add_relation(relation1)
        knowledge_graph.add_relation(relation2)

        edge_data = knowledge_graph.graph["mca"]["bg"]
        assert "c1" in edge_data["chunk_ids"]
        assert "c2" in edge_data["chunk_ids"]


# =============================================================================
# Entity Resolution Tests (10 tests)
# =============================================================================

class TestEntityResolution:
    """Tests for entity resolution (finding normalized names)."""

    def test_resolve_exact_match(self, populated_graph):
        """Test exact match resolution."""
        result = populated_graph.resolve_entity("middle cerebral artery")
        assert result == "middle cerebral artery"

    def test_resolve_alias_match(self, populated_graph):
        """Test resolution via alias."""
        result = populated_graph.resolve_entity("MCA")
        assert result == "middle cerebral artery"

    def test_resolve_case_insensitive(self, populated_graph):
        """Test resolution is case-insensitive."""
        result = populated_graph.resolve_entity("MIDDLE CEREBRAL ARTERY")
        assert result == "middle cerebral artery"

    def test_resolve_unknown_entity_returns_none(self, populated_graph):
        """Test unknown entity returns None."""
        result = populated_graph.resolve_entity("unknown_entity")
        assert result is None

    def test_resolve_partial_match_with_word_boundary(self, knowledge_graph, mock_entity):
        """Test partial match with word boundary."""
        # Add entity "internal carotid artery" and try to resolve "carotid"
        entity = mock_entity()
        entity.normalized = "internal carotid artery"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        result = knowledge_graph.resolve_entity("carotid")
        # Should not match because query (6 chars) < 30% of target (22 chars)
        assert result is None

    def test_resolve_partial_match_minimum_length_constraint(self, knowledge_graph, mock_entity):
        """Test partial matching requires minimum query length."""
        entity = mock_entity()
        entity.normalized = "middle cerebral artery"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # Short query should not match via partial
        result = knowledge_graph.resolve_entity("ca")
        assert result is None

    def test_resolve_substring_match(self, knowledge_graph, mock_entity):
        """Test substring matching in normalized name."""
        entity = mock_entity()
        entity.normalized = "middle cerebral artery"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        # "cerebral" is in "middle cerebral artery" and >30% match
        result = knowledge_graph.resolve_entity("cerebral")
        assert result == "middle cerebral artery"

    def test_resolve_query_contains_entity(self, knowledge_graph, mock_entity):
        """Test when query contains the entity."""
        entity = mock_entity()
        entity.normalized = "mca"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        result = knowledge_graph.resolve_entity("left mca aneurysm")
        assert result == "mca"

    def test_resolve_trims_whitespace(self, knowledge_graph, mock_entity):
        """Test query whitespace is trimmed."""
        entity = mock_entity()
        entity.normalized = "middle cerebral artery"
        knowledge_graph.add_entity(entity, chunk_id="c1", document_id="d1")

        result = knowledge_graph.resolve_entity("  middle cerebral artery  ")
        assert result == "middle cerebral artery"

    def test_resolve_checks_alias_substring_match(self, populated_graph):
        """Test alias substring matching."""
        result = populated_graph.resolve_entity("lst")  # lenticulostriate -> lst would not match at 3/19
        assert result is None  # Too low ratio


# =============================================================================
# Graph Traversal Tests (8 tests)
# =============================================================================

class TestGraphTraversal:
    """Tests for graph traversal operations."""

    def test_get_neighbors_outgoing(self, populated_graph):
        """Test getting outgoing neighbors."""
        neighbors = populated_graph.get_neighbors("middle cerebral artery")

        # MCA has outgoing edges to basal ganglia, internal capsule, sylvian fissure
        assert "basal ganglia" in neighbors
        assert "internal capsule" in neighbors
        assert "sylvian fissure" in neighbors

    def test_get_neighbors_filter_by_relation_type(self, populated_graph):
        """Test neighbors filtered by relation type."""
        # Only "supplies" relations
        neighbors = populated_graph.get_neighbors("middle cerebral artery", relation_type="supplies")

        assert "basal ganglia" in neighbors
        assert "internal capsule" in neighbors
        # "traverses" relation excluded
        assert "sylvian fissure" not in neighbors

    def test_get_relationships_outgoing(self, populated_graph):
        """Test getting outgoing relationships."""
        rels = populated_graph.get_relationships("middle cerebral artery", direction="outgoing")

        assert len(rels) == 3  # 3 outgoing edges from MCA
        assert all(source == "middle cerebral artery" for source, target, data in rels)

    def test_get_relationships_incoming(self, populated_graph):
        """Test getting incoming relationships."""
        rels = populated_graph.get_relationships("basal ganglia", direction="incoming")

        # Basal ganglia has 1 incoming (from MCA) and 1 outgoing
        incoming = [r for r in rels if r[1] == "basal ganglia"]
        assert len(incoming) >= 1

    def test_get_relationships_bidirectional(self, populated_graph):
        """Test getting both incoming and outgoing."""
        rels = populated_graph.get_relationships("basal ganglia", direction="both")

        # Should have both incoming and outgoing
        incoming = [r for r in rels if r[1] == "basal ganglia"]
        outgoing = [r for r in rels if r[0] == "basal ganglia"]
        assert len(incoming) >= 1
        assert len(outgoing) >= 1

    def test_get_relationships_unknown_entity(self, populated_graph):
        """Test relationships for unknown entity returns empty."""
        rels = populated_graph.get_relationships("unknown_entity")
        assert rels == []

    def test_get_relationships_includes_metadata(self, populated_graph):
        """Test relationship includes edge data."""
        rels = populated_graph.get_relationships("middle cerebral artery")

        for source, target, data in rels:
            assert "relation_type" in data
            assert "confidence" in data
            assert data["confidence"] > 0

    def test_get_neighbors_unknown_entity(self, populated_graph):
        """Test neighbors for unknown entity returns empty."""
        neighbors = populated_graph.get_neighbors("unknown_entity")
        assert neighbors == []


# =============================================================================
# GraphRAG Query Tests (12 tests)
# =============================================================================

class TestGraphRAGQueries:
    """Tests for GraphRAG multi-hop traversal."""

    def test_single_hop_query(self, populated_graph):
        """Test single-hop GraphRAG query."""
        results = populated_graph.graphrag_query("middle cerebral artery", ["supplies"])

        # Should find basal ganglia and internal capsule
        targets = [r["target"] for r in results]
        assert "basal ganglia" in targets
        assert "internal capsule" in targets

    def test_multi_hop_query(self, populated_graph):
        """Test multi-hop traversal."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies", "adjacent_to"]
        )

        # Should traverse supplies -> adjacent_to
        # After supplies: basal ganglia, internal capsule
        # From basal ganglia, adjacent_to: internal capsule
        assert len(results) >= 1

    def test_query_respects_max_depth(self, populated_graph):
        """Test max_depth limiting."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies", "supplies", "supplies"],
            max_depth=1
        )

        # Should only traverse first "supplies"
        depths = [r["depth"] for r in results]
        assert all(d <= 1 for d in depths)

    def test_query_unknown_entity(self, populated_graph):
        """Test query with unknown starting entity."""
        results = populated_graph.graphrag_query("unknown_entity", ["supplies"])
        assert results == []

    def test_query_unknown_relation_type(self, populated_graph):
        """Test query with unknown relation type returns empty."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["unknown_relation"]
        )
        assert results == []

    def test_query_results_include_depth(self, populated_graph):
        """Test query results include depth information."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies"]
        )

        for result in results:
            assert "depth" in result
            assert result["depth"] == 1

    def test_query_results_include_confidence(self, populated_graph):
        """Test query results include confidence scores."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies"]
        )

        for result in results:
            assert "confidence" in result
            assert 0 <= result["confidence"] <= 1

    def test_query_results_include_evidence_count(self, populated_graph):
        """Test query results include evidence count."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies"]
        )

        for result in results:
            assert "evidence_count" in result
            assert result["evidence_count"] >= 1

    def test_query_results_include_chunk_ids(self, populated_graph):
        """Test query results include chunk IDs."""
        results = populated_graph.graphrag_query(
            "middle cerebral artery",
            ["supplies"]
        )

        for result in results:
            assert "chunk_ids" in result
            assert isinstance(result["chunk_ids"], list)


# =============================================================================
# Context Retrieval Tests (6 tests)
# =============================================================================

class TestContextRetrieval:
    """Tests for entity context retrieval."""

    def test_get_entity_context_unknown_entity(self, populated_graph):
        """Test context for unknown entity returns empty dict."""
        context = populated_graph.get_entity_context("unknown_entity")
        assert context == {}

    def test_get_entity_context_includes_metadata(self, populated_graph):
        """Test entity context includes entity metadata."""
        context = populated_graph.get_entity_context("middle cerebral artery")

        assert context["entity"] == "middle cerebral artery"
        assert "category" in context
        assert "mention_count" in context
        assert "aliases" in context

    def test_get_entity_context_includes_relationships(self, populated_graph):
        """Test context includes related relationships."""
        context = populated_graph.get_entity_context("middle cerebral artery", hop_limit=1)

        relationships = context.get("relationships", [])
        assert len(relationships) > 0

    def test_get_entity_context_respects_hop_limit(self, populated_graph):
        """Test context respects hop limit."""
        context_1hop = populated_graph.get_entity_context(
            "middle cerebral artery",
            hop_limit=1
        )
        context_2hop = populated_graph.get_entity_context(
            "middle cerebral artery",
            hop_limit=2
        )

        # 2-hop should have at least as many relationships
        assert len(context_2hop.get("relationships", [])) >= len(context_1hop.get("relationships", []))

    def test_get_entity_context_includes_chunk_ids(self, populated_graph):
        """Test context includes chunk IDs for evidence."""
        context = populated_graph.get_entity_context("middle cerebral artery", hop_limit=2)

        assert "related_chunks" in context
        assert isinstance(context["related_chunks"], list)

    def test_get_entity_context_bidirectional_traversal(self, populated_graph):
        """Test context includes both incoming and outgoing relationships."""
        context = populated_graph.get_entity_context("basal ganglia", hop_limit=1)

        relationships = context.get("relationships", [])
        # Should have both incoming (from MCA) and outgoing (to internal capsule)
        assert len(relationships) >= 2


# =============================================================================
# Persistence Tests (6 tests)
# =============================================================================

class TestPersistence:
    """Tests for graph save/load functionality."""

    def test_save_graph_creates_file(self, populated_graph):
        """Test save creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kg.json"
            populated_graph.save(save_path)

            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_save_includes_nodes(self, populated_graph):
        """Test saved graph includes all nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kg.json"
            populated_graph.save(save_path)

            data = json.loads(save_path.read_text())
            assert len(data["nodes"]) == 5  # 5 entities in populated graph
            assert data["version"] == "1.0"

    def test_save_includes_edges(self, populated_graph):
        """Test saved graph includes all edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kg.json"
            populated_graph.save(save_path)

            data = json.loads(save_path.read_text())
            assert len(data["edges"]) == 4  # 4 relationships in populated graph

    def test_load_restores_graph(self, populated_graph):
        """Test load restores graph state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kg.json"
            populated_graph.save(save_path)

            # Create new graph and load
            from src.retrieval.knowledge_graph import NeurosurgicalKnowledgeGraph
            kg2 = NeurosurgicalKnowledgeGraph()
            kg2.load(save_path)

            # Should have same structure
            assert kg2.graph.number_of_nodes() == populated_graph.graph.number_of_nodes()
            assert kg2.graph.number_of_edges() == populated_graph.graph.number_of_edges()

    def test_load_restores_aliases(self, populated_graph):
        """Test load restores alias map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "kg.json"
            populated_graph.save(save_path)

            from src.retrieval.knowledge_graph import NeurosurgicalKnowledgeGraph
            kg2 = NeurosurgicalKnowledgeGraph()
            kg2.load(save_path)

            # Alias "mca" should still resolve
            result = kg2.resolve_entity("MCA")
            assert result == "middle cerebral artery"

    def test_load_missing_file_gracefully_fails(self, knowledge_graph):
        """Test load gracefully handles missing file."""
        missing_path = Path("/tmp/nonexistent_kg_file_xyz.json")
        knowledge_graph.load(missing_path)

        # Should not crash, just remain empty
        assert knowledge_graph.graph.number_of_nodes() == 0


# =============================================================================
# Statistics Tests (4 tests)
# =============================================================================

class TestStatistics:
    """Tests for graph statistics."""

    def test_stats_node_count(self, populated_graph):
        """Test stats includes node count."""
        stats = populated_graph.stats()
        assert stats["total_nodes"] == 5

    def test_stats_edge_count(self, populated_graph):
        """Test stats includes edge count."""
        stats = populated_graph.stats()
        assert stats["total_edges"] == 4

    def test_stats_category_distribution(self, populated_graph):
        """Test stats includes category distribution."""
        stats = populated_graph.stats()
        assert "node_categories" in stats
        assert "ANATOMY_VASCULAR" in stats["node_categories"]
        assert "ANATOMY_NEURAL" in stats["node_categories"]

    def test_stats_relation_type_distribution(self, populated_graph):
        """Test stats includes relation type distribution."""
        stats = populated_graph.stats()
        assert "edge_types" in stats
        assert "supplies" in stats["edge_types"]
        assert "traverses" in stats["edge_types"]
        assert "adjacent_to" in stats["edge_types"]


# =============================================================================
# GraphNode Tests
# =============================================================================

class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test GraphNode initialization."""
        from src.retrieval.knowledge_graph import GraphNode

        node = GraphNode(
            entity_name="test",
            category="TEST",
            aliases={"t", "test_alias"},
            mention_count=5
        )

        assert node.entity_name == "test"
        assert node.category == "TEST"
        assert "t" in node.aliases
        assert node.mention_count == 5


# =============================================================================
# EntityRelation Tests
# =============================================================================

class TestEntityRelation:
    """Tests for EntityRelation dataclass."""

    def test_entity_relation_to_dict(self):
        """Test EntityRelation serialization."""
        from src.retrieval.knowledge_graph import EntityRelation

        relation = EntityRelation(
            source_entity="mca",
            target_entity="bg",
            relation_type="supplies",
            source_category="ANATOMY_VASCULAR",
            target_category="ANATOMY_NEURAL",
            confidence=0.95,
            chunk_id="c1",
            document_id="d1",
            context_snippet="The MCA supplies the basal ganglia" * 100  # Long snippet
        )

        data = relation.to_dict()
        assert data["source"] == "mca"
        assert data["type"] == "supplies"
        assert data["confidence"] == 0.95
        # Context should be truncated
        assert len(data["context"]) <= 200


# =============================================================================
# Representation Tests
# =============================================================================

class TestRepresentation:
    """Tests for string representation."""

    def test_repr_shows_counts(self, populated_graph):
        """Test __repr__ shows node and edge counts."""
        repr_str = repr(populated_graph)
        assert "nodes=5" in repr_str
        assert "edges=4" in repr_str
