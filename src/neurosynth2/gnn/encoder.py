"""
NeuroSynth 2.0 - Graph Encoder
===============================

Converts database entities and relationships into PyTorch Geometric graph tensors.

This module bridges the gap between:
- PostgreSQL anatomical_entities table
- PostgreSQL causal_edges table
- PyTorch Geometric Data objects

Components:
- EntityVocabulary: Maps entity names to integer IDs
- PropertyEncoder: Encodes categorical physics properties
- GraphBuilder: Constructs PyG Data objects from DB records

Usage:
    encoder = GraphEncoder(database)
    await encoder.initialize()
    
    # Build graph for a surgical corridor
    graph = await encoder.build_corridor_graph("pterional")
    
    # Build full anatomical graph
    full_graph = await encoder.build_full_graph()
"""

import logging
from typing import Optional, List, Dict, Tuple, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field
import json

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None
    Batch = None


# =============================================================================
# VOCABULARY & ENCODING MAPS
# =============================================================================

# Mobility types
MOBILITY_VOCAB = {
    "fixed": 0,
    "slightly_mobile": 1,
    "elastic": 2,
    "tethered_by_perforators": 3,
    "freely_mobile": 4,
    "pulsatile": 5,
    "compressible": 6,
    "unknown": 7,
}

# Consistency types
CONSISTENCY_VOCAB = {
    "vascular": 0,
    "neural": 1,
    "soft_brain": 2,
    "firm_tumor": 3,
    "fibrous": 4,
    "calcified": 5,
    "solid_bone": 6,
    "cartilaginous": 7,
    "membranous": 8,
    "fluid_filled": 9,
    "mixed": 10,
    "unknown": 11,
}

# Eloquence grades
ELOQUENCE_VOCAB = {
    "non_eloquent": 0,
    "near_eloquent": 1,
    "eloquent": 2,
    "unknown": 3,
}

# Retraction tolerance
RETRACTION_VOCAB = {
    "none": 0,
    "minimal": 1,
    "moderate": 2,
    "significant": 3,
    "unlimited": 4,
    "unknown": 5,
}

# Collateral capacity
COLLATERAL_VOCAB = {
    "none": 0,
    "poor": 1,
    "moderate": 2,
    "good": 3,
    "excellent": 4,
    "unknown": 5,
}

# Relation types (for edges)
RELATION_VOCAB = {
    "supplies": 0,
    "drains": 1,
    "innervates": 2,
    "adjacent_to": 3,
    "contained_in": 4,
    "penetrates": 5,
    "wraps_around": 6,
    "supports": 7,
    "separates": 8,
    "connects": 9,
    "branches_from": 10,
    "merges_into": 11,
    "compresses": 12,
    "displaces": 13,
    "protects": 14,
    "covers": 15,
    "traverses": 16,
    "originates_from": 17,
    "terminates_at": 18,
    "anastomoses_with": 19,
    "unknown": 20,
}

# Surgical actions
ACTION_VOCAB = {
    "dissect": 0,
    "retract": 1,
    "coagulate": 2,
    "clip": 3,
    "cut": 4,
    "sacrifice": 5,
    "mobilize": 6,
    "preserve": 7,
    "resect": 8,
    "drill": 9,
    "open": 10,
    "identify": 11,
    "stimulate": 12,
    "expose": 13,
    "unknown": 14,
}


# =============================================================================
# ENTITY VOCABULARY
# =============================================================================

class EntityVocabulary:
    """
    Maps entity names to integer IDs for embedding lookup.
    
    Supports:
    - Known entities from database
    - Dynamic OOV (out-of-vocabulary) handling
    - Alias resolution
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._aliases: Dict[str, str] = {}
        
        # Reserved tokens
        self._name_to_id["<PAD>"] = 0
        self._name_to_id["<UNK>"] = 1
        self._id_to_name[0] = "<PAD>"
        self._id_to_name[1] = "<UNK>"
        
        self._next_id = 2
    
    def add(self, name: str, aliases: Optional[List[str]] = None) -> int:
        """Add entity to vocabulary."""
        normalized = self._normalize(name)
        
        if normalized in self._name_to_id:
            return self._name_to_id[normalized]
        
        if self._next_id >= self.max_size:
            logger.warning(f"Vocabulary full, returning UNK for: {name}")
            return 1  # UNK
        
        entity_id = self._next_id
        self._name_to_id[normalized] = entity_id
        self._id_to_name[entity_id] = normalized
        self._next_id += 1
        
        # Add aliases
        if aliases:
            for alias in aliases:
                self._aliases[self._normalize(alias)] = normalized
        
        return entity_id
    
    def get_id(self, name: str) -> int:
        """Get ID for entity name (resolves aliases)."""
        normalized = self._normalize(name)
        
        # Check aliases first
        if normalized in self._aliases:
            normalized = self._aliases[normalized]
        
        return self._name_to_id.get(normalized, 1)  # Default to UNK
    
    def get_name(self, entity_id: int) -> str:
        """Get entity name from ID."""
        return self._id_to_name.get(entity_id, "<UNK>")
    
    def _normalize(self, name: str) -> str:
        """Normalize entity name."""
        return name.lower().strip().replace(" ", "_")
    
    def __len__(self) -> int:
        return self._next_id
    
    def save(self, path: str):
        """Save vocabulary to JSON."""
        data = {
            "name_to_id": self._name_to_id,
            "aliases": self._aliases,
            "next_id": self._next_id,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._name_to_id = data["name_to_id"]
        self._id_to_name = {int(v): k for k, v in self._name_to_id.items()}
        self._aliases = data["aliases"]
        self._next_id = data["next_id"]


# =============================================================================
# PROPERTY ENCODER
# =============================================================================

class PropertyEncoder:
    """Encodes categorical and continuous entity properties."""
    
    def __init__(self):
        self.mobility_map = MOBILITY_VOCAB
        self.consistency_map = CONSISTENCY_VOCAB
        self.eloquence_map = ELOQUENCE_VOCAB
        self.retraction_map = RETRACTION_VOCAB
        self.collateral_map = COLLATERAL_VOCAB
        self.relation_map = RELATION_VOCAB
        self.action_map = ACTION_VOCAB
    
    def encode_mobility(self, value: str) -> int:
        return self.mobility_map.get(value.lower() if value else "unknown", 
                                      self.mobility_map["unknown"])
    
    def encode_consistency(self, value: str) -> int:
        return self.consistency_map.get(value.lower() if value else "unknown",
                                         self.consistency_map["unknown"])
    
    def encode_eloquence(self, value: str) -> int:
        return self.eloquence_map.get(value.lower() if value else "unknown",
                                       self.eloquence_map["unknown"])
    
    def encode_retraction(self, value: str) -> int:
        return self.retraction_map.get(value.lower() if value else "unknown",
                                        self.retraction_map["unknown"])
    
    def encode_collateral(self, value: str) -> int:
        return self.collateral_map.get(value.lower() if value else "unknown",
                                        self.collateral_map["unknown"])
    
    def encode_relation(self, value: str) -> int:
        return self.relation_map.get(value.lower() if value else "unknown",
                                      self.relation_map["unknown"])
    
    def encode_action(self, value: str) -> int:
        return self.action_map.get(value.lower() if value else "unknown",
                                    self.action_map["unknown"])
    
    def encode_flags(
        self,
        is_end_artery: bool = False,
        has_collaterals: bool = True,
        is_tethered: bool = False,
        is_eloquent: bool = False,
    ) -> List[float]:
        """Encode boolean flags as float vector."""
        return [
            float(is_end_artery),
            float(has_collaterals),
            float(is_tethered),
            float(is_eloquent),
        ]
    
    def encode_position(
        self,
        spatial_context: Optional[Dict]
    ) -> List[float]:
        """Extract position from spatial context."""
        if not spatial_context:
            return [0.0, 0.0, 0.0]
        
        # Try different keys
        if "centroid" in spatial_context:
            return list(spatial_context["centroid"])[:3]
        if "coordinates" in spatial_context:
            return list(spatial_context["coordinates"])[:3]
        if "position" in spatial_context:
            return list(spatial_context["position"])[:3]
        
        return [0.0, 0.0, 0.0]


# =============================================================================
# GRAPH BUILDER
# =============================================================================

@dataclass
class EntityNode:
    """Intermediate representation of an entity node."""
    name: str
    entity_id: int
    mobility: int
    consistency: int
    eloquence: int
    diameter: float
    confidence: float
    position: List[float]
    flags: List[float]


@dataclass
class RelationEdge:
    """Intermediate representation of a relation edge."""
    source_idx: int
    target_idx: int
    relation_type: int
    probability: float
    distance: float


class GraphEncoder:
    """
    Builds PyTorch Geometric graphs from database entities.
    
    Converts anatomical_entities and causal_edges tables into
    graph tensors suitable for GNN processing.
    """
    
    def __init__(self, database):
        self.database = database
        self.vocabulary = EntityVocabulary()
        self.property_encoder = PropertyEncoder()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize encoder, building vocabulary from database."""
        try:
            # Load all entities to build vocabulary
            rows = await self.database.fetch("""
                SELECT name, aliases FROM anatomical_entities
            """)
            
            for row in rows:
                aliases = row["aliases"] if row["aliases"] else []
                self.vocabulary.add(row["name"], aliases)
            
            logger.info(f"GraphEncoder initialized with {len(self.vocabulary)} entities")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"GraphEncoder initialization failed: {e}")
            return False
    
    async def build_full_graph(self) -> Optional[Data]:
        """
        Build complete anatomical graph from database.
        
        Includes all entities and their relationships.
        """
        if not TORCH_AVAILABLE or not PYG_AVAILABLE:
            logger.error("PyTorch/PyG not available")
            return None
        
        # Load entities
        entity_rows = await self.database.fetch("""
            SELECT * FROM anatomical_entities
            ORDER BY id
        """)
        
        if not entity_rows:
            logger.warning("No entities found in database")
            return None
        
        # Build node features
        nodes = []
        name_to_idx = {}
        
        for idx, row in enumerate(entity_rows):
            name = row["name"]
            name_to_idx[name.lower()] = idx
            
            # Decode spatial context
            spatial = row["spatial_context"]
            if isinstance(spatial, str):
                spatial = json.loads(spatial)
            
            node = EntityNode(
                name=name,
                entity_id=self.vocabulary.get_id(name),
                mobility=self.property_encoder.encode_mobility(row["mobility"]),
                consistency=self.property_encoder.encode_consistency(row["consistency"]),
                eloquence=self.property_encoder.encode_eloquence(row["eloquence_grade"]),
                diameter=float(row["vessel_diameter_mm"] or 0.0),
                confidence=float(row["confidence"] or 0.5),
                position=self.property_encoder.encode_position(spatial),
                flags=self.property_encoder.encode_flags(
                    is_end_artery=bool(row["is_end_artery"]),
                    has_collaterals=row["collateral_capacity"] not in ("none", None),
                    is_tethered="tethered" in str(row["mobility"] or "").lower(),
                    is_eloquent=row["eloquence_grade"] == "eloquent",
                ),
            )
            nodes.append(node)
        
        # Load edges
        edge_rows = await self.database.fetch("""
            SELECT source_entity, target_entity, relation_type, 
                   probability, mechanism_chain
            FROM causal_edges
        """)
        
        edges = []
        for row in edge_rows:
            source_name = row["source_entity"].lower()
            target_name = row["target_entity"].lower()
            
            if source_name not in name_to_idx or target_name not in name_to_idx:
                continue
            
            edges.append(RelationEdge(
                source_idx=name_to_idx[source_name],
                target_idx=name_to_idx[target_name],
                relation_type=self.property_encoder.encode_relation(row["relation_type"]),
                probability=float(row["probability"] or 0.5),
                distance=0.0,  # Would compute from positions
            ))
        
        # Convert to tensors
        return self._build_pyg_data(nodes, edges)
    
    async def build_corridor_graph(
        self,
        corridor_name: str,
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> Optional[Data]:
        """
        Build subgraph for a specific surgical corridor.
        
        Args:
            corridor_name: Name of surgical corridor
            include_neighbors: Include adjacent structures
            neighbor_depth: How many hops to include
        
        Returns:
            PyG Data object for corridor
        """
        if not TORCH_AVAILABLE or not PYG_AVAILABLE:
            return None
        
        # Get corridor structures
        corridor = await self.database.fetchrow("""
            SELECT structure_sequence, structures_at_risk
            FROM surgical_corridors
            WHERE LOWER(name) = LOWER($1)
        """, corridor_name)
        
        if not corridor:
            logger.warning(f"Corridor not found: {corridor_name}")
            return None
        
        # Collect all relevant entity names
        entity_names = set()
        entity_names.update(corridor["structure_sequence"] or [])
        entity_names.update(corridor["structures_at_risk"] or [])
        
        # Optionally expand to neighbors
        if include_neighbors:
            for _ in range(neighbor_depth):
                neighbors = await self._get_neighbors(list(entity_names))
                entity_names.update(neighbors)
        
        # Load entities
        placeholders = ', '.join(f'${i+1}' for i in range(len(entity_names)))
        entity_rows = await self.database.fetch(f"""
            SELECT * FROM anatomical_entities
            WHERE LOWER(name) = ANY(
                SELECT LOWER(unnest($1::text[]))
            )
        """, list(entity_names))
        
        if not entity_rows:
            return None
        
        # Build graph same as full graph
        nodes = []
        name_to_idx = {}
        
        for idx, row in enumerate(entity_rows):
            name = row["name"]
            name_to_idx[name.lower()] = idx
            
            spatial = row["spatial_context"]
            if isinstance(spatial, str):
                spatial = json.loads(spatial)
            
            node = EntityNode(
                name=name,
                entity_id=self.vocabulary.get_id(name),
                mobility=self.property_encoder.encode_mobility(row["mobility"]),
                consistency=self.property_encoder.encode_consistency(row["consistency"]),
                eloquence=self.property_encoder.encode_eloquence(row["eloquence_grade"]),
                diameter=float(row["vessel_diameter_mm"] or 0.0),
                confidence=float(row["confidence"] or 0.5),
                position=self.property_encoder.encode_position(spatial),
                flags=self.property_encoder.encode_flags(
                    is_end_artery=bool(row["is_end_artery"]),
                    has_collaterals=row["collateral_capacity"] not in ("none", None),
                    is_tethered="tethered" in str(row["mobility"] or "").lower(),
                    is_eloquent=row["eloquence_grade"] == "eloquent",
                ),
            )
            nodes.append(node)
        
        # Load edges between these entities
        edge_rows = await self.database.fetch("""
            SELECT source_entity, target_entity, relation_type, probability
            FROM causal_edges
            WHERE LOWER(source_entity) = ANY(SELECT LOWER(unnest($1::text[])))
              AND LOWER(target_entity) = ANY(SELECT LOWER(unnest($1::text[])))
        """, list(entity_names))
        
        edges = []
        for row in edge_rows:
            source_name = row["source_entity"].lower()
            target_name = row["target_entity"].lower()
            
            if source_name in name_to_idx and target_name in name_to_idx:
                edges.append(RelationEdge(
                    source_idx=name_to_idx[source_name],
                    target_idx=name_to_idx[target_name],
                    relation_type=self.property_encoder.encode_relation(row["relation_type"]),
                    probability=float(row["probability"] or 0.5),
                    distance=0.0,
                ))
        
        # Add corridor sequence edges
        sequence = corridor["structure_sequence"] or []
        for i in range(len(sequence) - 1):
            src = sequence[i].lower()
            tgt = sequence[i + 1].lower()
            if src in name_to_idx and tgt in name_to_idx:
                edges.append(RelationEdge(
                    source_idx=name_to_idx[src],
                    target_idx=name_to_idx[tgt],
                    relation_type=self.property_encoder.encode_relation("traverses"),
                    probability=1.0,
                    distance=0.0,
                ))
        
        return self._build_pyg_data(nodes, edges)
    
    async def _get_neighbors(self, entity_names: List[str]) -> Set[str]:
        """Get neighboring entities connected by edges."""
        rows = await self.database.fetch("""
            SELECT DISTINCT 
                CASE WHEN LOWER(source_entity) = ANY(SELECT LOWER(unnest($1::text[])))
                     THEN target_entity
                     ELSE source_entity
                END as neighbor
            FROM causal_edges
            WHERE LOWER(source_entity) = ANY(SELECT LOWER(unnest($1::text[])))
               OR LOWER(target_entity) = ANY(SELECT LOWER(unnest($1::text[])))
        """, entity_names)
        
        return {row["neighbor"] for row in rows}
    
    def _build_pyg_data(
        self,
        nodes: List[EntityNode],
        edges: List[RelationEdge],
    ) -> Data:
        """Convert intermediate representations to PyG Data."""
        n = len(nodes)
        
        # Node features
        entity_ids = torch.tensor([node.entity_id for node in nodes], dtype=torch.long)
        mobility = torch.tensor([node.mobility for node in nodes], dtype=torch.long)
        consistency = torch.tensor([node.consistency for node in nodes], dtype=torch.long)
        eloquence = torch.tensor([node.eloquence for node in nodes], dtype=torch.long)
        diameter = torch.tensor([[node.diameter] for node in nodes], dtype=torch.float)
        confidence = torch.tensor([[node.confidence] for node in nodes], dtype=torch.float)
        position = torch.tensor([node.position for node in nodes], dtype=torch.float)
        flags = torch.tensor([node.flags for node in nodes], dtype=torch.float)
        
        # Edge index
        if edges:
            edge_index = torch.tensor(
                [[e.source_idx for e in edges], [e.target_idx for e in edges]],
                dtype=torch.long
            )
            relation_type = torch.tensor([e.relation_type for e in edges], dtype=torch.long)
            probability = torch.tensor([[e.probability] for e in edges], dtype=torch.float)
            distance = torch.tensor([[e.distance] for e in edges], dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            relation_type = torch.zeros(0, dtype=torch.long)
            probability = torch.zeros((0, 1), dtype=torch.float)
            distance = torch.zeros((0, 1), dtype=torch.float)
        
        # Build Data object
        data = Data(
            x=torch.cat([
                entity_ids.unsqueeze(-1).float(),
                mobility.unsqueeze(-1).float(),
                consistency.unsqueeze(-1).float(),
                eloquence.unsqueeze(-1).float(),
                diameter,
                confidence,
                position,
                flags,
            ], dim=-1),
            edge_index=edge_index,
            edge_attr=torch.cat([
                relation_type.unsqueeze(-1).float(),
                probability,
                distance,
            ], dim=-1) if len(edges) > 0 else None,
            num_nodes=n,
        )
        
        # Store component tensors for EntityEncoder
        data.entity_ids = entity_ids
        data.mobility = mobility
        data.consistency = consistency
        data.eloquence = eloquence
        data.diameter = diameter
        data.confidence = confidence
        data.position = position
        data.flags = flags
        
        if len(edges) > 0:
            data.relation_type = relation_type
            data.probability = probability
            data.distance = distance
        
        # Store node names for debugging
        data.node_names = [node.name for node in nodes]
        
        return data
    
    def prepare_entity_features(self, data: Data) -> Dict[str, torch.Tensor]:
        """Prepare entity features dict for NeuroGAT."""
        return {
            "entity_ids": data.entity_ids,
            "mobility": data.mobility,
            "consistency": data.consistency,
            "eloquence": data.eloquence,
            "diameter": data.diameter,
            "confidence": data.confidence,
            "position": data.position,
            "flags": data.flags,
        }
    
    def prepare_edge_features(self, data: Data) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare edge features dict for NeuroGAT."""
        if data.edge_attr is None:
            return None
        
        return {
            "relation_type": data.relation_type,
            "probability": data.probability,
            "distance": data.distance,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Vocabularies
    "MOBILITY_VOCAB",
    "CONSISTENCY_VOCAB",
    "ELOQUENCE_VOCAB",
    "RETRACTION_VOCAB",
    "COLLATERAL_VOCAB",
    "RELATION_VOCAB",
    "ACTION_VOCAB",
    
    # Classes
    "EntityVocabulary",
    "PropertyEncoder",
    "EntityNode",
    "RelationEdge",
    "GraphEncoder",
]
