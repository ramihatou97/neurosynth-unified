"""
NeuroSynth 2.0 - GNN Module
============================

Graph Neural Networks for learning implicit anatomical relationships.

This module provides:
- NeuroGAT: Graph Attention Network for surgical risk reasoning
- GraphEncoder: Convert database entities to graph tensors
- Training utilities

The GNN complements explicit clinical rules by learning:
- Hidden dependencies between structures
- Risk propagation patterns
- Surgical outcome correlations

Dependencies (optional):
- torch >= 2.0
- torch_geometric (PyG)

Usage:
    from src.gnn import NeuroGAT, GraphEncoder
    
    # Build graph from database
    encoder = GraphEncoder(database)
    await encoder.initialize()
    graph = await encoder.build_corridor_graph("pterional")
    
    # Run GNN inference
    model = NeuroGAT()
    outputs = model(
        entity_features=encoder.prepare_entity_features(graph),
        edge_index=graph.edge_index,
        edge_features=encoder.prepare_edge_features(graph),
    )
"""

from src.neurosynth2.gnn.neuro_gat import (
    NeuroGATConfig,
    EntityEncoder,
    EdgeEncoder,
    NeuroGATLayer,
    RiskPredictionHead,
    LinkPredictionHead,
    OutcomePredictionHead,
    NeuroGAT,
    NeuroGATTrainer,
)

from src.neurosynth2.gnn.encoder import (
    EntityVocabulary,
    PropertyEncoder,
    EntityNode,
    RelationEdge,
    GraphEncoder,
    MOBILITY_VOCAB,
    CONSISTENCY_VOCAB,
    ELOQUENCE_VOCAB,
    RETRACTION_VOCAB,
    COLLATERAL_VOCAB,
    RELATION_VOCAB,
    ACTION_VOCAB,
)

__all__ = [
    # Main model
    "NeuroGAT",
    "NeuroGATConfig",
    "NeuroGATTrainer",
    
    # Model components
    "EntityEncoder",
    "EdgeEncoder",
    "NeuroGATLayer",
    "RiskPredictionHead",
    "LinkPredictionHead",
    "OutcomePredictionHead",
    
    # Graph encoding
    "GraphEncoder",
    "EntityVocabulary",
    "PropertyEncoder",
    "EntityNode",
    "RelationEdge",
    
    # Vocabularies
    "MOBILITY_VOCAB",
    "CONSISTENCY_VOCAB",
    "ELOQUENCE_VOCAB",
    "RETRACTION_VOCAB",
    "COLLATERAL_VOCAB",
    "RELATION_VOCAB",
    "ACTION_VOCAB",
]

__version__ = "2.0.0"
