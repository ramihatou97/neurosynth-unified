"""
NeuroSynth 2.0 - NeuroGAT (Graph Attention Network)
====================================================

Graph Neural Network for learning implicit relationships between
anatomical entities that aren't captured by explicit rules.

The GNN learns:
- Hidden dependencies between structures
- Risk propagation patterns
- Surgical outcome correlations

Architecture:
    Entity Embeddings → GAT Layers → Risk Prediction Head
                                  → Link Prediction Head
                                  → Outcome Prediction Head

Key Innovation:
    While the ClinicalReasoner uses explicit IF-THEN rules from clinical
    principles, NeuroGAT learns implicit patterns from surgical data that
    experts haven't codified into rules.

Dependencies:
    - torch >= 2.0
    - torch_geometric (PyG)
    - numpy

Usage:
    model = NeuroGAT(
        entity_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=3
    )
    
    # Forward pass
    node_embeddings, edge_predictions = model(
        x=entity_features,
        edge_index=adjacency,
        edge_attr=edge_features
    )
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    logger.warning("PyTorch not available")

# Dummy decorator for when torch is unavailable
def _no_grad_dummy(func):
    return func

if not TORCH_AVAILABLE:
    class _DummyTorch:
        @staticmethod
        def no_grad():
            return _no_grad_dummy
    torch = _DummyTorch()

try:
    import torch_geometric
    from torch_geometric.nn import (
        GATv2Conv,
        GCNConv,
        SAGEConv,
        TransformerConv,
        global_mean_pool,
        global_max_pool,
        global_add_pool,
    )
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import (
        add_self_loops,
        degree,
        to_dense_adj,
        negative_sampling,
    )
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("PyTorch Geometric not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NeuroGATConfig:
    """Configuration for NeuroGAT model."""
    # Entity encoding
    entity_vocab_size: int = 1000  # Max unique entities
    entity_dim: int = 128  # Entity embedding dimension
    
    # Physics properties encoding
    num_mobility_types: int = 7
    num_consistency_types: int = 11
    num_eloquence_grades: int = 3
    physics_dim: int = 32
    
    # Graph attention
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Output heads
    num_risk_levels: int = 5
    num_complication_types: int = 8
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


# =============================================================================
# ENTITY ENCODER
# =============================================================================

class EntityEncoder(nn.Module):
    """
    Encodes anatomical entities into dense vectors.
    
    Combines:
    - Learned entity embeddings (for frequent entities)
    - Physics property embeddings (mobility, consistency, etc.)
    - Spatial encoding (position in brain space)
    """
    
    def __init__(self, config: NeuroGATConfig):
        super().__init__()
        self.config = config
        
        # Entity name embedding (for known entities)
        self.entity_embedding = nn.Embedding(
            config.entity_vocab_size,
            config.entity_dim,
            padding_idx=0
        )
        
        # Physics property embeddings
        self.mobility_embedding = nn.Embedding(
            config.num_mobility_types,
            config.physics_dim
        )
        self.consistency_embedding = nn.Embedding(
            config.num_consistency_types,
            config.physics_dim
        )
        self.eloquence_embedding = nn.Embedding(
            config.num_eloquence_grades,
            config.physics_dim
        )
        
        # Continuous property projections
        self.diameter_proj = nn.Linear(1, config.physics_dim)
        self.confidence_proj = nn.Linear(1, config.physics_dim)
        
        # Spatial encoding (3D position)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, config.physics_dim),
            nn.ReLU(),
            nn.Linear(config.physics_dim, config.physics_dim),
        )
        
        # Binary flags projection
        self.flags_proj = nn.Linear(4, config.physics_dim)  # is_end_artery, has_collaterals, etc.
        
        # Combine all features
        total_dim = config.entity_dim + config.physics_dim * 6
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
    
    def forward(
        self,
        entity_ids: torch.Tensor,  # (N,)
        mobility: torch.Tensor,     # (N,)
        consistency: torch.Tensor,  # (N,)
        eloquence: torch.Tensor,    # (N,)
        diameter: torch.Tensor,     # (N, 1)
        confidence: torch.Tensor,   # (N, 1)
        position: torch.Tensor,     # (N, 3)
        flags: torch.Tensor,        # (N, 4)
    ) -> torch.Tensor:
        """
        Encode entities into hidden representations.
        
        Returns: (N, hidden_dim) tensor
        """
        # Embeddings
        entity_emb = self.entity_embedding(entity_ids)
        mobility_emb = self.mobility_embedding(mobility)
        consistency_emb = self.consistency_embedding(consistency)
        eloquence_emb = self.eloquence_embedding(eloquence)
        
        # Continuous projections
        diameter_emb = self.diameter_proj(diameter)
        confidence_emb = self.confidence_proj(confidence)
        
        # Spatial encoding
        spatial_emb = self.spatial_encoder(position)
        
        # Flags
        flags_emb = self.flags_proj(flags)
        
        # Concatenate all
        combined = torch.cat([
            entity_emb,
            mobility_emb,
            consistency_emb,
            eloquence_emb,
            diameter_emb,
            confidence_emb,
            spatial_emb,
            flags_emb,
        ], dim=-1)
        
        # Project to hidden dim
        return self.output_proj(combined)


# =============================================================================
# EDGE ENCODER
# =============================================================================

class EdgeEncoder(nn.Module):
    """
    Encodes relationship edges between entities.
    
    Edge features include:
    - Relation type (supplies, drains, adjacent, etc.)
    - Causal properties (probability, mechanism)
    - Spatial relationship (distance, direction)
    """
    
    def __init__(self, config: NeuroGATConfig, num_relation_types: int = 20):
        super().__init__()
        self.config = config
        
        # Relation type embedding
        self.relation_embedding = nn.Embedding(
            num_relation_types,
            config.physics_dim
        )
        
        # Continuous features
        self.probability_proj = nn.Linear(1, config.physics_dim)
        self.distance_proj = nn.Linear(1, config.physics_dim)
        
        # Combine
        self.output_proj = nn.Sequential(
            nn.Linear(config.physics_dim * 3, config.hidden_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        relation_type: torch.Tensor,  # (E,)
        probability: torch.Tensor,     # (E, 1)
        distance: torch.Tensor,        # (E, 1)
    ) -> torch.Tensor:
        """
        Encode edges into hidden representations.
        
        Returns: (E, hidden_dim) tensor
        """
        relation_emb = self.relation_embedding(relation_type)
        prob_emb = self.probability_proj(probability)
        dist_emb = self.distance_proj(distance)
        
        combined = torch.cat([relation_emb, prob_emb, dist_emb], dim=-1)
        return self.output_proj(combined)


# =============================================================================
# GRAPH ATTENTION LAYERS
# =============================================================================

class NeuroGATLayer(nn.Module):
    """
    Single layer of the NeuroGAT architecture.
    
    Uses GATv2 (improved attention mechanism) with edge features.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()
        
        self.use_edge_features = use_edge_features
        head_dim = out_dim // num_heads
        
        # GATv2 convolution
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=out_dim if use_edge_features else None,
            add_self_loops=True,
            concat=True,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(out_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(out_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GAT layer.
        
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_dim)
        
        Returns:
            Updated node features (N, out_dim)
        """
        # Attention + residual
        if self.use_edge_features and edge_attr is not None:
            h = self.gat(x, edge_index, edge_attr=edge_attr)
        else:
            h = self.gat(x, edge_index)
        
        # Residual connection (if dimensions match)
        if h.shape == x.shape:
            h = h + x
        h = self.norm(h)
        
        # FFN + residual
        h = h + self.ffn(h)
        h = self.ffn_norm(h)
        
        return h


# =============================================================================
# PREDICTION HEADS
# =============================================================================

class RiskPredictionHead(nn.Module):
    """
    Predicts risk level for a (structure, action) pair.
    
    Input: Concatenation of [structure_embedding, action_embedding, context_embedding]
    Output: Risk level distribution (5 classes)
    """
    
    def __init__(self, config: NeuroGATConfig):
        super().__init__()
        
        self.action_embedding = nn.Embedding(14, config.hidden_dim)  # 14 surgical actions
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_risk_levels),
        )
    
    def forward(
        self,
        structure_embedding: torch.Tensor,  # (B, hidden_dim)
        action_id: torch.Tensor,            # (B,)
        context_embedding: torch.Tensor,    # (B, hidden_dim)
    ) -> torch.Tensor:
        """Predict risk level logits."""
        action_emb = self.action_embedding(action_id)
        combined = torch.cat([structure_embedding, action_emb, context_embedding], dim=-1)
        return self.classifier(combined)


class LinkPredictionHead(nn.Module):
    """
    Predicts likelihood of implicit relationships between entities.
    
    Used to discover hidden dependencies not in the explicit knowledge graph.
    """
    
    def __init__(self, config: NeuroGATConfig, num_relation_types: int = 20):
        super().__init__()
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_relation_types),
        )
        
        self.existence_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        source_embedding: torch.Tensor,  # (B, hidden_dim)
        target_embedding: torch.Tensor,  # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edge existence and type.
        
        Returns:
            existence_logits: (B, 1) probability of edge existing
            relation_logits: (B, num_relations) relation type distribution
        """
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        
        existence = self.existence_classifier(combined)
        relation = self.relation_classifier(combined)
        
        return existence, relation


class OutcomePredictionHead(nn.Module):
    """
    Predicts surgical outcome from graph-level representation.
    
    Uses global pooling over all node embeddings.
    """
    
    def __init__(self, config: NeuroGATConfig):
        super().__init__()
        
        self.pool_proj = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        
        self.outcome_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 4),  # SAFE, CAUTION, HIGH_RISK, CONTRAINDICATED
        )
        
        self.complication_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_complication_types),
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # (N, hidden_dim)
        batch: torch.Tensor,            # (N,) node-to-graph assignment
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict overall surgical outcome.
        
        Returns:
            outcome_logits: (B, 4) verdict distribution
            complication_logits: (B, num_complications) complication probabilities
        """
        # Multi-pooling
        mean_pool = global_mean_pool(node_embeddings, batch)
        max_pool = global_max_pool(node_embeddings, batch)
        sum_pool = global_add_pool(node_embeddings, batch)
        
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        graph_embedding = self.pool_proj(pooled)
        
        outcome = self.outcome_classifier(graph_embedding)
        complications = self.complication_classifier(graph_embedding)
        
        return outcome, complications


# =============================================================================
# MAIN MODEL
# =============================================================================

class NeuroGAT(nn.Module):
    """
    Graph Attention Network for surgical risk reasoning.
    
    Learns implicit patterns from the anatomical graph that
    complement explicit clinical principles.
    """
    
    def __init__(self, config: Optional[NeuroGATConfig] = None):
        super().__init__()
        
        if not TORCH_AVAILABLE or not PYG_AVAILABLE:
            raise RuntimeError("PyTorch and PyTorch Geometric required")
        
        self.config = config or NeuroGATConfig()
        
        # Encoders
        self.entity_encoder = EntityEncoder(self.config)
        self.edge_encoder = EdgeEncoder(self.config)
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            NeuroGATLayer(
                in_dim=self.config.hidden_dim,
                out_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ])
        
        # Prediction heads
        self.risk_head = RiskPredictionHead(self.config)
        self.link_head = LinkPredictionHead(self.config)
        self.outcome_head = OutcomePredictionHead(self.config)
    
    def encode_entities(
        self,
        entity_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode raw entity features into embeddings."""
        return self.entity_encoder(
            entity_ids=entity_features["entity_ids"],
            mobility=entity_features["mobility"],
            consistency=entity_features["consistency"],
            eloquence=entity_features["eloquence"],
            diameter=entity_features["diameter"],
            confidence=entity_features["confidence"],
            position=entity_features["position"],
            flags=entity_features["flags"],
        )
    
    def encode_edges(
        self,
        edge_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode raw edge features into embeddings."""
        return self.edge_encoder(
            relation_type=edge_features["relation_type"],
            probability=edge_features["probability"],
            distance=edge_features["distance"],
        )
    
    def forward(
        self,
        entity_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_features: Optional[Dict[str, torch.Tensor]] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NeuroGAT.
        
        Args:
            entity_features: Dict of entity feature tensors
            edge_index: Graph connectivity (2, E)
            edge_features: Dict of edge feature tensors
            batch: Node-to-graph assignment for batched graphs
        
        Returns:
            Dict with:
            - node_embeddings: Final node representations
            - graph_embedding: Global graph representation
        """
        # Encode entities
        x = self.encode_entities(entity_features)
        
        # Encode edges
        edge_attr = None
        if edge_features is not None:
            edge_attr = self.encode_edges(edge_features)
        
        # Message passing
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
        
        result = {"node_embeddings": x}
        
        # Graph-level embedding (if batch provided)
        if batch is not None:
            graph_emb = global_mean_pool(x, batch)
            result["graph_embedding"] = graph_emb
        
        return result
    
    def predict_risk(
        self,
        node_embeddings: torch.Tensor,
        structure_idx: torch.Tensor,
        action_id: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict risk level for a specific action on a structure."""
        structure_emb = node_embeddings[structure_idx]
        return self.risk_head(structure_emb, action_id, context_embedding)
    
    def predict_link(
        self,
        node_embeddings: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict implicit relationship between two entities."""
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        return self.link_head(source_emb, target_emb)
    
    def predict_outcome(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict overall surgical outcome from graph."""
        return self.outcome_head(node_embeddings, batch)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class NeuroGATTrainer:
    """Training utilities for NeuroGAT."""
    
    def __init__(
        self,
        model: NeuroGAT,
        config: NeuroGATConfig,
        device: str = "auto",
    ):
        self.model = model
        self.config = config
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Loss functions
        self.risk_loss = nn.CrossEntropyLoss()
        self.link_loss = nn.BCEWithLogitsLoss()
        self.outcome_loss = nn.CrossEntropyLoss()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            entity_features=batch["entity_features"],
            edge_index=batch["edge_index"],
            edge_features=batch.get("edge_features"),
            batch=batch.get("batch"),
        )
        
        losses = {}
        total_loss = 0.0
        
        # Risk prediction loss
        if "risk_labels" in batch:
            risk_logits = self.model.predict_risk(
                outputs["node_embeddings"],
                batch["risk_structure_idx"],
                batch["risk_action_id"],
                outputs.get("graph_embedding", outputs["node_embeddings"].mean(0, keepdim=True)),
            )
            risk_loss = self.risk_loss(risk_logits, batch["risk_labels"])
            losses["risk"] = risk_loss.item()
            total_loss = total_loss + risk_loss
        
        # Link prediction loss (contrastive)
        if "link_labels" in batch:
            existence, _ = self.model.predict_link(
                outputs["node_embeddings"],
                batch["link_source_idx"],
                batch["link_target_idx"],
            )
            link_loss = self.link_loss(existence.squeeze(), batch["link_labels"].float())
            losses["link"] = link_loss.item()
            total_loss = total_loss + link_loss
        
        # Outcome prediction loss
        if "outcome_labels" in batch and "batch" in batch:
            outcome_logits, _ = self.model.predict_outcome(
                outputs["node_embeddings"],
                batch["batch"],
            )
            outcome_loss = self.outcome_loss(outcome_logits, batch["outcome_labels"])
            losses["outcome"] = outcome_loss.item()
            total_loss = total_loss + outcome_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        losses["total"] = total_loss.item()
        return losses
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self.model.eval()
        
        total_risk_correct = 0
        total_risk_samples = 0
        total_outcome_correct = 0
        total_outcome_samples = 0
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(
                entity_features=batch["entity_features"],
                edge_index=batch["edge_index"],
                edge_features=batch.get("edge_features"),
                batch=batch.get("batch"),
            )
            
            # Risk accuracy
            if "risk_labels" in batch:
                risk_logits = self.model.predict_risk(
                    outputs["node_embeddings"],
                    batch["risk_structure_idx"],
                    batch["risk_action_id"],
                    outputs.get("graph_embedding", outputs["node_embeddings"].mean(0, keepdim=True)),
                )
                pred = risk_logits.argmax(dim=-1)
                total_risk_correct += (pred == batch["risk_labels"]).sum().item()
                total_risk_samples += len(pred)
            
            # Outcome accuracy
            if "outcome_labels" in batch and "batch" in batch:
                outcome_logits, _ = self.model.predict_outcome(
                    outputs["node_embeddings"],
                    batch["batch"],
                )
                pred = outcome_logits.argmax(dim=-1)
                total_outcome_correct += (pred == batch["outcome_labels"]).sum().item()
                total_outcome_samples += len(pred)
        
        metrics = {}
        if total_risk_samples > 0:
            metrics["risk_accuracy"] = total_risk_correct / total_risk_samples
        if total_outcome_samples > 0:
            metrics["outcome_accuracy"] = total_outcome_correct / total_outcome_samples
        
        return metrics


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NeuroGATConfig",
    "EntityEncoder",
    "EdgeEncoder",
    "NeuroGATLayer",
    "RiskPredictionHead",
    "LinkPredictionHead",
    "OutcomePredictionHead",
    "NeuroGAT",
    "NeuroGATTrainer",
]
