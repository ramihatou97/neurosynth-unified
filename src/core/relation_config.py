"""
NeuroSynth Relation Extraction Configuration

Centralized configuration for all relation extraction enhancements.
All flags default to backward-compatible values.

Usage:
    from src.core.relation_config import RelationExtractionConfig
    
    # Default config (all safe features enabled)
    config = RelationExtractionConfig()
    
    # Full enhancement mode
    config = RelationExtractionConfig(
        enable_tiered_llm=True,
        enable_coreference=True,
    )
    
    # Minimal mode (for testing/debugging)
    config = RelationExtractionConfig(
        enable_coordination=False,
        enable_negation=False,
        enable_entity_first_ner=False,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ExtractionStrategy(str, Enum):
    """Primary extraction strategy to use."""
    DEPENDENCY_FIRST = "dependency_first"  # spaCy dependency parsing primary
    ENTITY_FIRST = "entity_first"          # NER-driven extraction primary
    HYBRID = "hybrid"                       # Both strategies, deduplicated


class LLMTier(str, Enum):
    """LLM usage tiers based on extraction confidence."""
    PASS_THROUGH = "pass_through"    # conf >= 0.9: No LLM needed
    VERIFY = "verify"                 # 0.5 <= conf < 0.9: LLM confirms/rejects
    COMPLETE = "complete"             # conf < 0.5 or complex: LLM extracts


@dataclass
class RelationExtractionConfig:
    """
    Configuration for enhanced relation extraction.
    
    All boolean flags default to backward-compatible values that
    enable safe improvements without requiring additional dependencies.
    
    Attributes:
        enable_coordination: Extract multiple targets from conjunctions
            "MCA supplies frontal and temporal lobes" → 2 relations
            
        enable_negation: Detect negated relations
            "No evidence X supplies Y" → is_negated=True
            
        enable_entity_first_ner: Use entity-first extraction strategy
            Find UMLS entities first, then classify relations between pairs
            
        enable_tiered_llm: Use tiered LLM verification
            Routes extractions through LLM based on confidence thresholds
            Requires llm_client to be provided
            
        enable_coreference: Resolve pronouns before extraction
            "The MCA... It supplies..." → resolves "It" to "MCA"
            Memory-intensive (~200MB model), disabled by default
    """
    
    # =========================================================================
    # Feature Flags (backward-compatible defaults)
    # =========================================================================
    
    # Core enhancements (safe, no extra dependencies beyond spaCy)
    enable_coordination: bool = True
    enable_negation: bool = True
    enable_entity_first_ner: bool = True
    
    # Advanced enhancements (require additional setup)
    enable_tiered_llm: bool = False      # Requires LLM client
    enable_coreference: bool = False     # Requires fastcoref (~200MB)
    
    # =========================================================================
    # Confidence Thresholds
    # =========================================================================
    
    # Minimum confidence for any extraction to be kept
    min_confidence: float = 0.5
    
    # LLM tier thresholds (only used if enable_tiered_llm=True)
    llm_verify_threshold: float = 0.9    # >= this: pass through without LLM
    llm_complete_threshold: float = 0.5  # < this: LLM does full extraction
    # Between complete and verify: LLM verifies spaCy extraction
    
    # =========================================================================
    # UMLS Integration
    # =========================================================================
    
    enable_umls_normalization: bool = True
    umls_min_score: float = 0.80
    
    # Semantic types to prioritize (TUIs)
    priority_semantic_types: List[str] = field(default_factory=lambda: [
        "T023",  # Body Part, Organ, or Organ Component
        "T029",  # Body Location or Region
        "T030",  # Body Space or Junction
        "T061",  # Therapeutic or Preventive Procedure
        "T047",  # Disease or Syndrome
        "T121",  # Pharmacologic Substance
    ])
    
    # =========================================================================
    # Extraction Strategy
    # =========================================================================
    
    strategy: ExtractionStrategy = ExtractionStrategy.HYBRID
    
    # Entity-first settings
    entity_pair_max_distance: int = 200  # Max chars between entity pairs
    
    # =========================================================================
    # Coreference Settings
    # =========================================================================
    
    # Pronouns to resolve
    coref_pronouns: List[str] = field(default_factory=lambda: [
        "it", "its", "they", "their", "this", "these", "which", "that"
    ])
    
    # Maximum text length for coreference (longer texts split)
    coref_max_length: int = 5000
    
    # =========================================================================
    # Negation Settings
    # =========================================================================
    
    # Additional negation cues beyond negspacy defaults
    additional_negation_cues: List[str] = field(default_factory=lambda: [
        "no evidence",
        "not demonstrated", 
        "absence of",
        "negative for",
        "unlikely",
        "failed to demonstrate",
        "without",
        "rules out",
        "excluded",
    ])
    
    # =========================================================================
    # Performance Settings
    # =========================================================================
    
    # Batch sizes for different operations
    batch_size: int = 50
    llm_batch_size: int = 10  # Smaller batches for LLM calls
    
    # Timeouts
    llm_timeout_seconds: float = 30.0
    
    # =========================================================================
    # Debug/Audit Settings
    # =========================================================================
    
    # Track extraction method for each relation
    track_extraction_method: bool = True
    
    # Log detailed extraction info
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure thresholds are ordered correctly
        if self.llm_complete_threshold >= self.llm_verify_threshold:
            raise ValueError(
                f"llm_complete_threshold ({self.llm_complete_threshold}) must be "
                f"< llm_verify_threshold ({self.llm_verify_threshold})"
            )
        
        # Warn if LLM enabled but thresholds seem off
        if self.enable_tiered_llm:
            if self.llm_verify_threshold < 0.7:
                import warnings
                warnings.warn(
                    f"llm_verify_threshold={self.llm_verify_threshold} is low; "
                    "most extractions will bypass LLM verification"
                )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/serialization."""
        return {
            "enable_coordination": self.enable_coordination,
            "enable_negation": self.enable_negation,
            "enable_entity_first_ner": self.enable_entity_first_ner,
            "enable_tiered_llm": self.enable_tiered_llm,
            "enable_coreference": self.enable_coreference,
            "min_confidence": self.min_confidence,
            "llm_verify_threshold": self.llm_verify_threshold,
            "llm_complete_threshold": self.llm_complete_threshold,
            "enable_umls_normalization": self.enable_umls_normalization,
            "umls_min_score": self.umls_min_score,
            "strategy": self.strategy.value,
            "track_extraction_method": self.track_extraction_method,
        }
    
    @classmethod
    def minimal(cls) -> "RelationExtractionConfig":
        """Create minimal config for testing/debugging."""
        return cls(
            enable_coordination=False,
            enable_negation=False,
            enable_entity_first_ner=False,
            enable_tiered_llm=False,
            enable_coreference=False,
        )
    
    @classmethod
    def full(cls, llm_client=None) -> "RelationExtractionConfig":
        """Create full-featured config."""
        return cls(
            enable_coordination=True,
            enable_negation=True,
            enable_entity_first_ner=True,
            enable_tiered_llm=llm_client is not None,
            enable_coreference=True,
        )


# =============================================================================
# Convenience Exports
# =============================================================================

DEFAULT_CONFIG = RelationExtractionConfig()
MINIMAL_CONFIG = RelationExtractionConfig.minimal()
