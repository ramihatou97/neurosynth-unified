"""
NeuroSynth Unified - Pipeline Configuration
============================================

Extended pipeline configuration with database integration options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class OutputMode(Enum):
    """Pipeline output mode."""
    FILE = "file"           # Export to pkl/json files only
    DATABASE = "database"   # Write to PostgreSQL only
    BOTH = "both"           # Database + file backup


@dataclass
class DatabaseConfig:
    """Database configuration for pipeline."""
    connection_string: str = ""
    min_connections: int = 2
    max_connections: int = 10
    
    # Write options
    batch_size: int = 100           # Batch size for inserts
    create_indexes: bool = True     # Create vector indexes after insert
    update_existing: bool = True    # Update if document exists
    
    @property
    def enabled(self) -> bool:
        return bool(self.connection_string)


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    # Text embeddings
    text_provider: str = "voyage"
    text_model: str = "voyage-3"
    text_dimension: int = 1024
    text_api_key: str = ""
    
    # Image embeddings
    image_provider: str = "biomedclip"
    image_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    image_dimension: int = 512
    image_device: str = "cpu"  # cpu, cuda, mps
    
    # VLM captioning
    vlm_provider: str = "anthropic"
    vlm_model: str = "claude-sonnet-4-20250514"
    vlm_api_key: str = ""


@dataclass
class TriageConfig:
    """Visual triage configuration."""
    enabled: bool = True
    
    # Size thresholds
    min_dimension: int = 100        # Minimum width/height
    max_aspect_ratio: float = 10.0  # Max width/height ratio
    
    # Content analysis
    min_entropy: float = 3.5        # Minimum image entropy
    min_edge_density: float = 0.05  # Minimum edge density
    
    # Duplicate detection
    detect_duplicates: bool = True
    duplicate_threshold: float = 0.95


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    # Chunk sizes
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    target_chunk_size: int = 800
    
    # Overlap
    overlap_size: int = 100
    
    # Medical-aware settings
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    protect_abbreviations: bool = True
    
    # Classification
    classify_chunks: bool = True
    detect_specialty: bool = True


@dataclass  
class LinkingConfig:
    """Chunk-image linking configuration."""
    # Link types
    enable_proximity: bool = True
    enable_semantic: bool = True
    enable_cui_match: bool = True
    
    # Thresholds
    proximity_pages: int = 1        # Link if within N pages
    semantic_threshold: float = 0.5  # Min semantic similarity
    cui_overlap_min: int = 1        # Min shared CUIs
    
    # Scoring weights
    proximity_weight: float = 0.3
    semantic_weight: float = 0.5
    cui_weight: float = 0.2


@dataclass
class UnifiedPipelineConfig:
    """
    Complete pipeline configuration.
    
    Usage:
        config = UnifiedPipelineConfig(
            output_mode=OutputMode.DATABASE,
            database=DatabaseConfig(
                connection_string="postgresql://..."
            ),
            output_dir=Path("./backup")
        )
        
        pipeline = UnifiedPipeline(config)
        result = await pipeline.process_document(pdf_path)
    """
    
    # Output settings
    output_mode: OutputMode = OutputMode.FILE
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    # Component configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    
    # Processing options
    enable_ocr: bool = True
    enable_tables: bool = True
    enable_umls: bool = True
    enable_knowledge_graph: bool = False
    
    # Performance
    max_workers: int = 4
    enable_metrics: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_progress: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate database config if in database mode
        if self.output_mode in (OutputMode.DATABASE, OutputMode.BOTH):
            if not self.database.connection_string:
                raise ValueError(
                    "database.connection_string required for DATABASE or BOTH output mode"
                )
    
    @classmethod
    def from_env(cls) -> 'UnifiedPipelineConfig':
        """Create config from environment variables."""
        import os
        
        # Determine output mode
        mode_str = os.getenv("PIPELINE_OUTPUT_MODE", "file").lower()
        output_mode = OutputMode(mode_str)
        
        return cls(
            output_mode=output_mode,
            output_dir=Path(os.getenv("PIPELINE_OUTPUT_DIR", "./output")),
            database=DatabaseConfig(
                connection_string=os.getenv("DATABASE_URL", ""),
                min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "2")),
                max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10"))
            ),
            embedding=EmbeddingConfig(
                text_api_key=os.getenv("VOYAGE_API_KEY", ""),
                vlm_api_key=os.getenv("ANTHROPIC_API_KEY", "")
            ),
            triage=TriageConfig(
                enabled=os.getenv("TRIAGE_ENABLED", "true").lower() == "true"
            ),
            enable_ocr=os.getenv("ENABLE_OCR", "true").lower() == "true",
            enable_tables=os.getenv("ENABLE_TABLES", "true").lower() == "true",
            enable_umls=os.getenv("ENABLE_UMLS", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    @classmethod
    def for_database(
        cls,
        connection_string: str,
        **kwargs
    ) -> 'UnifiedPipelineConfig':
        """Create config for database output mode."""
        return cls(
            output_mode=OutputMode.DATABASE,
            database=DatabaseConfig(connection_string=connection_string),
            **kwargs
        )
    
    @classmethod
    def for_files(
        cls,
        output_dir: Path,
        **kwargs
    ) -> 'UnifiedPipelineConfig':
        """Create config for file output mode."""
        return cls(
            output_mode=OutputMode.FILE,
            output_dir=output_dir,
            **kwargs
        )
    
    @classmethod
    def for_both(
        cls,
        connection_string: str,
        output_dir: Path,
        **kwargs
    ) -> 'UnifiedPipelineConfig':
        """Create config for database + file backup mode."""
        return cls(
            output_mode=OutputMode.BOTH,
            database=DatabaseConfig(connection_string=connection_string),
            output_dir=output_dir,
            **kwargs
        )
    
    def to_phase1_config(self):
        """
        Convert to Phase 1 PipelineConfig for backward compatibility.
        
        Returns a config object compatible with the original Phase 1 pipeline.
        """
        # Import Phase 1 config if available
        try:
            from src.ingest.pipeline import PipelineConfig
            
            return PipelineConfig(
                output_dir=self.output_dir,
                enable_ocr=self.enable_ocr,
                enable_tables=self.enable_tables,
                # Map other settings as needed
            )
        except ImportError:
            # Return self if Phase 1 config not available
            return self
