"""
NeuroSynth Unified - Ingestion Layer
=====================================

Document processing pipeline with database integration.

Components:
- config.py: Pipeline configuration
- database_writer.py: PostgreSQL writer adapter
- unified_pipeline.py: Main pipeline wrapper
- pipeline.py: Phase 1 extraction (from original)

Quick Start:
    # Database mode
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    config = UnifiedPipelineConfig.for_database(
        connection_string="postgresql://user:pass@localhost/neurosynth"
    )
    
    async with UnifiedPipeline(config) as pipeline:
        result = await pipeline.process_document("/path/to/file.pdf")
        print(f"Document ID: {result.document_id}")

    # File export mode (backward compatible)
    config = UnifiedPipelineConfig.for_files(output_dir=Path("./output"))
    
    async with UnifiedPipeline(config) as pipeline:
        result = await pipeline.process_document("/path/to/file.pdf")
        print(f"Export: {result.export_path}")
"""

# Configuration
from src.ingest.config import (
    UnifiedPipelineConfig,
    OutputMode,
    DatabaseConfig,
    EmbeddingConfig,
    TriageConfig,
    ChunkingConfig,
    LinkingConfig
)

# Database writer
from src.ingest.database_writer import (
    PipelineDatabaseWriter,
    write_phase1_result_to_database
)

# Unified pipeline
from src.ingest.unified_pipeline import (
    UnifiedPipeline,
    UnifiedPipelineResult,
    process_document_to_database,
    process_document_to_files
)

# Re-export Phase 1 components if available
try:
    from src.ingest.pipeline import (
        NeuroIngestPipeline,
        PipelineConfig,
        PipelineResult
    )
    _HAS_PHASE1 = True
except ImportError:
    _HAS_PHASE1 = False

__all__ = [
    # Config
    'UnifiedPipelineConfig',
    'OutputMode',
    'DatabaseConfig',
    'EmbeddingConfig',
    'TriageConfig',
    'ChunkingConfig',
    'LinkingConfig',
    
    # Database writer
    'PipelineDatabaseWriter',
    'write_phase1_result_to_database',
    
    # Pipeline
    'UnifiedPipeline',
    'UnifiedPipelineResult',
    'process_document_to_database',
    'process_document_to_files',
]

# Add Phase 1 exports if available
if _HAS_PHASE1:
    __all__.extend([
        'NeuroIngestPipeline',
        'PipelineConfig',
        'PipelineResult'
    ])
