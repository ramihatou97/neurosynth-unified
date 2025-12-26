"""
NeuroSynth Unified - Pipeline Wrapper
======================================

Unified pipeline that wraps Phase 1 extraction with database integration.
Supports file export, database write, or both.

Usage:
    from src.ingest.unified_pipeline import UnifiedPipeline
    from src.ingest.config import UnifiedPipelineConfig, OutputMode
    
    # Database mode
    config = UnifiedPipelineConfig.for_database(
        connection_string="postgresql://..."
    )
    
    pipeline = UnifiedPipeline(config)
    await pipeline.initialize()
    
    result = await pipeline.process_document("/path/to/file.pdf")
    
    await pipeline.close()
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
from uuid import UUID
import asyncio
from datetime import datetime

from src.ingest.config import UnifiedPipelineConfig, OutputMode
from src.ingest.database_writer import PipelineDatabaseWriter

logger = logging.getLogger(__name__)


class UnifiedPipelineResult:
    """
    Result from unified pipeline processing.
    
    Extends Phase 1 result with database information.
    """
    
    def __init__(
        self,
        phase1_result=None,
        document_id: UUID = None,
        source_path: str = None,
        export_path: Path = None
    ):
        self._phase1_result = phase1_result
        self.document_id = document_id
        self.source_path = source_path
        self.export_path = export_path
        
        # Copy Phase 1 result attributes
        if phase1_result:
            self.chunks = getattr(phase1_result, 'chunks', [])
            self.images = getattr(phase1_result, 'images', [])
            self.links = getattr(phase1_result, 'links', [])
            self.error = getattr(phase1_result, 'error', None)
            self.processing_time_seconds = getattr(phase1_result, 'processing_time_seconds', 0)
            self.total_pages = getattr(phase1_result, 'total_pages', 0)
        else:
            self.chunks = []
            self.images = []
            self.links = []
            self.error = None
            self.processing_time_seconds = 0
            self.total_pages = 0
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
    
    @property
    def image_count(self) -> int:
        return len(self.images)
    
    @property
    def link_count(self) -> int:
        return len(self.links)
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': str(self.document_id) if self.document_id else None,
            'source_path': self.source_path,
            'export_path': str(self.export_path) if self.export_path else None,
            'chunk_count': self.chunk_count,
            'image_count': self.image_count,
            'link_count': self.link_count,
            'total_pages': self.total_pages,
            'processing_time_seconds': self.processing_time_seconds,
            'success': self.success,
            'error': str(self.error) if self.error else None
        }


class UnifiedPipeline:
    """
    Unified document processing pipeline.
    
    Wraps Phase 1 extraction pipeline and adds:
    - PostgreSQL database integration
    - Configurable output modes (file, database, both)
    - Progress callbacks
    - Error recovery
    
    Usage:
        config = UnifiedPipelineConfig.for_database(connection_string)
        pipeline = UnifiedPipeline(config)
        await pipeline.initialize()
        
        result = await pipeline.process_document("/path/to/pdf")
        print(f"Document ID: {result.document_id}")
        print(f"Chunks: {result.chunk_count}")
        
        await pipeline.close()
    """
    
    def __init__(
        self,
        config: UnifiedPipelineConfig,
        on_progress: Callable = None
    ):
        """
        Initialize unified pipeline.
        
        Args:
            config: Pipeline configuration
            on_progress: Progress callback(stage, current, total, message)
        """
        self.config = config
        self.on_progress = on_progress
        
        # Components (initialized in initialize())
        self._phase1_pipeline = None
        self._db_writer: Optional[PipelineDatabaseWriter] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        if self._initialized:
            return
        
        logger.info(f"Initializing UnifiedPipeline (mode: {self.config.output_mode.value})")
        
        # Initialize Phase 1 pipeline
        try:
            from src.ingest.pipeline import NeuroIngestPipeline, PipelineConfig
            
            # Convert config
            phase1_config = PipelineConfig(
                output_dir=self.config.output_dir
            )
            
            self._phase1_pipeline = NeuroIngestPipeline(
                config=phase1_config,
                enable_triage=self.config.triage.enabled,
                enable_metrics=self.config.enable_metrics
            )
            
            logger.info("Phase 1 pipeline initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import Phase 1 pipeline: {e}")
            raise RuntimeError(
                "Phase 1 pipeline not available. "
                "Ensure src/ingest/pipeline.py is present."
            )
        
        # Initialize database writer if needed
        if self.config.output_mode in (OutputMode.DATABASE, OutputMode.BOTH):
            self._db_writer = PipelineDatabaseWriter(
                connection_string=self.config.database.connection_string,
                export_files=(self.config.output_mode == OutputMode.BOTH),
                export_dir=self.config.output_dir if self.config.output_mode == OutputMode.BOTH else None
            )
            
            await self._db_writer.connect()
            logger.info("Database writer initialized")
        
        self._initialized = True
        logger.info("UnifiedPipeline initialization complete")
    
    async def close(self) -> None:
        """Close pipeline resources."""
        if self._db_writer:
            await self._db_writer.close()
        
        self._initialized = False
        logger.info("UnifiedPipeline closed")
    
    async def process_document(
        self,
        pdf_path: str | Path,
        title: str = None,
        metadata: Dict[str, Any] = None
    ) -> UnifiedPipelineResult:
        """
        Process a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            title: Optional document title
            metadata: Optional additional metadata
        
        Returns:
            UnifiedPipelineResult with processing results
        """
        if not self._initialized:
            await self.initialize()
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        start_time = datetime.now()
        logger.info(f"Processing document: {pdf_path}")
        
        # Progress helper
        def report_progress(stage: str, current: int, total: int, message: str = ""):
            if self.on_progress:
                self.on_progress(stage, current, total, message)
            if self.config.log_progress:
                logger.info(f"[{stage}] {current}/{total} {message}")
        
        try:
            # Step 1: Phase 1 extraction
            report_progress("extraction", 0, 1, "Starting Phase 1 extraction...")
            
            phase1_result = await self._phase1_pipeline.process_document(str(pdf_path))
            
            report_progress("extraction", 1, 1, "Extraction complete")
            
            # Check for extraction errors
            if phase1_result.error:
                logger.error(f"Phase 1 extraction failed: {phase1_result.error}")
                return UnifiedPipelineResult(
                    phase1_result=phase1_result,
                    source_path=str(pdf_path)
                )
            
            # Log extraction results
            logger.info(
                f"Extracted: {len(phase1_result.chunks)} chunks, "
                f"{len(phase1_result.images)} images, "
                f"{len(phase1_result.links)} links"
            )
            
            # Step 2: Output based on mode
            document_id = None
            export_path = None
            
            if self.config.output_mode == OutputMode.FILE:
                # File export only
                report_progress("export", 0, 1, "Exporting to files...")
                export_path = await self._export_to_files(phase1_result, pdf_path)
                report_progress("export", 1, 1, f"Exported to {export_path}")
                
            elif self.config.output_mode == OutputMode.DATABASE:
                # Database only
                report_progress("database", 0, 1, "Writing to database...")
                document_id = await self._write_to_database(
                    phase1_result, 
                    str(pdf_path), 
                    title,
                    report_progress
                )
                report_progress("database", 1, 1, f"Document ID: {document_id}")
                
            elif self.config.output_mode == OutputMode.BOTH:
                # Database + file backup
                report_progress("database", 0, 1, "Writing to database...")
                document_id = await self._write_to_database(
                    phase1_result,
                    str(pdf_path),
                    title,
                    report_progress
                )
                report_progress("database", 1, 1, f"Document ID: {document_id}")
                
                # File export is handled by database writer when export_files=True
                export_path = self.config.output_dir / str(document_id)
            
            # Build result
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = UnifiedPipelineResult(
                phase1_result=phase1_result,
                document_id=document_id,
                source_path=str(pdf_path),
                export_path=export_path
            )
            
            logger.info(
                f"Document processed successfully in {elapsed:.1f}s: "
                f"{result.chunk_count} chunks, {result.image_count} images"
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline error processing {pdf_path}")
            
            # Return error result
            result = UnifiedPipelineResult(source_path=str(pdf_path))
            result.error = str(e)
            return result
    
    async def _export_to_files(
        self,
        phase1_result,
        pdf_path: Path
    ) -> Path:
        """Export Phase 1 result to files using existing export method."""
        if hasattr(self._phase1_pipeline, 'export_for_phase2'):
            export_dir = self._phase1_pipeline.export_for_phase2(phase1_result)
            return Path(export_dir)
        else:
            # Fallback: manual export
            from uuid import uuid4
            import pickle
            import json
            
            doc_id = uuid4()
            export_path = self.config.output_dir / str(doc_id)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Export chunks
            chunks = getattr(phase1_result, 'chunks', [])
            with open(export_path / "chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)
            
            # Export images
            images = getattr(phase1_result, 'images', [])
            with open(export_path / "images.pkl", "wb") as f:
                pickle.dump(images, f)
            
            # Export links
            links = getattr(phase1_result, 'links', [])
            with open(export_path / "links.json", "w") as f:
                json.dump(links, f, default=str)
            
            # Export manifest
            manifest = {
                'document_id': str(doc_id),
                'source_path': str(pdf_path),
                'chunk_count': len(chunks),
                'image_count': len(images),
                'link_count': len(links)
            }
            with open(export_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            return export_path
    
    async def _write_to_database(
        self,
        phase1_result,
        source_path: str,
        title: str,
        report_progress: Callable
    ) -> UUID:
        """Write Phase 1 result to database."""
        
        def db_progress(stage: str, current: int, total: int):
            report_progress(f"db_{stage}", current, total, "")
        
        document_id = await self._db_writer.write_pipeline_result(
            result=phase1_result,
            source_path=source_path,
            title=title,
            on_progress=db_progress
        )
        
        return document_id
    
    async def process_batch(
        self,
        pdf_paths: List[str | Path],
        on_document_complete: Callable = None
    ) -> List[UnifiedPipelineResult]:
        """
        Process multiple documents.
        
        Args:
            pdf_paths: List of PDF file paths
            on_document_complete: Callback(index, result) after each document
        
        Returns:
            List of UnifiedPipelineResult
        """
        results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Processing document {i+1}/{len(pdf_paths)}: {pdf_path}")
            
            try:
                result = await self.process_document(pdf_path)
                results.append(result)
                
                if on_document_complete:
                    on_document_complete(i, result)
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                error_result = UnifiedPipelineResult(source_path=str(pdf_path))
                error_result.error = str(e)
                results.append(error_result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successful}/{len(results)} successful")
        
        return results
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    async def __aenter__(self) -> 'UnifiedPipeline':
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_document_to_database(
    pdf_path: str | Path,
    connection_string: str,
    title: str = None,
    **config_kwargs
) -> UnifiedPipelineResult:
    """
    Convenience function: process PDF and write to database.
    
    Args:
        pdf_path: Path to PDF file
        connection_string: PostgreSQL connection string
        title: Optional document title
        **config_kwargs: Additional config options
    
    Returns:
        UnifiedPipelineResult
    """
    config = UnifiedPipelineConfig.for_database(
        connection_string=connection_string,
        **config_kwargs
    )
    
    async with UnifiedPipeline(config) as pipeline:
        return await pipeline.process_document(pdf_path, title=title)


async def process_document_to_files(
    pdf_path: str | Path,
    output_dir: Path,
    **config_kwargs
) -> UnifiedPipelineResult:
    """
    Convenience function: process PDF and export to files.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        **config_kwargs: Additional config options
    
    Returns:
        UnifiedPipelineResult
    """
    config = UnifiedPipelineConfig.for_files(
        output_dir=output_dir,
        **config_kwargs
    )
    
    async with UnifiedPipeline(config) as pipeline:
        return await pipeline.process_document(pdf_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Process PDF with unified pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--database", "-d", help="PostgreSQL connection string")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--mode", choices=["file", "database", "both"], default="file")
    parser.add_argument("--title", help="Document title")
    
    args = parser.parse_args()
    
    async def main():
        if args.mode == "database":
            if not args.database:
                print("Error: --database required for database mode")
                sys.exit(1)
            config = UnifiedPipelineConfig.for_database(args.database)
        elif args.mode == "both":
            if not args.database:
                print("Error: --database required for both mode")
                sys.exit(1)
            config = UnifiedPipelineConfig.for_both(args.database, Path(args.output))
        else:
            config = UnifiedPipelineConfig.for_files(Path(args.output))
        
        async with UnifiedPipeline(config) as pipeline:
            result = await pipeline.process_document(args.pdf_path, title=args.title)
            
            print()
            print("=" * 50)
            print("RESULT")
            print("=" * 50)
            print(f"Success: {result.success}")
            print(f"Document ID: {result.document_id}")
            print(f"Chunks: {result.chunk_count}")
            print(f"Images: {result.image_count}")
            print(f"Links: {result.link_count}")
            print(f"Export Path: {result.export_path}")
            if result.error:
                print(f"Error: {result.error}")
    
    asyncio.run(main())
