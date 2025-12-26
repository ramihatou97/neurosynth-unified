#!/usr/bin/env python3
"""
NeuroSynth Unified Pipeline - Usage Examples
=============================================

This script demonstrates the different ways to use the unified pipeline.
"""

import asyncio
from pathlib import Path
import os


async def example_database_mode():
    """Process PDF and write directly to PostgreSQL."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Database Mode")
    print("=" * 60)
    
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    # Get connection string from environment
    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth"
    )
    
    # Create config for database mode
    config = UnifiedPipelineConfig.for_database(
        connection_string=connection_string
    )
    
    print(f"Output mode: {config.output_mode.value}")
    print(f"Database: {connection_string.split('@')[-1] if '@' in connection_string else 'configured'}")
    
    # Process document
    pdf_path = "/path/to/your/document.pdf"
    
    async with UnifiedPipeline(config) as pipeline:
        result = await pipeline.process_document(pdf_path)
        
        print(f"\nResult:")
        print(f"  Document ID: {result.document_id}")
        print(f"  Chunks: {result.chunk_count}")
        print(f"  Images: {result.image_count}")
        print(f"  Links: {result.link_count}")
        print(f"  Success: {result.success}")
    
    return result


async def example_file_mode():
    """Process PDF and export to files (backward compatible)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: File Export Mode")
    print("=" * 60)
    
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    # Create config for file mode
    output_dir = Path("./output")
    config = UnifiedPipelineConfig.for_files(output_dir=output_dir)
    
    print(f"Output mode: {config.output_mode.value}")
    print(f"Output dir: {output_dir}")
    
    # Process document
    pdf_path = "/path/to/your/document.pdf"
    
    async with UnifiedPipeline(config) as pipeline:
        result = await pipeline.process_document(pdf_path)
        
        print(f"\nResult:")
        print(f"  Export path: {result.export_path}")
        print(f"  Chunks: {result.chunk_count}")
        print(f"  Images: {result.image_count}")
        print(f"  Success: {result.success}")
    
    return result


async def example_hybrid_mode():
    """Process PDF, write to database, AND export files."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Hybrid Mode (Database + Files)")
    print("=" * 60)
    
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth"
    )
    output_dir = Path("./backup")
    
    # Create config for both modes
    config = UnifiedPipelineConfig.for_both(
        connection_string=connection_string,
        output_dir=output_dir
    )
    
    print(f"Output mode: {config.output_mode.value}")
    print(f"Database: configured")
    print(f"Backup dir: {output_dir}")
    
    # Process document
    pdf_path = "/path/to/your/document.pdf"
    
    async with UnifiedPipeline(config) as pipeline:
        result = await pipeline.process_document(pdf_path)
        
        print(f"\nResult:")
        print(f"  Document ID: {result.document_id}")
        print(f"  Export path: {result.export_path}")
        print(f"  Chunks: {result.chunk_count}")
        print(f"  Images: {result.image_count}")
        print(f"  Links: {result.link_count}")
        print(f"  Success: {result.success}")
    
    return result


async def example_batch_processing():
    """Process multiple PDFs."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 60)
    
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    config = UnifiedPipelineConfig.for_files(output_dir=Path("./batch_output"))
    
    # List of PDFs
    pdf_paths = [
        "/path/to/document1.pdf",
        "/path/to/document2.pdf",
        "/path/to/document3.pdf",
    ]
    
    def on_complete(index, result):
        status = "✓" if result.success else "✗"
        print(f"  [{status}] Document {index + 1}: {result.chunk_count} chunks")
    
    async with UnifiedPipeline(config) as pipeline:
        results = await pipeline.process_batch(
            pdf_paths,
            on_document_complete=on_complete
        )
    
    print(f"\nBatch Summary:")
    successful = sum(1 for r in results if r.success)
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    
    return results


async def example_with_progress():
    """Process with progress callbacks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: With Progress Callbacks")
    print("=" * 60)
    
    from src.ingest import UnifiedPipeline, UnifiedPipelineConfig
    
    def progress_callback(stage, current, total, message=""):
        bar_width = 30
        filled = int(bar_width * current / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {stage}: {current}/{total} {message}", end="", flush=True)
        if current >= total:
            print()
    
    config = UnifiedPipelineConfig.for_files(output_dir=Path("./output"))
    
    pdf_path = "/path/to/your/document.pdf"
    
    pipeline = UnifiedPipeline(config, on_progress=progress_callback)
    
    try:
        await pipeline.initialize()
        result = await pipeline.process_document(pdf_path)
        
        print(f"\nCompleted: {result.chunk_count} chunks, {result.image_count} images")
    finally:
        await pipeline.close()


async def example_convenience_functions():
    """Use convenience functions for simple cases."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Convenience Functions")
    print("=" * 60)
    
    from src.ingest import process_document_to_database, process_document_to_files
    
    pdf_path = "/path/to/your/document.pdf"
    
    # Option A: To database
    # result = await process_document_to_database(
    #     pdf_path,
    #     connection_string="postgresql://...",
    #     title="My Document"
    # )
    
    # Option B: To files
    result = await process_document_to_files(
        pdf_path,
        output_dir=Path("./quick_output")
    )
    
    print(f"Result: {result.chunk_count} chunks, {result.image_count} images")


async def main():
    """Run examples (comment out ones you don't want to run)."""
    print("NeuroSynth Unified Pipeline Examples")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # await example_database_mode()
    # await example_file_mode()
    # await example_hybrid_mode()
    # await example_batch_processing()
    # await example_with_progress()
    # await example_convenience_functions()
    
    print("\nUncomment an example function in main() to run it.")
    print("\nQuick start:")
    print("  from src.ingest import UnifiedPipeline, UnifiedPipelineConfig")
    print("  config = UnifiedPipelineConfig.for_database(connection_string)")
    print("  async with UnifiedPipeline(config) as pipeline:")
    print("      result = await pipeline.process_document('file.pdf')")


if __name__ == "__main__":
    asyncio.run(main())
