#!/usr/bin/env python3
"""
NeuroSynth - Migrate Phase 1 Exports to Database
=================================================

Migrates existing Phase 1 file exports (chunks.pkl, images.pkl, links.json)
into the PostgreSQL database.

Usage:
    # Single export directory
    python migrate_phase1_exports.py /path/to/export/
    
    # Multiple exports
    python migrate_phase1_exports.py /path/to/exports/ --batch
    
    # With options
    python migrate_phase1_exports.py /path/to/export/ \\
        --database postgresql://user:pass@localhost/neurosynth \\
        --keep-files
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.database_writer import PipelineDatabaseWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase1ExportLoader:
    """Loads Phase 1 export files into a result-like object."""
    
    def __init__(self, export_dir: Path):
        self.export_dir = Path(export_dir)
        self.chunks = []
        self.images = []
        self.links = []
        self.manifest = {}
    
    def load(self) -> 'Phase1ExportLoader':
        """Load all export files."""
        if not self.export_dir.exists():
            raise FileNotFoundError(f"Export directory not found: {self.export_dir}")
        
        # Load manifest
        manifest_path = self.export_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            logger.info(f"Loaded manifest: document_id={self.manifest.get('document_id')}")
        
        # Load chunks
        chunks_path = self.export_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks")
        else:
            logger.warning("No chunks.pkl found")
        
        # Load images
        images_path = self.export_dir / "images.pkl"
        if images_path.exists():
            with open(images_path, 'rb') as f:
                self.images = pickle.load(f)
            logger.info(f"Loaded {len(self.images)} images")
        else:
            logger.warning("No images.pkl found")
        
        # Load links
        links_path = self.export_dir / "links.json"
        if links_path.exists():
            with open(links_path, 'r') as f:
                self.links = json.load(f)
            logger.info(f"Loaded {len(self.links)} links")
        else:
            logger.warning("No links.json found")
        
        return self
    
    @property
    def source_path(self) -> str:
        return self.manifest.get('source_path', str(self.export_dir))
    
    @property
    def document_id(self) -> str:
        return self.manifest.get('document_id')
    
    @property
    def text_embedding_dim(self) -> int:
        return self.manifest.get('text_embedding_dim', 1024)
    
    @property
    def image_embedding_dim(self) -> int:
        return self.manifest.get('image_embedding_dim', 512)
    
    @property
    def processing_time_seconds(self) -> float:
        return self.manifest.get('processing_time_seconds', 0)
    
    @property
    def total_pages(self) -> int:
        return self.manifest.get('total_pages', 0)


def find_export_directories(base_path: Path) -> List[Path]:
    """Find all directories containing Phase 1 exports."""
    exports = []
    
    # Check if base_path is an export itself
    if (base_path / "manifest.json").exists() or (base_path / "chunks.pkl").exists():
        exports.append(base_path)
        return exports
    
    # Search subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            if (subdir / "manifest.json").exists() or (subdir / "chunks.pkl").exists():
                exports.append(subdir)
    
    return exports


async def migrate_export(
    export_dir: Path,
    connection_string: str,
    keep_files: bool = True
) -> Dict[str, Any]:
    """
    Migrate a single Phase 1 export to database.
    
    Returns:
        Dict with migration results
    """
    result = {
        'export_dir': str(export_dir),
        'success': False,
        'document_id': None,
        'chunks': 0,
        'images': 0,
        'links': 0,
        'error': None
    }
    
    try:
        # Load export
        loader = Phase1ExportLoader(export_dir).load()
        
        if not loader.chunks and not loader.images:
            result['error'] = "No chunks or images found"
            logger.warning(f"Skipping {export_dir}: {result['error']}")
            return result
        
        # Write to database
        writer = PipelineDatabaseWriter(connection_string=connection_string)
        await writer.connect()
        
        try:
            doc_id = await writer.write_pipeline_result(
                result=loader,
                source_path=loader.source_path
            )
            
            result['success'] = True
            result['document_id'] = str(doc_id)
            result['chunks'] = len(loader.chunks)
            result['images'] = len(loader.images)
            result['links'] = len(loader.links)
            
            logger.info(f"✓ Migrated {export_dir.name}: {result['chunks']} chunks, {result['images']} images")
            
            # Optionally delete files
            if not keep_files:
                import shutil
                shutil.rmtree(export_dir)
                logger.info(f"  Deleted {export_dir}")
            
        finally:
            await writer.close()
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"✗ Failed to migrate {export_dir}: {e}")
    
    return result


async def migrate_batch(
    base_path: Path,
    connection_string: str,
    keep_files: bool = True
) -> List[Dict[str, Any]]:
    """Migrate multiple exports."""
    exports = find_export_directories(base_path)
    
    if not exports:
        logger.warning(f"No exports found in {base_path}")
        return []
    
    logger.info(f"Found {len(exports)} exports to migrate")
    
    results = []
    for i, export_dir in enumerate(exports):
        logger.info(f"\n[{i+1}/{len(exports)}] Processing {export_dir.name}")
        result = await migrate_export(export_dir, connection_string, keep_files)
        results.append(result)
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print migration summary."""
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_chunks = sum(r['chunks'] for r in successful)
    total_images = sum(r['images'] for r in successful)
    total_links = sum(r['links'] for r in successful)
    
    print(f"\nTotal exports: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    print(f"Total chunks migrated: {total_chunks}")
    print(f"Total images migrated: {total_images}")
    print(f"Total links migrated: {total_links}")
    
    if failed:
        print("\nFailed exports:")
        for r in failed:
            print(f"  - {r['export_dir']}: {r['error']}")
    
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate Phase 1 exports to database"
    )
    
    parser.add_argument(
        "path",
        help="Path to export directory or base directory for batch"
    )
    
    parser.add_argument(
        "--database", "-d",
        default="postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth",
        help="PostgreSQL connection string"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Migrate all exports in subdirectories"
    )
    
    parser.add_argument(
        "--keep-files", "-k",
        action="store_true",
        default=True,
        help="Keep original files after migration (default: True)"
    )
    
    parser.add_argument(
        "--delete-files",
        action="store_true",
        help="Delete original files after successful migration"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    keep_files = not args.delete_files
    
    print("NeuroSynth Phase 1 Export Migration")
    print("=" * 60)
    print(f"Source: {path}")
    print(f"Database: {args.database.split('@')[-1]}")
    print(f"Mode: {'batch' if args.batch else 'single'}")
    print(f"Keep files: {keep_files}")
    print()
    
    if args.batch:
        results = await migrate_batch(path, args.database, keep_files)
    else:
        result = await migrate_export(path, args.database, keep_files)
        results = [result]
    
    print_summary(results)
    
    # Exit code based on success
    if all(r['success'] for r in results):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
