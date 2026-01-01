#!/usr/bin/env python3
"""
NeuroSynth - Build FAISS Indexes from Database
===============================================

Builds FAISS indexes from PostgreSQL database for fast similarity search.

Usage:
    # Build all indexes
    python build_indexes.py
    
    # With options
    python build_indexes.py \
        --database postgresql://user:pass@localhost/neurosynth \
        --output ./indexes \
        --index-type IVFFlat
    
    # Rebuild specific index
    python build_indexes.py --only text
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import time
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.retrieval.faiss_manager import (
    FAISSManager,
    FAISSIndexConfig,
    TEXT_CONFIG,
    IMAGE_CONFIG,
    CAPTION_CONFIG
)
from src.database import init_database, close_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def fetch_chunk_embeddings(db) -> tuple:
    """Fetch chunk embeddings from database."""
    logger.info("Fetching chunk embeddings...")
    
    rows = await db.fetch("""
        SELECT id, embedding
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY id
    """)
    
    embeddings = []
    ids = []
    
    for row in rows:
        emb = row['embedding']
        if emb is not None:
            embeddings.append(np.array(emb, dtype=np.float32))
            ids.append(str(row['id']))
    
    logger.info(f"Fetched {len(embeddings)} chunk embeddings")
    return embeddings, ids


async def fetch_image_embeddings(db) -> tuple:
    """Fetch image embeddings from database."""
    logger.info("Fetching image embeddings...")

    rows = await db.fetch("""
        SELECT id, image_embedding, caption_embedding
        FROM images
        WHERE (is_decorative IS NULL OR NOT is_decorative)
        ORDER BY id
    """)

    visual_embeddings = []
    visual_ids = []
    caption_embeddings = []
    caption_ids = []

    for row in rows:
        img_id = str(row['id'])

        # Visual embeddings: image_embedding (512d BiomedCLIP)
        if row['image_embedding'] is not None:
            visual_embeddings.append(np.array(row['image_embedding'], dtype=np.float32))
            visual_ids.append(img_id)

        # Caption embeddings: caption_embedding (1024d Voyage)
        if row['caption_embedding'] is not None:
            caption_embeddings.append(np.array(row['caption_embedding'], dtype=np.float32))
            caption_ids.append(img_id)

    logger.info(f"Fetched {len(visual_embeddings)} visual embeddings, {len(caption_embeddings)} caption embeddings")
    return visual_embeddings, visual_ids, caption_embeddings, caption_ids


async def build_indexes(
    connection_string: str,
    output_dir: Path,
    index_type: str = "IVFFlat",
    only: str = None
) -> dict:
    """
    Build FAISS indexes from database.
    
    Args:
        connection_string: PostgreSQL connection string
        output_dir: Directory for index files
        index_type: FAISS index type (Flat, IVFFlat, IVFPQ, HNSW)
        only: Build only specific index (text, image, caption)
    
    Returns:
        Dict with index statistics
    """
    stats = {
        'text': 0,
        'image': 0,
        'caption': 0,
        'build_time_seconds': 0
    }
    
    start_time = time.time()
    
    # Connect to database
    logger.info(f"Connecting to database...")
    db = await init_database(connection_string)
    
    try:
        # Configure indexes
        text_config = FAISSIndexConfig(
            name="text",
            dimension=1024,
            index_type=index_type,
            nlist=100,
            nprobe=10
        )
        
        image_config = FAISSIndexConfig(
            name="image",
            dimension=512,
            index_type=index_type,
            nlist=50,
            nprobe=5
        )
        
        caption_config = FAISSIndexConfig(
            name="caption",
            dimension=1024,
            index_type=index_type,
            nlist=50,
            nprobe=10
        )
        
        # Create manager
        manager = FAISSManager(
            index_dir=output_dir,
            text_config=text_config,
            image_config=image_config,
            caption_config=caption_config
        )
        
        # Build text index
        if only is None or only == "text":
            embeddings, ids = await fetch_chunk_embeddings(db)
            
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                stats['text'] = manager.build_text_index(embeddings_array, ids)
                logger.info(f"Built text index: {stats['text']} vectors")
            else:
                logger.warning("No chunk embeddings found")
        
        # Build image indexes
        if only is None or only in ("image", "caption"):
            visual_emb, visual_ids, caption_emb, caption_ids = await fetch_image_embeddings(db)
            
            if only is None or only == "image":
                if visual_emb:
                    embeddings_array = np.vstack(visual_emb)
                    stats['image'] = manager.build_image_index(embeddings_array, visual_ids)
                    logger.info(f"Built image index: {stats['image']} vectors")
                else:
                    logger.warning("No visual embeddings found")
            
            if only is None or only == "caption":
                if caption_emb:
                    embeddings_array = np.vstack(caption_emb)
                    stats['caption'] = manager.build_caption_index(embeddings_array, caption_ids)
                    logger.info(f"Built caption index: {stats['caption']} vectors")
                else:
                    logger.warning("No caption embeddings found")
        
        # Save indexes
        manager.save()
        
        stats['build_time_seconds'] = time.time() - start_time
        
        return stats
        
    finally:
        await close_database()


def print_summary(stats: dict, output_dir: Path):
    """Print build summary."""
    print()
    print("=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Index Sizes:")
    print(f"  Text (chunks):  {stats['text']:,} vectors")
    print(f"  Image (visual): {stats['image']:,} vectors")
    print(f"  Caption (text): {stats['caption']:,} vectors")
    print()
    print(f"Build time: {stats['build_time_seconds']:.2f} seconds")
    print()
    
    # List files
    if output_dir.exists():
        print("Files created:")
        for f in sorted(output_dir.iterdir()):
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name}: {size_str}")


async def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes from database"
    )
    
    parser.add_argument(
        "--database", "-d",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://neurosynth:neurosynth@localhost:5432/neurosynth"
        ),
        help="PostgreSQL connection string"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./indexes",
        help="Output directory for index files"
    )
    
    parser.add_argument(
        "--index-type", "-t",
        choices=["Flat", "IVFFlat", "IVFPQ", "HNSW"],
        default="IVFFlat",
        help="FAISS index type"
    )
    
    parser.add_argument(
        "--only",
        choices=["text", "image", "caption"],
        help="Build only specific index"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("NeuroSynth FAISS Index Builder")
    print("=" * 60)
    print(f"Database: {args.database.split('@')[-1]}")
    print(f"Output: {output_dir}")
    print(f"Index type: {args.index_type}")
    if args.only:
        print(f"Building: {args.only} only")
    print()
    
    try:
        stats = await build_indexes(
            connection_string=args.database,
            output_dir=output_dir,
            index_type=args.index_type,
            only=args.only
        )
        
        print_summary(stats, output_dir)
        return 0
        
    except Exception as e:
        logger.exception(f"Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
