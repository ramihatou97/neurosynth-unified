#!/usr/bin/env python3
"""
NeuroSynth - Backfill UMLS CUIs
================================

Extracts UMLS Concept Unique Identifiers (CUIs) from chunks and images
using SciSpacy's UMLS entity linker.

Prerequisites:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

Usage:
    cd <project-root>
    source venv/bin/activate
    DATABASE_URL=$DATABASE_URL python scripts/backfill_cuis.py

    # For a specific document:
    python scripts/backfill_cuis.py --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

    # Chunks only (faster):
    python scripts/backfill_cuis.py --chunks-only

    # Dry run (preview only):
    python scripts/backfill_cuis.py --dry-run
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from uuid import UUID

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCISPACY UMLS EXTRACTOR
# =============================================================================

class SciSpacyExtractor:
    """Extract UMLS CUIs using SciSpacy."""
    
    def __init__(self, model: str = "en_core_sci_lg"):
        self.model_name = model
        self.nlp = None
        self._loaded = False

    def load(self) -> bool:
        """Load the SciSpacy model with UMLS linker."""
        if self._loaded:
            return True

        try:
            import spacy
            from scispacy.linking import EntityLinker
            
            logger.info(f"Loading SciSpacy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            
            # Add UMLS linker
            logger.info("Adding UMLS entity linker...")
            self.nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "threshold": 0.7,  # Confidence threshold
                    "max_entities_per_mention": 1
                }
            )
            
            self._loaded = True
            logger.info("✓ SciSpacy model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"SciSpacy not installed: {e}")
            logger.error("Install with:")
            logger.error("  pip install scispacy")
            logger.error("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
            return False
        except Exception as e:
            logger.error(f"Failed to load SciSpacy model: {e}")
            return False

    def extract_cuis(self, text: str, max_length: int = 10000) -> List[str]:
        """
        Extract UMLS CUIs from text.
        
        Returns list of unique CUI strings (e.g., ["C0016504", "C0018787"])
        """
        if not self._loaded:
            if not self.load():
                return []

        # Truncate very long texts
        if len(text) > max_length:
            text = text[:max_length]

        try:
            doc = self.nlp(text)
            cuis: Set[str] = set()

            for ent in doc.ents:
                if hasattr(ent, '_') and hasattr(ent._, 'kb_ents'):
                    for cui, score in ent._.kb_ents:
                        if score >= 0.7:  # Confidence threshold
                            cuis.add(cui)

            return list(cuis)

        except Exception as e:
            logger.warning(f"CUI extraction failed: {e}")
            return []

    def unload(self):
        """Unload model to free memory."""
        import gc
        if self.nlp:
            del self.nlp
            self.nlp = None
            self._loaded = False
            gc.collect()
            logger.info("SciSpacy model unloaded")


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def get_chunks_without_cuis(
    db,
    document_id: Optional[str] = None,
    limit: int = 0
) -> List[Dict[str, Any]]:
    """Get chunks that need CUI extraction."""
    
    query = """
        SELECT id, content, document_id
        FROM chunks
        WHERE (cuis IS NULL OR cuis = '{}')
    """
    params = []
    
    if document_id:
        query += " AND document_id = $1::uuid"
        params.append(document_id)
    
    query += " ORDER BY id"
    
    if limit > 0:
        query += f" LIMIT {limit}"
    
    rows = await db.fetch(query, *params)
    return [dict(row) for row in rows]


async def get_images_without_cuis(
    db,
    document_id: Optional[str] = None,
    limit: int = 0
) -> List[Dict[str, Any]]:
    """Get images that need CUI extraction (from captions)."""
    
    query = """
        SELECT id, vlm_caption, document_id
        FROM images
        WHERE vlm_caption IS NOT NULL
          AND (cuis IS NULL OR cuis = '{}')
          AND (is_decorative IS NULL OR NOT is_decorative)
    """
    params = []
    
    if document_id:
        query += " AND document_id = $1::uuid"
        params.append(document_id)
    
    query += " ORDER BY id"
    
    if limit > 0:
        query += f" LIMIT {limit}"
    
    rows = await db.fetch(query, *params)
    return [dict(row) for row in rows]


async def update_chunk_cuis(db, chunk_id: str, cuis: List[str]) -> bool:
    """Update a chunk's CUI array."""
    try:
        await db.execute("""
            UPDATE chunks
            SET cuis = $1
            WHERE id = $2::uuid
        """, cuis, chunk_id)
        return True
    except Exception as e:
        logger.warning(f"Failed to update chunk {chunk_id}: {e}")
        return False


async def update_image_cuis(db, image_id: str, cuis: List[str]) -> bool:
    """Update an image's CUI array."""
    try:
        await db.execute("""
            UPDATE images
            SET cuis = $1
            WHERE id = $2::uuid
        """, cuis, image_id)
        return True
    except Exception as e:
        logger.warning(f"Failed to update image {image_id}: {e}")
        return False


# =============================================================================
# MAIN BACKFILL LOGIC
# =============================================================================

async def backfill_cuis(
    connection_string: str,
    document_id: Optional[str] = None,
    chunks_only: bool = False,
    images_only: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    batch_size: int = 50
) -> Dict[str, Any]:
    """
    Backfill UMLS CUIs for chunks and images.

    Args:
        connection_string: PostgreSQL connection string
        document_id: Specific document ID (optional)
        chunks_only: Only process chunks
        images_only: Only process images
        dry_run: If True, don't modify database
        limit: Limit number of items to process (0 = all)
        batch_size: Log progress every N items

    Returns:
        Statistics dictionary
    """
    from src.database import init_database, close_database

    stats = {
        'chunks_processed': 0,
        'chunks_with_cuis': 0,
        'images_processed': 0,
        'images_with_cuis': 0,
        'total_cuis_found': 0,
        'errors': 0
    }

    # Initialize extractor
    extractor = SciSpacyExtractor()
    if not extractor.load():
        logger.error("Failed to load SciSpacy model")
        return stats

    logger.info("Connecting to database...")
    db = await init_database(connection_string)

    try:
        # Process chunks
        if not images_only:
            chunks = await get_chunks_without_cuis(db, document_id, limit)
            logger.info(f"\nFound {len(chunks)} chunks without CUIs")

            for i, chunk in enumerate(chunks):
                chunk_id = str(chunk['id'])
                content = chunk['content'] or ''

                # Extract CUIs
                cuis = extractor.extract_cuis(content)
                stats['chunks_processed'] += 1

                if cuis:
                    stats['chunks_with_cuis'] += 1
                    stats['total_cuis_found'] += len(cuis)

                    if not dry_run:
                        if await update_chunk_cuis(db, chunk_id, cuis):
                            pass
                        else:
                            stats['errors'] += 1
                    else:
                        logger.debug(f"  [DRY RUN] Would update chunk {chunk_id[:8]} with {len(cuis)} CUIs")

                # Progress logging
                if (i + 1) % batch_size == 0:
                    logger.info(f"  Processed {i + 1}/{len(chunks)} chunks, found {stats['chunks_with_cuis']} with CUIs")

            logger.info(f"✓ Processed {stats['chunks_processed']} chunks, {stats['chunks_with_cuis']} had CUIs")

        # Process images
        if not chunks_only:
            images = await get_images_without_cuis(db, document_id, limit)
            logger.info(f"\nFound {len(images)} images without CUIs")

            for i, image in enumerate(images):
                image_id = str(image['id'])
                caption = image['vlm_caption'] or ''

                # Extract CUIs from caption
                cuis = extractor.extract_cuis(caption)
                stats['images_processed'] += 1

                if cuis:
                    stats['images_with_cuis'] += 1
                    stats['total_cuis_found'] += len(cuis)

                    if not dry_run:
                        if await update_image_cuis(db, image_id, cuis):
                            pass
                        else:
                            stats['errors'] += 1
                    else:
                        logger.debug(f"  [DRY RUN] Would update image {image_id[:8]} with {len(cuis)} CUIs")

                # Progress logging
                if (i + 1) % batch_size == 0:
                    logger.info(f"  Processed {i + 1}/{len(images)} images, found {stats['images_with_cuis']} with CUIs")

            logger.info(f"✓ Processed {stats['images_processed']} images, {stats['images_with_cuis']} had CUIs")

        return stats

    finally:
        await close_database()
        extractor.unload()


def print_summary(stats: dict, dry_run: bool = False):
    """Print backfill summary."""
    print()
    print("=" * 60)
    print(f"CUI BACKFILL {'PREVIEW (DRY RUN)' if dry_run else 'COMPLETE'}")
    print("=" * 60)
    print()
    print("Statistics:")
    print(f"  Chunks processed:     {stats['chunks_processed']}")
    print(f"  Chunks with CUIs:     {stats['chunks_with_cuis']}")
    print(f"  Images processed:     {stats['images_processed']}")
    print(f"  Images with CUIs:     {stats['images_with_cuis']}")
    print(f"  Total CUIs found:     {stats['total_cuis_found']}")
    print(f"  Errors:               {stats['errors']}")
    print()

    if stats['total_cuis_found'] > 0 and not dry_run:
        print("Impact:")
        print("  - TriPassLinker Pass 2 (CUI overlap) now enabled")
        print("  - Entity-based search filtering available")
        print()
        print("Verification:")
        print("  SELECT COUNT(*) FROM chunks WHERE array_length(cuis, 1) > 0;")
        print("  SELECT COUNT(*) FROM images WHERE array_length(cuis, 1) > 0;")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill UMLS CUIs for chunks and images"
    )

    parser.add_argument(
        "--database", "-d",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth"
        ),
        help="PostgreSQL connection string"
    )

    parser.add_argument(
        "--document-id", "-doc",
        default=None,
        help="Process specific document ID only"
    )

    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only process chunks (skip images)"
    )

    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only process images (skip chunks)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items to process (0 = all)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Log progress every N items"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database"
    )

    args = parser.parse_args()

    print("NeuroSynth CUI Backfill")
    print("=" * 60)
    print(f"Database: {args.database.split('@')[-1] if '@' in args.database else 'configured'}")
    if args.document_id:
        print(f"Document: {args.document_id}")
    print(f"Chunks only: {args.chunks_only}")
    print(f"Images only: {args.images_only}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Dry Run: {args.dry_run}")
    print()

    stats = await backfill_cuis(
        connection_string=args.database,
        document_id=args.document_id,
        chunks_only=args.chunks_only,
        images_only=args.images_only,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size
    )

    print_summary(stats, args.dry_run)

    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
