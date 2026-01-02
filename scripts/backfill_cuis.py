#!/usr/bin/env python3
"""
Backfill UMLS CUIs for chunks missing them.

Usage:
    DATABASE_URL=... python scripts/backfill_cuis.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseConnection
from src.core.umls_extractor import UMLSExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backfill_cuis")


async def backfill_cuis():
    """Backfill UMLS CUIs for chunks that are missing them."""

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("Initializing database connection...")
    db = await DatabaseConnection.initialize(db_url)

    logger.info("Initializing UMLS Extractor...")
    extractor = UMLSExtractor()

    # Count chunks needing CUIs
    total_to_process = await db.fetchval("""
        SELECT COUNT(*) FROM chunks
        WHERE cuis IS NULL OR cuis = '{}'
    """)

    logger.info(f"Found {total_to_process} chunks missing CUIs")

    if total_to_process == 0:
        logger.info("All chunks already have CUIs. Nothing to do.")
        return

    batch_size = 50
    processed = 0
    updated = 0
    errors = 0

    while processed < total_to_process:
        # Fetch batch
        rows = await db.fetch("""
            SELECT id, content
            FROM chunks
            WHERE cuis IS NULL OR cuis = '{}'
            ORDER BY created_at
            LIMIT $1
        """, batch_size)

        if not rows:
            break

        for row in rows:
            chunk_id = row['id']
            content = row['content']

            if not content or len(content.strip()) < 50:
                # Skip very short chunks
                continue

            try:
                # Extract UMLS entities
                entities = extractor.extract(content)

                # Filter to high-confidence CUIs (score >= 0.7)
                # UMLSEntity has 'score' attribute, not 'confidence'
                cuis = list(set([
                    e.cui for e in entities
                    if e.cui and e.score >= 0.7
                ]))

                if cuis:
                    await db.execute("""
                        UPDATE chunks
                        SET cuis = $1
                        WHERE id = $2
                    """, cuis, chunk_id)
                    updated += 1

            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_id}: {e}")
                errors += 1

        processed += len(rows)
        logger.info(f"Progress: {processed}/{total_to_process} processed, {updated} updated, {errors} errors")

    # Final verification
    final_stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE cuis IS NOT NULL AND array_length(cuis, 1) > 0) as with_cuis
        FROM chunks
    """)

    await db.close()

    logger.info("=" * 50)
    logger.info("CUI BACKFILL COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Total chunks: {final_stats['total']}")
    logger.info(f"  Chunks with CUIs: {final_stats['with_cuis']}")
    logger.info(f"  Coverage: {final_stats['with_cuis']/final_stats['total']*100:.1f}%")
    logger.info(f"  Errors: {errors}")


if __name__ == "__main__":
    asyncio.run(backfill_cuis())
