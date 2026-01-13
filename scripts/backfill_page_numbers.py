#!/usr/bin/env python3
"""
NeuroSynth - Backfill Page Numbers
===================================

Estimates and populates page numbers for chunks that are missing them.

Methods:
1. If chunk has page_start/page_end, use those
2. If document has total_pages, distribute chunks proportionally
3. Fallback: estimate from chunk index within document

Usage:
    cd <project-root>
    source venv/bin/activate
    DATABASE_URL=$DATABASE_URL python scripts/backfill_page_numbers.py

    # For a specific document:
    python scripts/backfill_page_numbers.py --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

    # Dry run (preview only):
    python scripts/backfill_page_numbers.py --dry-run
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
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
# DATABASE OPERATIONS
# =============================================================================

async def get_documents_needing_page_numbers(
    db,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get documents with chunks missing page numbers."""
    
    if document_id:
        query = """
            SELECT 
                d.id,
                d.title,
                d.total_pages,
                COUNT(c.id) as total_chunks,
                SUM(CASE WHEN c.page_number IS NULL THEN 1 ELSE 0 END) as missing_pages
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.id = $1::uuid
            GROUP BY d.id, d.title, d.total_pages
        """
        rows = await db.fetch(query, document_id)
    else:
        query = """
            SELECT 
                d.id,
                d.title,
                d.total_pages,
                COUNT(c.id) as total_chunks,
                SUM(CASE WHEN c.page_number IS NULL THEN 1 ELSE 0 END) as missing_pages
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id, d.title, d.total_pages
            HAVING SUM(CASE WHEN c.page_number IS NULL THEN 1 ELSE 0 END) > 0
            ORDER BY missing_pages DESC
        """
        rows = await db.fetch(query)

    return [dict(row) for row in rows]


async def get_chunks_for_document(db, doc_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a document ordered by index."""
    query = """
        SELECT 
            id, 
            page_number,
            chunk_index,
            content,
            COALESCE(metadata->>'page_start', '0') as meta_page_start,
            COALESCE(metadata->>'page_end', '0') as meta_page_end
        FROM chunks
        WHERE document_id = $1::uuid
        ORDER BY chunk_index NULLS LAST, id
    """
    rows = await db.fetch(query, doc_id)
    return [dict(row) for row in rows]


async def update_chunk_page_number(db, chunk_id: str, page_number: int) -> bool:
    """Update a single chunk's page number."""
    try:
        await db.execute("""
            UPDATE chunks
            SET page_number = $1
            WHERE id = $2::uuid
        """, page_number, chunk_id)
        return True
    except Exception as e:
        logger.warning(f"Failed to update chunk {chunk_id}: {e}")
        return False


async def batch_update_page_numbers(
    db,
    updates: List[tuple]  # [(chunk_id, page_number), ...]
) -> int:
    """Batch update page numbers."""
    if not updates:
        return 0

    updated = 0
    
    # Use a transaction for batch updates
    async with db.transaction():
        for chunk_id, page_number in updates:
            try:
                await db.execute("""
                    UPDATE chunks
                    SET page_number = $1
                    WHERE id = $2::uuid
                """, page_number, chunk_id)
                updated += 1
            except Exception as e:
                logger.warning(f"Failed to update chunk {chunk_id}: {e}")

    return updated


# =============================================================================
# PAGE NUMBER ESTIMATION
# =============================================================================

def estimate_page_numbers(
    chunks: List[Dict[str, Any]],
    total_pages: Optional[int]
) -> List[tuple]:
    """
    Estimate page numbers for chunks.
    
    Returns list of (chunk_id, estimated_page_number)
    """
    updates = []
    
    if not chunks:
        return updates

    # First, check if we have metadata page info
    for i, chunk in enumerate(chunks):
        chunk_id = str(chunk['id'])
        
        # Already has page number
        if chunk['page_number'] is not None:
            continue

        # Try metadata page_start
        try:
            meta_start = int(chunk.get('meta_page_start', 0))
            if meta_start > 0:
                updates.append((chunk_id, meta_start))
                continue
        except (ValueError, TypeError):
            pass

        # Estimate from position
        if total_pages and total_pages > 0:
            # Distribute chunks proportionally across pages
            estimated_page = max(1, int((i / len(chunks)) * total_pages) + 1)
            estimated_page = min(estimated_page, total_pages)
            updates.append((chunk_id, estimated_page))
        else:
            # Fallback: assume 3-5 chunks per page
            estimated_page = max(1, (i // 4) + 1)
            updates.append((chunk_id, estimated_page))

    return updates


# =============================================================================
# MAIN BACKFILL LOGIC
# =============================================================================

async def backfill_page_numbers(
    connection_string: str,
    document_id: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Backfill page numbers for chunks.

    Args:
        connection_string: PostgreSQL connection string
        document_id: Specific document ID (optional)
        dry_run: If True, don't modify database

    Returns:
        Statistics dictionary
    """
    from src.database import init_database, close_database

    stats = {
        'documents_processed': 0,
        'total_chunks': 0,
        'chunks_updated': 0,
        'chunks_skipped': 0,
        'errors': 0
    }

    logger.info("Connecting to database...")
    db = await init_database(connection_string)

    try:
        # Get documents to process
        documents = await get_documents_needing_page_numbers(db, document_id)
        logger.info(f"Found {len(documents)} documents with missing page numbers")

        if not documents:
            logger.info("No documents need page number backfill")
            return stats

        for doc in documents:
            doc_id = str(doc['id'])
            title = doc['title'] or doc_id[:8]
            total_pages = doc['total_pages']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {title}")
            logger.info(f"  Total pages: {total_pages or 'unknown'}")
            logger.info(f"  Total chunks: {doc['total_chunks']}")
            logger.info(f"  Missing page numbers: {doc['missing_pages']}")

            try:
                # Fetch chunks
                chunks = await get_chunks_for_document(db, doc_id)
                stats['total_chunks'] += len(chunks)

                # Estimate page numbers
                updates = estimate_page_numbers(chunks, total_pages)
                
                already_set = len(chunks) - len(updates)
                stats['chunks_skipped'] += already_set
                
                logger.info(f"  Already have page numbers: {already_set}")
                logger.info(f"  Will update: {len(updates)}")

                if not updates:
                    stats['documents_processed'] += 1
                    continue

                if dry_run:
                    logger.info("  [DRY RUN] Would update page numbers")
                    # Show sample
                    for chunk_id, page in updates[:5]:
                        logger.info(f"    {chunk_id[:8]}... → page {page}")
                    if len(updates) > 5:
                        logger.info(f"    ... and {len(updates) - 5} more")
                    stats['chunks_updated'] += len(updates)
                else:
                    # Apply updates
                    updated = await batch_update_page_numbers(db, updates)
                    stats['chunks_updated'] += updated
                    logger.info(f"  Updated {updated} chunks")

                stats['documents_processed'] += 1

            except Exception as e:
                logger.error(f"  Error processing document: {e}")
                stats['errors'] += 1
                continue

        return stats

    finally:
        await close_database()


def print_summary(stats: dict, dry_run: bool = False):
    """Print backfill summary."""
    print()
    print("=" * 60)
    print(f"PAGE NUMBER BACKFILL {'PREVIEW (DRY RUN)' if dry_run else 'COMPLETE'}")
    print("=" * 60)
    print()
    print("Statistics:")
    print(f"  Documents processed:  {stats['documents_processed']}")
    print(f"  Total chunks:         {stats['total_chunks']}")
    print(f"  Chunks updated:       {stats['chunks_updated']}")
    print(f"  Chunks skipped:       {stats['chunks_skipped']}")
    print(f"  Errors:               {stats['errors']}")
    print()

    if stats['chunks_updated'] > 0 and not dry_run:
        print("Verification:")
        print("  SELECT COUNT(*) FROM chunks WHERE page_number IS NOT NULL;")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill page numbers for chunks"
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
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database"
    )

    args = parser.parse_args()

    print("NeuroSynth Page Number Backfill")
    print("=" * 60)
    print(f"Database: {args.database.split('@')[-1] if '@' in args.database else 'configured'}")
    if args.document_id:
        print(f"Document: {args.document_id}")
    print(f"Dry Run: {args.dry_run}")
    print()

    stats = await backfill_page_numbers(
        connection_string=args.database,
        document_id=args.document_id,
        dry_run=args.dry_run
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
