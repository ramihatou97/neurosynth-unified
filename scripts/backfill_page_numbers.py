#!/usr/bin/env python3
"""
Backfill page numbers for chunks based on position in document.

Usage:
    DATABASE_URL=... python scripts/backfill_page_numbers.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def backfill_page_numbers():
    """Estimate page numbers for chunks based on position in document."""

    from src.database.connection import DatabaseConnection

    print("=" * 60)
    print("PAGE NUMBER BACKFILL")
    print("=" * 60)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    db = await DatabaseConnection.initialize(db_url)

    # Check current state
    null_count = await db.fetchval("SELECT COUNT(*) FROM chunks WHERE page_number IS NULL")
    total_count = await db.fetchval("SELECT COUNT(*) FROM chunks")

    print(f"\nChunks missing page_number: {null_count}/{total_count}")

    if null_count == 0:
        print("All chunks already have page numbers!")
        await db.close()
        return

    # Get documents with page counts
    docs = await db.fetch("""
        SELECT d.id, d.title, d.total_pages,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) as chunk_count
        FROM documents d
        WHERE (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) > 0
    """)

    print(f"Processing {len(docs)} documents...\n")

    total_updated = 0

    for doc in docs:
        doc_id = doc['id']
        total_pages = doc['total_pages'] or 1
        chunk_count = doc['chunk_count']

        if chunk_count == 0:
            continue

        # Get chunks ordered by creation (approximates document order)
        chunks = await db.fetch("""
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY sequence_in_doc NULLS LAST, created_at) as position
            FROM chunks
            WHERE document_id = $1
            ORDER BY sequence_in_doc NULLS LAST, created_at
        """, doc_id)

        # Distribute chunks proportionally across pages
        chunks_per_page = max(1, chunk_count / total_pages)

        for chunk in chunks:
            estimated_page = min(
                int((chunk['position'] - 1) / chunks_per_page) + 1,
                total_pages
            )

            await db.execute("""
                UPDATE chunks SET page_number = $1 WHERE id = $2
            """, estimated_page, chunk['id'])
            total_updated += 1

        title = doc['title'][:40] if doc['title'] else str(doc_id)[:8]
        print(f"  {title}: {chunk_count} chunks -> pages 1-{total_pages}")

    # Also ensure images have page numbers (default to 1 if NULL)
    img_updated = await db.fetchval("""
        UPDATE images
        SET page_number = COALESCE(page_number, 1)
        WHERE page_number IS NULL
        RETURNING COUNT(*)
    """) or 0

    # Verification
    null_chunks = await db.fetchval("SELECT COUNT(*) FROM chunks WHERE page_number IS NULL")
    null_images = await db.fetchval("SELECT COUNT(*) FROM images WHERE page_number IS NULL")

    # Show page distribution for one document
    sample = await db.fetch("""
        SELECT page_number, COUNT(*) as count
        FROM chunks
        WHERE document_id = (SELECT id FROM documents LIMIT 1)
        GROUP BY page_number
        ORDER BY page_number
        LIMIT 10
    """)

    await db.close()

    print("\n" + "=" * 60)
    print("PAGE NUMBER BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Chunks updated: {total_updated}")
    print(f"  Images updated: {img_updated}")
    print(f"  Chunks still NULL: {null_chunks}")
    print(f"  Images still NULL: {null_images}")

    if sample:
        print("\n  Sample distribution (first doc):")
        for row in sample:
            print(f"    Page {row['page_number']}: {row['count']} chunks")


if __name__ == "__main__":
    asyncio.run(backfill_page_numbers())
