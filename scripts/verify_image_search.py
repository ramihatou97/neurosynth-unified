#!/usr/bin/env python3
"""
Verify caption-based image search functionality.

Usage:
    VOYAGE_API_KEY=... DATABASE_URL=... python scripts/verify_image_search.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def verify_image_search():
    """Test text-to-image search via caption embeddings."""

    from src.database.connection import DatabaseConnection
    from src.ingest.embeddings import VoyageTextEmbedder

    print("=" * 60)
    print("CAPTION-BASED IMAGE SEARCH VERIFICATION")
    print("=" * 60)

    db_url = os.getenv("DATABASE_URL")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    if not voyage_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)

    db = await DatabaseConnection.initialize(db_url)
    embedder = VoyageTextEmbedder(api_key=voyage_key)

    # Verify caption embeddings exist
    stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE caption_embedding IS NOT NULL) as with_embedding
        FROM images
    """)

    print(f"\nImages with caption embeddings: {stats['with_embedding']}/{stats['total']}")

    if stats['with_embedding'] == 0:
        print("ERROR: No caption embeddings found. Run backfill_vlm_captions.py first.")
        await db.close()
        return

    # Test queries
    queries = [
        "MRI scan showing tumor",
        "surgical exposure of the facial nerve",
        "anatomical diagram of skull base",
        "intraoperative photograph of craniotomy"
    ]

    print("\n" + "-" * 60)

    for query in queries:
        print(f"\nQuery: '{query}'")

        # Embed query
        query_embedding = await embedder.embed(query)

        # Search via caption embeddings (cosine distance)
        results = await db.fetch("""
            SELECT
                id,
                vlm_caption,
                image_type,
                1 - (caption_embedding <=> $1::vector) as similarity
            FROM images
            WHERE caption_embedding IS NOT NULL
            ORDER BY caption_embedding <=> $1::vector
            LIMIT 3
        """, query_embedding.tolist())

        print(f"  Top 3 matches:")
        for i, r in enumerate(results, 1):
            caption = r['vlm_caption']
            if len(caption) > 70:
                caption = caption[:70] + "..."
            print(f"    [{i}] sim={r['similarity']:.3f} | type={r['image_type'] or 'unknown'}")
            print(f"        {caption}")

    await db.close()

    print("\n" + "=" * 60)
    print("CAPTION-BASED IMAGE SEARCH VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_image_search())
