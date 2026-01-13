#!/usr/bin/env python3
"""
NeuroSynth - Backfill Summary Embeddings
=========================================

Generates embeddings for chunk summaries to enable multi-index retrieval.
This improves synthesis quality by allowing semantic search on both
content and summaries.

Usage:
    python scripts/backfill_summary_embeddings.py --all
    python scripts/backfill_summary_embeddings.py --document-id <uuid>
    python scripts/backfill_summary_embeddings.py --batch-size 100 --dry-run
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_chunks_needing_summary_embedding(
    conn,
    document_id: Optional[UUID] = None,
    limit: int = 500
) -> List[Dict]:
    """Find chunks with summary but no summary_embedding."""

    query = """
        SELECT id, document_id, summary
        FROM chunks
        WHERE summary IS NOT NULL
          AND length(summary) > 20
          AND summary_embedding IS NULL
    """

    params = []
    if document_id:
        query += " AND document_id = $1"
        params.append(document_id)

    query += f" ORDER BY document_id, chunk_index LIMIT ${len(params) + 1}"
    params.append(limit)

    rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def generate_embeddings_batch(texts: List[str], client) -> List[List[float]]:
    """Generate embeddings for a batch of texts using Voyage."""
    try:
        response = await asyncio.to_thread(
            client.embed,
            texts=texts,
            model="voyage-3",
            input_type="document"
        )
        return response.embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return [None] * len(texts)


async def update_summary_embedding(conn, chunk_id: UUID, embedding: List[float]) -> bool:
    """Update chunk with summary embedding."""
    try:
        # Convert list to PostgreSQL vector string format
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        await conn.execute(
            """
            UPDATE chunks
            SET summary_embedding = $1::vector
            WHERE id = $2
            """,
            embedding_str, chunk_id
        )
        return True
    except Exception as e:
        logger.error(f"Failed to update chunk {chunk_id}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Backfill summary embeddings")
    parser.add_argument("--document-id", type=str, help="Specific document UUID")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    parser.add_argument("--limit", type=int, default=1000, help="Max chunks to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if not args.document_id and not args.all:
        parser.error("Must specify --document-id or --all")

    import asyncpg
    from dotenv import load_dotenv

    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")

    # Initialize Voyage client
    try:
        import voyageai
        voyage_client = voyageai.Client()
    except Exception as e:
        logger.error(f"Failed to initialize Voyage client: {e}")
        logger.error("Set VOYAGE_API_KEY environment variable")
        return

    conn = await asyncpg.connect(database_url.replace("+asyncpg", ""))

    try:
        doc_id = UUID(args.document_id) if args.document_id else None
        chunks = await get_chunks_needing_summary_embedding(conn, doc_id, args.limit)

        print(f"\nFound {len(chunks)} chunks needing summary embeddings")

        if args.dry_run:
            print("\n[DRY RUN] Would process:")
            for chunk in chunks[:10]:
                print(f"  {chunk['id']}: {chunk['summary'][:60]}...")
            if len(chunks) > 10:
                print(f"  ... and {len(chunks) - 10} more")
            return

        # Process in batches
        success = 0
        failed = 0

        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i:i + args.batch_size]
            summaries = [c['summary'] for c in batch]

            print(f"\nProcessing batch {i // args.batch_size + 1} ({len(batch)} chunks)...")

            embeddings = await generate_embeddings_batch(summaries, voyage_client)

            for chunk, embedding in zip(batch, embeddings):
                if embedding:
                    if await update_summary_embedding(conn, chunk['id'], embedding):
                        success += 1
                    else:
                        failed += 1
                else:
                    failed += 1

            # Rate limiting
            await asyncio.sleep(0.5)

        print(f"\n{'='*50}")
        print(f"Results: {success} succeeded, {failed} failed")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
