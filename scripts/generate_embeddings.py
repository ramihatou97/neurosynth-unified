#!/usr/bin/env python
"""
Generate embeddings for existing chunks in database.

This script:
1. Fetches chunks without embeddings from database
2. Generates embeddings using Voyage AI API
3. Updates chunks with embeddings
4. Tracks progress and handles errors
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import asyncpg
from voyageai import AsyncClient as VoyageAsyncClient

# Configuration
BATCH_SIZE = 100  # Process 100 chunks at a time
VOYAGE_MODEL = "voyage-3"  # 1024 dimensions

async def fetch_chunks_without_embeddings(pool) -> List[Dict[str, Any]]:
    """Fetch all chunks that don't have embeddings yet."""
    query = """
        SELECT id, content
        FROM chunks
        WHERE embedding IS NULL
        ORDER BY id
    """

    rows = await pool.fetch(query)
    return [dict(row) for row in rows]

async def generate_embeddings_batch(
    client: VoyageAsyncClient,
    texts: List[str]
) -> List[List[float]]:
    """Generate embeddings for a batch of texts using Voyage AI."""
    try:
        result = await client.embed(
            texts=texts,
            model=VOYAGE_MODEL,
            input_type="document"  # For indexing/storage
        )
        return result.embeddings
    except Exception as e:
        print(f"   ⚠️  Error generating embeddings: {e}")
        raise

async def update_chunk_embeddings(
    pool,
    chunk_id: str,
    embedding: List[float]
) -> None:
    """Update a chunk with its embedding."""
    # Convert list to PostgreSQL vector format: '[0.1, 0.2, ...]'
    embedding_str = '[' + ','.join(map(str, embedding)) + ']'

    query = """
        UPDATE chunks
        SET embedding = $1::vector(1024)
        WHERE id = $2
    """

    await pool.execute(query, embedding_str, chunk_id)

async def main():
    print("="*70)
    print("GENERATE EMBEDDINGS FOR EXISTING CHUNKS")
    print("="*70)

    # Check configuration
    database_url = os.getenv("DATABASE_URL")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not database_url:
        print("❌ DATABASE_URL not configured in .env")
        return 1

    if not voyage_key:
        print("❌ VOYAGE_API_KEY not configured in .env")
        return 1

    print(f"\n✅ Configuration:")
    print(f"   Database: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
    print(f"   Voyage API: {voyage_key[:10]}...{voyage_key[-4:]}")
    print(f"   Model: {VOYAGE_MODEL}")
    print(f"   Batch size: {BATCH_SIZE}")

    # Connect to database
    print(f"\n[1] Connecting to database...")
    # Remove +asyncpg suffix for asyncpg.connect
    db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    try:
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
        print(f"✅ Connected to database")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return 1

    # Fetch chunks
    print(f"\n[2] Fetching chunks without embeddings...")
    chunks = await fetch_chunks_without_embeddings(pool)

    if not chunks:
        print(f"✅ All chunks already have embeddings!")
        await pool.close()
        return 0

    print(f"✅ Found {len(chunks)} chunks without embeddings")

    # Initialize Voyage client
    print(f"\n[3] Initializing Voyage AI client...")
    voyage_client = VoyageAsyncClient(api_key=voyage_key)
    print(f"✅ Voyage client ready")

    # Generate embeddings in batches
    print(f"\n[4] Generating embeddings (this may take 1-2 minutes)...")
    print(f"    Estimated cost: ${len(chunks) * 0.00013:.4f} (Voyage-3 pricing)")
    print()

    total_processed = 0
    total_errors = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        try:
            # Extract text content
            texts = [chunk['content'] for chunk in batch]

            # Generate embeddings
            embeddings = await generate_embeddings_batch(voyage_client, texts)

            # Update database
            for chunk, embedding in zip(batch, embeddings):
                await update_chunk_embeddings(pool, chunk['id'], embedding)
                total_processed += 1

            print(f"✅ ({total_processed}/{len(chunks)})")

        except Exception as e:
            print(f"❌ Error")
            print(f"      {str(e)}")
            total_errors += len(batch)
            continue

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.1)

    # Summary
    print(f"\n[5] Summary:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Processed: {total_processed}")
    print(f"   Errors: {total_errors}")
    print(f"   Success rate: {total_processed/len(chunks)*100:.1f}%")

    # Verify
    print(f"\n[6] Verifying embeddings in database...")
    verify_query = """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings
        FROM chunks
    """
    result = await pool.fetchrow(verify_query)

    print(f"✅ Verification:")
    print(f"   Total chunks: {result['total']}")
    print(f"   With embeddings: {result['with_embeddings']}")
    print(f"   Coverage: {result['with_embeddings']/result['total']*100:.1f}%")

    await pool.close()

    if result['with_embeddings'] == result['total']:
        print(f"\n" + "="*70)
        print("✅ ALL CHUNKS NOW HAVE EMBEDDINGS")
        print("="*70)
        print("\nNext steps:")
        print("  1. Build FAISS indexes: python scripts/build_faiss_indexes.py")
        print("  2. Test search: python tests/manual/stage12_search.py")
        print("  3. Test synthesis: python tests/manual/stage14_synthesis.py")
        return 0
    else:
        print(f"\n⚠️  Some chunks still missing embeddings")
        return 1

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
