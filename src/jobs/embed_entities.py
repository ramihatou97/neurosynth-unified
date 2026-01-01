"""
Background job to populate entity embeddings.

Run after ingestion or as a scheduled task to ensure all entities
have embeddings for semantic similarity ranking in Graph-RAG.

Usage:
    python -m src.jobs.embed_entities

Options:
    --batch-size N    Batch size for embedding (default: 100)
    --force           Re-embed all entities, not just missing ones
"""

import asyncio
import argparse
import logging
import os
import sys
from typing import Callable, Awaitable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def populate_entity_embeddings(
    db_pool,
    embed_fn: Callable[[str], Awaitable[list[float]]],
    batch_size: int = 100,
    force: bool = False,
) -> dict:
    """
    Populate embeddings for all entities missing them.

    Args:
        db_pool: asyncpg connection pool
        embed_fn: Async function that embeds text -> vector
        batch_size: Number of entities to embed per batch
        force: If True, re-embed all entities

    Returns:
        Dict with statistics
    """
    stats = {
        "total_entities": 0,
        "entities_embedded": 0,
        "entities_skipped": 0,
        "errors": 0,
    }

    async with db_pool.acquire() as conn:
        # Get entities without embeddings (or all if force)
        if force:
            rows = await conn.fetch("""
                SELECT e.id, e.name
                FROM entities e
                ORDER BY e.name
            """)
        else:
            rows = await conn.fetch("""
                SELECT e.id, e.name
                FROM entities e
                LEFT JOIN entity_embeddings ee ON e.id = ee.entity_id
                WHERE ee.entity_id IS NULL
                ORDER BY e.name
            """)

        stats["total_entities"] = len(rows)
        logger.info(f"Found {len(rows)} entities to embed")

        if not rows:
            logger.info("No entities to embed")
            return stats

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            for row in batch:
                try:
                    embedding = await embed_fn(row["name"])

                    await conn.execute("""
                        INSERT INTO entity_embeddings (entity_id, embedding)
                        VALUES ($1, $2)
                        ON CONFLICT (entity_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            updated_at = CURRENT_TIMESTAMP
                    """, row["id"], embedding)

                    stats["entities_embedded"] += 1

                except Exception as e:
                    logger.error(f"Failed to embed entity '{row['name']}': {e}")
                    stats["errors"] += 1

            processed = min(i + batch_size, len(rows))
            logger.info(f"Processed {processed}/{len(rows)} entities")

    # Create/update vector index after population
    async with db_pool.acquire() as conn:
        try:
            # Check if we have enough data for ivfflat
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_embeddings"
            )

            if count >= 100:
                await conn.execute("""
                    DROP INDEX IF EXISTS idx_entity_emb;
                    CREATE INDEX idx_entity_emb
                    ON entity_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                logger.info(f"Created ivfflat index on {count} embeddings")
            else:
                logger.info(
                    f"Skipping ivfflat index: only {count} embeddings "
                    "(need 100+ for effective indexing)"
                )
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

    logger.info(
        f"Entity embedding complete: {stats['entities_embedded']} embedded, "
        f"{stats['errors']} errors"
    )

    return stats


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate entity embeddings for Graph-RAG"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all entities, not just missing ones"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: DATABASE_URL env var)"
    )
    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        logger.error(
            "Database URL required. Set DATABASE_URL env var or use --database-url"
        )
        sys.exit(1)

    try:
        import asyncpg

        # Create connection pool
        logger.info("Connecting to database...")
        pool = await asyncpg.create_pool(database_url)

        # Create embedding function
        async def embed_fn(text: str) -> list[float]:
            """Embed text using Voyage AI."""
            import voyageai

            client = voyageai.AsyncClient()
            result = await client.embed([text], model="voyage-3")
            return result.embeddings[0]

        # Run embedding job
        stats = await populate_entity_embeddings(
            db_pool=pool,
            embed_fn=embed_fn,
            batch_size=args.batch_size,
            force=args.force,
        )

        logger.info(f"Final stats: {stats}")

        await pool.close()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install asyncpg voyageai")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
