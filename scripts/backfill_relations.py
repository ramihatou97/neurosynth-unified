"""
Backfill Relations Script

Reprocesses existing chunks with the enhanced relation extractor.
Safely handles the data consistency issue by deleting before inserting.

Usage:
    python scripts/backfill_relations.py --db-url "postgresql://..." --dry-run
    python scripts/backfill_relations.py --db-url "postgresql://..." --batch-size 100
    python scripts/backfill_relations.py --db-url "postgresql://..." --document-id <uuid>
"""

import asyncio
import argparse
import logging
from uuid import UUID
from typing import Optional

import asyncpg

from src.core.relation_config import RelationExtractionConfig
from src.ingest.relation_pipeline import RelationExtractionPipeline

logger = logging.getLogger(__name__)


async def get_chunks_to_reprocess(
    conn: asyncpg.Connection,
    document_id: Optional[UUID] = None,
    batch_size: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Get chunks that need reprocessing (old extraction_method)."""

    if document_id:
        return await conn.fetch(
            """
            SELECT DISTINCT c.id, c.content, c.document_id
            FROM chunks c
            LEFT JOIN entity_relations er ON c.id = ANY(er.chunk_ids)
            WHERE c.document_id = $1
            AND (er.extraction_method IS NULL OR er.extraction_method = '')
            ORDER BY c.id
            LIMIT $2 OFFSET $3
            """,
            document_id, batch_size, offset
        )
    else:
        return await conn.fetch(
            """
            SELECT DISTINCT c.id, c.content, c.document_id
            FROM chunks c
            LEFT JOIN entity_relations er ON c.id = ANY(er.chunk_ids)
            WHERE er.extraction_method IS NULL OR er.extraction_method = ''
            ORDER BY c.id
            LIMIT $1 OFFSET $2
            """,
            batch_size, offset
        )


async def cleanup_chunk_relations(conn: asyncpg.Connection, chunk_id: UUID) -> int:
    """Delete existing relations for a chunk before reprocessing."""
    result = await conn.execute(
        """
        DELETE FROM entity_relations
        WHERE $1 = ANY(chunk_ids)
        """,
        chunk_id
    )
    return int(result.split()[-1]) if result else 0


async def backfill_relations(
    db_url: str,
    document_id: Optional[UUID] = None,
    batch_size: int = 100,
    dry_run: bool = False,
):
    """Main backfill function."""

    pool = await asyncpg.create_pool(db_url)

    config = RelationExtractionConfig(
        enable_coordination=True,
        enable_negation=True,
        enable_entity_first_ner=True,
        enable_tiered_llm=False,
        enable_coreference=False,  # Set True if you have RAM
    )

    pipeline = RelationExtractionPipeline(
        db_pool=pool,
        config=config,
    )

    offset = 0
    total_processed = 0
    total_relations = 0
    total_deleted = 0

    async with pool.acquire() as conn:
        while True:
            chunks = await get_chunks_to_reprocess(
                conn, document_id, batch_size, offset
            )

            if not chunks:
                break

            logger.info(f"Processing batch of {len(chunks)} chunks (offset={offset})")

            for chunk in chunks:
                chunk_id = chunk['id']
                chunk_text = chunk['content']

                if dry_run:
                    logger.info(f"[DRY RUN] Would process chunk {chunk_id}")
                    total_processed += 1
                    continue

                # CRITICAL: Delete before insert to avoid duplicates
                deleted = await cleanup_chunk_relations(conn, chunk_id)
                total_deleted += deleted

                # Extract with new pipeline
                relations = await pipeline.process_chunk(chunk_id, chunk_text)
                total_relations += len(relations)
                total_processed += 1

                if total_processed % 50 == 0:
                    logger.info(
                        f"Progress: {total_processed} chunks, "
                        f"{total_relations} relations, "
                        f"{total_deleted} old relations deleted"
                    )

            # Flush batch to DB
            if not dry_run:
                await pipeline.flush()

            offset += batch_size

    logger.info(f"Backfill complete:")
    logger.info(f"  Chunks processed: {total_processed}")
    logger.info(f"  Relations created: {total_relations}")
    logger.info(f"  Old relations deleted: {total_deleted}")

    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill relations with enhanced extractor")
    parser.add_argument("--db-url", required=True, help="Database connection URL")
    parser.add_argument("--document-id", type=str, help="Specific document UUID to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Chunks per batch")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(backfill_relations(
        db_url=args.db_url,
        document_id=UUID(args.document_id) if args.document_id else None,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))
