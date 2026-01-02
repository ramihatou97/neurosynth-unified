#!/usr/bin/env python3
"""
Backfill chunk-image links using TriPassLinker.

Usage:
    DATABASE_URL=... python scripts/backfill_links.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import UUID, uuid4
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseConnection
from src.ingest.fusion import TriPassLinker
from src.shared.models import SemanticChunk, ExtractedImage, ChunkType, LinkResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backfill_links")


async def backfill_links():
    """Backfill chunk-image links using TriPassLinker strategy."""

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    db = await DatabaseConnection.initialize(db_url)
    linker = TriPassLinker()

    # Find documents with both chunks and non-decorative images
    docs = await db.fetch("""
        SELECT DISTINCT d.id, d.title
        FROM documents d
        WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
          AND EXISTS (SELECT 1 FROM images i WHERE i.document_id = d.id AND (i.is_decorative IS NULL OR NOT i.is_decorative))
    """)

    logger.info(f"Found {len(docs)} documents eligible for linking")

    if not docs:
        logger.info("No documents to process.")
        return

    total_links_created = 0

    for doc in docs:
        doc_id = doc['id']
        doc_title = doc['title'] or str(doc_id)[:8]

        logger.info(f"Processing: {doc_title}")

        # Fetch chunks with embeddings
        chunk_rows = await db.fetch("""
            SELECT id, content, page_number, chunk_type, cuis, embedding as text_embedding
            FROM chunks
            WHERE document_id = $1 AND embedding IS NOT NULL
        """, doc_id)

        # Fetch non-decorative images with embeddings
        # Note: images table uses clip_embedding, not embedding, and has no cuis column
        image_rows = await db.fetch("""
            SELECT id, vlm_caption, page_number, clip_embedding as embedding, caption_embedding
            FROM images
            WHERE document_id = $1
              AND (is_decorative IS NULL OR NOT is_decorative)
        """, doc_id)

        if not chunk_rows or not image_rows:
            logger.info(f"  Skipping: {len(chunk_rows)} chunks, {len(image_rows)} images")
            continue

        logger.info(f"  Found {len(chunk_rows)} chunks, {len(image_rows)} images")

        # Convert to model objects
        # SemanticChunk required: id, document_id, content, title, section_path, page_start, page_end, chunk_type
        chunks = []
        for row in chunk_rows:
            try:
                chunk_type = ChunkType(row['chunk_type']) if row['chunk_type'] else ChunkType.GENERAL
            except ValueError:
                chunk_type = ChunkType.GENERAL

            chunks.append(SemanticChunk(
                id=str(row['id']),
                document_id=str(doc_id),
                content=row['content'],
                title="",  # Required but can be empty
                section_path=[],  # Required but can be empty
                page_start=row['page_number'] or 0,
                page_end=row['page_number'] or 0,
                chunk_type=chunk_type,
                cuis=row['cuis'] or [],
                text_embedding=row['text_embedding']
            ))

        images = []
        for row in image_rows:
            images.append(ExtractedImage(
                id=str(row['id']),
                document_id=str(doc_id),
                file_path=Path("placeholder"),
                page_number=row['page_number'] or 0,
                width=0,
                height=0,
                format="unknown",
                content_hash="",
                caption="",
                vlm_caption=row['vlm_caption'] or "",
                image_type="unknown",
                quality_score=0.0,
                embedding=row['embedding'],
                caption_embedding=row['caption_embedding'],
                cuis=[]  # Images don't have CUIs in DB schema
            ))

        # Run TriPassLinker - returns TUPLE (chunks, images, links), NOT async
        try:
            _, _, links = linker.link(chunks, images)
        except Exception as e:
            logger.error(f"  Linking failed: {e}")
            continue

        logger.info(f"  Generated {len(links)} potential links")

        # Insert high-quality links
        inserted = 0
        for link in links:
            # LinkResult is a dataclass with 'strength' attribute (not 'score')
            if link.strength < 0.5:
                continue

            try:
                # Note: INSERT into chunk_image_links table (not the 'links' view)
                # The view aliases relevance_score -> score
                await db.execute("""
                    INSERT INTO chunk_image_links (
                        id, chunk_id, image_id, link_type, relevance_score, link_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (chunk_id, image_id)
                    DO UPDATE SET
                        relevance_score = EXCLUDED.relevance_score,
                        link_type = EXCLUDED.link_type
                """,
                    uuid4(),
                    UUID(link.chunk_id),
                    UUID(link.image_id),
                    link.match_type,
                    link.strength,  # 'strength' maps to relevance_score
                    {
                        'proximity_score': link.details.get('proximity_score'),
                        'semantic_score': link.details.get('semantic_score'),
                        'cui_overlap_score': link.details.get('cui_overlap_score')
                    }
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"  Failed to insert link: {e}")

        total_links_created += inserted
        logger.info(f"  Inserted {inserted} links")

    # Create/refresh materialized view
    logger.info("Creating/refreshing materialized view...")

    # Check if view exists
    view_exists = await db.fetchval("""
        SELECT EXISTS (
            SELECT FROM pg_matviews WHERE matviewname = 'top_chunk_links'
        )
    """)

    if not view_exists:
        await db.execute("""
            CREATE MATERIALIZED VIEW top_chunk_links AS
            SELECT
                chunk_id,
                array_agg(image_id ORDER BY score DESC) AS top_image_ids,
                array_agg(score ORDER BY score DESC) AS top_scores
            FROM (
                SELECT chunk_id, image_id, score,
                       ROW_NUMBER() OVER (PARTITION BY chunk_id ORDER BY score DESC) as rn
                FROM links
                WHERE score >= 0.5
            ) ranked
            WHERE rn <= 3
            GROUP BY chunk_id
        """)
        logger.info("  Created materialized view")
    else:
        await db.execute("REFRESH MATERIALIZED VIEW top_chunk_links")
        logger.info("  Refreshed materialized view")

    # Final stats
    link_count = await db.fetchval("SELECT COUNT(*) FROM links")
    view_count = await db.fetchval("SELECT COUNT(*) FROM top_chunk_links")

    await db.close()

    logger.info("=" * 50)
    logger.info("LINK BACKFILL COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Total links in table: {link_count}")
    logger.info(f"  Chunks with images: {view_count}")
    logger.info(f"  Links created this run: {total_links_created}")


if __name__ == "__main__":
    asyncio.run(backfill_links())
