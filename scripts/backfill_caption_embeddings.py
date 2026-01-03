#!/usr/bin/env python3
"""
Backfill caption_embedding for images that have vlm_caption but no caption_embedding.
"""
import asyncio
import os
import logging
from typing import List
import asyncpg
import voyageai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")


async def get_images_missing_embeddings(conn: asyncpg.Connection) -> List[dict]:
    """Get images with vlm_caption but no caption_embedding."""
    rows = await conn.fetch("""
        SELECT id, vlm_caption
        FROM images
        WHERE vlm_caption IS NOT NULL
          AND vlm_caption != ''
          AND caption_embedding IS NULL
        ORDER BY id
    """)
    return [{"id": row["id"], "caption": row["vlm_caption"]} for row in rows]


async def update_caption_embedding(conn: asyncpg.Connection, image_id: str, embedding: List[float]):
    """Update caption_embedding for an image."""
    # Format as pgvector string
    embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    await conn.execute(
        "UPDATE images SET caption_embedding = $1 WHERE id = $2",
        embedding_str, image_id
    )


async def main():
    if not VOYAGE_API_KEY:
        logger.error("VOYAGE_API_KEY not set")
        return

    logger.info("Connecting to database...")
    conn = await asyncpg.connect(DATABASE_URL)

    # Get images needing embeddings
    images = await get_images_missing_embeddings(conn)
    logger.info(f"Found {len(images)} images missing caption_embedding")

    if not images:
        logger.info("Nothing to backfill")
        await conn.close()
        return

    # Initialize Voyage client
    voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Process in batches
    batch_size = 128
    total_updated = 0

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        captions = [img["caption"] for img in batch]

        logger.info(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} captions)...")

        try:
            result = voyage.embed(
                texts=captions,
                model="voyage-3",
                input_type="document"
            )

            embeddings = result.embeddings

            # Update each image
            for j, (img, emb) in enumerate(zip(batch, embeddings)):
                await update_caption_embedding(conn, img["id"], emb)
                total_updated += 1

            logger.info(f"Updated {len(batch)} images in this batch")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue

    await conn.close()
    logger.info(f"Done! Total updated: {total_updated}")


if __name__ == "__main__":
    asyncio.run(main())
