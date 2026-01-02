#!/usr/bin/env python3
"""
Backfill VLM captions and caption embeddings for existing images.

Usage:
    ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... DATABASE_URL=... python scripts/backfill_vlm_captions.py

This script:
1. Finds images without VLM captions
2. Generates captions using Claude Vision
3. Embeds captions using Voyage-3
4. Updates the database
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backfill_vlm")


async def backfill_vlm_captions():
    """Backfill VLM captions and caption embeddings for images."""

    # Check required environment variables
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    if not anthropic_key:
        logger.error("ANTHROPIC_API_KEY not set - required for VLM captions")
        sys.exit(1)

    if not voyage_key:
        logger.warning("VOYAGE_API_KEY not set - caption embeddings will be skipped")

    from src.database.connection import DatabaseConnection

    logger.info("Connecting to database...")
    db = await DatabaseConnection.initialize(db_url)

    # Find images needing captions
    images_need_caption = await db.fetch("""
        SELECT id, file_path, storage_path, width, height, is_decorative
        FROM images
        WHERE vlm_caption IS NULL
          AND (is_decorative IS NULL OR NOT is_decorative)
    """)

    # Find images needing caption embeddings (have caption but no embedding)
    images_need_embedding = await db.fetch("""
        SELECT id, vlm_caption
        FROM images
        WHERE vlm_caption IS NOT NULL
          AND caption_embedding IS NULL
    """)

    logger.info(f"Images needing VLM captions: {len(images_need_caption)}")
    logger.info(f"Images needing caption embeddings: {len(images_need_embedding)}")

    if not images_need_caption and not images_need_embedding:
        logger.info("All images already have captions and embeddings!")
        await db.close()
        return

    # Phase 1: Generate VLM Captions
    captioned_count = 0
    caption_errors = 0

    if images_need_caption:
        logger.info("=" * 50)
        logger.info("PHASE 1: VLM Caption Generation")
        logger.info("=" * 50)

        from src.retrieval.vlm_captioner import VLMImageCaptioner, ImageInput, ImageType

        captioner = VLMImageCaptioner(
            api_key=anthropic_key,
            model="claude-sonnet-4-20250514",
            rate_limit_delay=0.5  # Be gentle with API
        )

        for i, img in enumerate(images_need_caption, 1):
            img_id = img['id']

            # Determine file path (storage_path or file_path)
            file_path_str = img.get('storage_path') or img.get('file_path')
            if not file_path_str:
                logger.warning(f"  [{i}/{len(images_need_caption)}] No file path for {img_id}")
                caption_errors += 1
                continue

            file_path = Path(file_path_str)

            # Check if file exists - try multiple locations
            if not file_path.exists():
                # Try as absolute path from project root
                project_root = Path(__file__).parent.parent
                candidates = [
                    project_root / file_path_str,  # output/images/...
                    Path("data/images") / file_path.name,
                    project_root / "data" / "images" / file_path.name,
                ]

                found = False
                for candidate in candidates:
                    if candidate.exists():
                        file_path = candidate
                        found = True
                        break

                if not found:
                    logger.warning(f"  [{i}/{len(images_need_caption)}] File not found: {file_path_str}")
                    caption_errors += 1
                    continue

            logger.info(f"  [{i}/{len(images_need_caption)}] Captioning: {file_path.name}")

            try:
                # Create ImageInput
                image_input = ImageInput(
                    id=str(img_id),
                    file_path=file_path,
                    width=img.get('width') or 0,
                    height=img.get('height') or 0,
                    image_type=ImageType.UNKNOWN,
                    is_decorative=img.get('is_decorative') or False
                )

                # Generate caption
                result = await captioner.caption_image(image_input, classify_type=True)

                if result.success:
                    # Update database
                    await db.execute("""
                        UPDATE images
                        SET vlm_caption = $1, image_type = $2
                        WHERE id = $3
                    """, result.caption, result.image_type.value, img_id)

                    captioned_count += 1
                    logger.info(f"      Caption: {result.caption[:80]}...")
                else:
                    logger.warning(f"      Failed: {result.error}")
                    caption_errors += 1

            except Exception as e:
                logger.error(f"      Error: {e}")
                caption_errors += 1

            # Progress checkpoint
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(images_need_caption)} ({captioned_count} success, {caption_errors} errors)")

        logger.info(f"Caption generation complete: {captioned_count} success, {caption_errors} errors")

    # Phase 2: Generate Caption Embeddings
    embedded_count = 0
    embed_errors = 0

    if voyage_key:
        # Re-query to include newly captioned images
        images_for_embedding = await db.fetch("""
            SELECT id, vlm_caption
            FROM images
            WHERE vlm_caption IS NOT NULL
              AND caption_embedding IS NULL
        """)

        if images_for_embedding:
            logger.info("=" * 50)
            logger.info("PHASE 2: Caption Embedding Generation")
            logger.info("=" * 50)

            from src.ingest.embeddings import VoyageTextEmbedder

            embedder = VoyageTextEmbedder(api_key=voyage_key)

            # Process in batches
            batch_size = 20

            for i in range(0, len(images_for_embedding), batch_size):
                batch = images_for_embedding[i:i+batch_size]
                captions = [img['vlm_caption'] for img in batch]

                try:
                    logger.info(f"  Embedding batch {i//batch_size + 1}/{(len(images_for_embedding) + batch_size - 1)//batch_size}")

                    embeddings = await embedder.embed_batch(captions)

                    # Update database
                    for img, emb in zip(batch, embeddings):
                        # Convert to list for storage
                        if hasattr(emb, 'tolist'):
                            emb_list = emb.tolist()
                        else:
                            emb_list = list(emb)

                        await db.execute("""
                            UPDATE images
                            SET caption_embedding = $1::vector
                            WHERE id = $2
                        """, emb_list, img['id'])

                        embedded_count += 1

                except Exception as e:
                    logger.error(f"  Batch embedding error: {e}")
                    embed_errors += len(batch)

            logger.info(f"Caption embedding complete: {embedded_count} success, {embed_errors} errors")

    # Final verification
    stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE vlm_caption IS NOT NULL) as with_caption,
            COUNT(*) FILTER (WHERE caption_embedding IS NOT NULL) as with_embedding,
            COUNT(*) FILTER (WHERE is_decorative) as decorative
        FROM images
    """)

    await db.close()

    logger.info("=" * 50)
    logger.info("VLM BACKFILL COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Total images: {stats['total']}")
    logger.info(f"  With VLM caption: {stats['with_caption']}")
    logger.info(f"  With caption embedding: {stats['with_embedding']}")
    logger.info(f"  Decorative (skipped): {stats['decorative']}")
    logger.info(f"  This run: {captioned_count} captioned, {embedded_count} embedded")


if __name__ == "__main__":
    asyncio.run(backfill_vlm_captions())
