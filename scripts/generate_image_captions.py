#!/usr/bin/env python3
"""
NeuroSynth - Generate Captions for Existing Images
===================================================

Generates VLM captions and caption embeddings for images in the database
that don't have them yet. This bypasses the memory-intensive BiomedCLIP
image embedder and uses only API-based services.

Usage:
    cd <project-root>
    source venv/bin/activate
    python scripts/generate_image_captions.py
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def generate_captions(
    connection_string: str,
    batch_size: int = 5,
    rate_limit_delay: float = 0.5
) -> dict:
    """
    Generate VLM captions and caption embeddings for images.

    Args:
        connection_string: PostgreSQL connection string
        batch_size: Number of images to process before committing
        rate_limit_delay: Delay between API calls

    Returns:
        Dict with statistics
    """
    from src.database import init_database, close_database
    from src.retrieval.vlm_captioner import VLMImageCaptioner, ImageInput, ImageType
    from src.ingest.embeddings import VoyageTextEmbedder

    stats = {
        'total_images': 0,
        'images_without_caption': 0,
        'captions_generated': 0,
        'embeddings_generated': 0,
        'errors': 0
    }

    # Initialize database
    logger.info("Connecting to database...")
    db = await init_database(connection_string)

    # Initialize captioner and embedder
    logger.info("Initializing VLM captioner and text embedder...")
    captioner = VLMImageCaptioner(
        model="claude-sonnet-4-20250514",
        rate_limit_delay=rate_limit_delay
    )

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        logger.error("VOYAGE_API_KEY not set!")
        return stats

    text_embedder = VoyageTextEmbedder(
        model="voyage-3",
        api_key=voyage_api_key
    )

    try:
        # Get images without captions
        logger.info("Fetching images without captions...")
        rows = await db.fetch("""
            SELECT id, file_path, width, height, image_type, is_decorative
            FROM images
            WHERE vlm_caption IS NULL
              AND NOT is_decorative
            ORDER BY id
        """)

        stats['images_without_caption'] = len(rows)
        logger.info(f"Found {len(rows)} images without captions")

        if not rows:
            logger.info("All images already have captions!")
            return stats

        # Get total image count
        total_count = await db.fetchval("SELECT COUNT(*) FROM images")
        stats['total_images'] = total_count

        # Process images
        for i, row in enumerate(rows):
            image_id = str(row['id'])
            file_path = Path(row['file_path'])

            logger.info(f"Processing image {i+1}/{len(rows)}: {file_path.name}")

            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Image file not found: {file_path}")
                stats['errors'] += 1
                continue

            try:
                # Map image type
                image_type_str = row['image_type'] or 'unknown'
                type_map = {
                    'surgical_photo': ImageType.SURGICAL_PHOTO,
                    'anatomy_diagram': ImageType.ANATOMY_DIAGRAM,
                    'imaging_scan': ImageType.IMAGING_SCAN,
                    'flowchart': ImageType.FLOWCHART,
                    'illustration': ImageType.ILLUSTRATION,
                }
                image_type = type_map.get(image_type_str.lower(), ImageType.UNKNOWN)

                # Create ImageInput
                image_input = ImageInput(
                    id=image_id,
                    file_path=file_path,
                    width=row['width'] or 0,
                    height=row['height'] or 0,
                    image_type=image_type,
                    is_decorative=row['is_decorative'] or False
                )

                # Generate caption
                result = await captioner.caption_image(image_input, classify_type=True)

                if not result.success:
                    logger.warning(f"Failed to caption image {image_id}: {result.error}")
                    stats['errors'] += 1
                    continue

                caption = result.caption
                confidence = result.confidence
                stats['captions_generated'] += 1

                logger.info(f"  Caption: {caption[:100]}...")

                # Generate caption embedding
                caption_embedding = await text_embedder.embed(caption)
                stats['embeddings_generated'] += 1

                # Update database
                await db.execute("""
                    UPDATE images
                    SET vlm_caption = $1,
                        vlm_confidence = $2,
                        caption_embedding = $3,
                        image_type = $4
                    WHERE id = $5
                """,
                    caption,
                    confidence,
                    caption_embedding.tolist(),
                    result.image_type.value,
                    row['id']
                )

                logger.info(f"  ✓ Updated database")

                # Rate limiting
                if i < len(rows) - 1:
                    await asyncio.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                stats['errors'] += 1
                continue

        return stats

    finally:
        await close_database()


def print_summary(stats: dict):
    """Print generation summary."""
    print()
    print("=" * 60)
    print("CAPTION GENERATION COMPLETE")
    print("=" * 60)
    print()
    print("Statistics:")
    print(f"  Total images in DB:     {stats['total_images']}")
    print(f"  Images without caption: {stats['images_without_caption']}")
    print(f"  Captions generated:     {stats['captions_generated']}")
    print(f"  Embeddings generated:   {stats['embeddings_generated']}")
    print(f"  Errors:                 {stats['errors']}")
    print()

    if stats['captions_generated'] > 0:
        print("Next step: Run build_indexes.py to rebuild FAISS indexes")
        print("  python scripts/build_indexes.py")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM captions for images in database"
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
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--rate-limit", "-r",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)"
    )

    args = parser.parse_args()

    print("NeuroSynth Image Caption Generator")
    print("=" * 60)
    print(f"Database: {args.database.split('@')[-1]}")
    print()

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set!")
        return 1

    if not os.getenv("VOYAGE_API_KEY"):
        print("ERROR: VOYAGE_API_KEY not set!")
        return 1

    print("API Keys: ✓ ANTHROPIC_API_KEY, ✓ VOYAGE_API_KEY")
    print()

    try:
        stats = await generate_captions(
            connection_string=args.database,
            batch_size=args.batch_size,
            rate_limit_delay=args.rate_limit
        )

        print_summary(stats)
        return 0

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
