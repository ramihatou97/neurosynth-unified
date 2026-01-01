#!/usr/bin/env python3
"""
NeuroSynth - Fix Image Pipeline
================================

Fixes the image pipeline by:
1. Scanning all image folders on disk
2. Generating VLM captions for images that don't have them
3. Creating caption embeddings via Voyage
4. Properly updating database records with file paths and embeddings

Usage:
    cd <project-root>
    source venv/bin/activate
    python scripts/fix_image_pipeline.py
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add parent to path (portable)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about an image on disk."""
    file_path: Path
    file_name: str
    content_hash: str
    document_id: str
    page: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    vlm_caption: Optional[str] = None
    vlm_type: Optional[str] = None


def scan_image_folders(images_dir: Path) -> Dict[str, List[ImageInfo]]:
    """Scan image folders and gather all image info."""
    results = {}

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return results

    for doc_folder in images_dir.iterdir():
        if not doc_folder.is_dir():
            continue

        doc_id = doc_folder.name
        images = []

        # Check for vlm_captions.json
        captions_file = doc_folder / "vlm_captions.json"
        captions_map = {}

        if captions_file.exists():
            try:
                with open(captions_file) as f:
                    captions_data = json.load(f)
                    for item in captions_data:
                        captions_map[item.get("image_file", "")] = item
            except Exception as e:
                logger.warning(f"Error reading captions file {captions_file}: {e}")

        # Scan images
        for img_file in doc_folder.glob("*.png"):
            file_name = img_file.name
            content_hash = img_file.stem  # Filename without extension is the hash

            # Get caption info if available
            caption_info = captions_map.get(file_name, {})

            # Parse size
            size_str = caption_info.get("size", "")
            width, height = None, None
            if "x" in size_str:
                try:
                    width, height = map(int, size_str.split("x"))
                except:
                    pass

            images.append(ImageInfo(
                file_path=img_file,
                file_name=file_name,
                content_hash=content_hash,
                document_id=doc_id,
                page=caption_info.get("page"),
                width=width,
                height=height,
                vlm_caption=caption_info.get("vlm_caption"),
                vlm_type=caption_info.get("vlm_type")
            ))

        if images:
            results[doc_id] = images
            logger.info(f"Found {len(images)} images in {doc_id}")

    return results


async def generate_missing_captions(
    images: List[ImageInfo],
    rate_limit_delay: float = 0.5
) -> int:
    """Generate VLM captions for images that don't have them."""
    from src.retrieval.vlm_captioner import VLMImageCaptioner, ImageInput, ImageType

    captioner = VLMImageCaptioner(
        model="claude-sonnet-4-20250514",
        rate_limit_delay=rate_limit_delay
    )

    needs_caption = [img for img in images if not img.vlm_caption]
    logger.info(f"Generating captions for {len(needs_caption)} images...")

    generated = 0
    for i, img in enumerate(needs_caption):
        logger.info(f"  [{i+1}/{len(needs_caption)}] {img.file_name}")

        try:
            image_input = ImageInput(
                id=img.content_hash,
                file_path=img.file_path,
                width=img.width or 0,
                height=img.height or 0,
                image_type=ImageType.UNKNOWN,
                is_decorative=False
            )

            result = await captioner.caption_image(image_input, classify_type=True)

            if result.success:
                img.vlm_caption = result.caption
                img.vlm_type = result.image_type.value
                generated += 1
                logger.info(f"    ✓ Caption: {result.caption[:80]}...")
            else:
                logger.warning(f"    ✗ Failed: {result.error}")

            # Rate limiting
            if i < len(needs_caption) - 1:
                await asyncio.sleep(rate_limit_delay)

        except Exception as e:
            logger.error(f"    ✗ Error: {e}")

    return generated


async def generate_caption_embeddings(
    images: List[ImageInfo]
) -> Dict[str, list]:
    """Generate caption embeddings for images with captions."""
    from src.ingest.embeddings import VoyageTextEmbedder

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        logger.error("VOYAGE_API_KEY not set!")
        return {}

    embedder = VoyageTextEmbedder(model="voyage-3", api_key=voyage_api_key)

    has_caption = [img for img in images if img.vlm_caption]
    logger.info(f"Generating embeddings for {len(has_caption)} captioned images...")

    embeddings = {}

    # Batch for efficiency
    captions = [img.vlm_caption for img in has_caption]

    try:
        results = await embedder.embed_batch(captions)

        for img, emb in zip(has_caption, results):
            embeddings[img.content_hash] = emb.tolist()

        logger.info(f"  ✓ Generated {len(embeddings)} embeddings")

    except Exception as e:
        logger.error(f"  ✗ Embedding error: {e}")
        # Fall back to individual embedding
        for img in has_caption:
            try:
                emb = await embedder.embed(img.vlm_caption)
                embeddings[img.content_hash] = emb.tolist()
            except Exception as e2:
                logger.warning(f"  ✗ Failed for {img.content_hash}: {e2}")

    return embeddings


async def update_database(
    images: List[ImageInfo],
    embeddings: Dict[str, list],
    connection_string: str
) -> int:
    """Update database with image info and embeddings."""
    from src.database import init_database, close_database

    logger.info("Connecting to database...")
    db = await init_database(connection_string)

    updated = 0
    created = 0

    try:
        for img in images:
            if not img.vlm_caption:
                continue

            caption_embedding = embeddings.get(img.content_hash)

            # Check if image record exists (by content_hash or file_path)
            existing = await db.fetchrow("""
                SELECT id FROM images
                WHERE content_hash = $1 OR file_path = $2
            """, img.content_hash, str(img.file_path))

            if existing:
                # Update existing record
                await db.execute("""
                    UPDATE images SET
                        file_path = $1,
                        file_name = $2,
                        content_hash = $3,
                        page_number = $4,
                        width = $5,
                        height = $6,
                        vlm_caption = $7,
                        vlm_confidence = 0.9,
                        image_type = $8,
                        caption_embedding = $9
                    WHERE id = $10
                """,
                    str(img.file_path),
                    img.file_name,
                    img.content_hash,
                    img.page,
                    img.width,
                    img.height,
                    img.vlm_caption,
                    img.vlm_type,
                    caption_embedding,
                    existing['id']
                )
                updated += 1
            else:
                # Check if document exists
                doc_exists = await db.fetchval("""
                    SELECT id FROM documents WHERE id = $1::uuid
                """, img.document_id)

                if doc_exists:
                    # Create new image record
                    await db.execute("""
                        INSERT INTO images (
                            document_id, file_path, file_name, content_hash,
                            page_number, width, height, vlm_caption, vlm_confidence,
                            image_type, caption_embedding, is_decorative
                        ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, 0.9, $9, $10, false)
                    """,
                        img.document_id,
                        str(img.file_path),
                        img.file_name,
                        img.content_hash,
                        img.page,
                        img.width,
                        img.height,
                        img.vlm_caption,
                        img.vlm_type,
                        caption_embedding
                    )
                    created += 1
                else:
                    logger.warning(f"Document {img.document_id} not found for image {img.file_name}")

        logger.info(f"Database: {updated} updated, {created} created")
        return updated + created

    finally:
        await close_database()


def save_captions_json(images: List[ImageInfo], doc_folder: Path):
    """Save updated captions back to JSON file."""
    captions_file = doc_folder / "vlm_captions.json"

    data = []
    for img in images:
        data.append({
            "image_file": img.file_name,
            "page": img.page,
            "size": f"{img.width}x{img.height}" if img.width and img.height else "",
            "vlm_type": img.vlm_type,
            "vlm_caption": img.vlm_caption,
            "pdf_caption": None,
            "figure_id": None
        })

    with open(captions_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved captions to {captions_file}")


async def fix_pipeline(
    images_dir: Path,
    connection_string: str,
    rate_limit_delay: float = 0.5,
    skip_caption_gen: bool = False
) -> dict:
    """Fix the entire image pipeline."""
    stats = {
        'folders_scanned': 0,
        'total_images': 0,
        'captions_generated': 0,
        'embeddings_generated': 0,
        'db_records_updated': 0
    }

    # Step 1: Scan image folders
    logger.info("=" * 60)
    logger.info("STEP 1: Scanning image folders")
    logger.info("=" * 60)

    folder_images = scan_image_folders(images_dir)
    stats['folders_scanned'] = len(folder_images)

    all_images = []
    for doc_id, images in folder_images.items():
        all_images.extend(images)
    stats['total_images'] = len(all_images)

    if not all_images:
        logger.warning("No images found!")
        return stats

    # Step 2: Generate missing captions
    if not skip_caption_gen:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Generating missing VLM captions")
        logger.info("=" * 60)

        stats['captions_generated'] = await generate_missing_captions(
            all_images, rate_limit_delay
        )

        # Save updated captions back to JSON files
        for doc_id, images in folder_images.items():
            doc_folder = images_dir / doc_id
            save_captions_json(images, doc_folder)

    # Step 3: Generate caption embeddings
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Generating caption embeddings")
    logger.info("=" * 60)

    embeddings = await generate_caption_embeddings(all_images)
    stats['embeddings_generated'] = len(embeddings)

    # Step 4: Update database
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Updating database")
    logger.info("=" * 60)

    stats['db_records_updated'] = await update_database(
        all_images, embeddings, connection_string
    )

    return stats


def print_summary(stats: dict):
    """Print fix summary."""
    print()
    print("=" * 60)
    print("IMAGE PIPELINE FIX COMPLETE")
    print("=" * 60)
    print()
    print("Statistics:")
    print(f"  Folders scanned:      {stats['folders_scanned']}")
    print(f"  Total images:         {stats['total_images']}")
    print(f"  Captions generated:   {stats['captions_generated']}")
    print(f"  Embeddings generated: {stats['embeddings_generated']}")
    print(f"  DB records updated:   {stats['db_records_updated']}")
    print()
    print("Next step: Rebuild FAISS indexes")
    print("  python scripts/build_indexes.py")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Fix image pipeline - generate captions and embeddings"
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
        "--images-dir", "-i",
        default="./output/images",
        help="Directory containing image folders"
    )

    parser.add_argument(
        "--rate-limit", "-r",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)"
    )

    parser.add_argument(
        "--skip-caption-gen", "-s",
        action="store_true",
        help="Skip caption generation (only generate embeddings)"
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = PROJECT_ROOT / images_dir

    print("NeuroSynth Image Pipeline Fix")
    print("=" * 60)
    print(f"Images dir: {images_dir}")
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
        stats = await fix_pipeline(
            images_dir=images_dir,
            connection_string=args.database,
            rate_limit_delay=args.rate_limit,
            skip_caption_gen=args.skip_caption_gen
        )

        print_summary(stats)
        return 0

    except Exception as e:
        logger.exception(f"Fix failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
