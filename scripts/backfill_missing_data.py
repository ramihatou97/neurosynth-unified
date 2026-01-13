#!/usr/bin/env python3
"""
NeuroSynth Data Backfill Script
================================

Addresses data gaps identified in v2.2 audit:
- 445 chunks missing embeddings
- 140 images missing VLM captions
- 233 images missing caption summaries
- 181 images missing caption embeddings

Usage:
    python scripts/backfill_missing_data.py --all
    python scripts/backfill_missing_data.py --chunk-embeddings
    python scripts/backfill_missing_data.py --vlm-captions
    python scripts/backfill_missing_data.py --caption-summaries
    python scripts/backfill_missing_data.py --caption-embeddings
    python scripts/backfill_missing_data.py --dry-run
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

import asyncpg
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/neurosynth")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

BATCH_SIZE = 20
EMBEDDING_DIM = 1024


# =============================================================================
# DATABASE HELPERS
# =============================================================================

async def get_connection():
    """Get database connection."""
    return await asyncpg.connect(DATABASE_URL)


async def get_missing_chunk_embeddings(conn) -> List[Dict[str, Any]]:
    """Get chunks that are missing embeddings."""
    rows = await conn.fetch("""
        SELECT id, content
        FROM chunks
        WHERE text_embedding IS NULL
        ORDER BY id
    """)
    return [dict(row) for row in rows]


async def get_missing_vlm_captions(conn) -> List[Dict[str, Any]]:
    """Get non-decorative images missing VLM captions."""
    rows = await conn.fetch("""
        SELECT id, image_path, image_type
        FROM images
        WHERE is_decorative = FALSE
          AND vlm_caption IS NULL
          AND image_path IS NOT NULL
        ORDER BY id
    """)
    return [dict(row) for row in rows]


async def get_missing_caption_summaries(conn) -> List[Dict[str, Any]]:
    """Get images with VLM captions but missing caption summaries."""
    rows = await conn.fetch("""
        SELECT id, vlm_caption
        FROM images
        WHERE vlm_caption IS NOT NULL
          AND caption_summary IS NULL
        ORDER BY id
    """)
    return [dict(row) for row in rows]


async def get_missing_caption_embeddings(conn) -> List[Dict[str, Any]]:
    """Get images with VLM captions but missing caption embeddings."""
    rows = await conn.fetch("""
        SELECT id, vlm_caption
        FROM images
        WHERE vlm_caption IS NOT NULL
          AND caption_embedding IS NULL
        ORDER BY id
    """)
    return [dict(row) for row in rows]


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

async def generate_embeddings_voyage(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings using Voyage API."""
    import httpx

    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY environment variable required")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": texts,
                "model": "voyage-3",
                "input_type": "document"
            }
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [
            np.array(item["embedding"], dtype=np.float32)
            for item in data["data"]
        ]
        return embeddings


# =============================================================================
# VLM CAPTIONING
# =============================================================================

async def generate_vlm_caption(image_path: str, image_type: str) -> Optional[str]:
    """Generate VLM caption for an image using Claude."""
    import anthropic
    import base64

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    # Check if file exists
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return None

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_types.get(ext, 'image/png')

    # Select prompt based on image type
    prompts = {
        "surgical_photo": "Describe this surgical photograph. Focus on: anatomical structures visible, surgical instruments, and current surgical step.",
        "anatomy_diagram": "Describe this anatomical diagram. Identify: labeled structures, spatial relationships, and educational focus.",
        "imaging_scan": "Describe this medical imaging scan. Identify: modality, anatomical region, and any pathology visible.",
        "flowchart": "Describe this medical flowchart or algorithm. Summarize: decision points and clinical pathway.",
        "illustration": "Describe this medical illustration. Focus on: subject matter and key anatomical or procedural details."
    }
    prompt = prompts.get(image_type, prompts["illustration"])

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"VLM caption failed for {image_path}: {e}")
        return None


# =============================================================================
# CAPTION SUMMARIZATION
# =============================================================================

async def generate_caption_summary(caption: str) -> str:
    """Generate summary for an image caption."""
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    if not caption or len(caption) < 50:
        return caption or "Medical image"

    prompt = """Summarize this medical image caption in ONE sentence (max 20 words).

Format: [What it shows] — [key detail]

Examples:
- "Pterional craniotomy — dural opening with sylvian fissure exposed"
- "MRI T1 axial — vestibular schwannoma compressing brainstem"
- "Surgical photograph — clip placement on MCA bifurcation aneurysm"

Caption:
{caption}

Summary:""".format(caption=caption[:1000])

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.content[0].text.strip().strip('"\'')
        return summary[:100]
    except Exception as e:
        logger.error(f"Caption summary failed: {e}")
        # Fallback: first clause
        first_part = caption.split(',')[0].split('.')[0]
        return first_part[:77] + "..." if len(first_part) > 80 else first_part


# =============================================================================
# BACKFILL FUNCTIONS
# =============================================================================

async def backfill_chunk_embeddings(conn, dry_run: bool = False):
    """Backfill missing chunk embeddings."""
    logger.info("=== Backfilling Chunk Embeddings ===")

    chunks = await get_missing_chunk_embeddings(conn)
    logger.info(f"Found {len(chunks)} chunks missing embeddings")

    if dry_run:
        logger.info("[DRY RUN] Would process chunks")
        return

    if not chunks:
        logger.info("No chunks to process")
        return

    # Process in batches
    processed = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c['content'][:8000] for c in batch]  # Truncate to Voyage limit

        try:
            embeddings = await generate_embeddings_voyage(texts)

            # Update database - format embedding as string for pgvector
            for chunk, embedding in zip(batch, embeddings):
                emb_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
                await conn.execute("""
                    UPDATE chunks
                    SET text_embedding = $1::vector
                    WHERE id = $2
                """, emb_str, chunk['id'])

            processed += len(batch)
            logger.info(f"Processed {processed}/{len(chunks)} chunks")

            # Rate limiting
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            continue

    logger.info(f"Completed: {processed} chunk embeddings backfilled")


async def backfill_vlm_captions(conn, dry_run: bool = False):
    """Backfill missing VLM captions."""
    logger.info("=== Backfilling VLM Captions ===")

    images = await get_missing_vlm_captions(conn)
    logger.info(f"Found {len(images)} images missing VLM captions")

    if dry_run:
        logger.info("[DRY RUN] Would process images")
        return

    if not images:
        logger.info("No images to process")
        return

    processed = 0
    for img in images:
        try:
            caption = await generate_vlm_caption(
                img['image_path'],
                img.get('image_type', 'illustration')
            )

            if caption:
                await conn.execute("""
                    UPDATE images
                    SET vlm_caption = $1
                    WHERE id = $2
                """, caption, img['id'])
                processed += 1

            logger.info(f"Processed {processed}/{len(images)} images")

            # Rate limiting for Claude API
            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"VLM caption failed for {img['id']}: {e}")
            continue

    logger.info(f"Completed: {processed} VLM captions backfilled")


async def backfill_caption_summaries(conn, dry_run: bool = False):
    """Backfill missing caption summaries."""
    logger.info("=== Backfilling Caption Summaries ===")

    images = await get_missing_caption_summaries(conn)
    logger.info(f"Found {len(images)} images missing caption summaries")

    if dry_run:
        logger.info("[DRY RUN] Would process images")
        return

    if not images:
        logger.info("No images to process")
        return

    processed = 0
    semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls

    async def process_image(img):
        nonlocal processed
        async with semaphore:
            try:
                summary = await generate_caption_summary(img['vlm_caption'])
                await conn.execute("""
                    UPDATE images
                    SET caption_summary = $1
                    WHERE id = $2
                """, summary, img['id'])
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(images)} caption summaries")
            except Exception as e:
                logger.error(f"Caption summary failed for {img['id']}: {e}")

    # Process with limited concurrency
    tasks = [process_image(img) for img in images]
    await asyncio.gather(*tasks)

    logger.info(f"Completed: {processed} caption summaries backfilled")


async def backfill_caption_embeddings(conn, dry_run: bool = False):
    """Backfill missing caption embeddings."""
    logger.info("=== Backfilling Caption Embeddings ===")

    images = await get_missing_caption_embeddings(conn)
    logger.info(f"Found {len(images)} images missing caption embeddings")

    if dry_run:
        logger.info("[DRY RUN] Would process images")
        return

    if not images:
        logger.info("No images to process")
        return

    # Process in batches
    processed = 0
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        captions = [img['vlm_caption'][:8000] for img in batch]

        try:
            embeddings = await generate_embeddings_voyage(captions)

            # Update database - format embedding as string for pgvector
            for img, embedding in zip(batch, embeddings):
                emb_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
                await conn.execute("""
                    UPDATE images
                    SET caption_embedding = $1::vector
                    WHERE id = $2
                """, emb_str, img['id'])

            processed += len(batch)
            logger.info(f"Processed {processed}/{len(images)} caption embeddings")

            # Rate limiting
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            continue

    logger.info(f"Completed: {processed} caption embeddings backfilled")


# =============================================================================
# STATUS CHECK
# =============================================================================

async def check_status(conn):
    """Print current data status."""
    logger.info("=== Current Data Status ===")

    # Chunks
    total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    chunks_with_embeddings = await conn.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE text_embedding IS NOT NULL"
    )
    chunks_with_summaries = await conn.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"
    )

    logger.info(f"Chunks: {total_chunks} total")
    logger.info(f"  - With embeddings: {chunks_with_embeddings} ({100*chunks_with_embeddings/total_chunks:.1f}%)")
    logger.info(f"  - With summaries: {chunks_with_summaries} ({100*chunks_with_summaries/total_chunks:.1f}%)")
    logger.info(f"  - Missing embeddings: {total_chunks - chunks_with_embeddings}")

    # Images
    total_images = await conn.fetchval("SELECT COUNT(*) FROM images")
    non_decorative = await conn.fetchval(
        "SELECT COUNT(*) FROM images WHERE is_decorative = FALSE"
    )
    with_vlm = await conn.fetchval(
        "SELECT COUNT(*) FROM images WHERE vlm_caption IS NOT NULL"
    )
    with_caption_summary = await conn.fetchval(
        "SELECT COUNT(*) FROM images WHERE caption_summary IS NOT NULL"
    )
    with_caption_embedding = await conn.fetchval(
        "SELECT COUNT(*) FROM images WHERE caption_embedding IS NOT NULL"
    )

    logger.info(f"\nImages: {total_images} total, {non_decorative} non-decorative")
    logger.info(f"  - With VLM captions: {with_vlm} ({100*with_vlm/non_decorative:.1f}% of non-decorative)")
    logger.info(f"  - With caption summaries: {with_caption_summary}")
    logger.info(f"  - With caption embeddings: {with_caption_embedding}")
    logger.info(f"  - Missing VLM captions: {non_decorative - with_vlm}")
    logger.info(f"  - Missing caption summaries: {with_vlm - with_caption_summary}")
    logger.info(f"  - Missing caption embeddings: {with_vlm - with_caption_embedding}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Backfill missing NeuroSynth data")
    parser.add_argument("--all", action="store_true", help="Run all backfills")
    parser.add_argument("--chunk-embeddings", action="store_true", help="Backfill chunk embeddings")
    parser.add_argument("--vlm-captions", action="store_true", help="Backfill VLM captions")
    parser.add_argument("--caption-summaries", action="store_true", help="Backfill caption summaries")
    parser.add_argument("--caption-embeddings", action="store_true", help="Backfill caption embeddings")
    parser.add_argument("--status", action="store_true", help="Show current status only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    args = parser.parse_args()

    conn = await get_connection()

    try:
        await check_status(conn)

        if args.status:
            return

        if args.all or args.chunk_embeddings:
            await backfill_chunk_embeddings(conn, args.dry_run)

        if args.all or args.vlm_captions:
            await backfill_vlm_captions(conn, args.dry_run)

        if args.all or args.caption_summaries:
            await backfill_caption_summaries(conn, args.dry_run)

        if args.all or args.caption_embeddings:
            await backfill_caption_embeddings(conn, args.dry_run)

        if not any([args.all, args.chunk_embeddings, args.vlm_captions,
                    args.caption_summaries, args.caption_embeddings]):
            logger.info("\nNo backfill operation specified. Use --all or specific flags.")
            logger.info("Use --status for status only, --dry-run to preview changes.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
