#!/usr/bin/env python3
"""
Generate BiomedCLIP 512d visual embeddings for NeuroSynth images.

This script populates the images.embedding column (vector 512d) which is
used by image.faiss for visual similarity search.

Memory-safe implementation:
- Loads model once at startup (not per-image)
- Processes images one at a time (no batch accumulation)
- Explicit garbage collection after each image
- Graceful error handling with continuation

Usage:
    cd <project-root>
    source venv/bin/activate
    python scripts/generate_biomedclip_embeddings.py

    # Then rebuild indexes:
    python scripts/build_indexes.py
"""

import asyncio
import gc
import sys
import os
from pathlib import Path
from io import BytesIO
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment before any other imports
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# BIOMEDCLIP MODEL LOADER
# =============================================================================

class BiomedCLIPEmbedder:
    """
    Memory-optimized BiomedCLIP embedder.

    Outputs 512-dimensional vectors optimized for medical images.
    """

    def __init__(self):
        self.model = None
        self.preprocess = None
        self._loaded = False

    def load(self) -> bool:
        """Load BiomedCLIP model with memory optimizations."""
        if self._loaded:
            return True

        logger.info("Loading BiomedCLIP model...")
        logger.info("  Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        logger.info("  This may take 1-2 minutes on first run (downloading ~890MB)")

        try:
            import open_clip
            import torch

            # Clear any existing tensors
            gc.collect()

            # Load BiomedCLIP via HuggingFace hub
            self.model, self.preprocess = open_clip.create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )

            # Force CPU to avoid GPU memory issues
            self.model = self.model.to("cpu")

            # Set to evaluation mode
            self.model.eval()

            self._loaded = True
            logger.info("✅ BiomedCLIP loaded successfully (512d output)")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install open_clip_torch torch pillow")
            return False

        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            logger.error("This may be a memory issue. Try closing other applications.")
            return False

    def embed(self, image_bytes: bytes) -> list:
        """
        Generate 512d embedding from image bytes.

        Returns:
            List of 512 floats (normalized embedding)
        """
        import torch
        from PIL import Image

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0)

        # Generate embedding
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            # L2 normalize for cosine similarity
            features = features / features.norm(dim=-1, keepdim=True)
            embedding = features[0].cpu().numpy().tolist()

        # Cleanup
        del img, img_tensor, features

        return embedding

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.preprocess
            self.model = None
            self.preprocess = None
            self._loaded = False
            gc.collect()
            logger.info("Model unloaded")


# =============================================================================
# MAIN PROCESSING
# =============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate BiomedCLIP visual embeddings")
    parser.add_argument("--force", action="store_true", help="Re-embed all images")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0=all)")
    parser.add_argument("--test", action="store_true", help="Test model loading only")
    args = parser.parse_args()

    print("=" * 60)
    print("NeuroSynth BiomedCLIP Embedding Generator")
    print("=" * 60)
    print("Output: 512-dimensional visual embeddings")
    print("Column: images.embedding (vector 512d)")
    print("Index:  image.faiss")
    print("=" * 60)

    # Test mode - just load model
    if args.test:
        embedder = BiomedCLIPEmbedder()
        if embedder.load():
            # Test with synthetic image
            from PIL import Image
            test_img = Image.new('RGB', (224, 224), color='blue')
            buf = BytesIO()
            test_img.save(buf, format='PNG')
            embedding = embedder.embed(buf.getvalue())
            logger.info(f"✅ Test embedding: {len(embedding)}d vector")
            logger.info(f"   First 5 values: {embedding[:5]}")
            embedder.unload()
        return 0

    # Connect to database
    try:
        from src.database import init_database, close_database
    except ImportError:
        # Fallback: direct asyncpg connection
        logger.warning("Using direct asyncpg connection")
        import asyncpg

        db_url = os.getenv("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
        db = await asyncpg.connect(db_url)

        async def close_database():
            await db.close()
    else:
        db_url = os.getenv("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
        logger.info(f"Connecting to database...")
        db = await init_database(db_url)

    try:
        # Get image count
        if args.force:
            count_query = """
                SELECT COUNT(*) FROM images
                WHERE file_path IS NOT NULL
                  AND (is_decorative IS NULL OR NOT is_decorative)
            """
        else:
            count_query = """
                SELECT COUNT(*) FROM images
                WHERE image_embedding IS NULL
                  AND file_path IS NOT NULL
                  AND (is_decorative IS NULL OR NOT is_decorative)
            """

        total = await db.fetchval(count_query)
        logger.info(f"Found {total} images to process")

        if total == 0:
            logger.info("All images already have embeddings!")
            logger.info("Use --force to re-embed all images.")
            return 0

        # Load model
        embedder = BiomedCLIPEmbedder()
        if not embedder.load():
            logger.error("Failed to load model. Exiting.")
            return 1

        # Fetch images
        if args.force:
            fetch_query = """
                SELECT id, file_path, storage_path
                FROM images
                WHERE file_path IS NOT NULL
                  AND (is_decorative IS NULL OR NOT is_decorative)
                ORDER BY document_id, page_number
            """
        else:
            fetch_query = """
                SELECT id, file_path, storage_path
                FROM images
                WHERE image_embedding IS NULL
                  AND file_path IS NOT NULL
                  AND (is_decorative IS NULL OR NOT is_decorative)
                ORDER BY document_id, page_number
            """

        if args.limit > 0:
            fetch_query += f" LIMIT {args.limit}"

        rows = await db.fetch(fetch_query)
        logger.info(f"Processing {len(rows)} images...")

        # Process images
        embedded = 0
        errors = 0
        start_time = time.time()

        for i, row in enumerate(rows):
            image_id = row['id']
            file_path = row['file_path']
            image_data = row.get('image_data')

            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"[{i+1}/{len(rows)}] "
                    f"{embedded} embedded, {errors} errors | "
                    f"{rate:.1f} img/s, ETA: {eta:.0f}s"
                )

            try:
                # Get image bytes from file (try file_path first, then storage_path)
                storage_path = row.get('storage_path')
                image_path = file_path or storage_path

                if not image_path:
                    logger.warning(f"  No file path for {image_id}")
                    errors += 1
                    continue

                path = Path(image_path)
                if not path.is_absolute():
                    path = project_root / path

                if not path.exists():
                    # Try storage_path if file_path didn't work
                    if storage_path and storage_path != image_path:
                        path = Path(storage_path)
                        if not path.is_absolute():
                            path = project_root / path

                    if not path.exists():
                        logger.warning(f"  File not found: {path}")
                        errors += 1
                        continue

                image_bytes = path.read_bytes()

                # Generate embedding
                embedding = embedder.embed(image_bytes)

                # Verify dimension
                if len(embedding) != 512:
                    logger.warning(f"  Wrong dimension: {len(embedding)} (expected 512)")
                    errors += 1
                    continue

                # Update database - column: image_embedding (512d for BiomedCLIP)
                await db.execute("""
                    UPDATE images
                    SET image_embedding = $1
                    WHERE id = $2
                """, embedding, image_id)

                embedded += 1

                # Memory cleanup
                del image_bytes, embedding
                if (i + 1) % 5 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"  Error processing {image_id}: {e}")
                errors += 1
                continue

        # Cleanup
        embedder.unload()
        gc.collect()

        # Summary
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("BIOMEDCLIP EMBEDDING COMPLETE")
        print("=" * 60)
        print(f"  Images processed: {len(rows)}")
        print(f"  Successfully embedded: {embedded}")
        print(f"  Errors: {errors}")
        print(f"  Time: {elapsed:.1f}s ({len(rows)/elapsed:.1f} img/s)")
        print()
        print("Next step: Rebuild FAISS indexes")
        print("  python scripts/build_indexes.py")
        print()

        return 0 if errors == 0 else 1

    finally:
        await close_database()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
