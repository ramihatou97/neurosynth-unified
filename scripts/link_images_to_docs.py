#!/usr/bin/env python3
"""
Link existing image folders to their documents in the database.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Get project root directory (portable)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Mapping: folder_id -> document_id
FOLDER_TO_DOC = {
    # Chapter 4_ The Cerebral Veins
    "1cae649c-6bf4-41e4-a17e-38bb1b0cb5e7": "c15bd6f5-44c2-4edb-ab69-196fe7fdf568",
    # Chapter 2_ The Supratentorial Arteries
    "8ebb9c41-7910-4eb2-bdcc-471c4a12dfe1": "bb0ea46d-b79e-451d-8f94-c998d0a80b54",
}

async def link_images():
    from src.database import init_database, close_database
    from src.ingest.embeddings import VoyageTextEmbedder

    images_dir = PROJECT_ROOT / "output" / "images"

    db = await init_database(os.getenv('DATABASE_URL', 'postgresql://neurosynth:neurosynth@localhost:5432/neurosynth'))

    # Initialize embedder
    embedder = VoyageTextEmbedder(model="voyage-3", api_key=os.getenv("VOYAGE_API_KEY"))

    created = 0
    embedded = 0

    try:
        for folder_id, doc_id in FOLDER_TO_DOC.items():
            folder = images_dir / folder_id
            if not folder.exists():
                print(f"Folder not found: {folder}")
                continue

            print(f"\nProcessing {folder_id} -> doc {doc_id}")

            # Verify document exists
            doc_exists = await db.fetchval("SELECT id FROM documents WHERE id = $1::uuid", doc_id)
            if not doc_exists:
                print(f"  Document {doc_id} not found!")
                continue

            # Read captions file
            captions_file = folder / "vlm_captions.json"
            captions_map = {}
            if captions_file.exists():
                with open(captions_file) as f:
                    for item in json.load(f):
                        captions_map[item.get("image_file", "")] = item

            # Process each image
            for img_file in folder.glob("*.png"):
                file_name = img_file.name
                content_hash = img_file.stem

                caption_info = captions_map.get(file_name, {})
                vlm_caption = caption_info.get("vlm_caption")
                vlm_type = caption_info.get("vlm_type")
                page = caption_info.get("page")

                # Parse size
                size_str = caption_info.get("size", "")
                width, height = None, None
                if "x" in size_str:
                    try:
                        width, height = map(int, size_str.split("x"))
                    except:
                        pass

                # Generate caption embedding if we have a caption
                caption_embedding = None
                if vlm_caption:
                    try:
                        emb = await embedder.embed(vlm_caption)
                        caption_embedding = emb.tolist()
                        embedded += 1
                    except Exception as e:
                        print(f"    Embedding error for {file_name}: {e}")

                # Insert image record
                try:
                    await db.execute("""
                        INSERT INTO images (
                            document_id, storage_path, file_path, file_name, content_hash,
                            page_number, width, height, vlm_caption, caption,
                            image_type, caption_embedding, is_decorative
                        ) VALUES ($1::uuid, $2, $2, $3, $4, $5, $6, $7, $8, $8, $9, $10, false)
                    """,
                        doc_id,
                        str(img_file),
                        file_name,
                        content_hash,
                        page,
                        width,
                        height,
                        vlm_caption,
                        vlm_type,
                        caption_embedding
                    )
                    created += 1
                except Exception as e:
                    print(f"    DB error for {file_name}: {e}")

            print(f"  Created {created} image records")

        # Update document stats
        for doc_id in FOLDER_TO_DOC.values():
            img_count = await db.fetchval(
                "SELECT COUNT(*) FROM images WHERE document_id = $1::uuid",
                doc_id
            )
            await db.execute(
                "UPDATE documents SET total_images = $1 WHERE id = $2::uuid",
                img_count,
                doc_id
            )
            print(f"Updated document {doc_id[:8]}... total_images = {img_count}")

    finally:
        await close_database()

    print(f"\n=== Summary ===")
    print(f"Images created: {created}")
    print(f"Captions embedded: {embedded}")
    print(f"\nNext: python scripts/build_indexes.py")


if __name__ == "__main__":
    asyncio.run(link_images())
