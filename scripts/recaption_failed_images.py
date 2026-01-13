#!/usr/bin/env python3
"""
NeuroSynth - Re-caption Failed Images
======================================

Identifies and re-processes images that:
1. Are NOT marked decorative
2. Have NULL vlm_caption
3. Have valid storage_path

Usage:
    python scripts/recaption_failed_images.py --document-id <uuid>
    python scripts/recaption_failed_images.py --all --limit 50
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def find_uncaptioned_images(conn, document_id: Optional[UUID] = None, limit: int = 100) -> List[Dict]:
    """Find images that need captioning."""
    
    query = """
        SELECT 
            i.id,
            i.document_id,
            i.storage_path,
            i.page_number,
            i.image_type,
            i.is_decorative,
            i.vlm_caption,
            d.title as document_title
        FROM images i
        JOIN documents d ON i.document_id = d.id
        WHERE i.is_decorative = FALSE
          AND i.vlm_caption IS NULL
          AND i.storage_path IS NOT NULL
    """
    
    params = []
    if document_id:
        query += " AND i.document_id = $1"
        params.append(document_id)
    
    query += " ORDER BY i.page_number LIMIT $" + str(len(params) + 1)
    params.append(limit)
    
    rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def generate_caption(image_path: str, client) -> Optional[str]:
    """Generate VLM caption for an image using Claude."""
    import base64
    
    try:
        path = Path(image_path)
        if not path.exists():
            # Try common base paths
            for base in [Path("data/images"), Path("/data/images"), Path(".")]:
                candidate = base / path.name
                if candidate.exists():
                    path = candidate
                    break
        
        if not path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Read and encode image
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Detect media type
        suffix = path.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(suffix, "image/png")
        
        # Generate caption
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
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
                            "text": """Describe this medical/surgical image in 2-3 sentences. Focus on:
- Anatomical structures visible
- Surgical technique or approach shown (if applicable)
- Key landmarks or labeled structures
- Clinical significance

Be specific and use proper medical terminology."""
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text.strip()
    
    except Exception as e:
        logger.error(f"Caption generation failed for {image_path}: {e}")
        return None


async def update_caption(conn, image_id: UUID, caption: str, embedding_client=None) -> bool:
    """Update image caption and optionally generate caption embedding."""
    try:
        # Update caption
        await conn.execute(
            """
            UPDATE images 
            SET vlm_caption = $1, 
                caption = $1,
                updated_at = NOW()
            WHERE id = $2
            """,
            caption, image_id
        )
        
        # Generate caption embedding if client provided
        if embedding_client:
            try:
                embedding = await asyncio.to_thread(
                    embedding_client.embed,
                    texts=[caption],
                    model="voyage-3",
                    input_type="document"
                )
                
                if embedding and embedding.embeddings:
                    emb_vector = embedding.embeddings[0]
                    await conn.execute(
                        "UPDATE images SET caption_embedding = $1::vector WHERE id = $2",
                        emb_vector, image_id
                    )
                    logger.debug(f"Updated caption embedding for {image_id}")
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to update caption for {image_id}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Re-caption failed images")
    parser.add_argument("--document-id", type=str, help="Specific document UUID")
    parser.add_argument("--all", action="store_true", help="Process all uncaptioned")
    parser.add_argument("--limit", type=int, default=50, help="Max images to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--with-embeddings", action="store_true", help="Also generate caption embeddings")
    args = parser.parse_args()
    
    if not args.document_id and not args.all:
        parser.error("Must specify --document-id or --all")
    
    import asyncpg
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")
    
    # Initialize clients
    try:
        from anthropic import Anthropic
        anthropic_client = Anthropic()
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        return
    
    embedding_client = None
    if args.with_embeddings:
        try:
            import voyageai
            embedding_client = voyageai.Client()
        except Exception as e:
            logger.warning(f"Voyage client not available, skipping embeddings: {e}")
    
    conn = await asyncpg.connect(database_url.replace("+asyncpg", ""))
    
    try:
        doc_id = UUID(args.document_id) if args.document_id else None
        images = await find_uncaptioned_images(conn, doc_id, args.limit)
        
        print(f"\nFound {len(images)} uncaptioned images:")
        for img in images:
            print(f"  Page {img['page_number']}: {img['id'][:8]}... ({img['document_title']})")
        
        if args.dry_run:
            print("\n[DRY RUN] Would process above images")
            return
        
        # Process images
        success = 0
        failed = 0
        
        for img in images:
            print(f"\nProcessing {img['id'][:8]}... (page {img['page_number']})")
            
            caption = await generate_caption(img['storage_path'], anthropic_client)
            
            if caption:
                if await update_caption(conn, img['id'], caption, embedding_client):
                    print(f"  ✓ Caption: {caption[:80]}...")
                    success += 1
                else:
                    print(f"  ✗ Failed to save caption")
                    failed += 1
            else:
                print(f"  ✗ Failed to generate caption")
                failed += 1
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        print(f"\n{'='*50}")
        print(f"Results: {success} succeeded, {failed} failed")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
