#!/usr/bin/env python3
"""
Re-link chunks and images with page proximity scoring.

Usage:
    DATABASE_URL=... python scripts/relink_with_proximity.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))


async def relink_with_proximity():
    """Re-run chunk-image linking with page proximity scoring."""

    from src.database.connection import DatabaseConnection

    print("=" * 60)
    print("RE-LINKING WITH PAGE PROXIMITY")
    print("=" * 60)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    db = await DatabaseConnection.initialize(db_url)

    # Check current state
    existing_count = await db.fetchval("SELECT COUNT(*) FROM chunk_image_links")
    print(f"\nExisting links: {existing_count}")

    # Get documents with both chunks and images that have page numbers
    docs = await db.fetch("""
        SELECT DISTINCT d.id, d.title
        FROM documents d
        JOIN chunks c ON c.document_id = d.id
        JOIN images i ON i.document_id = d.id
        WHERE c.page_number IS NOT NULL
          AND i.page_number IS NOT NULL
    """)

    print(f"Documents with page data: {len(docs)}\n")

    if not docs:
        print("No documents with page data found!")
        await db.close()
        return

    # Clear existing links for these documents
    for doc in docs:
        await db.execute("""
            DELETE FROM chunk_image_links
            WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = $1)
        """, doc['id'])

    new_links = 0
    skipped = 0

    for doc in docs:
        doc_id = doc['id']
        title = doc['title'][:35] if doc['title'] else str(doc_id)[:8]

        # Get chunks with embeddings and page numbers
        chunks = await db.fetch("""
            SELECT id, page_number, embedding, cuis
            FROM chunks
            WHERE document_id = $1
              AND embedding IS NOT NULL
              AND page_number IS NOT NULL
        """, doc_id)

        # Get images with embeddings and page numbers
        # Note: images table doesn't have cuis column
        images = await db.fetch("""
            SELECT id, page_number, caption_embedding
            FROM images
            WHERE document_id = $1
              AND NOT COALESCE(is_decorative, false)
              AND page_number IS NOT NULL
        """, doc_id)

        if not chunks or not images:
            continue

        print(f"  {title}: {len(chunks)} chunks, {len(images)} images")

        # Create links based on combined scoring
        for img in images:
            img_page = img['page_number']
            img_cuis = set()  # Images don't have CUIs in this schema

            chunk_scores = []

            for chunk in chunks:
                chunk_page = chunk['page_number']
                chunk_cuis = set(chunk['cuis'] or [])

                # 1. Proximity score (same page = 1.0, decay by distance)
                page_distance = abs(chunk_page - img_page)
                proximity_score = max(0, 1.0 - (page_distance * 0.15))
                # Same page: 1.0, Adjacent: 0.85, 2 pages: 0.70, >6 pages: 0.0

                # 2. CUI overlap score
                if img_cuis and chunk_cuis:
                    cui_overlap = len(img_cuis & chunk_cuis)
                    cui_score = min(1.0, cui_overlap * 0.3)
                else:
                    cui_score = 0.0

                # 3. Semantic score (caption embedding vs chunk embedding)
                semantic_score = 0.0
                img_emb = img['caption_embedding']
                chunk_emb = chunk['embedding']
                if img_emb is not None and chunk_emb is not None:
                    try:
                        sim_result = await db.fetchval("""
                            SELECT 1 - ($1::vector <=> $2::vector)
                        """, list(img_emb), list(chunk_emb))
                        semantic_score = float(sim_result) if sim_result else 0.0
                    except Exception:
                        semantic_score = 0.0

                # Combined score (weighted)
                combined = (
                    proximity_score * 0.4 +    # Spatial proximity
                    semantic_score * 0.4 +     # Caption-chunk similarity
                    cui_score * 0.2            # Medical concept overlap
                )

                chunk_scores.append({
                    'chunk_id': chunk['id'],
                    'score': combined,
                    'proximity': proximity_score,
                    'semantic': semantic_score,
                    'cui': cui_score
                })

            # Keep top 3 chunks per image (with score > threshold)
            top_chunks = sorted(chunk_scores, key=lambda x: x['score'], reverse=True)[:3]

            for match in top_chunks:
                if match['score'] < 0.3:
                    skipped += 1
                    continue

                try:
                    await db.execute("""
                        INSERT INTO chunk_image_links (
                            id, chunk_id, image_id, link_type, relevance_score, link_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (chunk_id, image_id)
                        DO UPDATE SET
                            relevance_score = EXCLUDED.relevance_score,
                            link_type = EXCLUDED.link_type,
                            link_metadata = EXCLUDED.link_metadata
                    """,
                        uuid4(),
                        match['chunk_id'],
                        img['id'],
                        'proximity+semantic',
                        match['score'],
                        json.dumps({
                            'proximity_score': round(match['proximity'], 3),
                            'semantic_score': round(match['semantic'], 3),
                            'cui_score': round(match['cui'], 3)
                        })
                    )
                    new_links += 1
                except Exception as e:
                    print(f"    Link error: {e}")

    # Refresh materialized view
    print("\nRefreshing materialized view...")
    try:
        await db.execute("REFRESH MATERIALIZED VIEW top_chunk_links")
        print("  Materialized view refreshed")
    except Exception as e:
        print(f"  View refresh error: {e}")

    # Final stats
    final_count = await db.fetchval("SELECT COUNT(*) FROM chunk_image_links")
    avg_score = await db.fetchval("SELECT AVG(relevance_score) FROM chunk_image_links")
    high_quality = await db.fetchval(
        "SELECT COUNT(*) FROM chunk_image_links WHERE relevance_score >= 0.5"
    )

    # Distribution
    dist = await db.fetch("""
        SELECT
            CASE
                WHEN relevance_score >= 0.7 THEN 'high (0.7+)'
                WHEN relevance_score >= 0.5 THEN 'medium (0.5-0.7)'
                ELSE 'low (<0.5)'
            END as quality,
            COUNT(*) as count
        FROM chunk_image_links
        GROUP BY 1
        ORDER BY 1
    """)

    await db.close()

    print("\n" + "=" * 60)
    print("RE-LINKING COMPLETE")
    print("=" * 60)
    print(f"  Previous links: {existing_count}")
    print(f"  Current links: {final_count}")
    print(f"  Links created: {new_links}")
    print(f"  Links skipped (low score): {skipped}")
    print(f"  Average score: {avg_score:.3f}" if avg_score else "  Average score: N/A")
    print(f"  High quality (>=0.5): {high_quality}")

    print("\n  Score distribution:")
    for row in dist:
        print(f"    {row['quality']}: {row['count']}")


if __name__ == "__main__":
    asyncio.run(relink_with_proximity())
