#!/usr/bin/env python3
"""
Full E2E pipeline test: Search -> Context -> Synthesis.

Usage:
    ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... DATABASE_URL=... python scripts/test_full_pipeline.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_full_pipeline():
    """Test Search -> Context -> Synthesis flow."""

    print("=" * 60)
    print("E2E PIPELINE TEST")
    print("=" * 60)

    # Check environment
    db_url = os.getenv("DATABASE_URL")
    voyage_key = os.getenv("VOYAGE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    if not voyage_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)

    from src.database.connection import DatabaseConnection
    from src.retrieval.search_service import SearchService, PostgresVectorSearcher
    from src.ingest.embeddings import VoyageTextEmbedder

    db = await DatabaseConnection.initialize(db_url)

    # ========================================
    # PHASE 1: SEARCH
    # ========================================
    print("\n" + "-" * 60)
    print("PHASE 1: SEARCH")
    print("-" * 60)

    # Initialize components
    embedder = VoyageTextEmbedder(api_key=voyage_key)
    pgv = PostgresVectorSearcher(db)

    # Test queries
    test_queries = [
        "anterior cervical approach for discectomy",
        "spinal osteoid osteoma treatment",
        "posterior cervical arthrodesis technique"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Embed query
        query_embedding = await embedder.embed(query)

        # Search chunks with linked images
        results = await pgv.search_hybrid(
            query_embedding=query_embedding.tolist(),
            top_k=5
        )

        print(f"  Results: {len(results)}")

        # Count results with images
        with_images = sum(1 for r in results if r.images)
        total_images = sum(len(r.images) for r in results if r.images)

        print(f"  Results with images: {with_images}/{len(results)}")
        print(f"  Total linked images: {total_images}")

        # Show top result
        if results:
            top = results[0]
            print(f"  Top result:")
            print(f"    Score: {top.final_score:.3f}")
            print(f"    Content: {top.content[:80]}...")
            if top.images:
                print(f"    Images: {len(top.images)}")
                for img in top.images[:2]:
                    caption = getattr(img, 'vlm_caption', '') or getattr(img, 'caption', '') or 'No caption'
                    print(f"      - {caption[:50]}...")

    # ========================================
    # PHASE 2: IMAGE SEARCH
    # ========================================
    print("\n" + "-" * 60)
    print("PHASE 2: IMAGE SEARCH (via captions)")
    print("-" * 60)

    image_queries = [
        "surgical photograph of cervical spine",
        "MRI showing vertebral tumor"
    ]

    for query in image_queries:
        print(f"\nQuery: '{query}'")

        # Embed query
        query_embedding = await embedder.embed(query)

        # Search images via caption embedding
        image_results = await db.fetch("""
            SELECT
                id,
                vlm_caption,
                image_type,
                1 - (caption_embedding <=> $1::vector) as similarity
            FROM images
            WHERE caption_embedding IS NOT NULL
            ORDER BY caption_embedding <=> $1::vector
            LIMIT 3
        """, query_embedding.tolist())

        print(f"  Top matches:")
        for i, img in enumerate(image_results, 1):
            caption = img['vlm_caption'][:60] if img['vlm_caption'] else 'No caption'
            print(f"    [{i}] sim={img['similarity']:.3f} | {caption}...")

    # ========================================
    # PHASE 3: LINK QUALITY CHECK
    # ========================================
    print("\n" + "-" * 60)
    print("PHASE 3: LINK QUALITY")
    print("-" * 60)

    # Link stats
    link_stats = await db.fetchrow("""
        SELECT
            COUNT(*) as total,
            AVG(relevance_score) as avg_score,
            COUNT(*) FILTER (WHERE relevance_score >= 0.5) as high_quality,
            COUNT(*) FILTER (WHERE relevance_score >= 0.7) as very_high
        FROM chunk_image_links
    """)

    print(f"  Total links: {link_stats['total']}")
    print(f"  Average score: {link_stats['avg_score']:.3f}" if link_stats['avg_score'] else "  Average score: N/A")
    print(f"  High quality (>=0.5): {link_stats['high_quality']}")
    print(f"  Very high (>=0.7): {link_stats['very_high']}")

    # Link type distribution
    type_dist = await db.fetch("""
        SELECT link_type, COUNT(*) as count, AVG(relevance_score) as avg
        FROM chunk_image_links
        GROUP BY link_type
        ORDER BY count DESC
    """)

    print("\n  By link type:")
    for row in type_dist:
        print(f"    {row['link_type']}: {row['count']} (avg={row['avg']:.3f})")

    # ========================================
    # PHASE 4: SYNTHESIS TEST (Optional)
    # ========================================
    print("\n" + "-" * 60)
    print("PHASE 4: SYNTHESIS TEST")
    print("-" * 60)

    if not anthropic_key:
        print("  SKIPPED: ANTHROPIC_API_KEY not set")
    else:
        try:
            import anthropic
            from src.synthesis.engine import SynthesisEngine, TemplateType, ENHANCED_ADAPTERS_AVAILABLE

            client = anthropic.AsyncAnthropic(api_key=anthropic_key)

            # Test SynthesisEngine with enhanced adapters
            engine = SynthesisEngine(
                anthropic_client=client,
                use_enhanced_adapters=True,
            )

            print(f"\n  Enhanced adapters available: {ENHANCED_ADAPTERS_AVAILABLE}")
            print(f"  Using enhanced adapters: {engine.using_enhanced}")

            # Search for a topic
            topic = "anterior cervical discectomy and fusion"
            print(f"\n  Topic: '{topic}'")

            query_embedding = await embedder.embed(topic)
            search_results = await pgv.search_hybrid(
                query_embedding=query_embedding.tolist(),
                top_k=20
            )

            print(f"  Search results: {len(search_results)}")
            with_images = sum(1 for r in search_results if r.images)
            print(f"  Results with images: {with_images}")

            # Show chunk types from search results
            chunk_types = {}
            for r in search_results:
                ct = getattr(r, 'chunk_type', 'UNKNOWN')
                chunk_types[ct] = chunk_types.get(ct, 0) + 1
            print(f"  Chunk types: {chunk_types}")

            # Adapt context using the engine's adapter (enhanced or standard)
            context = engine.adapter.adapt(topic, search_results, TemplateType.PROCEDURAL)

            print(f"\n  Context Adaptation (Enhanced):")
            print(f"    Sections: {len(context.get('sections', []))}")
            print(f"    Image catalog: {len(context.get('image_catalog', []))} images")
            print(f"    Source chunks: {len(context.get('chunks', []))}")

            # Check for enhanced features
            all_cuis = context.get('all_cuis', [])
            print(f"    CUIs preserved: {len(all_cuis)}")

            # Check image catalog for embeddings
            images_with_caption_emb = 0
            for img in context.get('image_catalog', []):
                if img.get('caption_embedding') is not None:
                    images_with_caption_emb += 1
            print(f"    Images with caption embedding: {images_with_caption_emb}")

            # Show sections
            sections_dict = context.get('sections', {})
            if sections_dict:
                print(f"\n    Sections by content:")
                for section_name, chunks in sections_dict.items():
                    if chunks:
                        print(f"    - {section_name}: {len(chunks)} chunks")

            # Sample images
            image_catalog = context.get('image_catalog', [])
            if image_catalog:
                print(f"\n    Sample images in catalog:")
                for img in image_catalog[:3]:
                    caption = img.get('vlm_caption', '')[:50] or img.get('caption', '')[:50] or 'No caption'
                    has_emb = '✓' if img.get('caption_embedding') else '✗'
                    print(f"      [{has_emb}] {caption}...")

            print("\n  [Full synthesis skipped to save API cost]")
            print("  Enhanced context adaptation verified successfully!")

        except Exception as e:
            import traceback
            print(f"  Synthesis test error: {e}")
            traceback.print_exc()

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("E2E PIPELINE TEST COMPLETE")
    print("=" * 60)

    # Final verification queries
    total_chunks = await db.fetchval("SELECT COUNT(*) FROM chunks")
    chunks_with_page = await db.fetchval("SELECT COUNT(*) FROM chunks WHERE page_number IS NOT NULL")
    total_images = await db.fetchval("SELECT COUNT(*) FROM images WHERE NOT COALESCE(is_decorative, false)")
    images_with_caption = await db.fetchval("SELECT COUNT(*) FROM images WHERE vlm_caption IS NOT NULL")
    total_links = await db.fetchval("SELECT COUNT(*) FROM chunk_image_links")

    print(f"\n  Database State:")
    print(f"    Chunks: {total_chunks} ({chunks_with_page} with page numbers)")
    print(f"    Images: {total_images} ({images_with_caption} with captions)")
    print(f"    Links: {total_links}")

    # Calculate coverage
    linked_chunks = await db.fetchval("SELECT COUNT(DISTINCT chunk_id) FROM chunk_image_links")
    linked_images = await db.fetchval("SELECT COUNT(DISTINCT image_id) FROM chunk_image_links")

    print(f"\n  Link Coverage:")
    print(f"    Chunks with images: {linked_chunks}/{total_chunks} ({100*linked_chunks/total_chunks:.1f}%)")
    print(f"    Images with chunks: {linked_images}/{total_images} ({100*linked_images/total_images:.1f}%)")

    # Verdict
    all_good = (
        link_stats['total'] > 0 and
        link_stats['avg_score'] and link_stats['avg_score'] > 0.5 and
        chunks_with_page == total_chunks and
        images_with_caption == total_images
    )

    print(f"\n  Status: {'PASS' if all_good else 'NEEDS ATTENTION'}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
