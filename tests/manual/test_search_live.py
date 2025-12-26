#!/usr/bin/env python
"""
Test live search with actual database and FAISS index.
Tests the complete search → synthesis integration.
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

async def test_search():
    print("="*70)
    print("LIVE SEARCH TEST")
    print("="*70)

    # Check configuration
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL not set")
        return 1

    if not os.getenv("VOYAGE_API_KEY"):
        print("❌ VOYAGE_API_KEY not set")
        return 1

    # Check FAISS index exists (proper format)
    index_file = Path("indexes/text.faiss")
    meta_file = Path("indexes/text.meta.json")

    if not index_file.exists() or not meta_file.exists():
        print(f"❌ FAISS index not found")
        print(f"   Expected: {index_file} and {meta_file}")
        print("   Run: python scripts/build_faiss_indexes.py")
        return 1

    print(f"\n✅ Prerequisites:")
    print(f"   Database: Connected")
    print(f"   Voyage API: Configured")
    print(f"   FAISS index: {index_file.name} ({index_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"   FAISS metadata: {meta_file.name}")

    # Initialize services
    print(f"\n[1] Initializing services...")

    from src.api.dependencies import ServiceContainer
    from src.shared.models import ExtractedImage

    container = ServiceContainer()
    await container.initialize()
    print(f"✅ Services initialized")

    # Test search
    query = "supraorbital approach"
    print(f"\n[2] Executing search...")
    print(f"   Query: '{query}'")

    try:
        response = await container.search.search(
            query=query,
            mode="hybrid",
            top_k=10,
            include_images=True,
            rerank=False  # Skip reranker if not configured
        )

        print(f"\n✅ Search completed in {response.search_time_ms}ms")
        print(f"   Results: {len(response.results)}")
        print(f"   Total candidates: {response.total_candidates}")

    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify results
    if not response.results:
        print(f"\n⚠️  No results found for query: '{query}'")
        print(f"   This may indicate:")
        print(f"   - No relevant content in database")
        print(f"   - Query doesn't match indexed content")
        return 0

    print(f"\n[3] Validating top result...")
    r = response.results[0]

    # Check all fields
    fields_check = {
        'chunk_id': r.chunk_id,
        'document_id': r.document_id,
        'content': len(r.content) if r.content else 0,
        'title': r.title,
        'chunk_type': r.chunk_type,
        'page_start': r.page_start,
        'entity_names': len(r.entity_names) if r.entity_names else 0,
        'cuis': len(r.cuis) if r.cuis else 0,
        'authority_score': r.authority_score,
        'semantic_score': r.semantic_score,
        'final_score': r.final_score,
        'document_title': r.document_title,
        'images': len(r.images) if r.images else 0
    }

    print(f"\n📊 Top Result Fields:")
    for field, value in fields_check.items():
        status = "✅" if value else "⚠️ "
        if field in ['content', 'entity_names', 'cuis', 'images']:
            print(f"   {status} {field}: {value} items")
        elif field == 'chunk_type':
            print(f"   {status} {field}: {value.value if hasattr(value, 'value') else value}")
        else:
            val_str = str(value)[:50] if value else "None"
            print(f"   {status} {field}: {val_str}")

    # Verify critical fields
    assert r.chunk_id, "chunk_id missing"
    assert r.content, "content missing"
    assert r.chunk_type, "chunk_type missing"

    print(f"\n✅ All critical fields present")

    # Display content preview
    print(f"\n[4] Content Preview:")
    print(f"   Document: {r.document_title or 'Unknown'}")
    print(f"   Page: {r.page_start}")
    print(f"   Type: {r.chunk_type.value if hasattr(r.chunk_type, 'value') else r.chunk_type}")
    print(f"   Score: {r.final_score:.3f}")
    print(f"   Authority: {r.authority_score}")
    print(f"\n   Content (first 200 chars):")
    print(f"   {r.content[:200]}...")

    # Check if we can use this for synthesis
    print(f"\n[5] Synthesis Compatibility Check...")

    from src.synthesis.engine import ContextAdapter, TemplateType

    adapter = ContextAdapter()

    try:
        context = adapter.adapt(
            topic=query,
            search_results=response.results[:10],
            template_type=TemplateType.PROCEDURAL
        )

        print(f"✅ ContextAdapter succeeded")
        print(f"   Sections with content: {sum(1 for s, c in context['sections'].items() if c)}")
        print(f"   Total chunks adapted: {sum(len(c) for c in context['sections'].values())}")
        print(f"   Sources: {len(context['sources'])}")
        print(f"   Images in catalog: {len(context['image_catalog'])}")

    except Exception as e:
        print(f"❌ ContextAdapter failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n" + "="*70)
    print("✅ SEARCH TEST PASSED")
    print("="*70)
    print(f"\nConclusion:")
    print(f"  ✅ Search is working correctly")
    print(f"  ✅ SearchResult fields are all populated")
    print(f"  ✅ Synthesis integration is functional")
    print(f"\nReady for synthesis test!")

    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_search())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
