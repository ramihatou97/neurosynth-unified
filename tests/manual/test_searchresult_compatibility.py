#!/usr/bin/env python
"""
Test SearchResult Model Compatibility
Verifies that SearchResult can be created and consumed by SynthesisEngine
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_searchresult_compatibility():
    print("="*70)
    print("SEARCHRESULT MODEL COMPATIBILITY TEST")
    print("="*70)

    from src.shared.models import SearchResult, ExtractedImage, ChunkType

    print("\n[Test 1] Create SearchResult with all required fields")

    # Create mock ExtractedImage
    test_image = ExtractedImage(
        id="img_test_001",
        document_id="doc_test",
        page_number=45,
        file_path=Path("/data/images/test_001.jpg"),
        width=1200,
        height=900,
        format="JPEG",
        content_hash="abc123",
        caption="Supraorbital craniotomy exposure",
        vlm_caption="Intraoperative photograph showing supraorbital craniotomy with frontal lobe retraction",
        image_type="surgical_photo",
        quality_score=0.92
    )

    # Create SearchResult with ALL fields (as returned by SearchService)
    search_result = SearchResult(
        chunk_id="chunk_test_001",
        document_id="doc_test",
        content="The supraorbital approach provides excellent access to the anterior cranial fossa. The key steps include patient positioning in supine with head rotation, frontotemporal skin incision, and supraorbital craniotomy.",
        title="Supraorbital Approach",
        chunk_type=ChunkType.PROCEDURE,
        page_start=45,
        entity_names=["supraorbital", "anterior cranial fossa", "frontal lobe"],
        image_ids=["img_test_001", "img_test_002"],
        cuis=["C0205094", "C0149566"],
        authority_score=0.85,
        keyword_score=0.0,
        semantic_score=0.89,
        final_score=0.89,
        document_title="Keyhole Approaches in Neurosurgery - Volume 1",
        images=[test_image]
    )

    print("✅ SearchResult created successfully")
    print(f"   chunk_id: {search_result.chunk_id}")
    print(f"   document_title: {search_result.document_title}")
    print(f"   authority_score: {search_result.authority_score}")
    print(f"   entity_names: {search_result.entity_names}")
    print(f"   cuis: {search_result.cuis}")
    print(f"   chunk_type: {search_result.chunk_type.value}")
    print(f"   images: {len(search_result.images)} attached")

    print("\n[Test 2] Verify SynthesisEngine can consume SearchResult")

    from src.synthesis.engine import ContextAdapter, TemplateType

    adapter = ContextAdapter()

    # This is the critical integration point that was broken before
    try:
        context = adapter.adapt(
            topic="supraorbital approach",
            search_results=[search_result],  # List[SearchResult]
            template_type=TemplateType.PROCEDURAL
        )

        print("✅ ContextAdapter.adapt() succeeded")
        print(f"   Topic: {context['topic']}")
        print(f"   Template: {context['template_type'].value}")
        print(f"   Sections: {list(context['sections'].keys())}")
        print(f"   Sources: {len(context['sources'])}")
        print(f"   Image catalog: {len(context['image_catalog'])}")

        # Verify chunk structure in context
        for section, chunks in context['sections'].items():
            if chunks:
                chunk = chunks[0]
                # ContextAdapter creates dict with 'id' (not 'chunk_id')
                assert 'id' in chunk, "Missing id in adapted chunk"
                assert 'authority_score' in chunk, "Missing authority_score in adapted chunk"
                assert 'entity_names' in chunk, "Missing entity_names in adapted chunk"
                assert 'content' in chunk, "Missing content in adapted chunk"
                assert 'document_title' in chunk, "Missing document_title in adapted chunk"
                print(f"\n✅ Section '{section}' chunks properly structured:")
                print(f"     id: {chunk['id']}")
                print(f"     authority_score: {chunk['authority_score']}")
                print(f"     entity_names: {chunk['entity_names'][:2]}")
                break

    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        print("   This indicates field mismatch between SearchResult and ContextAdapter")
        raise

    except TypeError as e:
        print(f"❌ TypeError: {e}")
        print("   This indicates type incompatibility")
        raise

    print("\n[Test 3] Verify image handling")

    # Check that images are accessed correctly
    for img_data in context['image_catalog']:
        assert 'id' in img_data, "Missing image id"
        assert 'caption' in img_data, "Missing caption"
        assert 'path' in img_data, "Missing path"
        print(f"✅ Image catalog entry: {img_data['id']}")
        break

    print("\n" + "="*70)
    print("✅ ALL COMPATIBILITY TESTS PASSED")
    print("="*70)
    print("\nConclusion:")
    print("  ✅ SearchResult model is synthesis-compatible")
    print("  ✅ No AttributeError crashes")
    print("  ✅ No TypeError crashes")
    print("  ✅ All fields accessible")
    print("  ✅ Images handled correctly")
    print("\n  The integration gap has been successfully bridged!")

if __name__ == "__main__":
    try:
        test_searchresult_compatibility()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ COMPATIBILITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
