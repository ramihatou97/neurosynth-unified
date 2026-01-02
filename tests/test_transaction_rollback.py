"""
Test Transaction Rollback in Database Writer
=============================================

Verifies that partial failures during pipeline writes result in complete rollback,
preventing orphaned chunks/images/links in the database.

Run with:
    DATABASE_URL="postgresql://..." python tests/test_transaction_rollback.py
"""

import asyncio
import os
import sys
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.connection import DatabaseConnection
from src.ingest.database_writer import PipelineDatabaseWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MockChunk:
    """Mock chunk for testing."""
    id: str
    content: str
    chunk_type: str = "general"
    page_number: int = 1
    cuis: List[str] = field(default_factory=list)
    entities: List[dict] = field(default_factory=list)
    text_embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockImage:
    """Mock image for testing."""
    id: str
    file_path: str
    file_name: str
    width: int = 100
    height: int = 100
    format: str = "png"
    page_number: int = 1
    image_type: str = "diagram"
    is_decorative: bool = False
    vlm_caption: Optional[str] = None
    embedding: Optional[List[float]] = None
    caption_embedding: Optional[List[float]] = None
    cuis: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    triage_skipped: bool = False
    triage_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockLink:
    """Mock link for testing."""
    chunk_id: str
    image_id: str
    link_type: str = "semantic"
    score: float = 0.8
    proximity_score: Optional[float] = None
    semantic_score: Optional[float] = None
    cui_overlap_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockPipelineResult:
    """Mock pipeline result for testing."""
    source_path: str
    chunks: List[MockChunk]
    images: List[MockImage]
    links: List[MockLink]
    document_id: str = None
    total_pages: int = 1
    processing_time_seconds: float = 1.0
    text_embedding_dim: int = 1024
    image_embedding_dim: int = 512
    text_embedding_provider: str = "voyage"


class FailingPipelineResult(MockPipelineResult):
    """Pipeline result that will cause a failure during image insertion."""

    def __init__(self, *args, fail_at_images: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_at_images = fail_at_images


async def get_table_counts(db: DatabaseConnection) -> dict:
    """Get current counts for all relevant tables."""
    counts = {}
    for table in ['documents', 'chunks', 'images', 'links', 'entities']:
        result = await db.fetchval(f"SELECT COUNT(*) FROM {table}")
        counts[table] = result
    return counts


async def test_successful_write():
    """Test that successful writes work correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Successful Write")
    logger.info("=" * 60)

    connection_string = os.getenv("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
    db = await DatabaseConnection.initialize(connection_string)

    # Get initial counts
    initial_counts = await get_table_counts(db)
    logger.info(f"Initial counts: {initial_counts}")

    # Create mock data
    test_id = str(uuid4())[:8]
    result = MockPipelineResult(
        source_path=f"/test/rollback_test_{test_id}.pdf",
        chunks=[
            MockChunk(id="c1", content="Test chunk 1 content"),
            MockChunk(id="c2", content="Test chunk 2 content"),
        ],
        images=[
            MockImage(id="i1", file_path="/test/image1.png", file_name="image1.png"),
        ],
        links=[
            MockLink(chunk_id="c1", image_id="i1"),
        ]
    )

    # Write to database
    writer = PipelineDatabaseWriter(db=db)
    await writer.connect()

    doc_id = await writer.write_pipeline_result(result)
    logger.info(f"Document created: {doc_id}")

    # Verify counts increased
    final_counts = await get_table_counts(db)
    logger.info(f"Final counts: {final_counts}")

    assert final_counts['documents'] == initial_counts['documents'] + 1, "Document should be created"
    assert final_counts['chunks'] == initial_counts['chunks'] + 2, "Chunks should be created"
    assert final_counts['images'] == initial_counts['images'] + 1, "Image should be created"
    assert final_counts['links'] == initial_counts['links'] + 1, "Link should be created"

    # Cleanup
    await db.execute("DELETE FROM links WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = $1)", doc_id)
    await db.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
    await db.execute("DELETE FROM images WHERE document_id = $1", doc_id)
    await db.execute("DELETE FROM documents WHERE id = $1", doc_id)

    logger.info("✓ TEST 1 PASSED: Successful write works correctly")
    return True


async def test_rollback_on_failure():
    """Test that failures during write result in complete rollback."""
    logger.info("=" * 60)
    logger.info("TEST 2: Rollback on Failure")
    logger.info("=" * 60)

    connection_string = os.getenv("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
    db = await DatabaseConnection.initialize(connection_string)

    # Get initial counts
    initial_counts = await get_table_counts(db)
    logger.info(f"Initial counts: {initial_counts}")

    # Create mock data with an image that will fail (invalid embedding dimension)
    test_id = str(uuid4())[:8]

    @dataclass
    class BadImage(MockImage):
        """Image with invalid embedding that will cause insert to fail."""
        embedding: List[float] = field(default_factory=lambda: [1.0] * 10)  # Wrong dimension

    result = MockPipelineResult(
        source_path=f"/test/rollback_test_fail_{test_id}.pdf",
        chunks=[
            MockChunk(id="c1", content="Test chunk 1 content"),
            MockChunk(id="c2", content="Test chunk 2 content"),
        ],
        images=[
            BadImage(id="i1", file_path="/test/bad_image.png", file_name="bad_image.png"),
        ],
        links=[
            MockLink(chunk_id="c1", image_id="i1"),
        ]
    )

    # Attempt write - should fail
    writer = PipelineDatabaseWriter(db=db)
    await writer.connect()

    try:
        await writer.write_pipeline_result(result)
        logger.error("✗ Expected failure but write succeeded!")
        return False
    except Exception as e:
        logger.info(f"Expected error occurred: {type(e).__name__}: {e}")

    # Verify counts unchanged (rollback worked)
    final_counts = await get_table_counts(db)
    logger.info(f"Final counts after rollback: {final_counts}")

    assert final_counts['documents'] == initial_counts['documents'], f"Documents should be unchanged: {final_counts['documents']} vs {initial_counts['documents']}"
    assert final_counts['chunks'] == initial_counts['chunks'], f"Chunks should be unchanged: {final_counts['chunks']} vs {initial_counts['chunks']}"
    assert final_counts['images'] == initial_counts['images'], f"Images should be unchanged: {final_counts['images']} vs {initial_counts['images']}"
    assert final_counts['links'] == initial_counts['links'], f"Links should be unchanged: {final_counts['links']} vs {initial_counts['links']}"

    logger.info("✓ TEST 2 PASSED: Rollback on failure works correctly")
    return True


async def test_rollback_leaves_no_orphans():
    """Test that failed writes leave no orphaned records."""
    logger.info("=" * 60)
    logger.info("TEST 3: No Orphaned Records After Rollback")
    logger.info("=" * 60)

    connection_string = os.getenv("DATABASE_URL", "postgresql://ramihatoum@localhost:5432/neurosynth")
    db = await DatabaseConnection.initialize(connection_string)

    # Get test source path
    test_id = str(uuid4())[:8]
    source_path = f"/test/orphan_test_{test_id}.pdf"

    # Count documents with this source path before
    before_count = await db.fetchval(
        "SELECT COUNT(*) FROM documents WHERE source_path = $1",
        source_path
    )
    assert before_count == 0, "Test source path should not exist before test"

    # Create mock data that will fail during link insertion
    # (by having mismatched IDs)
    result = MockPipelineResult(
        source_path=source_path,
        chunks=[
            MockChunk(id="c1", content="Test chunk content"),
        ],
        images=[
            MockImage(
                id="i1",
                file_path="/test/image.png",
                file_name="image.png",
                # Use invalid vector dimension to cause failure
                embedding=[1.0] * 10  # Wrong dimension (should be 512)
            ),
        ],
        links=[
            MockLink(chunk_id="c1", image_id="i1"),
        ]
    )

    # Attempt write
    writer = PipelineDatabaseWriter(db=db)
    await writer.connect()

    try:
        await writer.write_pipeline_result(result)
        # If we get here, the test setup didn't cause a failure
        # Clean up and report
        doc = await db.fetchrow("SELECT id FROM documents WHERE source_path = $1", source_path)
        if doc:
            await db.execute("DELETE FROM links WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = $1)", doc['id'])
            await db.execute("DELETE FROM chunks WHERE document_id = $1", doc['id'])
            await db.execute("DELETE FROM images WHERE document_id = $1", doc['id'])
            await db.execute("DELETE FROM documents WHERE id = $1", doc['id'])
        logger.warning("Note: Test data was valid, consider using different failure trigger")
    except Exception as e:
        logger.info(f"Expected error: {e}")

    # Verify no documents with this source path exist
    after_count = await db.fetchval(
        "SELECT COUNT(*) FROM documents WHERE source_path = $1",
        source_path
    )

    # Also check for orphaned chunks/images that might reference this source path
    orphan_check = await db.fetchval(
        """
        SELECT COUNT(*) FROM chunks c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
        """
    )

    assert after_count == 0, f"No documents should exist with test source path after rollback: found {after_count}"
    assert orphan_check == 0, f"No orphaned chunks should exist: found {orphan_check}"

    logger.info("✓ TEST 3 PASSED: No orphaned records after rollback")
    return True


async def run_all_tests():
    """Run all transaction rollback tests."""
    logger.info("\n" + "=" * 60)
    logger.info("TRANSACTION ROLLBACK TEST SUITE")
    logger.info("=" * 60 + "\n")

    results = []

    # Test 1: Successful write
    try:
        results.append(("Successful Write", await test_successful_write()))
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}")
        results.append(("Successful Write", False))

    # Test 2: Rollback on failure
    try:
        results.append(("Rollback on Failure", await test_rollback_on_failure()))
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}")
        results.append(("Rollback on Failure", False))

    # Test 3: No orphans after rollback
    try:
        results.append(("No Orphaned Records", await test_rollback_leaves_no_orphans()))
    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}")
        results.append(("No Orphaned Records", False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        logger.info(f"  {status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED - Transaction rollback is working correctly!")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED - Review the output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
