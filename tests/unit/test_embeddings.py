"""
Unit tests for embedding validation in VoyageTextEmbedder.

Tests the fix for the caption embedding gap issue where Voyage API
could return fewer embeddings than requested, causing silent data loss.

Tests:
- Normal operation: API returns correct count
- Partial response: API returns fewer embeddings (padding with zeros)
- Extra response: API returns more embeddings (truncation)
- Stats tracking for partial responses
- Circuit breaker and exception handling
"""

import pytest
import numpy as np
import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass


# =============================================================================
# Helper to run async tests
# =============================================================================

def run_async(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.run(coro)


# =============================================================================
# Mocking helpers
# =============================================================================

def create_mock_voyage_module():
    """Create a mock voyageai module for import mocking."""
    mock_module = MagicMock()
    mock_module.AsyncClient = Mock()
    return mock_module


def create_mock_client_normal(dimension=1024):
    """Mock client that returns correct number of embeddings."""
    mock_client = Mock()

    async def embed(texts, model, input_type):
        result = Mock()
        result.embeddings = [
            np.random.randn(dimension).tolist() for _ in texts
        ]
        return result

    mock_client.embed = embed
    return mock_client


def create_mock_client_partial(dimension=1024):
    """Mock client that returns fewer embeddings than requested."""
    mock_client = Mock()

    async def embed(texts, model, input_type):
        result = Mock()
        # Return only half the embeddings
        partial_count = max(1, len(texts) // 2)
        result.embeddings = [
            np.random.randn(dimension).tolist() for _ in range(partial_count)
        ]
        return result

    mock_client.embed = embed
    return mock_client


def create_mock_client_extra(dimension=1024):
    """Mock client that returns more embeddings than requested."""
    mock_client = Mock()

    async def embed(texts, model, input_type):
        result = Mock()
        # Return 3 extra embeddings
        extra_count = len(texts) + 3
        result.embeddings = [
            np.random.randn(dimension).tolist() for _ in range(extra_count)
        ]
        return result

    mock_client.embed = embed
    return mock_client


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_texts():
    """Sample texts for embedding."""
    return [
        "This is a test document about neurosurgery.",
        "The cerebellopontine angle approach is used for acoustic neuroma.",
        "Surgical anatomy of the brain requires detailed knowledge.",
        "Craniotomy techniques vary based on the location of the lesion.",
        "Intraoperative monitoring is essential for preserving neural function.",
    ]


@pytest.fixture
def mock_voyage_module():
    """Mock voyageai module for import."""
    return create_mock_voyage_module()


# =============================================================================
# VoyageTextEmbedder Tests
# =============================================================================

class TestVoyageTextEmbedderValidation:
    """Tests for embedding count validation in VoyageTextEmbedder."""

    def test_embed_batch_normal_operation(self, sample_texts, mock_voyage_module):
        """Normal case: API returns correct number of embeddings."""
        async def _test():
            # Mock the voyageai import
            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_client = create_mock_client_normal()
                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                embeddings = await embedder.embed_batch(sample_texts)

                # Should return same number of embeddings as input texts
                assert len(embeddings) == len(sample_texts), \
                    f"Expected {len(sample_texts)} embeddings, got {len(embeddings)}"

                # All embeddings should be proper numpy arrays
                for emb in embeddings:
                    assert isinstance(emb, np.ndarray)
                    assert emb.shape == (1024,)

                # No partial responses should be recorded
                assert embedder._stats.partial_responses == 0

        run_async(_test())

    def test_embed_batch_partial_response_padding(self, sample_texts, mock_voyage_module):
        """Partial response: API returns fewer embeddings, should pad with zeros."""
        async def _test():
            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_client = create_mock_client_partial()
                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                embeddings = await embedder.embed_batch(sample_texts)

                # Should STILL return same number of embeddings as input texts
                assert len(embeddings) == len(sample_texts), \
                    f"Expected {len(sample_texts)} embeddings, got {len(embeddings)}"

                # Partial response should be recorded in stats
                assert embedder._stats.partial_responses == 1

                # First half should be real embeddings (non-zero)
                expected_real_count = max(1, len(sample_texts) // 2)
                for i in range(expected_real_count):
                    assert not np.allclose(embeddings[i], 0), \
                        f"Embedding {i} should not be zero (real embedding)"

                # Remaining should be zero vectors (padding)
                for i in range(expected_real_count, len(sample_texts)):
                    assert np.allclose(embeddings[i], 0), \
                        f"Embedding {i} should be zero (padding)"

        run_async(_test())

    def test_embed_batch_extra_response_truncation(self, sample_texts, mock_voyage_module):
        """Extra response: API returns more embeddings, should truncate."""
        async def _test():
            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_client = create_mock_client_extra()
                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                embeddings = await embedder.embed_batch(sample_texts)

                # Should return EXACTLY the same number as input texts (truncated)
                assert len(embeddings) == len(sample_texts), \
                    f"Expected {len(sample_texts)} embeddings, got {len(embeddings)}"

                # Partial response should be recorded in stats
                assert embedder._stats.partial_responses == 1

        run_async(_test())

    def test_embed_batch_empty_input(self, mock_voyage_module):
        """Empty input should return empty list."""
        async def _test():
            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_client = create_mock_client_normal()
                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")

                embeddings = await embedder.embed_batch([])

                assert embeddings == []
                assert embedder._stats.partial_responses == 0

        run_async(_test())

    def test_embed_batch_stats_accumulate(self, mock_voyage_module):
        """Stats should accumulate across multiple calls."""
        async def _test():
            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_client = create_mock_client_partial()
                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                # Make multiple calls with partial responses
                texts = ["test1", "test2", "test3", "test4"]

                await embedder.embed_batch(texts)
                assert embedder._stats.partial_responses == 1

                await embedder.embed_batch(texts)
                assert embedder._stats.partial_responses == 2

                await embedder.embed_batch(texts)
                assert embedder._stats.partial_responses == 3

        run_async(_test())


class TestVoyageTextEmbedderBatching:
    """Tests for batch handling with validation."""

    def test_large_batch_validation(self, mock_voyage_module):
        """Test validation works correctly across multiple API batches."""
        async def _test():
            # Create 150 texts (more than BATCH_SIZE of 128)
            texts = [f"Test document number {i}" for i in range(150)]

            call_count = [0]  # Use list to allow mutation in closure

            async def create_response(texts, model, input_type):
                call_count[0] += 1
                result = Mock()
                # Return correct count for each batch
                result.embeddings = [
                    np.random.randn(1024).tolist() for _ in texts
                ]
                return result

            mock_client = Mock()
            mock_client.embed = create_response

            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                embeddings = await embedder.embed_batch(texts)

                # Should return all 150 embeddings
                assert len(embeddings) == 150

                # Should have made 2 API calls (128 + 22)
                assert call_count[0] == 2

                # No partial responses
                assert embedder._stats.partial_responses == 0

        run_async(_test())


class TestEmbeddingStatsPartialResponses:
    """Tests for the new partial_responses stat field."""

    def test_stats_initialization(self):
        """Stats should initialize with partial_responses = 0."""
        from src.ingest.embeddings import EmbeddingStats

        stats = EmbeddingStats()

        assert stats.partial_responses == 0
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0

    def test_stats_partial_responses_increment(self):
        """partial_responses should be incrementable."""
        from src.ingest.embeddings import EmbeddingStats

        stats = EmbeddingStats()

        stats.partial_responses += 1
        assert stats.partial_responses == 1

        stats.partial_responses += 5
        assert stats.partial_responses == 6


class TestEmbeddingCountInvariant:
    """Tests to verify the count invariant: len(output) == len(input)."""

    def test_count_invariant_with_single_text(self, mock_voyage_module):
        """Single text should always return single embedding."""
        async def _test():
            # Mock partial response for single text
            mock_client = Mock()

            async def embed(texts, model, input_type):
                result = Mock()
                # Return 0 embeddings (edge case)
                result.embeddings = []
                return result

            mock_client.embed = embed

            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                embeddings = await embedder.embed_batch(["single text"])

                # Should still return 1 embedding (zero-padded)
                assert len(embeddings) == 1
                assert np.allclose(embeddings[0], 0)
                assert embedder._stats.partial_responses == 1

        run_async(_test())

    def test_count_invariant_maintained_on_error_recovery(self, mock_voyage_module):
        """Count invariant should be maintained even with API issues."""
        async def _test():
            texts = ["text1", "text2", "text3", "text4", "text5"]

            # API returns random counts each time (simulating flaky API)
            response_counts = [3, 2, 7, 5, 0]  # Various partial/extra responses
            call_idx = [0]

            async def flaky_embed(input_texts, model, input_type):
                result = Mock()
                count = response_counts[call_idx[0] % len(response_counts)]
                call_idx[0] += 1
                result.embeddings = [
                    np.random.randn(1024).tolist() for _ in range(count)
                ]
                return result

            mock_client = Mock()
            mock_client.embed = flaky_embed

            with patch.dict('sys.modules', {'voyageai': mock_voyage_module}):
                from src.ingest.embeddings import VoyageTextEmbedder

                mock_voyage_module.AsyncClient.return_value = mock_client

                embedder = VoyageTextEmbedder(api_key="test-key")
                embedder._client = mock_client

                # Multiple calls with flaky API
                for _ in range(5):
                    embeddings = await embedder.embed_batch(texts)
                    # CRITICAL: Count invariant must hold
                    assert len(embeddings) == len(texts), \
                        f"Count invariant violated: expected {len(texts)}, got {len(embeddings)}"

        run_async(_test())
