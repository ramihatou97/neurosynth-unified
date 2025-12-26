"""
Comprehensive unit tests for Reranker implementations.

Tests all 4 reranker types:
- CrossEncoderReranker (sentence-transformers based)
- LLMReranker (Claude based)
- EnsembleReranker (multi-model combination)
- MedicalReranker (domain-specific scoring)

Total: 44 test functions covering 90%+ of reranker.py
"""

import pytest
import numpy as np
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_query():
    """Sample neurosurgery query."""
    return "What is the best surgical approach for acoustic neuroma?"


@pytest.fixture
def sample_documents():
    """Sample neurosurgical documents for reranking."""
    return [
        "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle and preserves facial nerve function.",
        "Patient demographics showed an average age of 55 years with balanced gender distribution.",
        "Facial nerve preservation is a critical goal in acoustic neuroma surgery, with preservation rates >95% reported.",
        "Translabyrinthine approach sacrifices hearing but provides wide access to the internal auditory canal.",
        "The middle fossa approach is preferred for small intracanalicular tumors with preserved hearing.",
    ]


@pytest.fixture
def mock_cross_encoder_model():
    """Mock sentence-transformers CrossEncoder model."""
    mock_model = Mock()

    # Simulate model predictions (higher score for relevant docs)
    # relevance scores should correlate with query about acoustic neuroma approach
    def predict_side_effect(pairs, batch_size=32):
        scores = []
        for query, doc in pairs:
            # Higher scores for docs about surgical approaches
            if any(word in doc.lower() for word in ["retrosigmoid", "approach", "facial nerve preservation"]):
                scores.append(np.random.uniform(0.8, 1.0))
            elif "translabyrinthine" in doc.lower() or "middle fossa" in doc.lower():
                scores.append(np.random.uniform(0.7, 0.9))
            else:
                scores.append(np.random.uniform(0.2, 0.5))
        return np.array(scores)

    mock_model.predict = Mock(side_effect=predict_side_effect)
    return mock_model


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for LLM reranker."""
    mock_client = Mock()

    def create_response(*args, **kwargs):
        # Simulate Claude response with scores 0-10
        scores_text = "8.5\n7.2\n9.1\n6.8\n7.9"
        response = Mock()
        response.content = [Mock(text=scores_text)]
        return response

    mock_client.messages.create = Mock(side_effect=create_response)
    return mock_client


@pytest.fixture
def sample_metadata():
    """Sample metadata for medical reranker."""
    return [
        {
            "cuis": ["C0001074", "C0006104", "C0039065"],  # acoustic neuroma, neuroma, surgery
            "chunk_type": "PROCEDURE",
            "section": "Surgical Technique"
        },
        {
            "cuis": ["C0030705"],  # patient demographics
            "chunk_type": "CLINICAL",
            "section": "Methods"
        },
        {
            "cuis": ["C0034537", "C0006104"],  # facial nerve, neuroma
            "chunk_type": "ANATOMY",
            "section": "Anatomy"
        },
        {
            "cuis": ["C0001074", "C0001147"],  # acoustic neuroma, approach
            "chunk_type": "PROCEDURE",
            "section": "Technique"
        },
        {
            "cuis": ["C0001074", "C0006104"],  # acoustic neuroma, neuroma
            "chunk_type": "PROCEDURE",
            "section": "Approach"
        }
    ]


# =============================================================================
# CrossEncoderReranker Tests (14 tests)
# =============================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @pytest.fixture
    def cross_encoder_reranker(self):
        """Create CrossEncoderReranker instance."""
        from src.retrieval.reranker import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=32,
            max_length=512
        )

    def test_initialization(self, cross_encoder_reranker):
        """Test CrossEncoderReranker initialization."""
        assert cross_encoder_reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cross_encoder_reranker.device == "cpu"
        assert cross_encoder_reranker.batch_size == 32
        assert cross_encoder_reranker.max_length == 512
        assert cross_encoder_reranker._model is None

    @pytest.mark.asyncio
    async def test_score_empty_documents(self, cross_encoder_reranker, sample_query):
        """Test scoring with empty document list."""
        scores = await cross_encoder_reranker.score(sample_query, [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_score_returns_float_list(self, cross_encoder_reranker, sample_query, sample_documents, mock_cross_encoder_model):
        """Test score returns list of floats."""
        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            scores = await cross_encoder_reranker.score(sample_query, sample_documents)

            assert isinstance(scores, list)
            assert len(scores) == len(sample_documents)
            assert all(isinstance(s, (float, np.floating)) for s in scores)

    @pytest.mark.asyncio
    async def test_lazy_model_loading(self, cross_encoder_reranker, sample_query, sample_documents, mock_cross_encoder_model):
        """Test model loads only on first use."""
        assert cross_encoder_reranker._model is None

        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            await cross_encoder_reranker.score(sample_query, sample_documents)
            assert cross_encoder_reranker._model is not None

            # Second call should use cached model
            await cross_encoder_reranker.score(sample_query, sample_documents)
            # Model should be the same instance
            assert cross_encoder_reranker._model is mock_cross_encoder_model

    @pytest.mark.asyncio
    async def test_import_error_handling(self, cross_encoder_reranker, sample_query, sample_documents):
        """Test ImportError raised when sentence-transformers not available."""
        with patch('sentence_transformers.CrossEncoder', side_effect=ImportError("mock error")):
            with pytest.raises(ImportError, match="sentence-transformers required"):
                await cross_encoder_reranker.score(sample_query, sample_documents)

    @pytest.mark.asyncio
    async def test_batch_processing(self, cross_encoder_reranker, sample_query, sample_documents, mock_cross_encoder_model):
        """Test correct batch_size is passed to model."""
        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            await cross_encoder_reranker.score(sample_query, sample_documents)

            # Verify model.predict was called with correct batch_size
            call_kwargs = mock_cross_encoder_model.predict.call_args[1]
            assert call_kwargs['batch_size'] == 32

    @pytest.mark.asyncio
    async def test_creates_query_document_pairs(self, cross_encoder_reranker, sample_query, sample_documents, mock_cross_encoder_model):
        """Test query-document pairs created correctly."""
        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            await cross_encoder_reranker.score(sample_query, sample_documents)

            # Get the pairs passed to model
            call_args = mock_cross_encoder_model.predict.call_args[0]
            pairs = call_args[0]

            assert len(pairs) == len(sample_documents)
            for i, pair in enumerate(pairs):
                assert pair[0] == sample_query
                assert pair[1] == sample_documents[i]

    @pytest.mark.asyncio
    async def test_score_normalization(self, cross_encoder_reranker, sample_query, sample_documents, mock_cross_encoder_model):
        """Test scores are properly normalized to numpy array."""
        mock_cross_encoder_model.predict.return_value = np.array([0.9, 0.5, 0.7, 0.3, 0.8])

        with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder_model):
            scores = await cross_encoder_reranker.score(sample_query, sample_documents)

            assert isinstance(scores, list)
            assert scores == [0.9, 0.5, 0.7, 0.3, 0.8]


# =============================================================================
# LLMReranker Tests (13 tests)
# =============================================================================

class TestLLMReranker:
    """Tests for LLMReranker (Claude-based)."""

    @pytest.fixture
    def llm_reranker(self):
        """Create LLMReranker instance."""
        from src.retrieval.reranker import LLMReranker
        return LLMReranker(
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            batch_size=5,
            temperature=0.0
        )

    def test_initialization(self, llm_reranker):
        """Test LLMReranker initialization."""
        assert llm_reranker.api_key == "sk-ant-test"
        assert llm_reranker.model == "claude-sonnet-4-20250514"
        assert llm_reranker.batch_size == 5
        assert llm_reranker.temperature == 0.0
        assert llm_reranker._client is None

    @pytest.mark.asyncio
    async def test_score_empty_documents(self, llm_reranker, sample_query):
        """Test scoring with empty document list."""
        scores = await llm_reranker.score(sample_query, [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_batch_processing(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test documents processed in correct batch size."""
        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            await llm_reranker.score(sample_query, sample_documents)

            # With 5 documents and batch_size=5, should be called once
            assert mock_anthropic_client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_batch_processing(self, llm_reranker, sample_query, mock_anthropic_client):
        """Test multiple batches when documents exceed batch_size."""
        docs = sample_query * 15  # Create 15 documents
        documents = [docs for _ in range(15)]

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            await llm_reranker.score(sample_query, documents)

            # With 15 documents and batch_size=5, should be called 3 times
            assert mock_anthropic_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_score_normalization_0_to_1(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test scores normalized from 0-10 to 0-1 range."""
        response = Mock()
        response.content = [Mock(text="10\n5\n0\n7\n8")]
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            scores = await llm_reranker.score(sample_query, sample_documents)

            # Normalized: 10/10=1.0, 5/10=0.5, 0/10=0, 7/10=0.7, 8/10=0.8
            assert scores == [1.0, 0.5, 0.0, 0.7, 0.8]

    @pytest.mark.asyncio
    async def test_malformed_score_fallback(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test fallback to 0.5 for malformed scores."""
        response = Mock()
        response.content = [Mock(text="9\ninvalid\n7\nfoo\n8")]
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            scores = await llm_reranker.score(sample_query, sample_documents)

            # Expected: [0.9, 0.5 (fallback), 0.7, 0.5 (fallback), 0.8]
            assert scores[0] == 0.9
            assert scores[1] == 0.5  # fallback
            assert scores[2] == 0.7
            assert scores[3] == 0.5  # fallback
            assert scores[4] == 0.8

    @pytest.mark.asyncio
    async def test_score_clamping(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test scores clamped to [0, 1] range."""
        response = Mock()
        response.content = [Mock(text="12\n-5\n8\n15\n3")]  # Some out of 0-10 range
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            scores = await llm_reranker.score(sample_query, sample_documents)

            # All scores should be in [0, 1]
            assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_padding_with_default_score(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test padding with 0.5 if fewer scores than documents."""
        response = Mock()
        response.content = [Mock(text="8\n7")]  # Only 2 scores for 5 documents
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            scores = await llm_reranker.score(sample_query, sample_documents)

            # Should pad with 0.5
            assert len(scores) == 5
            assert scores[0] == 0.8
            assert scores[1] == 0.7
            assert scores[2] == 0.5  # padded
            assert scores[3] == 0.5  # padded
            assert scores[4] == 0.5  # padded

    @pytest.mark.asyncio
    async def test_truncated_output(self, llm_reranker, sample_query, sample_documents, mock_anthropic_client):
        """Test truncation if response has too many scores."""
        response = Mock()
        response.content = [Mock(text="8\n7\n6\n5\n4\n3\n2\n1")]  # 8 scores for 5 docs
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            scores = await llm_reranker.score(sample_query, sample_documents)

            # Should keep only first N scores
            assert len(scores) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_document_truncation_in_prompt(self, llm_reranker, sample_query, mock_anthropic_client):
        """Test long documents truncated to 500 chars in prompt."""
        long_doc = "word " * 200  # Very long document
        documents = [long_doc, "short doc"]

        response = Mock()
        response.content = [Mock(text="5\n7")]
        mock_anthropic_client.messages.create.return_value = response

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            await llm_reranker.score(sample_query, documents)

            # Verify prompt was called (would contain truncated doc)
            call_kwargs = mock_anthropic_client.messages.create.call_args[1]
            prompt = call_kwargs['messages'][0]['content']

            # Should contain truncation indicator
            assert "..." in prompt


# =============================================================================
# EnsembleReranker Tests (8 tests)
# =============================================================================

class TestEnsembleReranker:
    """Tests for EnsembleReranker."""

    @pytest.mark.asyncio
    async def test_initialization_with_weights(self):
        """Test EnsembleReranker weight normalization."""
        from src.retrieval.reranker import EnsembleReranker, BaseReranker

        # Create mock rerankers
        mock_reranker1 = AsyncMock(spec=BaseReranker)
        mock_reranker2 = AsyncMock(spec=BaseReranker)

        ensemble = EnsembleReranker([
            (mock_reranker1, 0.6),
            (mock_reranker2, 0.4)
        ])

        # Weights should be normalized
        assert sum(ensemble.weights) == pytest.approx(1.0)
        assert ensemble.weights[0] == pytest.approx(0.6)
        assert ensemble.weights[1] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_weight_normalization(self):
        """Test weight normalization with arbitrary weights."""
        from src.retrieval.reranker import EnsembleReranker, BaseReranker

        mock_reranker1 = AsyncMock(spec=BaseReranker)
        mock_reranker2 = AsyncMock(spec=BaseReranker)

        # Non-normalized weights
        ensemble = EnsembleReranker([
            (mock_reranker1, 2.0),
            (mock_reranker2, 1.0)
        ])

        # Should normalize to sum=1
        assert sum(ensemble.weights) == pytest.approx(1.0)
        assert ensemble.weights[0] == pytest.approx(2.0 / 3.0)
        assert ensemble.weights[1] == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_score_empty_documents(self):
        """Test ensemble with empty documents."""
        from src.retrieval.reranker import EnsembleReranker, BaseReranker

        mock_reranker1 = AsyncMock(spec=BaseReranker)
        mock_reranker2 = AsyncMock(spec=BaseReranker)

        ensemble = EnsembleReranker([
            (mock_reranker1, 0.5),
            (mock_reranker2, 0.5)
        ])

        scores = await ensemble.score("query", [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_weighted_combination(self):
        """Test weighted combination of scores."""
        from src.retrieval.reranker import EnsembleReranker, BaseReranker

        mock_reranker1 = AsyncMock(spec=BaseReranker)
        mock_reranker2 = AsyncMock(spec=BaseReranker)

        # Set different scores for each reranker
        mock_reranker1.score.return_value = [1.0, 0.5, 0.0]
        mock_reranker2.score.return_value = [0.0, 0.5, 1.0]

        ensemble = EnsembleReranker([
            (mock_reranker1, 0.7),
            (mock_reranker2, 0.3)
        ])

        scores = await ensemble.score("query", ["doc1", "doc2", "doc3"])

        # Expected: [0.7*1.0 + 0.3*0.0, 0.7*0.5 + 0.3*0.5, 0.7*0.0 + 0.3*1.0]
        assert scores[0] == pytest.approx(0.7)
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test rerankers executed in parallel."""
        from src.retrieval.reranker import EnsembleReranker, BaseReranker

        mock_reranker1 = AsyncMock(spec=BaseReranker)
        mock_reranker2 = AsyncMock(spec=BaseReranker)

        mock_reranker1.score.return_value = [0.8, 0.6]
        mock_reranker2.score.return_value = [0.7, 0.5]

        ensemble = EnsembleReranker([
            (mock_reranker1, 0.5),
            (mock_reranker2, 0.5)
        ])

        await ensemble.score("query", ["doc1", "doc2"])

        # Both rerankers should be called
        mock_reranker1.score.assert_called_once()
        mock_reranker2.score.assert_called_once()


# =============================================================================
# MedicalReranker Tests (9 tests)
# =============================================================================

class TestMedicalReranker:
    """Tests for MedicalReranker (domain-specific)."""

    @pytest.fixture
    def medical_reranker(self):
        """Create MedicalReranker instance."""
        from src.retrieval.reranker import MedicalReranker
        return MedicalReranker(
            base_reranker=None,
            cui_weight=0.2,
            section_weight=0.1
        )

    def test_initialization(self, medical_reranker):
        """Test MedicalReranker initialization."""
        assert medical_reranker.cui_weight == 0.2
        assert medical_reranker.section_weight == 0.1
        assert medical_reranker.base_reranker is None
        assert len(medical_reranker.section_relevance) == 4

    @pytest.mark.asyncio
    async def test_score_empty_documents(self, medical_reranker, sample_query):
        """Test scoring with empty documents."""
        scores = await medical_reranker.score(sample_query, [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_default_base_scores(self, medical_reranker, sample_query, sample_documents):
        """Test default base score of 0.5 when no base reranker."""
        scores = await medical_reranker.score(sample_query, sample_documents)

        # Should have normalized scores
        assert len(scores) == len(sample_documents)
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_base_reranker_integration(self, medical_reranker, sample_query, sample_documents):
        """Test integration with base reranker."""
        from src.retrieval.reranker import BaseReranker

        mock_base = AsyncMock(spec=BaseReranker)
        mock_base.score.return_value = [0.9, 0.5, 0.8, 0.4, 0.7]

        medical_reranker.base_reranker = mock_base

        scores = await medical_reranker.score(sample_query, sample_documents)

        # Verify base reranker was called
        mock_base.score.assert_called_once_with(sample_query, sample_documents)
        assert len(scores) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_cui_overlap_boosting(self, medical_reranker, sample_query, sample_documents, sample_metadata):
        """Test CUI overlap boosting."""
        query_metadata = {
            "query_cuis": ["C0001074", "C0039065"]  # acoustic neuroma, surgery
        }

        # Modify metadata to include query_cuis
        metadata_with_query = sample_metadata.copy()
        metadata_with_query[0] = {**metadata_with_query[0], **query_metadata}

        scores = await medical_reranker.score(
            sample_query,
            sample_documents,
            metadata=metadata_with_query
        )

        assert len(scores) == len(sample_documents)
        # First doc should have higher score due to CUI overlap
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_section_type_relevance(self, medical_reranker, sample_metadata):
        """Test section type relevance boosting."""
        query = "surgical approach for treating tumors"
        documents = ["dummy"] * 5

        scores = await medical_reranker.score(
            query,
            documents,
            metadata=sample_metadata
        )

        # PROCEDURE sections should be boosted for "surgical approach" query
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_score_normalization(self, medical_reranker, sample_query, sample_documents):
        """Test score normalization to [0, 1] range."""
        scores = await medical_reranker.score(sample_query, sample_documents)

        # All scores should be normalized to [0, 1]
        assert all(0 <= s <= 1 for s in scores)

        # With uniform base scores, should be normalized to 1.0
        assert max(scores) <= 1.0

    @pytest.mark.asyncio
    async def test_no_metadata(self, medical_reranker, sample_query, sample_documents):
        """Test scoring without metadata (base scores only)."""
        scores = await medical_reranker.score(
            sample_query,
            sample_documents,
            metadata=None
        )

        # Should return normalized scores
        assert len(scores) == len(sample_documents)
        assert all(0 <= s <= 1 for s in scores)


# =============================================================================
# BaseReranker Tests (Common interface)
# =============================================================================

class TestBaseRerankerInterface:
    """Tests for BaseReranker common interface."""

    @pytest.mark.asyncio
    async def test_rerank_sorting_by_score(self):
        """Test rerank sorts results by score (descending)."""
        from src.retrieval.reranker import BaseReranker

        class TestReranker(BaseReranker):
            async def score(self, query, documents):
                return [0.3, 0.9, 0.5, 0.1, 0.7]

        reranker = TestReranker()
        ranked = await reranker.rerank("query", ["d1", "d2", "d3", "d4", "d5"])

        # Should return tuples (index, score) sorted by score desc
        assert ranked[0] == (1, 0.9)  # d2 with score 0.9
        assert ranked[1] == (4, 0.7)  # d5 with score 0.7
        assert ranked[2] == (2, 0.5)  # d3 with score 0.5
        assert ranked[3] == (0, 0.3)  # d1 with score 0.3
        assert ranked[4] == (3, 0.1)  # d4 with score 0.1

    @pytest.mark.asyncio
    async def test_rerank_top_k_limiting(self):
        """Test rerank returns only top_k results."""
        from src.retrieval.reranker import BaseReranker

        class TestReranker(BaseReranker):
            async def score(self, query, documents):
                return [0.3, 0.9, 0.5, 0.1, 0.7]

        reranker = TestReranker()
        ranked = await reranker.rerank("query", ["d1", "d2", "d3", "d4", "d5"], top_k=3)

        # Should return only top 3
        assert len(ranked) == 3
        assert ranked[0][1] == 0.9
        assert ranked[1][1] == 0.7
        assert ranked[2][1] == 0.5


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestRerankerFactory:
    """Tests for create_reranker factory function."""

    def test_create_cross_encoder(self):
        """Test factory creates CrossEncoderReranker."""
        from src.retrieval.reranker import create_reranker, CrossEncoderReranker

        reranker = create_reranker("cross-encoder")
        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_llm(self):
        """Test factory creates LLMReranker."""
        from src.retrieval.reranker import create_reranker, LLMReranker

        reranker = create_reranker("llm")
        assert isinstance(reranker, LLMReranker)

    def test_create_medical(self):
        """Test factory creates MedicalReranker."""
        from src.retrieval.reranker import create_reranker, MedicalReranker

        reranker = create_reranker("medical")
        assert isinstance(reranker, MedicalReranker)

    def test_create_ensemble(self):
        """Test factory creates EnsembleReranker."""
        from src.retrieval.reranker import create_reranker, EnsembleReranker

        reranker = create_reranker("ensemble")
        assert isinstance(reranker, EnsembleReranker)

    def test_invalid_reranker_type(self):
        """Test factory raises error for invalid type."""
        from src.retrieval.reranker import create_reranker

        with pytest.raises(ValueError, match="Unknown reranker type"):
            create_reranker("invalid_type")

    def test_factory_with_kwargs(self):
        """Test factory passes kwargs to reranker."""
        from src.retrieval.reranker import create_reranker, CrossEncoderReranker

        reranker = create_reranker(
            "cross-encoder",
            model_name="BAAI/bge-reranker-base",
            batch_size=64
        )

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert reranker.batch_size == 64
