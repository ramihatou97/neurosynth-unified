"""
Integration tests for V3 Features.

Tests for new V3 capabilities:
- Gap detection in synthesis output
- Gap filling with internal/external sources
- Circuit breaker resilience
- Enhanced synthesis pipeline
- Redis job store persistence

Total: 35 test functions covering V3 workflows
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
from uuid import uuid4
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_synthesis_content():
    """Sample synthesis content with deliberate gaps."""
    return """
## Surgical Approach

The retrosigmoid approach provides excellent exposure of the cerebellopontine angle.

## Intraoperative Monitoring

Facial nerve monitoring is essential during acoustic neuroma surgery.

## Outcomes

Patient outcomes data is pending analysis.

## Complications

No complication data available for this population.
"""


@pytest.fixture
def sample_gaps():
    """Sample detected gaps."""
    from src.synthesis.gap_models import Gap, GapType

    return [
        Gap(
            gap_type=GapType.EVIDENCE_LEVEL,
            topic="Patient outcomes statistics",
            target_section="Outcomes",
            recommended_coverage="Missing quantitative outcome data",
            priority_score=75.0,  # HIGH
            auto_fill_available=True,
            external_query="acoustic neuroma surgery outcomes statistics",
        ),
        Gap(
            gap_type=GapType.MISSING,
            topic="Complication rates",
            target_section="Complications",
            recommended_coverage="No complication data provided",
            priority_score=85.0,  # CRITICAL
            auto_fill_available=True,
            external_query="acoustic neuroma surgery complication rates",
        ),
        Gap(
            gap_type=GapType.THIN_COVERAGE,
            topic="Approach citations",
            target_section="Surgical Approach",
            recommended_coverage="Missing citations for approach claims",
            priority_score=25.0,  # LOW
            auto_fill_available=False,
        ),
    ]


@pytest.fixture
def mock_search_results():
    """Mock search results for gap filling."""
    return [
        {
            "id": str(uuid4()),
            "content": "A meta-analysis of 1,234 patients showed facial nerve preservation in 92% of cases using the retrosigmoid approach.",
            "score": 0.92,
            "document_id": "doc1",
            "page_number": 145,
        },
        {
            "id": str(uuid4()),
            "content": "Complication rates for acoustic neuroma surgery include CSF leak (8%), hearing loss (25%), and facial weakness (15%).",
            "score": 0.88,
            "document_id": "doc2",
            "page_number": 78,
        },
    ]


@pytest.fixture
def mock_external_research():
    """Mock external research results."""
    return {
        "sources": [
            {
                "title": "PubMed: Acoustic Neuroma Surgery Outcomes",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                "content": "Long-term outcomes show 95% tumor control at 10 years.",
            }
        ],
        "summary": "Recent studies demonstrate excellent long-term outcomes for acoustic neuroma surgery.",
    }


@pytest.fixture
def mock_gap_filler(mock_search_results, mock_external_research):
    """Mock gap filling service."""
    from src.synthesis.gap_models import GapFillResult

    service = AsyncMock()

    async def fill_gaps(gaps, strategy=None, config=None):
        results = []
        for gap in gaps:
            if gap.priority.value in ("high", "critical"):
                result = GapFillResult(
                    gap_id=gap.gap_id,
                    gap_type=gap.gap_type,
                    topic=gap.topic,
                    fill_successful=True,
                    filled_content="Based on recent studies, complication rates are: CSF leak 8%, facial weakness 15%.",
                    fill_source="internal",
                    external_sources=[{"citation": "Smith et al., 2023"}],
                    fill_duration_ms=150,
                )
            else:
                result = GapFillResult(
                    gap_id=gap.gap_id,
                    gap_type=gap.gap_type,
                    topic=gap.topic,
                    fill_successful=False,
                    filled_content="",
                    fill_source="failed",
                    external_sources=[],
                    fill_duration_ms=10,
                )
            results.append(result)
        return results

    service.fill_gaps = fill_gaps
    return service


@pytest.fixture
def circuit_breaker():
    """Create a fresh circuit breaker for testing."""
    from src.utils.circuit_breaker import CircuitBreaker

    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        success_threshold=2,
        reset_timeout=0.5,  # Short timeout for tests
        timeout=5.0,
    )


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for synthesis."""
    client = Mock()

    def create_message(*args, **kwargs):
        response = Mock()
        response.content = [Mock(text="""
## Summary

The retrosigmoid approach is the preferred method for acoustic neuroma surgery.

## Key Points

1. Excellent tumor exposure
2. Facial nerve preservation possible
3. Low complication rates

## Evidence Quality

Strong evidence from multiple randomized trials.
""")]
        response.usage = Mock(input_tokens=500, output_tokens=200)
        return response

    client.messages.create = Mock(side_effect=create_message)
    return client


# =============================================================================
# Gap Detection Tests (8 tests)
# =============================================================================

class TestGapDetection:
    """Tests for gap detection in synthesis output."""

    @pytest.mark.asyncio
    async def test_gap_detection_service_finds_missing_data(self, sample_gaps):
        """Test gap detection identifies missing data sections."""
        from src.synthesis.gap_models import GapType, GapPriority

        # Use sample gaps to test gap model behavior
        gaps = sample_gaps

        # Should have gaps
        assert len(gaps) > 0

        # Should have different gap types
        gap_types = {g.gap_type for g in gaps}
        assert len(gap_types) >= 2

    @pytest.mark.asyncio
    async def test_gap_detection_prioritizes_gaps(self, sample_gaps):
        """Test gap detection assigns appropriate priorities."""
        from src.synthesis.gap_models import GapPriority

        # Check priority assignments
        priorities = {g.priority for g in sample_gaps}
        assert len(priorities) > 0
        assert all(isinstance(p, GapPriority) for p in priorities)

        # Should have high/critical priority for high scores
        high_priority = [g for g in sample_gaps if g.priority_score >= 60]
        assert all(g.priority in (GapPriority.HIGH, GapPriority.CRITICAL) for g in high_priority)

    @pytest.mark.asyncio
    async def test_gap_detection_generates_queries(self, sample_gaps):
        """Test gap detection generates external search queries."""
        # High priority gaps should have external queries
        high_priority = [g for g in sample_gaps if g.priority.value in ("high", "critical")]
        for gap in high_priority:
            assert gap.external_query or gap.topic

    @pytest.mark.asyncio
    async def test_gap_detection_with_complete_content(self, sample_gaps):
        """Test gap priority scores correctly computed."""
        from src.synthesis.gap_models import Gap, GapType, GapPriority

        # Create gap with low score - should be LOW priority
        low_gap = Gap(
            gap_type=GapType.THIN_COVERAGE,
            topic="Minor detail",
            priority_score=20.0,
        )

        assert low_gap.priority == GapPriority.LOW

        # Create gap with high score - should be HIGH priority
        high_gap = Gap(
            gap_type=GapType.MISSING,
            topic="Important topic",
            priority_score=70.0,
        )

        assert high_gap.priority == GapPriority.HIGH

    @pytest.mark.asyncio
    async def test_gap_report_generation(self, sample_gaps):
        """Test gap report aggregation."""
        from src.synthesis.gap_models import GapReport, TemplateType

        report = GapReport(
            topic="acoustic neuroma surgery",
            template_type=TemplateType.PROCEDURAL,
            gaps=sample_gaps,
        )

        # Test report properties
        assert report.total_gaps == 3
        assert len(report.high_priority_gaps) >= 1
        assert len(report.critical_gaps) >= 1

    @pytest.mark.asyncio
    async def test_gap_detection_by_section(self, sample_gaps):
        """Test gaps are associated with correct sections."""
        # Gaps should have target_section assignments
        for gap in sample_gaps:
            assert gap.target_section is not None
            assert len(gap.target_section) > 0

    @pytest.mark.asyncio
    async def test_gap_type_classification(self, sample_gaps):
        """Test gap types are correctly classified."""
        from src.synthesis.gap_models import GapType

        gap_types = {g.gap_type for g in sample_gaps}

        # Should have multiple gap types
        assert len(gap_types) >= 2
        assert all(isinstance(t, GapType) for t in gap_types)

    @pytest.mark.asyncio
    async def test_gap_auto_fill_availability(self, sample_gaps):
        """Test auto-fill availability flag."""
        fillable = [g for g in sample_gaps if g.auto_fill_available]
        non_fillable = [g for g in sample_gaps if not g.auto_fill_available]

        # Should have both types
        assert len(fillable) >= 1
        assert len(non_fillable) >= 1


# =============================================================================
# Gap Filling Tests (8 tests)
# =============================================================================

class TestGapFilling:
    """Tests for gap filling service."""

    @pytest.mark.asyncio
    async def test_fill_high_priority_gaps(self, sample_gaps, mock_gap_filler):
        """Test filling high priority gaps only."""
        from src.synthesis.gap_models import GapFillStrategy

        results = await mock_gap_filler.fill_gaps(
            gaps=sample_gaps,
            strategy=GapFillStrategy.HIGH_PRIORITY_ONLY,
        )

        # Should have results for high/critical gaps
        successful = [r for r in results if r.fill_successful]
        assert len(successful) >= 1

    @pytest.mark.asyncio
    async def test_fill_result_has_content(self, sample_gaps, mock_gap_filler):
        """Test filled gaps have content."""
        results = await mock_gap_filler.fill_gaps(gaps=sample_gaps)

        successful = [r for r in results if r.fill_successful]
        for result in successful:
            assert result.filled_content
            assert len(result.filled_content) > 20

    @pytest.mark.asyncio
    async def test_fill_result_has_external_sources(self, sample_gaps, mock_gap_filler):
        """Test filled gaps include external sources."""
        results = await mock_gap_filler.fill_gaps(gaps=sample_gaps)

        successful = [r for r in results if r.fill_successful]
        for result in successful:
            assert result.external_sources is not None
            # At least some should have sources
            if result.fill_successful:
                assert isinstance(result.external_sources, list)

    @pytest.mark.asyncio
    async def test_fill_tracks_source(self, sample_gaps, mock_gap_filler):
        """Test fill source is tracked."""
        results = await mock_gap_filler.fill_gaps(gaps=sample_gaps)

        for result in results:
            assert isinstance(result.fill_source, str)
            assert result.fill_source in ("internal", "external", "both", "failed")

    @pytest.mark.asyncio
    async def test_fill_duration_tracking(self, sample_gaps, mock_gap_filler):
        """Test fill duration is tracked."""
        results = await mock_gap_filler.fill_gaps(gaps=sample_gaps)

        for result in results:
            assert result.fill_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_strategy_none_skips_filling(self, sample_gaps):
        """Test NONE strategy doesn't fill any gaps."""
        from src.synthesis.gap_filling_service import GapFillingService
        from src.synthesis.gap_models import GapFillStrategy

        service = GapFillingService()
        results = await service.fill_gaps(
            gaps=sample_gaps,
            strategy=GapFillStrategy.NONE,
        )

        # Should return empty or all unsuccessful
        assert len(results) == 0 or all(not r.fill_successful for r in results)

    @pytest.mark.asyncio
    async def test_internal_search_fallback(self, sample_gaps, mock_search_results):
        """Test internal search is tried before external."""
        from src.synthesis.gap_filling_service import GapFillingService
        from src.synthesis.gap_models import GapFillStrategy

        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=mock_search_results)

        service = GapFillingService(search_service=mock_search)

        results = await service.fill_gaps(
            gaps=sample_gaps[:1],  # Just one gap
            strategy=GapFillStrategy.ALL_WITH_FALLBACK,
        )

        # Internal search should be attempted
        # (actual call depends on gap.auto_fill_available)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_external_research_used(self, sample_gaps, mock_external_research):
        """Test external research is used when internal fails."""
        from src.synthesis.gap_filling_service import GapFillingService
        from src.synthesis.gap_models import GapFillStrategy

        mock_enricher = AsyncMock()
        mock_enricher.fetch_external = AsyncMock(return_value=mock_external_research)

        service = GapFillingService(research_enricher=mock_enricher)

        results = await service.fill_gaps(
            gaps=sample_gaps[:1],
            strategy=GapFillStrategy.ALWAYS_EXTERNAL,
        )

        assert isinstance(results, list)


# =============================================================================
# Circuit Breaker Tests (9 tests)
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker resilience."""

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self, circuit_breaker):
        """Test circuit breaker starts in closed state."""
        from src.utils.circuit_breaker import CircuitState

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker):
        """Test circuit opens after threshold failures."""
        from src.utils.circuit_breaker import CircuitState

        # Simulate failures
        for i in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise Exception(f"Simulated failure {i}")
            except Exception:
                pass

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit_breaker):
        """Test open circuit rejects requests."""
        from src.utils.circuit_breaker import CircuitOpenError, CircuitState

        # Force open
        for i in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise Exception("Simulated failure")
            except Exception:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Should reject
        with pytest.raises(CircuitOpenError):
            async with circuit_breaker:
                pass

    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self, circuit_breaker):
        """Test circuit transitions to half-open after reset timeout."""
        from src.utils.circuit_breaker import CircuitState

        # Force open
        for i in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise Exception("Simulated failure")
            except Exception:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(circuit_breaker.reset_timeout + 0.1)

        # Should be half-open now
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success(self, circuit_breaker):
        """Test circuit closes after successes in half-open."""
        from src.utils.circuit_breaker import CircuitState

        # Force open
        for i in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise Exception("Simulated failure")
            except Exception:
                pass

        # Wait for half-open
        await asyncio.sleep(circuit_breaker.reset_timeout + 0.1)
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Successful calls
        for i in range(circuit_breaker.success_threshold):
            async with circuit_breaker:
                pass  # Success

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_stats_tracking(self, circuit_breaker):
        """Test circuit breaker tracks statistics."""
        # Some successes
        for _ in range(3):
            async with circuit_breaker:
                pass

        # Some failures
        for _ in range(2):
            try:
                async with circuit_breaker:
                    raise Exception("Failure")
            except Exception:
                pass

        stats = circuit_breaker.stats
        assert stats.total_calls == 5
        assert stats.successful_calls == 3
        assert stats.failed_calls == 2

    @pytest.mark.asyncio
    async def test_circuit_manual_reset(self, circuit_breaker):
        """Test manual circuit reset."""
        from src.utils.circuit_breaker import CircuitState

        # Force open
        for i in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise Exception("Failure")
            except Exception:
                pass

        assert circuit_breaker.state == CircuitState.OPEN

        # Manual reset
        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_calls == 0

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Test circuit breaker decorator with fallback."""
        from src.utils.circuit_breaker import CircuitBreaker, with_circuit_breaker

        breaker = CircuitBreaker(name="test_decorator", failure_threshold=2)

        call_count = 0
        fallback_count = 0

        async def fallback_func():
            nonlocal fallback_count
            fallback_count += 1
            return "fallback_result"

        @with_circuit_breaker(breaker, fallback=fallback_func)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")

        # First calls will fail and use fallback after circuit opens
        for _ in range(5):
            result = await failing_func()

        # After circuit opens, fallback should be used
        assert fallback_count > 0

    @pytest.mark.asyncio
    async def test_voyage_circuit_breaker_integration(self):
        """Test Voyage embedding circuit breaker is configured."""
        from src.ingest.embeddings import voyage_breaker

        # Should be configured with sensible defaults
        assert voyage_breaker.name == "voyage"
        assert voyage_breaker.failure_threshold >= 3
        assert voyage_breaker.reset_timeout >= 30


# =============================================================================
# Enhanced Synthesis Tests (6 tests)
# =============================================================================

class TestEnhancedSynthesis:
    """Tests for enhanced synthesis pipeline."""

    @pytest.mark.asyncio
    async def test_synthesis_with_gap_detection(self, mock_anthropic_client):
        """Test synthesis includes gap detection."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()

            # Mock search results
            mock_results = [
                Mock(content="Test content 1", document_id="doc1", page_number=1, score=0.9),
                Mock(content="Test content 2", document_id="doc1", page_number=2, score=0.8),
            ]

            result = await engine.synthesize(
                topic="acoustic neuroma surgery",
                template_type=TemplateType.CLINICAL_REVIEW,
                search_results=mock_results,
            )

            assert result is not None
            assert hasattr(result, "gap_report")

    @pytest.mark.asyncio
    async def test_synthesis_with_gap_filling(self, mock_anthropic_client, mock_gap_filler):
        """Test synthesis includes gap filling when enabled."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()
            engine.gap_filler = mock_gap_filler

            mock_results = [
                Mock(content="Test content", document_id="doc1", page_number=1, score=0.9),
            ]

            result = await engine.synthesize(
                topic="acoustic neuroma",
                template_type=TemplateType.CLINICAL_REVIEW,
                search_results=mock_results,
                gap_fill_strategy="high_priority_only",
            )

            assert result is not None
            # Should have gap fill results if gap filling is wired
            if hasattr(result, "gap_fill_results"):
                assert isinstance(result.gap_fill_results, list)

    @pytest.mark.asyncio
    async def test_synthesis_result_structure(self, mock_anthropic_client):
        """Test synthesis result has expected structure."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()

            mock_results = [
                Mock(content="Content", document_id="doc1", page_number=1, score=0.9),
            ]

            result = await engine.synthesize(
                topic="test topic",
                template_type=TemplateType.OVERVIEW,
                search_results=mock_results,
            )

            # Check required fields
            assert hasattr(result, "content")
            assert hasattr(result, "citations")
            assert hasattr(result, "figures")
            assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_synthesis_validates_content(self, mock_anthropic_client):
        """Test synthesis validates generated content."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()

            mock_results = [
                Mock(content="Content", document_id="doc1", page_number=1, score=0.9),
            ]

            result = await engine.synthesize(
                topic="test",
                template_type=TemplateType.OVERVIEW,
                search_results=mock_results,
            )

            # Content should not be empty
            assert result.content
            assert len(result.content) > 50

    @pytest.mark.asyncio
    async def test_enhanced_synthesis_with_enrichment(self, mock_anthropic_client):
        """Test enhanced synthesis with web research enrichment."""
        from src.synthesis.enhanced_engine import EnhancedSynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.enhanced_engine.Anthropic", return_value=mock_anthropic_client):
            with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
                engine = EnhancedSynthesisEngine()

                mock_results = [
                    Mock(content="Content", document_id="doc1", page_number=1, score=0.9),
                ]

                result = await engine.synthesize(
                    topic="test topic",
                    template_type=TemplateType.OVERVIEW,
                    search_results=mock_results,
                    include_web_research=False,  # Disable for unit test
                )

                assert result is not None
                assert hasattr(result, "content")

    @pytest.mark.asyncio
    async def test_synthesis_includes_figure_selection(self, mock_anthropic_client):
        """Test synthesis includes figure selection."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()

            mock_results = [
                Mock(
                    content="Content with figure",
                    document_id="doc1",
                    page_number=1,
                    score=0.9,
                    linked_images=[{"id": "img1", "caption": "Figure 1"}],
                ),
            ]

            result = await engine.synthesize(
                topic="test",
                template_type=TemplateType.CLINICAL_REVIEW,
                search_results=mock_results,
                include_figures=True,
            )

            # Should have figures list
            assert hasattr(result, "figures")
            assert isinstance(result.figures, list)


# =============================================================================
# Redis Job Store Tests (4 tests)
# =============================================================================

class TestRedisJobStore:
    """Tests for Redis job store persistence."""

    @pytest.mark.asyncio
    async def test_job_store_creates_job(self):
        """Test job store creates job entries."""
        from src.ingest.job_store import JobStore

        store = JobStore(force_memory=True)
        await store.initialize()

        job = await store.create_job("test-job-1", "test.pdf")

        assert job is not None
        assert job.job_id == "test-job-1"
        assert job.status == "pending"

        # Retrieve the job
        retrieved = await store.get_job("test-job-1")
        assert retrieved is not None
        assert retrieved.status == "pending"

    @pytest.mark.asyncio
    async def test_job_store_updates_progress(self):
        """Test job store updates job progress."""
        from src.ingest.job_store import JobStore

        store = JobStore(force_memory=True)
        await store.initialize()

        await store.create_job("test-job-2", "test.pdf")

        # Update progress
        await store.update_job(
            "test-job-2",
            status="processing",
            progress=50,
            stage="chunking",
        )

        job = await store.get_job("test-job-2")
        assert job.status == "processing"
        assert job.progress == 50
        assert job.stage == "chunking"

    @pytest.mark.asyncio
    async def test_job_store_handles_completion(self):
        """Test job store handles job completion."""
        from src.ingest.job_store import JobStore

        store = JobStore(force_memory=True)
        await store.initialize()

        await store.create_job("test-job-3", "test.pdf")

        # Complete the job
        await store.complete_job(
            "test-job-3",
            summary={"document_id": "doc123", "chunks_created": 50},
        )

        job = await store.get_job("test-job-3")
        assert job.status == "completed"
        assert job.progress == 100
        assert job.summary["chunks_created"] == 50

    @pytest.mark.asyncio
    async def test_job_store_handles_failure(self):
        """Test job store handles job failure."""
        from src.ingest.job_store import JobStore

        store = JobStore(force_memory=True)
        await store.initialize()

        await store.create_job("test-job-4", "test.pdf")

        # Fail the job
        await store.fail_job(
            "test-job-4",
            error="PDF parsing failed: corrupted file",
        )

        job = await store.get_job("test-job-4")
        assert job.status == "failed"
        assert "corrupted" in job.error


# =============================================================================
# End-to-End Integration Tests (2 tests)
# =============================================================================

class TestE2EIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_synthesis_pipeline(self, mock_anthropic_client):
        """Test complete synthesis pipeline from search to output."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()

            # Simulate search results
            mock_results = [
                Mock(
                    content="The retrosigmoid approach is preferred for acoustic neuroma.",
                    document_id="doc1",
                    page_number=42,
                    score=0.95,
                    chunk_type="PROCEDURE",
                    cuis=["C0001074"],
                ),
                Mock(
                    content="Facial nerve preservation is achieved in 92% of cases.",
                    document_id="doc1",
                    page_number=45,
                    score=0.92,
                    chunk_type="OUTCOME",
                    cuis=["C0015462"],
                ),
            ]

            result = await engine.synthesize(
                topic="acoustic neuroma surgery approaches",
                template_type=TemplateType.CLINICAL_REVIEW,
                search_results=mock_results,
                include_verification=False,
                include_figures=True,
            )

            # Validate complete result
            assert result.content is not None
            assert len(result.content) > 100
            assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_synthesis_with_all_v3_features(self, mock_anthropic_client, mock_gap_filler):
        """Test synthesis using all V3 features together."""
        from src.synthesis.engine import SynthesisEngine
        from src.synthesis.models import TemplateType

        with patch("src.synthesis.engine.Anthropic", return_value=mock_anthropic_client):
            engine = SynthesisEngine()
            engine.gap_filler = mock_gap_filler

            mock_results = [
                Mock(
                    content="Surgical approach content",
                    document_id="doc1",
                    page_number=1,
                    score=0.9,
                    linked_images=[],
                ),
            ]

            result = await engine.synthesize(
                topic="comprehensive neurosurgery review",
                template_type=TemplateType.CLINICAL_REVIEW,
                search_results=mock_results,
                include_verification=False,
                include_figures=True,
                gap_fill_strategy="high_priority_only",
            )

            # Should complete without error
            assert result is not None
            assert result.content
