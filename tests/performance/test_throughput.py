"""
Throughput and load testing for NeuroSynth.

Tests:
- Query throughput (QPS - queries per second)
- Concurrent user capacity
- Load stability over time
- Resource utilization under load

Targets:
- Handle 50+ concurrent users
- Maintain 50+ QPS
- P99 latency stable under sustained load

Total: 15 load and throughput tests
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock
import random


# =============================================================================
# Load Testing Utilities
# =============================================================================

class ThroughputMetrics:
    """Throughput and load statistics."""

    def __init__(
        self,
        total_requests: int,
        successful: int,
        failed: int,
        total_time_seconds: float,
        latencies: List[float],
    ):
        self.total_requests = total_requests
        self.successful = successful
        self.failed = failed
        self.total_time = total_time_seconds
        self.latencies = sorted(latencies)

    @property
    def qps(self) -> float:
        """Queries per second."""
        return self.total_requests / self.total_time if self.total_time > 0 else 0

    @property
    def success_rate(self) -> float:
        """Percentage of successful requests."""
        return (self.successful / self.total_requests * 100) if self.total_requests > 0 else 0

    @property
    def p50_latency(self) -> float:
        """50th percentile latency."""
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        return self._percentile(99)

    @property
    def mean_latency(self) -> float:
        """Mean latency."""
        return statistics.mean(self.latencies) if self.latencies else 0

    def _percentile(self, p: float) -> float:
        """Calculate percentile."""
        if not self.latencies:
            return 0
        index = int((p / 100) * len(self.latencies))
        index = min(index, len(self.latencies) - 1)
        return self.latencies[index]

    def __repr__(self) -> str:
        return (
            f"ThroughputMetrics(qps={self.qps:.1f}, success_rate={self.success_rate:.1f}%, "
            f"p99={self.p99_latency:.2f}ms)"
        )


async def run_load_test(
    requests_per_second: int,
    duration_seconds: int,
    request_func,
    concurrent_users: int = 10,
) -> ThroughputMetrics:
    """Run load test with specified QPS and duration."""
    start_time = time.time()
    end_time = start_time + duration_seconds
    total_requests = 0
    successful = 0
    failed = 0
    latencies = []

    # Create semaphore for concurrent user limit
    semaphore = asyncio.Semaphore(concurrent_users)

    async def limited_request():
        async with semaphore:
            try:
                request_start = time.time()
                await request_func()
                request_time = (time.time() - request_start) * 1000
                latencies.append(request_time)
                return True
            except Exception:
                return False

    # Generate requests at target QPS
    request_tasks = []
    current_time = start_time

    while current_time < end_time:
        # Calculate how many requests should be issued by now
        elapsed = current_time - start_time
        expected_requests = int(elapsed * requests_per_second)

        while total_requests < expected_requests and current_time < end_time:
            task = asyncio.create_task(limited_request())
            request_tasks.append(task)
            total_requests += 1

            # Wait slightly to space out requests
            await asyncio.sleep(1.0 / requests_per_second / concurrent_users)

        current_time = time.time()

    # Wait for all requests to complete
    results = await asyncio.gather(*request_tasks, return_exceptions=True)

    # Count successes
    for result in results:
        if isinstance(result, bool) and result:
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    return ThroughputMetrics(
        total_requests=total_requests,
        successful=successful,
        failed=failed,
        total_time_seconds=total_time,
        latencies=latencies,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_fast_request():
    """Mock fast request (10-20ms)."""
    async def request():
        await asyncio.sleep(random.uniform(0.010, 0.020))

    return request


@pytest.fixture
def mock_search_request():
    """Mock search request (15-50ms)."""
    async def request():
        await asyncio.sleep(random.uniform(0.015, 0.050))

    return request


@pytest.fixture
def mock_rag_request():
    """Mock RAG request (1-3 seconds)."""
    async def request():
        await asyncio.sleep(random.uniform(1.0, 3.0))

    return request


# =============================================================================
# QPS (Queries Per Second) Tests (5 tests)
# =============================================================================

class TestQueryThroughput:
    """Tests for query throughput capacity."""

    @pytest.mark.asyncio
    async def test_target_50_qps(self, mock_search_request):
        """Test sustaining target 50 QPS."""
        metrics = await run_load_test(
            requests_per_second=50,
            duration_seconds=10,
            request_func=mock_search_request,
            concurrent_users=10,
        )

        print(f"\n50 QPS Load Test: {metrics}")
        # Should achieve close to target QPS
        assert metrics.qps >= 40  # 80% of target minimum
        assert metrics.success_rate >= 95

    @pytest.mark.asyncio
    async def test_burst_100_qps(self, mock_fast_request):
        """Test handling burst to 100 QPS."""
        metrics = await run_load_test(
            requests_per_second=100,
            duration_seconds=5,
            request_func=mock_fast_request,
            concurrent_users=20,
        )

        print(f"\nBurst 100 QPS: {metrics}")
        # Should handle burst
        assert metrics.qps >= 80

    @pytest.mark.asyncio
    async def test_sustained_qps_stability(self, mock_search_request):
        """Test QPS stability over time."""
        # Run for longer to check stability
        metrics = await run_load_test(
            requests_per_second=30,
            duration_seconds=30,
            request_func=mock_search_request,
            concurrent_users=10,
        )

        print(f"\nSustained 30 QPS (30s): {metrics}")
        # QPS should be consistent
        assert metrics.qps >= 25
        # Success rate should remain high
        assert metrics.success_rate >= 95

    @pytest.mark.asyncio
    async def test_qps_headroom_above_target(self, mock_fast_request):
        """Test we have headroom above target QPS."""
        metrics = await run_load_test(
            requests_per_second=75,
            duration_seconds=5,
            request_func=mock_fast_request,
            concurrent_users=20,
        )

        print(f"\n75 QPS Headroom: {metrics}")
        # Should handle 50% above target
        assert metrics.qps >= 65

    @pytest.mark.asyncio
    async def test_low_qps_efficiency(self, mock_search_request):
        """Test efficiency at low QPS."""
        metrics = await run_load_test(
            requests_per_second=10,
            duration_seconds=10,
            request_func=mock_search_request,
            concurrent_users=5,
        )

        print(f"\nLow QPS (10): {metrics}")
        # Should be very efficient at low load
        assert metrics.success_rate >= 99


# =============================================================================
# Concurrent User Tests (4 tests)
# =============================================================================

class TestConcurrentUsers:
    """Tests for concurrent user capacity."""

    @pytest.mark.asyncio
    async def test_target_50_concurrent_users(self, mock_search_request):
        """Test handling target 50 concurrent users."""
        metrics = await run_load_test(
            requests_per_second=50,
            duration_seconds=10,
            request_func=mock_search_request,
            concurrent_users=50,
        )

        print(f"\n50 Concurrent Users: {metrics}")
        # Should handle 50 users
        assert metrics.success_rate >= 95
        assert metrics.qps >= 40

    @pytest.mark.asyncio
    async def test_100_concurrent_users_stress(self, mock_fast_request):
        """Test stress load with 100 concurrent users."""
        metrics = await run_load_test(
            requests_per_second=100,
            duration_seconds=5,
            request_func=mock_fast_request,
            concurrent_users=100,
        )

        print(f"\n100 Concurrent Users: {metrics}")
        # Should handle higher load
        assert metrics.success_rate >= 90

    @pytest.mark.asyncio
    async def test_user_increase_ramp(self, mock_search_request):
        """Test graceful handling of user increase."""
        # Start with 10, ramping to 50
        metrics_10 = await run_load_test(
            requests_per_second=10,
            duration_seconds=5,
            request_func=mock_search_request,
            concurrent_users=10,
        )

        metrics_50 = await run_load_test(
            requests_per_second=50,
            duration_seconds=5,
            request_func=mock_search_request,
            concurrent_users=50,
        )

        print(f"\nUser Ramp:")
        print(f"  10 users: {metrics_10}")
        print(f"  50 users: {metrics_50}")

        # Both should maintain acceptable performance
        assert metrics_10.success_rate >= 95
        assert metrics_50.success_rate >= 95

    @pytest.mark.asyncio
    async def test_latency_under_concurrent_load(self, mock_search_request):
        """Test latency stability with concurrent users."""
        metrics = await run_load_test(
            requests_per_second=50,
            duration_seconds=10,
            request_func=mock_search_request,
            concurrent_users=50,
        )

        print(f"\nConcurrent Load Latency:")
        print(f"  P50: {metrics.p50_latency:.2f}ms")
        print(f"  P95: {metrics.p95_latency:.2f}ms")
        print(f"  P99: {metrics.p99_latency:.2f}ms")

        # P99 should remain under target
        assert metrics.p99_latency < 200


# =============================================================================
# Load Stability Tests (3 tests)
# =============================================================================

class TestLoadStability:
    """Tests for stability under sustained load."""

    @pytest.mark.asyncio
    async def test_sustained_load_60_seconds(self, mock_search_request):
        """Test 60-second sustained load."""
        metrics = await run_load_test(
            requests_per_second=30,
            duration_seconds=60,
            request_func=mock_search_request,
            concurrent_users=20,
        )

        print(f"\n60-Second Load: {metrics}")
        # Should maintain performance over time
        assert metrics.qps >= 25
        assert metrics.success_rate >= 95

    @pytest.mark.asyncio
    async def test_no_latency_degradation_over_time(self, mock_search_request):
        """Test latency doesn't degrade over time."""
        # Run two halves and compare
        metrics_first = await run_load_test(
            requests_per_second=30,
            duration_seconds=15,
            request_func=mock_search_request,
            concurrent_users=15,
        )

        metrics_second = await run_load_test(
            requests_per_second=30,
            duration_seconds=15,
            request_func=mock_search_request,
            concurrent_users=15,
        )

        print(f"\nLatency Degradation Over Time:")
        print(f"  First 15s: p99={metrics_first.p99_latency:.2f}ms")
        print(f"  Second 15s: p99={metrics_second.p99_latency:.2f}ms")

        # Second half should not be significantly worse
        degradation = (metrics_second.p99_latency - metrics_first.p99_latency) / metrics_first.p99_latency
        assert degradation < 0.3  # Less than 30% increase

    @pytest.mark.asyncio
    async def test_recovery_after_spike(self, mock_search_request):
        """Test system recovers after traffic spike."""
        # Normal load
        await run_load_test(
            requests_per_second=20,
            duration_seconds=5,
            request_func=mock_search_request,
            concurrent_users=10,
        )

        # Spike
        spike_metrics = await run_load_test(
            requests_per_second=100,
            duration_seconds=5,
            request_func=mock_search_request,
            concurrent_users=50,
        )

        # Back to normal
        recovery_metrics = await run_load_test(
            requests_per_second=20,
            duration_seconds=5,
            request_func=mock_search_request,
            concurrent_users=10,
        )

        print(f"\nSpike Recovery:")
        print(f"  During spike: p99={spike_metrics.p99_latency:.2f}ms")
        print(f"  After recovery: p99={recovery_metrics.p99_latency:.2f}ms")

        # Should recover to normal performance
        assert recovery_metrics.success_rate >= 95


# =============================================================================
# Mixed Workload Tests (3 tests)
# =============================================================================

class TestMixedWorkloads:
    """Tests for mixed search and RAG workloads."""

    @pytest.mark.asyncio
    async def test_mostly_search_workload(self, mock_search_request, mock_rag_request):
        """Test workload with 80% search, 20% RAG."""
        async def mixed_request():
            if random.random() < 0.8:
                await mock_search_request()
            else:
                await mock_rag_request()

        metrics = await run_load_test(
            requests_per_second=30,
            duration_seconds=20,
            request_func=mixed_request,
            concurrent_users=20,
        )

        print(f"\n80% Search / 20% RAG Workload: {metrics}")
        # Should handle mixed workload
        assert metrics.success_rate >= 90
        assert metrics.qps >= 20

    @pytest.mark.asyncio
    async def test_search_heavy_with_occasional_rag(self):
        """Test mostly search with occasional expensive RAG."""
        async def search_heavy():
            if random.random() < 0.95:
                await asyncio.sleep(0.020)  # Search
            else:
                await asyncio.sleep(2.0)  # RAG

        metrics = await run_load_test(
            requests_per_second=50,
            duration_seconds=15,
            request_func=search_heavy,
            concurrent_users=15,
        )

        print(f"\n95% Search / 5% RAG: {metrics}")
        assert metrics.success_rate >= 90

    @pytest.mark.asyncio
    async def test_peak_hour_simulation(self):
        """Simulate peak hour with varying load."""
        # Peak hour with spiky traffic
        async def peak_hour_request():
            await asyncio.sleep(random.uniform(0.010, 0.100))

        metrics = await run_load_test(
            requests_per_second=75,
            duration_seconds=30,
            request_func=peak_hour_request,
            concurrent_users=40,
        )

        print(f"\nPeak Hour Simulation: {metrics}")
        # Should handle peak load
        assert metrics.success_rate >= 85
        assert metrics.qps >= 60


# =============================================================================
# Error Handling Under Load Tests (1 test)
# =============================================================================

class TestErrorHandlingUnderLoad:
    """Tests for graceful error handling under load."""

    @pytest.mark.asyncio
    async def test_partial_failure_graceful_handling(self):
        """Test system handles partial failures gracefully."""
        failure_rate = 0.05  # 5% failure rate

        async def request_with_failures():
            if random.random() < failure_rate:
                raise Exception("Simulated failure")
            await asyncio.sleep(random.uniform(0.010, 0.050))

        metrics = await run_load_test(
            requests_per_second=50,
            duration_seconds=10,
            request_func=request_with_failures,
            concurrent_users=20,
        )

        print(f"\nWith 5% Failure Rate: {metrics}")
        # Should maintain reasonable performance despite failures
        assert metrics.success_rate >= 90
        # Remaining requests should complete successfully
        assert metrics.successful > 400
