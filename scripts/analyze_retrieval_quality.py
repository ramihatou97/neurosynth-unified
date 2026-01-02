#!/usr/bin/env python3
"""
NeuroSynth Retrieval Quality Benchmark
=======================================

Validates retrieval quality with current 600-token configuration.
Decision gate: PASS = current config works, FAIL = investigate further.

Usage:
    ./venv/bin/python scripts/analyze_retrieval_quality.py
"""

import asyncio
import sys
import logging
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass
from typing import List, Set

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark")


@dataclass
class TestCase:
    """A retrieval benchmark test case."""
    query: str
    expected_terms: List[str]
    domain: str  # vascular, spine, tumor, skull_base


@dataclass
class TestResult:
    """Result of a single test case."""
    query: str
    domain: str
    latency_ms: float
    result_count: int
    expected_terms: Set[str]
    found_terms: Set[str]
    passed: bool


# Neurosurgical benchmark test cases
TEST_CASES = [
    # Vascular domain
    TestCase(
        query="surgical management of petroclival meningioma",
        expected_terms=["transpetrosal", "kawase", "approach", "clivus", "trigeminal"],
        domain="skull_base"
    ),
    TestCase(
        query="flow diversion for intracranial aneurysms",
        expected_terms=["pipeline", "surpass", "endothelialization", "coil", "stent"],
        domain="vascular"
    ),
    TestCase(
        query="sylvian fissure dissection technique",
        expected_terms=["arachnoid", "MCA", "insular", "temporal", "frontal"],
        domain="vascular"
    ),
    # Spine domain
    TestCase(
        query="complications of pedicle screw fixation",
        expected_terms=["breach", "loosening", "infection", "nerve", "screw"],
        domain="spine"
    ),
    TestCase(
        query="anterior cervical discectomy and fusion",
        expected_terms=["vertebral", "disc", "cage", "plate", "decompression"],
        domain="spine"
    ),
    # Tumor domain
    TestCase(
        query="resection of vestibular schwannoma",
        expected_terms=["facial", "hearing", "translabyrinthine", "retrosigmoid", "tumor"],
        domain="tumor"
    ),
    TestCase(
        query="high-grade glioma surgical treatment",
        expected_terms=["resection", "eloquent", "mapping", "awake", "margin"],
        domain="tumor"
    ),
    # Skull base domain
    TestCase(
        query="expanded endonasal approach to skull base",
        expected_terms=["transsphenoidal", "sellar", "cavernous", "endoscope", "nasoseptal"],
        domain="skull_base"
    ),
]


async def run_benchmark() -> tuple[List[TestResult], dict]:
    """Run the complete retrieval benchmark."""
    from src.api.dependencies import ServiceContainer, get_settings

    logger.info("Initializing services...")

    settings = get_settings()
    container = ServiceContainer.get_instance()
    await container.initialize(settings)

    search_service = container.search
    if not search_service:
        logger.error("Search service not available")
        await container.shutdown()
        return [], {"error": "Search service not initialized"}

    results: List[TestResult] = []

    print("\n" + "=" * 60)
    print("  NEUROSYNTH RETRIEVAL QUALITY BENCHMARK")
    print("  Configuration: 600-token chunks")
    print("=" * 60 + "\n")

    for case in TEST_CASES:
        start = perf_counter()

        try:
            search_results = await search_service.search(
                query=case.query,
                top_k=10,
                score_threshold=0.55
            )
            latency = (perf_counter() - start) * 1000

            # Check which expected terms were found
            found = set()
            for res in search_results:
                content_lower = res.content.lower()
                for term in case.expected_terms:
                    if term.lower() in content_lower:
                        found.add(term)

            # Pass if at least 2 expected terms found
            passed = len(found) >= 2

            result = TestResult(
                query=case.query,
                domain=case.domain,
                latency_ms=latency,
                result_count=len(search_results),
                expected_terms=set(case.expected_terms),
                found_terms=found,
                passed=passed
            )
            results.append(result)

            # Print result
            status = "PASS" if passed else "FAIL"
            status_color = "\033[92m" if passed else "\033[91m"
            reset = "\033[0m"

            print(f"[{case.domain.upper():10}] {case.query[:50]}...")
            print(f"  Latency: {latency:.1f}ms | Results: {len(search_results)}")
            print(f"  Terms: {len(found)}/{len(case.expected_terms)} ({', '.join(sorted(found)) or 'none'})")
            print(f"  Status: {status_color}{status}{reset}\n")

        except Exception as e:
            logger.error(f"Test case failed: {case.query} - {e}")
            results.append(TestResult(
                query=case.query,
                domain=case.domain,
                latency_ms=0,
                result_count=0,
                expected_terms=set(case.expected_terms),
                found_terms=set(),
                passed=False
            ))

    await container.shutdown()

    # Calculate summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    avg_results = sum(r.result_count for r in results) / total if total > 0 else 0

    # Per-domain stats
    domains = set(r.domain for r in results)
    domain_stats = {}
    for domain in domains:
        domain_results = [r for r in results if r.domain == domain]
        domain_stats[domain] = {
            "total": len(domain_results),
            "passed": sum(1 for r in domain_results if r.passed),
            "avg_latency": sum(r.latency_ms for r in domain_results) / len(domain_results)
        }

    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total * 100 if total > 0 else 0,
        "avg_latency_ms": avg_latency,
        "avg_results": avg_results,
        "domain_stats": domain_stats
    }

    return results, summary


def print_summary(summary: dict) -> bool:
    """Print summary and return overall pass status."""
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    total = summary["total_tests"]
    passed = summary["passed"]
    pass_rate = summary["pass_rate"]

    print(f"\n  Total Tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {summary['failed']}")
    print(f"  Pass Rate:   {pass_rate:.1f}%")
    print(f"\n  Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"  Avg Results: {summary['avg_results']:.1f}")

    print("\n  Per-Domain Results:")
    for domain, stats in summary.get("domain_stats", {}).items():
        domain_rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {domain:12} {stats['passed']}/{stats['total']} ({domain_rate:.0f}%) @ {stats['avg_latency']:.1f}ms")

    print("\n" + "=" * 60)

    # Decision: Pass if >= 75% of tests pass
    all_pass = pass_rate >= 75

    if all_pass:
        print("\033[92m  DECISION: V2 (600-token) configuration WORKS.")
        print("  Migration to 500-token is NOT REQUIRED.\033[0m")
    else:
        print("\033[93m  DECISION: Retrieval quality below threshold.")
        print("  Consider investigating specific failing domains.\033[0m")

    print("=" * 60 + "\n")

    return all_pass


async def main():
    """Main entry point."""
    try:
        results, summary = await run_benchmark()

        if "error" in summary:
            print(f"\nError: {summary['error']}")
            sys.exit(1)

        all_pass = print_summary(summary)
        sys.exit(0 if all_pass else 1)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
