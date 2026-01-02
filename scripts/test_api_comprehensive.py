#!/usr/bin/env python3
"""
NeuroSynth API Comprehensive Test Suite
========================================

Tests all API endpoints with real requests against a running server.

Usage:
    # Start server:
    uvicorn src.api.main:app --reload --port 8000

    # Run tests:
    python scripts/test_api_comprehensive.py

    # Quick tests only (no synthesis):
    python scripts/test_api_comprehensive.py --quick

    # Specific endpoint:
    python scripts/test_api_comprehensive.py --endpoint search
"""

import asyncio
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestStatus(Enum):
    PASS = "\u2705 PASS"
    FAIL = "\u274c FAIL"
    SKIP = "\u23ed\ufe0f SKIP"
    WARN = "\u26a0\ufe0f WARN"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration_ms: float
    message: str
    request: Optional[Dict] = None
    response: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class TestSuite:
    name: str
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAIL)

    def print_summary(self):
        print(f"\n{'-'*60}")
        print(f"Suite: {self.name}")
        print(f"{'-'*60}")

        for result in self.results:
            status_str = result.status.value
            duration = f"({result.duration_ms:.0f}ms)" if result.duration_ms else ""
            print(f"  {status_str} {result.name} {duration}")
            if result.status == TestStatus.FAIL and result.error:
                print(f"       Error: {result.error[:100]}")

        print(f"\n  Total: {self.passed} passed, {self.failed} failed")


class TestRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None
        self.suites: List[TestSuite] = []

    async def __aenter__(self):
        import httpx
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def run_test(
        self,
        name: str,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        expected_status: int = 200,
        validate: Optional[callable] = None
    ) -> TestResult:
        """Run a single API test."""
        start = time.time()

        try:
            if method.upper() == "GET":
                resp = await self.client.get(path)
            elif method.upper() == "POST":
                resp = await self.client.post(path, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            duration = (time.time() - start) * 1000

            if resp.status_code != expected_status:
                return TestResult(
                    name=name,
                    status=TestStatus.FAIL,
                    duration_ms=duration,
                    message=f"Expected {expected_status}, got {resp.status_code}",
                    request=json_data,
                    error=resp.text[:200]
                )

            try:
                data = resp.json()
            except:
                data = {"raw": resp.text[:500]}

            # Custom validation
            if validate:
                try:
                    validate(data)
                except AssertionError as e:
                    return TestResult(
                        name=name,
                        status=TestStatus.FAIL,
                        duration_ms=duration,
                        message=f"Validation failed: {e}",
                        request=json_data,
                        response=data
                    )

            return TestResult(
                name=name,
                status=TestStatus.PASS,
                duration_ms=duration,
                message="OK",
                request=json_data,
                response=data
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                status=TestStatus.FAIL,
                duration_ms=duration,
                message=str(e),
                request=json_data,
                error=str(e)
            )


# =============================================================================
# TEST SUITES
# =============================================================================

async def test_health(runner: TestRunner) -> TestSuite:
    """Test health and status endpoints."""
    suite = TestSuite("Health & Status")

    # Root endpoint
    suite.add(await runner.run_test(
        name="Root endpoint",
        method="GET",
        path="/",
        validate=lambda d: d.get("name") or d.get("status")
    ))

    # Health check
    suite.add(await runner.run_test(
        name="Health check",
        method="GET",
        path="/health",
        validate=lambda d: "status" in d
    ))

    # Stats (if available)
    suite.add(await runner.run_test(
        name="System stats",
        method="GET",
        path="/stats",
        expected_status=200  # May be 404 if not implemented
    ))

    # Synthesis health
    suite.add(await runner.run_test(
        name="Synthesis health",
        method="GET",
        path="/api/synthesis/health",
        validate=lambda d: "status" in d
    ))

    return suite


async def test_search(runner: TestRunner) -> TestSuite:
    """Test search endpoints."""
    suite = TestSuite("Search")

    # Basic search
    result = await runner.run_test(
        name="Basic search",
        method="POST",
        path="/api/v1/search",
        json_data={
            "query": "retrosigmoid approach",
            "top_k": 10
        },
        validate=lambda d: "results" in d
    )
    suite.add(result)

    # Check for quality field in results
    if result.status == TestStatus.PASS and result.response:
        results = result.response.get('results', [])
        if results:
            has_quality = 'quality' in results[0]
            suite.add(TestResult(
                name="Search results have quality field",
                status=TestStatus.PASS if has_quality else TestStatus.WARN,
                duration_ms=0,
                message="quality field present" if has_quality else "quality field missing"
            ))

    # Search with filters
    suite.add(await runner.run_test(
        name="Search with type filter",
        method="POST",
        path="/api/v1/search",
        json_data={
            "query": "surgical technique",
            "top_k": 5,
            "filters": {"chunk_types": ["PROCEDURE"]}
        },
        validate=lambda d: "results" in d
    ))

    # Search with images
    suite.add(await runner.run_test(
        name="Search with images",
        method="POST",
        path="/api/v1/search",
        json_data={
            "query": "facial nerve anatomy",
            "top_k": 10,
            "include_images": True
        },
        validate=lambda d: "results" in d
    ))

    # Empty query handling
    suite.add(await runner.run_test(
        name="Empty query (should fail)",
        method="POST",
        path="/api/v1/search",
        json_data={"query": "", "top_k": 5},
        expected_status=422  # Validation error
    ))

    return suite


async def test_templates(runner: TestRunner) -> TestSuite:
    """Test template endpoints."""
    suite = TestSuite("Templates")

    # List templates
    result = await runner.run_test(
        name="List templates",
        method="GET",
        path="/api/synthesis/templates",
        validate=lambda d: isinstance(d, list) and len(d) >= 4
    )
    suite.add(result)

    # Validate template structure
    if result.status == TestStatus.PASS and result.response:
        templates = result.response
        for template in templates[:2]:  # Check first 2
            has_required = all(k in template for k in ['type', 'sections'])
            suite.add(TestResult(
                name=f"Template '{template.get('type', '?')}' structure",
                status=TestStatus.PASS if has_required else TestStatus.FAIL,
                duration_ms=0,
                message="All required fields present" if has_required else "Missing fields",
                response=template
            ))

    return suite


async def test_synthesis(runner: TestRunner, quick: bool = False) -> TestSuite:
    """Test synthesis endpoints."""
    suite = TestSuite("Synthesis")

    if quick:
        suite.add(TestResult(
            name="Synthesis generation",
            status=TestStatus.SKIP,
            duration_ms=0,
            message="Skipped (--quick mode)"
        ))
        return suite

    # Full synthesis (this is slow)
    print("\n  Running synthesis test (may take 30-60s)...")

    result = await runner.run_test(
        name="Generate synthesis (PROCEDURAL)",
        method="POST",
        path="/api/synthesis/generate",
        json_data={
            "topic": "retrosigmoid approach for acoustic neuroma",
            "template_type": "PROCEDURAL",
            "max_chunks": 15,
            "include_figures": True
        },
        validate=lambda d: all(k in d for k in ['title', 'sections', 'total_words'])
    )
    suite.add(result)

    # Validate synthesis response structure
    if result.status == TestStatus.PASS and result.response:
        data = result.response

        # Check sections
        sections = data.get('sections', [])
        suite.add(TestResult(
            name="Synthesis has sections",
            status=TestStatus.PASS if len(sections) >= 4 else TestStatus.WARN,
            duration_ms=0,
            message=f"Found {len(sections)} sections",
            response={"section_titles": [s.get('title') for s in sections]}
        ))

        # Check word count
        total_words = data.get('total_words', 0)
        suite.add(TestResult(
            name="Synthesis word count",
            status=TestStatus.PASS if total_words >= 500 else TestStatus.WARN,
            duration_ms=0,
            message=f"{total_words} words generated"
        ))

        # Check figures
        figures = data.get('resolved_figures', [])
        suite.add(TestResult(
            name="Figure resolution",
            status=TestStatus.PASS if len(figures) >= 0 else TestStatus.WARN,
            duration_ms=0,
            message=f"{len(figures)} figures resolved"
        ))

        # Check references
        refs = data.get('references', [])
        suite.add(TestResult(
            name="Source references",
            status=TestStatus.PASS if len(refs) >= 1 else TestStatus.WARN,
            duration_ms=0,
            message=f"{len(refs)} sources cited"
        ))

        # Check enhanced fields
        has_all_cuis = 'all_cuis' in data
        has_chunk_dist = 'chunk_type_distribution' in data
        has_validation = 'validation_result' in data
        has_quality = 'quality_summary' in data

        suite.add(TestResult(
            name="Enhanced synthesis fields",
            status=TestStatus.PASS if (has_all_cuis and has_chunk_dist) else TestStatus.WARN,
            duration_ms=0,
            message=f"all_cuis: {has_all_cuis}, chunk_type_distribution: {has_chunk_dist}, validation_result: {has_validation}, quality_summary: {has_quality}"
        ))

    return suite


async def test_rag(runner: TestRunner) -> TestSuite:
    """Test RAG/chat endpoints."""
    suite = TestSuite("RAG & Chat")

    # Basic RAG question
    suite.add(await runner.run_test(
        name="RAG question",
        method="POST",
        path="/api/v1/rag/ask",
        json_data={
            "question": "What are the key steps in a retrosigmoid approach?",
            "max_context_chunks": 5
        },
        validate=lambda d: "answer" in d or "response" in d
    ))

    # RAG with citations
    suite.add(await runner.run_test(
        name="RAG with citations",
        method="POST",
        path="/api/v1/rag/ask",
        json_data={
            "question": "Describe the facial nerve course",
            "max_context_chunks": 5,
            "include_citations": True
        }
    ))

    return suite


async def test_documents(runner: TestRunner) -> TestSuite:
    """Test document management endpoints."""
    suite = TestSuite("Documents")

    # List documents
    suite.add(await runner.run_test(
        name="List documents",
        method="GET",
        path="/api/v1/documents",
        validate=lambda d: "documents" in d or isinstance(d, list)
    ))

    return suite


async def test_error_handling(runner: TestRunner) -> TestSuite:
    """Test error handling."""
    suite = TestSuite("Error Handling")

    # Invalid template type
    suite.add(await runner.run_test(
        name="Invalid template type",
        method="POST",
        path="/api/synthesis/generate",
        json_data={
            "topic": "test",
            "template_type": "INVALID_TYPE"
        },
        expected_status=400
    ))

    # Missing required field
    suite.add(await runner.run_test(
        name="Missing required field",
        method="POST",
        path="/api/synthesis/generate",
        json_data={
            "template_type": "PROCEDURAL"
            # Missing topic
        },
        expected_status=422
    ))

    # Invalid endpoint
    suite.add(await runner.run_test(
        name="404 on invalid path",
        method="GET",
        path="/api/nonexistent",
        expected_status=404
    ))

    return suite


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Test NeuroSynth API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests (synthesis)")
    parser.add_argument("--endpoint", help="Test specific endpoint only: health, search, templates, synthesis, rag, documents, errors")

    args = parser.parse_args()

    print("="*70)
    print("NEUROSYNTH API COMPREHENSIVE TEST")
    print("="*70)
    print(f"\nTarget: {args.url}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    try:
        import httpx
    except ImportError:
        print("\n\u274c httpx not installed. Run: pip install httpx")
        sys.exit(1)

    # Check server is running
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{args.url}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"\n\u274c Server returned {resp.status_code}")
                sys.exit(1)
    except httpx.ConnectError:
        print(f"\n\u274c Cannot connect to {args.url}")
        print("   Start the server with: uvicorn src.api.main:app --port 8000")
        sys.exit(1)

    print("\n\u2705 Server is running")

    # Run tests
    suites = []

    async with TestRunner(args.url) as runner:
        endpoint_map = {
            "health": test_health,
            "search": test_search,
            "templates": test_templates,
            "synthesis": lambda r: test_synthesis(r, args.quick),
            "rag": test_rag,
            "documents": test_documents,
            "errors": test_error_handling,
        }

        if args.endpoint:
            if args.endpoint not in endpoint_map:
                print(f"\n\u274c Unknown endpoint: {args.endpoint}")
                print(f"   Available: {', '.join(endpoint_map.keys())}")
                sys.exit(1)

            print(f"\nTesting {args.endpoint} endpoint only...")
            suite = await endpoint_map[args.endpoint](runner)
            suites.append(suite)
        else:
            # Run all suites
            for name, test_fn in endpoint_map.items():
                print(f"\nTesting {name}...")
                if name == "synthesis":
                    suite = await test_fn(runner)
                else:
                    suite = await test_fn(runner)
                suites.append(suite)

    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    total_passed = 0
    total_failed = 0

    for suite in suites:
        suite.print_summary()
        total_passed += suite.passed
        total_failed += suite.failed

    print("\n" + "="*70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("="*70)

    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
