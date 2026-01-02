#!/usr/bin/env python3
"""
NeuroSynth API Alignment Verification
======================================

Verifies that the existing API routes work correctly with:
- Enhanced ContextAdapter (synthesis fixes)
- Upgraded SearchService (authority/type/CUI boosting)
- New data structures (quality scores, caption embeddings)

Usage:
    # Start server first:
    uvicorn src.api.main:app --reload --port 8000

    # Then run tests:
    python scripts/verify_api_alignment.py

Or run without server (direct import test):
    python scripts/verify_api_alignment.py --direct
"""

import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# VERIFICATION RESULTS
# =============================================================================

class CheckStatus(Enum):
    PASS = "\u2705"
    FAIL = "\u274c"
    WARN = "\u26a0\ufe0f"
    SKIP = "\u23ed\ufe0f"


@dataclass
class Check:
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check):
        self.checks.append(check)

    def print_report(self):
        print("\n" + "="*70)
        print("API ALIGNMENT VERIFICATION REPORT")
        print("="*70)

        for check in self.checks:
            print(f"\n{check.status.value} {check.name}")
            print(f"   {check.message}")
            if check.details:
                for k, v in list(check.details.items())[:5]:
                    print(f"   - {k}: {v}")

        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

        print("\n" + "-"*70)
        print(f"RESULT: {passed} passed, {failed} failed")
        print("-"*70)


# =============================================================================
# DIRECT VERIFICATION (No Server Required)
# =============================================================================

async def verify_direct(report: VerificationReport):
    """Verify API alignment by directly importing and testing components."""

    print("\n[1] Checking imports...")

    # Check synthesis engine imports
    try:
        from src.synthesis.engine import (
            SynthesisEngine,
            TemplateType,
            SynthesisResult,
            SynthesisSection,
            ContextAdapter,
            FigureResolver,
            TEMPLATE_SECTIONS,
            TEMPLATE_REQUIREMENTS,
        )
        report.add(Check(
            name="Synthesis Engine Imports",
            status=CheckStatus.PASS,
            message="All synthesis engine components importable"
        ))
    except ImportError as e:
        report.add(Check(
            name="Synthesis Engine Imports",
            status=CheckStatus.FAIL,
            message=f"Import failed: {e}"
        ))
        return

    # Check API route imports
    try:
        from src.api.routes.synthesis import (
            router,
            SynthesisRequest,
            SynthesisResponse,
            SectionResponse,
            ReferenceResponse,
        )
        report.add(Check(
            name="API Route Imports",
            status=CheckStatus.PASS,
            message="Synthesis API routes importable"
        ))
    except ImportError as e:
        report.add(Check(
            name="API Route Imports",
            status=CheckStatus.FAIL,
            message=f"Import failed: {e}"
        ))
        return

    # Check API models imports
    print("\n[2] Checking API models...")

    try:
        from src.api.models import (
            QualityBreakdown,
            ValidationResult,
            SearchResultItem,
        )
        report.add(Check(
            name="Enhanced API Models",
            status=CheckStatus.PASS,
            message="QualityBreakdown, ValidationResult models available"
        ))

        # Check QualityBreakdown fields
        qb_fields = set(QualityBreakdown.model_fields.keys())
        expected_qb = {'readability', 'coherence', 'completeness', 'composite'}
        if expected_qb.issubset(qb_fields):
            report.add(Check(
                name="QualityBreakdown Fields",
                status=CheckStatus.PASS,
                message=f"All expected fields present: {expected_qb}"
            ))
        else:
            report.add(Check(
                name="QualityBreakdown Fields",
                status=CheckStatus.WARN,
                message=f"Missing fields: {expected_qb - qb_fields}"
            ))

        # Check ValidationResult fields
        vr_fields = set(ValidationResult.model_fields.keys())
        expected_vr = {'validated', 'hallucination_risk', 'generated_cuis', 'source_cuis', 'unsupported_cuis'}
        if expected_vr.issubset(vr_fields):
            report.add(Check(
                name="ValidationResult Fields",
                status=CheckStatus.PASS,
                message=f"All expected fields present: {expected_vr}"
            ))
        else:
            report.add(Check(
                name="ValidationResult Fields",
                status=CheckStatus.WARN,
                message=f"Missing fields: {expected_vr - vr_fields}"
            ))

        # Check SearchResultItem has quality field
        sri_fields = set(SearchResultItem.model_fields.keys())
        if 'quality' in sri_fields:
            report.add(Check(
                name="SearchResultItem.quality",
                status=CheckStatus.PASS,
                message="quality field present in SearchResultItem"
            ))
        else:
            report.add(Check(
                name="SearchResultItem.quality",
                status=CheckStatus.FAIL,
                message="quality field MISSING from SearchResultItem"
            ))

    except ImportError as e:
        report.add(Check(
            name="Enhanced API Models",
            status=CheckStatus.FAIL,
            message=f"Import failed: {e}"
        ))

    # Check API schema alignment
    print("\n[3] Checking schema alignment...")

    # Check SynthesisResponse fields
    response_fields = set(SynthesisResponse.model_fields.keys())

    # Key fields that should be in response
    expected_fields = {
        'title', 'abstract', 'sections', 'total_words',
        'total_figures', 'synthesis_time_ms'
    }

    # Enhanced fields
    enhanced_fields = {'all_cuis', 'chunk_type_distribution', 'validation_result', 'quality_summary'}

    missing_core = expected_fields - response_fields
    if missing_core:
        report.add(Check(
            name="Core Schema Fields",
            status=CheckStatus.WARN,
            message=f"Missing core fields: {missing_core}",
            details={'response_fields': list(response_fields)[:10]}
        ))
    else:
        report.add(Check(
            name="Core Schema Fields",
            status=CheckStatus.PASS,
            message="All core fields present in SynthesisResponse"
        ))

    present_enhanced = enhanced_fields & response_fields
    if present_enhanced:
        report.add(Check(
            name="Enhanced Schema Fields",
            status=CheckStatus.PASS,
            message=f"Enhanced fields present: {present_enhanced}"
        ))
    else:
        report.add(Check(
            name="Enhanced Schema Fields",
            status=CheckStatus.WARN,
            message="No enhanced fields found in SynthesisResponse"
        ))

    # Check service container
    print("\n[4] Checking service container...")

    try:
        from src.api.dependencies import ServiceContainer, get_container

        report.add(Check(
            name="Service Container",
            status=CheckStatus.PASS,
            message="ServiceContainer available"
        ))

    except ImportError as e:
        report.add(Check(
            name="Service Container",
            status=CheckStatus.FAIL,
            message=f"Import failed: {e}"
        ))

    # Check route registration
    print("\n[5] Checking route registration...")

    routes = [r.path for r in router.routes]
    expected_routes = ['/templates', '/generate', '/generate/stream', '/health']

    missing_routes = [r for r in expected_routes if r not in routes]

    if missing_routes:
        report.add(Check(
            name="Route Registration",
            status=CheckStatus.WARN,
            message=f"Missing routes: {missing_routes}",
            details={'registered': routes}
        ))
    else:
        report.add(Check(
            name="Route Registration",
            status=CheckStatus.PASS,
            message=f"All expected routes registered ({len(routes)} total)",
            details={'routes': routes}
        ))


# =============================================================================
# HTTP VERIFICATION (Requires Running Server)
# =============================================================================

async def verify_http(report: VerificationReport, base_url: str = "http://localhost:8000"):
    """Verify API by making HTTP requests to running server."""

    try:
        import httpx
    except ImportError:
        report.add(Check(
            name="HTTP Client",
            status=CheckStatus.SKIP,
            message="httpx not installed - run: pip install httpx"
        ))
        return

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:

        # Health check
        print("\n[1] Checking API health...")
        try:
            resp = await client.get("/health")
            if resp.status_code == 200:
                data = resp.json()
                report.add(Check(
                    name="API Health",
                    status=CheckStatus.PASS,
                    message="API is healthy",
                    details=data
                ))
            else:
                report.add(Check(
                    name="API Health",
                    status=CheckStatus.FAIL,
                    message=f"Health check failed: {resp.status_code}"
                ))
                return
        except httpx.ConnectError:
            report.add(Check(
                name="API Health",
                status=CheckStatus.FAIL,
                message=f"Cannot connect to {base_url} - is server running?"
            ))
            return

        # Synthesis health
        print("\n[2] Checking synthesis health...")
        try:
            resp = await client.get("/api/synthesis/health")
            if resp.status_code == 200:
                data = resp.json()
                report.add(Check(
                    name="Synthesis Health",
                    status=CheckStatus.PASS,
                    message="Synthesis subsystem healthy",
                    details=data
                ))
            else:
                report.add(Check(
                    name="Synthesis Health",
                    status=CheckStatus.WARN,
                    message=f"Synthesis health: {resp.status_code}"
                ))
        except Exception as e:
            report.add(Check(
                name="Synthesis Health",
                status=CheckStatus.WARN,
                message=f"Synthesis health check failed: {e}"
            ))

        # Templates endpoint
        print("\n[3] Checking templates endpoint...")
        try:
            resp = await client.get("/api/synthesis/templates")
            if resp.status_code == 200:
                templates = resp.json()
                report.add(Check(
                    name="Templates Endpoint",
                    status=CheckStatus.PASS,
                    message=f"Retrieved {len(templates)} templates",
                    details={'types': [t['type'] for t in templates]}
                ))
            else:
                report.add(Check(
                    name="Templates Endpoint",
                    status=CheckStatus.FAIL,
                    message=f"Templates failed: {resp.status_code}"
                ))
        except Exception as e:
            report.add(Check(
                name="Templates Endpoint",
                status=CheckStatus.FAIL,
                message=f"Templates endpoint error: {e}"
            ))

        # Search endpoint (prerequisite for synthesis)
        print("\n[4] Checking search endpoint...")
        try:
            resp = await client.post("/api/v1/search", json={
                "query": "retrosigmoid approach",
                "top_k": 5
            })
            if resp.status_code == 200:
                data = resp.json()
                result_count = len(data.get('results', []))

                # Check for quality field in results
                has_quality = False
                if result_count > 0:
                    first_result = data['results'][0]
                    has_quality = 'quality' in first_result

                report.add(Check(
                    name="Search Endpoint",
                    status=CheckStatus.PASS if result_count > 0 else CheckStatus.WARN,
                    message=f"Search returned {result_count} results",
                    details={'has_quality_field': has_quality, 'search_time_ms': data.get('search_time_ms')}
                ))
            else:
                report.add(Check(
                    name="Search Endpoint",
                    status=CheckStatus.FAIL,
                    message=f"Search failed: {resp.status_code} - {resp.text[:100]}"
                ))
        except Exception as e:
            report.add(Check(
                name="Search Endpoint",
                status=CheckStatus.FAIL,
                message=f"Search endpoint error: {e}"
            ))

        # Synthesis endpoint (the main test)
        print("\n[5] Testing synthesis endpoint...")
        print("   This may take 30-60 seconds...")

        try:
            start = time.time()
            resp = await client.post("/api/synthesis/generate", json={
                "topic": "retrosigmoid approach for acoustic neuroma",
                "template_type": "PROCEDURAL",
                "max_chunks": 20,
                "include_figures": True
            })
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                report.add(Check(
                    name="Synthesis Endpoint",
                    status=CheckStatus.PASS,
                    message=f"Synthesis completed in {elapsed:.1f}s",
                    details={
                        'title': data.get('title', '')[:50],
                        'sections': len(data.get('sections', [])),
                        'total_words': data.get('total_words'),
                        'total_figures': data.get('total_figures'),
                        'synthesis_time_ms': data.get('synthesis_time_ms')
                    }
                ))

                # Check for enhanced fields
                has_all_cuis = 'all_cuis' in data
                has_chunk_dist = 'chunk_type_distribution' in data
                has_validation = 'validation_result' in data

                report.add(Check(
                    name="Enhanced Fields in Synthesis Response",
                    status=CheckStatus.PASS if (has_all_cuis and has_chunk_dist) else CheckStatus.WARN,
                    message=f"all_cuis: {has_all_cuis}, chunk_type_distribution: {has_chunk_dist}, validation_result: {has_validation}"
                ))

            elif resp.status_code == 504:
                report.add(Check(
                    name="Synthesis Endpoint",
                    status=CheckStatus.WARN,
                    message=f"Synthesis timed out after {elapsed:.1f}s"
                ))
            else:
                report.add(Check(
                    name="Synthesis Endpoint",
                    status=CheckStatus.FAIL,
                    message=f"Synthesis failed: {resp.status_code}",
                    details={'error': resp.text[:200]}
                ))

        except Exception as e:
            report.add(Check(
                name="Synthesis Endpoint",
                status=CheckStatus.FAIL,
                message=f"Synthesis error: {e}"
            ))


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Verify API alignment")
    parser.add_argument("--direct", action="store_true", help="Direct import test (no server)")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")

    args = parser.parse_args()

    print("="*70)
    print("NEUROSYNTH API ALIGNMENT VERIFICATION")
    print("="*70)

    report = VerificationReport()

    if args.direct:
        print("\nMode: Direct Import Test")
        await verify_direct(report)
    else:
        print(f"\nMode: HTTP Test against {args.url}")
        print("Note: Start server first with 'uvicorn src.api.main:app --port 8000'")
        await verify_http(report, args.url)

    report.print_report()

    # Exit code
    failed = sum(1 for c in report.checks if c.status == CheckStatus.FAIL)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
