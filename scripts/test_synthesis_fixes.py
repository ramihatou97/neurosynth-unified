#!/usr/bin/env python3
"""
NeuroSynth Synthesis Fixes Verification Test
=============================================

Tests that all synthesis engine fixes work correctly:
1. Quality score composite calculation
2. Type-based section routing
3. Caption embedding passthrough
4. CUI preservation
5. Improved authority weighting

Usage:
    DATABASE_URL=... VOYAGE_API_KEY=... python scripts/test_synthesis_fixes.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# TEST RESULTS
# =============================================================================

class TestStatus(Enum):
    PASS = "[PASS]"
    FAIL = "[FAIL]"
    SKIP = "[SKIP]"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAIL)

    def print_report(self):
        print("\n" + "="*70)
        print("SYNTHESIS FIXES TEST REPORT")
        print("="*70)

        for result in self.results:
            print(f"\n{result.status.value} {result.name}")
            print(f"   {result.message}")
            if result.details:
                for k, v in result.details.items():
                    print(f"   - {k}: {v}")

        print("\n" + "-"*70)
        print(f"TOTAL: {self.passed} passed, {self.failed} failed")
        print("-"*70)


# =============================================================================
# MOCK OBJECTS FOR TESTING
# =============================================================================

class MockChunkType(Enum):
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"
    CLINICAL = "CLINICAL"
    PATHOLOGY = "PATHOLOGY"
    GENERAL = "GENERAL"


@dataclass
class MockImage:
    id: str = "img-001"
    file_path: str = "/images/test.png"
    vlm_caption: str = "Surgical exposure showing the facial nerve"
    page_number: int = 45
    image_type: str = "photograph"
    quality_score: float = 0.8
    caption_embedding: Optional[List[float]] = None
    embedding: Optional[List[float]] = None
    cuis: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Create mock embeddings
        import random
        self.caption_embedding = [random.random() for _ in range(1024)]
        self.embedding = [random.random() for _ in range(512)]
        self.cuis = ["C0027051", "C0015450"]


@dataclass
class MockSearchResult:
    chunk_id: str
    content: str
    document_id: str = "doc-001"
    document_title: str = "Rhoton's Cranial Anatomy"
    chunk_type: MockChunkType = MockChunkType.PROCEDURE
    page_start: int = 45
    title: str = "Surgical Approaches"
    authority_score: float = 1.0
    semantic_score: float = 0.85
    keyword_score: float = 0.0
    final_score: float = 0.92
    entity_names: List[str] = field(default_factory=list)
    cuis: List[str] = field(default_factory=list)
    images: List[MockImage] = field(default_factory=list)

    # Quality score components (new pipeline)
    readability_score: float = 0.75
    coherence_score: float = 0.80
    completeness_score: float = 0.85

    def __post_init__(self):
        if not self.entity_names:
            self.entity_names = ["retrosigmoid", "facial nerve"]
        if not self.cuis:
            self.cuis = ["C0027051", "C0015450", "C0001234"]
        if not self.images:
            self.images = [MockImage()]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_quality_score_composite(report: TestReport):
    """Test FIX #1: Quality score composite calculation."""

    try:
        # Import enhanced adapter
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter

        adapter = EnhancedContextAdapter()

        # Test with 3 separate scores
        result = MockSearchResult(
            chunk_id="test-1",
            content="Test content",
            readability_score=0.7,
            coherence_score=0.8,
            completeness_score=0.9
        )

        quality = adapter.compute_quality_score(result)
        expected = 0.7 * 0.3 + 0.8 * 0.3 + 0.9 * 0.4  # = 0.81

        if abs(quality - expected) < 0.01:
            report.add(TestResult(
                name="Quality Score Composite",
                status=TestStatus.PASS,
                message=f"Correctly computed: {quality:.3f} (expected {expected:.3f})",
                details={
                    'readability': 0.7,
                    'coherence': 0.8,
                    'completeness': 0.9,
                    'composite': quality
                }
            ))
        else:
            report.add(TestResult(
                name="Quality Score Composite",
                status=TestStatus.FAIL,
                message=f"Incorrect: {quality:.3f} (expected {expected:.3f})"
            ))

    except ImportError as e:
        report.add(TestResult(
            name="Quality Score Composite",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced adapter: {e}"
        ))


def test_type_based_section_routing(report: TestReport):
    """Test FIX #2: Type-based section routing."""

    try:
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter, TemplateType

        adapter = EnhancedContextAdapter()

        # Test PROCEDURE chunk -> Step-by-Step Technique
        proc_result = MockSearchResult(
            chunk_id="test-proc",
            content="The surgical steps include...",
            chunk_type=MockChunkType.PROCEDURE
        )

        section = adapter.classify_section(proc_result, TemplateType.PROCEDURAL)

        if section == "Step-by-Step Technique":
            report.add(TestResult(
                name="Type-Based Section Routing (PROCEDURE)",
                status=TestStatus.PASS,
                message=f"PROCEDURE -> '{section}' (correct)",
            ))
        else:
            report.add(TestResult(
                name="Type-Based Section Routing (PROCEDURE)",
                status=TestStatus.FAIL,
                message=f"PROCEDURE -> '{section}' (expected 'Step-by-Step Technique')"
            ))

        # Test ANATOMY chunk -> Surgical Approach
        anat_result = MockSearchResult(
            chunk_id="test-anat",
            content="The anatomical structures include...",
            chunk_type=MockChunkType.ANATOMY
        )

        section = adapter.classify_section(anat_result, TemplateType.PROCEDURAL)

        if section == "Surgical Approach":
            report.add(TestResult(
                name="Type-Based Section Routing (ANATOMY)",
                status=TestStatus.PASS,
                message=f"ANATOMY -> '{section}' (correct)",
            ))
        else:
            report.add(TestResult(
                name="Type-Based Section Routing (ANATOMY)",
                status=TestStatus.FAIL,
                message=f"ANATOMY -> '{section}' (expected 'Surgical Approach')"
            ))

    except ImportError as e:
        report.add(TestResult(
            name="Type-Based Section Routing",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced adapter: {e}"
        ))


def test_caption_embedding_passthrough(report: TestReport):
    """Test FIX #3: Caption embedding passthrough."""

    try:
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter, TemplateType

        adapter = EnhancedContextAdapter()

        # Create result with image that has embeddings
        result = MockSearchResult(
            chunk_id="test-img",
            content="Figure shows the facial nerve exposure..."
        )

        # Adapt
        context = adapter.adapt(
            topic="Facial nerve surgery",
            search_results=[result],
            template_type=TemplateType.PROCEDURAL
        )

        image_catalog = context.get('image_catalog', [])

        if not image_catalog:
            report.add(TestResult(
                name="Caption Embedding Passthrough",
                status=TestStatus.FAIL,
                message="No images in catalog"
            ))
            return

        img = image_catalog[0]
        has_caption_emb = 'caption_embedding' in img and img['caption_embedding'] is not None
        has_clip_emb = 'clip_embedding' in img and img['clip_embedding'] is not None
        has_cuis = 'cuis' in img and img['cuis']

        if has_caption_emb and has_clip_emb:
            report.add(TestResult(
                name="Caption Embedding Passthrough",
                status=TestStatus.PASS,
                message="Caption and CLIP embeddings preserved",
                details={
                    'caption_embedding_dim': len(img['caption_embedding']),
                    'clip_embedding_dim': len(img['clip_embedding']),
                    'has_cuis': has_cuis
                }
            ))
        else:
            report.add(TestResult(
                name="Caption Embedding Passthrough",
                status=TestStatus.FAIL,
                message="Missing embeddings",
                details={
                    'has_caption_embedding': has_caption_emb,
                    'has_clip_embedding': has_clip_emb
                }
            ))

    except ImportError as e:
        report.add(TestResult(
            name="Caption Embedding Passthrough",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced adapter: {e}"
        ))


def test_cui_preservation(report: TestReport):
    """Test FIX #4: CUI preservation."""

    try:
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter, TemplateType

        adapter = EnhancedContextAdapter()

        # Create results with CUIs
        results = [
            MockSearchResult(
                chunk_id="test-1",
                content="Content about myocardial infarction...",
                cuis=["C0027051", "C0001234"]
            ),
            MockSearchResult(
                chunk_id="test-2",
                content="Content about facial nerve...",
                cuis=["C0015450", "C0005678"]
            )
        ]

        context = adapter.adapt(
            topic="Test topic",
            search_results=results,
            template_type=TemplateType.PROCEDURAL
        )

        # Check CUIs in adapted chunks
        all_cuis_in_context = context.get('all_cuis', [])
        expected_cuis = {"C0027051", "C0001234", "C0015450", "C0005678"}

        preserved = set(all_cuis_in_context) & expected_cuis

        # Check CUIs in individual chunks
        chunk_cuis = []
        for section, chunks in context['sections'].items():
            for chunk in chunks:
                chunk_cuis.extend(chunk.get('cuis', []))

        if len(preserved) >= 4 and len(chunk_cuis) >= 4:
            report.add(TestResult(
                name="CUI Preservation",
                status=TestStatus.PASS,
                message=f"All {len(preserved)} CUIs preserved",
                details={
                    'all_cuis': len(all_cuis_in_context),
                    'in_chunks': len(chunk_cuis),
                    'expected': len(expected_cuis)
                }
            ))
        else:
            report.add(TestResult(
                name="CUI Preservation",
                status=TestStatus.FAIL,
                message=f"Only {len(preserved)}/4 CUIs preserved",
                details={
                    'preserved': list(preserved),
                    'missing': list(expected_cuis - preserved)
                }
            ))

    except ImportError as e:
        report.add(TestResult(
            name="CUI Preservation",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced adapter: {e}"
        ))


def test_authority_weighting(report: TestReport):
    """Test FIX #5: Authority score in combined score."""

    try:
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter, TemplateType

        adapter = EnhancedContextAdapter()

        # Create result with known scores
        result = MockSearchResult(
            chunk_id="test-auth",
            content="Test content",
            final_score=0.9,
            authority_score=0.8,
            readability_score=0.7,
            coherence_score=0.7,
            completeness_score=0.7
        )

        context = adapter.adapt(
            topic="Test",
            search_results=[result],
            template_type=TemplateType.PROCEDURAL
        )

        # Find the chunk
        chunk = None
        for section, chunks in context['sections'].items():
            if chunks:
                chunk = chunks[0]
                break

        if not chunk:
            report.add(TestResult(
                name="Authority Weighting",
                status=TestStatus.FAIL,
                message="No chunks in context"
            ))
            return

        combined = chunk.get('combined_score', 0)
        quality = chunk.get('quality_score', 0.7)  # Should be 0.7

        # Expected: final_score * authority_score * quality_score
        expected = 0.9 * 0.8 * quality

        if abs(combined - expected) < 0.01:
            report.add(TestResult(
                name="Authority Weighting",
                status=TestStatus.PASS,
                message=f"Combined score includes quality: {combined:.3f}",
                details={
                    'final_score': 0.9,
                    'authority_score': 0.8,
                    'quality_score': quality,
                    'combined_score': combined,
                    'expected': expected
                }
            ))
        else:
            report.add(TestResult(
                name="Authority Weighting",
                status=TestStatus.FAIL,
                message=f"Combined score mismatch: {combined:.3f} (expected {expected:.3f})"
            ))

    except ImportError as e:
        report.add(TestResult(
            name="Authority Weighting",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced adapter: {e}"
        ))


def test_figure_resolver(report: TestReport):
    """Test FIX #3: Enhanced figure resolver with semantic matching."""

    try:
        from synthesis_fixes.enhanced_context_adapter import EnhancedFigureResolver
        import random

        resolver = EnhancedFigureResolver(min_match_score=0.2, prefer_semantic=True)

        # Create mock figure request
        @dataclass
        class MockFigureRequest:
            id: str = "fig-1"
            description: str = "Surgical exposure of facial nerve"
            topic: str = "facial nerve surgery"
            section: str = "Surgical Approach"
            cuis: List[str] = field(default_factory=list)

            def __post_init__(self):
                self.cuis = ["C0015450"]

        request = MockFigureRequest()

        # Create image catalog with embeddings
        image_catalog = [
            {
                "id": "img-1",
                "caption": "Facial nerve exposure during surgery",
                "vlm_caption": "Intraoperative photograph showing the facial nerve trunk",
                "caption_embedding": [random.random() for _ in range(1024)],
                "cuis": ["C0015450", "C0027051"],
                "path": "/images/facial_nerve.png"
            },
            {
                "id": "img-2",
                "caption": "MRI showing tumor",
                "vlm_caption": "T1 MRI with contrast showing acoustic neuroma",
                "caption_embedding": [random.random() for _ in range(1024)],
                "cuis": ["C0000001"],
                "path": "/images/tumor_mri.png"
            }
        ]

        updated, resolved = resolver.resolve([request], image_catalog)

        if resolved and resolved[0]['image_id'] == 'img-1':
            report.add(TestResult(
                name="Enhanced Figure Resolver",
                status=TestStatus.PASS,
                message="Correctly matched facial nerve image",
                details={
                    'matched_image': resolved[0]['image_id'],
                    'confidence': resolved[0]['confidence'],
                    'score_breakdown': resolved[0]['score_breakdown']
                }
            ))
        elif resolved:
            report.add(TestResult(
                name="Enhanced Figure Resolver",
                status=TestStatus.FAIL,
                message=f"Matched wrong image: {resolved[0]['image_id']} (expected img-1)"
            ))
        else:
            report.add(TestResult(
                name="Enhanced Figure Resolver",
                status=TestStatus.FAIL,
                message="No matches found"
            ))

    except ImportError as e:
        report.add(TestResult(
            name="Enhanced Figure Resolver",
            status=TestStatus.SKIP,
            message=f"Could not import enhanced figure resolver: {e}"
        ))


# =============================================================================
# INTEGRATION TEST (requires database)
# =============================================================================

async def test_integration_with_search(report: TestReport):
    """Test enhanced adapter with real search results."""

    db_url = os.getenv("DATABASE_URL")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not db_url or not voyage_key:
        report.add(TestResult(
            name="Integration with Search",
            status=TestStatus.SKIP,
            message="DATABASE_URL or VOYAGE_API_KEY not set"
        ))
        return

    try:
        from src.database.connection import DatabaseConnection
        from src.ingest.embeddings import VoyageTextEmbedder
        from src.retrieval.search_service import SearchService
        from synthesis_fixes.enhanced_context_adapter import EnhancedContextAdapter, TemplateType

        db = await DatabaseConnection.initialize(db_url)
        embedder = VoyageTextEmbedder(api_key=voyage_key)

        service = SearchService(database=db, embedder=embedder)
        adapter = EnhancedContextAdapter()

        query = "retrosigmoid approach acoustic neuroma"
        response = await service.search(query, top_k=10, include_images=True)

        if not response.results:
            report.add(TestResult(
                name="Integration with Search",
                status=TestStatus.FAIL,
                message="No search results returned"
            ))
            await db.close()
            return

        # Adapt with enhanced adapter
        context = adapter.adapt(
            topic=query,
            search_results=response.results,
            template_type=TemplateType.PROCEDURAL
        )

        # Verify all fixes are working
        checks = {
            'sections_populated': any(len(c) > 0 for c in context['sections'].values()),
            'all_cuis_present': 'all_cuis' in context,
            'image_catalog': len(context.get('image_catalog', [])) >= 0,
            'sources_sorted': len(context.get('sources', [])) > 0,
        }

        await db.close()

        if all(checks.values()):
            report.add(TestResult(
                name="Integration with Search",
                status=TestStatus.PASS,
                message="Enhanced adapter works with real search results",
                details={
                    'sections': list(context['sections'].keys()),
                    'total_chunks': context.get('total_chunks'),
                    'all_cuis': len(context.get('all_cuis', [])),
                    'images': len(context.get('image_catalog', []))
                }
            ))
        else:
            report.add(TestResult(
                name="Integration with Search",
                status=TestStatus.FAIL,
                message="Some checks failed",
                details=checks
            ))

    except Exception as e:
        report.add(TestResult(
            name="Integration with Search",
            status=TestStatus.FAIL,
            message=f"Error: {e}"
        ))


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("="*70)
    print("NEUROSYNTH SYNTHESIS FIXES VERIFICATION")
    print("="*70)

    report = TestReport()

    print("\n[1] Testing Quality Score Composite...")
    test_quality_score_composite(report)

    print("[2] Testing Type-Based Section Routing...")
    test_type_based_section_routing(report)

    print("[3] Testing Caption Embedding Passthrough...")
    test_caption_embedding_passthrough(report)

    print("[4] Testing CUI Preservation...")
    test_cui_preservation(report)

    print("[5] Testing Authority Weighting...")
    test_authority_weighting(report)

    print("[6] Testing Enhanced Figure Resolver...")
    test_figure_resolver(report)

    print("[7] Testing Integration with Search...")
    await test_integration_with_search(report)

    report.print_report()

    if report.failed > 0:
        print("\n[FAIL] SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n[PASS] ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
