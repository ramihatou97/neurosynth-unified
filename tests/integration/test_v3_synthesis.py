"""
V3 Synthesis Pipeline Integration Tests
========================================

Tests all V3 enhancements:
1. Robust JSON parsing (no more crashes from LLM markdown)
2. MMR deduplication (diverse chunk selection)
3. Semantic Router (vector-based section classification)
4. Semantic Figure Resolver (embedding-based figure matching)
5. Conflict-Aware Merger (internal vs external reconciliation)
6. Adversarial Reviewer (Tier 1 safety validation)
7. Enrichment Models (Pydantic schemas)
"""

import pytest
import numpy as np
from uuid import uuid4


# =============================================================================
# PARSING TESTS
# =============================================================================

class TestRobustParsing:
    """Tests for src/shared/parsing.py"""

    def test_strip_markdown_fences(self):
        """Should strip markdown code fences from JSON."""
        from src.shared.parsing import strip_markdown_fences

        raw = '```json\n{"key": "value"}\n```'
        result = strip_markdown_fences(raw)
        assert result == '{"key": "value"}'

    def test_strip_markdown_fences_with_language(self):
        """Should handle various fence formats."""
        from src.shared.parsing import strip_markdown_fences

        raw = '```JSON\n{"a": 1}\n```'
        assert strip_markdown_fences(raw) == '{"a": 1}'

        raw2 = "```\n[1, 2, 3]\n```"
        assert strip_markdown_fences(raw2) == "[1, 2, 3]"

    def test_find_json_boundaries_object(self):
        """Should find JSON object boundaries."""
        from src.shared.parsing import find_json_boundaries

        text = 'Here is the result: {"gaps": []} and more text'
        start, end = find_json_boundaries(text)
        assert text[start:end+1] == '{"gaps": []}'

    def test_find_json_boundaries_array(self):
        """Should find JSON array boundaries."""
        from src.shared.parsing import find_json_boundaries

        text = 'The data: [1, 2, 3] is here'
        start, end = find_json_boundaries(text)
        assert text[start:end+1] == '[1, 2, 3]'

    def test_extract_json_string_simple(self):
        """Should extract JSON string from markdown."""
        from src.shared.parsing import extract_json_string
        import json

        raw = '```json\n{"name": "test", "value": 42}\n```'
        json_str = extract_json_string(raw)
        result = json.loads(json_str)
        assert result == {"name": "test", "value": 42}

    def test_extract_json_string_with_preamble(self):
        """Should extract JSON ignoring preamble text."""
        from src.shared.parsing import extract_json_string
        import json

        raw = 'Here is the JSON:\n{"items": [1, 2, 3]}\nLet me know if you need more.'
        json_str = extract_json_string(raw)
        result = json.loads(json_str)
        assert result["items"] == [1, 2, 3]

    def test_extract_and_parse_json_with_model(self):
        """Should validate against Pydantic model."""
        from src.shared.parsing import extract_and_parse_json
        from pydantic import BaseModel
        from typing import List

        class GapResult(BaseModel):
            gaps: List[str]

        raw = '{"gaps": ["gap1", "gap2"]}'
        result = extract_and_parse_json(raw, GapResult)
        assert isinstance(result, GapResult)
        assert result.gaps == ["gap1", "gap2"]


# =============================================================================
# MMR DEDUPLICATION TESTS
# =============================================================================

class TestMMRDeduplication:
    """Tests for src/shared/mmr.py"""

    def test_mmr_sort_basic(self):
        """Should reorder for diversity."""
        from src.shared.mmr import mmr_sort

        np.random.seed(42)
        # Create embeddings where items 0,1,2 are similar, 3 is different
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # Very similar to 0
            [0.98, 0.15, 0.0],  # Very similar to 0,1
            [0.0, 1.0, 0.0],  # Different
            [0.0, 0.0, 1.0],  # Different
        ]
        candidate_indices = [0, 1, 2, 3, 4]

        query = [1.0, 0.0, 0.0]

        # With high lambda (0.9), should favor relevance
        indices_high = mmr_sort(query, embeddings, candidate_indices, top_k=3, lambda_mult=0.9)
        assert indices_high[0] == 0  # Most relevant first

        # With low lambda (0.3), should favor diversity more
        indices_low = mmr_sort(query, embeddings, candidate_indices, top_k=3, lambda_mult=0.3)
        # Should include diverse items earlier
        assert 3 in indices_low or 4 in indices_low

    def test_mmr_rerank_with_items(self):
        """Should rerank list of items based on embeddings."""
        from src.shared.mmr import mmr_rerank

        # Create dicts with embedding key
        items = [
            {"id": "doc_a", "embedding": [1.0, 0.0]},
            {"id": "doc_b", "embedding": [0.99, 0.1]},
            {"id": "doc_c", "embedding": [0.0, 1.0]},
            {"id": "doc_d", "embedding": [-1.0, 0.0]},
        ]

        query = [1.0, 0.0]

        reranked = mmr_rerank(query, items, embedding_key="embedding", top_k=3, lambda_mult=0.5)

        assert len(reranked) == 3
        assert reranked[0]["id"] == "doc_a"  # Most relevant

    def test_mmr_selector_class(self):
        """Should work with MMRSelector class interface."""
        from src.shared.mmr import MMRSelector

        selector = MMRSelector(lambda_mult=0.7)

        embeddings = [[float(x) for x in row] for row in np.random.randn(10, 64)]
        query = [float(x) for x in np.random.randn(64)]

        indices = selector.select(query, embeddings, top_k=5)

        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique


# =============================================================================
# SEMANTIC ROUTER TESTS
# =============================================================================

class TestSemanticRouter:
    """Tests for src/synthesis/router.py"""

    def test_section_prototypes_exist(self):
        """Should have prototypes for all template types."""
        from src.synthesis.router import SectionPrototypes

        assert hasattr(SectionPrototypes, 'PROCEDURAL')
        assert hasattr(SectionPrototypes, 'ANATOMY')
        assert hasattr(SectionPrototypes, 'DISORDER')

        # Check procedural has expected sections (uppercase keys)
        assert 'EQUIPMENT' in SectionPrototypes.PROCEDURAL
        assert 'TECHNIQUE' in SectionPrototypes.PROCEDURAL
        assert 'APPROACH' in SectionPrototypes.PROCEDURAL

    def test_keyword_fallback_router(self):
        """Should route based on keywords when no embedder."""
        from src.synthesis.router import KeywordFallbackRouter

        router = KeywordFallbackRouter()

        # Equipment content (positioning)
        content = "The patient is positioned supine with the head rotated. The microscope is configured."
        result = router.route_chunk(content)
        assert result.section_name in ["EQUIPMENT", "APPROACH"]  # Position/microscope keywords

        # Anatomy content
        content2 = "The middle cerebral artery bifurcates into superior and inferior trunks. The nerve courses along."
        result2 = router.route_chunk(content2)
        assert result2.section_name == "ANATOMY"

    def test_route_result_structure(self):
        """Should return properly structured RouteResult."""
        from src.synthesis.router import KeywordFallbackRouter, RouteResult

        router = KeywordFallbackRouter()
        result = router.route_chunk("test content with anatomy nerve artery")

        assert isinstance(result, RouteResult)
        assert hasattr(result, 'section_name')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'is_confident')
        assert 0 <= result.confidence_score <= 1.0


# =============================================================================
# FIGURE RESOLVER TESTS
# =============================================================================

class TestSemanticFigureResolver:
    """Tests for src/synthesis/figure_resolver_semantic.py"""

    def test_legacy_resolver_basic(self):
        """Should match figures by keyword overlap."""
        from src.synthesis.figure_resolver_semantic import LegacyFigureResolver
        from dataclasses import dataclass

        @dataclass
        class MockImage:
            id: str
            vlm_caption: str
            authority_tier: int = 2

        images = [
            MockImage("img1", "MCA aneurysm surgical approach", 1),
            MockImage("img2", "Patient positioning for craniotomy", 2),
            MockImage("img3", "Arterial blood supply diagram", 3),
        ]

        resolver = LegacyFigureResolver()

        # Should match aneurysm image
        matches = resolver.resolve("MCA aneurysm clipping technique", images, top_k=2)

        assert len(matches) <= 2
        if matches:
            assert matches[0].image_id == "img1"  # Best match

    def test_figure_match_structure(self):
        """Should return properly structured FigureMatch."""
        from src.synthesis.figure_resolver_semantic import FigureMatch

        match = FigureMatch(
            image_id="img1",
            score=0.85,
            semantic_score=0.9,
            keyword_score=0.7,
            tier_boost=1.0
        )

        assert match.image_id == "img1"
        assert match.score == 0.85
        assert match.tier_boost == 1.0


# =============================================================================
# CONFLICT MERGER TESTS
# =============================================================================

class TestConflictMerger:
    """Tests for src/synthesis/conflict_merger.py"""

    def test_extracted_fact_structure(self):
        """Should create ExtractedFact correctly."""
        from src.synthesis.conflict_merger import ExtractedFact

        fact = ExtractedFact(
            claim="The artery should be clipped before dissection",
            source_type="internal",
            is_recommendation=True
        )

        assert fact.claim == "The artery should be clipped before dissection"
        assert fact.source_type == "internal"
        assert fact.is_recommendation is True

    def test_detected_conflict_structure(self):
        """Should create DetectedConflict correctly."""
        from src.synthesis.conflict_merger import (
            DetectedConflict,
            ConflictCategory,
            ResolutionStrategy
        )

        conflict = DetectedConflict(
            category=ConflictCategory.APPROACH,
            description="Different approaches to clipping order",
            internal_claim="Clip before dissection",
            external_claim="Simultaneous approach acceptable",
            resolution_strategy=ResolutionStrategy.PREFER_INTERNAL
        )

        assert conflict.category == ConflictCategory.APPROACH
        assert conflict.resolution_strategy == ResolutionStrategy.PREFER_INTERNAL

    def test_merge_result_structure(self):
        """Should create MergeResult with proper structure."""
        from src.synthesis.conflict_merger import (
            MergeResult,
            DetectedConflict,
            ConflictCategory,
            ResolutionStrategy
        )

        conflict = DetectedConflict(
            category=ConflictCategory.RECOMMENDATION,
            description="Different values",
            internal_claim="Clip first",
            external_claim="Dissect first",
            resolution_strategy=ResolutionStrategy.PREFER_INTERNAL
        )

        result = MergeResult(
            topic="aneurysm clipping",
            resolved_content="Based on Tier 1 sources, clip before dissection.",
            merge_strategy_used="prefer_internal",
            conflicts=[conflict],
            conflict_count=1,
            internal_facts_used=1,
            external_facts_used=0
        )

        assert result is not None
        assert result.topic == "aneurysm clipping"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution_strategy == ResolutionStrategy.PREFER_INTERNAL


# =============================================================================
# ADVERSARIAL REVIEWER TESTS
# =============================================================================

class TestAdversarialReviewer:
    """Tests for src/synthesis/agents/reviewer.py"""

    def test_controversy_warning_structure(self):
        """Should create ControversyWarning correctly."""
        from src.synthesis.agents.reviewer import ControversyWarning

        warning = ControversyWarning(
            severity="HIGH",
            topic="proximal control",
            draft_claim="Clip without proximal control",
            contradicting_source="Lawton",
            source_quote="Always obtain proximal control first",
            recommendation="Add proximal control step"
        )

        assert warning.severity == "HIGH"
        assert warning.topic == "proximal control"

        # Test serialization
        d = warning.to_dict()
        assert d["severity"] == "HIGH"
        assert "draft_claim" in d

    def test_review_result_structure(self):
        """Should create ReviewResult correctly."""
        from src.synthesis.agents.reviewer import ReviewResult, ControversyWarning

        result = ReviewResult(
            has_issues=True,
            warnings=[
                ControversyWarning(
                    severity="HIGH",
                    topic="test",
                    draft_claim="claim",
                    contradicting_source="source",
                    source_quote="quote",
                    recommendation="rec"
                )
            ],
            section_reviewed="Surgical Technique",
            confidence=0.85
        )

        assert result.has_issues is True
        assert len(result.warnings) == 1
        assert result.get_critical_count() == 1

        d = result.to_dict()
        assert "high_severity_count" in d
        assert d["high_severity_count"] == 1

    def test_heuristic_reviewer_patterns(self):
        """Should detect dangerous patterns."""
        from src.synthesis.agents.reviewer import HeuristicReviewer

        reviewer = HeuristicReviewer()

        # Safe content
        safe = "Apply the clip after obtaining proximal control."
        warnings = reviewer.review(safe)
        assert len(warnings) == 0

        # Dangerous pattern
        dangerous = "Clip the aneurysm without proximal control."
        warnings = reviewer.review(dangerous)
        assert len(warnings) >= 1
        assert warnings[0].severity == "MEDIUM"


# =============================================================================
# ENRICHMENT MODELS TESTS
# =============================================================================

class TestEnrichmentModels:
    """Tests for src/synthesis/models_enrichment.py"""

    def test_gap_item_validation(self):
        """Should validate GapItem fields."""
        from src.synthesis.models_enrichment import GapItem

        gap = GapItem(
            gap_type="missing_data",
            priority="high",
            confidence=0.85,
            description="Information about MCA aneurysm surgical complications rates is missing"
        )

        assert gap.gap_type == "missing_data"
        assert gap.priority == "high"
        assert gap.confidence == 0.85

    def test_gap_analysis_result(self):
        """Should create GapAnalysisResult correctly."""
        from src.synthesis.models_enrichment import GapAnalysisResult, GapItem

        result = GapAnalysisResult(
            summary="Internal sources provide good coverage but lack recent trial data",
            gaps=[
                GapItem(
                    gap_type="recent_developments",
                    priority="medium",
                    confidence=0.75,
                    description="Recent clinical trials for aneurysm surgery outcomes are missing"
                )
            ],
            internal_coverage_score=0.7
        )

        assert len(result.gaps) == 1
        assert result.internal_coverage_score == 0.7

    def test_v3_synthesis_metadata(self):
        """Should track all V3 component usage."""
        from src.synthesis.models_enrichment import V3SynthesisMetadata

        metadata = V3SynthesisMetadata(
            mode="hybrid",
            enrichment_used=True,
            total_external_sources=3,
            conflict_count=1,
            high_severity_conflicts=0,
            requires_review=False
        )

        assert metadata.mode == "hybrid"
        assert metadata.enrichment_used is True
        assert metadata.conflict_count == 1

        # Test frontend conversion
        frontend_dict = metadata.to_frontend_dict()
        assert "enrichmentUsed" in frontend_dict
        assert frontend_dict["conflictCount"] == 1


# =============================================================================
# INTEGRATION SMOKE TEST
# =============================================================================

class TestV3IntegrationSmoke:
    """End-to-end smoke tests for V3 pipeline."""

    def test_all_v3_imports(self):
        """Should import all V3 components without error."""
        from src.synthesis import (
            # Semantic Router
            SemanticRouter,
            SectionPrototypes,
            RouteResult,
            KeywordFallbackRouter,
            # Figure Resolver
            SemanticFigureResolver,
            FigureMatch,
            LegacyFigureResolver,
            # Conflict Merger
            ConflictAwareMerger,
            MergeResult,
            DetectedConflict,
            ExtractedFact,
            merge_internal_external,
            # Adversarial Reviewer
            AdversarialReviewer,
            ReviewResult,
            ControversyWarning,
            HeuristicReviewer,
            # Enrichment Models
            GapItem,
            GapAnalysisResult,
            V3SynthesisMetadata,
        )

        # All imports successful
        assert SemanticRouter is not None
        assert ConflictAwareMerger is not None
        assert AdversarialReviewer is not None

    def test_shared_utilities_import(self):
        """Should import shared utilities."""
        from src.shared.parsing import (
            extract_and_parse_json,
            extract_json_string,
            strip_markdown_fences,
            find_json_boundaries,
            LLMParsingError,
        )
        from src.shared.mmr import (
            mmr_sort,
            mmr_rerank,
            MMRSelector,
        )

        assert extract_and_parse_json is not None
        assert extract_json_string is not None
        assert mmr_sort is not None

    def test_v3_pipeline_components_instantiate(self):
        """Should instantiate all V3 components."""
        from src.synthesis.router import KeywordFallbackRouter
        from src.synthesis.agents.reviewer import HeuristicReviewer
        from src.shared.mmr import MMRSelector

        router = KeywordFallbackRouter()
        reviewer = HeuristicReviewer()
        mmr = MMRSelector(lambda_mult=0.7)

        # Quick functionality check
        result = router.route_chunk("Patient positioned supine with microscope")
        assert result.section_name is not None

        warnings = reviewer.review("Safe surgical content here.")
        assert isinstance(warnings, list)

        # MMR with minimal data
        embeddings = [[float(x) for x in row] for row in np.random.randn(5, 32)]
        query = [float(x) for x in np.random.randn(32)]
        indices = mmr.select(query, embeddings, top_k=3)
        assert len(indices) == 3
