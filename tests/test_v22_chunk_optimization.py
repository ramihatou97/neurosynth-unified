"""
NeuroSynth v2.2 Chunk Optimization - Comprehensive Test Suite
=============================================================

Tests all components of the v2.2 enhanced chunking system:
1. chunk_config.py - Types, phases, safe-cut rules
2. quality_scorer.py - 4-dimension scoring, orphan detection
3. enhanced_chunker.py - Adaptive limits, complexity estimation
4. enhanced_models.py - Serialization, computed properties
"""

import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Any


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", error: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.error = error


class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []

    def add(self, name: str, passed: bool, message: str = "", error: str = ""):
        self.results.append(TestResult(name, passed, message, error))

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        return f"{self.name}: {passed}/{len(self.results)} passed, {failed} failed"


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(f"Assertion failed: {msg}")


def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{msg}: {item!r} not in {container!r}")


def assert_ge(actual, minimum, msg=""):
    if actual < minimum:
        raise AssertionError(f"{msg}: {actual} < {minimum}")


def assert_le(actual, maximum, msg=""):
    if actual > maximum:
        raise AssertionError(f"{msg}: {actual} > {maximum}")


def assert_approx(actual, expected, tolerance=0.01, msg=""):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{msg}: {actual} != {expected} (tolerance {tolerance})")


# =============================================================================
# MOCK CHUNK FOR TESTING
# =============================================================================

@dataclass
class MockChunk:
    """Mock chunk for testing quality scorer."""
    content: str
    chunk_type: Any = None
    entities: List[Any] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    specialty_tags: List[str] = field(default_factory=list)
    readability_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    type_specific_score: float = 0.0  # v2.2: Fourth dimension
    is_orphan: bool = False  # v2.2: Orphan status
    cuis: List[str] = field(default_factory=list)
    figure_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Computed quality score (v2.2: 4 dimensions + orphan penalty)."""
        base_score = (
            self.readability_score * 0.20 +
            self.coherence_score * 0.25 +
            self.completeness_score * 0.35 +
            self.type_specific_score * 0.20
        )
        if self.is_orphan:
            base_score = max(0.0, base_score - 0.20)
        return base_score


# =============================================================================
# TEST: chunk_config.py
# =============================================================================

def test_chunk_config():
    suite = TestSuite("chunk_config")

    try:
        from src.core.chunk_config import (
            ChunkType, SurgicalPhase, SafeCutRule, ChunkTypeConfig,
            get_type_config, get_all_safe_cut_rules, detect_surgical_phase,
            extract_step_number, TYPE_CONFIGS
        )

        # Test 1: ChunkType enum has expected values
        try:
            expected_types = ['procedure', 'anatomy', 'pathology', 'clinical',
                            'case', 'general', 'differential', 'imaging']
            for t in expected_types:
                ChunkType(t)
            suite.add("ChunkType_core_values", True, f"All {len(expected_types)} core types exist")
        except Exception as e:
            suite.add("ChunkType_core_values", False, error=str(e))

        # Test 2: ChunkType count
        try:
            assert_eq(len(ChunkType), 24, "ChunkType count")
            suite.add("ChunkType_count", True, "24 types defined")
        except Exception as e:
            suite.add("ChunkType_count", False, error=str(e))

        # Test 3: SurgicalPhase enum
        try:
            expected_phases = ['indication', 'positioning', 'exposure', 'approach',
                             'resection', 'closure', 'other']
            for p in expected_phases:
                SurgicalPhase(p)
            assert_eq(len(SurgicalPhase), 12, "SurgicalPhase count")
            suite.add("SurgicalPhase_values", True, "12 phases defined")
        except Exception as e:
            suite.add("SurgicalPhase_values", False, error=str(e))

        # Test 4: SafeCutRule matching logic
        try:
            rule = SafeCutRule(
                name="test",
                description="Test rule",
                prev_contains=["warning"],
                next_contains=["can cause"]
            )
            # Should match
            assert_true(rule.matches("Warning: be careful", "This can cause damage"),
                       "Should match warning->can cause")
            # Should not match - missing prev
            assert_true(not rule.matches("Normal sentence", "This can cause damage"),
                       "Should not match without prev")
            # Should not match - missing next
            assert_true(not rule.matches("Warning: be careful", "Normal sentence"),
                       "Should not match without next")
            suite.add("SafeCutRule_matching", True, "Matching logic correct")
        except Exception as e:
            suite.add("SafeCutRule_matching", False, error=str(e))

        # Test 5: SafeCutRule with patterns
        try:
            rule = SafeCutRule(
                name="step_test",
                description="Step pattern test",
                prev_patterns=[r"Step\s+\d+"],
                next_patterns=[r"^\s*(This|The)"]
            )
            assert_true(rule.matches("Step 1: Open the dura.", "This allows exposure."),
                       "Pattern matching")
            assert_true(not rule.matches("Random text.", "This allows exposure."),
                       "Should not match without step")
            suite.add("SafeCutRule_patterns", True, "Pattern matching works")
        except Exception as e:
            suite.add("SafeCutRule_patterns", False, error=str(e))

        # Test 6: get_type_config returns correct type
        try:
            config = get_type_config(ChunkType.PROCEDURE)
            assert_eq(config.chunk_type, ChunkType.PROCEDURE, "Config type")
            assert_eq(config.target_tokens, 750, "Procedure target tokens")
            assert_ge(len(config.safe_cut_rules), 1, "Has safe-cut rules")
            suite.add("get_type_config_procedure", True, "Procedure config correct")
        except Exception as e:
            suite.add("get_type_config_procedure", False, error=str(e))

        # Test 7: get_type_config fallback to GENERAL
        try:
            # Try a type not in TYPE_CONFIGS
            config = get_type_config(ChunkType.DOSAGE)  # Not in TYPE_CONFIGS
            assert_eq(config.chunk_type, ChunkType.GENERAL, "Should fallback to GENERAL")
            suite.add("get_type_config_fallback", True, "Fallback to GENERAL works")
        except Exception as e:
            suite.add("get_type_config_fallback", False, error=str(e))

        # Test 8: get_all_safe_cut_rules
        try:
            rules = get_all_safe_cut_rules()
            assert_ge(len(rules), 5, "At least 5 rules")
            # Check sorted by priority (descending)
            for i in range(1, len(rules)):
                assert_ge(rules[i-1].priority, rules[i].priority, "Priority order")
            suite.add("get_all_safe_cut_rules", True, f"{len(rules)} rules, sorted by priority")
        except Exception as e:
            suite.add("get_all_safe_cut_rules", False, error=str(e))

        # Test 9: detect_surgical_phase
        try:
            phase = detect_surgical_phase("Patient positioned supine with Mayfield head holder")
            assert_eq(phase, SurgicalPhase.POSITIONING, "Positioning detection")

            phase = detect_surgical_phase("Make a linear skin incision over the mastoid")
            assert_eq(phase, SurgicalPhase.EXPOSURE, "Exposure detection")

            phase = detect_surgical_phase("Complete tumor resection was achieved")
            assert_eq(phase, SurgicalPhase.RESECTION, "Resection detection")

            phase = detect_surgical_phase("Random unrelated text")
            assert_eq(phase, SurgicalPhase.OTHER, "Fallback to OTHER")

            suite.add("detect_surgical_phase", True, "Phase detection works")
        except Exception as e:
            suite.add("detect_surgical_phase", False, error=str(e))

        # Test 10: extract_step_number
        try:
            assert_eq(extract_step_number("Step 3: Open the dura."), 3, "Explicit step")
            assert_eq(extract_step_number("1. Make the incision."), 1, "Numbered step")
            assert_eq(extract_step_number("First, position the patient."), 1, "Ordinal first")
            assert_eq(extract_step_number("Third, close the wound."), 3, "Ordinal third")
            assert_eq(extract_step_number("The dura is then opened."), None, "No step")
            suite.add("extract_step_number", True, "Step extraction works")
        except Exception as e:
            suite.add("extract_step_number", False, error=str(e))

        # Test 11: TYPE_CONFIGS completeness
        try:
            # Check that common types have configs
            required_types = [ChunkType.PROCEDURE, ChunkType.ANATOMY, ChunkType.PATHOLOGY,
                            ChunkType.CLINICAL, ChunkType.GENERAL]
            for ct in required_types:
                assert_in(ct, TYPE_CONFIGS, f"{ct} in TYPE_CONFIGS")
            suite.add("TYPE_CONFIGS_completeness", True, f"{len(TYPE_CONFIGS)} types configured")
        except Exception as e:
            suite.add("TYPE_CONFIGS_completeness", False, error=str(e))

    except ImportError as e:
        suite.add("import", False, error=f"Import failed: {e}")

    return suite


# =============================================================================
# TEST: quality_scorer.py
# =============================================================================

def test_quality_scorer():
    suite = TestSuite("quality_scorer")

    try:
        from src.core.quality_scorer import (
            ChunkQualityScorer, QualityConfig, get_quality_scorer,
            ORPHAN_PATTERNS, ELEMENT_PATTERNS, TYPE_REQUIREMENTS
        )
        from src.core.chunk_config import ChunkType

        # Test 1: Weight sum validation
        try:
            config = QualityConfig()
            total = (config.readability_weight + config.coherence_weight +
                    config.completeness_weight + config.type_specific_weight)
            assert_approx(total, 1.0, 0.001, "Weights should sum to 1.0")
            suite.add("weight_sum", True, f"Weights sum to {total}")
        except Exception as e:
            suite.add("weight_sum", False, error=str(e))

        # Test 2: Orphan pattern detection
        try:
            scorer = get_quality_scorer()

            # Should be orphans
            orphan_tests = [
                "Step 2: Continue the dissection.",
                "2. Open the dura next.",
                "Then proceed with closure.",
                "Next, remove the retractor.",
                "Subsequently, the wound was closed.",
                "Third, check the hemostasis.",
            ]
            for text in orphan_tests:
                assert_true(scorer._is_orphan(text), f"Should be orphan: {text[:30]}...")

            # Should NOT be orphans
            non_orphan_tests = [
                "Step 1: Begin the procedure.",
                "1. Make the initial incision.",
                "First, position the patient.",
                "The facial nerve is identified.",
            ]
            for text in non_orphan_tests:
                assert_true(not scorer._is_orphan(text), f"Should NOT be orphan: {text[:30]}...")

            suite.add("orphan_detection", True, "Orphan patterns work correctly")
        except Exception as e:
            suite.add("orphan_detection", False, error=str(e))

        # Test 3: Element detection
        try:
            scorer = get_quality_scorer()

            # Procedure elements
            proc_text = "Dissect the nerve carefully using bipolar forceps to avoid injury."
            elements = scorer._detect_elements(proc_text)
            assert_in('action', elements, "Should detect action")
            assert_in('anatomy', elements, "Should detect anatomy")
            assert_in('instrument', elements, "Should detect instrument")

            # Anatomy elements
            anat_text = "The facial nerve courses lateral to the tumor and innervates the face."
            elements = scorer._detect_elements(anat_text)
            assert_in('anatomy', elements, "Should detect anatomy")
            assert_in('function', elements, "Should detect function")

            suite.add("element_detection", True, f"Detected elements correctly")
        except Exception as e:
            suite.add("element_detection", False, error=str(e))

        # Test 4: TYPE_REQUIREMENTS coverage
        try:
            required_types = ['procedure', 'anatomy', 'pathology', 'clinical', 'case', 'general']
            for t in required_types:
                assert_in(t, TYPE_REQUIREMENTS, f"{t} in TYPE_REQUIREMENTS")

            # Check procedure has expected requirements
            assert_in('action', TYPE_REQUIREMENTS['procedure'], "Procedure needs action")
            assert_in('anatomy', TYPE_REQUIREMENTS['procedure'], "Procedure needs anatomy")

            suite.add("TYPE_REQUIREMENTS", True, "All types have requirements")
        except Exception as e:
            suite.add("TYPE_REQUIREMENTS", False, error=str(e))

        # Test 5: compute_readability
        try:
            scorer = get_quality_scorer()

            # Good readability - optimal sentence length
            good_chunk = MockChunk(
                content="The facial nerve is identified at the stylomastoid foramen. "
                       "It courses anteriorly through the parotid gland. "
                       "The nerve branches into five main divisions.",
                chunk_type=ChunkType.ANATOMY
            )
            score = scorer.compute_readability(good_chunk)
            assert_ge(score, 0.5, "Good text should score >= 0.5")
            assert_le(score, 1.0, "Score should be <= 1.0")

            # Empty content
            empty_chunk = MockChunk(content="", chunk_type=ChunkType.GENERAL)
            score = scorer.compute_readability(empty_chunk)
            assert_eq(score, 0.0, "Empty content should score 0")

            suite.add("compute_readability", True, "Readability scoring works")
        except Exception as e:
            suite.add("compute_readability", False, error=str(e))

        # Test 6: compute_coherence
        try:
            scorer = get_quality_scorer()

            # Coherent text with flow markers
            coherent_chunk = MockChunk(
                content="The tumor is located in the cerebellopontine angle. "
                       "Furthermore, it compresses the facial nerve. "
                       "Therefore, the surgical approach must preserve nerve function.",
                chunk_type=ChunkType.PATHOLOGY
            )
            score = scorer.compute_coherence(coherent_chunk)
            assert_ge(score, 0.5, "Coherent text should score >= 0.5")

            # Single sentence - should still be coherent
            single_chunk = MockChunk(
                content="The nerve is preserved.",
                chunk_type=ChunkType.PROCEDURE
            )
            score = scorer.compute_coherence(single_chunk)
            assert_ge(score, 0.8, "Single sentence should be coherent")

            suite.add("compute_coherence", True, "Coherence scoring works")
        except Exception as e:
            suite.add("compute_coherence", False, error=str(e))

        # Test 7: compute_completeness
        try:
            scorer = get_quality_scorer()

            # Complete text - ends properly, no dangling refs
            complete_chunk = MockChunk(
                content="The facial nerve is carefully preserved during tumor resection. "
                       "Continuous monitoring ensures functional integrity.",
                chunk_type=ChunkType.PROCEDURE
            )
            score = scorer.compute_completeness(complete_chunk)
            assert_ge(score, 0.4, "Complete text should score >= 0.4")

            # Dangling reference - starts with "This"
            dangling_chunk = MockChunk(
                content="This technique is essential for preservation. "
                       "The nerve must be carefully dissected.",
                chunk_type=ChunkType.PROCEDURE
            )
            score_dangling = scorer.compute_completeness(dangling_chunk)
            # Should be penalized but not zero
            assert_le(score_dangling, score, "Dangling reference should lower score")

            suite.add("compute_completeness", True, "Completeness scoring works")
        except Exception as e:
            suite.add("compute_completeness", False, error=str(e))

        # Test 8: compute_type_specific
        try:
            scorer = get_quality_scorer()

            # Procedure chunk with required elements
            proc_chunk = MockChunk(
                content="Dissect the facial nerve using bipolar forceps to avoid thermal injury. "
                       "The dissector is used to separate the tumor from the nerve.",
                chunk_type=ChunkType.PROCEDURE
            )
            score = scorer.compute_type_specific(proc_chunk)
            assert_ge(score, 0.5, "Procedure with elements should score >= 0.5")

            # General chunk (no requirements)
            gen_chunk = MockChunk(
                content="This is general information about neurosurgery.",
                chunk_type=ChunkType.GENERAL
            )
            score = scorer.compute_type_specific(gen_chunk)
            assert_eq(score, 0.8, "General type should return 0.8 baseline")

            suite.add("compute_type_specific", True, "Type-specific scoring works")
        except Exception as e:
            suite.add("compute_type_specific", False, error=str(e))

        # Test 9: score_chunk with orphan penalty
        try:
            scorer = get_quality_scorer()

            # Non-orphan chunk
            normal_chunk = MockChunk(
                content="Step 1: Begin the dissection of the facial nerve. "
                       "The nerve is identified at the stylomastoid foramen. "
                       "Use bipolar forceps to coagulate small vessels.",
                chunk_type=ChunkType.PROCEDURE
            )
            scorer.score_chunk(normal_chunk)
            normal_score = normal_chunk.quality_score

            # Orphan chunk (starts with Step 2)
            orphan_chunk = MockChunk(
                content="Step 2: Continue the dissection of the facial nerve. "
                       "The nerve is identified at the stylomastoid foramen. "
                       "Use bipolar forceps to coagulate small vessels.",
                chunk_type=ChunkType.PROCEDURE
            )
            scorer.score_chunk(orphan_chunk)
            orphan_score = orphan_chunk.quality_score

            # Orphan should be penalized
            assert_true(orphan_score < normal_score,
                       f"Orphan ({orphan_score:.2f}) should score lower than normal ({normal_score:.2f})")

            # Penalty should be approximately -0.20
            expected_diff = 0.20
            actual_diff = normal_score - orphan_score
            assert_ge(actual_diff, expected_diff - 0.05,
                     f"Penalty diff should be ~{expected_diff}, got {actual_diff:.2f}")

            suite.add("score_chunk_orphan_penalty", True,
                     f"Orphan penalty applied: {actual_diff:.2f}")
        except Exception as e:
            suite.add("score_chunk_orphan_penalty", False, error=str(e))

        # Test 10: get_quality_scorer singleton
        try:
            scorer1 = get_quality_scorer()
            scorer2 = get_quality_scorer()
            assert_true(scorer1 is scorer2, "Should return same instance")

            # New config should create new instance
            scorer3 = get_quality_scorer(QualityConfig(readability_weight=0.30))
            assert_true(scorer3 is not scorer1, "New config should create new instance")

            suite.add("get_quality_scorer_singleton", True, "Singleton pattern works")
        except Exception as e:
            suite.add("get_quality_scorer_singleton", False, error=str(e))

    except ImportError as e:
        suite.add("import", False, error=f"Import failed: {e}")

    return suite


# =============================================================================
# TEST: enhanced_chunker.py
# =============================================================================

def test_enhanced_chunker():
    suite = TestSuite("enhanced_chunker")

    try:
        from src.core.enhanced_chunker import (
            EnhancedNeuroChunker, EnhancedChunkerConfig, ContentComplexity,
            CATEGORY_SPECIALTY_MAP, SPECIALTY_PRIORITY
        )
        from src.core.chunk_config import ChunkType, get_type_config

        # Test 1: ContentComplexity values
        try:
            assert_eq(ContentComplexity.SIMPLE.value, 0.85, "SIMPLE factor")
            assert_eq(ContentComplexity.MODERATE.value, 1.0, "MODERATE factor")
            assert_eq(ContentComplexity.COMPLEX.value, 1.15, "COMPLEX factor")
            assert_eq(ContentComplexity.HIGHLY_COMPLEX.value, 1.30, "HIGHLY_COMPLEX factor")
            suite.add("ContentComplexity_values", True, "All complexity factors correct")
        except Exception as e:
            suite.add("ContentComplexity_values", False, error=str(e))

        # Test 2: Complexity estimation
        try:
            chunker = EnhancedNeuroChunker()

            # Simple text
            simple = "The patient was positioned supine."
            complexity = chunker._estimate_complexity(simple)
            assert_eq(complexity, ContentComplexity.SIMPLE, "Simple text")

            # Moderate - has spatial terms
            moderate = "The nerve is lateral to the artery and medial to the bone."
            complexity = chunker._estimate_complexity(moderate)
            assert_in(complexity, [ContentComplexity.MODERATE, ContentComplexity.SIMPLE],
                     "Moderate text")

            # Complex - steps + spatial + measurements
            complex_text = """Step 1: Make incision 3 cm lateral to midline.
            Step 2: Dissect through layers. The nerve is lateral and superior.
            Measure 5 mm from the foramen. Grade III tumor identified."""
            complexity = chunker._estimate_complexity(complex_text)
            assert_in(complexity, [ContentComplexity.COMPLEX, ContentComplexity.HIGHLY_COMPLEX],
                     "Complex text")

            suite.add("complexity_estimation", True, "Complexity estimation works")
        except Exception as e:
            suite.add("complexity_estimation", False, error=str(e))

        # Test 3: Adaptive limits
        try:
            chunker = EnhancedNeuroChunker()
            type_config = get_type_config(ChunkType.PROCEDURE)  # target=750

            # Simple content - should reduce limits
            simple_text = "A simple procedure description."
            limits = chunker._get_adaptive_limits(type_config, simple_text)
            assert_le(limits['target'], type_config.target_tokens,
                     "Simple should reduce target")

            # With adaptive disabled
            config = EnhancedChunkerConfig(enable_adaptive_limits=False)
            chunker_no_adapt = EnhancedNeuroChunker(config=config)
            limits_fixed = chunker_no_adapt._get_adaptive_limits(type_config, simple_text)
            assert_eq(limits_fixed['target'], type_config.target_tokens,
                     "Disabled adaptive should use fixed")

            suite.add("adaptive_limits", True, "Adaptive limits work")
        except Exception as e:
            suite.add("adaptive_limits", False, error=str(e))

        # Test 4: Sentence splitting
        try:
            chunker = EnhancedNeuroChunker()

            text = "Dr. Smith performed surgery. The patient recovered. Fig. 1 shows results."
            sentences = chunker._split_sentences(text)

            # Should preserve Dr. and Fig. abbreviations
            assert_ge(len(sentences), 1, "Should produce sentences")
            # Check that sentences end with periods
            for s in sentences:
                assert_true(s.endswith('.') or s.endswith('!') or s.endswith('?'),
                           f"Sentence should end properly: {s}")

            suite.add("sentence_splitting", True, f"{len(sentences)} sentences parsed")
        except Exception as e:
            suite.add("sentence_splitting", False, error=str(e))

        # Test 5: Dependency checking
        try:
            chunker = EnhancedNeuroChunker()

            # Dependent sentences
            dependent_tests = [
                "This technique is critical.",
                "However, complications may occur.",
                "Therefore, monitoring is essential.",
                "Step 2: Continue dissection.",
                "Then close the wound.",
            ]
            for sent in dependent_tests:
                assert_true(chunker._check_dependency(sent),
                           f"Should be dependent: {sent[:20]}...")

            # Independent sentences
            independent_tests = [
                "The facial nerve exits the stylomastoid foramen.",
                "A linear incision is made.",
                "Complications include facial weakness.",
            ]
            for sent in independent_tests:
                assert_true(not chunker._check_dependency(sent),
                           f"Should be independent: {sent[:20]}...")

            suite.add("dependency_checking", True, "Dependency detection works")
        except Exception as e:
            suite.add("dependency_checking", False, error=str(e))

        # Test 6: Type detection from title
        try:
            chunker = EnhancedNeuroChunker()

            test_cases = [
                ("Surgical Technique", ChunkType.PROCEDURE),
                ("Pterional Approach", ChunkType.PROCEDURE),
                ("Anatomy of the Facial Nerve", ChunkType.ANATOMY),
                ("Tumor Pathology", ChunkType.PATHOLOGY),
                ("Clinical Presentation", ChunkType.CLINICAL),
                ("Case Report", ChunkType.CASE),
                ("Differential Diagnosis", ChunkType.DIFFERENTIAL),
                ("MRI Imaging Features", ChunkType.IMAGING),
                ("Random Title", ChunkType.GENERAL),
            ]

            for title, expected in test_cases:
                detected = chunker._detect_type_from_title(title)
                assert_eq(detected, expected, f"Title '{title}'")

            suite.add("type_detection_from_title", True, f"{len(test_cases)} cases correct")
        except Exception as e:
            suite.add("type_detection_from_title", False, error=str(e))

        # Test 7: Safe cut checking
        try:
            chunker = EnhancedNeuroChunker()
            type_config = get_type_config(ChunkType.PROCEDURE)

            # Should NOT be safe to cut (instrument -> action)
            prev = "The bipolar forceps are used."
            next_sent = "To coagulate the vessel carefully."
            assert_true(not chunker._is_safe_cut(prev, next_sent, type_config),
                       "Should not cut between instrument and action")

            # Should be safe to cut
            prev = "The tumor was removed completely."
            next_sent = "Postoperative imaging confirmed resection."
            assert_true(chunker._is_safe_cut(prev, next_sent, type_config),
                       "Should be safe to cut between unrelated sentences")

            suite.add("safe_cut_checking", True, "Safe cut detection works")
        except Exception as e:
            suite.add("safe_cut_checking", False, error=str(e))

        # Test 8: Pitfall detection
        try:
            chunker = EnhancedNeuroChunker()

            pitfall_texts = [
                "Avoid injury to the facial nerve.",
                "Caution: the vessel is fragile.",
                "Warning: this can cause complications.",
                "A critical pitfall is bleeding.",
                "Pearl: always identify the nerve first.",
            ]
            for text in pitfall_texts:
                assert_true(chunker._detect_pitfall(text),
                           f"Should detect pitfall: {text[:30]}...")

            non_pitfall = "The nerve courses laterally."
            assert_true(not chunker._detect_pitfall(non_pitfall),
                       "Should not detect pitfall in normal text")

            suite.add("pitfall_detection", True, "Pitfall detection works")
        except Exception as e:
            suite.add("pitfall_detection", False, error=str(e))

        # Test 9: Teaching point detection
        try:
            chunker = EnhancedNeuroChunker()

            teaching_texts = [
                "This lesson demonstrates proper technique.",
                "Remember to check hemostasis.",
                "The key concept is nerve preservation.",
                "This illustrates the importance of exposure.",
            ]
            for text in teaching_texts:
                assert_true(chunker._detect_teaching_point(text),
                           f"Should detect teaching: {text[:30]}...")

            suite.add("teaching_point_detection", True, "Teaching point detection works")
        except Exception as e:
            suite.add("teaching_point_detection", False, error=str(e))

        # Test 10: Full chunk_section integration
        try:
            chunker = EnhancedNeuroChunker()

            section_text = """
            Step 1: Position the patient supine with the head in a Mayfield clamp.
            The head is rotated 30 degrees to the contralateral side.
            Step 2: Make a curvilinear incision behind the ear.
            The incision extends from the mastoid tip superiorly.
            Dissect through the soft tissues to expose the mastoid bone.
            Step 3: Perform the mastoidectomy using a high-speed drill.
            The sigmoid sinus is identified and protected.
            Caution: avoid injury to the facial nerve in the fallopian canal.
            """

            chunks = chunker.chunk_section(
                section_text=section_text,
                section_title="Surgical Technique",
                page_num=42,
                doc_id="test-doc-123"
            )

            assert_ge(len(chunks), 1, "Should produce at least 1 chunk")

            # Check chunk structure
            chunk = chunks[0]
            if isinstance(chunk, dict):
                assert_in('content', chunk, "Chunk should have content")
                assert_in('chunk_type', chunk, "Chunk should have type")
                assert_in('document_id', chunk, "Chunk should have doc_id")
            else:
                assert_true(hasattr(chunk, 'content'), "Chunk should have content")

            suite.add("chunk_section_integration", True, f"Produced {len(chunks)} chunks")
        except Exception as e:
            suite.add("chunk_section_integration", False, error=str(e))

        # Test 11: CATEGORY_SPECIALTY_MAP coverage
        try:
            assert_ge(len(CATEGORY_SPECIALTY_MAP), 10, "At least 10 mappings")
            assert_in("ANATOMY_VASCULAR_ARTERIAL", CATEGORY_SPECIALTY_MAP, "Vascular anatomy")
            assert_in("PATHOLOGY_TUMOR", CATEGORY_SPECIALTY_MAP, "Tumor pathology")
            suite.add("CATEGORY_SPECIALTY_MAP", True, f"{len(CATEGORY_SPECIALTY_MAP)} mappings")
        except Exception as e:
            suite.add("CATEGORY_SPECIALTY_MAP", False, error=str(e))

    except ImportError as e:
        suite.add("import", False, error=f"Import failed: {e}")

    return suite


# =============================================================================
# TEST: enhanced_models.py
# =============================================================================

def test_enhanced_models():
    suite = TestSuite("enhanced_models")

    try:
        from src.shared.enhanced_models import (
            EnhancedSemanticChunk, ChunkType, SurgicalPhase, get_schema_additions
        )

        # Test 1: EnhancedSemanticChunk creation
        try:
            chunk = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test content about facial nerve.",
                title="Test Title",
                section_path=["Chapter 1", "Section 1"],
                page_start=10,
                page_end=11,
                chunk_type=ChunkType.ANATOMY
            )
            assert_eq(chunk.id, "test-123", "ID")
            assert_eq(chunk.document_id, "doc-456", "Document ID")
            assert_eq(chunk.chunk_type, ChunkType.ANATOMY, "Chunk type")
            suite.add("EnhancedSemanticChunk_creation", True, "Creation works")
        except Exception as e:
            suite.add("EnhancedSemanticChunk_creation", False, error=str(e))

        # Test 2: Enhanced metadata fields
        try:
            chunk = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test content",
                title="Test",
                section_path=[],
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.PROCEDURE,
                surgical_phase="exposure",
                step_number=3,
                step_sequence="3_of_8",
                has_pitfall=True,
                has_teaching_point=True,
                grading_scale="spetzler_martin",
                grade_value="III",
                molecular_markers=["IDH1", "MGMT"]
            )

            assert_eq(chunk.surgical_phase, "exposure", "Surgical phase")
            assert_eq(chunk.step_number, 3, "Step number")
            assert_eq(chunk.step_sequence, "3_of_8", "Step sequence")
            assert_eq(chunk.has_pitfall, True, "Has pitfall")
            assert_eq(chunk.has_teaching_point, True, "Has teaching point")
            assert_eq(chunk.grading_scale, "spetzler_martin", "Grading scale")
            assert_eq(chunk.molecular_markers, ["IDH1", "MGMT"], "Molecular markers")

            suite.add("enhanced_metadata_fields", True, "All enhanced fields work")
        except Exception as e:
            suite.add("enhanced_metadata_fields", False, error=str(e))

        # Test 3: Computed properties
        try:
            chunk = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test",
                title="Test",
                section_path=[],
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.PROCEDURE,
                readability_score=0.8,
                coherence_score=0.7,
                completeness_score=0.9,
                has_pitfall=True,
                step_number=1,
                molecular_markers=["IDH1"]
            )

            # quality_score = 0.8*0.25 + 0.7*0.35 + 0.9*0.40 = 0.805
            expected_quality = 0.8*0.25 + 0.7*0.35 + 0.9*0.40
            assert_approx(chunk.quality_score, expected_quality, 0.001, "Quality score")

            assert_eq(chunk.is_high_value, True, "is_high_value (has_pitfall)")
            assert_eq(chunk.has_step_context, True, "has_step_context")
            assert_eq(chunk.is_molecular_pathology, True, "is_molecular_pathology")

            suite.add("computed_properties", True, "All computed properties work")
        except Exception as e:
            suite.add("computed_properties", False, error=str(e))

        # Test 4: to_dict serialization
        try:
            chunk = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test content",
                title="Test Title",
                section_path=["Ch1"],
                page_start=1,
                page_end=2,
                chunk_type=ChunkType.PROCEDURE,
                has_pitfall=True,
                molecular_markers=["IDH1"]
            )

            d = chunk.to_dict()
            assert_eq(d['id'], "test-123", "ID in dict")
            assert_eq(d['chunk_type'], "procedure", "Chunk type serialized")
            assert_eq(d['has_pitfall'], True, "has_pitfall in dict")
            assert_in('quality_score', d, "Computed quality_score in dict")
            assert_in('is_high_value', d, "Computed is_high_value in dict")

            suite.add("to_dict_serialization", True, "to_dict works")
        except Exception as e:
            suite.add("to_dict_serialization", False, error=str(e))

        # Test 5: to_db serialization
        try:
            chunk = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test",
                title="Test",
                section_path=["Ch1", "Sec1"],
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.ANATOMY,
                molecular_markers=["IDH1", "MGMT"]
            )

            db_dict = chunk.to_db()

            # Lists should be JSON strings
            import json
            assert_eq(json.loads(db_dict['section_path']), ["Ch1", "Sec1"], "section_path as JSON")
            assert_eq(json.loads(db_dict['molecular_markers']), ["IDH1", "MGMT"], "molecular_markers as JSON")

            # Computed properties should be removed
            assert_true('quality_score' not in db_dict, "quality_score removed")
            assert_true('is_high_value' not in db_dict, "is_high_value removed")

            suite.add("to_db_serialization", True, "to_db works")
        except Exception as e:
            suite.add("to_db_serialization", False, error=str(e))

        # Test 6: from_dict deserialization
        try:
            data = {
                'id': 'test-123',
                'document_id': 'doc-456',
                'content': 'Test content',
                'title': 'Test Title',
                'section_path': ['Ch1'],
                'page_start': 1,
                'page_end': 2,
                'chunk_type': 'procedure',
                'has_pitfall': True,
                'step_number': 3,
                'molecular_markers': ['IDH1']
            }

            chunk = EnhancedSemanticChunk.from_dict(data)
            assert_eq(chunk.id, 'test-123', "ID")
            assert_eq(chunk.chunk_type, ChunkType.PROCEDURE, "Chunk type enum")
            assert_eq(chunk.has_pitfall, True, "has_pitfall")
            assert_eq(chunk.step_number, 3, "step_number")

            suite.add("from_dict_deserialization", True, "from_dict works")
        except Exception as e:
            suite.add("from_dict_deserialization", False, error=str(e))

        # Test 7: Round-trip serialization
        try:
            original = EnhancedSemanticChunk(
                id="test-123",
                document_id="doc-456",
                content="Test content for round trip",
                title="Round Trip Test",
                section_path=["Ch1", "Sec1", "SubSec1"],
                page_start=42,
                page_end=45,
                chunk_type=ChunkType.PATHOLOGY,
                surgical_phase="resection",
                step_number=5,
                has_pitfall=True,
                has_teaching_point=True,
                grading_scale="who",
                grade_value="IV",
                molecular_markers=["IDH1", "MGMT", "1p/19q"],
                readability_score=0.85,
                coherence_score=0.90
            )

            # Round trip
            d = original.to_dict()
            restored = EnhancedSemanticChunk.from_dict(d)

            assert_eq(restored.id, original.id, "ID preserved")
            assert_eq(restored.chunk_type, original.chunk_type, "chunk_type preserved")
            assert_eq(restored.surgical_phase, original.surgical_phase, "surgical_phase preserved")
            assert_eq(restored.step_number, original.step_number, "step_number preserved")
            assert_eq(restored.has_pitfall, original.has_pitfall, "has_pitfall preserved")
            assert_eq(restored.molecular_markers, original.molecular_markers, "molecular_markers preserved")

            suite.add("round_trip_serialization", True, "Round trip preserves data")
        except Exception as e:
            suite.add("round_trip_serialization", False, error=str(e))

        # Test 8: get_schema_additions SQL
        try:
            sql = get_schema_additions()
            assert_true(len(sql) > 100, "SQL should be non-trivial")
            assert_in("surgical_phase", sql, "Should include surgical_phase")
            assert_in("step_number", sql, "Should include step_number")
            assert_in("has_pitfall", sql, "Should include has_pitfall")
            assert_in("molecular_markers", sql, "Should include molecular_markers")
            assert_in("CREATE INDEX", sql, "Should include indexes")
            suite.add("get_schema_additions", True, "Schema SQL generated")
        except Exception as e:
            suite.add("get_schema_additions", False, error=str(e))

        # Test 9: ChunkType in enhanced_models matches chunk_config
        try:
            from src.core.chunk_config import ChunkType as ConfigChunkType

            # Check that core types exist in both
            core_types = ['procedure', 'anatomy', 'pathology', 'clinical', 'case', 'general']
            for t in core_types:
                ChunkType(t)  # enhanced_models
                ConfigChunkType(t)  # chunk_config

            suite.add("ChunkType_consistency", True, "ChunkType consistent between modules")
        except Exception as e:
            suite.add("ChunkType_consistency", False, error=str(e))

    except ImportError as e:
        suite.add("import", False, error=f"Import failed: {e}")

    return suite


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("NeuroSynth v2.2 Chunk Optimization - Comprehensive Test Suite")
    print("=" * 70)
    print()

    all_suites = []

    # Run all test suites
    print("Running tests...")
    print()

    suites = [
        ("chunk_config", test_chunk_config),
        ("quality_scorer", test_quality_scorer),
        ("enhanced_chunker", test_enhanced_chunker),
        ("enhanced_models", test_enhanced_models),
    ]

    for name, test_fn in suites:
        try:
            suite = test_fn()
            all_suites.append(suite)
        except Exception as e:
            print(f"FATAL: {name} crashed: {e}")
            traceback.print_exc()

    # Print results
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    total_passed = 0
    total_failed = 0
    issues = []

    for suite in all_suites:
        print(f"\n{suite.summary()}")
        for result in suite.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}", end="")
            if result.message:
                print(f" - {result.message}", end="")
            if result.error:
                print(f" - ERROR: {result.error}", end="")
            print()

            if result.passed:
                total_passed += 1
            else:
                total_failed += 1
                issues.append((suite.name, result.name, result.error or result.message))

    print()
    print("=" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if issues:
        print()
        print("ISSUES TO FIX:")
        for suite_name, test_name, error in issues:
            print(f"  - {suite_name}/{test_name}: {error}")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
