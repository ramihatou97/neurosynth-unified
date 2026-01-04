"""
NeuroSynth v2.2 - Enhanced Medical-Aware Semantic Chunker
==========================================================

Enhanced chunking with type-specific configuration for optimal
neurosurgical synthesis and QA quality.

Key enhancements over v2.1:
1. Type-specific token limits and overlap
2. Comprehensive safe-cut rules per content type
3. Surgical phase detection
4. Step number extraction and preservation
5. Enhanced metadata capture
6. Complexity-based adaptive token limits (NEW in v2.2)
7. Quality gates for chunk validation (NEW in v2.2)

Usage:
    from src.core.enhanced_chunker import EnhancedNeuroChunker

    chunker = EnhancedNeuroChunker()
    chunks = chunker.chunk_section(
        section_text="...",
        section_title="Surgical Technique",
        page_num=42,
        doc_id="doc-123"
    )
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from uuid import uuid4
from enum import Enum

from src.core.chunk_config import (
    ChunkType, SurgicalPhase, ChunkTypeConfig, SafeCutRule,
    get_type_config, get_all_safe_cut_rules, detect_surgical_phase,
    extract_step_number,
)
from src.core.quality_scorer import get_quality_scorer


# =============================================================================
# CONTENT COMPLEXITY (NEW in v2.2)
# =============================================================================

class ContentComplexity(Enum):
    """Content complexity levels with scaling factors."""
    SIMPLE = 0.85
    MODERATE = 1.0
    COMPLEX = 1.15
    HIGHLY_COMPLEX = 1.30


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancedChunkerConfig:
    """Master configuration for the enhanced chunker."""
    default_target_tokens: int = 600
    default_max_tokens: int = 1000
    default_min_tokens: int = 150
    default_overlap_sentences: int = 2
    enable_early_type_detection: bool = True
    enable_quality_scoring: bool = True
    enable_adaptive_limits: bool = True  # NEW in v2.2
    enable_quality_gates: bool = True    # NEW in v2.2
    min_quality_threshold: float = 0.4
    extract_surgical_phase: bool = True
    extract_step_numbers: bool = True
    detect_pitfalls: bool = True
    detect_teaching_points: bool = True


# =============================================================================
# SPECIALTY MAPPING
# =============================================================================

CATEGORY_SPECIALTY_MAP = {
    "ANATOMY_VASCULAR_ARTERIAL": "vascular",
    "ANATOMY_VASCULAR_VENOUS": "vascular",
    "ANATOMY_SKULL_BASE": "skull_base",
    "ANATOMY_CRANIAL_NERVES": "skull_base",
    "ANATOMY_SPINE_BONE": "spine",
    "ANATOMY_SPINE_NEURAL": "spine",
    "PATHOLOGY_VASCULAR": "vascular",
    "PATHOLOGY_TUMOR": "oncology",
    "PATHOLOGY_TRAUMA": "trauma",
    "PATHOLOGY_FUNCTIONAL": "functional",
    "PROCEDURE_APPROACH": "general",
    "PROCEDURE_ACTION": "general",
}

SPECIALTY_PRIORITY = [
    "vascular", "skull_base", "spine", "oncology",
    "functional", "trauma", "peripheral", "pediatric", "general"
]


# =============================================================================
# ENHANCED CHUNKER CLASS
# =============================================================================

class EnhancedNeuroChunker:
    """
    Enhanced semantic chunker with type-specific optimization.
    Includes complexity-based adaptive limits (v2.2).
    """

    SENTENCE_SPLIT_PATTERN = re.compile(
        r'(?<!\bDr)(?<!\bFig)(?<!\bNo)(?<!\bal)(?<!\bapprox)(?<!\bvs)'
        r'(?<!\bet)(?<!\bi\.e)(?<!\be\.g)(?<!\bSt)(?<!\bVol)'
        r'\.\s+(?=[A-Z])'
    )

    PITFALL_PATTERNS = [
        r"\b(avoid|caution|warning|careful|critical|important|danger)\b",
        r"\b(pitfall|pearl|tip|key point|do not|never)\b",
        r"\b(risk of|complication|injury to|damage to)\b",
    ]

    TEACHING_PATTERNS = [
        r"\b(lesson|teaching point|remember|note that|importantly)\b",
        r"\b(highlights?|illustrates?|demonstrates?|emphasizes?)\b",
        r"\b(key concept|fundamental|essential|principle)\b",
    ]

    def __init__(self, config: EnhancedChunkerConfig = None, extractor=None):
        self.config = config or EnhancedChunkerConfig()
        self.extractor = extractor
        self.quality_scorer = get_quality_scorer()
        self._pitfall_patterns = [re.compile(p, re.IGNORECASE) for p in self.PITFALL_PATTERNS]
        self._teaching_patterns = [re.compile(p, re.IGNORECASE) for p in self.TEACHING_PATTERNS]
        self._all_safe_cut_rules = get_all_safe_cut_rules()

    # =========================================================================
    # COMPLEXITY ESTIMATION (NEW in v2.2)
    # =========================================================================

    def _estimate_complexity(self, text: str) -> ContentComplexity:
        """Estimate content complexity for adaptive token limits."""
        signals = 0

        # Step markers indicate procedural complexity
        if re.search(r"Step\s+\d|^\d+\.\s+", text, re.M):
            signals += 2

        # Multiple spatial relationships
        spatial_count = len(re.findall(
            r"\b(lateral|medial|superior|inferior|anterior|posterior)\b",
            text, re.I
        ))
        if spatial_count > 5:
            signals += 2
        elif spatial_count > 2:
            signals += 1

        # Grading scales
        if re.search(r"(Grade|WHO|Spetzler|Hunt.Hess)\s+[IVX\d]+", text, re.I):
            signals += 1

        # Molecular markers
        if re.search(r"\b(IDH|MGMT|1p.?19q|EGFR|BRAF)\b", text, re.I):
            signals += 1

        # Multiple measurements
        measurement_count = len(re.findall(r"\d+\s*(mm|cm|degrees?|°)", text, re.I))
        if measurement_count > 3:
            signals += 1

        # Entity density (if extractor available)
        if self.extractor:
            try:
                entities = self.extractor.extract(text[:1000])
                word_count = len(text.split())
                if word_count > 0:
                    density = len(entities) / word_count * 100
                    if density > 15:
                        signals += 1
            except:
                pass

        # Classify
        if signals >= 5:
            return ContentComplexity.HIGHLY_COMPLEX
        elif signals >= 3:
            return ContentComplexity.COMPLEX
        elif signals >= 1:
            return ContentComplexity.MODERATE
        return ContentComplexity.SIMPLE

    def _get_adaptive_limits(self, type_config: ChunkTypeConfig, text: str) -> dict:
        """Get token limits adjusted for content complexity."""
        if not self.config.enable_adaptive_limits:
            return {
                'target': type_config.target_tokens,
                'min': type_config.min_tokens,
                'max': type_config.max_tokens,
                'overlap': type_config.overlap_sentences,
            }

        complexity = self._estimate_complexity(text)
        factor = complexity.value

        return {
            'target': int(type_config.target_tokens * factor),
            'min': int(type_config.min_tokens * factor),
            'max': int(type_config.max_tokens * min(factor, 1.2)),  # Cap max expansion
            'overlap': type_config.overlap_sentences,
        }

    # =========================================================================
    # MAIN CHUNKING
    # =========================================================================

    def chunk_section(
        self,
        section_text: str,
        section_title: str,
        page_num: int,
        doc_id: str,
        tables: List = None
    ) -> List:
        """Process a section into semantic chunks."""
        tables = tables or []

        # Early type detection
        early_type = self._detect_type_from_title(section_title)
        type_config = get_type_config(early_type)

        # Get adaptive limits (NEW in v2.2)
        limits = self._get_adaptive_limits(type_config, section_text)
        target_tokens = limits['target']
        max_tokens = limits['max']
        min_tokens = limits['min']
        overlap_sentences = limits['overlap']

        # Split sentences
        sentences = self._split_sentences(section_text)
        if not sentences:
            return []

        chunks = []
        current_buffer: List[str] = []
        current_word_count = 0
        current_step = 0
        total_steps = self._count_steps_in_section(sentences)

        for i, sentence in enumerate(sentences):
            sent_word_count = len(sentence.split())

            step_num = extract_step_number(sentence)
            if step_num:
                current_step = step_num

            is_dependent = self._check_dependency(sentence)
            prev_sentence = current_buffer[-1] if current_buffer else ""

            if current_word_count + sent_word_count > target_tokens:

                if current_word_count + sent_word_count > max_tokens:
                    if current_buffer:
                        chunk = self._finalize_chunk(
                            current_buffer, section_title, page_num, doc_id,
                            early_type, current_step, total_steps
                        )
                        chunks.append(chunk)
                    current_buffer = [sentence]
                    current_word_count = sent_word_count

                elif is_dependent or not self._is_safe_cut(prev_sentence, sentence, type_config):
                    current_buffer.append(sentence)
                    current_word_count += sent_word_count

                else:
                    if current_buffer:
                        chunk = self._finalize_chunk(
                            current_buffer, section_title, page_num, doc_id,
                            early_type, current_step, total_steps
                        )
                        chunks.append(chunk)

                    overlap = current_buffer[-overlap_sentences:] if current_buffer else []
                    current_buffer = overlap + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_buffer)
            else:
                current_buffer.append(sentence)
                current_word_count += sent_word_count

        # Flush remaining
        if current_buffer:
            if current_word_count >= min_tokens or not chunks:
                chunk = self._finalize_chunk(
                    current_buffer, section_title, page_num, doc_id,
                    early_type, current_step, total_steps
                )
                chunks.append(chunk)
            elif chunks:
                chunks[-1].content += "\n\n" + " ".join(current_buffer)
                if self.config.enable_quality_scoring:
                    self.quality_scorer.score_chunk(chunks[-1])

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = re.sub(r'\s+', ' ', text.strip())
        parts = self.SENTENCE_SPLIT_PATTERN.split(text)
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                if not part.endswith(('.', '!', '?')):
                    part += '.'
                sentences.append(part)
        return sentences

    def _check_dependency(self, sentence: str) -> bool:
        """Check if sentence depends on previous context."""
        triggers = [
            r"^(This|That|These|Those)\s",
            r"^(However|Therefore|Thus|Furthermore|Moreover|Additionally)\b",
            r"^(Step\s+\d|Then|Next|Finally|Subsequently)\b",
            r"^(It\s+is|They\s+are)\b",
        ]
        return any(re.search(t, sentence, re.IGNORECASE) for t in triggers)

    def _is_safe_cut(self, prev_sent: str, next_sent: str, type_config: ChunkTypeConfig) -> bool:
        """Check if safe to cut between sentences."""
        if not prev_sent or not next_sent:
            return True

        for rule in type_config.safe_cut_rules:
            if rule.matches(prev_sent, next_sent):
                return False

        for rule in self._all_safe_cut_rules:
            if rule.matches(prev_sent, next_sent):
                return False

        return True

    def _detect_type_from_title(self, section_title: str) -> ChunkType:
        """Detect chunk type from section title."""
        title_lower = section_title.lower()

        if any(kw in title_lower for kw in ["technique", "approach", "procedure", "surgical", "step"]):
            return ChunkType.PROCEDURE
        if any(kw in title_lower for kw in ["anatomy", "nerve", "artery", "vein"]):
            return ChunkType.ANATOMY
        if any(kw in title_lower for kw in ["pathology", "tumor", "disease", "grade"]):
            return ChunkType.PATHOLOGY
        if any(kw in title_lower for kw in ["clinical", "presentation", "management", "treatment"]):
            return ChunkType.CLINICAL
        if any(kw in title_lower for kw in ["case", "patient"]):
            return ChunkType.CASE
        if any(kw in title_lower for kw in ["differential", "versus"]):
            return ChunkType.DIFFERENTIAL
        if any(kw in title_lower for kw in ["imaging", "mri", "ct"]):
            return ChunkType.IMAGING
        if any(kw in title_lower for kw in ["outcome", "evidence", "study"]):
            return ChunkType.EVIDENCE

        return ChunkType.GENERAL

    def _count_steps_in_section(self, sentences: List[str]) -> int:
        """Count explicit steps in section."""
        count = 0
        for sent in sentences:
            if extract_step_number(sent):
                count += 1
        return count

    def _finalize_chunk(
        self,
        buffer: List[str],
        section_title: str,
        page_num: int,
        doc_id: str,
        early_type: ChunkType,
        current_step: int,
        total_steps: int
    ) -> dict:
        """Create chunk with metadata."""
        content = " ".join(buffer)

        # Extract entities if extractor available
        entities = []
        if self.extractor:
            entities = self.extractor.extract(content)

        # Classify content
        chunk_type = self._classify_content(content, section_title, entities, early_type)

        # Extract metadata
        surgical_phase = None
        if self.config.extract_surgical_phase and chunk_type == ChunkType.PROCEDURE:
            surgical_phase = detect_surgical_phase(content, section_title)

        step_number = None
        step_sequence = None
        if self.config.extract_step_numbers and current_step > 0:
            step_number = current_step
            if total_steps > 0:
                step_sequence = f"{current_step}_of_{total_steps}"

        has_pitfall = self._detect_pitfall(content)
        has_teaching = self._detect_teaching_point(content)

        # Build chunk
        chunk = self._build_chunk_object(
            content=content,
            doc_id=doc_id,
            section_title=section_title,
            page_num=page_num,
            chunk_type=chunk_type,
            entities=entities,
            surgical_phase=surgical_phase,
            step_number=step_number,
            step_sequence=step_sequence,
            has_pitfall=has_pitfall,
            has_teaching=has_teaching
        )

        # Quality scoring
        if self.config.enable_quality_scoring:
            self.quality_scorer.score_chunk(chunk)

        return chunk

    def _classify_content(self, content: str, section_title: str, entities: list, early_type: ChunkType) -> ChunkType:
        """Classify chunk content."""
        content_lower = content.lower()

        # Count entity categories
        category_counts = {}
        for e in entities:
            cat = getattr(e, 'category', 'UNKNOWN')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        procedure_words = ["first", "then", "next", "step", "incision", "dissect", "retract"]
        if any(w in content_lower for w in procedure_words):
            if any("PROCEDURE" in cat for cat in category_counts):
                return ChunkType.PROCEDURE

        if sum(1 for cat in category_counts if "PATHOLOGY" in cat) >= 2:
            return ChunkType.PATHOLOGY

        if sum(1 for cat in category_counts if "ANATOMY" in cat) >= 3:
            return ChunkType.ANATOMY

        clinical_words = ["present", "symptom", "deficit", "outcome", "prognosis"]
        if any(w in content_lower for w in clinical_words):
            return ChunkType.CLINICAL

        if early_type != ChunkType.GENERAL:
            return early_type

        return ChunkType.GENERAL

    def _detect_pitfall(self, content: str) -> bool:
        """Detect if content contains pitfalls/pearls."""
        return any(p.search(content) for p in self._pitfall_patterns)

    def _detect_teaching_point(self, content: str) -> bool:
        """Detect if content contains teaching points."""
        return any(p.search(content) for p in self._teaching_patterns)

    def _build_chunk_object(self, **kwargs):
        """Build chunk object (compatible with SemanticChunk)."""
        from dataclasses import dataclass, field as dc_field

        # Try to import actual SemanticChunk
        try:
            from src.shared.models import SemanticChunk

            chunk = SemanticChunk(
                id=str(uuid4()),
                document_id=kwargs['doc_id'],
                content=kwargs['content'],
                title=self._generate_title(kwargs['content'], kwargs['section_title'], kwargs['entities']),
                section_path=[kwargs['section_title']],
                page_start=kwargs['page_num'],
                page_end=kwargs['page_num'],
                chunk_type=kwargs['chunk_type'],
                entities=kwargs['entities'],
                entity_names=[getattr(e, 'text', str(e)) for e in kwargs['entities']],
                specialty_tags=self._detect_specialties(kwargs['content'], kwargs['entities']),
                keywords=self._extract_keywords(kwargs['content'], kwargs['entities']),
            )

            # Add enhanced metadata if fields exist
            if hasattr(chunk, 'surgical_phase'):
                chunk.surgical_phase = kwargs.get('surgical_phase')
            if hasattr(chunk, 'step_number'):
                chunk.step_number = kwargs.get('step_number')
            if hasattr(chunk, 'step_sequence'):
                chunk.step_sequence = kwargs.get('step_sequence')
            if hasattr(chunk, 'has_pitfall'):
                chunk.has_pitfall = kwargs.get('has_pitfall', False)
            if hasattr(chunk, 'has_teaching_point'):
                chunk.has_teaching_point = kwargs.get('has_teaching', False)

            return chunk

        except ImportError:
            # Return dict if SemanticChunk not available
            return {
                'id': str(uuid4()),
                'document_id': kwargs['doc_id'],
                'content': kwargs['content'],
                'title': kwargs['section_title'],
                'chunk_type': kwargs['chunk_type'].value,
                'page_start': kwargs['page_num'],
                'surgical_phase': kwargs.get('surgical_phase'),
                'step_number': kwargs.get('step_number'),
                'step_sequence': kwargs.get('step_sequence'),
                'has_pitfall': kwargs.get('has_pitfall', False),
                'has_teaching_point': kwargs.get('has_teaching', False),
                'entities': kwargs['entities'],
            }

    def _generate_title(self, content: str, section_title: str, entities: list) -> str:
        """Generate chunk title."""
        if section_title and len(section_title) > 3:
            if entities:
                key_entity = entities[0]
                entity_text = getattr(key_entity, 'normalized', None) or getattr(key_entity, 'text', '')
                if entity_text and entity_text.lower() not in section_title.lower():
                    return f"{section_title}: {entity_text}"
            return section_title

        sentences = content.split('.')
        for sent in sentences:
            sent = sent.strip()
            if 10 < len(sent) < 100:
                return sent[:80]

        return content[:60].split('.')[0] + "..."

    def _detect_specialties(self, content: str, entities: list) -> List[str]:
        """Detect neurosurgical specialties."""
        detected = set()

        for entity in entities:
            cat = getattr(entity, 'category', '')
            specialty = CATEGORY_SPECIALTY_MAP.get(cat, 'general')
            detected.add(specialty)

        if len(detected) > 1 and "general" in detected:
            detected.discard("general")

        return sorted(detected, key=lambda s: SPECIALTY_PRIORITY.index(s) if s in SPECIALTY_PRIORITY else 99)

    def _extract_keywords(self, content: str, entities: list) -> List[str]:
        """Extract searchable keywords."""
        keywords = set()

        for entity in entities:
            text = getattr(entity, 'text', '')
            if text:
                keywords.add(text.lower())
            normalized = getattr(entity, 'normalized', '')
            if normalized and normalized != text:
                keywords.add(normalized.lower())

        important_terms = [
            "complication", "technique", "approach", "outcome",
            "indication", "contraindication", "management"
        ]
        content_lower = content.lower()
        for term in important_terms:
            if term in content_lower:
                keywords.add(term)

        return list(keywords)[:30]
