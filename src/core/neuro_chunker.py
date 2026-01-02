"""
NeuroSynth v2.0 - Medical-Aware Semantic Chunker
=================================================

Chunks text preserving medical semantic completeness.

Key principles:
1. Never split mid-sentence
2. Never split anatomical descriptions
3. Keep cause-effect pairs together
4. Keep procedure steps together
5. Preserve figure references with context
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from uuid import uuid4

from src.shared.models import SemanticChunk, ChunkType, NeuroEntity
from .neuro_extractor import NeuroExpertTextExtractor


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED CATEGORY TO SPECIALTY MAPPING (50 Categories)
# ═══════════════════════════════════════════════════════════════════════════════

EXPANDED_CATEGORY_SPECIALTY_MAP = {
    # ─────────────────────────────────────────────────────────────────────────
    # EXISTING CATEGORIES (14)
    # ─────────────────────────────────────────────────────────────────────────
    "ANATOMY_VASCULAR_ARTERIAL": "vascular",
    "ANATOMY_VASCULAR_VENOUS": "vascular",
    "ANATOMY_SKULL_BASE": "skull_base",
    "ANATOMY_CRANIAL_NERVES": "skull_base",
    "ANATOMY_SPINE_BONE": "spine",
    "ANATOMY_SPINE_NEURAL": "spine",
    "PATHOLOGY_VASCULAR": "vascular",
    "PATHOLOGY_TUMOR": "oncology",
    "PROCEDURE_APPROACH": "general",
    "PROCEDURE_ACTION": "general",
    "INSTRUMENT": "general",
    "MEASUREMENT": "general",
    "IMAGING": "general",
    "PATHOLOGY_FUNCTIONAL": "functional",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - DEVICE/TECHNOLOGY & MEASUREMENTS (3)
    # ─────────────────────────────────────────────────────────────────────────
    "DEVICE_TECHNOLOGY": "general",
    "MEASUREMENT_PARAMETER": "general",
    "SURGICAL_SPECIFICATION": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - PERIPHERAL NERVE (3)
    # ─────────────────────────────────────────────────────────────────────────
    "ANATOMY_PERIPHERAL_UPPER": "peripheral",
    "ANATOMY_PERIPHERAL_LOWER": "peripheral",
    "PATHOLOGY_PERIPHERAL": "peripheral",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - RADIOSURGERY & RADIATION (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PROCEDURE_RADIOSURGERY": "oncology",
    "PROCEDURE_RADIATION": "oncology",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - ENDOVASCULAR (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PROCEDURE_ENDOVASCULAR": "vascular",
    "SCORE_ENDOVASCULAR": "vascular",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - PAIN (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_PAIN": "functional",
    "PROCEDURE_PAIN": "functional",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - TUMOR MOLECULAR & CLASSIFICATION (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_TUMOR_MOLECULAR": "oncology",
    "PATHOLOGY_TUMOR_CLASSIFICATION": "oncology",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - SPINE DEFORMITY & BIOMECHANICS (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_SPINE_DEFORMITY": "spine",
    "SPINE_BIOMECHANICS": "spine",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - VASCULAR MALFORMATIONS (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_AVM": "vascular",
    "PATHOLOGY_VASCULAR_OTHER": "vascular",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - CRITICAL CARE & PHYSIOLOGY (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PHYSIOLOGY_CRITICAL_CARE": "trauma",
    "PHYSIOLOGY_CSF": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - PEDIATRIC (2)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_PEDIATRIC": "pediatric",
    "ANATOMY_PEDIATRIC": "pediatric",

    # ─────────────────────────────────────────────────────────────────────────
    # NEW CATEGORIES - SKULL BASE EXPANDED (3)
    # ─────────────────────────────────────────────────────────────────────────
    "ANATOMY_SKULL_BASE_ANTERIOR": "skull_base",
    "ANATOMY_SKULL_BASE_MIDDLE": "skull_base",
    "ANATOMY_SKULL_BASE_POSTERIOR": "skull_base",

    # ─────────────────────────────────────────────────────────────────────────
    # BRAIN ANATOMY (5)
    # ─────────────────────────────────────────────────────────────────────────
    "ANATOMY_BRAIN_CORTICAL": "general",
    "ANATOMY_BRAIN_SUBCORTICAL": "functional",
    "ANATOMY_BRAIN_WHITE_MATTER": "general",
    "ANATOMY_BRAIN_BRAINSTEM": "general",
    "ANATOMY_BRAIN_CEREBELLUM": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # MENINGES, VENTRICLES, CISTERNS (3)
    # ─────────────────────────────────────────────────────────────────────────
    "ANATOMY_MENINGES": "general",
    "ANATOMY_VENTRICULAR": "general",
    "ANATOMY_CISTERNAL": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # PATHOLOGY - TRAUMA, DEGENERATIVE, INFECTIOUS, CONGENITAL, HYDROCEPHALUS (5)
    # ─────────────────────────────────────────────────────────────────────────
    "PATHOLOGY_TRAUMA": "trauma",
    "PATHOLOGY_DEGENERATIVE": "spine",
    "PATHOLOGY_INFECTIOUS": "general",
    "PATHOLOGY_CONGENITAL": "pediatric",
    "PATHOLOGY_HYDROCEPHALUS": "pediatric",

    # ─────────────────────────────────────────────────────────────────────────
    # FUNCTIONAL & MONITORING (2)
    # ─────────────────────────────────────────────────────────────────────────
    "FUNCTIONAL_NEURO": "functional",
    "PHYSIOLOGY_MONITORING": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # MEDICATIONS & OUTCOMES (2)
    # ─────────────────────────────────────────────────────────────────────────
    "MEDICATION": "general",
    "OUTCOME_COMPLICATION": "general",

    # ─────────────────────────────────────────────────────────────────────────
    # PROCEDURE MODALITY (for backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────
    "PROCEDURE_MODALITY": "functional",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALTY DISPLAY NAMES
# ═══════════════════════════════════════════════════════════════════════════════

SPECIALTY_DISPLAY_NAMES = {
    "vascular": "Cerebrovascular & Endovascular",
    "skull_base": "Skull Base Surgery",
    "spine": "Spine Surgery",
    "oncology": "Neuro-Oncology & Radiosurgery",
    "functional": "Functional & Pain Neurosurgery",
    "peripheral": "Peripheral Nerve Surgery",
    "pediatric": "Pediatric Neurosurgery",
    "trauma": "Neurotrauma & Critical Care",
    "general": "General Neurosurgery",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALTY PRIORITY FOR CHUNK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
# When multiple specialties are present, use this priority order
# Higher index = higher priority (more specific takes precedence)

SPECIALTY_PRIORITY = [
    "general",       # 0 - lowest priority (default)
    "oncology",      # 1
    "vascular",      # 2
    "spine",         # 3
    "skull_base",    # 4
    "functional",    # 5
    "peripheral",    # 6
    "pediatric",     # 7
    "trauma"         # 8 - highest priority (most urgent)
]

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE GENERATION PRIORITY CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════
# Categories to prioritize when generating chunk titles
# Order matters - earlier categories preferred for title generation

TITLE_PRIORITY_CATEGORIES = [
    # Specific pathology first
    "PATHOLOGY_TUMOR_CLASSIFICATION",
    "PATHOLOGY_TUMOR_MOLECULAR",
    "PATHOLOGY_AVM",
    "PATHOLOGY_VASCULAR",
    "PATHOLOGY_VASCULAR_OTHER",
    "PATHOLOGY_TRAUMA",
    "PATHOLOGY_PAIN",
    "PATHOLOGY_SPINE_DEFORMITY",
    "PATHOLOGY_PEDIATRIC",
    "PATHOLOGY_HYDROCEPHALUS",
    "PATHOLOGY_INFECTIOUS",
    "PATHOLOGY_DEGENERATIVE",
    "PATHOLOGY_PERIPHERAL",
    "PATHOLOGY_CONGENITAL",
    "PATHOLOGY_FUNCTIONAL",
    "PATHOLOGY_TUMOR",
    # Then procedures
    "PROCEDURE_ENDOVASCULAR",
    "PROCEDURE_RADIOSURGERY",
    "PROCEDURE_PAIN",
    "PROCEDURE_RADIATION",
    "PROCEDURE_APPROACH",
    "PROCEDURE_ACTION",
    # Then anatomy
    "ANATOMY_SKULL_BASE_MIDDLE",
    "ANATOMY_SKULL_BASE_POSTERIOR",
    "ANATOMY_SKULL_BASE_ANTERIOR",
    "ANATOMY_SKULL_BASE",
    "ANATOMY_BRAIN_CORTICAL",
    "ANATOMY_BRAIN_SUBCORTICAL",
    "ANATOMY_PERIPHERAL_UPPER",
    "ANATOMY_PERIPHERAL_LOWER",
    "ANATOMY_VASCULAR_ARTERIAL",
    "ANATOMY_VENTRICULAR",
    # Then scores/measurements
    "SCORE_ENDOVASCULAR",
    "MEASUREMENT",
    "MEASUREMENT_PARAMETER",
    "SURGICAL_SPECIFICATION",
    # Then general/devices
    "DEVICE_TECHNOLOGY",
    "INSTRUMENT",
    "IMAGING",
    "MEDICATION"
]


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALTY HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_specialty_for_category(category: str) -> str:
    """
    Get the specialty tag for a given entity category.

    Args:
        category: Entity category string (e.g., "PATHOLOGY_TRAUMA")

    Returns:
        Specialty string (e.g., "trauma")
    """
    return EXPANDED_CATEGORY_SPECIALTY_MAP.get(category, "general")


def get_dominant_specialty(category_counts: Dict[str, int]) -> str:
    """
    Determine the dominant specialty from a collection of entity categories.

    Uses weighted priority - more specific specialties take precedence,
    but frequency still matters.

    Args:
        category_counts: Dict of {category: count}

    Returns:
        The dominant specialty string
    """
    specialty_scores: Dict[str, float] = {}

    for category, count in category_counts.items():
        specialty = get_specialty_for_category(category)
        priority = SPECIALTY_PRIORITY.index(specialty) if specialty in SPECIALTY_PRIORITY else 0

        # Score = count * (1 + priority_bonus)
        # This weights more specific specialties higher
        score = count * (1 + priority * 0.2)

        if specialty not in specialty_scores:
            specialty_scores[specialty] = 0
        specialty_scores[specialty] += score

    if not specialty_scores:
        return "general"

    return max(specialty_scores, key=specialty_scores.get)


def get_all_specialties(entities: List) -> List[str]:
    """
    Get list of all unique specialties represented in entities.

    Args:
        entities: List of Entity objects with .category attribute

    Returns:
        Sorted list of unique specialty strings (most specific first)
    """
    specialties = set()
    for entity in entities:
        specialty = get_specialty_for_category(entity.category)
        specialties.add(specialty)

    # Sort by priority (most specific first)
    return sorted(
        specialties,
        key=lambda s: SPECIALTY_PRIORITY.index(s) if s in SPECIALTY_PRIORITY else -1,
        reverse=True
    )


def get_title_entity(entities: List) -> Optional[object]:
    """
    Select the best entity to use for chunk title generation.

    Args:
        entities: List of Entity objects

    Returns:
        The highest-priority Entity for title, or None
    """
    if not entities:
        return None

    # Build priority lookup
    category_priority = {cat: idx for idx, cat in enumerate(TITLE_PRIORITY_CATEGORIES)}

    # Sort entities by priority (lower index = higher priority)
    sorted_entities = sorted(
        entities,
        key=lambda e: category_priority.get(e.category, 999)
    )

    return sorted_entities[0] if sorted_entities else None


@dataclass
class ChunkerConfig:
    """Configuration for the semantic chunker (synthesis-optimized)."""
    target_tokens: int = 600      # Soft limit - increased for richer context
    max_tokens: int = 1000        # Hard limit - increased for complete thoughts
    min_tokens: int = 150         # Minimum chunk size - avoid fragments
    overlap_sentences: int = 2    # Sentences to overlap - better continuity

    # Type-specific limits (used when chunk type is detected early)
    procedure_target_tokens: int = 700    # Procedures need more context
    anatomy_target_tokens: int = 550      # Anatomy can be more concise
    pathology_target_tokens: int = 650    # Pathology needs clinical context
    clinical_target_tokens: int = 600     # Clinical matches default


class NeuroSemanticChunker:
    """
    Chunks text based on semantic completeness and neurosurgical logical flow.
    
    Key features:
    1. Safe boundary detection (prevents "lobotomizing" knowledge)
    2. Dependency checking (keeps related sentences together)
    3. Entity-aware chunking (preserves complete medical thoughts)
    4. Figure reference preservation
    """
    
    def __init__(self, config: ChunkerConfig = None):
        self.config = config or ChunkerConfig()
        self.extractor = NeuroExpertTextExtractor()
        
        # Regex for splitting sentences while protecting medical abbreviations
        # Protects: Dr., Fig., No., et al., approx., vs., etc.
        self.sentence_split_pattern = re.compile(
            r'(?<!\bDr)(?<!\bFig)(?<!\bNo)(?<!\bal)(?<!\bapprox)(?<!\bvs)(?<!\bet)(?<!\bi\.e)(?<!\be\.g)\.\s+(?=[A-Z])'
        )

    def _get_target_for_section(self, title: str, text: str) -> int:
        """
        Determine the optimal target token count based on content type.

        Uses a two-pass detection strategy:
        1. Title keywords (high confidence) - check section title first
        2. Content signals (fallback) - analyze text for procedural/anatomical density

        Returns:
            Target token count (procedure=700, anatomy=550, pathology=650, default=600)
        """
        title_lower = title.lower() if title else ""
        text_lower = text.lower() if text else ""

        # ========== PASS 1: Title-based detection (high confidence) ==========

        # Procedure indicators in title
        procedure_title_signals = [
            "technique", "procedure", "surgical", "approach", "operative",
            "step", "method", "how to", "positioning", "exposure"
        ]
        if any(signal in title_lower for signal in procedure_title_signals):
            return self.config.procedure_target_tokens  # 700

        # Anatomy indicators in title
        anatomy_title_signals = [
            "anatomy", "anatomic", "structure", "relationship", "topograph",
            "morpholog", "course of", "origin", "branches of"
        ]
        if any(signal in title_lower for signal in anatomy_title_signals):
            return self.config.anatomy_target_tokens  # 550

        # Pathology indicators in title
        pathology_title_signals = [
            "pathology", "pathological", "disease", "lesion", "tumor",
            "malformation", "syndrome", "deficit", "disorder"
        ]
        if any(signal in title_lower for signal in pathology_title_signals):
            return self.config.pathology_target_tokens  # 650

        # Clinical indicators in title
        clinical_title_signals = [
            "clinical", "presentation", "symptom", "diagnosis", "management",
            "treatment", "outcome", "prognosis", "complication"
        ]
        if any(signal in title_lower for signal in clinical_title_signals):
            return self.config.clinical_target_tokens  # 600

        # ========== PASS 2: Content-based detection (fallback) ==========
        # Count signal density to determine content type

        # Procedure signals in content
        procedure_signals = [
            "incision", "dissect", "retract", "expose", "identify",
            "mobilize", "ligate", "resect", "clip", "coagulate",
            "elevate", "drill", "remove", "position the patient",
            "make a", "next step", "then"
        ]
        procedure_density = sum(1 for s in procedure_signals if s in text_lower)

        # Anatomy signals in content
        anatomy_signals = [
            "arises from", "courses", "enters", "exits", "passes",
            "runs along", "lies", "situated", "medial to", "lateral to",
            "anterior to", "posterior to", "superficial to", "deep to",
            "originates", "terminates", "divides into", "gives off"
        ]
        anatomy_density = sum(1 for s in anatomy_signals if s in text_lower)

        # Pathology signals in content
        pathology_signals = [
            "causes", "results in", "leads to", "presents with",
            "manifests as", "characterized by", "associated with",
            "compresses", "invades", "infiltrates", "displaces"
        ]
        pathology_density = sum(1 for s in pathology_signals if s in text_lower)

        # Decision based on highest density (threshold of 2)
        max_density = max(procedure_density, anatomy_density, pathology_density)

        if max_density >= 2:
            if procedure_density == max_density:
                return self.config.procedure_target_tokens  # 700
            elif anatomy_density == max_density:
                return self.config.anatomy_target_tokens  # 550
            elif pathology_density == max_density:
                return self.config.pathology_target_tokens  # 650

        # Default target
        return self.config.target_tokens  # 600

    def chunk_section(
        self,
        section_text: str,
        section_title: str,
        page_num: int,
        doc_id: str
    ) -> List[SemanticChunk]:
        """
        Process a section of text into semantic chunks.

        Args:
            section_text: The text content to chunk
            section_title: Title of the section (for context)
            page_num: Page number for reference
            doc_id: Document ID for linking

        Returns:
            List of SemanticChunk objects
        """
        # Clean and split into sentences
        sentences = self._split_sentences(section_text)

        if not sentences:
            return []

        # Get type-specific target tokens based on content analysis
        target_tokens = self._get_target_for_section(section_title, section_text)
        logger.debug(f"Chunking section '{section_title[:50]}...' with target={target_tokens} tokens")

        chunks = []
        current_buffer: List[str] = []
        current_word_count = 0

        for i, sentence in enumerate(sentences):
            sent_word_count = len(sentence.split())

            # Check if this sentence depends on the previous one
            is_dependent = self._check_dependency(sentence)

            # Decision: Should we cut here?
            if current_word_count + sent_word_count > target_tokens:
                
                # CASE A: Hard limit reached - must cut
                if current_word_count + sent_word_count > self.config.max_tokens:
                    if current_buffer:
                        chunk = self._finalize_chunk(
                            current_buffer, section_title, page_num, doc_id
                        )
                        chunks.append(chunk)
                    
                    current_buffer = [sentence]
                    current_word_count = sent_word_count
                
                # CASE B: Soft limit - check if safe to cut
                elif is_dependent or not self._is_safe_cut(
                    current_buffer[-1] if current_buffer else "",
                    sentence
                ):
                    # Unsafe to cut - extend chunk
                    current_buffer.append(sentence)
                    current_word_count += sent_word_count
                
                # CASE C: Safe to cut
                else:
                    if current_buffer:
                        chunk = self._finalize_chunk(
                            current_buffer, section_title, page_num, doc_id
                        )
                        chunks.append(chunk)
                    
                    # Start new buffer with overlap
                    overlap = current_buffer[-self.config.overlap_sentences:] if current_buffer else []
                    current_buffer = overlap + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_buffer)
            
            else:
                # Under limit - add to buffer
                current_buffer.append(sentence)
                current_word_count += sent_word_count
        
        # Flush remaining buffer
        if current_buffer:
            # Check minimum size
            if current_word_count >= self.config.min_tokens or not chunks:
                chunk = self._finalize_chunk(
                    current_buffer, section_title, page_num, doc_id
                )
                chunks.append(chunk)
            elif chunks:
                # Merge with previous chunk if too small
                prev_chunk = chunks[-1]
                prev_chunk.content += "\n\n" + " ".join(current_buffer)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, protecting medical abbreviations.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        parts = self.sentence_split_pattern.split(text)
        
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                # Ensure sentence ends with period
                if not part.endswith(('.', '!', '?')):
                    part += '.'
                sentences.append(part)
        
        return sentences
    
    def _check_dependency(self, sentence: str) -> bool:
        """
        Returns True if sentence strongly depends on previous context.
        
        These patterns indicate the sentence cannot stand alone.
        """
        dependency_triggers = [
            # Demonstrative pronouns
            r"^(This|That|These|Those)\s",
            # Connectors
            r"^(However|Therefore|Thus|Furthermore|Moreover|Additionally|Consequently)\b",
            r"^(In\s+contrast|Conversely|Similarly|Likewise)\b",
            r"^(As\s+a\s+result|For\s+this\s+reason)\b",
            # Sequence markers
            r"^(Step\s+\d|Then|Next|Finally|Subsequently|Following\s+this)\b",
            # Continuation
            r"^(It\s+is|They\s+are|He\s+is|She\s+is)\b",
            # Causation (mid-sentence)
            r"resulting\s+in",
            r"due\s+to\s+this",
            r"leads\s+to",
            r"which\s+causes",
        ]
        
        return any(re.search(t, sentence, re.IGNORECASE) for t in dependency_triggers)
    
    def _is_safe_cut(self, prev_sent: str, next_sent: str) -> bool:
        """
        Heuristics to prevent breaking critical knowledge connections.
        
        Returns False if cutting between these sentences would break
        important medical context.
        """
        if not prev_sent or not next_sent:
            return True
        
        prev_lower = prev_sent.lower()
        next_lower = next_sent.lower()
        
        # Rule 1: Anatomy -> Function/Supply
        # "The artery arises here." + "It supplies X." = Don't split
        supply_words = ["supplies", "innervates", "drains", "perfuses", "feeds"]
        if any(w in next_lower for w in supply_words):
            if any(w in prev_lower for w in ["artery", "nerve", "vein", "vessel"]):
                return False
        
        # Rule 2: Action -> Consequence
        # "The clip was applied." + "This resulted in..." = Don't split
        consequence_words = ["resulted in", "caused", "leading to", "producing"]
        if any(w in next_lower for w in consequence_words):
            return False
        
        # Rule 3: Figure reference -> Description
        # "Figure 1 shows the exposure." + "Note the optic nerve." = Don't split
        if re.search(r"figure|fig\.", prev_lower):
            if re.search(r"(show|demonstrate|illustrate)", prev_lower):
                if any(w in next_lower for w in ["note", "observe", "see", "arrow"]):
                    return False
        
        # Rule 4: Warning/Caution continuation
        # "Avoid excessive retraction." + "This can cause venous infarction." = Don't split
        if any(w in prev_lower for w in ["avoid", "caution", "warning", "careful"]):
            if any(w in next_lower for w in ["can cause", "may result", "risk of"]):
                return False
        
        # Rule 5: Measurement context
        # "The incision is 3 cm." + "This allows adequate exposure." = Don't split
        if re.search(r"\d+\s*(mm|cm|degrees?)", prev_lower):
            if any(w in next_lower for w in ["this allows", "this provides", "this ensures"]):
                return False
        
        # Rule 6: List continuation
        # "The branches include: 1) xxx." + "2) yyy." = Don't split
        if re.search(r"[:\(]\s*\d\)", prev_sent):
            if re.search(r"^\d\)", next_sent.strip()):
                return False

        # ===== NEW SYNTHESIS-OPTIMIZED RULES =====

        # Rule 7: Surgical step sequence
        # "The dura is opened." + "The tumor is then visualized." = Don't split
        step_markers = ["then", "next", "subsequently", "following this", "after this"]
        if any(f" {m} " in f" {next_lower} " or next_lower.startswith(m) for m in step_markers):
            surgical_actions = ["is opened", "is exposed", "is identified", "is dissected",
                              "is retracted", "is incised", "is elevated", "is removed"]
            if any(a in prev_lower for a in surgical_actions):
                return False

        # Rule 8: Anatomy-pathology relationship
        # "The tumor involves the M1 segment." + "This results in motor deficit." = Don't split
        pathology_verbs = ["involves", "compresses", "displaces", "encases", "infiltrates",
                          "invades", "erodes", "obstructs", "occludes"]
        if any(v in prev_lower for v in pathology_verbs):
            effect_markers = ["results in", "causing", "leading to", "produces", "manifests as"]
            if any(e in next_lower for e in effect_markers):
                return False

        # Rule 9: Instrument-technique pairing
        # "The bipolar is used to coagulate." + "Care is taken to avoid the nerve." = Don't split
        instruments = ["bipolar", "suction", "dissector", "microscope", "drill",
                      "clip", "cautery", "retractor", "speculum", "forceps"]
        if any(i in prev_lower for i in instruments):
            technique_markers = ["care is taken", "to avoid", "to prevent", "ensuring",
                               "while preserving", "without injuring"]
            if any(t in next_lower for t in technique_markers):
                return False

        # Rule 10: Grading/classification context
        # "The lesion was Spetzler-Martin grade III." + "This indicates high risk." = Don't split
        grading_terms = ["grade", "score", "spetzler", "hunt-hess", "fisher", "who grade",
                        "house-brackmann", "karnofsky", "modified rankin"]
        if any(g in prev_lower for g in grading_terms):
            interpretation_markers = ["indicates", "suggests", "means", "implies",
                                     "associated with", "correlates with", "predicts"]
            if any(i in next_lower for i in interpretation_markers):
                return False

        # Rule 11: Complication-prevention pairs
        # "Injury to the nerve causes weakness." + "This is avoided by careful dissection." = Don't split
        complication_words = ["injury", "damage", "complication", "deficit", "hemorrhage",
                             "infarction", "ischemia", "paralysis", "blindness"]
        if any(c in prev_lower for c in complication_words):
            prevention_markers = ["avoided by", "prevented by", "minimized by", "reduced by",
                                "to prevent", "to avoid", "by careful", "through meticulous"]
            if any(p in next_lower for p in prevention_markers):
                return False

        return True
    
    def _finalize_chunk(
        self,
        buffer: List[str],
        section_title: str,
        page_num: int,
        doc_id: str
    ) -> SemanticChunk:
        """
        Create a SemanticChunk with full metadata extraction.
        """
        content = " ".join(buffer)
        
        # Extract medical entities
        entities = self.extractor.extract(content)
        
        # Extract figure references
        figure_refs = self.extractor.extract_figure_references(content)
        
        # Extract table references
        table_refs = self.extractor.extract_table_references(content)
        
        # Determine chunk type
        chunk_type = self._classify_chunk(content, section_title, entities)
        
        # Extract keywords
        keywords = self._extract_keywords(content, entities)
        
        # Detect specialty tags
        specialty_tags = self._detect_specialties(content, entities)
        
        # Generate title
        title = self._generate_title(content, section_title, entities)
        
        return SemanticChunk(
            id=str(uuid4()),
            document_id=doc_id,
            content=content,
            title=title,
            section_path=[section_title],
            page_start=page_num,
            page_end=page_num,
            chunk_type=chunk_type,
            specialty_tags=specialty_tags,
            entities=entities,
            entity_names=[e.text for e in entities],
            figure_refs=figure_refs,
            table_refs=table_refs,
            keywords=keywords
        )
    
    def _classify_chunk(
        self,
        content: str,
        section_title: str,
        entities: List[NeuroEntity]
    ) -> ChunkType:
        """
        Classify chunk by content type.
        """
        content_lower = content.lower()
        section_lower = section_title.lower()
        
        # Check section title first
        if any(w in section_lower for w in ["technique", "approach", "procedure", "surgical"]):
            return ChunkType.PROCEDURE
        
        # Check for procedure indicators
        procedure_words = [
            "first", "then", "next", "step", "incision", "approach",
            "dissect", "retract", "expose", "clip", "resect"
        ]
        if any(w in content_lower for w in procedure_words):
            procedure_entities = [e for e in entities if "PROCEDURE" in e.category]
            if procedure_entities or "step" in content_lower:
                return ChunkType.PROCEDURE
        
        # Check for pathology
        pathology_entities = [e for e in entities if "PATHOLOGY" in e.category]
        if len(pathology_entities) >= 2:
            return ChunkType.PATHOLOGY
        
        # Check for anatomy
        anatomy_entities = [e for e in entities if "ANATOMY" in e.category]
        if len(anatomy_entities) >= 3:
            return ChunkType.ANATOMY
        
        # Check for clinical correlation
        clinical_words = ["present", "symptom", "deficit", "outcome", "prognosis", "complication"]
        if any(w in content_lower for w in clinical_words):
            return ChunkType.CLINICAL
        
        # Check for case
        case_words = ["case", "patient", "year-old", "presented with", "admitted"]
        if any(w in content_lower for w in case_words):
            return ChunkType.CASE
        
        return ChunkType.GENERAL
    
    def _generate_title(
        self,
        content: str,
        section_title: str,
        entities: List[NeuroEntity]
    ) -> str:
        """
        Generate human-readable title for the chunk.

        Uses expanded TITLE_PRIORITY_CATEGORIES for selecting the most
        relevant entity for title generation.
        """
        # If section title is informative, use it with key entity
        if section_title and len(section_title) > 3:
            # Use the new get_title_entity helper for priority-based selection
            key_entity = get_title_entity(entities)

            if key_entity:
                entity_text = key_entity.normalized or key_entity.text
                if entity_text.lower() not in section_title.lower():
                    return f"{section_title}: {entity_text}"

            return section_title

        # Extract first meaningful sentence as title
        sentences = content.split('.')
        for sent in sentences:
            sent = sent.strip()
            if 10 < len(sent) < 100 and not sent.startswith('('):
                # Clean up
                title = re.sub(r'^\d+[\.\)]\s*', '', sent)
                title = re.sub(r'[:\-–]\s*$', '', title)
                return title[:80]

        # Use priority-based entity selection
        key_entity = get_title_entity(entities)
        if key_entity:
            specialty = get_specialty_for_category(key_entity.category)
            display_name = SPECIALTY_DISPLAY_NAMES.get(specialty, specialty.title())
            return f"{display_name}: {key_entity.text}"

        # Fallback
        return content[:60].split('.')[0] + "..."
    
    def _extract_keywords(
        self,
        content: str,
        entities: List[NeuroEntity]
    ) -> List[str]:
        """
        Extract searchable keywords from content.
        """
        keywords = set()
        
        # Add entity text (normalized and original)
        for entity in entities:
            keywords.add(entity.text.lower())
            if entity.normalized != entity.text:
                keywords.add(entity.normalized.lower())
        
        # Add important general terms
        important_terms = [
            "complication", "technique", "approach", "outcome",
            "indication", "contraindication", "management",
            "diagnosis", "treatment", "prognosis", "risk",
            "bleeding", "infection", "recurrence"
        ]
        
        content_lower = content.lower()
        for term in important_terms:
            if term in content_lower:
                keywords.add(term)
        
        return list(keywords)[:30]
    
    def _detect_specialties(
        self,
        content: str,
        entities: List[NeuroEntity]
    ) -> List[str]:
        """
        Detect neurosurgical subspecialties relevant to this chunk.

        Uses the expanded 50-category specialty mapping for entity-based detection,
        combined with keyword-based detection for comprehensive coverage.
        """
        detected = set()
        content_lower = content.lower()

        # Use expanded category-to-specialty mapping (50 categories)
        for entity in entities:
            specialty = get_specialty_for_category(entity.category)
            detected.add(specialty)

        # Enhanced keyword-based detection (expanded for all 9 specialties)
        specialty_keywords = {
            "vascular": [
                "aneurysm", "avm", "bypass", "carotid", "stroke", "hemorrhage",
                "thrombectomy", "coiling", "embolization", "sah", "ich", "vasospasm",
                "dural fistula", "moyamoya", "flow diversion", "tici", "aspects"
            ],
            "oncology": [
                "glioma", "meningioma", "resection", "tumor", "oncology", "gbm",
                "gamma knife", "radiosurgery", "cyberknife", "srs", "idh", "mgmt",
                "schwannoma", "metastasis", "who grade", "chordoma"
            ],
            "spine": [
                "cervical", "lumbar", "fusion", "disc", "laminectomy", "spondyl",
                "pedicle screw", "acdf", "tlif", "plif", "stenosis", "kyphosis",
                "scoliosis", "sva", "pelvic incidence", "myelopathy"
            ],
            "functional": [
                "dbs", "epilepsy", "parkinson", "stimulation", "tremor", "dystonia",
                "stn", "gpi", "vim", "mvd", "scs", "trigeminal neuralgia",
                "pain", "crps", "rhizotomy", "drez"
            ],
            "skull_base": [
                "skull base", "pituitary", "acoustic", "petroclival", "transsphenoidal",
                "cavernous sinus", "clivus", "cerebellopontine", "cpa", "meckel",
                "foramen", "cribriform", "planum"
            ],
            "peripheral": [
                "brachial plexus", "peripheral nerve", "carpal tunnel", "cubital",
                "neuroma", "ulnar", "median nerve", "radial nerve", "sciatic",
                "femoral nerve", "peroneal"
            ],
            "pediatric": [
                "pediatric", "child", "congenital", "shunt", "craniosynostosis",
                "chiari", "myelomeningocele", "tethered cord", "fontanelle",
                "hydrocephalus", "nph", "mmc"
            ],
            "trauma": [
                "trauma", "tbi", "concussion", "contusion", "edh", "sdh",
                "epidural hematoma", "subdural", "skull fracture", "dai",
                "herniation", "icp", "cpp"
            ],
        }

        for specialty, keywords in specialty_keywords.items():
            if any(kw in content_lower for kw in keywords):
                detected.add(specialty)

        # Remove 'general' if more specific specialties detected
        if len(detected) > 1 and "general" in detected:
            detected.discard("general")

        # Sort by priority (most specific first)
        if detected:
            return sorted(
                detected,
                key=lambda s: SPECIALTY_PRIORITY.index(s) if s in SPECIALTY_PRIORITY else -1,
                reverse=True
            )
        return ["general"]


class TableAwareChunker(NeuroSemanticChunker):
    """
    Extended chunker that handles tables specially.
    
    Tables are kept as single chunks and never split.
    """
    
    TABLE_MARKER_START = "\n[TABLE_START]\n"
    TABLE_MARKER_END = "\n[TABLE_END]\n"
    
    def chunk_section(
        self,
        section_text: str,
        section_title: str,
        page_num: int,
        doc_id: str,
        tables: List = None
    ) -> List[SemanticChunk]:
        """
        Chunk section with table awareness.
        
        Tables are extracted as standalone chunks.
        """
        tables = tables or []
        
        # If tables present, preprocess text
        if tables:
            section_text = self._preprocess_with_tables(section_text, tables)
        
        # Check for table markers
        if self.TABLE_MARKER_START in section_text:
            return self._chunk_with_tables(section_text, section_title, page_num, doc_id)
        
        # Standard chunking
        return super().chunk_section(section_text, section_title, page_num, doc_id)
    
    def _preprocess_with_tables(self, text: str, tables: List) -> str:
        """
        Replace table regions with marked markdown versions.
        """
        for table in tables:
            if hasattr(table, 'markdown_content') and table.markdown_content:
                # Create marked table block
                marked = (
                    f"{self.TABLE_MARKER_START}"
                    f"**{table.title or 'Table'}**\n\n"
                    f"{table.markdown_content}"
                    f"{self.TABLE_MARKER_END}"
                )
                
                # Try to find and replace the raw text
                # This is a simplification - production would use bbox matching
                if hasattr(table, 'raw_text') and len(table.raw_text) > 20:
                    # Find approximate location and replace
                    text = text.replace(table.raw_text[:50], marked[:50])
        
        return text
    
    def _chunk_with_tables(
        self,
        text: str,
        section_title: str,
        page_num: int,
        doc_id: str
    ) -> List[SemanticChunk]:
        """
        Handle text with embedded table markers.
        """
        chunks = []
        
        # Split on table markers
        parts = re.split(
            f"({re.escape(self.TABLE_MARKER_START)}.*?{re.escape(self.TABLE_MARKER_END)})",
            text,
            flags=re.DOTALL
        )
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if part.startswith(self.TABLE_MARKER_START.strip()):
                # This is a table - create single chunk
                table_content = part.replace(self.TABLE_MARKER_START.strip(), "").replace(self.TABLE_MARKER_END.strip(), "").strip()
                
                chunk = SemanticChunk(
                    id=str(uuid4()),
                    document_id=doc_id,
                    content=table_content,
                    title=f"{section_title}: Table",
                    section_path=[section_title],
                    page_start=page_num,
                    page_end=page_num,
                    chunk_type=ChunkType.GENERAL,
                    specialty_tags=["general"],
                    keywords=["table", "data"]
                )
                chunks.append(chunk)
            else:
                # Regular text - standard chunking
                text_chunks = super().chunk_section(part, section_title, page_num, doc_id)
                chunks.extend(text_chunks)
        
        return chunks
