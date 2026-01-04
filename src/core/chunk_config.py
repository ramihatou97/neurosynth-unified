"""
NeuroSynth v2.2 - Enhanced Chunk Configuration
===============================================

Type-specific chunking configuration optimized for expert-level
neurosurgical synthesis and QA.

Each chunk type has tuned parameters for:
- Token limits (target, min, max)
- Overlap requirements
- Safe-cut rules
- Quality score weights
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import re


# =============================================================================
# CHUNK TYPE ENUM
# =============================================================================

class ChunkType(Enum):
    """Semantic classification of chunk content."""
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    PATHOLOGY = "pathology"
    CLINICAL = "clinical"
    CASE = "case"
    GENERAL = "general"
    DIFFERENTIAL = "differential"
    IMAGING = "imaging"
    EVIDENCE = "evidence"
    COMPARATIVE = "comparative"
    GRADING_SCALE = "grading_scale"
    MEASUREMENT = "measurement"
    INSTRUMENT = "instrument"
    MONITORING = "monitoring"
    COMPLICATION = "complication"
    OUTCOME = "outcome"
    DOSAGE = "dosage"
    TABLE = "table"
    ALGORITHM = "algorithm"
    HISTORICAL = "historical"
    EPIDEMIOLOGY = "epidemiology"
    PEARL_PITFALL = "pearl_pitfall"
    PHARMACOLOGY = "pharmacology"
    REFERENCE = "reference"


class SurgicalPhase(Enum):
    """Phases of a surgical procedure."""
    INDICATION = "indication"
    PREOPERATIVE = "preoperative"
    POSITIONING = "positioning"
    EXPOSURE = "exposure"
    APPROACH = "approach"
    RESECTION = "resection"
    RECONSTRUCTION = "reconstruction"
    CLOSURE = "closure"
    POSTOPERATIVE = "postoperative"
    COMPLICATION = "complication"
    OUTCOME = "outcome"
    OTHER = "other"


# =============================================================================
# SAFE-CUT RULE
# =============================================================================

@dataclass
class SafeCutRule:
    """Defines when NOT to split between two sentences."""
    name: str
    description: str
    prev_patterns: List[str] = field(default_factory=list)
    prev_contains: List[str] = field(default_factory=list)
    next_patterns: List[str] = field(default_factory=list)
    next_contains: List[str] = field(default_factory=list)
    priority: int = 5

    def matches(self, prev_sent: str, next_sent: str) -> bool:
        prev_lower = prev_sent.lower()
        next_lower = next_sent.lower()

        prev_match = False
        if self.prev_patterns:
            prev_match = any(re.search(p, prev_sent, re.IGNORECASE) for p in self.prev_patterns)
        if not prev_match and self.prev_contains:
            prev_match = any(c in prev_lower for c in self.prev_contains)
        if not prev_match and (self.prev_patterns or self.prev_contains):
            return False

        next_match = False
        if self.next_patterns:
            next_match = any(re.search(p, next_sent, re.IGNORECASE) for p in self.next_patterns)
        if not next_match and self.next_contains:
            next_match = any(c in next_lower for c in self.next_contains)
        if not next_match and (self.next_patterns or self.next_contains):
            return False

        return True


# =============================================================================
# TYPE CONFIG
# =============================================================================

@dataclass
class ChunkTypeConfig:
    """Configuration for a specific chunk type."""
    chunk_type: ChunkType
    target_tokens: int
    min_tokens: int
    max_tokens: int
    overlap_sentences: int
    safe_cut_rules: List[SafeCutRule] = field(default_factory=list)
    quality_weights: Dict[str, float] = field(default_factory=dict)
    classification_keywords: List[str] = field(default_factory=list)


# =============================================================================
# SAFE-CUT RULES BY TYPE
# =============================================================================

PROCEDURE_RULES = [
    SafeCutRule("step_sequence", "Keep steps together",
                prev_patterns=[r"Step\s+\d+", r"^\d+\.\s+"],
                next_patterns=[r"^\s*(This|The\s+\w+\s+is)"], priority=10),
    SafeCutRule("instrument_action", "Keep instrument with action",
                prev_contains=["bipolar", "suction", "dissector", "microscope", "retractor"],
                next_contains=["to coagulate", "to dissect", "to expose", "carefully"], priority=9),
    SafeCutRule("pitfall_prevention", "Keep warning with prevention",
                prev_contains=["avoid", "caution", "warning", "critical"],
                next_contains=["can cause", "risk of", "to prevent"], priority=9),
]

ANATOMY_RULES = [
    SafeCutRule("spatial_chain", "Keep spatial relationships",
                prev_contains=["lateral to", "medial to", "superior to", "inferior to"],
                next_contains=["lateral to", "medial to", "and", "which is"], priority=9),
    SafeCutRule("vascular_course", "Keep vessel course",
                prev_contains=["arises from", "originates", "emerges"],
                next_contains=["courses", "travels", "terminates", "supplies"], priority=9),
]

PATHOLOGY_RULES = [
    SafeCutRule("grading_criteria", "Keep grade with criteria",
                prev_patterns=[r"(Grade|WHO|Spetzler)\s+[IVX\d]+"],
                next_contains=["characterized by", "defined as", "criteria"], priority=10),
    SafeCutRule("molecular_prognosis", "Keep markers with prognosis",
                prev_contains=["IDH", "1p/19q", "MGMT", "mutation"],
                next_contains=["prognosis", "survival", "outcome"], priority=9),
]


# =============================================================================
# TYPE CONFIGURATIONS
# =============================================================================

TYPE_CONFIGS: Dict[ChunkType, ChunkTypeConfig] = {
    ChunkType.PROCEDURE: ChunkTypeConfig(
        chunk_type=ChunkType.PROCEDURE, target_tokens=750, min_tokens=200, max_tokens=1100,
        overlap_sentences=2, safe_cut_rules=PROCEDURE_RULES,
        classification_keywords=["step", "incision", "dissect", "retract", "technique"]
    ),
    ChunkType.ANATOMY: ChunkTypeConfig(
        chunk_type=ChunkType.ANATOMY, target_tokens=600, min_tokens=150, max_tokens=900,
        overlap_sentences=3, safe_cut_rules=ANATOMY_RULES,
        classification_keywords=["anatomy", "nerve", "artery", "lateral", "medial"]
    ),
    ChunkType.PATHOLOGY: ChunkTypeConfig(
        chunk_type=ChunkType.PATHOLOGY, target_tokens=700, min_tokens=180, max_tokens=1000,
        overlap_sentences=2, safe_cut_rules=PATHOLOGY_RULES,
        classification_keywords=["tumor", "lesion", "grade", "WHO", "pathology"]
    ),
    ChunkType.CLINICAL: ChunkTypeConfig(
        chunk_type=ChunkType.CLINICAL, target_tokens=650, min_tokens=150, max_tokens=950,
        overlap_sentences=3, classification_keywords=["patient", "symptom", "management"]
    ),
    ChunkType.CASE: ChunkTypeConfig(
        chunk_type=ChunkType.CASE, target_tokens=800, min_tokens=250, max_tokens=1200,
        overlap_sentences=2, classification_keywords=["case", "year-old", "presented"]
    ),
    ChunkType.GENERAL: ChunkTypeConfig(
        chunk_type=ChunkType.GENERAL, target_tokens=600, min_tokens=150, max_tokens=1000,
        overlap_sentences=2
    ),
    ChunkType.DIFFERENTIAL: ChunkTypeConfig(
        chunk_type=ChunkType.DIFFERENTIAL, target_tokens=650, min_tokens=150, max_tokens=900,
        overlap_sentences=2, classification_keywords=["differential", "versus", "distinguish"]
    ),
    ChunkType.IMAGING: ChunkTypeConfig(
        chunk_type=ChunkType.IMAGING, target_tokens=550, min_tokens=120, max_tokens=800,
        overlap_sentences=2, classification_keywords=["MRI", "CT", "T1", "T2", "enhancement"]
    ),
    ChunkType.EVIDENCE: ChunkTypeConfig(
        chunk_type=ChunkType.EVIDENCE, target_tokens=600, min_tokens=150, max_tokens=900,
        overlap_sentences=2, classification_keywords=["study", "trial", "evidence"]
    ),
    ChunkType.COMPARATIVE: ChunkTypeConfig(
        chunk_type=ChunkType.COMPARATIVE, target_tokens=700, min_tokens=180, max_tokens=1000,
        overlap_sentences=2, classification_keywords=["compare", "versus", "advantage"]
    ),
    ChunkType.GRADING_SCALE: ChunkTypeConfig(
        chunk_type=ChunkType.GRADING_SCALE, target_tokens=500, min_tokens=100, max_tokens=800,
        overlap_sentences=2, classification_keywords=["grade", "scale", "Spetzler", "Hunt-Hess"]
    ),
    ChunkType.COMPLICATION: ChunkTypeConfig(
        chunk_type=ChunkType.COMPLICATION, target_tokens=600, min_tokens=150, max_tokens=900,
        overlap_sentences=3, classification_keywords=["complication", "risk", "avoid", "prevent"]
    ),
    ChunkType.PEARL_PITFALL: ChunkTypeConfig(
        chunk_type=ChunkType.PEARL_PITFALL, target_tokens=400, min_tokens=80, max_tokens=600,
        overlap_sentences=2, classification_keywords=["pearl", "pitfall", "tip", "key point"]
    ),
}


def get_type_config(chunk_type: ChunkType) -> ChunkTypeConfig:
    """Get configuration for a chunk type."""
    return TYPE_CONFIGS.get(chunk_type, TYPE_CONFIGS[ChunkType.GENERAL])


def get_all_safe_cut_rules() -> List[SafeCutRule]:
    """Get all safe-cut rules sorted by priority."""
    all_rules = []
    seen = set()
    for config in TYPE_CONFIGS.values():
        for rule in config.safe_cut_rules:
            if rule.name not in seen:
                all_rules.append(rule)
                seen.add(rule.name)
    return sorted(all_rules, key=lambda r: r.priority, reverse=True)


# =============================================================================
# SURGICAL PHASE DETECTION
# =============================================================================

PHASE_PATTERNS = {
    SurgicalPhase.POSITIONING: [r"position", r"prone", r"supine", r"lateral", r"mayfield"],
    SurgicalPhase.EXPOSURE: [r"incision", r"skin", r"flap", r"craniotomy", r"exposure"],
    SurgicalPhase.APPROACH: [r"approach", r"corridor", r"dura", r"opening"],
    SurgicalPhase.RESECTION: [r"resection", r"removal", r"dissection", r"clipping"],
    SurgicalPhase.CLOSURE: [r"closure", r"close", r"dural closure", r"wound"],
    SurgicalPhase.COMPLICATION: [r"complication", r"risk", r"avoid", r"prevent"],
}


def detect_surgical_phase(content: str, section_title: str = "") -> SurgicalPhase:
    """Detect surgical phase from content."""
    combined = f"{section_title} {content}".lower()
    scores = {}
    for phase, patterns in PHASE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, combined))
        if score > 0:
            scores[phase] = score
    return max(scores, key=scores.get) if scores else SurgicalPhase.OTHER


# =============================================================================
# STEP NUMBER EXTRACTION
# =============================================================================

STEP_PATTERNS = [
    (r"^Step\s+(\d+)", "explicit"),
    (r"^(\d+)\.\s+", "numbered"),
    (r"^(First|Second|Third|Fourth|Fifth)", "ordinal"),
]

ORDINAL_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def extract_step_number(content: str) -> Optional[int]:
    """Extract step number from content."""
    for pattern, step_type in STEP_PATTERNS:
        match = re.match(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1)
            if step_type in ("explicit", "numbered"):
                return int(value)
            elif step_type == "ordinal":
                return ORDINAL_MAP.get(value.lower())
    return None
