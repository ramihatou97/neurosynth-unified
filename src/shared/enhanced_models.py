"""
NeuroSynth v2.2 - Enhanced Semantic Chunk Model
================================================

Extended SemanticChunk model with additional metadata fields
for optimal synthesis and QA quality.

New fields:
- surgical_phase: Phase of surgical procedure
- step_number: Explicit step number if present
- step_sequence: Position in step sequence (e.g., "3_of_8")
- has_pitfall: Contains surgical pitfall/pearl
- has_teaching_point: Contains teaching content
- grading_scale: For pathology chunks with grading
- molecular_markers: For pathology chunks with molecular data

Usage:
    This file provides the extended dataclass definition.
    Add these fields to your existing SemanticChunk in src/shared/models.py
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


# =============================================================================
# ENHANCED CHUNK TYPE ENUM
# =============================================================================

class ChunkType(Enum):
    """Semantic classification of chunk content - extended."""
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    PATHOLOGY = "pathology"
    CLINICAL = "clinical"
    CASE = "case"
    GENERAL = "general"
    # New types
    DIFFERENTIAL = "differential"
    IMAGING = "imaging"
    EVIDENCE = "evidence"
    COMPARATIVE = "comparative"


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
# ENHANCED SEMANTIC CHUNK
# =============================================================================

@dataclass
class EnhancedSemanticChunk:
    """
    Enhanced semantic chunk with additional metadata for synthesis optimization.

    This extends the base SemanticChunk with fields that enable:
    - Procedural step reconstruction
    - Surgical phase ordering
    - High-value content identification (pitfalls, teaching points)
    - Pathology grading context
    """

    # -------------------------------------------------------------------------
    # Core Fields (from base SemanticChunk)
    # -------------------------------------------------------------------------
    id: str
    document_id: str
    content: str
    title: str
    section_path: List[str]
    page_start: int
    page_end: int
    chunk_type: ChunkType

    # Entity fields
    entities: List[Any] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)

    # Reference fields
    figure_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)

    # Classification fields
    specialty_tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    cuis: List[str] = field(default_factory=list)

    # Quality scores
    readability_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0

    # Embedding (optional)
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None

    # -------------------------------------------------------------------------
    # NEW Enhanced Metadata Fields
    # -------------------------------------------------------------------------

    # Procedural metadata
    surgical_phase: Optional[str] = None
    """Phase of surgical procedure (positioning, exposure, approach, etc.)"""

    step_number: Optional[int] = None
    """Explicit step number if present in content (e.g., Step 3)"""

    step_sequence: Optional[str] = None
    """Position in sequence (e.g., '3_of_8' for ordering during synthesis)"""

    # High-value content flags
    has_pitfall: bool = False
    """Contains surgical pitfall, pearl, or critical warning"""

    has_teaching_point: bool = False
    """Contains explicit teaching point or key concept"""

    has_key_measurement: bool = False
    """Contains critical measurements (distances, angles, etc.)"""

    # Pathology-specific metadata
    grading_scale: Optional[str] = None
    """Grading scale used (e.g., 'spetzler_martin', 'who', 'hunt_hess')"""

    grade_value: Optional[str] = None
    """Specific grade if mentioned (e.g., 'III', '4')"""

    molecular_markers: List[str] = field(default_factory=list)
    """Molecular markers mentioned (IDH, MGMT, 1p/19q, etc.)"""

    # Anatomy-specific metadata
    anatomical_region: Optional[str] = None
    """Broad anatomical region (skull_base, spine, vascular, etc.)"""

    spatial_relationships: List[str] = field(default_factory=list)
    """Key spatial relationships mentioned (e.g., 'lateral_to:optic_nerve')"""

    has_variation: bool = False
    """Describes anatomical variation"""

    # Clinical-specific metadata
    has_decision_point: bool = False
    """Contains clinical decision point or algorithm branch"""

    has_evidence_citation: bool = False
    """Contains reference to study or evidence"""

    # Imaging-specific metadata
    imaging_modality: Optional[str] = None
    """Primary imaging modality discussed (MRI, CT, etc.)"""

    imaging_sequences: List[str] = field(default_factory=list)
    """Specific sequences mentioned (T1, T2, FLAIR, etc.)"""

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------

    @property
    def quality_score(self) -> float:
        """Weighted aggregate quality score."""
        return (
            self.readability_score * 0.25 +
            self.coherence_score * 0.35 +
            self.completeness_score * 0.40
        )

    @property
    def is_high_value(self) -> bool:
        """Check if chunk contains high-value content."""
        return self.has_pitfall or self.has_teaching_point or self.has_key_measurement

    @property
    def has_step_context(self) -> bool:
        """Check if chunk has procedural step information."""
        return self.step_number is not None or self.surgical_phase is not None

    @property
    def is_molecular_pathology(self) -> bool:
        """Check if chunk contains molecular pathology data."""
        return len(self.molecular_markers) > 0

    @property
    def is_graded_pathology(self) -> bool:
        """Check if chunk contains grading information."""
        return self.grading_scale is not None

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'title': self.title,
            'section_path': self.section_path,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'chunk_type': self.chunk_type.value if hasattr(self.chunk_type, 'value') else str(self.chunk_type),
            'entity_names': self.entity_names,
            'specialty_tags': self.specialty_tags,
            'keywords': self.keywords,
            'cuis': self.cuis,
            # Quality scores
            'readability_score': self.readability_score,
            'coherence_score': self.coherence_score,
            'completeness_score': self.completeness_score,
            'quality_score': self.quality_score,
            # Enhanced metadata
            'surgical_phase': self.surgical_phase,
            'step_number': self.step_number,
            'step_sequence': self.step_sequence,
            'has_pitfall': self.has_pitfall,
            'has_teaching_point': self.has_teaching_point,
            'has_key_measurement': self.has_key_measurement,
            'grading_scale': self.grading_scale,
            'grade_value': self.grade_value,
            'molecular_markers': self.molecular_markers,
            'anatomical_region': self.anatomical_region,
            'has_variation': self.has_variation,
            'has_decision_point': self.has_decision_point,
            'imaging_modality': self.imaging_modality,
            'imaging_sequences': self.imaging_sequences,
            # Computed
            'is_high_value': self.is_high_value,
            'has_step_context': self.has_step_context,
        }

    def to_db(self) -> Dict[str, Any]:
        """Convert to database-ready dictionary."""
        base = self.to_dict()

        # Convert lists to JSON-compatible format
        base['section_path'] = json.dumps(self.section_path)
        base['entity_names'] = json.dumps(self.entity_names)
        base['specialty_tags'] = json.dumps(self.specialty_tags)
        base['keywords'] = json.dumps(self.keywords)
        base['cuis'] = json.dumps(self.cuis)
        base['molecular_markers'] = json.dumps(self.molecular_markers)
        base['spatial_relationships'] = json.dumps(self.spatial_relationships)
        base['imaging_sequences'] = json.dumps(self.imaging_sequences)

        # Remove computed properties (not stored)
        base.pop('quality_score', None)
        base.pop('is_high_value', None)
        base.pop('has_step_context', None)

        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedSemanticChunk':
        """Create from dictionary."""
        # Convert chunk_type string to enum
        chunk_type = data.get('chunk_type', 'general')
        if isinstance(chunk_type, str):
            try:
                chunk_type = ChunkType(chunk_type)
            except ValueError:
                chunk_type = ChunkType.GENERAL

        return cls(
            id=data['id'],
            document_id=data['document_id'],
            content=data['content'],
            title=data.get('title', ''),
            section_path=data.get('section_path', []),
            page_start=data.get('page_start', 0),
            page_end=data.get('page_end', 0),
            chunk_type=chunk_type,
            entity_names=data.get('entity_names', []),
            specialty_tags=data.get('specialty_tags', []),
            keywords=data.get('keywords', []),
            cuis=data.get('cuis', []),
            readability_score=data.get('readability_score', 0.0),
            coherence_score=data.get('coherence_score', 0.0),
            completeness_score=data.get('completeness_score', 0.0),
            surgical_phase=data.get('surgical_phase'),
            step_number=data.get('step_number'),
            step_sequence=data.get('step_sequence'),
            has_pitfall=data.get('has_pitfall', False),
            has_teaching_point=data.get('has_teaching_point', False),
            has_key_measurement=data.get('has_key_measurement', False),
            grading_scale=data.get('grading_scale'),
            grade_value=data.get('grade_value'),
            molecular_markers=data.get('molecular_markers', []),
            anatomical_region=data.get('anatomical_region'),
            spatial_relationships=data.get('spatial_relationships', []),
            has_variation=data.get('has_variation', False),
            has_decision_point=data.get('has_decision_point', False),
            has_evidence_citation=data.get('has_evidence_citation', False),
            imaging_modality=data.get('imaging_modality'),
            imaging_sequences=data.get('imaging_sequences', []),
        )


# =============================================================================
# DATABASE SCHEMA ADDITIONS
# =============================================================================

SCHEMA_ADDITIONS_SQL = """
-- Add enhanced metadata columns to chunks table
-- Run this migration after the base schema is in place

-- Procedural metadata
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS surgical_phase VARCHAR(50);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS step_number INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS step_sequence VARCHAR(20);

-- High-value content flags
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_pitfall BOOLEAN DEFAULT FALSE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_teaching_point BOOLEAN DEFAULT FALSE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_key_measurement BOOLEAN DEFAULT FALSE;

-- Pathology metadata
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS grading_scale VARCHAR(50);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS grade_value VARCHAR(20);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS molecular_markers JSONB DEFAULT '[]';

-- Anatomy metadata
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS anatomical_region VARCHAR(50);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS spatial_relationships JSONB DEFAULT '[]';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_variation BOOLEAN DEFAULT FALSE;

-- Clinical metadata
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_decision_point BOOLEAN DEFAULT FALSE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS has_evidence_citation BOOLEAN DEFAULT FALSE;

-- Imaging metadata
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS imaging_modality VARCHAR(50);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS imaging_sequences JSONB DEFAULT '[]';

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_surgical_phase ON chunks(surgical_phase) WHERE surgical_phase IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_step_number ON chunks(step_number) WHERE step_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_has_pitfall ON chunks(has_pitfall) WHERE has_pitfall = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_grading_scale ON chunks(grading_scale) WHERE grading_scale IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_anatomical_region ON chunks(anatomical_region) WHERE anatomical_region IS NOT NULL;

-- Comment documentation
COMMENT ON COLUMN chunks.surgical_phase IS 'Phase of surgical procedure (positioning, exposure, approach, etc.)';
COMMENT ON COLUMN chunks.step_number IS 'Explicit step number if present (e.g., Step 3)';
COMMENT ON COLUMN chunks.step_sequence IS 'Position in sequence (e.g., 3_of_8)';
COMMENT ON COLUMN chunks.has_pitfall IS 'Contains surgical pitfall, pearl, or critical warning';
COMMENT ON COLUMN chunks.has_teaching_point IS 'Contains explicit teaching point';
COMMENT ON COLUMN chunks.grading_scale IS 'Grading scale used (spetzler_martin, who, hunt_hess, etc.)';
COMMENT ON COLUMN chunks.molecular_markers IS 'Molecular markers mentioned (IDH, MGMT, etc.)';
COMMENT ON COLUMN chunks.anatomical_region IS 'Broad anatomical region (skull_base, spine, vascular)';
"""


def get_schema_additions() -> str:
    """Get SQL for schema additions."""
    return SCHEMA_ADDITIONS_SQL
