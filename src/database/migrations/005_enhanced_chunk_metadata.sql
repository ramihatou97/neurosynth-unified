-- Migration 005: Add enhanced chunk metadata columns (NeuroSynth v2.2)
-- Adds fields for procedural step tracking, pathology grading, and high-value content flags

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

-- v2.2 Orphan detection and type-specific scoring
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_orphan BOOLEAN DEFAULT FALSE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS type_specific_score FLOAT DEFAULT 0.0;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chunks_surgical_phase ON chunks(surgical_phase) WHERE surgical_phase IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_step_number ON chunks(step_number) WHERE step_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_has_pitfall ON chunks(has_pitfall) WHERE has_pitfall = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_has_teaching_point ON chunks(has_teaching_point) WHERE has_teaching_point = TRUE;
CREATE INDEX IF NOT EXISTS idx_chunks_grading_scale ON chunks(grading_scale) WHERE grading_scale IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_anatomical_region ON chunks(anatomical_region) WHERE anatomical_region IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_imaging_modality ON chunks(imaging_modality) WHERE imaging_modality IS NOT NULL;

-- Comment documentation
COMMENT ON COLUMN chunks.surgical_phase IS 'Phase of surgical procedure (positioning, exposure, approach, etc.)';
COMMENT ON COLUMN chunks.step_number IS 'Explicit step number if present (e.g., Step 3)';
COMMENT ON COLUMN chunks.step_sequence IS 'Position in sequence (e.g., 3_of_8)';
COMMENT ON COLUMN chunks.has_pitfall IS 'Contains surgical pitfall, pearl, or critical warning';
COMMENT ON COLUMN chunks.has_teaching_point IS 'Contains explicit teaching point';
COMMENT ON COLUMN chunks.has_key_measurement IS 'Contains critical measurements (distances, angles, etc.)';
COMMENT ON COLUMN chunks.grading_scale IS 'Grading scale used (spetzler_martin, who, hunt_hess, etc.)';
COMMENT ON COLUMN chunks.grade_value IS 'Specific grade value (e.g., III, 4)';
COMMENT ON COLUMN chunks.molecular_markers IS 'Molecular markers mentioned (IDH, MGMT, etc.)';
COMMENT ON COLUMN chunks.anatomical_region IS 'Broad anatomical region (skull_base, spine, vascular)';
COMMENT ON COLUMN chunks.spatial_relationships IS 'Key spatial relationships (lateral_to:optic_nerve)';
COMMENT ON COLUMN chunks.has_variation IS 'Describes anatomical variation';
COMMENT ON COLUMN chunks.has_decision_point IS 'Contains clinical decision point or algorithm branch';
COMMENT ON COLUMN chunks.has_evidence_citation IS 'Contains reference to study or evidence';
COMMENT ON COLUMN chunks.imaging_modality IS 'Primary imaging modality discussed (MRI, CT, etc.)';
COMMENT ON COLUMN chunks.imaging_sequences IS 'Specific sequences mentioned (T1, T2, FLAIR, etc.)';
COMMENT ON COLUMN chunks.is_orphan IS 'Chunk appears to be mid-sequence (starts with Step 2+, Then, etc.)';
COMMENT ON COLUMN chunks.type_specific_score IS 'Type-specific quality score (v2.2 fourth dimension)';
