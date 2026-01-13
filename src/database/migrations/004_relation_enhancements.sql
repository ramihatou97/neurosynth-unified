-- Migration: 004_relation_enhancements
-- Purpose: Add negation tracking and extraction method to entity_relations
-- Date: 2025-01-12
-- Depends on: 003_entity_relations.sql

-- =============================================================================
-- Schema Changes
-- =============================================================================

-- Add negation tracking columns
ALTER TABLE entity_relations
ADD COLUMN IF NOT EXISTS is_negated BOOLEAN DEFAULT FALSE;

ALTER TABLE entity_relations
ADD COLUMN IF NOT EXISTS negation_cue TEXT;

-- Add extraction method tracking
ALTER TABLE entity_relations
ADD COLUMN IF NOT EXISTS extraction_method TEXT DEFAULT 'dependency';

-- Add column comments for documentation
COMMENT ON COLUMN entity_relations.is_negated IS 
    'True if relation was expressed in negative form (e.g., "no evidence X supplies Y")';

COMMENT ON COLUMN entity_relations.negation_cue IS 
    'The negation phrase that triggered is_negated (e.g., "no evidence", "not", "without")';

COMMENT ON COLUMN entity_relations.extraction_method IS 
    'Extraction method used. Values: dependency, entity_first, llm_complete, llm_verified, hybrid';

-- =============================================================================
-- Indexes
-- =============================================================================

-- Index for filtering negated relations (partial index for efficiency)
CREATE INDEX IF NOT EXISTS idx_er_negated
ON entity_relations(is_negated) 
WHERE is_negated = TRUE;

-- Index for extraction method analysis
CREATE INDEX IF NOT EXISTS idx_er_extraction_method
ON entity_relations(extraction_method);

-- Composite index for common query pattern: non-negated relations by type
CREATE INDEX IF NOT EXISTS idx_er_type_not_negated
ON entity_relations(relation_type, confidence DESC)
WHERE is_negated = FALSE;

-- =============================================================================
-- Data Migration (set defaults for existing rows)
-- =============================================================================

-- Ensure existing relations have extraction_method set
UPDATE entity_relations 
SET extraction_method = 'dependency' 
WHERE extraction_method IS NULL;

-- Ensure existing relations have is_negated set
UPDATE entity_relations 
SET is_negated = FALSE 
WHERE is_negated IS NULL;

-- =============================================================================
-- Verification Query (run after migration)
-- =============================================================================

-- Verify migration success:
-- SELECT 
--     COUNT(*) as total_relations,
--     COUNT(*) FILTER (WHERE is_negated IS NOT NULL) as has_negation_flag,
--     COUNT(*) FILTER (WHERE extraction_method IS NOT NULL) as has_method,
--     COUNT(DISTINCT extraction_method) as method_count
-- FROM entity_relations;

-- =============================================================================
-- Rollback Script (if needed)
-- =============================================================================

-- To rollback this migration:
-- ALTER TABLE entity_relations DROP COLUMN IF EXISTS is_negated;
-- ALTER TABLE entity_relations DROP COLUMN IF EXISTS negation_cue;
-- ALTER TABLE entity_relations DROP COLUMN IF EXISTS extraction_method;
-- DROP INDEX IF EXISTS idx_er_negated;
-- DROP INDEX IF EXISTS idx_er_extraction_method;
-- DROP INDEX IF EXISTS idx_er_type_not_negated;
