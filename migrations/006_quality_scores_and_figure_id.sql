-- Migration 006: Add quality scores and figure_id columns
-- Date: 2026-01-13
-- Purpose: Ensure PipelineDatabaseWriter has schema parity with NeuroDatabase
--
-- This migration adds columns that may already exist from other migrations or
-- manual schema updates. Uses IF NOT EXISTS pattern for idempotency.

-- =============================================================================
-- CHUNKS TABLE: Quality Score Columns (4 dimensions)
-- =============================================================================

-- Readability score (0.0-1.0): How clear and readable the chunk is
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'readability_score'
    ) THEN
        ALTER TABLE chunks ADD COLUMN readability_score FLOAT DEFAULT 0.0;
        RAISE NOTICE 'Added readability_score column to chunks';
    ELSE
        RAISE NOTICE 'readability_score column already exists in chunks';
    END IF;
END $$;

-- Coherence score (0.0-1.0): How well sentences connect logically
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'coherence_score'
    ) THEN
        ALTER TABLE chunks ADD COLUMN coherence_score FLOAT DEFAULT 0.0;
        RAISE NOTICE 'Added coherence_score column to chunks';
    ELSE
        RAISE NOTICE 'coherence_score column already exists in chunks';
    END IF;
END $$;

-- Completeness score (0.0-1.0): Whether chunk is self-contained
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'completeness_score'
    ) THEN
        ALTER TABLE chunks ADD COLUMN completeness_score FLOAT DEFAULT 0.0;
        RAISE NOTICE 'Added completeness_score column to chunks';
    ELSE
        RAISE NOTICE 'completeness_score column already exists in chunks';
    END IF;
END $$;

-- Type-specific score (0.0-1.0): Quality based on chunk type requirements
-- Note: May already exist from migration 005_enhanced_chunk_metadata.sql
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'type_specific_score'
    ) THEN
        ALTER TABLE chunks ADD COLUMN type_specific_score FLOAT DEFAULT 0.0;
        RAISE NOTICE 'Added type_specific_score column to chunks';
    ELSE
        RAISE NOTICE 'type_specific_score column already exists in chunks';
    END IF;
END $$;

-- Is orphan flag: Chunk appears to be mid-sequence
-- Note: May already exist from migration 005_enhanced_chunk_metadata.sql
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'is_orphan'
    ) THEN
        ALTER TABLE chunks ADD COLUMN is_orphan BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added is_orphan column to chunks';
    ELSE
        RAISE NOTICE 'is_orphan column already exists in chunks';
    END IF;
END $$;

-- =============================================================================
-- IMAGES TABLE: Figure ID and Quality Score
-- =============================================================================

-- Figure ID: Extracted from VLM caption or PDF caption (e.g., "Figure 6.3", "Fig. 2A")
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'images' AND column_name = 'figure_id'
    ) THEN
        ALTER TABLE images ADD COLUMN figure_id TEXT;
        RAISE NOTICE 'Added figure_id column to images';
    ELSE
        RAISE NOTICE 'figure_id column already exists in images';
    END IF;
END $$;

-- Quality score for images (0.0-1.0)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'images' AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE images ADD COLUMN quality_score FLOAT;
        RAISE NOTICE 'Added quality_score column to images';
    ELSE
        RAISE NOTICE 'quality_score column already exists in images';
    END IF;
END $$;

-- =============================================================================
-- INDEXES for performance
-- =============================================================================

-- Composite quality score index for ranking chunks
CREATE INDEX IF NOT EXISTS idx_chunks_quality_composite
ON chunks (((readability_score + coherence_score + completeness_score + type_specific_score) / 4.0) DESC)
WHERE readability_score IS NOT NULL;

-- Figure ID index for fast lookups
CREATE INDEX IF NOT EXISTS idx_images_figure_id
ON images (figure_id)
WHERE figure_id IS NOT NULL;

-- Orphan chunks index
CREATE INDEX IF NOT EXISTS idx_chunks_is_orphan
ON chunks (is_orphan)
WHERE is_orphan = TRUE;

-- =============================================================================
-- COMMENTS for documentation
-- =============================================================================

COMMENT ON COLUMN chunks.readability_score IS 'Readability score (0-1): clarity and readability of content';
COMMENT ON COLUMN chunks.coherence_score IS 'Coherence score (0-1): logical flow between sentences';
COMMENT ON COLUMN chunks.completeness_score IS 'Completeness score (0-1): self-containment of content';
COMMENT ON COLUMN chunks.type_specific_score IS 'Type-specific score (0-1): meets requirements for chunk type';
COMMENT ON COLUMN chunks.is_orphan IS 'True if chunk appears to be mid-sequence (starts with Step 2+, Then, etc.)';
COMMENT ON COLUMN images.figure_id IS 'Figure identifier extracted from caption (e.g., Figure 6.3, Fig. 2A, Plate 1)';
COMMENT ON COLUMN images.quality_score IS 'Overall quality score for image/caption (0-1)';

-- =============================================================================
-- VERIFICATION
-- =============================================================================

SELECT
    'Schema verification' as check_type,
    (SELECT COUNT(*) FROM information_schema.columns
     WHERE table_name = 'chunks'
     AND column_name IN ('readability_score', 'coherence_score', 'completeness_score', 'type_specific_score', 'is_orphan')
    ) as chunks_quality_columns,
    (SELECT COUNT(*) FROM information_schema.columns
     WHERE table_name = 'images'
     AND column_name IN ('figure_id', 'quality_score')
    ) as images_columns;
