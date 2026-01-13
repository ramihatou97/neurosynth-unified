-- Migration: Add summary_embedding column for multi-index retrieval
-- Date: 2024-01-13
-- Purpose: Enable summary-based semantic search alongside content search

-- Add summary_embedding column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'summary_embedding'
    ) THEN
        ALTER TABLE chunks ADD COLUMN summary_embedding vector(1024);
        RAISE NOTICE 'Added summary_embedding column';
    ELSE
        RAISE NOTICE 'summary_embedding column already exists';
    END IF;
END $$;

-- Create HNSW index for fast approximate nearest neighbor search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_summary_embedding_hnsw
ON chunks USING hnsw (summary_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Add caption_quality_score to images if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'images' AND column_name = 'caption_quality_score'
    ) THEN
        ALTER TABLE images ADD COLUMN caption_quality_score FLOAT;
        RAISE NOTICE 'Added caption_quality_score column';
    END IF;
END $$;

-- Add needs_recaption flag
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'images' AND column_name = 'needs_recaption'
    ) THEN
        ALTER TABLE images ADD COLUMN needs_recaption BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added needs_recaption column';
    END IF;
END $$;

-- Verify migration
SELECT
    'chunks.summary_embedding' as column_check,
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='summary_embedding') as exists
UNION ALL
SELECT
    'images.caption_quality_score',
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='images' AND column_name='caption_quality_score')
UNION ALL
SELECT
    'images.needs_recaption',
    EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='images' AND column_name='needs_recaption');
