-- ============================================================================
-- Migration: Add authority_score column to documents table
-- ============================================================================
--
-- The Document model already has authority_score field (shared/models.py:279),
-- but the database schema is missing the column. This migration adds it.
--
-- Background:
-- Phase 1 pipeline computes authority_score during ingestion based on
-- document title patterns (Rhoton, Youmans, Schmidek, etc.) but the score
-- was never persisted to the database. This caused SearchService to be unable
-- to query it, breaking synthesis integration which relies on authority-
-- weighted source ranking.
--
-- Run: psql $DATABASE_URL -f migrations/002_add_authority_score_column.sql
--

BEGIN;

-- Add authority_score column to documents table
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS authority_score FLOAT DEFAULT 1.0;

-- Extract from metadata JSONB if it was stored there previously
-- (Phase 1 may have stored it in metadata instead of dedicated column)
UPDATE documents
SET authority_score = COALESCE((metadata->>'authority_score')::float, 1.0)
WHERE authority_score = 1.0
  AND metadata->>'authority_score' IS NOT NULL;

-- ============================================================================
-- Backfill based on document title patterns
-- ============================================================================
-- These patterns match the authority detection logic in Phase 1 pipeline
-- (src/ingest/pipeline.py _compute_authority_score method)

-- Rhoton (highest authority for microsurgical anatomy)
-- Score: 0.95
UPDATE documents SET authority_score = 0.95
WHERE (title ILIKE '%rhoton%' OR title ILIKE '%microsurgical anatomy%')
  AND authority_score = 1.0;

-- Youmans (comprehensive neurosurgery textbook)
-- Score: 0.90
UPDATE documents SET authority_score = 0.90
WHERE (title ILIKE '%youmans%' OR title ILIKE '%neurological surgery%')
  AND authority_score = 1.0;

-- Schmidek & Sweet (operative techniques)
-- Score: 0.88
UPDATE documents SET authority_score = 0.88
WHERE (title ILIKE '%schmidek%' OR title ILIKE '%operative neurosurgical%' OR title ILIKE '%sweet%')
  AND authority_score = 1.0;

-- Greenberg (handbook - quick reference)
-- Score: 0.85
UPDATE documents SET authority_score = 0.85
WHERE (title ILIKE '%greenberg%' OR title ILIKE '%handbook of neurosurgery%')
  AND authority_score = 1.0;

-- Journal articles (peer-reviewed)
-- Score: 0.80
UPDATE documents SET authority_score = 0.80
WHERE (title ILIKE '%journal%' OR title ILIKE '%j neurosurg%' OR title ILIKE '%j. neurosurg%')
  AND authority_score = 1.0;

-- General textbooks
-- Score: 0.75
UPDATE documents SET authority_score = 0.75
WHERE (title ILIKE '%textbook%' OR title ILIKE '%principles%')
  AND authority_score = 1.0;

-- Create index for authority-weighted queries
-- This supports ORDER BY authority_score DESC in synthesis ranking
CREATE INDEX IF NOT EXISTS idx_documents_authority_score
ON documents(authority_score DESC);

COMMIT;

-- ============================================================================
-- Verification Queries
-- ============================================================================
-- Run these to verify the migration succeeded:

-- Check column exists
-- \d documents

-- Check authority score distribution
-- SELECT authority_score, COUNT(*) as count
-- FROM documents
-- GROUP BY authority_score
-- ORDER BY authority_score DESC;

-- Check high-authority sources
-- SELECT title, authority_score
-- FROM documents
-- WHERE authority_score >= 0.85
-- ORDER BY authority_score DESC
-- LIMIT 10;
