-- Migration 007: Add summary columns for chunk and image caption summarization
-- These columns store AI-generated summaries from Stages 4.5 and 8.5 of the pipeline

-- Add summary column to chunks table (Stage 4.5 output)
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS summary TEXT;

-- Add caption_summary column to images table (Stage 8.5 output)
ALTER TABLE images
ADD COLUMN IF NOT EXISTS caption_summary TEXT;

-- Add comment for documentation
COMMENT ON COLUMN chunks.summary IS 'AI-generated summary highlighting unique aspects (Stage 4.5)';
COMMENT ON COLUMN images.caption_summary IS 'AI-generated caption summary (Stage 8.5)';
