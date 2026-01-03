-- =============================================================================
-- Migration: 004_authority_registry
-- Purpose: Persist authority registry configuration for neurosurgical sources
-- Date: 2025-01-03
-- =============================================================================
--
-- This migration creates a singleton table to store the AuthorityRegistry
-- configuration, allowing customization of authority scores and sources
-- to persist across application restarts.
--
-- The registry supports:
-- - Modifying scores for built-in sources (RHOTON, LAWTON, YOUMANS, etc.)
-- - Adding custom authority sources with keywords
-- - Tiered authority levels (1=Master, 2=Textbook, 3=Reference)
--
-- Usage:
--   -- View current config
--   SELECT config FROM authority_registry WHERE id = 1;
--
--   -- Update config (typically done via API)
--   UPDATE authority_registry SET config = '{"scores": {...}}' WHERE id = 1;
--
-- =============================================================================

-- Authority registry singleton table
-- Uses CHECK constraint to ensure only one row exists
CREATE TABLE IF NOT EXISTS authority_registry (
    id INTEGER PRIMARY KEY DEFAULT 1,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Singleton constraint: only id=1 allowed
    CONSTRAINT authority_registry_singleton CHECK (id = 1)
);

-- Add comment for documentation
COMMENT ON TABLE authority_registry IS 'Singleton table storing authority registry configuration for neurosurgical sources';
COMMENT ON COLUMN authority_registry.config IS 'JSONB config with scores, keywords, and custom_sources';

-- Seed with empty config (application will use code defaults if empty)
INSERT INTO authority_registry (id, config)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- Auto-update trigger for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_authority_registry_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS authority_registry_updated ON authority_registry;
CREATE TRIGGER authority_registry_updated
    BEFORE UPDATE ON authority_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_authority_registry_timestamp();

-- =============================================================================
-- Helper function to get registry config
-- =============================================================================

CREATE OR REPLACE FUNCTION get_authority_config()
RETURNS JSONB AS $$
    SELECT COALESCE(config, '{}'::jsonb) FROM authority_registry WHERE id = 1;
$$ LANGUAGE sql STABLE;

COMMENT ON FUNCTION get_authority_config() IS 'Get current authority registry configuration';

-- =============================================================================
-- Example config structure (for documentation)
-- =============================================================================
--
-- {
--   "scores": {
--     "RHOTON": 1.0,
--     "LAWTON": 1.0,
--     "YOUMANS": 0.9,
--     "GENERAL": 0.7
--   },
--   "keywords": {
--     "RHOTON": ["rhoton", "cranial anatomy"],
--     "LAWTON": ["lawton", "seven avms"]
--   },
--   "custom_sources": {
--     "MY_SOURCE": {
--       "name": "MY_SOURCE",
--       "score": 0.85,
--       "keywords": ["custom", "source"],
--       "tier": 3
--     }
--   }
-- }
--
-- =============================================================================
