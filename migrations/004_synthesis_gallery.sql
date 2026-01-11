-- =============================================================================
-- Migration 004: Synthesis Gallery
-- =============================================================================
-- Persistent storage for completed synthesis results with image previews
-- =============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- SYNTHESIS GALLERY TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS synthesis_gallery (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Core metadata
    topic TEXT NOT NULL,
    template_type TEXT NOT NULL,  -- PROCEDURAL, DISORDER, ANATOMY, ENCYCLOPEDIA
    title TEXT NOT NULL,
    abstract TEXT,

    -- Content storage
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Format: [{"title": "...", "content": "...", "level": 2, "word_count": 0}]

    source_references JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Format: [{"source": "...", "document_id": "...", "authority": "..."}]

    -- Image data for gallery preview
    resolved_figures JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Format: [{"image_path": "...", "image_caption": "...", "placeholder_id": "..."}]

    figure_requests JSONB DEFAULT '[]'::jsonb,
    -- Format: [{"placeholder_id": "...", "topic": "...", "figure_type": "..."}]

    -- Rendered content (for quick display)
    markdown_content TEXT,

    -- Statistics
    total_words INTEGER DEFAULT 0,
    total_figures INTEGER DEFAULT 0,
    total_citations INTEGER DEFAULT 0,
    synthesis_time_ms INTEGER DEFAULT 0,

    -- Verification (if available)
    verification_score FLOAT,
    verified BOOLEAN DEFAULT FALSE,

    -- Conflict detection
    conflict_count INTEGER DEFAULT 0,
    conflict_report JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- User tracking (optional)
    user_id TEXT,

    -- Status
    is_favorite BOOLEAN DEFAULT FALSE,
    tags TEXT[] DEFAULT '{}'
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Time-based queries (most recent first)
CREATE INDEX IF NOT EXISTS idx_synthesis_gallery_created
    ON synthesis_gallery(created_at DESC);

-- Topic search
CREATE INDEX IF NOT EXISTS idx_synthesis_gallery_topic
    ON synthesis_gallery USING gin(to_tsvector('english', topic));

-- Template type filter
CREATE INDEX IF NOT EXISTS idx_synthesis_gallery_template
    ON synthesis_gallery(template_type);

-- Favorites
CREATE INDEX IF NOT EXISTS idx_synthesis_gallery_favorite
    ON synthesis_gallery(is_favorite) WHERE is_favorite = TRUE;

-- Tags (GIN for array containment)
CREATE INDEX IF NOT EXISTS idx_synthesis_gallery_tags
    ON synthesis_gallery USING GIN(tags);

-- =============================================================================
-- TRIGGER: Update updated_at on modification
-- =============================================================================

CREATE OR REPLACE FUNCTION update_synthesis_gallery_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_synthesis_gallery_updated ON synthesis_gallery;
CREATE TRIGGER trigger_synthesis_gallery_updated
    BEFORE UPDATE ON synthesis_gallery
    FOR EACH ROW EXECUTE FUNCTION update_synthesis_gallery_timestamp();

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE synthesis_gallery IS 'Persistent storage for completed synthesis results with image previews and full content';
COMMENT ON COLUMN synthesis_gallery.sections IS 'Array of section objects with title, content, level, word_count';
COMMENT ON COLUMN synthesis_gallery.resolved_figures IS 'Array of resolved images with paths and captions for preview';
COMMENT ON COLUMN synthesis_gallery.markdown_content IS 'Pre-rendered markdown for quick display';
