-- Migration 006: Add pages table for Deep Research mode
-- Required by: src/rag/unified_engine.py (DeepResearchEngine)
-- Purpose: Store full page text for Gemini 2M context window analysis

CREATE TABLE IF NOT EXISTS pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    raw_text TEXT,  -- Original unprocessed text (optional)
    word_count INTEGER GENERATED ALWAYS AS (array_length(string_to_array(content, ' '), 1)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, page_number)
);

-- Index for efficient document lookups
CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages(document_id);

-- Index for page ordering
CREATE INDEX IF NOT EXISTS idx_pages_document_page ON pages(document_id, page_number);

-- Comment for documentation
COMMENT ON TABLE pages IS 'Full page content for Deep Research mode (V3). Stores complete page text for Gemini 2M context analysis.';
COMMENT ON COLUMN pages.content IS 'Processed page text for LLM consumption';
COMMENT ON COLUMN pages.raw_text IS 'Original extracted text before processing (optional)';
