-- =============================================================================
-- NeuroSynth Phase 2 - PostgreSQL Schema
-- =============================================================================
-- Requires: PostgreSQL 14+ with pgvector extension
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- DOCUMENTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path TEXT NOT NULL,
    title TEXT,
    total_pages INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    total_images INTEGER DEFAULT 0,
    processing_time_seconds FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for source path lookups
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_path);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);

-- =============================================================================
-- CHUNKS TABLE (Text chunks with embeddings)
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Content
    content TEXT NOT NULL,
    content_hash TEXT,  -- For deduplication
    
    -- Position
    page_number INTEGER,
    chunk_index INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    
    -- Classification
    chunk_type TEXT,      -- PROCEDURE, ANATOMY, PATHOLOGY, CLINICAL, CASE, etc.
    specialty TEXT,       -- Neurosurgery subspecialty
    
    -- Embeddings (1024d Voyage-3)
    embedding vector(1024),
    
    -- UMLS/Entities
    cuis TEXT[] DEFAULT '{}',  -- Array of UMLS CUIs
    entities JSONB DEFAULT '[]'::jsonb,  -- Full entity details
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_specialty ON chunks(specialty);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- Vector similarity index (IVFFlat with cosine distance)
-- Note: Create after data is loaded for better clustering
-- CREATE INDEX idx_chunks_embedding ON chunks 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- GIN index for CUI array searches
CREATE INDEX IF NOT EXISTS idx_chunks_cuis ON chunks USING GIN(cuis);

-- =============================================================================
-- IMAGES TABLE (Medical images with dual embeddings)
-- =============================================================================

CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- File info
    file_path TEXT NOT NULL,
    file_name TEXT,
    content_hash TEXT,  -- For deduplication
    
    -- Dimensions
    width INTEGER,
    height INTEGER,
    format TEXT,  -- JPEG, PNG, etc.
    
    -- Position
    page_number INTEGER,
    image_index INTEGER,
    
    -- Classification
    image_type TEXT,  -- MRI_CT, SURGICAL_PHOTO, ANATOMICAL_DIAGRAM, HISTOLOGY, etc.
    is_decorative BOOLEAN DEFAULT FALSE,
    
    -- VLM Caption
    vlm_caption TEXT,
    vlm_confidence FLOAT,
    
    -- Embeddings
    embedding vector(512),        -- BiomedCLIP visual embedding
    caption_embedding vector(1024),  -- Voyage-3 caption embedding
    
    -- UMLS/Entities (from caption)
    cuis TEXT[] DEFAULT '{}',
    entities JSONB DEFAULT '[]'::jsonb,
    
    -- Triage info (Phase 1)
    triage_skipped BOOLEAN DEFAULT FALSE,
    triage_reason TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_images_document ON images(document_id);
CREATE INDEX IF NOT EXISTS idx_images_page ON images(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_images_type ON images(image_type);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(content_hash);
CREATE INDEX IF NOT EXISTS idx_images_decorative ON images(is_decorative) WHERE NOT is_decorative;

-- Vector similarity indexes
-- CREATE INDEX idx_images_embedding ON images 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
-- CREATE INDEX idx_images_caption_embedding ON images 
--     USING ivfflat (caption_embedding vector_cosine_ops) WITH (lists = 50);

-- GIN index for CUI searches
CREATE INDEX IF NOT EXISTS idx_images_cuis ON images USING GIN(cuis);

-- =============================================================================
-- LINKS TABLE (Chunk-Image relationships)
-- =============================================================================

CREATE TABLE IF NOT EXISTS links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    
    -- Link details
    link_type TEXT NOT NULL,  -- proximity, semantic, cui_match, explicit_ref
    score FLOAT NOT NULL,     -- Link strength (0-1)
    
    -- Scoring components
    proximity_score FLOAT,
    semantic_score FLOAT,
    cui_overlap_score FLOAT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Unique constraint to prevent duplicates
    UNIQUE(chunk_id, image_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_links_chunk ON links(chunk_id);
CREATE INDEX IF NOT EXISTS idx_links_image ON links(image_id);
CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_score ON links(score DESC);

-- =============================================================================
-- ENTITIES TABLE (UMLS concepts for faceting)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cui TEXT UNIQUE NOT NULL,  -- UMLS Concept Unique Identifier
    name TEXT NOT NULL,        -- Preferred name
    semantic_type TEXT,        -- Semantic type name
    tui TEXT,                  -- Type Unique Identifier
    
    -- Statistics
    chunk_count INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_cui ON entities(cui);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(semantic_type);
CREATE INDEX IF NOT EXISTS idx_entities_count ON entities(chunk_count DESC);

-- =============================================================================
-- SEARCH HISTORY TABLE (Optional - for analytics)
-- =============================================================================

CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    query_embedding vector(1024),
    search_mode TEXT,  -- text, image, hybrid
    filters JSONB,
    result_count INTEGER,
    latency_ms INTEGER,
    user_id TEXT,  -- Optional user tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to update document statistics
CREATE OR REPLACE FUNCTION update_document_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE documents SET
        total_chunks = (SELECT COUNT(*) FROM chunks WHERE document_id = NEW.document_id),
        total_images = (SELECT COUNT(*) FROM images WHERE document_id = NEW.document_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.document_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to update stats
DROP TRIGGER IF EXISTS trigger_update_doc_stats_chunks ON chunks;
CREATE TRIGGER trigger_update_doc_stats_chunks
    AFTER INSERT OR DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_stats();

DROP TRIGGER IF EXISTS trigger_update_doc_stats_images ON images;
CREATE TRIGGER trigger_update_doc_stats_images
    AFTER INSERT OR DELETE ON images
    FOR EACH ROW EXECUTE FUNCTION update_document_stats();

-- Function for semantic search
CREATE OR REPLACE FUNCTION search_chunks_semantic(
    query_embedding vector(1024),
    match_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    chunk_type TEXT,
    specialty TEXT,
    page_number INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.document_id,
        c.content,
        c.chunk_type,
        c.specialty,
        c.page_number,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function for hybrid search (text + CUI matching)
CREATE OR REPLACE FUNCTION search_chunks_hybrid(
    query_embedding vector(1024),
    query_cuis TEXT[] DEFAULT '{}',
    match_count INTEGER DEFAULT 10,
    cui_boost FLOAT DEFAULT 1.2
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    chunk_type TEXT,
    page_number INTEGER,
    similarity FLOAT,
    cui_overlap INTEGER,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.document_id,
        c.content,
        c.chunk_type,
        c.page_number,
        1 - (c.embedding <=> query_embedding) AS similarity,
        COALESCE(array_length(c.cuis & query_cuis, 1), 0) AS cui_overlap,
        (1 - (c.embedding <=> query_embedding)) * 
            CASE WHEN array_length(c.cuis & query_cuis, 1) > 0 
                 THEN cui_boost 
                 ELSE 1.0 
            END AS combined_score
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View for chunk statistics
CREATE OR REPLACE VIEW chunk_stats AS
SELECT 
    document_id,
    COUNT(*) as total_chunks,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as chunks_with_embedding,
    COUNT(*) FILTER (WHERE array_length(cuis, 1) > 0) as chunks_with_cuis,
    COUNT(DISTINCT chunk_type) as unique_types,
    COUNT(DISTINCT specialty) as unique_specialties
FROM chunks
GROUP BY document_id;

-- View for image statistics
CREATE OR REPLACE VIEW image_stats AS
SELECT
    document_id,
    COUNT(*) as total_images,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as images_with_embedding,
    COUNT(*) FILTER (WHERE caption_embedding IS NOT NULL) as images_with_caption_embedding,
    COUNT(*) FILTER (WHERE vlm_caption IS NOT NULL) as images_with_caption,
    COUNT(*) FILTER (WHERE NOT is_decorative) as non_decorative_images,
    COUNT(DISTINCT image_type) as unique_types
FROM images
GROUP BY document_id;

-- =============================================================================
-- MATERIALIZED VIEWS (for performance optimization)
-- =============================================================================

-- Materialized view for top 3 image links per chunk
-- Pre-computes the most common query pattern: "get top 3 images for each chunk"
-- Expected improvement: 50% faster image linking (10ms → 5ms)
CREATE MATERIALIZED VIEW IF NOT EXISTS top_chunk_links AS
SELECT
    chunk_id,
    image_id,
    link_type,
    score,
    proximity_score,
    semantic_score,
    cui_overlap_score,
    created_at,
    metadata
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY chunk_id
            ORDER BY score DESC
        ) as rank
    FROM links
    WHERE score >= 0.5  -- Only include meaningful links
) ranked
WHERE rank <= 3;  -- Top 3 per chunk

-- Index on the materialized view
CREATE INDEX IF NOT EXISTS idx_top_chunk_links_chunk ON top_chunk_links(chunk_id);
CREATE INDEX IF NOT EXISTS idx_top_chunk_links_image ON top_chunk_links(image_id);

-- Refresh function (call after bulk link inserts)
-- Usage: REFRESH MATERIALIZED VIEW CONCURRENTLY top_chunk_links;
-- Note: Requires UNIQUE index for CONCURRENTLY option
CREATE UNIQUE INDEX IF NOT EXISTS idx_top_chunk_links_unique ON top_chunk_links(chunk_id, image_id);

-- =============================================================================
-- GRANTS (adjust as needed)
-- =============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO neurosynth;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO neurosynth;

-- =============================================================================
-- POST-LOAD INDEX CREATION
-- =============================================================================
-- Run these AFTER loading data for optimal index performance:
--
-- CREATE INDEX idx_chunks_embedding ON chunks 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--
-- CREATE INDEX idx_images_embedding ON images 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
--
-- CREATE INDEX idx_images_caption_embedding ON images 
--     USING ivfflat (caption_embedding vector_cosine_ops) WITH (lists = 50);
--
-- ANALYZE chunks;
-- ANALYZE images;
-- =============================================================================
