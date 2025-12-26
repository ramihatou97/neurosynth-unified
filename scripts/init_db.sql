-- NeuroSynth Unified - Database Initialization
-- =============================================
-- PostgreSQL with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Documents Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path TEXT NOT NULL UNIQUE,
    title TEXT,
    total_pages INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

-- =============================================================================
-- Chunks Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    chunk_type TEXT,
    specialty TEXT,
    cuis TEXT[] DEFAULT '{}',
    entities JSONB DEFAULT '{}',
    embedding VECTOR(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_specialty ON chunks(specialty);
CREATE INDEX IF NOT EXISTS idx_chunks_cuis ON chunks USING GIN(cuis);

-- Vector similarity index (IVFFlat for better recall)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- =============================================================================
-- Images Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    file_path TEXT,
    page_number INTEGER,
    image_index INTEGER,
    bbox JSONB,
    image_type TEXT,
    is_decorative BOOLEAN DEFAULT FALSE,
    triage_reason TEXT,
    vlm_caption TEXT,
    ocr_text TEXT,
    cuis TEXT[] DEFAULT '{}',
    embedding VECTOR(512),
    caption_embedding VECTOR(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_document_id ON images(document_id);
CREATE INDEX IF NOT EXISTS idx_images_page_number ON images(page_number);
CREATE INDEX IF NOT EXISTS idx_images_image_type ON images(image_type);
CREATE INDEX IF NOT EXISTS idx_images_is_decorative ON images(is_decorative);
CREATE INDEX IF NOT EXISTS idx_images_cuis ON images USING GIN(cuis);

-- Vector indexes for images
CREATE INDEX IF NOT EXISTS idx_images_embedding ON images 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_images_caption_embedding ON images 
    USING ivfflat (caption_embedding vector_cosine_ops)
    WITH (lists = 50);

-- =============================================================================
-- Links Table (Chunk-Image relationships)
-- =============================================================================
CREATE TABLE IF NOT EXISTS links (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    score FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(chunk_id, image_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_links_chunk_id ON links(chunk_id);
CREATE INDEX IF NOT EXISTS idx_links_image_id ON links(image_id);
CREATE INDEX IF NOT EXISTS idx_links_link_type ON links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_score ON links(score);

-- =============================================================================
-- Functions
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to documents
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Views
-- =============================================================================

-- Document statistics view
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    d.id,
    d.source_path,
    d.title,
    d.total_pages,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT i.id) as image_count,
    COUNT(DISTINCT i.id) FILTER (WHERE NOT i.is_decorative) as medical_image_count,
    COUNT(DISTINCT l.id) as link_count,
    d.created_at
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
LEFT JOIN images i ON i.document_id = d.id
LEFT JOIN links l ON l.chunk_id = c.id
GROUP BY d.id;

-- Chunk type distribution view
CREATE OR REPLACE VIEW chunk_type_distribution AS
SELECT 
    document_id,
    chunk_type,
    COUNT(*) as count
FROM chunks
WHERE chunk_type IS NOT NULL
GROUP BY document_id, chunk_type;

-- =============================================================================
-- Sample Queries for Testing
-- =============================================================================

-- Vector similarity search
-- SELECT id, content, 1 - (embedding <=> $1::vector) as similarity
-- FROM chunks
-- WHERE document_id = $2
-- ORDER BY embedding <=> $1::vector
-- LIMIT 10;

-- Hybrid search with CUI filter
-- SELECT c.id, c.content, 1 - (c.embedding <=> $1::vector) as similarity
-- FROM chunks c
-- WHERE c.cuis && $2::text[]
-- ORDER BY c.embedding <=> $1::vector
-- LIMIT 10;

-- =============================================================================
-- Grants
-- =============================================================================

-- Grant permissions (adjust as needed for your user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO neurosynth;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO neurosynth;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO neurosynth;

-- =============================================================================
-- Completion
-- =============================================================================
SELECT 'NeuroSynth database initialized successfully' as status;
