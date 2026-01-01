-- Migration: 003_entity_relations
-- Purpose: Add entity relations table for knowledge graph support
-- Date: 2024-12-29

-- Entity relations table for knowledge graph edges
CREATE TABLE IF NOT EXISTS entity_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    chunk_ids UUID[] DEFAULT '{}',
    context_snippet TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

-- Entity embeddings for semantic similarity ranking
CREATE TABLE IF NOT EXISTS entity_embeddings (
    entity_id UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    embedding vector(1536),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient graph traversal
CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_target ON entity_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_type ON entity_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_er_chunks ON entity_relations USING GIN(chunk_ids);

-- Index for entity embedding similarity search
CREATE INDEX IF NOT EXISTS idx_entity_embeddings_vector
    ON entity_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Add source column to entities if not exists (for tracking extraction source)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'entities' AND column_name = 'source'
    ) THEN
        ALTER TABLE entities ADD COLUMN source TEXT DEFAULT 'manual';
    END IF;
END $$;
