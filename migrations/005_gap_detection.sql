-- Migration 005: Gap Detection System Tables
-- Supports the 14-stage neurosurgical gap detection system

-- =============================================================================
-- Q&A Interactions Table (Stage 5: User Demand Analysis)
-- =============================================================================
-- Tracks questions asked and whether they were satisfactorily answered.
-- Used to identify recurring knowledge gaps based on user demand patterns.

CREATE TABLE IF NOT EXISTS qa_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Question data
    question TEXT NOT NULL,
    question_embedding vector(1024),  -- For semantic similarity matching

    -- Answer data
    answer TEXT,
    was_answered BOOLEAN DEFAULT FALSE,
    answer_quality_score FLOAT,  -- 0-1 scale, how well answered
    answer_source TEXT,  -- 'internal', 'external', 'both', 'none'

    -- Context
    chapter_topic TEXT,
    subspecialty TEXT,  -- skull_base, vascular, spine, tumor, functional, pediatric, trauma, peripheral_nerve
    template_type TEXT,  -- PROCEDURAL, DISORDER, ANATOMY, CONCEPT
    related_document_ids UUID[],
    related_chunk_ids UUID[],

    -- User tracking
    session_id TEXT,
    user_id TEXT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Q&A interactions
CREATE INDEX IF NOT EXISTS idx_qa_topic ON qa_interactions(chapter_topic);
CREATE INDEX IF NOT EXISTS idx_qa_subspecialty ON qa_interactions(subspecialty);
CREATE INDEX IF NOT EXISTS idx_qa_template ON qa_interactions(template_type);
CREATE INDEX IF NOT EXISTS idx_qa_answered ON qa_interactions(was_answered);
CREATE INDEX IF NOT EXISTS idx_qa_quality ON qa_interactions(answer_quality_score) WHERE answer_quality_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_qa_created ON qa_interactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_interactions(session_id) WHERE session_id IS NOT NULL;

-- Embedding index for semantic search
CREATE INDEX IF NOT EXISTS idx_qa_embedding ON qa_interactions
    USING ivfflat (question_embedding vector_cosine_ops)
    WITH (lists = 100);

-- =============================================================================
-- Gap Analysis Cache Table
-- =============================================================================
-- Caches gap analysis results to avoid repeated expensive analysis.
-- Invalidated when source documents change (via source_hash).

CREATE TABLE IF NOT EXISTS gap_analysis_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Query parameters
    topic TEXT NOT NULL,
    template_type TEXT NOT NULL,
    subspecialty TEXT,

    -- Gap report (full JSON)
    gap_report JSONB NOT NULL,

    -- Summary statistics (for quick filtering)
    total_gaps INTEGER DEFAULT 0,
    critical_gap_count INTEGER DEFAULT 0,
    high_gap_count INTEGER DEFAULT 0,
    safety_gaps_flagged BOOLEAN DEFAULT FALSE,
    requires_expert_review BOOLEAN DEFAULT FALSE,

    -- Cache invalidation
    source_hash TEXT,  -- Hash of source document IDs used
    source_document_ids UUID[],
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    analysis_duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for gap cache
CREATE INDEX IF NOT EXISTS idx_gap_cache_topic ON gap_analysis_cache(topic, template_type);
CREATE INDEX IF NOT EXISTS idx_gap_cache_subspecialty ON gap_analysis_cache(subspecialty) WHERE subspecialty IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_gap_cache_hash ON gap_analysis_cache(source_hash);
CREATE INDEX IF NOT EXISTS idx_gap_cache_expires ON gap_analysis_cache(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_gap_cache_critical ON gap_analysis_cache(critical_gap_count) WHERE critical_gap_count > 0;
CREATE INDEX IF NOT EXISTS idx_gap_cache_review ON gap_analysis_cache(requires_expert_review) WHERE requires_expert_review = TRUE;

-- =============================================================================
-- Gap Fill History Table
-- =============================================================================
-- Tracks which gaps were filled and from what sources.
-- Useful for auditing and improving the gap filling system.

CREATE TABLE IF NOT EXISTS gap_fill_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Gap identification
    gap_id TEXT NOT NULL,
    gap_type TEXT NOT NULL,
    gap_topic TEXT NOT NULL,
    gap_priority TEXT NOT NULL,

    -- Context
    original_topic TEXT NOT NULL,
    template_type TEXT NOT NULL,
    subspecialty TEXT,

    -- Fill result
    fill_strategy TEXT NOT NULL,  -- none, high, fallback, always
    fill_source TEXT,  -- internal, external, both, failed
    fill_successful BOOLEAN DEFAULT FALSE,

    -- Content
    filled_content TEXT,
    external_sources_used JSONB,  -- Array of {url, title, snippet}

    -- Metadata
    fill_duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fill history
CREATE INDEX IF NOT EXISTS idx_fill_gap_type ON gap_fill_history(gap_type);
CREATE INDEX IF NOT EXISTS idx_fill_priority ON gap_fill_history(gap_priority);
CREATE INDEX IF NOT EXISTS idx_fill_topic ON gap_fill_history(original_topic);
CREATE INDEX IF NOT EXISTS idx_fill_strategy ON gap_fill_history(fill_strategy);
CREATE INDEX IF NOT EXISTS idx_fill_success ON gap_fill_history(fill_successful);
CREATE INDEX IF NOT EXISTS idx_fill_created ON gap_fill_history(created_at DESC);

-- =============================================================================
-- Unanswered Questions View
-- =============================================================================
-- Aggregates frequently asked but poorly answered questions by topic.
-- Used for Stage 5: User Demand Analysis.

CREATE OR REPLACE VIEW unanswered_questions_by_topic AS
SELECT
    chapter_topic,
    subspecialty,
    COUNT(*) as total_questions,
    COUNT(*) FILTER (WHERE NOT was_answered OR answer_quality_score < 0.5) as unanswered_count,
    AVG(answer_quality_score) as avg_quality,
    array_agg(DISTINCT question ORDER BY question) FILTER (WHERE NOT was_answered) as sample_questions
FROM qa_interactions
WHERE created_at > NOW() - INTERVAL '90 days'
GROUP BY chapter_topic, subspecialty
HAVING COUNT(*) FILTER (WHERE NOT was_answered OR answer_quality_score < 0.5) >= 2
ORDER BY unanswered_count DESC;

-- =============================================================================
-- Critical Gaps Summary View
-- =============================================================================
-- Quick view of topics with unresolved critical gaps.

CREATE OR REPLACE VIEW critical_gaps_summary AS
SELECT
    topic,
    template_type,
    subspecialty,
    critical_gap_count,
    (gap_report->>'safety_flags')::jsonb as safety_flags,
    created_at,
    expires_at
FROM gap_analysis_cache
WHERE critical_gap_count > 0
  AND (expires_at IS NULL OR expires_at > NOW())
ORDER BY critical_gap_count DESC, created_at DESC;

-- =============================================================================
-- Helper Function: Update Q&A Timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_qa_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_qa_updated
    BEFORE UPDATE ON qa_interactions
    FOR EACH ROW
    EXECUTE FUNCTION update_qa_timestamp();

-- =============================================================================
-- Helper Function: Cleanup Expired Cache
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_expired_gap_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM gap_analysis_cache
    WHERE expires_at IS NOT NULL AND expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments for documentation
-- =============================================================================

COMMENT ON TABLE qa_interactions IS 'Tracks Q&A interactions for user demand gap analysis (Stage 5)';
COMMENT ON TABLE gap_analysis_cache IS 'Caches gap analysis results with invalidation support';
COMMENT ON TABLE gap_fill_history IS 'Audit trail of gap filling operations';
COMMENT ON VIEW unanswered_questions_by_topic IS 'Aggregates poorly-answered questions for gap detection';
COMMENT ON VIEW critical_gaps_summary IS 'Quick view of topics with critical safety gaps';
