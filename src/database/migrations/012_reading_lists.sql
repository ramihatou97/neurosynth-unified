-- ============================================================================
-- Migration 012: Reading Lists & Evidence Levels
-- ============================================================================
-- Purpose: Enable user curation of documents and future evidence tracking.
-- Supports the "Flight Plan" preparation workflow.
-- ============================================================================

-- ============================================================================
-- 1. READING LISTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS reading_lists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,

    -- Optional context links
    procedure_slug VARCHAR(100),              -- Link to procedure_taxonomy
    specialty VARCHAR(50),                    -- Specialty filter

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_shared BOOLEAN DEFAULT FALSE,

    -- Stats (denormalized for quick display)
    item_count INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0
);

COMMENT ON TABLE reading_lists IS 'User-curated collections of library documents for procedure prep';

-- ============================================================================
-- 2. READING LIST ITEMS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS reading_list_items (
    list_id UUID NOT NULL REFERENCES reading_lists(id) ON DELETE CASCADE,
    document_id VARCHAR(500) NOT NULL,        -- file_path from LibraryCatalog

    -- Ordering & Priority
    position INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 2 CHECK (priority BETWEEN 1 AND 3),
    -- 1 = Essential (must read)
    -- 2 = Recommended (should read)
    -- 3 = Optional (nice to have)

    -- User notes
    notes TEXT,

    -- Metadata
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (list_id, document_id)
);

COMMENT ON TABLE reading_list_items IS 'Documents within a reading list with priority and ordering';
COMMENT ON COLUMN reading_list_items.priority IS '1=Essential, 2=Recommended, 3=Optional';

-- ============================================================================
-- 3. EVIDENCE LEVEL COLUMN (Add to documents if exists)
-- ============================================================================

-- Check if documents table exists before altering
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'documents') THEN
        -- Add evidence_level column if it doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'documents' AND column_name = 'evidence_level'
        ) THEN
            ALTER TABLE documents ADD COLUMN evidence_level VARCHAR(30);
        END IF;
    END IF;
END $$;

-- ============================================================================
-- 4. INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_reading_list_procedure ON reading_lists(procedure_slug);
CREATE INDEX IF NOT EXISTS idx_reading_list_specialty ON reading_lists(specialty);
CREATE INDEX IF NOT EXISTS idx_reading_list_updated ON reading_lists(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_reading_list_items_list ON reading_list_items(list_id);
CREATE INDEX IF NOT EXISTS idx_reading_list_items_position ON reading_list_items(list_id, position);
CREATE INDEX IF NOT EXISTS idx_reading_list_items_priority ON reading_list_items(list_id, priority);

-- ============================================================================
-- 5. TRIGGER: Update list stats on item changes
-- ============================================================================

CREATE OR REPLACE FUNCTION update_reading_list_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update item_count for the affected list
    IF TG_OP = 'DELETE' THEN
        UPDATE reading_lists
        SET item_count = (
            SELECT COUNT(*) FROM reading_list_items WHERE list_id = OLD.list_id
        ),
        updated_at = NOW()
        WHERE id = OLD.list_id;
        RETURN OLD;
    ELSE
        UPDATE reading_lists
        SET item_count = (
            SELECT COUNT(*) FROM reading_list_items WHERE list_id = NEW.list_id
        ),
        updated_at = NOW()
        WHERE id = NEW.list_id;
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_reading_list_stats ON reading_list_items;
CREATE TRIGGER trg_reading_list_stats
    AFTER INSERT OR DELETE ON reading_list_items
    FOR EACH ROW
    EXECUTE FUNCTION update_reading_list_stats();

-- ============================================================================
-- 6. HELPER FUNCTIONS
-- ============================================================================

-- Get reading list with full item details
CREATE OR REPLACE FUNCTION get_reading_list_with_items(p_list_id UUID)
RETURNS TABLE (
    list_id UUID,
    list_name VARCHAR(200),
    list_description TEXT,
    procedure_slug VARCHAR(100),
    item_document_id VARCHAR(500),
    item_position INTEGER,
    item_priority INTEGER,
    item_notes TEXT,
    item_added_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        rl.id,
        rl.name,
        rl.description,
        rl.procedure_slug,
        rli.document_id,
        rli.position,
        rli.priority,
        rli.notes,
        rli.added_at
    FROM reading_lists rl
    LEFT JOIN reading_list_items rli ON rl.id = rli.list_id
    WHERE rl.id = p_list_id
    ORDER BY rli.position ASC, rli.added_at DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 7. SEED DATA: Example Reading Lists
-- ============================================================================

-- Create a sample reading list for pterional approach
INSERT INTO reading_lists (name, description, procedure_slug, specialty) VALUES
(
    'Pterional Craniotomy Essentials',
    'Core reading for mastering the pterional approach. Includes Rhoton anatomy, Yasargil technique, and common pitfalls.',
    'pterional-craniotomy',
    'vascular'
)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
